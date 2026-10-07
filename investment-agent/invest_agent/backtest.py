"""Backtests: (1) your current portfolio, held over the past N years / max history,
and (2) the agent's own strategy, walk-forward with no look-ahead."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .metrics import TRADING_DAYS, beta, calendar_year_returns, cagr, max_drawdown, rolling_beta, summary
from .optimizer import optimize

log = logging.getLogger(__name__)

REBALANCE_FREQS = {"none": None, "monthly": "M", "quarterly": "Q", "annual": "Y"}


# --------------------------------------------------------------------------- holdings
def load_holdings(path: str) -> pd.DataFrame:
    """CSV with a `symbol` column plus one of `shares`, `value`, or `weight`."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "symbol" not in df.columns:
        raise ValueError("holdings CSV needs a 'symbol' column")
    df["symbol"] = df["symbol"].str.strip().str.upper()
    if not {"shares", "value", "weight"} & set(df.columns):
        raise ValueError("holdings CSV needs a 'shares', 'value' or 'weight' column")
    return df.groupby("symbol", as_index=False).sum(numeric_only=True)


def holdings_to_weights(holdings: pd.DataFrame, last_prices: pd.Series) -> pd.Series:
    h = holdings.set_index("symbol")
    if "weight" in h.columns and h["weight"].notna().all() and h["weight"].sum() > 0:
        w = h["weight"].astype(float)
    elif "value" in h.columns and h["value"].notna().all():
        w = h["value"].astype(float)
    else:
        w = h["shares"].astype(float) * last_prices.reindex(h.index)
        missing = w[w.isna()].index.tolist()
        if missing:
            log.warning("No price for %s; excluded", missing)
            w = w.dropna()
    w = w[w > 0]
    return w / w.sum()


# --------------------------------------------------------------------------- engine
def _period_key(ts: pd.Timestamp, freq: str | None):
    if freq is None:
        return None
    return {"M": (ts.year, ts.month), "Q": (ts.year, (ts.month - 1) // 3), "Y": ts.year}[freq]


def simulate(
    prices: pd.DataFrame,
    weights_fn,
    rebalance: str = "none",
    cost_bps: float = 0.0,
    start_value: float = 10_000.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """Generic daily portfolio simulator.

    `weights_fn(date, available_symbols)` returns target weights at a rebalance.
    A rebalance happens on the first day, at each new calendar period
    (per `rebalance`), and whenever a new asset first gets a price (so assets
    that IPO'd mid-window join at their target weight).
    Returns (equity curve, weights history at rebalances).
    """
    freq = REBALANCE_FREQS[rebalance]
    px = prices.ffill()
    dates = px.index
    cols = px.columns
    units = pd.Series(0.0, index=cols)
    equity = np.empty(len(dates))
    history = {}
    prev_key, prev_avail = None, None

    for i, d in enumerate(dates):
        row = px.loc[d]
        avail = frozenset(row.dropna().index)
        value = start_value if i == 0 else float((units * row.fillna(0)).sum())
        key = _period_key(d, freq)
        due = i == 0 or (freq and key != prev_key) or avail != prev_avail
        if due:
            target = weights_fn(d, sorted(avail))
            target = target[target.index.isin(avail)]
            target = target[target > 0]
            if target.sum() > 0:
                target = target / target.sum()
                new_units = (value * target / row[target.index]).reindex(cols).fillna(0.0)
                traded = float((abs(new_units - units) * row.fillna(0)).sum())
                cost = traded * cost_bps / 1e4 if i > 0 else 0.0
                units = new_units * (1 - cost / value) if value > 0 else new_units
                history[d] = target
        equity[i] = float((units * row.fillna(0)).sum())
        prev_key, prev_avail = key, avail

    weights_hist = pd.DataFrame(history).T.fillna(0.0) if history else pd.DataFrame()
    return pd.Series(equity, index=dates, name="equity"), weights_hist


# --------------------------------------------------------------------------- results
@dataclass
class BacktestResult:
    title: str
    curves: pd.DataFrame  # columns: Your portfolio, Benchmark, [Agent strategy]
    stats: pd.DataFrame  # rows: series, columns: metric
    yearly: pd.DataFrame
    rolling_beta: pd.DataFrame
    asset_stats: pd.DataFrame
    weights: pd.Series
    correlation: pd.DataFrame
    notes: list[str] = field(default_factory=list)
    strategy_weights: pd.DataFrame | None = None


def backtest_portfolio(
    weights: pd.Series,
    prices: pd.DataFrame,
    benchmark: str = "SPY",
    rebalance: str = "none",
    align: str = "dynamic",
    cost_bps: float = 5.0,
    risk_free_rate: float = 0.04,
    strategy_cfg: Config | None = None,
    start_value: float = 10_000.0,
    start: pd.Timestamp | None = None,
) -> BacktestResult:
    """Backtest fixed target weights (your holdings) against a benchmark.

    align="dynamic": start at the requested window; assets without history yet
                     are skipped and join when they start trading.
    align="common":  start when every holding has data (shorter but cleaner).
    `prices` may include extra warm-up history before `start` (used by the
    agent-strategy walk-forward); the analysis window begins at `start`.
    """
    notes: list[str] = []
    weights = weights[weights > 0]
    syms = list(weights.index)
    missing = [s for s in syms if s not in prices.columns or prices[s].isna().all()]
    if missing:
        notes.append(f"No price history for {missing}; their weight was redistributed.")
        weights = weights.drop(missing)
        weights = weights / weights.sum()
        syms = list(weights.index)

    full_prices = prices
    if start is not None:
        prices = prices.loc[start:]
    firsts = {s: prices[s].first_valid_index() for s in syms}
    window_start = prices.index[0]
    late = {s: d for s, d in firsts.items() if d > window_start + pd.Timedelta(days=7)}
    if align == "common" and late:
        window_start = max(late.values())
        notes.append(
            f"Window starts {window_start.date()} because "
            + ", ".join(f"{s} (from {d.date()})" for s, d in late.items())
            + " had shorter history."
        )
    elif late:
        notes.append(
            "Joined late (weight spread over other holdings until then): "
            + ", ".join(f"{s} {d.date()}" for s, d in sorted(late.items(), key=lambda x: x[1]))
        )

    px = prices.loc[window_start:]
    port_eq, _ = simulate(px[syms], lambda d, a: weights, rebalance, cost_bps, start_value)
    bench_eq = px[benchmark].ffill().dropna()
    bench_eq = bench_eq / bench_eq.iloc[0] * start_value

    curves = {"Your portfolio": port_eq, f"Benchmark ({benchmark})": bench_eq}
    strat_w = None
    if strategy_cfg is not None:
        strat_eq, strat_w = backtest_strategy(full_prices, strategy_cfg, start=window_start, start_value=start_value)
        if strat_eq.index[0] > window_start + pd.Timedelta(days=7):
            # Rebase so the line starts on the benchmark's value that day.
            strat_eq = strat_eq / strat_eq.iloc[0] * float(bench_eq.asof(strat_eq.index[0]))
            notes.append(
                f"Agent strategy starts {strat_eq.index[0].date()} (needs 1 year of warm-up data); "
                "its stats cover that shorter window."
            )
        curves["Agent strategy"] = strat_eq
        notes.append(
            "Agent strategy is walk-forward (monthly re-optimization using only past data), "
            f"{strategy_cfg.transaction_cost_bps:.0f} bps costs. Its ETF universe was chosen today, "
            "so it carries some hindsight bias."
        )
    curves_df = pd.DataFrame(curves)
    curves_df = curves_df.loc[curves_df["Your portfolio"].first_valid_index():].ffill()

    bench_col = f"Benchmark ({benchmark})"
    stats = pd.DataFrame(
        {name: summary(curves_df[name].dropna(), curves_df[bench_col], risk_free_rate) for name in curves_df.columns}
    ).T

    yearly = pd.DataFrame({name: calendar_year_returns(curves_df[name].dropna()) for name in curves_df.columns})
    rets = curves_df.pct_change()
    rb = pd.DataFrame(
        {name: rolling_beta(rets[name], rets[bench_col]) for name in curves_df.columns if name != bench_col}
    ).dropna(how="all")

    asset_rets = px[syms].pct_change()
    mkt = px[benchmark].pct_change()
    asset_stats = pd.DataFrame(
        {
            "weight": weights,
            "first_date": pd.Series({s: firsts[s].date() for s in syms}),
            "cagr": pd.Series({s: cagr(px[s].dropna()) for s in syms}),
            "volatility": asset_rets.std() * np.sqrt(TRADING_DAYS),
            "beta": pd.Series({s: beta(asset_rets[s], mkt) for s in syms}),
            "max_drawdown": pd.Series({s: max_drawdown(px[s].dropna()) for s in syms}),
        }
    ).sort_values("weight", ascending=False)
    asset_stats["beta_contribution"] = asset_stats["weight"] * asset_stats["beta"]

    years = (curves_df.index[-1] - curves_df.index[0]).days / 365.25
    title = f"{curves_df.index[0].date()} to {curves_df.index[-1].date()} ({years:.1f} years), rebalance: {rebalance}"
    return BacktestResult(
        title=title, curves=curves_df, stats=stats, yearly=yearly, rolling_beta=rb, asset_stats=asset_stats,
        weights=weights, correlation=asset_rets.corr().round(2), notes=notes, strategy_weights=strat_w,
    )


def backtest_strategy(
    prices: pd.DataFrame, cfg: Config, start: pd.Timestamp | None = None, start_value: float = 10_000.0
) -> tuple[pd.Series, pd.DataFrame]:
    """Walk-forward test of the optimizer: each month, re-optimize with data up to
    the previous close only, then hold. Needs >=1y of history before `start`."""
    universe = [t for t in cfg.universe if t in prices.columns]
    cols = sorted(set(universe) | {cfg.benchmark})
    px = prices[cols]
    start = start or px.index[0]
    min_start = px.index[0] + pd.Timedelta(days=380)
    if start < min_start:
        start = min_start
    sim_px = px.loc[start:, universe]
    cache: dict = {}

    def weights_fn(d, avail):
        key = (d.year, d.month)
        if key not in cache:
            hist = px.loc[: d - pd.Timedelta(days=1)]
            try:
                cache[key] = optimize(hist, cfg).weights
            except (ValueError, RuntimeError) as e:  # not enough history yet
                log.debug("optimize failed at %s: %s", d.date(), e)
                cache[key] = pd.Series({cfg.benchmark: 1.0}) if cfg.benchmark in avail else pd.Series(1.0, index=avail) / len(avail)
        return cache[key]

    if cfg.benchmark not in universe:
        sim_px = px.loc[start:, sorted(set(universe) | {cfg.benchmark})]
    return simulate(sim_px, weights_fn, "monthly", cfg.transaction_cost_bps, start_value)
