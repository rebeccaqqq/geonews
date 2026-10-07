"""Portfolio performance and risk statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    df = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if len(df) < 20:
        return float("nan")
    a, m = df.iloc[:, 0], df.iloc[:, 1]
    return float(np.cov(a, m, ddof=1)[0, 1] / np.var(m, ddof=1))


def betas(returns: pd.DataFrame, market_returns: pd.Series, blume: bool = True) -> pd.Series:
    """Per-asset beta vs market. Blume adjustment shrinks toward 1 (0.67*raw + 0.33)."""
    raw = pd.Series({c: beta(returns[c], market_returns) for c in returns.columns})
    return 0.67 * raw + 0.33 if blume else raw


def rolling_beta(returns: pd.Series, market_returns: pd.Series, window: int = 252) -> pd.Series:
    cov = returns.rolling(window).cov(market_returns)
    var = market_returns.rolling(window).var()
    return cov / var


def cagr(equity: pd.Series) -> float:
    equity = equity.dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def max_drawdown(equity: pd.Series) -> float:
    dd = equity / equity.cummax() - 1
    return float(dd.min())


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1


def summary(
    equity: pd.Series,
    benchmark_equity: pd.Series | None = None,
    risk_free_rate: float = 0.04,
) -> dict[str, float]:
    """Headline stats for an equity curve (and relative stats vs a benchmark)."""
    r = equity.pct_change().dropna()
    rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    excess = r - rf_daily
    vol = float(r.std() * np.sqrt(TRADING_DAYS))
    downside = r[r < 0].std() * np.sqrt(TRADING_DAYS)
    c = cagr(equity)
    mdd = max_drawdown(equity)
    out: dict[str, float] = {
        "start_value": float(equity.iloc[0]),
        "end_value": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "cagr": c,
        "volatility": vol,
        "sharpe": float(excess.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else float("nan"),
        "sortino": float((c - risk_free_rate) / downside) if downside > 0 else float("nan"),
        "max_drawdown": mdd,
        "calmar": float(c / abs(mdd)) if mdd < 0 else float("nan"),
        "best_day": float(r.max()),
        "worst_day": float(r.min()),
        "pct_positive_days": float((r > 0).mean()),
    }
    if benchmark_equity is not None:
        m = benchmark_equity.reindex(equity.index).pct_change().dropna()
        r2, m = r.align(m, join="inner")
        b = beta(r2, m)
        mc = cagr(benchmark_equity.reindex(equity.index).dropna())
        active = r2 - m
        te = float(active.std() * np.sqrt(TRADING_DAYS))
        up, down = m > 0, m < 0
        out.update(
            {
                "beta": b,
                "correlation": float(r2.corr(m)),
                "alpha_annual": float(c - (risk_free_rate + b * (mc - risk_free_rate))),
                "tracking_error": te,
                "information_ratio": float((c - mc) / te) if te > 0 else float("nan"),
                "upside_capture": float(r2[up].mean() / m[up].mean()) if up.any() else float("nan"),
                "downside_capture": float(r2[down].mean() / m[down].mean()) if down.any() else float("nan"),
                "benchmark_cagr": mc,
            }
        )
    return out


def calendar_year_returns(equity: pd.Series) -> pd.Series:
    yearly = equity.resample("YE").last()
    first = equity.iloc[:1]
    yearly = pd.concat([first, yearly])
    rets = yearly.pct_change().dropna()
    rets.index = rets.index.year
    return rets.groupby(level=0).last()
