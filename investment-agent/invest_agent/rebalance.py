"""Turn target weights + current holdings into a guarded list of orders."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .config import Config


@dataclass
class Position:
    symbol: str
    quantity: float
    price: float

    @property
    def value(self) -> float:
        return self.quantity * self.price


@dataclass
class Trade:
    symbol: str
    side: str  # "buy" | "sell"
    notional_usd: float
    price: float
    reason: str

    @property
    def est_quantity(self) -> float:
        return self.notional_usd / self.price if self.price else 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return f"{self.side.upper():<4} {self.symbol:<6} ${self.notional_usd:>10,.2f} (~{self.est_quantity:.4f} sh) - {self.reason}"


@dataclass
class RebalancePlan:
    trades: list[Trade]
    total_value: float
    cash: float
    current_weights: pd.Series
    target_weights: pd.Series  # of the whole account, incl. locked holdings
    locked: list[str]
    notes: list[str]

    @property
    def gross_notional(self) -> float:
        return sum(t.notional_usd for t in self.trades)


def plan_rebalance(
    positions: dict[str, Position],
    cash: float,
    target_weights: pd.Series,
    cfg: Config,
    prices: dict[str, float],
) -> RebalancePlan:
    notes: list[str] = []
    total = sum(p.value for p in positions.values()) + cash
    if total <= 0:
        raise ValueError("Account has no value")

    universe = set(cfg.universe)
    locked = [s for s in positions if s not in universe and not cfg.sell_outside_universe]
    locked_value = sum(positions[s].value for s in locked)
    if locked:
        notes.append(f"Left untouched (outside universe): {', '.join(sorted(locked))} = ${locked_value:,.0f}")

    investable = total * (1 - cfg.cash_buffer_pct) - locked_value
    if investable <= 0:
        raise ValueError("Nothing investable after cash buffer and locked holdings")

    target_value = (target_weights / target_weights.sum()) * investable
    symbols = sorted((set(positions) | set(target_value.index)) - set(locked))

    current = pd.Series({s: positions[s].value if s in positions else 0.0 for s in symbols})
    desired = pd.Series({s: float(target_value.get(s, 0.0)) for s in symbols})

    trades: list[Trade] = []
    for s in symbols:
        diff = desired[s] - current[s]
        drift = abs(diff) / total
        exiting = desired[s] == 0 and current[s] > 0
        if drift < cfg.drift_threshold and not exiting:
            continue
        if abs(diff) < cfg.min_trade_usd:
            continue
        side = "buy" if diff > 0 else "sell"
        if side == "sell" and s in cfg.do_not_sell:
            notes.append(f"Skipped sell of {s} (do_not_sell)")
            continue
        if side == "buy" and s in cfg.blocklist:
            continue
        price = prices.get(s) or (positions[s].price if s in positions else 0.0)
        if not price:
            notes.append(f"Skipped {s}: no price")
            continue
        amount = abs(diff)
        # Never sell more than held.
        if side == "sell":
            amount = min(amount, current[s])
        reason = f"{current[s] / total:.1%} -> {desired[s] / total:.1%}"
        trades.append(Trade(s, side, amount, price, reason))

    trades = _apply_limits(trades, total, cash, cfg, notes)
    target_full = pd.concat(
        [desired / total, pd.Series({s: positions[s].value / total for s in locked}, dtype=float)]
    )
    current_full = pd.Series({s: p.value / total for s, p in positions.items()}, dtype=float)
    return RebalancePlan(trades, total, cash, current_full, target_full, locked, notes)


def _scale(trades: list[Trade], factor: float) -> None:
    for t in trades:
        t.notional_usd *= factor


def _apply_limits(trades: list[Trade], total: float, cash: float, cfg: Config, notes: list[str]) -> list[Trade]:
    sells = [t for t in trades if t.side == "sell"]
    buys = [t for t in trades if t.side == "buy"]

    # Turnover cap: one-way turnover = max(buys, sells) / total.
    one_way = max(sum(t.notional_usd for t in buys), sum(t.notional_usd for t in sells))
    cap = cfg.max_turnover_pct * total
    if one_way > cap > 0:
        _scale(trades, cap / one_way)
        notes.append(f"Scaled trades to {cfg.max_turnover_pct:.0%} turnover cap")

    # Daily notional cap across all orders.
    gross = sum(t.notional_usd for t in trades)
    if gross > cfg.max_daily_notional_usd:
        _scale(trades, cfg.max_daily_notional_usd / gross)
        notes.append(f"Scaled trades to ${cfg.max_daily_notional_usd:,.0f} daily notional cap")

    # Per-order cap.
    for t in trades:
        if t.notional_usd > cfg.max_order_usd:
            t.notional_usd = cfg.max_order_usd
            t.reason += f" (capped at ${cfg.max_order_usd:,.0f})"

    # Buys must be funded by cash + sells, minus the cash buffer.
    available = cash + sum(t.notional_usd for t in sells) - cfg.cash_buffer_pct * total
    buy_total = sum(t.notional_usd for t in buys)
    if buy_total > available:
        if available <= 0:
            notes.append("No cash available for buys")
            buys = []
        else:
            _scale(buys, available / buy_total)
            notes.append("Scaled buys to available cash")

    out = [t for t in sells + buys if t.notional_usd >= cfg.min_trade_usd]
    for t in out:
        t.notional_usd = round(t.notional_usd, 2)
    return out  # sells first so they fund the buys
