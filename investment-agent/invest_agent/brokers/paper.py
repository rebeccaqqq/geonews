"""Simulated broker backed by a JSON file. Useful for dry runs and testing."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from ..rebalance import Position
from .base import Broker, OrderResult


class PaperBroker(Broker):
    name = "paper"

    def __init__(self, state_dir: str = "./state", starting_cash: float = 100_000.0, price_source=None):
        self.path = Path(state_dir) / "paper_account.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._price_source = price_source
        if not self.path.exists():
            self._save({"cash": starting_cash, "positions": {}, "orders": []})

    def _load(self) -> dict:
        return json.loads(self.path.read_text())

    def _save(self, state: dict) -> None:
        self.path.write_text(json.dumps(state, indent=2))

    def quotes(self, symbols: list[str]) -> dict[str, float]:
        if self._price_source:
            return {s: float(self._price_source(s)) for s in symbols}
        from ..data import latest_prices

        return latest_prices(symbols)

    def positions(self) -> dict[str, Position]:
        state = self._load()
        held = {s: q for s, q in state["positions"].items() if q > 1e-9}
        px = self.quotes(list(held)) if held else {}
        return {s: Position(s, q, px.get(s, 0.0)) for s, q in held.items()}

    def cash(self) -> float:
        return float(self._load()["cash"])

    def market_open(self) -> bool:
        return True

    def _fill(self, symbol: str, side: str, amount: float) -> OrderResult:
        state = self._load()
        price = self.quotes([symbol])[symbol]
        qty = amount / price
        held = state["positions"].get(symbol, 0.0)
        if side == "buy":
            if amount > state["cash"] + 1e-6:
                return OrderResult(symbol, side, amount, False, message="insufficient cash")
            state["cash"] -= amount
            state["positions"][symbol] = held + qty
        else:
            qty = min(qty, held)
            amount = qty * price
            state["cash"] += amount
            state["positions"][symbol] = held - qty
        oid = uuid.uuid4().hex[:12]
        state["orders"].append(
            {"id": oid, "symbol": symbol, "side": side, "qty": qty, "price": price,
             "at": datetime.now(timezone.utc).isoformat()}
        )
        self._save(state)
        return OrderResult(symbol, side, amount, True, order_id=oid, message=f"filled {qty:.4f} @ {price:.2f}")

    def buy_dollars(self, symbol: str, amount: float) -> OrderResult:
        return self._fill(symbol, "buy", amount)

    def sell_dollars(self, symbol: str, amount: float) -> OrderResult:
        return self._fill(symbol, "sell", amount)
