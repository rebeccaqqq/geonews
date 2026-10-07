from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..rebalance import Position


@dataclass
class OrderResult:
    symbol: str
    side: str
    notional_usd: float
    ok: bool
    order_id: str | None = None
    message: str = ""
    raw: dict = field(default_factory=dict)


class Broker(ABC):
    name: str = "base"

    @abstractmethod
    def positions(self) -> dict[str, Position]: ...

    @abstractmethod
    def cash(self) -> float: ...

    @abstractmethod
    def quotes(self, symbols: list[str]) -> dict[str, float]: ...

    @abstractmethod
    def market_open(self) -> bool: ...

    @abstractmethod
    def buy_dollars(self, symbol: str, amount: float) -> OrderResult: ...

    @abstractmethod
    def sell_dollars(self, symbol: str, amount: float) -> OrderResult: ...
