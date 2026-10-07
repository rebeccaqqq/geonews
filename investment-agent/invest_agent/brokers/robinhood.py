"""Robinhood execution via the community `robin_stocks` library.

IMPORTANT: Robinhood has no official public API for stock trading. robin_stocks
uses the same private endpoints as the Robinhood app. It works, but it can
break without notice, and automated access may conflict with Robinhood's
Terms of Service. Use at your own risk and keep the agent in `approve` mode.
"""

from __future__ import annotations

import logging
import math
import os

from ..data import to_broker
from ..rebalance import Position
from .base import Broker, OrderResult

log = logging.getLogger(__name__)


class RobinhoodBroker(Broker):
    name = "robinhood"

    def __init__(self, state_dir: str = "./state"):
        import robin_stocks.robinhood as rh

        self.rh = rh
        self.state_dir = state_dir
        self._logged_in = False

    def login(self) -> None:
        if self._logged_in:
            return
        user, pw = os.environ.get("RH_USERNAME"), os.environ.get("RH_PASSWORD")
        if not user or not pw:
            raise RuntimeError("RH_USERNAME / RH_PASSWORD not set")
        mfa = None
        secret = os.environ.get("RH_MFA_SECRET")
        if secret:
            import pyotp

            mfa = pyotp.TOTP(secret).now()
        # store_session caches the token in ~/.tokens so we don't log in every run.
        self.rh.login(user, pw, mfa_code=mfa, store_session=True, expiresIn=86400)
        self._logged_in = True

    # --- account ---------------------------------------------------------
    def positions(self) -> dict[str, Position]:
        self.login()
        holdings = self.rh.account.build_holdings() or {}
        out = {}
        for sym, h in holdings.items():
            qty = float(h.get("quantity") or 0)
            if qty <= 0:
                continue
            out[sym.upper()] = Position(sym.upper(), qty, float(h.get("price") or 0))
        return out

    def cash(self) -> float:
        self.login()
        profile = self.rh.profiles.load_account_profile() or {}
        # Settled cash; buying_power can include margin, which we never use.
        for key in ("cash", "portfolio_cash", "buying_power"):
            if profile.get(key) not in (None, ""):
                return float(profile[key])
        return 0.0

    def quotes(self, symbols: list[str]) -> dict[str, float]:
        self.login()
        syms = [to_broker(s) for s in symbols]
        prices = self.rh.stocks.get_latest_price(syms, includeExtendedHours=False) or []
        return {s: float(p) for s, p in zip(symbols, prices) if p}

    def market_open(self) -> bool:
        self.login()
        try:
            hours = self.rh.markets.get_market_today_hours("XNYS")
            return bool(hours and hours.get("is_open")) and _within_regular_hours(hours)
        except Exception:  # noqa: BLE001
            log.exception("Could not fetch market hours; assuming closed")
            return False

    # --- orders ----------------------------------------------------------
    def buy_dollars(self, symbol: str, amount: float) -> OrderResult:
        return self._order(symbol, "buy", amount)

    def sell_dollars(self, symbol: str, amount: float) -> OrderResult:
        return self._order(symbol, "sell", amount)

    def _order(self, symbol: str, side: str, amount: float) -> OrderResult:
        self.login()
        sym = to_broker(symbol)
        fn = self.rh.orders.order_buy_fractional_by_price if side == "buy" else self.rh.orders.order_sell_fractional_by_price
        resp = fn(sym, round(amount, 2), timeInForce="gfd", extendedHours=False) or {}
        if resp.get("id"):
            return OrderResult(symbol, side, amount, True, resp["id"], resp.get("state", ""), resp)

        # Some tickers aren't fractional-eligible: fall back to whole shares.
        detail = str(resp.get("detail") or resp)
        price = self.quotes([symbol]).get(symbol)
        if price:
            qty = math.floor(amount / price)
            if qty >= 1:
                whole = self.rh.orders.order_buy_market if side == "buy" else self.rh.orders.order_sell_market
                resp2 = whole(sym, qty, timeInForce="gfd") or {}
                if resp2.get("id"):
                    return OrderResult(symbol, side, qty * price, True, resp2["id"],
                                       f"whole-share fallback ({qty} sh)", resp2)
                detail += f" | whole-share fallback failed: {resp2.get('detail') or resp2}"
        return OrderResult(symbol, side, amount, False, message=detail, raw=resp)


def _within_regular_hours(hours: dict) -> bool:
    from datetime import datetime, timezone

    try:
        now = datetime.now(timezone.utc)
        opens = datetime.fromisoformat(hours["opens_at"].replace("Z", "+00:00"))
        closes = datetime.fromisoformat(hours["closes_at"].replace("Z", "+00:00"))
        return opens <= now <= closes
    except Exception:  # noqa: BLE001
        return True
