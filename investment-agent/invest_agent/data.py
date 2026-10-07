"""Price data: dividend-adjusted daily closes from Yahoo Finance, cached on disk."""

from __future__ import annotations

import hashlib
import logging
from datetime import date
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

# Yahoo uses '-' for share classes (BRK-B); Robinhood uses '.' (BRK.B).
def to_yahoo(ticker: str) -> str:
    return ticker.upper().replace(".", "-")


def to_broker(ticker: str) -> str:
    return ticker.upper().replace("-", ".")


def load_prices(
    tickers: list[str],
    start: str | None = None,
    period: str | None = None,
    cache_dir: str | Path | None = "./state/cache",
) -> pd.DataFrame:
    """Return a DataFrame of adjusted closes (index=date, columns=tickers).

    Pass either `start` (YYYY-MM-DD) or `period` (e.g. "10y", "max").
    Columns keep the caller's ticker spelling. Assets that did not exist yet
    are NaN before their first trade.
    """
    import yfinance as yf

    tickers = sorted({t.upper() for t in tickers})
    key = f"{'_'.join(tickers)}_{start or period}_{date.today().isoformat()}"
    cache_file = None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_file = Path(cache_dir) / f"{hashlib.sha1(key.encode()).hexdigest()[:16]}.csv"
        if cache_file.exists():
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    yahoo = [to_yahoo(t) for t in tickers]
    kwargs = {"start": start} if start else {"period": period or "10y"}
    raw = yf.download(yahoo, auto_adjust=True, progress=False, group_by="column", **kwargs)
    if raw.empty:
        raise RuntimeError(f"No price data returned for {tickers}")
    closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if not isinstance(closes, pd.DataFrame):
        closes = closes.to_frame()
    if list(closes.columns) == ["Close"]:
        closes.columns = yahoo
    closes = closes.rename(columns=dict(zip(yahoo, tickers)))[tickers]
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    closes = closes.dropna(how="all")

    missing = [t for t in tickers if closes[t].isna().all()]
    if missing:
        log.warning("No data for: %s", missing)

    if cache_file is not None:
        closes.to_csv(cache_file)
    return closes


def latest_prices(tickers: list[str]) -> dict[str, float]:
    px = load_prices(tickers, period="5d", cache_dir=None)
    return {t: float(px[t].dropna().iloc[-1]) for t in px.columns if px[t].notna().any()}
