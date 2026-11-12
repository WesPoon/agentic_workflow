from __future__ import annotations

import pandas as pd
import yfinance as yf


def load_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Returns adjusted close prices as a DataFrame with DatetimeIndex and tickers as columns.
    Drops tickers with insufficient data (< 252 trading days).
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw

    prices = prices.dropna(axis=1, thresh=252)
    prices.index = pd.to_datetime(prices.index)
    return prices
