from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
from polygon import RESTClient


def load_prices(
    tickers: list[str],
    start: str,
    end: str,
    timespan: str = "day",
    adjusted: bool = True,
) -> pd.DataFrame:
    """
    Returns adjusted close prices via Polygon.io REST API.
    Reads POLYGON_API_KEY from environment.
    Returns DataFrame with DatetimeIndex and tickers as columns.
    """
    api_key = os.environ.get("POLYGON_API_KEY") or os.environ.get("MASSIVE_API_KEY")
    if not api_key:
        raise EnvironmentError("POLYGON_API_KEY or MASSIVE_API_KEY must be set")

    client = RESTClient(api_key)
    frames: dict[str, pd.Series] = {}

    for ticker in tickers:
        try:
            aggs = client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan=timespan,
                from_=start,
                to=end,
                adjusted=adjusted,
                limit=50000,
            )
            if not aggs:
                continue
            index = pd.to_datetime([a.timestamp for a in aggs], unit="ms", utc=True).tz_localize(None)
            closes = [a.close for a in aggs]
            frames[ticker] = pd.Series(closes, index=index, name=ticker)
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().dropna(axis=1, thresh=252)
    return df
