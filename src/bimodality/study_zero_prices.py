"""Diagnostic: find and characterize zero / null prices in the cached bars.

Run *after* a build has cached ``bars_<start>_<end>.parquet``:

    .venv/bin/python -m src.bimodality.study_zero_prices

Answers: how many zero closes, how many nulls, which tickers and dates, and
whether they cluster (a few bad tickers vs. scattered ticks). Read-only — it
touches the cache only, never the API.
"""
from __future__ import annotations

import sys

import pandas as pd

from . import constant as C


def _load_bars(cfg: C.StrategyConfig) -> pd.DataFrame:
    path = cfg.cache_dir / f"bars_{cfg.start}_{cfg.end}.parquet"
    if not path.exists():
        sys.exit(f"No bars cache at {path}. Run a build first so it gets written.")
    print(f"Loading {path}")
    return pd.read_parquet(path)  # long form: (date, ticker) index, price/volume cols


def main() -> None:
    cfg = C.StrategyConfig()
    store = _load_bars(cfg)
    price = store["price"]

    total = len(price)
    n_null = int(price.isna().sum())
    is_zero = price == 0
    is_neg = price < 0
    n_zero = int(is_zero.sum())
    n_neg = int(is_neg.sum())

    print("\n=== Price quality ===")
    print(f"rows (date,ticker) : {total:,}")
    print(f"null  prices       : {n_null:,} ({n_null / total:.4%})")
    print(f"zero  prices       : {n_zero:,} ({n_zero / total:.4%})")
    print(f"neg   prices       : {n_neg:,} ({n_neg / total:.4%})")

    bad = price[(is_zero | is_neg)]
    if bad.empty:
        print("\nNo zero/negative prices in cache.")
        return

    by_ticker = bad.groupby(level=1).size().sort_values(ascending=False)
    by_date = bad.groupby(level=0).size().sort_values(ascending=False)

    print("\n=== Top 20 tickers with zero/neg closes ===")
    print(by_ticker.head(20).to_string())
    print(f"\ndistinct tickers affected : {by_ticker.size}")

    print("\n=== Top 20 month-ends with zero/neg closes ===")
    print(by_date.head(20).to_string())

    print("\n=== Sample offending rows ===")
    print(bad.head(20).to_string())


if __name__ == "__main__":
    main()
