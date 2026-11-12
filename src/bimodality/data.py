"""Data layer — Polygon Massive -> cached monthly cross-sectional panels.

Builds the full-universe monthly panels (price, volume, market cap), derives
monthly returns, and computes a per-month ``universe_mask`` encoding every
dataset filter. All network results are cached to parquet so re-runs are offline.

Lookahead is *not* handled here (panels are point-in-time observations); the
``feature`` layer enforces the t-1 shift when it builds momentum.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import constant as C
from .util import get_logger

logger = get_logger("data")


@dataclass
class MarketData:
    """Bundle of month-end panels. Index = month-end dates, columns = tickers."""

    prices: pd.DataFrame        # adjusted close
    volume: pd.DataFrame        # monthly volume
    market_cap: pd.DataFrame    # close * shares outstanding
    returns: pd.DataFrame       # monthly simple returns (derived)
    universe_mask: pd.DataFrame  # bool: passes every filter that month

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.prices.index

    def eligible_tickers(self, date: pd.Timestamp) -> pd.Index:
        """Tickers passing every filter on ``date``."""
        row = self.universe_mask.loc[date]
        return row.index[row.fillna(False).to_numpy()]


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def load_market_data(cfg: C.StrategyConfig, refresh: bool = False) -> MarketData:
    """Load (or build + cache) the monthly panels for the config's date span.

    Parameters
    ----------
    cfg : StrategyConfig
    refresh : bool
        If True, ignore cache and rebuild from Polygon.
    """
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cfg.cache_dir / f"panels_{cfg.start}_{cfg.end}.parquet"

    if cache.exists() and not refresh:
        logger.info("Loading cached panels from %s", cache)
        store = pd.read_parquet(cache)
        return _unstack_store(store)

    logger.info("Building panels from Polygon (%s -> %s)", cfg.start, cfg.end)
    prices, volume, shares, list_dates = _fetch_from_polygon(cfg)

    market_cap = prices * shares
    # Non-positive prices are bad ticks; null them so they don't create
    # divide-by-zero / negative returns in pct_change.
    clean_prices = prices.where(prices > 0)
    returns = clean_prices.pct_change()
    mask = _build_universe_mask(prices, volume, market_cap, returns, list_dates, cfg)

    market = MarketData(prices, volume, market_cap, returns, mask)
    _stack_store(market).to_parquet(cache)
    logger.info("Cached panels to %s", cache)
    return market


# --------------------------------------------------------------------------- #
# Polygon fetch (grouped-daily build)
# --------------------------------------------------------------------------- #
def _fetch_from_polygon(
    cfg: C.StrategyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return (prices, volume, shares, list_dates) month-end panels/series.

    Strategy: one ``get_grouped_daily_aggs`` call per month-end trading day gives
    every ticker's close + volume at once (~12 calls/year). Shares outstanding and
    list date come from ``get_ticker_details`` — fetched concurrently and cached, and
    only for tickers that actually appear in the price panels.
    """
    from massive import RESTClient  # imported lazily so tests can run offline

    api_key = os.environ.get("POLYGON_API_KEY") or os.environ.get("MASSIVE_API_KEY")
    if not api_key:
        raise EnvironmentError("POLYGON_API_KEY or MASSIVE_API_KEY must be set")
    client = RESTClient(api_key)

    cs_tickers = _load_cs_universe(client, cfg)

    prices, volume = _load_grouped_bars(client, cs_tickers, cfg)

    # Per-ticker reference (shares + list_date) only for names that actually traded.
    meta = _fetch_ticker_meta(client, prices.columns, cfg)
    shares = pd.DataFrame(
        np.tile(meta["shares"].to_numpy(), (len(prices.index), 1)),
        index=prices.index,
        columns=prices.columns,
    )
    return prices, volume, shares, meta["list_date"]


# --------------------------------------------------------------------------- #
# Grouped-daily bars (cached — a crash after this loop must not re-pull bars)
# --------------------------------------------------------------------------- #
def _load_grouped_bars(
    client, cs_tickers: set[str], cfg: C.StrategyConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Month-end (price, volume) panels from grouped-daily aggs, cached to parquet.

    One ``get_grouped_daily_aggs`` call per month-end gives every ticker at once.
    The result is cached (``bars_<start>_<end>.parquet``) so a downstream crash
    never forces a re-pull. Bars are coerced to float64 — raw aggs can carry
    ``None`` closes that would otherwise poison the column to object dtype and turn
    a benign divide-by-zero into a ``ZeroDivisionError``.
    """
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cfg.cache_dir / f"bars_{cfg.start}_{cfg.end}.parquet"
    if cache_path.exists():
        store = pd.read_parquet(cache_path)
        logger.info("Loaded grouped-daily bars from cache: %s", cache_path)
        return store["price"].unstack(), store["volume"].unstack()

    month_ends = _month_end_trading_days(cfg.start, cfg.end)
    close_rows: dict[pd.Timestamp, pd.Series] = {}
    vol_rows: dict[pd.Timestamp, pd.Series] = {}

    logger.info("Pulling grouped-daily bars for %d month-ends ...", len(month_ends))
    for n, dt in enumerate(month_ends, start=1):
        try:
            aggs = client.get_grouped_daily_aggs(dt.strftime("%Y-%m-%d"), adjusted=True)
        except Exception as exc:  # noqa: BLE001 - skip missing/holiday days
            logger.warning("grouped_daily failed for %s: %s", dt.date(), exc)
            continue
        closes, vols = {}, {}
        for a in aggs:
            tkr = getattr(a, "ticker", None)
            if tkr in cs_tickers:
                closes[tkr] = a.close
                vols[tkr] = a.volume
        close_rows[dt] = pd.Series(closes)
        vol_rows[dt] = pd.Series(vols)
        if n % 24 == 0 or n == len(month_ends):
            logger.info("  ... grouped-daily %d/%d (%s)", n, len(month_ends), dt.date())

    prices = pd.DataFrame(close_rows).T.sort_index().apply(pd.to_numeric, errors="coerce")
    volume = pd.DataFrame(vol_rows).T.sort_index().apply(pd.to_numeric, errors="coerce")

    bars = pd.DataFrame(
        {
            "price": prices.stack(future_stack=True),
            "volume": volume.stack(future_stack=True),
        }
    )
    # fastparquet resets the index by name; unnamed MultiIndex levels (None)
    # break its assign(**{None: ...}). Name them explicitly.
    bars.index = bars.index.set_names(["date", "ticker"])
    bars.to_parquet(cache_path)
    logger.info("Cached grouped-daily bars to %s (%d months)", cache_path, len(prices))
    return prices, volume


# --------------------------------------------------------------------------- #
# CS universe (cached — avoids re-paginating the full delisted list every run)
# --------------------------------------------------------------------------- #
def _load_cs_universe(client, cfg: C.StrategyConfig) -> set[str]:
    """Active + delisted US common-stock tickers, cached to parquet.

    ``list_tickers`` paginates the entire (huge) delisted universe, so we do it once
    and reuse it. Progress is logged so the call never looks dead.
    """
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cfg.cache_dir / "cs_universe.parquet"
    if cache_path.exists():
        cs = set(pd.read_parquet(cache_path)["ticker"])
        logger.info("Loaded CS universe from cache: %d tickers (%s)", len(cs), cache_path)
        return cs

    logger.info("Listing CS universe (active + delisted) from Polygon — one-time ...")
    cs: set[str] = set()
    for active in (True, False):
        for t in client.list_tickers(market="stocks", type="CS", active=active, limit=1000):
            cs.add(t.ticker)
        logger.info("  ... universe so far: %d (active=%s done)", len(cs), active)
    pd.DataFrame({"ticker": sorted(cs)}).to_parquet(cache_path)
    logger.info("Cached CS universe: %d tickers -> %s", len(cs), cache_path)
    return cs


# --------------------------------------------------------------------------- #
# Per-ticker reference fetch (threaded + parquet-cached)
# --------------------------------------------------------------------------- #
def _fetch_ticker_meta(
    client,
    tickers: pd.Index,
    cfg: C.StrategyConfig,
) -> pd.DataFrame:
    """Return per-ticker ``shares`` and ``list_date``, ticker-indexed.

    Incrementally cached to parquet: only tickers absent from the cache hit the API,
    and those are fetched concurrently via a thread pool (the calls are I/O-bound,
    so threads — not asyncio — give the speedup against the blocking SDK).
    """
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cfg.cache_dir / "ticker_meta.parquet"

    cached = (
        pd.read_parquet(cache_path)
        if cache_path.exists()
        else pd.DataFrame(columns=["shares", "list_date"])
    )
    missing = [t for t in tickers if t not in cached.index]

    if missing:
        logger.info("Fetching ticker meta: %d new (%d cached) with %d workers",
                    len(missing), len(cached), cfg.fetch_max_workers)
        fetched = _fetch_meta_concurrent(client, missing, cfg.fetch_max_workers)
        cached = pd.concat([cached, fetched])
        cached = cached[~cached.index.duplicated(keep="last")]
        # Normalize dtypes before serialization: a mixed object column of
        # Timestamp/NaT is not parquet-writable, so coerce explicitly.
        cached["shares"] = pd.to_numeric(cached["shares"], errors="coerce")
        cached["list_date"] = pd.to_datetime(cached["list_date"], errors="coerce")
        cached.to_parquet(cache_path)
        logger.info("Cached ticker meta to %s (%d total)", cache_path, len(cached))

    return cached.reindex(tickers)


def _fetch_meta_concurrent(client, tickers: list[str], max_workers: int) -> pd.DataFrame:
    """Concurrently fetch (shares, list_date) for ``tickers`` via a thread pool.

    Interruptible: on Ctrl-C we cancel pending futures (``cancel_futures=True``,
    ``wait=False``) instead of letting the default ``with``-block join drain all
    queued work — so the process actually stops when you ask it to.
    """
    rows: dict[str, tuple[float, pd.Timestamp]] = {}
    total = len(tickers)
    pool = ThreadPoolExecutor(max_workers=max_workers)
    futures = {pool.submit(_fetch_one_meta, client, t): t for t in tickers}
    try:
        for done, fut in enumerate(as_completed(futures), start=1):
            rows[futures[fut]] = fut.result()
            if done % 200 == 0 or done == total:
                logger.info("  ... ticker meta %d/%d", done, total)
    except KeyboardInterrupt:
        logger.warning("Interrupted — cancelling %d pending ticker fetches", total - len(rows))
        raise
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return pd.DataFrame.from_dict(rows, orient="index", columns=["shares", "list_date"])


def _fetch_one_meta(client, ticker: str) -> tuple[float, pd.Timestamp]:
    """One ``get_ticker_details`` call -> (shares_outstanding, list_date)."""
    try:
        d = client.get_ticker_details(ticker)
        shares = getattr(d, "share_class_shares_outstanding", None)
        if not shares:
            shares = getattr(d, "weighted_shares_outstanding", None)
        ld = getattr(d, "list_date", None)
        return (
            float(shares) if shares else np.nan,
            pd.Timestamp(ld) if ld else pd.NaT,
        )
    except Exception:  # noqa: BLE001 - missing/delisted detail -> NaN/NaT
        return (np.nan, pd.NaT)


def _month_end_trading_days(start: str, end: str) -> pd.DatetimeIndex:
    """Business month-end dates in [start, end]."""
    return pd.date_range(start=start, end=end, freq="BME")


# --------------------------------------------------------------------------- #
# Universe filtering
# --------------------------------------------------------------------------- #
def _build_universe_mask(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    market_cap: pd.DataFrame,
    returns: pd.DataFrame,
    list_dates: pd.Series,
    cfg: C.StrategyConfig,
) -> pd.DataFrame:
    """Bool panel: True where a ticker passes every dataset filter that month."""
    has_price = prices.notna() & (prices > 0)

    # Market-cap floor: drop bottom X% within each month's cross-section.
    cap_rank = market_cap.rank(axis=1, pct=True)
    above_floor = cap_rank > cfg.market_cap_floor_pct

    # Listing age >= min_listing_months.
    age_ok = _listing_age_mask(prices.index, prices.columns, list_dates, cfg.min_listing_months)

    # Return outliers.
    ret_ok = (returns <= C.RETURN_UPPER_CAP) & (returns >= C.RETURN_LOWER_CAP)
    ret_ok = ret_ok.reindex_like(has_price).fillna(True)

    # Zero-volume + zero-return staleness filter (this month AND prior month).
    zero_now = (volume.fillna(0) == 0) & (returns.fillna(0) == 0)
    zero_prev = zero_now.shift(1).fillna(False)
    stale = zero_now & zero_prev
    volume_ok = ~stale

    mask = has_price & above_floor.fillna(False) & age_ok & ret_ok & volume_ok
    return mask.fillna(False)


def _listing_age_mask(
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    list_dates: pd.Series,
    min_months: int,
) -> pd.DataFrame:
    """True where the stock has been listed >= ``min_months`` as of each date.

    Vectorized: compute each ticker's seasoning threshold once, then broadcast the
    ``dates (rows) >= threshold (cols)`` comparison across the whole panel. Tickers
    with an unknown list date are treated as seasoned (True).
    """
    thresholds = pd.to_datetime(list_dates.reindex(tickers)) + pd.DateOffset(months=min_months)
    seasoned = dates.values[:, None] >= thresholds.values[None, :]  # (n_dates, n_tickers)
    out = pd.DataFrame(seasoned, index=dates, columns=tickers)

    unknown = thresholds.isna().to_numpy()  # NaT compares False -> override to True
    if unknown.any():
        out.loc[:, unknown] = True
    return out


# --------------------------------------------------------------------------- #
# Cache (de)serialization — tidy long form for parquet round-trip
# --------------------------------------------------------------------------- #
def _stack_store(m: MarketData) -> pd.DataFrame:
    store = pd.DataFrame(
        {
            "prices": m.prices.stack(future_stack=True),
            "volume": m.volume.stack(future_stack=True),
            "market_cap": m.market_cap.stack(future_stack=True),
            "returns": m.returns.stack(future_stack=True),
            "universe_mask": m.universe_mask.stack(future_stack=True),
        }
    )
    store.index = store.index.set_names(["date", "ticker"])
    return store


def _unstack_store(store: pd.DataFrame) -> MarketData:
    return MarketData(
        prices=store["prices"].unstack(),
        volume=store["volume"].unstack(),
        market_cap=store["market_cap"].unstack(),
        returns=store["returns"].unstack(),
        universe_mask=store["universe_mask"].unstack().astype(bool),
    )
