"""Point-in-time monthly panels of ALL US common stocks (Polygon), read from the local cache only.

This is the fixed data library for cross-sectional US equity strategies. Strategy code must use it
instead of hand-picked ticker lists (a list of today's large caps is survivorship/hindsight bias).

    from src.data.us_equity_panels import load_us_equity_panels
    panels = load_us_equity_panels(start="2004-01-01")
    panels.returns          # month-end x ticker, simple monthly returns
    panels.universe         # month-end x ticker, bool: tradable that month (point-in-time filters)

What the data is (state these limitations in meta["assumptions"]):
- Source: Polygon grouped daily bars at each month-end, US common stock (type CS), active AND
  delisted tickers, 2003-12 .. 2026-05. ~11.5k tickers, ~1.8k eligible per month (median).
- Prices are split-adjusted closes; returns EXCLUDE dividends (price returns). The SPY benchmark
  from `load_benchmark_returns()` is a total return, so long-only strategies are slightly
  penalised vs the benchmark.
- No delisting returns: a ticker's series simply stops. If a stock you hold stops trading, its
  final partial month is unknown. `delisting_return` lets you impute it (CRSP practice: -30% when
  the delisting return is missing), but ticker changes and mergers also look like delistings
  here, so imputing a loss also hits renamed and acquired names. Choose and record it.
- Six month-ends fell on market holidays and are empty in the cache (2004-05, 2010-05, 2013-03,
  2018-03, 2021-05, 2024-03). Their prices are imputed as the geometric midpoint of the adjacent
  month-ends (so the two-month return is split evenly and compounding is exact); eligibility in
  those months is carried from the previous month. `imputed_months` lists them.
- No reliable point-in-time market cap: Polygon only gives today's share count, and none for
  most delisted names. Use `dollar_volume` (price x monthly volume, point-in-time) as the size /
  liquidity proxy. `market_cap_current_shares` is provided for live names only and is NOT
  point-in-time; do not use it for historical size sorts.

The universe mask uses only information available at each month-end:
  price > 0 · listed >= `min_months_listed` months (months since the first observed price)
  · monthly return within [-95%, +300%] · not stale (zero volume and zero return this AND last month)
  · trailing `dollar_volume_months`-month mean dollar volume above the `dollar_volume_floor`
    percentile of that month's cross-section.

Benchmarks: `load_benchmark_returns()` gives SPY / QQQ monthly TOTAL returns (yfinance, dividends
reinvested) already on the panel's business month-end dates. Use it instead of resampling yourself.

The panel cache is never refreshed from here. Without it this raises; build it with src/bimodality
(`python -m src.bimodality.main`), which pulls Polygon once.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

RETURN_UPPER = 3.00
RETURN_LOWER = -0.95


@dataclass
class USEquityPanels:
    prices: pd.DataFrame                 # split-adjusted month-end close
    volume: pd.DataFrame                 # month-end-day volume as stored by the cache builder
    dollar_volume: pd.DataFrame          # trailing mean of price x volume (point-in-time size proxy)
    returns: pd.DataFrame                # simple monthly price returns (no dividends)
    universe: pd.DataFrame               # bool, point-in-time eligibility
    first_seen: pd.Series                # first month-end with a price, per ticker
    last_seen: pd.Series                 # last month-end with a price, per ticker
    market_cap_current_shares: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    imputed_months: list[pd.Timestamp] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.prices.index

    def eligible(self, date) -> pd.Index:
        row = self.universe.loc[pd.Timestamp(date)]
        return row.index[row.to_numpy()]


def cache_dir() -> Path:
    env = os.environ.get("US_EQUITY_CACHE_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[2] / "backtests" / "cache"


def _latest_panels_file(directory: Path) -> Path:
    files = sorted(directory.glob("panels_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"no cached Polygon panels in {directory}. Build them once with src/bimodality "
            "(it pulls Polygon), or set US_EQUITY_CACHE_DIR.")

    def span(p: Path) -> tuple[pd.Timestamp, pd.Timestamp]:
        _, start, end = p.stem.split("_", 2)
        return pd.Timestamp(start), pd.Timestamp(end)

    # widest coverage: earliest start, then latest end
    return min(files, key=lambda p: (span(p)[0], -span(p)[1].value))


def load_us_equity_panels(
    start: str | None = None,
    end: str | None = None,
    *,
    min_months_listed: int = 3,
    dollar_volume_floor: float = 0.10,
    dollar_volume_months: int = 12,
    delisting_return: float | None = None,
) -> USEquityPanels:
    """Load the cached panels and build a point-in-time universe. Never calls the Polygon API.

    start / end: optional date bounds applied AFTER the filters are computed (so trailing windows
    near `start` still see earlier data).
    dollar_volume_floor: drop the bottom fraction by trailing dollar volume each month
      (e.g. 0.05 for Han & Qin's bottom-5% market-cap rule, 0.10 for a stricter liquidity floor).
    delisting_return: if set, the month after a ticker's last observed price (when that is before
      the end of the data) gets this return, e.g. -0.30. Default None leaves it missing.
    """
    path = _latest_panels_file(cache_dir())
    store = pd.read_parquet(path)
    prices = store["prices"].unstack().sort_index()
    # every business month-end in range, so a month-end missing from the cache becomes an empty row
    calendar = pd.date_range(prices.index[0], prices.index[-1], freq="BME").union(prices.index)
    prices = prices.reindex(calendar)
    volume = store["volume"].unstack().reindex_like(prices)
    mcap_now = store["market_cap"].unstack().reindex_like(prices)

    prices = prices.where(prices > 0)
    prices, volume, imputed = _fill_empty_months(prices, volume)
    returns = prices.pct_change(fill_method=None)
    has_price = prices.notna()
    first_seen = prices.apply(lambda c: c.first_valid_index())
    last_seen = prices.apply(lambda c: c.last_valid_index())

    months_listed = has_price.cumsum().where(has_price)            # 1 on the first priced month
    seasoned = months_listed > min_months_listed

    ret_ok = returns.isna() | ((returns <= RETURN_UPPER) & (returns >= RETURN_LOWER))

    zero = (volume.fillna(0) == 0) & (returns.fillna(0) == 0)
    stale = zero & zero.shift(1, fill_value=False)

    dollar_volume = (prices * volume).rolling(dollar_volume_months, min_periods=1).mean().where(has_price)
    dv_rank = dollar_volume.rank(axis=1, pct=True)
    liquid = dv_rank > dollar_volume_floor

    universe = (has_price & seasoned & ret_ok & ~stale & liquid).fillna(False).astype(bool)
    for d in imputed:                                   # carry eligibility through holiday holes
        i = universe.index.get_loc(d)
        if i > 0:
            universe.iloc[i] = universe.iloc[i - 1] & has_price.iloc[i]

    if delisting_return is not None:
        end_of_data = prices.index[-1]
        for ticker, last in last_seen.items():
            if last is None or pd.isna(last) or last >= end_of_data:
                continue
            nxt = prices.index[prices.index.get_loc(last) + 1]
            if pd.isna(returns.at[nxt, ticker]):
                returns.at[nxt, ticker] = delisting_return

    lo = pd.Timestamp(start) if start else prices.index[0]
    hi = pd.Timestamp(end) if end else prices.index[-1]
    window = (prices.index >= lo) & (prices.index <= hi)

    def cut(df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[window]

    notes = [
        f"source: {path.name} (Polygon grouped daily, US common stock incl. delisted)",
        "returns are split-adjusted price returns; dividends excluded",
        "size proxy: trailing mean dollar volume; no point-in-time market cap",
        f"universe: listed > {min_months_listed} months, |ret| filter, not stale, "
        f"dollar volume above the {dollar_volume_floor:.0%} percentile",
        f"delisting return: {'not imputed' if delisting_return is None else delisting_return}",
        "holiday month-ends imputed (geometric midpoint of neighbours): "
        + ", ".join(d.strftime("%Y-%m") for d in imputed),
    ]
    return USEquityPanels(
        prices=cut(prices), volume=cut(volume), dollar_volume=cut(dollar_volume), returns=cut(returns),
        universe=cut(universe), first_seen=first_seen, last_seen=last_seen,
        market_cap_current_shares=cut(mcap_now), imputed_months=imputed, notes=notes,
    )


def _fill_empty_months(prices: pd.DataFrame, volume: pd.DataFrame
                       ) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.Timestamp]]:
    """Impute month-ends with no prices at all (holiday month-ends missing from the cache).

    Price = geometric midpoint of the neighbouring month-ends, per ticker where both exist;
    volume = mean of the neighbours. Only interior months with priced neighbours are filled."""
    prices, volume = prices.copy(), volume.copy()
    counts = prices.notna().sum(axis=1)
    imputed = []
    for i in range(1, len(prices) - 1):
        if counts.iloc[i] == 0 and counts.iloc[i - 1] > 0 and counts.iloc[i + 1] > 0:
            prev, nxt = prices.iloc[i - 1], prices.iloc[i + 1]
            prices.iloc[i] = np.sqrt(prev * nxt)
            volume.iloc[i] = (volume.iloc[i - 1] + volume.iloc[i + 1]) / 2
            imputed.append(prices.index[i])
    return prices, volume, imputed


BENCHMARK_FILE = "benchmarks_daily.parquet"


def load_benchmark_returns(tickers: tuple[str, ...] = ("SPY", "QQQ"), start: str | None = None,
                           end: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """Monthly total returns of benchmark ETFs on the panel's business month-end dates.

    Daily dividend-adjusted closes come from yfinance once and are cached to
    `<cache>/benchmarks_daily.parquet`; later calls are offline. A month's return is
    last close of the month / last close of the previous month - 1, labelled with that
    month's last business day (the same dates as `load_us_equity_panels().dates`).
    """
    path = cache_dir() / BENCHMARK_FILE
    daily = _read_benchmark_cache(path) if path.exists() and not refresh else pd.DataFrame()
    missing = [t for t in tickers if t not in daily.columns]
    if missing or daily.empty:
        import yfinance as yf

        raw = yf.download(sorted(set(tickers) | set(daily.columns)), start="2003-12-01",
                          auto_adjust=True, progress=False)["Close"]
        if isinstance(raw, pd.Series):
            raw = raw.to_frame(tickers[0])
        daily = raw.sort_index()
        daily.index = pd.DatetimeIndex(daily.index).tz_localize(None).normalize()
        path.parent.mkdir(parents=True, exist_ok=True)
        # dates as ISO text: the parquet engine here mangles non-nanosecond datetime indexes
        out = daily.copy()
        out.index = out.index.strftime("%Y-%m-%d")
        out.index.name = "date"
        out.to_parquet(path)
    closes = daily[list(tickers)].dropna(how="all")
    last_day = closes.index[-1]
    if last_day < pd.offsets.BMonthEnd().rollforward(last_day):    # drop an unfinished current month
        closes = closes[closes.index.to_period("M") < last_day.to_period("M")]
    monthly = closes.groupby(closes.index.to_period("M")).last().pct_change()
    month_end = pd.date_range(monthly.index[0].to_timestamp(), monthly.index[-1].to_timestamp("M"), freq="BME")
    label = pd.Series(month_end, index=month_end.to_period("M"))
    monthly.index = pd.DatetimeIndex(label.reindex(monthly.index).to_numpy())
    monthly = monthly.dropna(how="all")
    lo = pd.Timestamp(start) if start else monthly.index[0]
    hi = pd.Timestamp(end) if end else monthly.index[-1]
    return monthly.loc[(monthly.index >= lo) & (monthly.index <= hi)]


def _read_benchmark_cache(path: Path) -> pd.DataFrame:
    """Read the daily benchmark cache; an unreadable or implausible file counts as missing."""
    try:
        daily = pd.read_parquet(path)
        daily.index = pd.DatetimeIndex(pd.to_datetime(daily.index.astype(str)))
    except (ValueError, TypeError, OSError):
        return pd.DataFrame()
    if len(daily) == 0 or daily.index.min() < pd.Timestamp("2000-01-01"):
        return pd.DataFrame()                 # e.g. a file written with mangled dates: refetch
    return daily.sort_index()
