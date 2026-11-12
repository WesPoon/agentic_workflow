"""CLI trigger for the bimodality Deep Momentum pipeline.

Runs the 0->4 pipeline (data -> features -> strategy -> report) for one config,
or a lightweight self-contained smoke test on synthetic data (no Polygon needed).

Usage
-----
    # full run against Polygon Massive (needs MASSIVE_API_KEY)
    python -m bimodality.main --name deep_momentum --start 2004-01-01 --end 2025-12-31

    # offline plumbing check on synthetic data
    python -m bimodality.main --smoke
"""
from __future__ import annotations

import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from . import constant as C
from .data import MarketData, load_market_data
from .feature import build_feature_matrix
from .report_generator import generate_report
from .strategy import DeepMomentumStrategy, attach_forward_returns
from .util import get_logger

logger = get_logger("main")


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def load_benchmarks_monthly(start: str, end: str) -> dict[str, pd.Series]:
    """Monthly total returns for every ticker in ``C.BENCHMARK_TICKERS``.

    All tickers are pulled in a single grouped-daily pass per month-end (no extra
    API cost over one benchmark). Falls back to yfinance if Massive is unavailable.
    Returns a dict keyed by ticker; the first ticker is the primary benchmark.
    """
    from .data import _month_end_trading_days  # reuse the month-end calendar

    tickers = list(C.BENCHMARK_TICKERS)
    wanted = set(tickers)
    try:
        import os

        from massive import RESTClient

        key = os.environ.get("POLYGON_API_KEY") or os.environ.get("MASSIVE_API_KEY")
        client = RESTClient(key)
        closes: dict[str, dict] = {t: {} for t in tickers}
        for dt in _month_end_trading_days(start, end):
            aggs = client.get_grouped_daily_aggs(dt.strftime("%Y-%m-%d"), adjusted=True)
            for a in aggs:
                tkr = getattr(a, "ticker", None)
                if tkr in wanted:
                    closes[tkr][dt] = a.close
        out = {}
        for t in tickers:
            s = pd.Series(closes[t]).sort_index()
            out[t] = s.pct_change().dropna()
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("Polygon benchmark load failed (%s); trying yfinance", exc)
        import yfinance as yf

        px = yf.download(tickers, start=start, end=end,
                         auto_adjust=True, progress=False)["Close"]
        monthly = px.resample("BME").last().pct_change().dropna(how="all")
        if isinstance(monthly, pd.Series):  # single ticker
            return {tickers[0]: monthly.dropna()}
        return {t: monthly[t].dropna() for t in tickers if t in monthly}


# --------------------------------------------------------------------------- #
# Runners
# --------------------------------------------------------------------------- #
def run_full(cfg: C.StrategyConfig, refresh: bool) -> dict:
    """Full Polygon-backed pipeline for one config."""
    market = load_market_data(cfg, refresh=refresh)
    matrix = build_feature_matrix(market, cfg)
    matrix = attach_forward_returns(matrix, market)
    result = DeepMomentumStrategy(cfg).run(matrix, market)
    benchmarks = load_benchmarks_monthly(cfg.start, cfg.end)
    primary = benchmarks[C.BENCHMARK_TICKERS[0]]
    extras = {t: benchmarks[t] for t in C.BENCHMARK_TICKERS[1:] if t in benchmarks}
    report = generate_report(result, primary, cfg, extra_benchmarks=extras)
    _print_summary(report)
    return report


def run_smoke() -> dict:
    """Offline plumbing check: synthetic panels -> features -> backtest -> report."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2004-01-31", "2021-12-31", freq="BME")
    tickers = [f"T{i:03d}" for i in range(120)]

    rets = pd.DataFrame(rng.normal(0.01, 0.06, (len(dates), len(tickers))),
                        index=dates, columns=tickers)
    prices = 100 * (1 + rets).cumprod()
    volume = pd.DataFrame(rng.integers(1e5, 1e6, (len(dates), len(tickers))).astype(float),
                          index=dates, columns=tickers)
    market_cap = prices * rng.integers(1e6, 1e8, len(tickers))
    mask = pd.DataFrame(True, index=dates, columns=tickers)
    market = MarketData(prices, volume, market_cap, prices.pct_change(), mask)

    cfg = C.StrategyConfig(name="smoke", n_training_runs=2, out_sample_start="2018-01-01")
    matrix = build_feature_matrix(market, cfg)
    matrix = attach_forward_returns(matrix, market)
    result = DeepMomentumStrategy(cfg).run(matrix, market, backtest_start="2018-01-01")

    benchmark = market.returns.mean(axis=1)  # equal-weight market proxy
    report = generate_report(result, benchmark, cfg)
    _print_summary(report)
    return report


def _print_summary(report: dict) -> None:
    gate = report["pm_gate"]
    logger.info("PM gate: %s", "PASS" if gate.passed else "FAIL")
    for check, ok in gate.checks.items():
        logger.info("  [%s] %s", "x" if ok else " ", check)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bimodality Deep Momentum pipeline")
    p.add_argument("--name", default="deep_momentum")
    p.add_argument("--start", default=C.IN_SAMPLE_START)
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--bimodality", action="store_true",
                   help="ENABLE the BM gross-sizing overlay (OFF by default — not in "
                        "the papers; diagnostic only). Trades Sharpe for lower drawdown.")
    p.add_argument("--vol-momentum", action="store_true",
                   help="add vol-adjusted momentum features (zvMOM/MvMOM/SvMOM) "
                        "alongside raw momentum (A/B experiment).")
    p.add_argument("--refresh", action="store_true", help="rebuild cached panels")
    p.add_argument("--smoke", action="store_true", help="offline synthetic smoke test")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if args.smoke:
        run_smoke()
        return
    # Timestamp the run name so artifacts are archived per-run (no overwrite).
    run_name = f"{args.name}_{datetime.now():%Y%m%d_%H%M%S}"
    cfg = C.StrategyConfig(
        name=run_name,
        start=args.start,
        end=args.end,
        use_bimodality_filter=args.bimodality,
        use_vol_adjusted_momentum=args.vol_momentum,
    )
    logger.info("Run name: %s", run_name)
    run_full(cfg, refresh=args.refresh)


if __name__ == "__main__":
    main()
