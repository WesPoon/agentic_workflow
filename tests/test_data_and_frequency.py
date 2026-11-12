"""Offline tests: the US equity panel library (synthetic cache) and monthly-frequency evaluation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _write_cache(tmp_path, months=30):
    dates = pd.date_range("2010-01-31", periods=months, freq="BME")
    rng = np.random.default_rng(1)
    rows = []
    for t in ["LIVE", "DEAD", "THIN"]:
        price = 10.0
        for i, d in enumerate(dates):
            if t == "DEAD" and i >= 20:          # delisted after month 20
                continue
            price *= 1 + rng.normal(0.01, 0.05)
            vol = 1e4 if t == "THIN" else 1e6
            rows.append({"date": d, "ticker": t, "prices": price, "volume": vol, "market_cap": np.nan,
                         "returns": np.nan, "universe_mask": False})
    df = pd.DataFrame(rows).set_index(["date", "ticker"])
    hole = dates[10]                               # a holiday month-end missing from the cache
    df = df.drop(index=hole, level="date")
    df.to_parquet(tmp_path / "panels_2010-01-01_2012-12-31.parquet")
    return dates, hole


def test_us_equity_panels_point_in_time_universe(tmp_path, monkeypatch):
    dates, hole = _write_cache(tmp_path)
    monkeypatch.setenv("US_EQUITY_CACHE_DIR", str(tmp_path))
    from src.data.us_equity_panels import load_us_equity_panels

    p = load_us_equity_panels(dollar_volume_floor=0.34)
    assert p.imputed_months == [hole]                              # holiday hole filled
    assert p.prices.loc[hole].notna().all()
    assert not p.universe.iloc[:3].any().any()                     # seasoning: first 3 months excluded
    assert p.universe.loc[dates[5], "DEAD"] and p.universe.loc[dates[5], "LIVE"]
    assert not p.universe.loc[dates[3]:dates[19], "THIN"].any()    # bottom third by dollar volume
    assert p.universe.loc[dates[25], "THIN"]                       # percentile floor: alone with LIVE now
    assert not p.universe.loc[dates[25], "DEAD"]                   # gone after delisting
    assert p.last_seen["DEAD"] == dates[19]
    assert np.isnan(p.returns.loc[dates[20], "DEAD"])              # no delisting return by default

    q = load_us_equity_panels(dollar_volume_floor=0.34, delisting_return=-0.30)
    assert q.returns.loc[dates[20], "DEAD"] == -0.30


def test_evaluator_scores_monthly_and_daily_consistently():
    from src.jobs.evaluate import evaluate, infer_periods

    rng = np.random.default_rng(0)
    d = pd.bdate_range("2010-01-01", "2023-12-31")
    daily = pd.Series(rng.normal(0.0004, 0.01, len(d)), index=d)
    bench = pd.Series(rng.normal(0.0003, 0.011, len(d)), index=d)
    monthly = (1 + daily).groupby(d.to_period("M")).prod() - 1
    monthly.index = monthly.index.to_timestamp("M")
    assert infer_periods(daily.index) == 252 and infer_periods(monthly.index) == 12
    rd, rm = evaluate(daily, bench), evaluate(monthly, bench)
    assert rm["frequency"] == "monthly" and rm["periods_per_year"] == 12
    assert abs(rd["full"]["sharpe"] - rm["full"]["sharpe"]) < 0.15     # same series, same Sharpe
    assert abs(rd["full"]["cagr"] - rm["full"]["cagr"]) < 0.005
    with pytest.raises(ValueError, match="daily or monthly"):
        infer_periods(pd.date_range("2010-01-01", periods=20, freq="W"))


def test_worker_accepts_monthly_returns(tmp_path):
    from src.jobs.worker import _check_contract

    m = pd.date_range("2010-01-31", periods=48, freq="ME")
    out = {"returns": pd.Series(0.01, index=m), "benchmark_returns": pd.Series(0.005, index=m)}
    _check_contract(out)
    short = {"returns": pd.Series(0.01, index=m[:20]), "benchmark_returns": pd.Series(0.005, index=m[:20])}
    with pytest.raises(ValueError, match="need >= 36"):
        _check_contract(short)


def test_benchmark_cache_round_trip_and_month_alignment(tmp_path, monkeypatch):
    """The benchmark cache keeps its dates (it once read back as 1970), drops an unfinished
    month, labels months with business month-ends, and the evaluator matches by month."""
    monkeypatch.setenv("US_EQUITY_CACHE_DIR", str(tmp_path))
    days = pd.bdate_range("2010-01-04", "2011-03-15")           # March 2011 is unfinished
    closes = pd.DataFrame({"SPY": np.linspace(100, 130, len(days)), "QQQ": np.linspace(50, 70, len(days))},
                          index=days.strftime("%Y-%m-%d"))
    closes.index.name = "date"
    closes.to_parquet(tmp_path / "benchmarks_daily.parquet")
    from src.data.us_equity_panels import load_benchmark_returns

    b = load_benchmark_returns()
    assert b.index[0] == pd.Timestamp("2010-02-26") and b.index[-1] == pd.Timestamp("2011-02-28")
    assert (b["SPY"] > 0).all() and len(b) == 13                 # Jan 2010 has no prior month; Mar 2011 dropped
    assert b.index.isin(pd.date_range("2010-01-01", "2011-03-31", freq="BME")).all()

    (tmp_path / "benchmarks_daily.parquet").write_bytes(b"not a parquet file")   # corrupt cache → refetch
    import sys, types
    fake = types.SimpleNamespace(download=lambda *a, **k: pd.concat(
        {"Close": closes.set_axis(pd.to_datetime(closes.index))}, axis=1))
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    assert load_benchmark_returns().equals(b)

    from src.jobs.evaluate import align_benchmark
    ret_idx = pd.DatetimeIndex(["2014-05-30", "2014-06-30"])          # business month-ends
    cal = pd.Series([0.02, 0.01], index=pd.DatetimeIndex(["2014-05-31", "2014-06-30"]))   # calendar
    assert np.allclose(align_benchmark(cal, ret_idx, 12), [0.02, 0.01])   # was [0.0, 0.01] before the fix
