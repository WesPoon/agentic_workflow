"""Deterministic, offline strategy used by tests and smoke runs of the worker."""
import numpy as np
import pandas as pd


def run_backtest(backtests_dir: str, params: dict) -> dict:
    seed = int(params.get("seed", 1))
    drift = float(params.get("drift", 0.0004))
    idx = pd.bdate_range("2012-01-02", periods=252 * 12)
    rng = np.random.default_rng(seed)
    bench = pd.Series(rng.normal(0.0003, 0.011, len(idx)), index=idx)
    ret = 0.5 * bench + pd.Series(rng.normal(drift, 0.008, len(idx)), index=idx)
    return {"returns": ret, "benchmark_returns": bench, "positions": None, "meta": {"source": "synthetic"}}
