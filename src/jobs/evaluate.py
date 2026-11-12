"""Fixed evaluator. Every attempt is scored here, the same way, from a daily return series.

The agent never computes its own headline metrics. Inputs are net-of-cost daily strategy returns
and benchmark returns on the same calendar. Outputs: in-sample / out-of-sample / full-period
metrics (the CLAUDE.md table), a yearly sub-period table, a bootstrap 95% interval for the
out-of-sample Sharpe, and a deflated Sharpe ratio that accounts for the number of attempts.
"""
from __future__ import annotations

import logging
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329
METRIC_NAMES = [
    "sharpe", "sortino", "calmar", "cagr", "max_drawdown", "max_drawdown_days", "volatility",
    "win_rate", "avg_win", "avg_loss", "beta", "alpha", "value_at_risk", "total_return", "n_days",
]


# ----------------------------------------------------------------------------- helpers

def _clean(series: pd.Series, name: str) -> pd.Series:
    if not isinstance(series, pd.Series):
        raise TypeError(f"{name} must be a pandas Series of daily returns")
    s = pd.Series(series).astype(float)
    s.index = pd.to_datetime(s.index)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.replace([np.inf, -np.inf], np.nan)


def _safe(fn, *args, **kwargs) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val = fn(*args, **kwargs)
        if isinstance(val, pd.Series):
            val = val.iloc[0] if len(val) else np.nan
        val = float(val)
        return val if math.isfinite(val) else float("nan")
    except Exception as e:  # metric failures must not kill the evaluation
        log.debug("metric %s failed: %s", getattr(fn, "__name__", fn), e)
        return float("nan")


def _max_dd_days(ret: pd.Series) -> float:
    try:
        import quantstats as qs

        details = qs.stats.drawdown_details(qs.stats.to_drawdown_series(ret))
        return float(details["days"].max()) if len(details) else 0.0
    except Exception:
        # fallback: longest run below the running peak
        equity = (1 + ret.fillna(0)).cumprod()
        peak = equity.cummax()
        under = equity < peak
        longest = run = 0
        for flag in under:
            run = run + 1 if flag else 0
            longest = max(longest, run)
        return float(longest)


def _greek(ret: pd.Series, bench: pd.Series, name: str) -> float:
    """Beta / annualized alpha vs benchmark. quantstats exposes these via stats.greeks()."""
    import quantstats as qs

    def _calc():
        return qs.stats.greeks(ret, bench)[name]

    return _safe(_calc)


def segment_metrics(ret: pd.Series, bench: pd.Series) -> dict:
    import quantstats as qs

    ret = ret.fillna(0.0)
    bench = bench.reindex(ret.index).fillna(0.0)
    if len(ret) < 2:
        return {m: float("nan") for m in METRIC_NAMES} | {"n_days": len(ret)}
    return {
        "sharpe": _safe(qs.stats.sharpe, ret),
        "sortino": _safe(qs.stats.sortino, ret),
        "calmar": _safe(qs.stats.calmar, ret),
        "cagr": _safe(qs.stats.cagr, ret),
        "max_drawdown": _safe(qs.stats.max_drawdown, ret),
        "max_drawdown_days": _max_dd_days(ret),
        "volatility": _safe(qs.stats.volatility, ret),
        "win_rate": _safe(qs.stats.win_rate, ret),
        "avg_win": _safe(qs.stats.avg_win, ret),
        "avg_loss": _safe(qs.stats.avg_loss, ret),
        "beta": _greek(ret, bench, "beta"),
        "alpha": _greek(ret, bench, "alpha"),
        "value_at_risk": _safe(qs.stats.value_at_risk, ret),
        "total_return": float((1 + ret).prod() - 1),
        "n_days": int(len(ret)),
    }


def annualized_sharpe(ret: pd.Series) -> float:
    r = ret.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * math.sqrt(TRADING_DAYS))


def bootstrap_sharpe_ci(ret: pd.Series, n_boot: int = 1000, seed: int = 7) -> tuple[float, float]:
    r = ret.dropna().to_numpy()
    if len(r) < 20:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(r), size=(n_boot, len(r)))
    samples = r[idx]
    std = samples.std(axis=1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(std > 0, samples.mean(axis=1) / std * math.sqrt(TRADING_DAYS), np.nan)
    lo, hi = np.nanpercentile(sr, [2.5, 97.5])
    return (float(lo), float(hi))


def deflated_sharpe(ret: pd.Series, n_trials: int, prior_sharpes_annualized: list[float] | None = None) -> dict:
    """Bailey & López de Prado (2014) deflated Sharpe ratio.

    Returns the probability that the true Sharpe exceeds zero after `n_trials` attempts, and the
    expected maximum Sharpe (annualized) that pure noise would produce across that many trials.
    The variance of Sharpe across trials is taken from prior attempts when at least two exist,
    otherwise from the estimator variance of this series (an approximation; stated in the output).
    """
    from scipy.stats import kurtosis, norm, skew

    r = ret.dropna()
    T = len(r)
    if T < 20 or r.std(ddof=1) == 0:
        return {"deflated_sharpe_prob": float("nan"), "sr0_annualized": float("nan"),
                "n_trials": int(n_trials), "variance_source": "insufficient_data"}
    sr = float(r.mean() / r.std(ddof=1))          # daily, not annualized
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))          # non-excess kurtosis
    denom = 1 - g3 * sr + (g4 - 1) / 4 * sr * sr
    denom = max(denom, 1e-12)
    est_var = denom / max(T - 1, 1)

    priors = [p / math.sqrt(TRADING_DAYS) for p in (prior_sharpes_annualized or []) if p is not None and math.isfinite(p)]
    if len(priors) >= 2:
        var_trials, source = float(np.var(priors, ddof=1)), "prior_attempts"
    else:
        var_trials, source = est_var, "estimator_variance"

    N = max(int(n_trials), 1)
    if N == 1:
        sr0 = 0.0
    else:
        z1 = norm.ppf(1 - 1 / N)
        z2 = norm.ppf(1 - 1 / (N * math.e))
        sr0 = math.sqrt(max(var_trials, 0.0)) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    stat = (sr - sr0) * math.sqrt(max(T - 1, 1)) / math.sqrt(denom)
    return {
        "deflated_sharpe_prob": float(norm.cdf(stat)),
        "sr0_annualized": float(sr0 * math.sqrt(TRADING_DAYS)),
        "n_trials": N,
        "variance_source": source,
    }


def yearly_table(ret: pd.Series, bench: pd.Series) -> list[dict]:
    rows = []
    bench = bench.reindex(ret.index).fillna(0.0)
    for year, r in ret.fillna(0.0).groupby(ret.index.year):
        b = bench.loc[r.index]
        equity = (1 + r).cumprod()
        rows.append({
            "year": int(year),
            "n_days": int(len(r)),
            "return": float(equity.iloc[-1] - 1),
            "benchmark_return": float((1 + b).prod() - 1),
            "sharpe": annualized_sharpe(r),
            "max_drawdown": float((equity / equity.cummax() - 1).min()),
        })
    return rows


# ----------------------------------------------------------------------------- main entry

def evaluate(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    n_trials: int = 1,
    prior_oos_sharpes: list[float] | None = None,
    split: float = 0.7,
) -> dict:
    ret = _clean(returns, "returns").dropna()
    if len(ret) < 2 * TRADING_DAYS:
        log.warning("only %d days of returns; metrics will be unstable", len(ret))
    bench = _clean(benchmark_returns, "benchmark_returns").reindex(ret.index).fillna(0.0)

    n_is = int(len(ret) * split)
    is_ret, oos_ret = ret.iloc[:n_is], ret.iloc[n_is:]
    is_b, oos_b = bench.iloc[:n_is], bench.iloc[n_is:]

    oos_ci = bootstrap_sharpe_ci(oos_ret)
    dsr = deflated_sharpe(oos_ret, n_trials, prior_oos_sharpes)

    def _period(s: pd.Series) -> list[str]:
        return [s.index[0].strftime("%Y-%m-%d"), s.index[-1].strftime("%Y-%m-%d")] if len(s) else ["", ""]

    return {
        "split": split,
        "is_period": _period(is_ret),
        "oos_period": _period(oos_ret),
        "n_days": int(len(ret)),
        "in_sample": segment_metrics(is_ret, is_b),
        "out_of_sample": segment_metrics(oos_ret, oos_b),
        "full": segment_metrics(ret, bench),
        "benchmark": {
            "in_sample": segment_metrics(is_b, is_b),
            "out_of_sample": segment_metrics(oos_b, oos_b),
        },
        "oos_sharpe_ci95": list(oos_ci),
        **dsr,
        "subperiods": yearly_table(ret, bench),
    }


def flatten_metrics(result: dict) -> pd.DataFrame:
    rows = []
    for segment in ("in_sample", "out_of_sample", "full"):
        for k, v in result.get(segment, {}).items():
            rows.append({"segment": segment, "metric": k, "value": v})
    for segment in ("in_sample", "out_of_sample"):
        for k, v in result.get("benchmark", {}).get(segment, {}).items():
            rows.append({"segment": f"benchmark_{segment}", "metric": k, "value": v})
    for k in ("deflated_sharpe_prob", "sr0_annualized", "n_trials", "variance_source", "split"):
        rows.append({"segment": "out_of_sample", "metric": k, "value": result.get(k)})
    lo, hi = result.get("oos_sharpe_ci95", [None, None])
    rows.append({"segment": "out_of_sample", "metric": "sharpe_ci95_low", "value": lo})
    rows.append({"segment": "out_of_sample", "metric": "sharpe_ci95_high", "value": hi})
    return pd.DataFrame(rows)


def write_reports(result: dict, returns: pd.Series, benchmark_returns: pd.Series, out_dir: Path, title: str) -> dict:
    """Write metrics.csv, subperiods.csv, returns.csv and the quantstats tearsheet. Returns file names."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    flatten_metrics(result).to_csv(out_dir / "metrics.csv", index=False)
    files["metrics"] = "metrics.csv"
    pd.DataFrame(result.get("subperiods", [])).to_csv(out_dir / "subperiods.csv", index=False)
    files["subperiods"] = "subperiods.csv"
    ret = _clean(returns, "returns")
    bench = _clean(benchmark_returns, "benchmark_returns").reindex(ret.index)
    pd.DataFrame({"strategy": ret, "benchmark": bench}).to_csv(out_dir / "returns.csv")
    files["returns"] = "returns.csv"
    try:
        import quantstats as qs

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            qs.reports.html(ret.fillna(0.0), benchmark=bench.fillna(0.0), output=str(out_dir / "tearsheet.html"),
                            title=title, download_filename=str(out_dir / "tearsheet.html"))
        files["tearsheet"] = "tearsheet.html"
    except Exception as e:
        log.warning("tearsheet failed: %s", e)
        files["tearsheet_error"] = repr(e)
    return files
