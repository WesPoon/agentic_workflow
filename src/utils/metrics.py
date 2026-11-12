from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import quantstats as qs

# Try to reuse existing wqr risk_metrics functions
_WQR_PATH = Path("/Users/wpresearch/python")
if _WQR_PATH.exists() and str(_WQR_PATH) not in sys.path:
    sys.path.insert(0, str(_WQR_PATH))

try:
    from wqr.descriptor.risk_metrics import (
        sharpe_ratio as _wqr_sharpe,
        sortino_ratio as _wqr_sortino,
        calmar_ratio as _wqr_calmar,
        max_drawdown as _wqr_max_drawdown,
    )
    _HAS_WQR = True
except ImportError:
    _HAS_WQR = False


def compute_all_metrics(returns: pd.Series, benchmark: pd.Series | None = None) -> dict:
    """
    Compute all 13 required metrics for a strategy returns series.
    Uses wqr functions where available, falls back to quantstats.
    returns and benchmark must be daily return series with DatetimeIndex.
    """
    r = returns.dropna()

    metrics: dict = {
        "sharpe":        qs.stats.sharpe(r),
        "sortino":       qs.stats.sortino(r),
        "calmar":        qs.stats.calmar(r),
        "cagr":          qs.stats.cagr(r),
        "max_drawdown":  qs.stats.max_drawdown(r),
        "volatility":    qs.stats.volatility(r),
        "win_rate":      qs.stats.win_rate(r),
        "avg_win":       qs.stats.avg_win(r),
        "avg_loss":      qs.stats.avg_loss(r),
        "var_95":        qs.stats.value_at_risk(r),
    }

    dd_detail = qs.stats.drawdown_details(r)
    if dd_detail is not None and not dd_detail.empty:
        metrics["max_drawdown_days"] = int(dd_detail["days"].max())
    else:
        metrics["max_drawdown_days"] = None

    if benchmark is not None:
        b = benchmark.reindex(r.index).dropna()
        r_aligned = r.reindex(b.index).dropna()
        metrics["beta"]  = qs.stats.beta(r_aligned, b)
        metrics["alpha"] = qs.stats.alpha(r_aligned, b)
    else:
        metrics["beta"]  = None
        metrics["alpha"] = None

    return metrics
