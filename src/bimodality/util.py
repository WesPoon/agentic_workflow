"""Stateless helpers shared across the bimodality package.

Pure cross-sectional transforms (operate on one month's slice at a time) plus a
small logging factory. No I/O, no strategy logic, no global state.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import constant as C

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
_LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a configured module logger (idempotent)."""
    logger = logging.getLogger(f"bimodality.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_rebalance(
    logger: logging.Logger,
    date: pd.Timestamp,
    n_long: int,
    n_short: int,
    turnover: float,
) -> None:
    """Standard rebalance-event log line (date, sizes, turnover)."""
    logger.info(
        "REBALANCE %s | long=%d short=%d turnover=%.3f",
        date.date(),
        n_long,
        n_short,
        turnover,
    )


# --------------------------------------------------------------------------- #
# Cross-sectional transforms (one month's cross-section at a time)
# --------------------------------------------------------------------------- #
def cross_sectional_zscore(s: pd.Series) -> pd.Series:
    """Z-score a single cross-section. Zero/NaN std -> all zeros (no signal)."""
    mu = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=s.index)
    return (s - mu) / sd


def cross_sectional_decile(
    s: pd.Series,
    n: int = C.N_DECILES,
    ascending: bool = False,
) -> pd.Series:
    """Rank a cross-section into ``1..n`` deciles.

    ``ascending=False`` (default) => label 1 is the *highest* value, matching the
    "decile 1 = highest forward return" convention. NaNs propagate as NaN.
    """
    valid = s.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=s.index)
    ranks = valid.rank(ascending=ascending, method="first")
    labels = np.ceil(ranks / len(valid) * n).clip(1, n).astype(int)
    return labels.reindex(s.index)


def winsorize(
    s: pd.Series,
    lower: float = C.WINSOR_LOWER,
    upper: float = C.WINSOR_UPPER,
) -> pd.Series:
    """Clip a series to its [lower, upper] empirical quantiles."""
    valid = s.dropna()
    if valid.empty:
        return s
    lo, hi = valid.quantile(lower), valid.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def slice_period(s: pd.Series, start: str, end: str) -> pd.Series:
    """Inclusive date-slice of a DatetimeIndex series."""
    return s.loc[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
