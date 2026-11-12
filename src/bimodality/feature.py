"""Feature layer — monthly panels -> tidy 16-feature ML matrix.

Implements the Han (2022) feature set:

    zMOM_{1,3,6,9,12}  cross-sectional z-scores of momentum
    MMOM_{1,3,6,9,12}  cross-sectional mean of each momentum (market regime)
    SMOM_{1,3,6,9,12}  cross-sectional std  of each momentum (dispersion)
    size_decile        market-cap decile 1-10 (categorical)

The target is the forward 1-month return decile (0..9, decile 1 = highest return).
All momentum at month ``t`` is built from returns realized at or before ``t`` (the
12-month lookback additionally skips the most recent month). The target peeks one
month ahead by construction and is therefore present only on training rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import constant as C
from .data import MarketData
from .util import cross_sectional_decile, cross_sectional_zscore, get_logger

logger = get_logger("feature")

# All multi-month momenta skip the most recent month (paper formula:
# MOM_m = prod_{j=t-m..t-1}(1+r) - 1), avoiding 1-month reversal contamination.
# MOM_1 is the current month itself (the reversal term), so it does not skip.
_SKIP_MONTHS: dict[int, int] = {1: 0, 3: 1, 6: 1, 9: 1, 12: 1}


def _cumulative_return(gross: pd.DataFrame, window: int, skip: int) -> pd.DataFrame:
    """Product of gross monthly returns over ``window`` months ending ``skip`` months ago."""
    shifted = gross.shift(skip)
    return shifted.rolling(window).apply(np.prod, raw=True) - 1.0


def compute_momentum_panels(returns: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Return ``{lookback -> MOM panel}`` for every configured lookback."""
    gross = 1.0 + returns
    return {
        m: _cumulative_return(gross, window=m, skip=_SKIP_MONTHS[m])
        for m in C.LOOKBACKS
    }


def compute_vol_panels(returns: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Return ``{lookback -> trailing return-vol panel}`` matching each MOM window.

    Volatility is the std of monthly returns over the same window/skip as the
    momentum, so ``MOM_m / vol_m`` is a like-for-like risk-adjusted (t-stat) momentum.
    """
    return {
        m: returns.shift(_SKIP_MONTHS[m]).rolling(m).std(ddof=0)
        for m in C.VOL_ADJ_LOOKBACKS
    }


def build_feature_matrix(market: MarketData, cfg: C.StrategyConfig) -> pd.DataFrame:
    """Build the tidy ``(date, ticker)`` feature matrix + ``target_decile``.

    Only universe-eligible stocks are kept per month, and cross-sectional stats
    (z-score, mean, std, size decile, target decile) are computed on that eligible
    set so filtered names never contaminate the normalization.
    """
    mom_panels = compute_momentum_panels(market.returns)
    vol_panels = (compute_vol_panels(market.returns)
                  if cfg.use_vol_adjusted_momentum else None)
    forward_return = market.returns.shift(-1)  # realized over the held month

    rows: list[pd.DataFrame] = []
    for date in market.dates:
        eligible = market.eligible_tickers(date)
        if len(eligible) < C.N_DECILES * 2:  # too thin to rank into deciles
            continue
        rows.append(_features_for_month(date, eligible, mom_panels, vol_panels,
                                        market, forward_return))

    if not rows:
        logger.warning("No eligible months produced features.")
        return pd.DataFrame(columns=[*C.FEATURE_NAMES, C.TARGET_COLUMN])

    matrix = pd.concat(rows).sort_index()
    logger.info("Built feature matrix: %d rows over %d months",
                len(matrix), matrix.index.get_level_values(0).nunique())
    return matrix


def _features_for_month(
    date: pd.Timestamp,
    eligible: pd.Index,
    mom_panels: dict[int, pd.DataFrame],
    vol_panels: dict[int, pd.DataFrame] | None,
    market: MarketData,
    forward_return: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble one month's eligible cross-section into feature rows."""
    out = pd.DataFrame(index=eligible)

    if vol_panels is None:
        # Baseline: raw momentum (z-score / cross-sectional mean / std).
        for m in C.LOOKBACKS:
            mom = mom_panels[m].loc[date, eligible]
            out[f"zMOM_{m}"] = cross_sectional_zscore(mom)
            out[f"MMOM_{m}"] = mom.mean()      # broadcast scalar (regime level)
            out[f"SMOM_{m}"] = mom.std(ddof=0)  # broadcast scalar (dispersion)
    else:
        # Vol-adjusted momentum REPLACES raw: MOM_m / sigma_m, same z/mean/std forms.
        # No 1-month variant (vol is undefined over a single observation).
        for m in C.VOL_ADJ_LOOKBACKS:
            vol = vol_panels[m].loc[date, eligible]
            vmom = mom_panels[m].loc[date, eligible] / vol.where(vol > 0)  # NaN if vol<=0
            out[f"zvMOM_{m}"] = cross_sectional_zscore(vmom)
            out[f"MvMOM_{m}"] = vmom.mean()
            out[f"SvMOM_{m}"] = vmom.std(ddof=0)

    out[C.SIZE_FEATURE] = cross_sectional_decile(
        market.market_cap.loc[date, eligible], ascending=True
    )  # 1 = smallest, 10 = largest

    fwd = forward_return.loc[date, eligible]
    decile = cross_sectional_decile(fwd, ascending=False)  # 1 = highest return
    out[C.TARGET_COLUMN] = decile - 1                       # XGBoost labels 0..9

    out.index = pd.MultiIndex.from_product([[date], eligible], names=["date", "ticker"])
    return out


def split_features(
    matrix: pd.DataFrame, cfg: C.StrategyConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the matrix into in-sample (<= in_sample_end) and OOS (>= out_sample_start)."""
    dates = matrix.index.get_level_values("date")
    in_sample = matrix[dates <= pd.Timestamp(cfg.in_sample_end)]
    out_sample = matrix[dates >= pd.Timestamp(cfg.out_sample_start)]
    return in_sample, out_sample
