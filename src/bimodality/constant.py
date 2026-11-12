"""Immutable configuration for the bimodality Deep Momentum strategy.

This module holds *only* constants and the frozen ``StrategyConfig`` dataclass.
No logic, no I/O, no global mutable state. Every other module imports its knobs
from here so the dataset constraints live in exactly one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
BACKTEST_DIR: Path = PROJECT_ROOT / "backtests"
CACHE_DIR: Path = BACKTEST_DIR / "cache"

# --------------------------------------------------------------------------- #
# Momentum lookbacks & feature names (Han 2022 16-feature set)
# --------------------------------------------------------------------------- #
LOOKBACKS: tuple[int, ...] = (1, 3, 6, 9, 12)

ZMOM_FEATURES: tuple[str, ...] = tuple(f"zMOM_{m}" for m in LOOKBACKS)
MMOM_FEATURES: tuple[str, ...] = tuple(f"MMOM_{m}" for m in LOOKBACKS)
SMOM_FEATURES: tuple[str, ...] = tuple(f"SMOM_{m}" for m in LOOKBACKS)
SIZE_FEATURE: str = "size_decile"

FEATURE_NAMES: tuple[str, ...] = (
    *ZMOM_FEATURES,
    *MMOM_FEATURES,
    *SMOM_FEATURES,
    SIZE_FEATURE,
)
TARGET_COLUMN: str = "target_decile"

# Vol-adjusted (risk-adjusted) momentum: MOM_m / sigma_m, treated with the same
# z-score / cross-sectional-mean / std forms. 1-month is skipped (no vol over 1 obs).
# Added only when ``cfg.use_vol_adjusted_momentum`` is True (A/B experiment).
VOL_ADJ_LOOKBACKS: tuple[int, ...] = (3, 6, 9, 12)
ZVMOM_FEATURES: tuple[str, ...] = tuple(f"zvMOM_{m}" for m in VOL_ADJ_LOOKBACKS)
MVMOM_FEATURES: tuple[str, ...] = tuple(f"MvMOM_{m}" for m in VOL_ADJ_LOOKBACKS)
SVMOM_FEATURES: tuple[str, ...] = tuple(f"SvMOM_{m}" for m in VOL_ADJ_LOOKBACKS)

# --------------------------------------------------------------------------- #
# Decile convention
# --------------------------------------------------------------------------- #
N_DECILES: int = 10                 # 10 forward-return classes
# Decile 1 = highest forward return ... Decile 10 = lowest.
# XGBoost class labels are 0..9 (= decile - 1).
DECILE_MEAN_LOOKBACK_YEARS: int = 10  # trailing window for mu_hat_k

# --------------------------------------------------------------------------- #
# Filters / dataset constraints
# --------------------------------------------------------------------------- #
MARKET_CAP_FLOOR_PCT: float = 0.10   # drop bottom 10% by cap each month
MIN_LISTING_MONTHS: int = 3          # exclude stocks listed < 3 months
RETURN_UPPER_CAP: float = 3.00       # drop monthly return > +300%
RETURN_LOWER_CAP: float = -0.95      # drop monthly return < -95%
WINSOR_LOWER: float = 0.01
WINSOR_UPPER: float = 0.99
DELIST_RETURN_ASSUMPTION: float = -0.30  # assumed return when delist data missing

# Concurrency for per-ticker reference fetches (get_ticker_details).
FETCH_MAX_WORKERS: int = 20

# --------------------------------------------------------------------------- #
# Periods
# --------------------------------------------------------------------------- #
IN_SAMPLE_START: str = "2004-01-01"
IN_SAMPLE_END: str = "2015-12-31"
OUT_SAMPLE_START: str = "2016-01-01"

# Named OOS stress sub-periods (start, end) for the report breakdown.
SUBPERIODS: dict[str, tuple[str, str]] = {
    "OOS_full":        (OUT_SAMPLE_START, "2100-12-31"),
    "2019":            ("2019-01-01", "2019-12-31"),
    "COVID_2020_2021": ("2020-01-01", "2021-12-31"),
    "INFLATION_2022_2024": ("2022-01-01", "2024-12-31"),
}

# --------------------------------------------------------------------------- #
# Trading economics
# --------------------------------------------------------------------------- #
TRANSACTION_COST_BPS: float = 10.0   # per trade, per leg
LONG_QUANTILE: float = 0.10          # top 10% by expected return
SHORT_QUANTILE: float = 0.10         # bottom 10% by expected return
BENCHMARK_TICKERS: list[str] = ["SPY", "QQQ"]  # first = primary (PM gate, tearsheet)

# --------------------------------------------------------------------------- #
# Bimodality regime overlay thresholds
# --------------------------------------------------------------------------- #
BM_HALF_SIZE_THRESHOLD: float = 0.0  # BM > 0.0 -> scale positions to 50%
BM_FLAT_THRESHOLD: float = 0.1       # BM > 0.1 -> go flat

# --------------------------------------------------------------------------- #
# XGBoost / training
# --------------------------------------------------------------------------- #
N_TRAINING_RUNS: int = 100           # train N times, average probabilities
VALIDATION_FRACTION: float = 0.20    # random 80/20 split within training window
RETRAIN_MONTH: int = 1               # retrain every January (expanding window)

XGB_PARAMS: dict = {
    "objective": "multi:softprob",
    "num_class": N_DECILES,
    "max_depth": 5,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
}
XGB_NUM_ROUNDS: int = 300
XGB_EARLY_STOPPING_ROUNDS: int = 30


@dataclass(frozen=True)
class StrategyConfig:
    """Frozen end-to-end configuration passed through every layer.

    Frozen by design: parameters must be fixed *before* OOS scoring. Construct a
    new config for an experiment rather than mutating one mid-run.
    """

    name: str = "deep_momentum"

    # date span to load / backtest
    start: str = IN_SAMPLE_START
    end: str = "2025-12-31"
    in_sample_end: str = IN_SAMPLE_END
    out_sample_start: str = OUT_SAMPLE_START

    # filters
    market_cap_floor_pct: float = MARKET_CAP_FLOOR_PCT
    min_listing_months: int = MIN_LISTING_MONTHS

    # portfolio
    long_quantile: float = LONG_QUANTILE
    short_quantile: float = SHORT_QUANTILE
    transaction_cost_bps: float = TRANSACTION_COST_BPS

    # model
    n_training_runs: int = N_TRAINING_RUNS
    xgb_params: dict = field(default_factory=lambda: dict(XGB_PARAMS))
    xgb_num_rounds: int = XGB_NUM_ROUNDS

    # data fetch
    fetch_max_workers: int = FETCH_MAX_WORKERS

    # overlays / toggles
    # BM gross-sizing is OFF by default: neither paper sizes positions by BM (it is
    # a diagnostic only). BM is still computed and reported. Enable only as an
    # optional drawdown-control switch that trades Sharpe for lower MaxDD.
    use_bimodality_filter: bool = False
    # Add vol-adjusted momentum features (zvMOM/MvMOM/SvMOM) alongside raw momentum.
    use_vol_adjusted_momentum: bool = False
    seed: int = 42

    @property
    def cache_dir(self) -> Path:
        return CACHE_DIR

    @property
    def backtest_dir(self) -> Path:
        return BACKTEST_DIR
