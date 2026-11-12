"""Bimodality Deep Momentum strategy package.

Layered, single-responsibility modules (see ``implementation_spec.md``):

    constant          frozen config + dataset constraints
    util              stateless cross-sectional helpers + logging
    data              Polygon -> cached monthly panels + universe mask
    feature           panels -> 16-feature ML matrix + decile target
    strategy          XGBoost multiclass -> RET reclassification -> BM overlay
    report_generator  metrics, sub-periods, PM gate, tearsheet

End-to-end convenience entry point: :func:`run_pipeline`.
"""
from __future__ import annotations

import pandas as pd

from . import constant, data, feature, report_generator, strategy, util
from .constant import StrategyConfig
from .data import MarketData, load_market_data
from .feature import build_feature_matrix, split_features
from .report_generator import generate_report
from .strategy import BacktestResult, DeepMomentumStrategy, attach_forward_returns

__all__ = [
    "constant", "util", "data", "feature", "strategy", "report_generator",
    "StrategyConfig", "MarketData", "load_market_data",
    "build_feature_matrix", "split_features",
    "DeepMomentumStrategy", "BacktestResult", "attach_forward_returns",
    "generate_report", "run_pipeline",
]


def run_pipeline(cfg: StrategyConfig, benchmark: pd.Series, refresh: bool = False) -> dict:
    """0->4 end-to-end run for one strategy config.

    Parameters
    ----------
    cfg : StrategyConfig
        Frozen configuration.
    benchmark : pd.Series
        Monthly SPY total-return series (DatetimeIndex aligned to month-ends).
    refresh : bool
        Force a rebuild of cached Polygon panels.
    """
    market = load_market_data(cfg, refresh=refresh)            # 1. data
    matrix = build_feature_matrix(market, cfg)                 # 2. features
    matrix = attach_forward_returns(matrix, market)           # mu_hat_k support
    result = DeepMomentumStrategy(cfg).run(matrix, market)    # 3. strategy
    return generate_report(result, benchmark, cfg)            # 4. report
