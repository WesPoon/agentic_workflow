"""Strategy layer — Deep Momentum engine.

Pipeline per rebalance month:
    1. annual expanding-window XGBoost multiclass, trained N times, probs averaged
    2. RET reclassification: mu_hat_i = sum_k  y_hat_{i,k} * mu_hat_k   (no argmax)
    3. portfolio: long top decile / short bottom decile by mu_hat_i, equal weight
    4. bimodality regime overlay scales position size from the prior month's BM
    5. costs applied on turnover; returns + diagnostics returned

The IS/OOS firewall: models and ``mu_hat_k`` for year Y are fit only on data strictly
before January of Y (expanding from 2004). OOS months are scored, never fitted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import constant as C
from .data import MarketData
from .util import get_logger, log_rebalance

logger = get_logger("strategy")


@dataclass
class BacktestResult:
    """Outputs of a backtest run. Return series are monthly, DatetimeIndex."""

    long_short: pd.Series
    long_leg: pd.Series
    short_leg: pd.Series
    bm_series: pd.Series
    turnover: pd.Series
    weights_log: list[dict] = field(default_factory=list)
    # Per-(date, ticker) OOS diagnostics: expected_return, pred_decile,
    # actual_decile, leg (+1 long / -1 short / 0 none), scale, realized_return.
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)


class DeepMomentumStrategy:
    """XGBoost decile classifier + RET reclassification + bimodality overlay."""

    def __init__(self, cfg: C.StrategyConfig) -> None:
        self.cfg = cfg
        self._models: dict[int, list] = {}        # year -> [boosters]
        self._mu_hat_k: dict[int, np.ndarray] = {}  # year -> decile mean returns

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(
        self,
        matrix: pd.DataFrame,
        market: MarketData,
        backtest_start: str | None = None,
    ) -> BacktestResult:
        """Walk forward over rebalance months and produce portfolio returns.

        ``backtest_start`` defaults to the OOS start; pass an earlier date to also
        score in-sample months for diagnostics.
        """
        forward_return = market.returns.shift(-1)  # realized over the held month
        start = pd.Timestamp(backtest_start or self.cfg.out_sample_start)
        dates = sorted(d for d in matrix.index.get_level_values("date").unique() if d >= start)

        ls_ret, long_ret, short_ret = {}, {}, {}
        bm_hist, turnover_hist, logs = {}, {}, []
        pred_frames: list[pd.DataFrame] = []
        prev_long: set = set()
        prev_short: set = set()
        prev_bm: float = -1.0  # benign default -> full position
        prev_scale: float = 0.0  # start flat -> first month pays to build the book

        for date in dates:
            month = matrix.xs(date, level="date")
            self._ensure_trained(matrix, date.year)

            mu_hat_i, pred_decile, _ = self._score_month(month, date.year)
            longs, shorts = self._select_legs(mu_hat_i)

            realized = forward_return.loc[date]
            if realized.reindex(longs).isna().all():
                continue  # no realized month to score (end of sample)

            scale = self._position_scale(prev_bm)
            l_ret = realized.reindex(longs).mean()
            s_ret = realized.reindex(shorts).mean()
            traded = self._traded_notional(prev_long, longs, prev_short, shorts,
                                           prev_scale, scale)
            cost = traded * (self.cfg.transaction_cost_bps / 1e4)
            turnover = traded

            long_ret[date] = scale * l_ret
            short_ret[date] = scale * s_ret
            ls_ret[date] = scale * (l_ret - s_ret) - cost
            turnover_hist[date] = turnover

            bm = self._bimodality(longs, shorts, month)
            bm_hist[date] = bm
            pred_frames.append(
                self._month_predictions(date, month, mu_hat_i, pred_decile,
                                        longs, shorts, scale, realized)
            )
            prev_bm = bm
            prev_scale = scale
            prev_long, prev_short = set(longs), set(shorts)

            log_rebalance(logger, date, len(longs), len(shorts), turnover)
            logs.append({"date": date, "n_long": len(longs), "n_short": len(shorts),
                         "bm": bm, "scale": scale, "turnover": turnover})

        predictions = (
            pd.concat(pred_frames).sort_index() if pred_frames else pd.DataFrame()
        )
        return BacktestResult(
            long_short=pd.Series(ls_ret).sort_index(),
            long_leg=pd.Series(long_ret).sort_index(),
            short_leg=pd.Series(short_ret).sort_index(),
            bm_series=pd.Series(bm_hist).sort_index(),
            turnover=pd.Series(turnover_hist).sort_index(),
            weights_log=logs,
            predictions=predictions,
        )

    @staticmethod
    def _month_predictions(
        date: pd.Timestamp,
        month: pd.DataFrame,
        mu_hat_i: pd.Series,
        pred_decile: pd.Series,
        longs: pd.Index,
        shorts: pd.Index,
        scale: float,
        realized: pd.Series,
    ) -> pd.DataFrame:
        """Per-ticker diagnostics for one rebalance month (full cross-section)."""
        frame = pd.DataFrame(index=month.index)
        frame["expected_return"] = mu_hat_i
        frame["pred_decile"] = pred_decile + 1          # report as 1..10
        frame["actual_decile"] = month[C.TARGET_COLUMN] + 1
        frame["realized_return"] = realized.reindex(month.index)
        frame["leg"] = 0
        frame.loc[longs, "leg"] = 1
        frame.loc[shorts, "leg"] = -1
        frame["scale"] = scale
        frame.index = pd.MultiIndex.from_product(
            [[date], frame.index], names=["date", "ticker"]
        )
        return frame

    # ------------------------------------------------------------------ #
    # Training (annual, expanding window)
    # ------------------------------------------------------------------ #
    def _ensure_trained(self, matrix: pd.DataFrame, year: int) -> None:
        if year in self._models:
            return
        import xgboost as xgb

        dates = matrix.index.get_level_values("date")
        train = matrix[dates < pd.Timestamp(year=year, month=1, day=1)].dropna(
            subset=[C.TARGET_COLUMN]
        )
        if len(train) < C.N_DECILES * 100:
            logger.warning("Year %d: thin training set (%d rows)", year, len(train))

        X = train[_feature_columns(train)].to_numpy(dtype=float)
        y = train[C.TARGET_COLUMN].to_numpy(dtype=int)

        boosters = []
        rng = np.random.default_rng(self.cfg.seed + year)
        for run_i in range(self.cfg.n_training_runs):
            tr_idx, va_idx = self._random_split(len(train), rng)
            dtrain = xgb.DMatrix(X[tr_idx], label=y[tr_idx])
            dvalid = xgb.DMatrix(X[va_idx], label=y[va_idx])
            params = dict(self.cfg.xgb_params, seed=int(rng.integers(0, 2**31)))
            booster = xgb.train(
                params,
                dtrain,
                num_boost_round=self.cfg.xgb_num_rounds,
                evals=[(dvalid, "valid")],
                early_stopping_rounds=C.XGB_EARLY_STOPPING_ROUNDS,
                verbose_eval=False,
            )
            boosters.append(booster)

        self._models[year] = boosters
        self._mu_hat_k[year] = self._decile_mean_returns(matrix, year)
        logger.info("Trained %d boosters for year %d on %d rows",
                    len(boosters), year, len(train))

    def _random_split(self, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Random (not chronological) 80/20 split within the training window."""
        idx = rng.permutation(n)
        cut = int(n * (1 - C.VALIDATION_FRACTION))
        return idx[:cut], idx[cut:]

    def _decile_mean_returns(self, matrix: pd.DataFrame, year: int) -> np.ndarray:
        """``mu_hat_k`` = trailing 10y mean forward return of stocks that landed in decile k."""
        dates = matrix.index.get_level_values("date")
        lo = pd.Timestamp(year=year - C.DECILE_MEAN_LOOKBACK_YEARS, month=1, day=1)
        hi = pd.Timestamp(year=year, month=1, day=1)
        window = matrix[(dates >= lo) & (dates < hi)].dropna(subset=[C.TARGET_COLUMN])
        if "fwd_return" not in window.columns:
            # forward return is not stored on the matrix; approximate mu_hat_k from
            # the decile rank itself is invalid -> require fwd_return column.
            raise KeyError(
                "matrix must carry a 'fwd_return' column for mu_hat_k; "
                "attach it via attach_forward_returns()."
            )
        means = window.groupby(C.TARGET_COLUMN)["fwd_return"].mean()
        return means.reindex(range(C.N_DECILES)).to_numpy()

    # ------------------------------------------------------------------ #
    # Prediction / reclassification
    # ------------------------------------------------------------------ #
    def _predict_proba(self, month: pd.DataFrame, year: int) -> np.ndarray:
        """Average predicted decile probabilities across the year's boosters."""
        import xgboost as xgb

        X = xgb.DMatrix(month[_feature_columns(month)].to_numpy(dtype=float))
        probs = np.zeros((len(month), C.N_DECILES))
        for booster in self._models[year]:
            probs += booster.predict(X)
        return probs / len(self._models[year])

    def _score_month(
        self, month: pd.DataFrame, year: int
    ) -> tuple[pd.Series, pd.Series, np.ndarray]:
        """Score one cross-section.

        Returns ``(mu_hat_i, pred_decile, proba)``:
          * ``mu_hat_i``    RET reclassification  sum_k y_hat_{i,k} * mu_hat_k
          * ``pred_decile`` argmax of the predicted distribution (for diagnostics)
          * ``proba``       the full (n, 10) probability matrix
        One prediction pass feeds both the expected return and the diagnostics.
        """
        proba = self._predict_proba(month, year)        # (n, 10)
        mu_hat_k = np.nan_to_num(self._mu_hat_k[year])  # (10,)
        mu_hat_i = pd.Series(proba @ mu_hat_k, index=month.index)
        pred_decile = pd.Series(proba.argmax(axis=1), index=month.index)
        return mu_hat_i, pred_decile, proba

    def _select_legs(self, mu_hat_i: pd.Series) -> tuple[pd.Index, pd.Index]:
        ranked = mu_hat_i.sort_values(ascending=False)
        n = len(ranked)
        n_long = max(1, int(n * self.cfg.long_quantile))
        n_short = max(1, int(n * self.cfg.short_quantile))
        return ranked.index[:n_long], ranked.index[-n_short:]

    # ------------------------------------------------------------------ #
    # Bimodality regime overlay
    # ------------------------------------------------------------------ #
    def _bimodality(self, longs: pd.Index, shorts: pd.Index, month: pd.DataFrame) -> float:
        """BM from realized deciles of this month's selected legs (known after t).

        Winners/losers = actual top/bottom return decile (target 0 / 9).
        """
        actual = month[C.TARGET_COLUMN]
        win, lose = 0, C.N_DECILES - 1

        tp = int((actual.reindex(longs) == win).sum())
        fp = int((actual.reindex(longs) == lose).sum())
        tn = int((actual.reindex(shorts) == lose).sum())
        fn = int((actual.reindex(shorts) == win).sum())

        hh = _ratio(tp, tp + fp) - 0.1
        hl = _ratio(fp, tp + fp) - 0.1
        ll = _ratio(tn, tn + fn) - 0.1
        lh = _ratio(fn, tn + fn) - 0.1
        return -((hh - hl) + (ll - lh)) / 2.0

    def _position_scale(self, prev_bm: float) -> float:
        """Translate the prior month's BM into a position multiplier."""
        if not self.cfg.use_bimodality_filter or not np.isfinite(prev_bm):
            return 1.0
        if prev_bm > C.BM_FLAT_THRESHOLD:
            return 0.0
        if prev_bm > C.BM_HALF_SIZE_THRESHOLD:
            return 0.5
        return 1.0

    # ------------------------------------------------------------------ #
    # Costs / turnover
    # ------------------------------------------------------------------ #
    def _traded_notional(self, prev_long, longs, prev_short, shorts,
                         prev_scale: float, scale: float) -> float:
        """Round-trip notional traded this rebalance, as a fraction of one leg's gross.

        Costs the *deployed* (scale-weighted) book: a flat month (scale=0) pays ~0,
        a full->flat transition pays the liquidation, flat->full pays the rebuild.
        Both legs summed; ``turnover`` and ``cost`` are derived from this one number
        so they always reconcile (cost = traded * rate).
        """
        return (_scaled_leg_turnover(prev_long, longs, prev_scale, scale)
                + _scaled_leg_turnover(prev_short, shorts, prev_scale, scale))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def attach_forward_returns(matrix: pd.DataFrame, market: MarketData) -> pd.DataFrame:
    """Attach realized forward 1-month return to the matrix (needed for mu_hat_k)."""
    fwd = market.returns.shift(-1).stack()
    fwd.index = fwd.index.set_names(["date", "ticker"])
    out = matrix.copy()
    out["fwd_return"] = fwd.reindex(matrix.index)
    return out


_NON_FEATURE_COLS = {C.TARGET_COLUMN, "fwd_return"}


def _feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature columns = every column except the target and attached fwd_return.

    Derived from the matrix so optional feature sets (e.g. vol-adjusted momentum)
    flow through to training/prediction without editing this layer. ``train`` and
    ``month`` share the matrix's column order, so the two stay aligned.
    """
    return [c for c in df.columns if c not in _NON_FEATURE_COLS]


def _ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def _scaled_leg_turnover(prev: set, current: pd.Index,
                         prev_scale: float, scale: float) -> float:
    """L1 change in deployed equal-weight leg weights (buys + sells + rebalances).

    Per-name weight = scale / n. New names enter at this month's weight, dropped
    names exit at last month's weight, and held names are charged the weight delta
    (so a scale change rebalances even an unchanged name set). With scale=prev_scale=1
    and equal counts this reduces to the plain (buys+sells)/n turnover.
    """
    cur = set(current)
    if not cur and not prev:
        return 0.0
    w_prev = prev_scale / max(len(prev), 1)   # per-name weight last month
    w_cur = scale / max(len(cur), 1)          # per-name weight this month
    bought = w_cur * len(cur - prev)          # new names entered
    sold = w_prev * len(prev - cur)           # dropped names exited
    rebalanced = abs(w_cur - w_prev) * len(cur & prev)  # weight delta on holdovers
    return bought + sold + rebalanced
