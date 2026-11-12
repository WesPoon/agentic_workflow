from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier


class XGBoostMomentumSignal:
    """
    Cross-sectional momentum signal using XGBoost.
    Features: multi-horizon returns + volatility, all computed on train data only.
    Target: sign of forward 1-month return (binary classification).
    Fit on train window, predict on test window — no leakage.
    """

    DEFAULT_FEATURES = ["ret_1m", "ret_3m", "ret_6m", "ret_12m_1m", "vol_21d", "vol_63d"]

    def __init__(self, feature_config: dict | None = None):
        cfg = feature_config or {}
        self.features = cfg.get("features", self.DEFAULT_FEATURES)
        self.n_estimators = cfg.get("n_estimators", 300)
        self.max_depth = cfg.get("max_depth", 4)
        self.learning_rate = cfg.get("learning_rate", 0.05)
        self.model: XGBClassifier | None = None
        self._train_end: pd.Timestamp | None = None

    def _build_features(self, prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
        hist = prices.loc[:as_of]
        rows: dict[str, dict] = {}

        for ticker in prices.columns:
            s = hist[ticker].dropna()
            if len(s) < 252:
                continue
            row: dict = {}
            if "ret_1m" in self.features:
                row["ret_1m"] = s.pct_change(21).iloc[-1]
            if "ret_3m" in self.features:
                row["ret_3m"] = s.pct_change(63).iloc[-1]
            if "ret_6m" in self.features:
                row["ret_6m"] = s.pct_change(126).iloc[-1]
            if "ret_12m_1m" in self.features:
                ret_12 = s.pct_change(252).iloc[-1]
                ret_1 = s.pct_change(21).iloc[-1]
                row["ret_12m_1m"] = ret_12 - ret_1
            if "vol_21d" in self.features:
                row["vol_21d"] = s.pct_change().rolling(21).std().iloc[-1]
            if "vol_63d" in self.features:
                row["vol_63d"] = s.pct_change().rolling(63).std().iloc[-1]
            rows[ticker] = row

        return pd.DataFrame(rows).T.dropna()

    def fit(self, prices: pd.DataFrame, train_end: pd.Timestamp) -> None:
        """Fit on all monthly rebalance dates within the train window."""
        self._train_end = train_end
        train_prices = prices.loc[:train_end]

        # Build monthly feature/target pairs
        monthly_dates = pd.date_range(
            start=train_prices.index[252],
            end=train_prices.index[-22],
            freq="ME",
        )

        X_parts, y_parts = [], []
        for date in monthly_dates:
            X_date = self._build_features(train_prices, date)
            if X_date.empty:
                continue
            fwd_ret = prices.loc[date:].pct_change(21).iloc[1]
            fwd_ret = fwd_ret.reindex(X_date.index).dropna()
            X_aligned = X_date.loc[fwd_ret.index]
            if len(X_aligned) < 10:
                continue
            X_parts.append(X_aligned)
            y_parts.append((fwd_ret > 0).astype(int))

        if not X_parts:
            raise ValueError("Insufficient train data to fit XGBoost signal.")

        X = pd.concat(X_parts).reset_index(drop=True)
        y = pd.concat(y_parts).reset_index(drop=True)

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X[self.features], y)

    def predict_ranks(self, prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        """
        Cross-sectional probability of positive return for each ticker.
        Returns a Series indexed by ticker, values in [0, 1].
        Higher = stronger buy signal.
        """
        if self.model is None:
            raise RuntimeError("Call fit() before predict_ranks()")
        X = self._build_features(prices, as_of)
        if X.empty:
            return pd.Series(dtype=float)
        probs = self.model.predict_proba(X[self.features])[:, 1]
        return pd.Series(probs, index=X.index, name="xgb_score")

    def get_feature_importances(self) -> pd.Series:
        if self.model is None:
            return pd.Series(dtype=float)
        return pd.Series(
            self.model.feature_importances_,
            index=self.features,
            name="importance",
        ).sort_values(ascending=False)

    def compute_oos_ic(self, prices: pd.DataFrame, train_end: pd.Timestamp) -> float:
        """
        Compute average rank IC (Spearman) on OOS monthly dates.
        IC measures how well the signal ranks tickers vs actual forward returns.
        """
        oos_prices = prices.loc[train_end:]
        monthly_dates = pd.date_range(
            start=oos_prices.index[0],
            end=oos_prices.index[-22],
            freq="ME",
        )
        ics: list[float] = []
        for date in monthly_dates:
            signal = self.predict_ranks(prices, date)
            fwd_ret = prices.loc[date:].pct_change(21).iloc[1].reindex(signal.index).dropna()
            signal_aligned = signal.reindex(fwd_ret.index).dropna()
            if len(signal_aligned) < 10:
                continue
            ic, _ = spearmanr(signal_aligned, fwd_ret.loc[signal_aligned.index])
            ics.append(ic)
        return float(np.mean(ics)) if ics else float("nan")
