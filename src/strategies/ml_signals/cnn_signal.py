from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


class CNNPriceSeriesSignal:
    """
    1D CNN on log-return sequences for momentum signal generation.
    Architecture: Conv1D → BatchNorm → MaxPool → Dropout → Dense → Sigmoid
    Input: (lookback_days,) log returns per ticker
    Output: probability of positive forward return

    CRITICAL: normalization stats computed from train window only.
    self._train_mean and self._train_std are fixed after fit() and never updated.
    """

    def __init__(self, lookback_days: int = 60):
        self.lookback_days = lookback_days
        self.model = None
        self._train_mean: float | None = None
        self._train_std: float | None = None

    def _build_model(self):
        import torch
        import torch.nn as nn

        class MomentumCNN(nn.Module):
            def __init__(self, seq_len: int):
                super().__init__()
                self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
                self.bn1 = nn.BatchNorm1d(32)
                self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm1d(64)
                self.pool = nn.AdaptiveMaxPool1d(8)
                self.dropout = nn.Dropout(0.3)
                self.fc = nn.Linear(64 * 8, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = x.unsqueeze(1)
                x = torch.relu(self.bn1(self.conv1(x)))
                x = torch.relu(self.bn2(self.conv2(x)))
                x = self.pool(x)
                x = self.dropout(x.flatten(1))
                return self.sigmoid(self.fc(x)).squeeze(1)

        return MomentumCNN(self.lookback_days)

    def _make_sequences(
        self, prices: pd.DataFrame, end_date: pd.Timestamp
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Build (X, y, tickers) arrays for all tickers up to end_date.
        X shape: (n_samples, lookback_days)
        y shape: (n_samples,) — binary: 1 if forward 21-day return > 0
        """
        returns = np.log(prices / prices.shift(1)).dropna()
        hist = returns.loc[:end_date]

        X_list, y_list, tickers = [], [], []
        for ticker in hist.columns:
            s = hist[ticker].dropna()
            if len(s) < self.lookback_days + 22:
                continue
            for i in range(self.lookback_days, len(s) - 21, 21):
                seq = s.iloc[i - self.lookback_days : i].values
                fwd = prices[ticker].iloc[
                    prices.index.get_loc(s.index[i]) :
                    prices.index.get_loc(s.index[i]) + 22
                ]
                if len(fwd) < 22:
                    continue
                fwd_ret = fwd.iloc[-1] / fwd.iloc[0] - 1
                X_list.append(seq)
                y_list.append(1 if fwd_ret > 0 else 0)
                tickers.append(ticker)

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32), tickers

    def fit(self, prices: pd.DataFrame, train_end: pd.Timestamp, epochs: int = 20) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        X, y, _ = self._make_sequences(prices, train_end)
        if len(X) == 0:
            raise ValueError("Insufficient train data to fit CNN signal.")

        # Compute normalization stats from train data ONLY — never touch test data here
        self._train_mean = float(X.mean())
        self._train_std = float(X.std()) or 1.0
        X_norm = (X - self._train_mean) / self._train_std

        dataset = TensorDataset(
            torch.tensor(X_norm), torch.tensor(y)
        )
        loader = DataLoader(dataset, batch_size=256, shuffle=True)

        self.model = self._build_model()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        self.model.train()
        for _ in range(epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

    def predict_ranks(self, prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.Series:
        """
        Returns probability of positive forward return for each ticker as of as_of.
        Uses ONLY self._train_mean / self._train_std from fit() — no leakage.
        """
        import torch

        if self.model is None or self._train_mean is None:
            raise RuntimeError("Call fit() before predict_ranks()")

        returns = np.log(prices / prices.shift(1)).dropna()
        hist = returns.loc[:as_of]

        seqs, tickers = [], []
        for ticker in hist.columns:
            s = hist[ticker].dropna()
            if len(s) < self.lookback_days:
                continue
            seq = s.iloc[-self.lookback_days :].values.astype(np.float32)
            seqs.append(seq)
            tickers.append(ticker)

        if not seqs:
            return pd.Series(dtype=float)

        X = np.stack(seqs)
        X_norm = (X - self._train_mean) / self._train_std

        self.model.eval()
        with torch.no_grad():
            probs = self.model(torch.tensor(X_norm)).numpy()

        return pd.Series(probs, index=tickers, name="cnn_score")

    def evaluate_oos(self, prices: pd.DataFrame, train_end: pd.Timestamp) -> dict:
        """Compute OOS accuracy and IC on monthly rebalance dates after train_end."""
        oos_prices = prices.loc[train_end:]
        monthly_dates = pd.date_range(
            start=oos_prices.index[0],
            end=oos_prices.index[-22],
            freq="ME",
        )
        accuracies, ics = [], []
        for date in monthly_dates:
            signal = self.predict_ranks(prices, date)
            fwd_ret = prices.loc[date:].pct_change(21).iloc[1].reindex(signal.index).dropna()
            if len(fwd_ret) < 10:
                continue
            signal_aligned = signal.reindex(fwd_ret.index).dropna()
            pred_sign = (signal_aligned > 0.5).astype(int)
            actual_sign = (fwd_ret.loc[signal_aligned.index] > 0).astype(int)
            accuracies.append((pred_sign == actual_sign).mean())
            ic, _ = spearmanr(signal_aligned, fwd_ret.loc[signal_aligned.index])
            ics.append(ic)

        return {
            "oos_accuracy": float(np.mean(accuracies)) if accuracies else float("nan"),
            "oos_ic": float(np.mean(ics)) if ics else float("nan"),
        }
