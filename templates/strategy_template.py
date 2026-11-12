"""Strategy template. Copy, rename and edit. The worker calls run_backtest(backtests_dir, params).

Contract (enforced by src/jobs/worker.py):
  return {
      "returns": pd.Series,            # daily strategy returns NET of transaction costs, DatetimeIndex
      "benchmark_returns": pd.Series,  # daily SPY total returns on the same dates
      "positions": pd.DataFrame|None,  # optional: weights by date x ticker
      "meta": dict,                    # optional: universe, cost_bps, data source, notes
  }
Do NOT compute Sharpe/CAGR/drawdown or split in/out-of-sample yourself: the evaluator does that,
identically for every attempt (70/30 by date, bootstrap CI, deflated Sharpe).

Lookahead rules: compute the signal from data up to t, shift it one day before it drives weights,
never use full-sample statistics (means, std, model fits) to build features for earlier dates.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_PARAMS = {
    "tickers": ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "JPM", "JNJ", "V", "XOM",
                "PG", "UNH", "HD", "KO", "PEP", "MRK", "CVX", "WMT", "BAC", "CSCO"],
    "start": "2005-01-01",
    "lookback_days": 252,     # 12-month momentum
    "skip_days": 21,          # skip the most recent month (short-term reversal)
    "top_n": 5,
    "rebalance": "ME",        # month-end (pandas >= 2.2 alias)
    "cost_bps": 10,
}


def _load_prices(tickers: list[str], start: str) -> pd.DataFrame:
    from src.data.yfinance_loader import load_prices

    end = pd.Timestamp.today().strftime("%Y-%m-%d")
    return load_prices(tickers, start, end)


def run_backtest(backtests_dir: str, params: dict) -> dict:
    import bt

    p = {**DEFAULT_PARAMS, **(params or {})}
    prices = _load_prices(list(p["tickers"]) + ["SPY"], p["start"]).dropna(how="all")
    spy = prices.pop("SPY")
    prices = prices.dropna(axis=1, thresh=int(len(prices) * 0.8))

    # Signal: 12-1 momentum, known at the close of day t, applied from t+1.
    momentum = prices.shift(p["skip_days"]).pct_change(p["lookback_days"] - p["skip_days"], fill_method=None)
    ranks = momentum.rank(axis=1, ascending=False)
    target = (ranks <= p["top_n"]).astype(float)
    target = target.div(target.sum(axis=1).replace(0, float("nan")), axis=0).fillna(0.0)
    target = target.shift(1)  # no lookahead

    rebalance_dates = target.resample(p["rebalance"]).last().index
    weights = target.reindex(rebalance_dates, method="ffill").dropna(how="all")
    weights = weights.loc[weights.sum(axis=1) > 0]   # start at the first date with a real signal

    strategy = bt.Strategy(
        "strategy",
        [bt.algos.RunOnDate(*weights.index), bt.algos.SelectAll(),
         bt.algos.WeighTarget(weights), bt.algos.Rebalance()],
    )
    cost = p["cost_bps"] / 10_000
    test = bt.Backtest(strategy, prices.loc[weights.index[0]:],
                       commissions=lambda q, price: abs(q) * price * cost, integer_positions=False)
    res = bt.run(test)
    equity = res.prices["strategy"]          # equity curve rebased to 100, net of commissions
    returns = equity.pct_change().dropna()
    bench = spy.pct_change().reindex(returns.index).fillna(0.0)

    return {
        "returns": returns,
        "benchmark_returns": bench,
        "positions": weights,
        "meta": {"universe": list(prices.columns), "cost_bps": p["cost_bps"], "source": "yfinance",
                 "survivorship_note": "current large caps; survivorship bias present"},
    }
