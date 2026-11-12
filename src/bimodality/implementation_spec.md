# Bimodality Deep Momentum — Implementation Spec

Implementation of the Deep Momentum strategy derived from Han (2022) and Han & Qin (2026).
The package lives in `src/bimodality/` and is split into single-responsibility modules.
This document is the contract every module must satisfy.

> **Goal:** HIGH SHARPE / HIGH CAGR / CONTROLLABLE DRAWDOWN on US equities via Polygon Massive,
> with momentum-crash protection through RET reclassification over the full predicted decile
> distribution (not `argmax`).

---

## Module Map

| # | Module | Responsibility | Depends on |
|---|---|---|---|
| 0 | `constant.py` | Immutable config: periods, filters, decile/feature definitions, paths. No logic. | — |
| 0 | `util.py` | Stateless cross-sectional helpers + logging. Pure functions. | constant |
| 1 | `data.py` | Polygon → cached monthly panels (price, volume, market cap) + universe mask. | constant, util |
| 2 | `feature.py` | Panels → tidy `(date, ticker)` feature matrix + decile target. | constant, util |
| 3 | `strategy.py` | XGBoost multiclass → RET reclassification → BM regime filter → portfolio returns. | constant, util, feature |
| 4 | `report_generator.py` | Metrics, sub-period breakdown, PM gate, quantstats tearsheet, BM series. | constant, util |

Data flows strictly left-to-right. No module imports a module to its right. No global mutable state.

---

## 0. Constants & Utilities (`constant.py`, `util.py`)

### Dataset constraints (fixed — never tuned)
| Constraint | Value |
|---|---|
| Universe | US common stock (`type='CS'`), NYSE / AMEX / NASDAQ |
| Market-cap floor | drop bottom **10%** by cap, recomputed each month |
| New-listing filter | exclude if listed **< 3 months** |
| Return outliers | drop monthly ret `> 300%` or `< -95%`; winsorize 1%/99% |
| Zero-volume filter | drop if vol = 0 this month AND prior month AND ret = 0 |
| Delisted stocks | included; missing delist return → assume **-30%** |
| In-sample | **2004-01 … 2018-12** (14y train) |
| Out-of-sample | **2019-01 … present** (untouched until final eval) |
| Rebalance | monthly, last trading day |
| Transaction cost | **10 bps** per trade per leg |
| Benchmark | SPY total return |

### Decile convention
- 10 deciles. **Decile 1 = highest forward return, Decile 10 = lowest** (XGBoost class label `0..9`).
- `µ̂_k` = trailing **10-year** mean realized return of stocks that landed in decile `k`,
  recomputed each January from the training window only.

### `util.py` pure functions
- `cross_sectional_zscore(s)` — z-score within a single month's cross-section.
- `cross_sectional_decile(s, n=10, ascending=False)` — rank → 1..n labels.
- `winsorize(s, lower=0.01, upper=0.99)`.
- `get_logger(name)` — module logger; rebalance events log `date, n_long, n_short, turnover`.

---

## 1. Data Layer (`data.py`)

Builds and caches the monthly cross-sectional panels from Polygon Massive.

**Inputs:** date range, Polygon API key (`MASSIVE_API_KEY` / `POLYGON_API_KEY`).

**Outputs** (`MarketData` dataclass), all month-end `DatetimeIndex` × ticker columns:
- `prices` — adjusted close
- `volume` — monthly volume
- `market_cap` — close × shares outstanding
- `returns` — monthly simple returns (derived)
- `universe_mask` — bool: passes every filter that month

**Build method (efficient full-universe):**
1. `list_tickers(type='CS', market='stocks')` for active **and** delisted → the CS universe set.
2. For each month-end trading day, `get_grouped_daily_aggs(date)` → all-ticker close + volume in one call (~12 calls/yr).
3. Per-ticker `get_ticker_details` → `shares` (for market cap) **and** `list_date` (for the seasoning
   filter — `list_tickers` returns `list_date=None`, so it must come from details). Fetched only for
   tickers that actually traded, **concurrently** via a `ThreadPoolExecutor` (`cfg.fetch_max_workers`,
   default 20 — the calls are I/O-bound, so threads beat asyncio against the blocking SDK).
4. Apply filters (cap floor, listing age, outliers, zero-volume) to produce `universe_mask`.

**Caching:** monthly panels and per-ticker meta (`ticker_meta.parquet`) persisted to `backtests/cache/`.
Ticker meta is cached **incrementally** — only names absent from the cache hit the API — so the slow
per-ticker pull happens once and every later run is effectively offline.
No lookahead: month-`t` features use data through month-`t-1` close only (enforced in `feature.py`).

---

## 2. Feature Layer (`feature.py`)

Turns panels into the **16-feature** ML matrix (Han 2022 feature set).

| # | Feature | Formula | Norm |
|---|---|---|---|
| 1-5 | `zMOM_{1,3,6,9,12}` | cum return over m months (12 skips last month) | cross-sectional z-score |
| 6-10 | `MMOM_{1,3,6,9,12}` | cross-sectional **mean** of `MOM_m` | raw (regime info) |
| 11-15 | `SMOM_{1,3,6,9,12}` | cross-sectional **std** of `MOM_m` | raw (regime info) |
| 16 | `size_decile` | market-cap decile 1-10 | categorical |

- `MOM_12` skips the most recent month (12-1) to avoid short-term reversal.
- All `MOM_m` at month `t` use returns through month `t-1` (no lookahead).
- **Target:** forward 1-month return decile (`0..9`), computed from `t → t+1` return — present only
  on training rows, `NaN` on the live prediction row.

**Output:** tidy `DataFrame`, MultiIndex `(date, ticker)`, 16 feature cols + `target_decile`.

---

## 3. Strategy Layer (`strategy.py`)

`DeepMomentumStrategy` — the core engine.

**3a. Model training**
- XGBoost multiclass (`multi:softprob`, 10 classes), default params + early stopping.
- **Random 80/20** train/val split inside the training window (not chronological).
- **Train 100×** per rebalance year, average predicted probabilities `ŷ_{i,k}` (stability).
- Retrain **annually each January**, expanding window (all data ≥ 2004).

**3b. RET reclassification (crash protection — never `argmax`)**
```
µ̂_k = trailing 10y mean return of decile k   (from training window only)
µ̂_i = Σ_k  ŷ_{i,k} · µ̂_k
rank stocks by µ̂_i  →  long top 10%, short bottom 10%
```

**3c. Bimodality regime filter (overlay)**
```
HH = TP/(TP+FP) − 0.1 ;  HL = FP/(TP+FP) − 0.1
LL = TN/(TN+FN) − 0.1 ;  LH = FN/(TN+FN) − 0.1
BM = −[(HH − HL) + (LL − LH)] / 2
BM > 0.0 → scale positions to 50% ;  BM > 0.1 → go flat
```
BM series is returned for reporting.

**3d. Portfolio → returns**
- Equal-weighted (primary). Long, short, and long-short returned separately.
- 10 bps × turnover deducted each rebalance. Rebalance events logged.

**Output:** `BacktestResult` — long/short/long-short monthly returns, BM series, turnover, weights log.

---

## 4. Reporting Layer (`report_generator.py`)

Consumes `BacktestResult` + SPY benchmark.

- **Metrics** (quantstats): Sharpe, Sortino, Calmar, CAGR, MaxDD, MaxDD duration, Vol, Win rate,
  Avg win/loss, Beta, Alpha, VaR95 → `backtests/<name>_metrics.csv`.
- **Sub-period breakdown:** full OOS, 2019, COVID 2020-21, inflation 2022-24 — Sharpe / CAGR / MaxDD each.
- **PM performance gate** (OOS must clear all):

| Metric | Min | Target |
|---|---|---|
| OOS Sharpe | ≥ 1.0 | ≥ 1.5 |
| CAGR vs SPY | > +5% | > +15% |
| Max DD | < 20% | < 15% |
| Calmar | > 0.5 | > 1.0 |
| COVID 2020 DD | < 30% | < 20% |
| 2022 DD | < 25% | < 15% |
| Sub-period Sharpe | positive ≥2 of {2019, 2020-21, 2022-24} | all 3 |

- **Tearsheet:** `qs.reports.html(..., output='backtests/<name>_tearsheet.html')`.
- **BM series:** `backtests/<name>_bimodality.csv`.
- **Per-ticker diagnostics** (OOS, every rebalance month):
  - `backtests/<name>_predictions.csv` — full cross-section: `expected_return` (µ̂_i),
    `pred_decile` (argmax of ŷ, 1-10), `actual_decile` (realized, 1-10),
    `realized_return`, `leg` (+1 long / -1 short / 0), `scale` (BM overlay).
  - `backtests/<name>_trades.csv` — held names only (`leg != 0`): the exact long/short
    book each month with side, predicted expected return, predicted vs actual decile,
    and realized return. This is the audit trail for "what did we hold and why."

---

## Combined Implementation Spec Summary

End-to-end pipeline (single entry per strategy):

```
constant.StrategyConfig                       # 0. fixed knobs, periods, paths
        │
data.load_market_data(cfg)                    # 1. Polygon → cached panels + universe_mask
        │   MarketData(prices, volume, market_cap, returns, universe_mask)
        │
feature.build_feature_matrix(market, cfg)     # 2. → tidy (date,ticker) × 16 feats + target_decile
        │
strategy.DeepMomentumStrategy(cfg).run(...)   # 3. annual XGBoost ×100 → RET reclass → BM filter → portfolio
        │   BacktestResult(long, short, long_short, bm_series, turnover, weights_log)
        │
report_generator.generate_report(result, spy) # 4. metrics + sub-periods + PM gate + tearsheet
```

**Invariants enforced across the stack**
- Lookahead-free: month-`t` features use data ≤ end of month `t-1`; target is forward-looking and
  only present on training rows.
- IS/OOS firewall: anything fit on data (models, `µ̂_k`, deciles) uses the **training window only**,
  refit annually on an expanding basis; OOS 2019+ is scored, never fitted.
- No post-hoc parameter tuning to maximize backtest Sharpe — config is frozen before OOS scoring.
- Survivorship handled by including Polygon delisted tickers; missing delist return assumed -30%.

**Combination hooks (recursive study — research-agent Levels 1-4)**
- *Level 1 Feature Union:* add columns in `feature.build_feature_matrix` → same XGBoost.
- *Level 2 ER Fusion:* blend `µ̂_i` from two strategies (`α·µ̂_mom + (1-α)·µ̂_other`).
- *Level 3 Prob Ensemble:* average `ŷ_{i,k}` across specialist models before reclassification.
- *Level 4 Variants:* swap the booster in `strategy` (LightGBM / CatBoost / ordinal / stacked).

Bimodality is applied here because the base signal is price momentum. For non-momentum
combinations (growth, macro), disable the BM overlay via `cfg.use_bimodality_filter=False`.
