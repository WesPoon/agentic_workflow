# Momentum Trading Agentic Workflow

## Project Purpose

Three-agent pipeline for researching, implementing, and reviewing momentum trading strategies derived from academic literature.

1. **Research Agent** — reads academic papers, extracts tradeable signals
2. **Implementation Agent** — implements strategies in Python, backtests with `bt` + `quantstats`
3. **Review Agent** — cross-validates research fidelity and implementation correctness

---

## Directory Layout

```
agentic_workflow/
├── papers/          # Drop your 10 PDFs here before running research-agent
├── research/        # Research agent outputs (structured briefs, signal specs)
├── src/
│   ├── strategies/  # bt Strategy classes, one file per strategy
│   ├── data/        # Data loaders (yfinance, Polygon)
│   └── utils/       # Shared helpers (metrics, logging)
├── backtests/       # Backtest result CSVs and quantstats HTML tearsheets
├── reviews/         # Review agent findings and sign-off reports
└── agents/          # Agent prompt files and configs
```

---

## Tech Stack

| Purpose | Library |
|---|---|
| Backtesting engine | `bt` |
| Performance reporting | `quantstats` |
| Free EOD data | `yfinance` |
| Institutional data | `polygon-api-client` |
| Data wrangling | `pandas`, `numpy` |
| Visualization | `matplotlib` |

Install: `pip install bt quantstats yfinance polygon-api-client pandas numpy matplotlib`

---

## Agent Roles & Responsibilities

### Agent 1: Research Agent

**Trigger:** User drops PDFs into `papers/` and invokes the research agent.

**Inputs:** PDF files in `papers/`

**Task:**
- Extract the core momentum signal from each paper (e.g., 12-1 month lookback, cross-sectional rank, time-series momentum)
- Document: lookback period, holding period, rebalance frequency, universe (equities, futures, etc.), signal construction, risk management rules
- Flag any parameters the paper tested across a range (important: do not data-mine — pick the paper's primary result)
- Output a structured research brief to `research/brief_<paper_title>.md` per paper
- Output a consolidated `research/strategy_spec.md` summarizing all signals into one unified spec

**Output format for `strategy_spec.md`:**
```
## Signal: <name>
- Lookback: <N> months
- Holding period: <N> months
- Universe: <description>
- Entry rule: <exact condition>
- Exit rule: <exact condition>
- Position sizing: <equal weight / volatility scaled / etc.>
- Risk controls: <stop loss, max position, sector limits>
- Source paper: <title, authors, year>
```

**Constraints:**
- Do not invent parameters not in the papers
- Note if the paper's results are in-sample only (no out-of-sample validation)
- Flag any data-snooping concerns explicitly

---

### Agent 2: Implementation Agent

**Trigger:** `research/strategy_spec.md` exists and is complete.

**Inputs:** `research/strategy_spec.md`

**Task:**
- Implement each signal as a `bt.Strategy` subclass in `src/strategies/`
- Wire up data pipelines in `src/data/`:
  - `yfinance_loader.py` for free EOD data (default)
  - `polygon_loader.py` for Polygon.io (requires `POLYGON_API_KEY` env var)
- Run backtests using `bt.run()` — NO forward-looking data leakage
- Generate `quantstats` tearsheets for each strategy

**Backtest requirements:**
- In-sample: first 70% of available data
- Out-of-sample: last 30% — report these separately
- Benchmark: SPY total return
- Minimum 10 years of history

**Required metrics (output to `backtests/<strategy_name>_metrics.csv`):**

| Metric | quantstats call |
|---|---|
| Sharpe Ratio (annualized) | `qs.stats.sharpe(returns)` |
| Sortino Ratio | `qs.stats.sortino(returns)` |
| Calmar Ratio | `qs.stats.calmar(returns)` |
| CAGR | `qs.stats.cagr(returns)` |
| Max Drawdown | `qs.stats.max_drawdown(returns)` |
| Max Drawdown Duration | `qs.stats.drawdown_details(returns)` |
| Volatility (annualized) | `qs.stats.volatility(returns)` |
| Win Rate | `qs.stats.win_rate(returns)` |
| Avg Win / Avg Loss | `qs.stats.avg_win(returns)` / `qs.stats.avg_loss(returns)` |
| Beta vs SPY | `qs.stats.greeks(returns, benchmark)['beta']` |
| Alpha vs SPY | `qs.stats.greeks(returns, benchmark)['alpha']` |
| Value at Risk (95%) | `qs.stats.value_at_risk(returns)` |

**Full tearsheet:** `qs.reports.html(returns, benchmark=spy_returns, output='backtests/<name>_tearsheet.html')`

**Code standards:**
- Each strategy file: one class, typed, no global state
- Data loaders: return `pd.DataFrame` with DatetimeIndex, columns = tickers
- No hardcoded dates — use relative lookbacks from the data end date
- Log all rebalance events with date, tickers in/out, weights

---

### Agent 3: Review Agent

**Trigger:** Both `research/strategy_spec.md` and at least one file in `backtests/` exist.

**Inputs:** All files in `research/`, `src/`, `backtests/`

**Task:**
- **Research fidelity check:** For each implemented strategy, verify the code matches the spec (lookback period, signal construction, rebalance frequency, position sizing)
- **Methodology check:**
  - No lookahead bias (data aligned correctly, no `shift()` errors)
  - Transaction costs applied (default: 10bps per trade)
  - Survivorship bias acknowledged if using point-in-time data vs current constituents
  - In-sample vs out-of-sample split respected
- **Results sanity check:**
  - Flag suspiciously high Sharpe (>3.0 on out-of-sample) as likely data leakage
  - Flag max drawdown inconsistency with stated risk controls
  - Verify benchmark comparison is apples-to-apples (same period, total return)
- **Output:** `reviews/review_<strategy_name>.md` with PASS / FAIL / NEEDS_REVISION per check

**Review output format:**
```markdown
## Strategy: <name>
### Research Fidelity: PASS | FAIL
- [ ] Lookback matches paper
- [ ] Signal construction matches paper
- [ ] Rebalance frequency matches paper

### Methodology: PASS | FAIL
- [ ] No lookahead bias detected
- [ ] Transaction costs applied
- [ ] In/out-of-sample split correct

### Results Sanity: PASS | FAIL
- [ ] Sharpe ratio plausible
- [ ] Drawdown consistent with risk controls
- [ ] Benchmark comparison valid

### Overall: APPROVED | NEEDS_REVISION
**Revision notes:** <if any>
```

---

## Environment Variables

```bash
POLYGON_API_KEY=<your_key>   # Required for Polygon data loader
```

Set in shell or `.env` file (never commit `.env`).

---

## Running the Pipeline

```bash
python -m src.runner run --papers papers/<file>.pdf     # local, see .env.example
python -m src.runner resume <thread_id>                 # continue from the last checkpoint
python -m pytest tests -q                               # offline tests, scripted model
```

Run artifacts go to `runs/<thread_id>/` (or S3), never into `src/`. AWS deployment: see `DEPLOY.md`.

How the current code works (supersedes parts of the agent descriptions above):
- Metrics are computed by the fixed evaluator `src/jobs/evaluate.py`, not by strategy code. Strategy
  modules implement `run_backtest(backtests_dir, params)` and return daily net returns
  (`templates/strategy_template.py`). The evaluator does the 70/30 split, bootstrap CI and deflated Sharpe.
- Every backtest attempt is logged in `attempts.csv` with commit, change note, OOS Sharpe and verdict.
- A strategy that fails review `MAX_REVISIONS` times is skipped; there is no human pause.
- Config is env-driven (`src/config.py`); the model id is `LLM_MODEL_ID`.
- Agent prompts live in `prompts/<agent>/*.md` with `{{placeholder}}` fields (see `prompts/README.md`).
  `agents/*.md` are design notes and are not loaded by the pipeline.

---

## Anti-patterns to Avoid

- **No** `df.iloc[-1]` inside a loop that iterates forward in time (lookahead)
- **No** fitting parameters on the full dataset before splitting train/test
- **No** ignoring delisted stocks in a historical backtest (survivorship bias)
- **No** reporting only in-sample Sharpe — always show out-of-sample separately
- **No** strategy parameters tuned to maximize backtest Sharpe (overfitting)
