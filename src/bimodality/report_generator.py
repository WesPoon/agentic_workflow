"""Reporting layer — metrics, sub-period breakdown, PM gate, tearsheet.

Consumes a ``BacktestResult`` (monthly returns) plus a monthly SPY benchmark and
writes every artifact under ``backtests/``. quantstats stats are computed with
``periods=12`` since the strategy rebalances monthly.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

try:
    import quantstats as qs
    _HAS_QS = True
except ImportError:  # keep the module importable without quantstats
    _HAS_QS = False

from . import constant as C
from .strategy import BacktestResult
from .util import get_logger, slice_period

logger = get_logger("report")

_PERIODS_PER_YEAR = 12


@dataclass
class PMGate:
    """Evaluated PM performance gate. ``passed`` is the AND of all checks."""

    checks: dict[str, bool]
    detail: dict[str, float]

    @property
    def passed(self) -> bool:
        return all(self.checks.values())


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(returns: pd.Series, benchmark: pd.Series | None = None) -> dict:
    """The 13 required metrics for a monthly return series."""
    if not _HAS_QS:
        raise ImportError("quantstats is required for compute_metrics")
    r = returns.dropna()
    if r.empty:
        return {}

    cagr = qs.stats.cagr(r, periods=_PERIODS_PER_YEAR)
    mdd = qs.stats.max_drawdown(r)
    metrics = {
        "sharpe":       qs.stats.sharpe(r, periods=_PERIODS_PER_YEAR),
        "sortino":      qs.stats.sortino(r, periods=_PERIODS_PER_YEAR),
        # Calmar computed directly (qs.stats.calmar mis-scales on monthly series).
        "calmar":       cagr / abs(mdd) if mdd else float("nan"),
        "cagr":         cagr,
        "max_drawdown": mdd,
        "volatility":   qs.stats.volatility(r, periods=_PERIODS_PER_YEAR),
        "win_rate":     qs.stats.win_rate(r),
        "avg_win":      qs.stats.avg_win(r),
        "avg_loss":     qs.stats.avg_loss(r),
        "var_95":       qs.stats.value_at_risk(r),
    }

    dd = qs.stats.drawdown_details(r)
    metrics["max_drawdown_months"] = int(dd["days"].max()) if dd is not None and not dd.empty else None

    if benchmark is not None:
        b = benchmark.reindex(r.index).dropna()
        ra = r.reindex(b.index).dropna()
        greeks = qs.stats.greeks(ra, b, periods=_PERIODS_PER_YEAR)
        metrics["beta"] = float(greeks.get("beta", float("nan")))
        metrics["alpha"] = float(greeks.get("alpha", float("nan")))
    else:
        metrics["beta"] = metrics["alpha"] = None
    return metrics


def benchmark_comparison(
    returns: pd.Series, benchmarks: dict[str, pd.Series]
) -> pd.DataFrame:
    """OOS head-to-head: strategy vs each benchmark on CAGR / Sharpe / MaxDD / Vol.

    Lets you answer "does the strategy actually beat NASDAQ-100?" at a glance — a
    strategy whose OOS CAGR trails QQQ has no edge worth the complexity.
    """
    def _oos_stats(s: pd.Series) -> dict:
        x = slice_period(s.dropna(), C.OUT_SAMPLE_START, "2100-12-31")
        if x.empty:
            return {"cagr": float("nan"), "sharpe": float("nan"),
                    "max_drawdown": float("nan"), "volatility": float("nan")}
        return {
            "cagr":         qs.stats.cagr(x, periods=_PERIODS_PER_YEAR),
            "sharpe":       qs.stats.sharpe(x, periods=_PERIODS_PER_YEAR),
            "max_drawdown": qs.stats.max_drawdown(x),
            "volatility":   qs.stats.volatility(x, periods=_PERIODS_PER_YEAR),
        }

    rows = {"strategy": _oos_stats(returns)}
    for name, b in benchmarks.items():
        rows[name] = _oos_stats(b)
    table = pd.DataFrame(rows).T
    # Excess CAGR of the strategy over each benchmark (positive = strategy wins).
    table["cagr_vs_strategy"] = table.loc["strategy", "cagr"] - table["cagr"]
    table.loc["strategy", "cagr_vs_strategy"] = float("nan")
    return table


def subperiod_breakdown(returns: pd.Series) -> pd.DataFrame:
    """Sharpe / CAGR / MaxDD for each named OOS sub-period."""
    rows = {}
    for name, (start, end) in C.SUBPERIODS.items():
        seg = slice_period(returns.dropna(), start, end)
        if seg.empty:
            continue
        rows[name] = {
            "sharpe": qs.stats.sharpe(seg, periods=_PERIODS_PER_YEAR),
            "cagr": qs.stats.cagr(seg, periods=_PERIODS_PER_YEAR),
            "max_drawdown": qs.stats.max_drawdown(seg),
            "n_months": len(seg),
        }
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- #
# PM performance gate
# --------------------------------------------------------------------------- #
def evaluate_pm_gate(returns: pd.Series, benchmark: pd.Series) -> PMGate:
    """Evaluate the OOS PM gate from implementation_spec.md section 4."""
    oos = slice_period(returns.dropna(), C.OUT_SAMPLE_START, "2100-12-31")
    spy = slice_period(benchmark.dropna(), C.OUT_SAMPLE_START, "2100-12-31")

    sharpe = qs.stats.sharpe(oos, periods=_PERIODS_PER_YEAR)
    cagr = qs.stats.cagr(oos, periods=_PERIODS_PER_YEAR)
    spy_cagr = qs.stats.cagr(spy, periods=_PERIODS_PER_YEAR)
    max_dd = qs.stats.max_drawdown(oos)
    calmar = cagr / abs(max_dd) if max_dd else float("nan")

    covid = slice_period(oos, *C.SUBPERIODS["COVID_2020_2021"])
    inflation = slice_period(oos, *C.SUBPERIODS["INFLATION_2022_2024"])
    covid_dd = qs.stats.max_drawdown(covid) if not covid.empty else 0.0
    inflation_dd = qs.stats.max_drawdown(inflation) if not inflation.empty else 0.0

    sub = subperiod_breakdown(returns)
    stress_keys = ["2019", "COVID_2020_2021", "INFLATION_2022_2024"]
    positive = int(sum(sub.loc[k, "sharpe"] > 0 for k in stress_keys if k in sub.index))

    checks = {
        "oos_sharpe>=1.0": sharpe >= 1.0,
        "cagr>spy+5%": (cagr - spy_cagr) > 0.05,
        "max_dd<20%": abs(max_dd) < 0.20,
        "calmar>0.5": calmar > 0.5,
        "covid_dd<30%": abs(covid_dd) < 0.30,
        "inflation_dd<25%": abs(inflation_dd) < 0.25,
        "subperiod_sharpe_pos>=2": positive >= 2,
    }
    detail = {
        "oos_sharpe": sharpe, "cagr": cagr, "spy_cagr": spy_cagr,
        "max_dd": max_dd, "calmar": calmar, "covid_dd": covid_dd,
        "inflation_dd": inflation_dd, "n_positive_subperiods": positive,
    }
    return PMGate(checks=checks, detail=detail)


# --------------------------------------------------------------------------- #
# Top-level report
# --------------------------------------------------------------------------- #
def generate_report(
    result: BacktestResult,
    benchmark: pd.Series,
    cfg: C.StrategyConfig,
    extra_benchmarks: dict[str, pd.Series] | None = None,
) -> dict:
    """Write all artifacts under ``backtests/`` and return a summary dict.

    ``benchmark`` is the primary (SPY) — drives the PM gate and the main tearsheet.
    ``extra_benchmarks`` (e.g. QQQ) add a head-to-head comparison CSV and their own
    tearsheets, so the strategy can be judged against tougher bars than SPY.
    """
    if not _HAS_QS:
        raise ImportError("quantstats is required for generate_report")
    cfg.backtest_dir.mkdir(parents=True, exist_ok=True)
    name = cfg.name
    ls = result.long_short
    extra_benchmarks = extra_benchmarks or {}

    metrics = compute_metrics(ls, benchmark)
    pd.Series(metrics).to_csv(cfg.backtest_dir / f"{name}_metrics.csv")

    sub = subperiod_breakdown(ls)
    sub.to_csv(cfg.backtest_dir / f"{name}_subperiods.csv")

    result.bm_series.to_csv(cfg.backtest_dir / f"{name}_bimodality.csv")

    gate = evaluate_pm_gate(ls, benchmark)
    pd.DataFrame({"check": gate.checks}).to_csv(cfg.backtest_dir / f"{name}_pm_gate.csv")

    # Head-to-head vs every benchmark (SPY + extras like QQQ).
    all_benchmarks = {C.BENCHMARK_TICKERS[0]: benchmark, **extra_benchmarks}
    comparison = benchmark_comparison(ls, all_benchmarks)
    comparison.to_csv(cfg.backtest_dir / f"{name}_benchmark_comparison.csv")

    pred_paths = export_predictions(result, cfg)

    tearsheet = cfg.backtest_dir / f"{name}_tearsheet.html"
    try:
        qs.reports.html(ls, benchmark=benchmark.reindex(ls.index),
                        output=str(tearsheet), title=name)
    except Exception as exc:  # noqa: BLE001 - tearsheet is best-effort
        logger.warning("Tearsheet generation failed: %s", exc)

    # One extra tearsheet per additional benchmark (e.g. _tearsheet_QQQ.html).
    for bname, b in extra_benchmarks.items():
        try:
            qs.reports.html(ls, benchmark=b.reindex(ls.index),
                            output=str(cfg.backtest_dir / f"{name}_tearsheet_{bname}.html"),
                            title=f"{name} vs {bname}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tearsheet vs %s failed: %s", bname, exc)

    logger.info("PM gate %s | OOS Sharpe=%.2f CAGR=%.1f%%",
                "PASS" if gate.passed else "FAIL",
                gate.detail["oos_sharpe"], gate.detail["cagr"] * 100)

    return {
        "metrics": metrics,
        "subperiods": sub,
        "pm_gate": gate,
        "benchmark_comparison": comparison,
        "artifacts": {
            "metrics_csv": str(cfg.backtest_dir / f"{name}_metrics.csv"),
            "subperiods_csv": str(cfg.backtest_dir / f"{name}_subperiods.csv"),
            "bimodality_csv": str(cfg.backtest_dir / f"{name}_bimodality.csv"),
            "pm_gate_csv": str(cfg.backtest_dir / f"{name}_pm_gate.csv"),
            "benchmark_comparison_csv": str(cfg.backtest_dir / f"{name}_benchmark_comparison.csv"),
            "tearsheet_html": str(tearsheet),
            **{f"tearsheet_{b}_html": str(cfg.backtest_dir / f"{name}_tearsheet_{b}.html")
               for b in extra_benchmarks},
            **pred_paths,
        },
    }


def export_predictions(result, cfg: C.StrategyConfig) -> dict[str, str]:
    """Write per-ticker OOS diagnostics.

    ``<name>_predictions.csv`` — full cross-section each month: expected return,
    predicted decile, actual decile, realized return, leg, position scale.
    ``<name>_trades.csv`` — only the held names (leg != 0), the long/short book.
    """
    preds = result.predictions
    if preds is None or preds.empty:
        logger.warning("No predictions to export.")
        return {}

    full_path = cfg.backtest_dir / f"{cfg.name}_predictions.csv"
    preds.to_csv(full_path)

    trades = preds[preds["leg"] != 0].copy()
    trades["side"] = trades["leg"].map({1: "LONG", -1: "SHORT"})
    trades = trades.sort_values(["date", "leg", "expected_return"],
                                ascending=[True, False, False])
    cols = ["side", "expected_return", "pred_decile", "actual_decile",
            "realized_return", "scale"]
    trades_path = cfg.backtest_dir / f"{cfg.name}_trades.csv"
    trades[cols].to_csv(trades_path)

    logger.info("Exported %d prediction rows (%d trades) to %s",
                len(preds), len(trades), cfg.backtest_dir)
    return {"predictions_csv": str(full_path), "trades_csv": str(trades_path)}
