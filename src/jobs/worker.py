"""Execute one backtest job. Runs on the EC2 (GitHub Actions self-hosted runner) or locally.

    python -m src.jobs.worker path/to/job.json

The job JSON (written by src/jobs/runner.py) contains:
  job_id, thread_id, strategy_name, revision, attempt, params, change_note, spec_hash,
  strategy_path (absolute, local backend) or strategy_relpath (relative to the checkout),
  output: {"kind": "local", "dir": ...} | {"kind": "s3", "bucket": ..., "prefix": ...},
  attempts_csv: local path or S3 key.

The strategy file must define run_backtest(backtests_dir: str, params: dict) -> dict with keys
  returns (pd.Series daily net returns), benchmark_returns (pd.Series), positions (optional),
  meta (optional dict).
The worker always writes result.json; a failing strategy is a result, not a worker failure.
"""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import logging
import os
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.jobs.evaluate import evaluate, write_reports

log = logging.getLogger("worker")

ATTEMPT_COLUMNS = [
    "attempt_id", "timestamp", "strategy", "revision", "attempt", "commit_sha", "branch", "change_note",
    "oos_sharpe", "oos_sharpe_ci95_low", "oos_sharpe_ci95_high", "deflated_sharpe_prob",
    "cagr", "max_drawdown", "n_trials", "status", "verdict", "family", "eval_mode",
]


# ----------------------------------------------------------------------------- IO backends

class _Output:
    def __init__(self, spec: dict, attempts_ref: str):
        self.kind = spec.get("kind", "local")
        self.spec = spec
        self.attempts_ref = attempts_ref
        if self.kind == "local":
            self.dir = Path(spec["dir"])
        else:
            self.dir = Path(tempfile.mkdtemp(prefix="job_"))
        self.dir.mkdir(parents=True, exist_ok=True)

    # attempts.csv ----------------------------------------------------------
    def read_attempts(self) -> pd.DataFrame:
        try:
            if self.kind == "local":
                p = Path(self.attempts_ref)
                return pd.read_csv(p) if p.exists() else pd.DataFrame(columns=ATTEMPT_COLUMNS)
            import boto3

            body = boto3.client("s3").get_object(Bucket=self.spec["bucket"], Key=self.attempts_ref)["Body"].read()
            return pd.read_csv(io.BytesIO(body))
        except Exception as e:  # missing object or unreadable file
            log.info("no attempts.csv yet (%s)", e.__class__.__name__)
            return pd.DataFrame(columns=ATTEMPT_COLUMNS)

    def append_attempt(self, row: dict) -> None:
        df = self.read_attempts()
        df = pd.concat([df, pd.DataFrame([row])[ATTEMPT_COLUMNS]], ignore_index=True)
        if self.kind == "local":
            p = Path(self.attempts_ref)
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p, index=False)
        else:
            import boto3

            boto3.client("s3").put_object(Bucket=self.spec["bucket"], Key=self.attempts_ref,
                                          Body=df.to_csv(index=False).encode("utf-8"))

    # job outputs --------------------------------------------------------------
    def finalize(self) -> None:
        if self.kind != "s3":
            return
        import boto3

        s3 = boto3.client("s3")
        prefix = self.spec["prefix"].rstrip("/")
        for path in self.dir.rglob("*"):
            if path.is_file():
                s3.upload_file(str(path), self.spec["bucket"], f"{prefix}/{path.relative_to(self.dir).as_posix()}")


# ----------------------------------------------------------------------------- core

def _load_strategy(path: Path):
    spec = importlib.util.spec_from_file_location(f"strategy_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load strategy from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run_backtest"):
        raise AttributeError("strategy file must define run_backtest(backtests_dir, params)")
    return module


def _check_contract(out) -> tuple[pd.Series, pd.Series, pd.DataFrame | None, dict]:
    if not isinstance(out, dict):
        raise TypeError("run_backtest must return a dict")
    missing = [k for k in ("returns", "benchmark_returns") if k not in out]
    if missing:
        raise KeyError(f"run_backtest result missing required keys: {missing}")
    returns, bench = out["returns"], out["benchmark_returns"]
    for name, s in (("returns", returns), ("benchmark_returns", bench)):
        if not isinstance(s, pd.Series):
            raise TypeError(f"{name} must be a pandas Series indexed by date")
    from src.jobs.evaluate import infer_periods

    periods = infer_periods(returns.dropna().index)          # raises unless daily or monthly
    minimum = 60 if periods == 252 else 36                    # ~3 months daily / 3 years monthly
    if len(returns.dropna()) < minimum:
        raise ValueError(f"returns has {len(returns.dropna())} non-null observations; need >= {minimum} "
                         f"({'daily' if periods == 252 else 'monthly'})")
    if len(bench.dropna()) < minimum:
        raise ValueError(f"benchmark_returns has fewer than {minimum} non-null observations")
    positions = out.get("positions")
    if positions is not None and not isinstance(positions, pd.DataFrame):
        raise TypeError("positions must be a DataFrame or None")
    meta = out.get("meta") or {}
    if not isinstance(meta, dict):
        raise TypeError("meta must be a dict")
    return returns, bench, positions, meta


def apply_window(window: dict, returns: pd.Series, bench: pd.Series, positions):
    """Cut the strategy output to the evaluation window BEFORE anything is scored or written.

    split   : default; evaluator splits returns 70/30 by date (in-sample / out-of-sample).
    design  : keep returns up to window["end"] (e.g. 2018-12-31); everything is scored as one
              design segment (reported in the out_of_sample slot). Later returns are discarded
              even if the strategy computed them, so the holdout never leaks into results.
    holdout : keep returns from window["start"] (e.g. 2019-01-01); scored as one segment.
    Returns (returns, bench, positions, split) for evaluate().
    """
    mode = window.get("mode", "split")
    if mode == "split":
        return returns, bench, positions, float(window.get("split", 0.7))
    idx = pd.to_datetime(returns.index)
    if mode == "design":
        keep = idx <= pd.Timestamp(window["end"])
    elif mode == "holdout":
        keep = idx >= pd.Timestamp(window["start"])
    else:
        raise ValueError(f"unknown evaluation window mode {mode!r}")
    returns = returns[keep]
    if len(returns.dropna()) < 12:
        raise ValueError(f"only {len(returns.dropna())} returns inside the {mode} window {window}")
    lo, hi = pd.to_datetime(returns.index).min(), pd.to_datetime(returns.index).max()
    bidx = pd.to_datetime(bench.index)
    bench = bench[(bidx >= lo - pd.Timedelta(days=31)) & (bidx <= hi)]
    if positions is not None:
        pidx = pd.to_datetime(positions.index)
        positions = positions[(pidx >= lo - pd.Timedelta(days=31)) & (pidx <= hi)]
    return returns, bench, positions, 0.0


def run(job: dict) -> dict:
    strategy_path = Path(job.get("strategy_path") or (Path.cwd() / job["strategy_relpath"])).resolve()
    out = _Output(job["output"], job["attempts_csv"])
    strategy = job["strategy_name"]
    commit_sha = job.get("commit_sha") or os.environ.get("GITHUB_SHA", "")
    branch = job.get("branch") or os.environ.get("GITHUB_REF_NAME", "")
    base = {
        "status": "ok",
        "job_id": job["job_id"],
        "strategy_name": strategy,
        "revision": job.get("revision"),
        "attempt": job.get("attempt"),
        "commit_sha": commit_sha,
        "branch": branch,
        "change_note": job.get("change_note", ""),
        "params": job.get("params", {}),
        "spec_hash": job.get("spec_hash", ""),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "strategy_file": str(strategy_path),
    }
    # Trials are counted per family: the strategy itself by default, the whole paper group in group
    # mode (baseline + every variant + finalists all count against the winner's deflated Sharpe).
    family = job.get("family") or strategy
    window = job.get("eval") or {"mode": "split", "split": 0.7}
    attempts = out.read_attempts()
    if len(attempts):
        fam_col = attempts["family"] if "family" in attempts.columns else pd.Series(index=attempts.index, dtype=object)
        fam = fam_col.where(fam_col.notna() & (fam_col.astype(str) != ""), attempts["strategy"])
        prior = attempts[(fam == family) & (attempts["status"] == "ok")]
    else:
        prior = attempts
    n_trials = int(len(prior)) + 1
    prior_sharpes = [float(x) for x in pd.to_numeric(prior.get("oos_sharpe", pd.Series(dtype=float)), errors="coerce").dropna()] if len(prior) else []

    result: dict
    try:
        module = _load_strategy(strategy_path)
        raw = module.run_backtest(backtests_dir=str(out.dir), params=dict(job.get("params", {})))
        returns, bench, positions, meta = _check_contract(raw)
        returns, bench, positions, split = apply_window(window, returns, bench, positions)
        metrics = evaluate(returns, bench, n_trials=n_trials, prior_oos_sharpes=prior_sharpes, split=split)
        metrics["eval_window"] = window
        files = write_reports(metrics, returns, bench, out.dir, title=f"{strategy} r{job.get('revision')} a{job.get('attempt')}")
        if positions is not None:
            positions.to_csv(out.dir / "positions.csv")
            files["positions"] = "positions.csv"
        result = {**base, "meta": meta, "files": files, **metrics}
    except Exception as e:
        tb = traceback.format_exc()
        log.error("strategy %s failed: %s", strategy, e)
        result = {**base, "status": "error", "error": f"{e.__class__.__name__}: {e}", "traceback": tb[-6000:],
                  "n_trials": n_trials}

    (out.dir / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    oos = result.get("out_of_sample", {})
    ci = result.get("oos_sharpe_ci95", [None, None])
    out.append_attempt({
        "attempt_id": job["job_id"],
        "timestamp": result["evaluated_at"],
        "strategy": strategy,
        "revision": job.get("revision"),
        "attempt": job.get("attempt"),
        "commit_sha": commit_sha,
        "branch": branch,
        "change_note": job.get("change_note", ""),
        "oos_sharpe": oos.get("sharpe"),
        "oos_sharpe_ci95_low": ci[0],
        "oos_sharpe_ci95_high": ci[1],
        "deflated_sharpe_prob": result.get("deflated_sharpe_prob"),
        "cagr": oos.get("cagr"),
        "max_drawdown": oos.get("max_drawdown"),
        "n_trials": n_trials,
        "status": result["status"],
        "verdict": "",
        "family": family,
        "eval_mode": window.get("mode", "split"),
    })
    out.finalize()
    return result


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if len(argv) != 1:
        print(__doc__)
        return 2
    from src.secrets import load_secrets_into_env

    load_secrets_into_env(os.environ.get("AWS_REGION"))
    job = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    result = run(job)
    print(json.dumps({k: result.get(k) for k in ("status", "job_id", "strategy_name", "error")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
