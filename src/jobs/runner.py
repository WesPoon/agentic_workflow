"""Dispatch a backtest job and return the worker's result as a string for the agent.

COMPUTE_BACKEND=local  → run src.jobs.worker as a subprocess (dev, tests).
COMPUTE_BACKEND=github → commit the strategy + job spec to a branch; the GitHub Actions workflow
                         runs the worker on the EC2 runner; poll S3 for result.json.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

from src.config import get_settings
from src.storage import run_prefix, s3_download_file, s3_download_prefix, s3_exists, s3_get_text
from src.workdir import RunPaths

log = logging.getLogger(__name__)

COPY_BACK = ("result.json", "metrics.csv", "subperiods.csv", "returns.csv", "tearsheet.html", "positions.csv")


def new_job_id() -> str:
    return uuid.uuid4().hex[:10]


def _summarize(result: dict, local_files: dict[str, str]) -> str:
    """Trim the result for the agent: drop the traceback body and keep headline numbers first."""
    keep = {k: v for k, v in result.items() if k not in ("traceback",)}
    if result.get("status") == "error":
        tb = result.get("traceback", "")
        keep["traceback_tail"] = tb[-2500:]
    keep["local_files"] = local_files
    return json.dumps(keep, indent=1, default=str)


def _copy_back(job_dir: Path, paths: RunPaths, strategy: str, attempt: int) -> dict[str, str]:
    files: dict[str, str] = {}
    for name in COPY_BACK:
        src = job_dir / name
        if src.exists():
            dest = paths.backtests / f"{strategy}_a{attempt}_{name}"
            shutil.copy2(src, dest)
            files[name] = str(dest)
    return files


_SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "SESSION")


def worker_env() -> dict[str, str]:
    """Environment for the backtest subprocess: everything except credentials.

    The worker runs LLM-written code, so API keys (LLM providers, Polygon, GitHub, AWS) are not
    passed on. The strategy reads market data from the local cache and yfinance only."""
    import os

    return {k: v for k, v in os.environ.items()
            if not any(h in k.upper() for h in _SECRET_HINTS) and not k.upper().startswith("AWS_")}


def run_job(job: dict, strategy_file: Path, paths: RunPaths) -> str:
    s = get_settings()
    job = dict(job)
    job.setdefault("job_id", new_job_id())
    paths.jobs.mkdir(parents=True, exist_ok=True)
    job_dir = paths.jobs / job["job_id"]

    if s.compute_backend == "local":
        job["strategy_path"] = str(strategy_file)
        job["output"] = {"kind": "local", "dir": str(job_dir)}
        job["attempts_csv"] = str(paths.attempts_csv)
        job_file = paths.jobs / f"{job['job_id']}.json"
        job_file.write_text(json.dumps(job, indent=2, default=str), encoding="utf-8")
        from src.jobs.local_git import snapshot

        sha = snapshot(paths.root, [strategy_file, job_file],
                       f"{job['strategy_name']} r{job.get('revision', 0)} a{job.get('attempt', 0)} "
                       f"[{job['job_id']}]: {job.get('change_note', '')}")
        if sha:
            job["commit_sha"], job["branch"] = sha, "local"
            job_file.write_text(json.dumps(job, indent=2, default=str), encoding="utf-8")
        timeout_s = int(job.get("timeout_s") or s.backtest_timeout_s)
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "src.jobs.worker", str(job_file)],
                cwd=str(s.code_root), capture_output=True, text=True, timeout=timeout_s, env=worker_env(),
            )
        except subprocess.TimeoutExpired:
            return f"BACKTEST TIMEOUT: exceeded {timeout_s} seconds."
        result_file = job_dir / "result.json"
        if not result_file.exists():
            return f"WORKER ERROR (exit {proc.returncode}):\n{proc.stderr[-3000:]}"
        result = json.loads(result_file.read_text(encoding="utf-8"))
        return _summarize(result, _copy_back(job_dir, paths, job["strategy_name"], job.get("attempt", 0)))

    if s.compute_backend == "github":
        from src.jobs.github_runner import GitHubRunner

        if s.storage != "s3" or not s.artifact_bucket:
            raise RuntimeError("COMPUTE_BACKEND=github requires STORAGE=s3 and ARTIFACT_BUCKET")
        prefix = run_prefix(job["thread_id"], "jobs", job["job_id"])
        job["strategy_relpath"] = f"strategies/{job['strategy_name']}.py"
        job["output"] = {"kind": "s3", "bucket": s.artifact_bucket, "prefix": prefix}
        job["attempts_csv"] = run_prefix(job["thread_id"], "attempts.csv")
        branch = f"run/{job['thread_id']}/{job['strategy_name']}-r{job.get('revision', 0)}-{job['job_id']}"
        job["branch"] = branch
        message = f"{job['strategy_name']} r{job.get('revision', 0)} a{job.get('attempt', 0)}: {job.get('change_note', '')}".strip()
        gh = GitHubRunner(s.github_repo, s.github_base_branch, s.github_api_url)
        sha = gh.submit(branch, {
            job["strategy_relpath"]: strategy_file.read_text(encoding="utf-8"),
            f"jobs/{job['job_id']}.json": json.dumps({**job, "commit_sha": ""}, indent=2, default=str),
        }, message)
        job["commit_sha"] = sha
        (paths.jobs / f"{job['job_id']}.json").write_text(json.dumps(job, indent=2, default=str), encoding="utf-8")

        deadline = time.time() + s.backtest_timeout_s
        key = f"{prefix}/result.json"
        while time.time() < deadline:
            text = s3_get_text(key)
            if text is not None:
                result = json.loads(text)
                s3_download_prefix(prefix, job_dir)
                attempts_key = run_prefix(job["thread_id"], "attempts.csv")
                if s3_exists(attempts_key):  # the worker appended to the S3 copy; refresh ours
                    s3_download_file(attempts_key, paths.attempts_csv)
                return _summarize(result, _copy_back(job_dir, paths, job["strategy_name"], job.get("attempt", 0)))
            time.sleep(s.job_poll_s)
        return (f"BACKTEST TIMEOUT: no result after {s.backtest_timeout_s} s for branch {branch} "
                f"(commit {sha[:8]}). Check the GitHub Actions run and the EC2 runner.")

    raise ValueError(f"unknown COMPUTE_BACKEND {s.compute_backend!r}")
