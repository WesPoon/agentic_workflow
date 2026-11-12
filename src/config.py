"""Single source of configuration. Every env var the pipeline reads is declared here.

Local defaults keep `python -m src.runner` working with no AWS at all:
in-memory persistence, local storage, Anthropic API, subprocess backtests.
"""
from __future__ import annotations

import contextvars
import logging

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

CODE_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(CODE_ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    code_root: Path
    aws_region: str

    # LLM
    llm_provider: str          # anthropic | bedrock | xai | deepseek
    llm_model_id: str
    llm_max_tokens: int
    llm_timeout_s: float
    llm_max_retries: int
    llm_effort: str            # '' (model default) | none (deepseek only) | low | medium | high | xhigh | max

    # Persistence (checkpointer + store)
    persistence: str           # memory | sqlite | dynamodb
    sqlite_path: Path
    checkpoint_table: str
    store_table: str
    checkpoint_ttl_days: int

    # Artifact storage
    storage: str               # local | s3
    artifact_bucket: str
    artifact_prefix: str
    work_dir: Path

    # Graph bounds
    recursion_limit: int
    max_revisions: int
    max_phase_steps: int

    # Backtest compute
    compute_backend: str       # local | github
    github_repo: str
    github_base_branch: str
    github_api_url: str
    backtest_timeout_s: int
    job_poll_s: int

    log_level: str

    # ------------------------------------------------------------------ paths
    @property
    def runs_root(self) -> Path:
        return self.work_dir

    def run_dir(self, thread_id: str) -> Path:
        return self.work_dir / thread_id


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    work_dir = Path(_env("WORK_DIR", str(CODE_ROOT / "runs"))).expanduser()
    return Settings(
        code_root=CODE_ROOT,
        aws_region=_env("AWS_REGION", _env("AWS_DEFAULT_REGION", "us-east-1")),
        llm_provider=_env("LLM_PROVIDER", "anthropic").lower(),
        llm_model_id=_env("LLM_MODEL_ID", "claude-opus-5-5"),
        llm_max_tokens=_env_int("LLM_MAX_TOKENS", 8192),
        llm_timeout_s=_env_float("LLM_TIMEOUT_S", 600.0),
        llm_max_retries=_env_int("LLM_MAX_RETRIES", 3),
        llm_effort=_env("LLM_EFFORT", "").lower(),
        persistence=_env("PERSISTENCE", "memory").lower(),
        sqlite_path=Path(_env("SQLITE_PATH", str(CODE_ROOT / "checkpoints" / "momentum_research.db"))),
        checkpoint_table=_env("CHECKPOINT_TABLE", "momentum-checkpoints"),
        store_table=_env("STORE_TABLE", "momentum-store"),
        checkpoint_ttl_days=_env_int("CHECKPOINT_TTL_DAYS", 30),
        storage=_env("STORAGE", "local").lower(),
        artifact_bucket=_env("ARTIFACT_BUCKET", ""),
        artifact_prefix=_env("ARTIFACT_PREFIX", "").strip("/"),
        work_dir=work_dir,
        recursion_limit=_env_int("RECURSION_LIMIT", 300),
        max_revisions=_env_int("MAX_REVISIONS", 3),
        max_phase_steps=_env_int("MAX_PHASE_STEPS", 80),
        compute_backend=_env("COMPUTE_BACKEND", "local").lower(),
        github_repo=_env("GITHUB_REPO", ""),
        github_base_branch=_env("GITHUB_BASE_BRANCH", "main"),
        github_api_url=_env("GITHUB_API_URL", "https://api.github.com").rstrip("/"),
        backtest_timeout_s=_env_int("BACKTEST_TIMEOUT_S", 1800),
        job_poll_s=_env_int("JOB_POLL_S", 15),
        log_level=_env("LOG_LEVEL", "INFO").upper(),
    )


_RUN_ID = contextvars.ContextVar("run_id", default="")
_process_run_id = ""          # fallback for threads that do not inherit the context (one run per process)


def set_run_id(run_id: str) -> None:
    """Tag every later log line of this run (this context and, as a fallback, this process) with run_id."""
    global _process_run_id
    _RUN_ID.set(run_id)
    _process_run_id = run_id


class _RunContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _RUN_ID.get() or _process_run_id or "-"
        return True


def configure_logging() -> None:
    """Every line: time, level, [run=<batch id> pid=<process id>], logger, message."""
    logging.basicConfig(
        level=get_settings().log_level,
        format="%(asctime)s %(levelname)s [run=%(run_id)s pid=%(process)d] %(name)s: %(message)s",
        force=True,
    )
    for handler in logging.getLogger().handlers:   # handler-level, so library loggers get it too
        handler.addFilter(_RunContextFilter())
