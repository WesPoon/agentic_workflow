from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Annotated, Literal

import pandas as pd
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState, InjectedStore
from langgraph.store.base import BaseStore
from langgraph.types import Command

from src.state import current_spec
from src.tools.context import paths_from, thread_id_from

_DENY_NAMES = {".env", "env", ".git", ".venv"}
_SKIP_DIRS = {".venv", ".git", "__pycache__", "node_modules", "cache"}
MAX_READ_CHARS = 200_000


def _denied(path) -> bool:
    return any(part in _DENY_NAMES or part.startswith(".env") for part in path.parts)


@tool
def read_file(file_path: str, config: RunnableConfig) -> str:
    """Read a file from this run's directory (strategies/, backtests/, research/, reviews/) or the
    code repository. Relative paths are resolved against the run directory first."""
    try:
        path = paths_from(config).resolve_inside(file_path)
    except PermissionError as e:
        return f"ERROR: {e}"
    if _denied(path) or path.is_dir():
        return f"ERROR: cannot read {file_path}"
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        return f"ERROR reading file: {e}"
    if len(text) > MAX_READ_CHARS:
        return text[:MAX_READ_CHARS] + f"\n\n[truncated: file is {len(text)} characters]"
    return text


@tool
def list_directory(path: str, config: RunnableConfig) -> str:
    """List files under a directory of this run (e.g. 'strategies', 'backtests', 'research') or the
    code repository. Returns newline-separated paths."""
    try:
        p = paths_from(config).resolve_inside(path)
    except PermissionError as e:
        return f"ERROR: {e}"
    if not p.is_dir():
        return f"ERROR: not a directory: {path}"
    files = sorted(
        str(f) for f in p.rglob("*")
        if f.is_file() and not any(part in _SKIP_DIRS for part in f.relative_to(p).parts) and not _denied(f)
    )
    return "\n".join(files[:500]) if files else "(empty directory)"


@tool
def write_review(strategy_name: str, content: str, config: RunnableConfig) -> str:
    """Write review findings to reviews/review_<strategy_name>_r<revision>.md. Returns file path."""
    paths = paths_from(config)
    existing = sorted(paths.reviews.glob(f"review_{strategy_name}_r*.md"))
    path = paths.reviews / f"review_{strategy_name}_r{len(existing)}.md"
    path.write_text(content, encoding="utf-8")
    return str(path)


@tool
def flag_overfitting_pattern(
    pattern: str,
    strategy_name: str,
    technique: str,
    description: str,
    config: RunnableConfig,
    store: Annotated[BaseStore, InjectedStore()],
    fix_applied: str = "",
) -> str:
    """
    Persist a detected overfitting or methodology error pattern to the Store.
    This is the primary mechanism by which the review agent teaches future runs.
    Always call this when you find a real methodology failure.
    """
    key = hashlib.md5(pattern.encode()).hexdigest()[:16]
    store.put(("review", "overfitting_patterns"), key, {
        "pattern": pattern,
        "strategy_name": strategy_name,
        "technique": technique,
        "description": description,
        "fix_applied": fix_applied,
        "run_id": thread_id_from(config),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return f"Overfitting pattern saved: '{pattern[:60]}'"


def _set_verdict(attempts_csv, attempt_id: str, verdict: str) -> None:
    if not attempt_id or not attempts_csv.exists():
        return
    df = pd.read_csv(attempts_csv, dtype={"attempt_id": str})
    if "verdict" not in df.columns:
        df["verdict"] = ""
    df["verdict"] = df["verdict"].astype("object")
    df.loc[df["attempt_id"] == attempt_id, "verdict"] = verdict
    df.to_csv(attempts_csv, index=False)


@tool
def record_technique_outcome(
    technique: str,
    verdict: Literal["APPROVED", "NEEDS_REVISION"],
    oos_sharpe: float,
    what_worked: str,
    overfitting_flags: list[str],
    revision_notes: str,
    attempt_id: str,
    config: RunnableConfig,
    state: Annotated[dict, InjectedState],
    store: Annotated[BaseStore, InjectedStore()],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Record the review verdict for the strategy under review. This ends the review: call it
    exactly once, after write_review.
    technique: short model-family label taken from the spec's model line, e.g. 'rule_based',
      'linear', 'tree_ensemble', 'neural_net'.
    verdict: 'APPROVED' | 'NEEDS_REVISION'.
    revision_notes: specific, line-level fixes for the implementation agent ('' if approved).
    attempt_id: the attempt_id of the backtest you reviewed (from the run_backtest result).
    """
    spec = current_spec(state)
    strategy_name = spec.get("name", "unknown")
    iteration = state.get("revision_count", 0)
    paths = paths_from(config)
    commit_sha = ""
    for r in reversed(state.get("backtest_results", []) or []):
        if r.get("attempt_id") == attempt_id:
            commit_sha = r.get("commit_sha", "")
            break
    store.put(("review", "technique_outcomes"), f"{technique}_{strategy_name}", {
        "technique": technique,
        "strategy_name": strategy_name,
        "verdict": verdict,
        "oos_sharpe": oos_sharpe,
        "what_worked": what_worked,
        "overfitting_flags": overfitting_flags,
        "revision_notes": revision_notes,
        "attempt_id": attempt_id,
        "commit_sha": commit_sha,
        "run_id": thread_id_from(config),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    _set_verdict(paths.attempts_csv, attempt_id, verdict)
    finding = {
        "strategy_name": strategy_name,
        "verdict": verdict,
        "attempt_id": attempt_id,
        "overfitting_flags": overfitting_flags,
        "revision_notes": revision_notes,
        "iteration": iteration,
    }
    return Command(update={
        "review_findings": [finding],
        "messages": [ToolMessage(f"Verdict recorded: {strategy_name} → {verdict}", tool_call_id=tool_call_id)],
    })


@tool
def query_past_overfitting_patterns(store: Annotated[BaseStore, InjectedStore()]) -> str:
    """Query all stored overfitting patterns from past runs. Returns JSON."""
    return json.dumps([i.value for i in store.search(("review", "overfitting_patterns"), limit=200)], indent=2, default=str)


REVIEW_TOOLS = [read_file, list_directory, write_review, flag_overfitting_pattern,
                record_technique_outcome, query_past_overfitting_patterns]
