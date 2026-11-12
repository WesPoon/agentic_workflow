from __future__ import annotations

import ast
import json
from typing import Annotated

import pandas as pd
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState, InjectedStore
from langgraph.store.base import BaseStore
from langgraph.types import Command

from src.config import get_settings
from src.jobs.runner import new_job_id, run_job
from src.state import current_spec
from src.tools.context import paths_from, thread_id_from
from src.workdir import safe_name


@tool
def read_strategy_spec(config: RunnableConfig) -> str:
    """Read research/strategy_spec.md for this run and return its contents."""
    path = paths_from(config).research / "strategy_spec.md"
    if not path.exists():
        return "ERROR: strategy_spec.md not found. Research agent must run first."
    return path.read_text(encoding="utf-8")


@tool
def read_strategy_template() -> str:
    """Return the strategy template that shows the run_backtest(backtests_dir, params) contract."""
    return (get_settings().code_root / "templates" / "strategy_template.py").read_text(encoding="utf-8")


@tool
def write_strategy_file(strategy_name: str, code: str, config: RunnableConfig) -> str:
    """
    Write the strategy module to strategies/<strategy_name>.py in the run directory.
    strategy_name: 1-64 characters from [A-Za-z0-9_]. Validates Python syntax and that
    run_backtest(backtests_dir, params) is defined. Returns the file path or an error.
    """
    try:
        safe_name(strategy_name)
    except ValueError as e:
        return f"ERROR: {e}"
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SYNTAX ERROR — file not written: {e}"
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_backtest"), None)
    if fn is None:
        return "ERROR — file not written: module must define run_backtest(backtests_dir, params)"
    arg_names = [a.arg for a in fn.args.args]
    if arg_names[:2] != ["backtests_dir", "params"]:
        return f"ERROR — file not written: run_backtest signature must be (backtests_dir, params), got {arg_names}"
    path = paths_from(config).strategies / f"{strategy_name}.py"
    path.write_text(code, encoding="utf-8")
    return str(path)


def _next_attempt(attempts_csv, strategy_name: str) -> int:
    if not attempts_csv.exists():
        return 1
    try:
        df = pd.read_csv(attempts_csv)
        return int((df["strategy"] == strategy_name).sum()) + 1
    except Exception:
        return 1


@tool
def run_backtest(
    strategy_name: str,
    params: dict,
    change_note: str,
    config: RunnableConfig,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """
    Run strategies/<strategy_name>.py on the backtest runner and return the evaluator's result.
    params: the parameter dict passed to run_backtest(backtests_dir, params). Put every tunable
            value here (lookbacks, universe, model hyper-parameters) so each attempt is reproducible.
    change_note: one or two sentences — what changed versus the previous attempt and why.
                 It becomes the commit message; required.
    The evaluator computes all metrics (70/30 split, OOS Sharpe with bootstrap CI, deflated Sharpe).
    """
    def _reply(text: str, extra: dict | None = None) -> Command:
        return Command(update={**(extra or {}), "messages": [ToolMessage(text, tool_call_id=tool_call_id)]})

    try:
        safe_name(strategy_name)
    except ValueError as e:
        return _reply(f"ERROR: {e}")
    if not (change_note or "").strip():
        return _reply("ERROR: change_note is required — say what changed and why.")
    paths = paths_from(config)
    strategy_file = paths.strategies / f"{strategy_name}.py"
    if not strategy_file.exists():
        return _reply(f"ERROR: {strategy_file} not found; call write_strategy_file first.")
    try:
        json.dumps(params)
    except TypeError as e:
        return _reply(f"ERROR: params must be JSON-serializable: {e}")

    spec = current_spec(state)
    job = {
        "job_id": new_job_id(),
        "thread_id": thread_id_from(config),
        "strategy_name": strategy_name,
        "revision": state.get("revision_count", 0),
        "attempt": _next_attempt(paths.attempts_csv, strategy_name),
        "params": params or {},
        "change_note": change_note.strip(),
        "spec_hash": spec.get("spec_hash", ""),
    }
    text = run_job(job, strategy_file, paths)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = {"status": "error"}
    record = {
        "strategy_name": strategy_name,
        "revision": job["revision"],
        "attempt": job["attempt"],
        "attempt_id": job["job_id"],
        "commit_sha": parsed.get("commit_sha", ""),
        "status": parsed.get("status", "error"),
        "oos_sharpe": (parsed.get("out_of_sample") or {}).get("sharpe"),
        "deflated_sharpe_prob": parsed.get("deflated_sharpe_prob"),
    }
    return _reply(text, {"backtest_results": [record]})


@tool
def query_past_technique_outcomes(store: Annotated[BaseStore, InjectedStore()]) -> str:
    """Query the Store for past technique outcomes (approved and rejected). Returns JSON."""
    return json.dumps([i.value for i in store.search(("review", "technique_outcomes"), limit=200)], indent=2, default=str)


IMPL_TOOLS = [read_strategy_spec, read_strategy_template, write_strategy_file, run_backtest,
              query_past_technique_outcomes]
