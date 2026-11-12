"""Helpers that give tools their run context from the graph config instead of module globals."""
from __future__ import annotations

from langchain_core.runnables import RunnableConfig

from src.workdir import RunPaths, run_paths


def thread_id_from(config: RunnableConfig) -> str:
    tid = (config or {}).get("configurable", {}).get("thread_id")
    if not tid:
        raise RuntimeError("thread_id missing from graph config")
    return str(tid)


def paths_from(config: RunnableConfig) -> RunPaths:
    return run_paths(thread_id_from(config)).ensure()
