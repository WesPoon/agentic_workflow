"""Local CLI. The same graph and factories the AgentCore entrypoint (main.py) uses.

  python -m src.runner run --papers papers/a.pdf papers/b.pdf [--thread-id ID]
  python -m src.runner resume <thread_id>            # continue from the last checkpoint
  python -m src.runner status <thread_id>
  python -m src.runner inspect

Persistence and storage follow the env (see src/config.py). For a run you can resume locally:
  PERSISTENCE=sqlite python -m src.runner run --papers ...
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import uuid
from datetime import date

from src.config import configure_logging, get_settings, set_run_id
from src.graph import build_graph
from src.persistence import make_checkpointer, make_store
from src.secrets import load_secrets_into_env
from src.state import initial_state
from src.storage import hydrate_run, persist_run, read_status, write_status

log = logging.getLogger("runner")

STORE_NAMESPACES = [
    ("research", "extracted_signals"),
    ("review", "overfitting_patterns"),
    ("review", "technique_outcomes"),
    ("review", "skipped_strategies"),
    ("runs", "manifests"),
]


def new_thread_id(paper_keys: list[str]) -> str:
    papers_hash = hashlib.md5("|".join(sorted(paper_keys)).encode()).hexdigest()[:8]
    return f"batch_{date.today().isoformat()}_{papers_hash}_{uuid.uuid4().hex[:8]}"


def graph_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": get_settings().recursion_limit}


def execute(graph, thread_id: str, inputs) -> dict:
    set_run_id(thread_id)
    """Stream the graph to completion, keeping status.json current. Returns the final status."""
    config = graph_config(thread_id)
    write_status(thread_id, "running", node="start")
    try:
        for event in graph.stream(inputs, config, stream_mode="updates"):
            for node in event:
                log.info("[%s] done", node)
                write_status(thread_id, "running", node=node)
        values = graph.get_state(config).values
        return write_status(
            thread_id, "complete",
            strategies=[s.get("name") for s in values.get("strategy_specs", []) or []],
            approved=sorted({f["strategy_name"] for f in values.get("review_findings", []) or []
                             if f.get("verdict") == "APPROVED"}),
            skipped=values.get("skipped_strategies", []),
            excluded=[{"name": x.get("name"), "reason": x.get("exclusion_reason")}
                      for x in values.get("excluded_strategies", []) or []],
            attempts=len(values.get("backtest_results", []) or []),
        )
    except Exception as e:
        write_status(thread_id, "failed", error=f"{e.__class__.__name__}: {e}")
        raise
    finally:
        persist_run(thread_id)


def start_run(paper_keys: list[str], thread_id: str | None = None, graph=None) -> str:
    thread_id = thread_id or new_thread_id(paper_keys)
    paper_paths = hydrate_run(thread_id, paper_keys)
    graph = graph or build_graph(make_checkpointer(), make_store())
    set_run_id(thread_id)
    log.info("starting run %s (pid %d) with %d paper(s)", thread_id, os.getpid(), len(paper_paths))
    execute(graph, thread_id, initial_state(thread_id, paper_paths))
    return thread_id


def resume_run(thread_id: str, graph=None) -> None:
    set_run_id(thread_id)
    graph = graph or build_graph(make_checkpointer(), make_store())
    snapshot = graph.get_state(graph_config(thread_id))
    if not snapshot.values:
        raise SystemExit(f"no checkpoint for thread {thread_id} (PERSISTENCE={get_settings().persistence})")
    hydrate_run(thread_id, [])
    if not snapshot.next:
        log.info("thread %s already finished", thread_id)
        return
    log.info("resuming %s (pid %d) at %s", thread_id, os.getpid(), snapshot.next)
    execute(graph, thread_id, None)


def inspect_store() -> None:
    store = make_store()
    for ns in STORE_NAMESPACES:
        items = store.search(ns, limit=1000)
        print(f"\n=== {'/'.join(ns)} ({len(items)} items) ===")
        for item in items:
            print(json.dumps(item.value, indent=2, default=str))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Momentum Research Pipeline")
    sub = parser.add_subparsers(dest="command")
    run_p = sub.add_parser("run", help="Start a new pipeline run")
    run_p.add_argument("--papers", nargs="+", required=True,
                       help="Local PDF paths (STORAGE=local) or S3 keys like papers/x.pdf (STORAGE=s3)")
    run_p.add_argument("--thread-id", default=None)
    resume_p = sub.add_parser("resume", help="Continue a run from its last checkpoint")
    resume_p.add_argument("thread_id")
    status_p = sub.add_parser("status", help="Print a run's status.json")
    status_p.add_argument("thread_id")
    sub.add_parser("inspect", help="Print Store contents")
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    load_secrets_into_env(get_settings().aws_region)
    args = _parse_args()
    if args.command == "run":
        print(start_run(args.papers, args.thread_id))
    elif args.command == "resume":
        resume_run(args.thread_id)
    elif args.command == "status":
        print(json.dumps(read_status(args.thread_id), indent=2))
    elif args.command == "inspect":
        inspect_store()
    else:
        print(__doc__)
        sys.exit(1)
