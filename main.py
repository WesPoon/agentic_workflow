"""Amazon Bedrock AgentCore Runtime entrypoint.

Payloads:
  {"action": "run",    "paper_keys": ["papers/a.pdf", ...], "thread_id": optional}
  {"action": "resume", "thread_id": "..."}      continue from the last checkpoint
  {"action": "status", "thread_id": "..."}      read runs/<thread_id>/status.json

"run" and "resume" return immediately with the thread_id; the graph runs on a background thread
registered as an AgentCore async task, so /ping reports HealthyBusy and the session is not reaped
while it works (up to the 8 h microVM limit). Progress is written to S3 status.json.
"""
from __future__ import annotations

import logging
import threading

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from src.config import configure_logging, get_settings
from src.secrets import load_secrets_into_env

configure_logging()
load_secrets_into_env(get_settings().aws_region)

from src.graph import build_graph  # noqa: E402  (after secrets are in the env)
from src.persistence import make_checkpointer, make_store  # noqa: E402
from src.runner import execute, new_thread_id  # noqa: E402
from src.state import initial_state  # noqa: E402
from src.storage import hydrate_run, read_status, write_status  # noqa: E402

log = logging.getLogger("entrypoint")

app = BedrockAgentCoreApp()
graph = build_graph(make_checkpointer(), make_store())

_lock = threading.Lock()
_active: dict[str, int] = {}          # thread_id -> async task id, for this microVM


def _work(action: str, thread_id: str, paper_keys: list[str], task_id: int) -> None:
    try:
        if action == "run":
            paper_paths = hydrate_run(thread_id, paper_keys)
            execute(graph, thread_id, initial_state(thread_id, paper_paths))
        else:
            hydrate_run(thread_id, [])
            config = {"configurable": {"thread_id": thread_id}}
            snapshot = graph.get_state(config)
            if not snapshot.values:
                write_status(thread_id, "failed", error="no checkpoint for this thread_id")
                return
            if not snapshot.next:
                write_status(thread_id, "complete", note="thread had already finished")
                return
            execute(graph, thread_id, None)
    except Exception:
        log.exception("run %s failed", thread_id)   # execute() already wrote status=failed
    finally:
        with _lock:
            _active.pop(thread_id, None)
        app.complete_async_task(task_id)


@app.entrypoint
def invoke(payload: dict, context=None) -> dict:
    action = (payload or {}).get("action", "run")
    if action == "status":
        return read_status(payload["thread_id"])
    if action not in ("run", "resume"):
        return {"error": f"unknown action {action!r}; use run | resume | status"}

    paper_keys = list(payload.get("paper_keys") or [])
    if action == "run" and not paper_keys:
        return {"error": "run needs paper_keys, e.g. [\"papers/my_paper.pdf\"]"}
    if action == "resume" and not payload.get("thread_id"):
        return {"error": "resume needs thread_id"}
    thread_id = payload.get("thread_id") or new_thread_id(paper_keys)

    with _lock:
        if thread_id in _active:
            return {"thread_id": thread_id, "status": "already_running"}
        task_id = app.add_async_task("pipeline", {"thread_id": thread_id, "action": action})
        _active[thread_id] = task_id
    write_status(thread_id, "started", action=action, paper_keys=paper_keys)
    threading.Thread(target=_work, args=(action, thread_id, paper_keys, task_id), daemon=True,
                     name=f"pipeline-{thread_id}").start()
    return {"thread_id": thread_id, "status": "started"}


if __name__ == "__main__":
    app.run()
