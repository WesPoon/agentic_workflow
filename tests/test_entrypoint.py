"""The AgentCore entrypoint returns immediately, holds the session busy while the graph runs,
and releases it when done."""
from __future__ import annotations

import importlib
import json
import sys
import time

from tests.test_pipeline import ScriptedModel


def test_run_is_async_and_session_busy(tmp_path, monkeypatch):
    monkeypatch.setenv("PERSISTENCE", "sqlite")
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "ckpt.db"))
    monkeypatch.setenv("STORAGE", "local")
    monkeypatch.setenv("COMPUTE_BACKEND", "local")
    monkeypatch.setenv("WORK_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("MAX_REVISIONS", "2")
    from src.config import get_settings
    get_settings.cache_clear()
    import src.nodes.common as common
    monkeypatch.setattr(common, "get_llm", lambda: ScriptedModel())
    ScriptedModel.calls, ScriptedModel.fail_once_on, ScriptedModel.last_by_phase = [], None, {}
    paper = tmp_path / "p.pdf"
    paper.write_bytes(b"%PDF-1.4")

    sys.modules.pop("main", None)
    main = importlib.import_module("main")
    t0 = time.time()
    resp = main.invoke({"action": "run", "paper_keys": [str(paper)], "thread_id": "e2e_thread_000000000000000000000001"})
    assert resp == {"thread_id": "e2e_thread_000000000000000000000001", "status": "started"}
    assert time.time() - t0 < 2, "entrypoint must not block"
    assert str(main.app.get_current_ping_status()).endswith("HEALTHY_BUSY") or "Busy" in str(main.app.get_current_ping_status())
    again = main.invoke({"action": "run", "paper_keys": [str(paper)], "thread_id": "e2e_thread_000000000000000000000001"})
    assert again["status"] == "already_running"

    deadline = time.time() + 120
    while time.time() < deadline and main._active:
        time.sleep(0.2)
    assert not main._active
    status = main.invoke({"action": "status", "thread_id": "e2e_thread_000000000000000000000001"})
    assert status["status"] == "complete", json.dumps(status)
    assert status["approved"] == ["alpha_mom"] and status["skipped"] == ["beta_mom"]
    assert "Busy" not in str(main.app.get_current_ping_status()) and "BUSY" not in str(main.app.get_current_ping_status())
    get_settings.cache_clear()
