"""Offline end-to-end test of the graph with a scripted model.

Real: graph, routing, tools, Command updates, local backtest worker + evaluator, SQLite checkpoints,
JSON-file store, crash + resume. Fake: only the LLM.
"""
from __future__ import annotations

import itertools
import json
import re
import sys
from typing import ClassVar
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

ROOT = Path(__file__).resolve().parents[1]
STRATEGY_CODE = (ROOT / "tests" / "jobs" / "synthetic_strategy.py").read_text()
def _spec_block(title: str, model: str, lookback: str) -> str:
    return f"""## Strategy: {title}
- implementable: yes — empirical momentum paper with full parameters
- document_type: empirical
- strategy_type: cross_sectional_momentum
- model: {model}
- Lookback: {lookback} (p. 3: "formation period")
- Reference result: Sharpe 0.8 annualized, long-short, Table 2
- Data requirements: CRSP -> yfinance; survivorship bias
- Implementation notes: skip the most recent month
"""


SPEC = ("# Consolidated spec\n" + _spec_block("Alpha Mom", "rule-based 12-1 ranking", "12 months")
        + _spec_block("Beta Mom", "XGBoost classifier", "6 months") + """## Strategy: Gamma Review
- implementable: no — review paper, no parameters or results
- document_type: review
- strategy_type: cross_sectional_momentum
- model: random forest consensus
- Source paper: A Review, Someone, 2025
""")
_ids = itertools.count(1)


def _call(name: str, **args) -> dict:
    return {"name": name, "args": args, "id": f"call_{next(_ids)}", "type": "tool_call"}


class ScriptedModel(BaseChatModel):
    """Decides the next message from the system prompt (which agent / strategy / revision)
    and from how far the current phase has progressed. Also enforces API invariants."""

    calls: list = []
    nudges: ClassVar[list] = []       # (phase key, user message the graph appended after a stop)
    fail_once_on: str | None = None   # e.g. "review:beta_mom:0" → raise once (simulated crash)
    last_by_phase: dict = {}          # phase key -> (system prompt, serialized conversation)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        assert isinstance(messages[0], SystemMessage)
        convo = messages[1:]
        # Invariants the Anthropic Messages API enforces
        assert isinstance(convo[0], HumanMessage), f"conversation must start on a user turn: {convo[0]!r}"
        assert not isinstance(convo[-1], AIMessage), "request must not end on an assistant turn (prefill)"
        seen_calls = {tc["id"] for m in convo if isinstance(m, AIMessage)
                      for tc in list(m.tool_calls) + list(m.invalid_tool_calls or [])}
        for m in convo:
            if isinstance(m, ToolMessage):
                assert m.tool_call_id in seen_calls, "tool result without its tool call"

        system = messages[0].content
        ai_turns = sum(isinstance(m, AIMessage) for m in convo)
        if "research analyst" in system:
            agent, key = "research", "research"
        elif "quantitative developer" in system:
            agent = "impl"
            name = re.search(r"Name \(use exactly this as strategy_name\): (\w+)", system).group(1)
            key = f"impl:{name}"
        else:
            agent = "review"
            name = re.search(r"^Name: (\w+)", system, re.M).group(1)
            rev = int(re.search(r"Revision iteration: (\d+)", system).group(1))
            key = f"review:{name}:{rev}"
        # Opus 5.5 preserved thinking: within one conversation the system prompt is frozen and the
        # history only grows (no edits, no trimming). A new phase starts a fresh conversation.
        def _ser(m):
            return (type(m).__name__, str(m.content), tuple(tc["id"] for tc in getattr(m, "tool_calls", []) or []),
                    getattr(m, "tool_call_id", None))
        convo_ser = [_ser(m) for m in convo]
        prev = ScriptedModel.last_by_phase.get(key)
        if prev is not None and ai_turns > 0:
            prev_system, prev_convo = prev
            assert system == prev_system, f"system prompt changed within phase {key}"
            assert convo_ser[:len(prev_convo)] == prev_convo, f"history edited within phase {key}"
            assert len(convo_ser) > len(prev_convo)
        ScriptedModel.last_by_phase[key] = (system, convo_ser)
        ScriptedModel.calls.append((key, ai_turns))
        if ai_turns > 0 and isinstance(convo[-1], HumanMessage):
            ScriptedModel.nudges.append((key, convo[-1].content))
        if ScriptedModel.fail_once_on == key and ai_turns == 0:
            ScriptedModel.fail_once_on = None
            raise RuntimeError("simulated crash")

        msg = self._next(agent, system, ai_turns)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _next(self, agent: str, system: str, step: int) -> AIMessage:
        if agent == "research":
            if step == 0:
                return AIMessage(content="", tool_calls=[_call("write_research_brief", title="Paper A", content="brief")])
            if step == 1:
                return AIMessage(content="", tool_calls=[_call("write_strategy_spec", content=SPEC)])
            return AIMessage(content="Research done.")
        if agent == "impl":
            name = re.search(r"Name \(use exactly this as strategy_name\): (\w+)", system).group(1)
            if name == "beta_mom" and step == 0:   # cut off at LLM_MAX_TOKENS before any tool call
                return AIMessage(content="", response_metadata={"finish_reason": "length"})
            if name == "beta_mom" and step == 1:   # stops early: nothing written, nothing backtested
                return AIMessage(content="Ready for review.")
            if name == "alpha_mom" and "Revision" not in system and step == 0:
                return AIMessage(content="", invalid_tool_calls=[{"type": "invalid_tool_call", "id": "cut1",
                                 "name": "write_strategy_file", "args": '{"strategy_name": "alpha_mom", "code": "imp',
                                 "error": "truncated"}], response_metadata={"finish_reason": "length"})
            step -= 2 if name == "beta_mom" else (1 if name == "alpha_mom" and "Revision" not in system else 0)
            if step == 0:
                return AIMessage(content="", tool_calls=[_call("write_strategy_file", strategy_name=name, code=STRATEGY_CODE)])
            if step == 1:
                return AIMessage(content="", tool_calls=[_call("run_backtest", strategy_name=name,
                                                               params={"seed": 3, "drift": 0.0003},
                                                               change_note="first implementation")])
            return AIMessage(content="Ready for review.")
        # review
        name = re.search(r"^Name: (\w+)", system, re.M).group(1)
        rev = int(re.search(r"Revision iteration: (\d+)", system).group(1))
        attempt_ids = re.findall(r"attempt_id=(\w+)", system)
        verdict = "APPROVED" if (name == "alpha_mom" and rev >= 1) else "NEEDS_REVISION"
        if step == 0:
            return AIMessage(content="", tool_calls=[_call("read_file", file_path=f"strategies/{name}.py"),
                                                     _call("write_review", strategy_name=name, content="review")])
        if step == 1:
            return AIMessage(content="Thinking out loud without finishing.")   # forces a nudge
        if step == 2:
            return AIMessage(content="", tool_calls=[_call(
                "record_technique_outcome", technique="traditional", verdict=verdict, oos_sharpe=1.0,
                what_worked="n/a", overfitting_flags=[], revision_notes="fix X" if verdict != "APPROVED" else "",
                attempt_id=attempt_ids[-1] if attempt_ids else "")])
        return AIMessage(content="Review complete.")


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("PERSISTENCE", "sqlite")
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "ckpt.db"))
    monkeypatch.setenv("STORAGE", "local")
    monkeypatch.setenv("COMPUTE_BACKEND", "local")
    monkeypatch.setenv("WORK_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("MAX_REVISIONS", "2")
    monkeypatch.setenv("MAX_PHASE_STEPS", "10")
    from src.config import get_settings
    get_settings.cache_clear()
    import src.nodes.common as common
    monkeypatch.setattr(common, "get_llm", lambda: ScriptedModel())
    ScriptedModel.calls = []
    ScriptedModel.nudges = []
    ScriptedModel.fail_once_on = None
    ScriptedModel.last_by_phase = {}
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4 placeholder")
    yield tmp_path, paper
    get_settings.cache_clear()


def test_full_run_with_revision_skip_crash_and_resume(env):
    tmp_path, paper = env
    from src import runner
    from src.persistence import make_checkpointer, make_store
    from src.graph import build_graph

    ScriptedModel.fail_once_on = "review:beta_mom:0"
    with pytest.raises(RuntimeError, match="simulated crash"):
        runner.start_run([str(paper)], thread_id="t_test_000000000000000000000000001")
    status = json.loads((tmp_path / "runs" / "t_test_000000000000000000000000001" / "status.json").read_text())
    assert status["status"] == "failed"

    # New process: fresh graph + checkpointer from the same SQLite file
    runner.resume_run("t_test_000000000000000000000000001", graph=build_graph(make_checkpointer(), make_store()))

    run_dir = tmp_path / "runs" / "t_test_000000000000000000000000001"
    status = json.loads((run_dir / "status.json").read_text())
    assert status["status"] == "complete", status
    assert status["strategies"] == ["alpha_mom", "beta_mom"]          # gamma_review never implemented
    assert status["excluded"] == [{"name": "gamma_review", "reason": "review paper, no parameters or results"}]
    assert status["approved"] == ["alpha_mom"]
    assert status["skipped"] == ["beta_mom"]          # failed review MAX_REVISIONS=2 times

    # Files land in the run dir, not the source tree
    assert (run_dir / "research" / "strategy_spec.md").exists()
    assert (run_dir / "strategies" / "alpha_mom.py").exists()
    assert not (ROOT / "src" / "strategies" / "alpha_mom.py").exists()
    reviews = sorted(p.name for p in (run_dir / "reviews").iterdir())
    assert "review_alpha_mom_r0.md" in reviews and "review_alpha_mom_r1.md" in reviews

    # attempts.csv: one row per backtest, verdicts filled in by the reviewer
    import pandas as pd
    attempts = pd.read_csv(run_dir / "attempts.csv")
    assert list(attempts["strategy"]) == ["alpha_mom", "alpha_mom", "beta_mom", "beta_mom"]
    assert list(attempts["n_trials"]) == [1, 2, 1, 2]
    assert list(attempts["verdict"]) == ["NEEDS_REVISION", "APPROVED", "NEEDS_REVISION", "NEEDS_REVISION"]
    assert (attempts["status"] == "ok").all()

    # Store survives across graph instances (JSON-backed for sqlite mode)
    store = make_store()
    manifest = store.get(("runs", "manifests"), "t_test_000000000000000000000000001").value
    assert manifest["strategies_approved"] == ["alpha_mom"] and manifest["strategies_skipped"] == ["beta_mom"]
    assert [x["name"] for x in manifest["strategies_excluded"]] == ["gamma_review"]
    assert len(store.search(("review", "technique_outcomes"))) == 2
    assert len(store.search(("review", "skipped_strategies"))) == 1

    # Resume continued from the checkpoint: research ran once (3 model calls), not again after the crash
    assert [k for k, _ in ScriptedModel.calls].count("research") == 3

    # Phases restarted cleanly: every review phase began at step 0 and needed exactly one nudge
    review_phases = {k for k, _ in ScriptedModel.calls if k.startswith("review:")}
    assert review_phases == {"review:alpha_mom:0", "review:alpha_mom:1", "review:beta_mom:0", "review:beta_mom:1"}

    # beta_mom's implementation was cut off, then stopped early: neither stop reached review (it was
    # nudged back until a backtest existed), so every review above saw a real attempt
    beta_nudges = [text for k, text in ScriptedModel.nudges if k == "impl:beta_mom"]
    assert len(beta_nudges) == 4, beta_nudges            # 2 per revision
    assert "cut off at the output token limit" in beta_nudges[0] and "cut off" in beta_nudges[2]
    assert "No backtest attempt exists" in beta_nudges[1] and "No backtest attempt exists" in beta_nudges[3]
    assert [k for k, _ in ScriptedModel.calls].count("impl:beta_mom") == 10   # (2 stops + write + run + ready) × 2
    assert [k for k, _ in ScriptedModel.calls].count("impl:alpha_mom") == 7   # +1: resumed after a mid-call cut-off
    alpha_nudges = [text for k, text in ScriptedModel.nudges if k == "impl:alpha_mom"]
    assert "cut off at the output token limit" in alpha_nudges[0]


def test_tool_guards(env):
    tmp_path, _ = env
    from src.tools.impl_tools import write_strategy_file
    from src.tools.review_tools import list_directory, read_file

    cfg = {"configurable": {"thread_id": "guards"}}
    assert read_file.invoke({"file_path": "/etc/passwd"}, cfg).startswith("ERROR")
    assert read_file.invoke({"file_path": ".env"}, cfg).startswith("ERROR")
    assert read_file.invoke({"file_path": "../../../../etc/hosts"}, cfg).startswith("ERROR")
    assert "ERROR" in write_strategy_file.invoke({"strategy_name": "../evil", "code": "x=1"}, cfg)
    assert "must define run_backtest" in write_strategy_file.invoke({"strategy_name": "ok", "code": "x=1"}, cfg)
    assert "signature" in write_strategy_file.invoke(
        {"strategy_name": "ok", "code": "def run_backtest(a, b):\n    pass\n"}, cfg)
    assert list_directory.invoke({"path": "/"}, cfg).startswith("ERROR")


def test_evaluator_contract_errors(env):
    tmp_path, _ = env
    from src.jobs.worker import run
    out = tmp_path / "job"
    result = run({
        "job_id": "b1", "thread_id": "x", "strategy_name": "broken", "revision": 0, "attempt": 1, "params": {},
        "strategy_path": str(ROOT / "tests" / "jobs" / "broken_strategy.py"),
        "output": {"kind": "local", "dir": str(out)}, "attempts_csv": str(tmp_path / "attempts.csv"),
    })
    assert result["status"] == "error" and "returns" in result["error"]
    assert (out / "result.json").exists()


def test_xai_provider_factory(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "xai")
    monkeypatch.setenv("LLM_MODEL_ID", "grok-4.7")
    from src.config import get_settings
    from src.llm import get_llm
    get_settings.cache_clear(); get_llm.cache_clear()
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="XAI_API_KEY"):
        get_llm()
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    get_llm.cache_clear()
    llm = get_llm()
    assert type(llm).__name__ == "ChatXAI" and llm.model_name == "grok-4.7"
    assert type(llm.bind_tools([])).__name__ == "RunnableBinding" or llm.bind_tools([]) is not None
    get_settings.cache_clear(); get_llm.cache_clear()


def test_deepseek_provider_factory_and_reasoning_replay(monkeypatch):
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import tool

    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("LLM_MODEL_ID", "deepseek-v4-pro")
    monkeypatch.setenv("LLM_EFFORT", "high")
    from src.config import get_settings
    from src.llm import get_llm
    get_settings.cache_clear(); get_llm.cache_clear()
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="DEEPSEEK_API_KEY"):
        get_llm()
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    get_llm.cache_clear()
    llm = get_llm()
    assert llm.model_name == "deepseek-v4-pro" and llm.reasoning_effort == "high" and llm.streaming

    @tool
    def ping(x: str) -> str:
        """Echo x."""
        return x

    history = [
        SystemMessage(content="sys"),
        HumanMessage(content="go"),
        AIMessage(content="", additional_kwargs={"reasoning_content": "think-1"},
                  tool_calls=[{"name": "ping", "args": {"x": "a"}, "id": "c1", "type": "tool_call"}]),
        ToolMessage(content="a", tool_call_id="c1"),
        AIMessage(content="done", additional_kwargs={"reasoning_content": "think-2"}),
        HumanMessage(content="continue"),
    ]
    bound = llm.bind_tools([ping])
    payload = bound.bound._get_request_payload(history, **bound.kwargs)  # no network call
    assistants = [m for m in payload["messages"] if m["role"] == "assistant"]
    assert [m["reasoning_content"] for m in assistants] == ["think-1", "think-2"]
    assert payload["reasoning_effort"] == "high" and payload["tools"]

    monkeypatch.setenv("LLM_EFFORT", "none")
    get_settings.cache_clear(); get_llm.cache_clear()
    assert get_llm().extra_body == {"thinking": {"type": "disabled"}}
    get_settings.cache_clear(); get_llm.cache_clear()


def test_json_file_store_parallel_puts_stay_valid(tmp_path):
    import json
    from concurrent.futures import ThreadPoolExecutor
    from src.local_store import JsonFileStore

    path = tmp_path / "s.store.json"
    store = JsonFileStore(path)
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(lambda i: store.put(("research", "extracted_signals"), f"k{i}", {"i": i, "pad": "x" * (i * 37)}),
                      range(64)))
    data = json.loads(path.read_text(encoding="utf-8"))          # must parse: no interleaved writes
    assert len(data) == 64 and not list(tmp_path.glob("*.tmp"))
    assert len(JsonFileStore(path).search(("research", "extracted_signals"), limit=100)) == 64

    # a file corrupted by the old non-atomic writer still loads (first complete snapshot)
    path.write_text(json.dumps(data[:3]) + '  {"namespace": ["x"], "key": "tail"', encoding="utf-8")
    assert len(JsonFileStore(path).search(("research", "extracted_signals"), limit=100)) == 3


def test_spec_parsing_and_validation():
    from src.tools.research_tools import parse_strategy_spec, spec_problems
    specs = parse_strategy_spec(SPEC)
    assert [(x["name"], x["implementable"]) for x in specs] == [
        ("alpha_mom", True), ("beta_mom", True), ("gamma_review", False)]
    assert specs[1]["model"] == "XGBoost classifier" and specs[0]["strategy_type"] == "cross_sectional_momentum"
    assert specs[2]["exclusion_reason"] == "review paper, no parameters or results"
    assert spec_problems(specs) == []

    bad = parse_strategy_spec("""## Strategy: No Flag
- model: x
## Strategy: Thin
- implementable: yes — looks fine
- model: rule-based
## Strategy: Bare No
- implementable: no
""")
    problems = spec_problems(bad)
    assert len(problems) == 3
    assert "missing or invalid '- implementable" in problems[0]
    assert "strategy_type" in problems[1] and "reference result" in problems[1] and "implementation notes" in problems[1]
    assert "give the reason" in problems[2]


def test_write_strategy_spec_rejects_then_excludes(env):
    from langchain_core.messages import ToolMessage
    from src.tools.research_tools import write_strategy_spec
    tmp_path, _ = env
    from src.workdir import run_paths
    run_paths("t_spec").research.mkdir(parents=True, exist_ok=True)
    cfg = {"configurable": {"thread_id": "t_spec", "run_id": "t_spec"}}

    def call(content):
        return write_strategy_spec.invoke({"type": "tool_call", "id": "c1", "name": "write_strategy_spec",
                                           "args": {"content": content}}, config=cfg)

    rejected = call("## Strategy: Thin\n- implementable: yes — ok\n")
    assert "strategy_specs" not in rejected.update
    assert rejected.update["messages"][0].content.startswith("ERROR: spec not saved")

    ok = call(SPEC)
    assert [x["name"] for x in ok.update["strategy_specs"]] == ["alpha_mom", "beta_mom"]
    assert [x["name"] for x in ok.update["excluded_strategies"]] == ["gamma_review"]
    assert ok.update["spec_written"] is True and "Excluded: gamma_review" in ok.update["messages"][0].content


def test_run_ends_after_research_when_nothing_is_implementable(env, monkeypatch):
    tmp_path, paper = env
    from src import runner
    only_review = """## Strategy: Gamma Review
- implementable: no — review paper, no parameters or results
- document_type: review
- strategy_type: other
- model: none
"""
    monkeypatch.setattr(sys.modules[__name__], "SPEC", only_review)
    runner.start_run([str(paper)], thread_id="t_test_000000000000000000000000009")
    status = json.loads((tmp_path / "runs" / "t_test_000000000000000000000000009" / "status.json").read_text())
    assert status["status"] == "complete" and status["strategies"] == [] and status["attempts"] == 0
    assert status["excluded"][0]["name"] == "gamma_review"
    assert not any(k.startswith(("impl:", "review:")) for k, _ in ScriptedModel.calls)
