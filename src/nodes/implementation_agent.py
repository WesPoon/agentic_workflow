from __future__ import annotations

from langgraph.prebuilt import ToolNode

from src.config import get_settings
from src.nodes.common import run_agent
from src.state import PipelineState, current_spec, has_backtest_for_current
from src.tools.impl_tools import IMPL_TOOLS
from src.prompts import render
from src.utils.prompt_utils import build_memory_context_block


def _last_review(state: PipelineState, name: str) -> dict | None:
    return next((f for f in reversed(state.get("review_findings", []) or []) if f.get("strategy_name") == name), None)


def _build_system_prompt(state: PipelineState) -> str:
    spec = current_spec(state)
    name = spec.get("name", "unknown")
    revision_count = state.get("revision_count", 0)
    last_review = _last_review(state, name)
    revision_block = ""
    if last_review and revision_count > 0:
        revision_block = render(
            "impl/revision.md",
            revision_count=revision_count,
            verdict=last_review.get("verdict"),
            flags=last_review.get("overfitting_flags", []),
            notes=last_review.get("revision_notes", ""),
        )
    return render(
        "impl/system.md",
        backtest_timeout_min=max(1, get_settings().backtest_timeout_s // 60),
        name=name,
        spec_block=spec.get("block", ""),
        revision_block=revision_block,
        memory_block=build_memory_context_block(state),
    )


def implementation_agent_node(state: PipelineState) -> dict:
    name = current_spec(state).get("name", "unknown")
    rev = state.get("revision_count", 0)
    kickoff = (render("impl/kickoff.md", name=name) if rev == 0
               else render("impl/kickoff_revision.md", name=name, revision_count=rev))
    nudge = render("impl/nudge.md") if has_backtest_for_current(state) else render("impl/nudge_no_backtest.md")
    return run_agent("impl", state, _build_system_prompt, IMPL_TOOLS, kickoff=kickoff, nudge=nudge)


impl_tool_node = ToolNode(IMPL_TOOLS)
