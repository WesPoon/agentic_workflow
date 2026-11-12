from __future__ import annotations

from langgraph.prebuilt import ToolNode

from src.nodes.common import run_agent
from src.state import PipelineState, current_spec
from src.tools.review_tools import REVIEW_TOOLS
from src.prompts import render
from src.utils.prompt_utils import build_memory_context_block


def _attempts_block(state: PipelineState, name: str) -> str:
    rows = [r for r in state.get("backtest_results", []) or [] if r.get("strategy_name") == name]
    if not rows:
        return "No backtest attempts were recorded for this strategy."
    lines = [f"  - attempt_id={r.get('attempt_id')} revision={r.get('revision')} attempt={r.get('attempt')} "
             f"status={r.get('status')} oos_sharpe={r.get('oos_sharpe')} "
             f"deflated_sharpe_prob={r.get('deflated_sharpe_prob')} commit={str(r.get('commit_sha', ''))[:8]}"
             for r in rows[-10:]]
    return "\n".join(lines)


def _build_system_prompt(state: PipelineState) -> str:
    spec = current_spec(state)
    strategy_name = spec.get("name", "unknown")
    return render(
        "review/system.md",
        strategy_name=strategy_name,
        revision_count=state.get("revision_count", 0),
        spec_block=spec.get("block", ""),
        attempts_block=_attempts_block(state, strategy_name),
        memory_block=build_memory_context_block(state),
    )


def review_agent_node(state: PipelineState) -> dict:
    name = current_spec(state).get("name", "unknown")
    return run_agent(
        "review", state, _build_system_prompt, REVIEW_TOOLS,
        kickoff=render("review/kickoff.md", name=name),
        nudge=render("review/nudge.md"),
    )


review_tool_node = ToolNode(REVIEW_TOOLS)
