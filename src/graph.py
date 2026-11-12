from __future__ import annotations

import logging
from datetime import datetime, timezone

from langgraph.graph import END, START, StateGraph
from langgraph.store.base import BaseStore

from src.config import get_settings
from src.nodes.common import last_has_tool_calls, over_budget
from src.nodes.implementation_agent import impl_tool_node, implementation_agent_node
from src.nodes.load_memory import load_memory_node
from src.nodes.research_agent import research_agent_node, research_tool_node
from src.nodes.review_agent import review_agent_node, review_tool_node
from src.nodes.save_memory import save_memory_node
from src.state import PipelineState, current_spec, has_backtest_for_current

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_research(state: PipelineState) -> str:
    if last_has_tool_calls(state):
        return "research_tools"
    if state.get("strategy_specs"):
        return "implementation_agent"
    if state.get("spec_written"):
        log.warning("research found no implementable strategy (%d excluded); ending run",
                    len(state.get("excluded_strategies", []) or []))
        return "save_memory"
    if over_budget(state):
        log.warning("research produced no strategy spec within the step budget; ending run")
        return "save_memory"
    return "research_agent"          # nudged to finish write_strategy_spec


def route_impl(state: PipelineState) -> str:
    if last_has_tool_calls(state):
        return "impl_tools"
    if over_budget(state):
        return "skip_strategy_node"
    if not has_backtest_for_current(state):
        # The model stopped (or was cut off at the output limit) before run_backtest produced an
        # attempt: reviewing nothing would burn a revision, so send it back with a nudge instead.
        log.info("implementation stopped without a backtest for this revision; nudging it to continue")
        return "implementation_agent"
    return "review_agent"


def _current_finding(state: PipelineState) -> dict | None:
    name = current_spec(state).get("name")
    rev = state.get("revision_count", 0)
    return next(
        (f for f in reversed(state.get("review_findings", []) or [])
         if f.get("strategy_name") == name and f.get("iteration") == rev),
        None,
    )


def route_review(state: PipelineState) -> str:
    if last_has_tool_calls(state):
        return "review_tools"
    finding = _current_finding(state)
    if finding is None:
        return "skip_strategy_node" if over_budget(state) else "review_agent"
    if finding.get("verdict") == "APPROVED":
        return "approved_node"
    if state.get("revision_count", 0) + 1 >= get_settings().max_revisions:
        return "skip_strategy_node"
    return "needs_revision_node"


def route_next_strategy(state: PipelineState) -> str:
    # approved_node / skip_strategy_node have already advanced the index
    if state.get("current_strategy_index", 0) < len(state.get("strategy_specs", []) or []):
        return "implementation_agent"
    return "save_memory"


# ---------------------------------------------------------------------------
# State transition nodes
# ---------------------------------------------------------------------------

def needs_revision_node(state: PipelineState) -> dict:
    return {"revision_count": state.get("revision_count", 0) + 1}


def approved_node(state: PipelineState) -> dict:
    return {"current_strategy_index": state.get("current_strategy_index", 0) + 1, "revision_count": 0}


def skip_strategy_node(state: PipelineState, store: BaseStore) -> dict:
    """Give up on the current strategy (failed review MAX_REVISIONS times, or an agent ran out of
    steps) and move on. Recorded in state and in the Store so later runs can see it."""
    spec = current_spec(state)
    name = spec.get("name", "unknown")
    finding = _current_finding(state) or {}
    reason = ("failed review" if finding.get("verdict") == "NEEDS_REVISION" else "step budget exhausted")
    store.put(("review", "skipped_strategies"), f"{state.get('run_id')}_{name}", {
        "strategy_name": name,
        "run_id": state.get("run_id"),
        "revisions": state.get("revision_count", 0),
        "reason": reason,
        "last_revision_notes": finding.get("revision_notes", ""),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    log.warning("skipping strategy %s (%s)", name, reason)
    return {
        "skipped_strategies": [name],
        "current_strategy_index": state.get("current_strategy_index", 0) + 1,
        "revision_count": 0,
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(checkpointer, store):
    builder = StateGraph(PipelineState)

    builder.add_node("load_memory", load_memory_node)
    builder.add_node("research_agent", research_agent_node)
    builder.add_node("research_tools", research_tool_node)
    builder.add_node("implementation_agent", implementation_agent_node)
    builder.add_node("impl_tools", impl_tool_node)
    builder.add_node("review_agent", review_agent_node)
    builder.add_node("review_tools", review_tool_node)
    builder.add_node("needs_revision_node", needs_revision_node)
    builder.add_node("approved_node", approved_node)
    builder.add_node("skip_strategy_node", skip_strategy_node)
    builder.add_node("save_memory", save_memory_node)

    builder.add_edge(START, "load_memory")
    builder.add_conditional_edges(
        "load_memory",
        lambda s: "implementation_agent" if s.get("strategy_specs") else "research_agent",
        ["implementation_agent", "research_agent"],
    )

    builder.add_conditional_edges("research_agent", route_research,
                                  ["research_tools", "implementation_agent", "research_agent", "save_memory"])
    builder.add_edge("research_tools", "research_agent")

    builder.add_conditional_edges("implementation_agent", route_impl,
                                  ["impl_tools", "review_agent", "skip_strategy_node", "implementation_agent"])
    builder.add_edge("impl_tools", "implementation_agent")

    builder.add_conditional_edges("review_agent", route_review,
                                  ["review_tools", "review_agent", "approved_node",
                                   "needs_revision_node", "skip_strategy_node"])
    builder.add_edge("review_tools", "review_agent")

    builder.add_edge("needs_revision_node", "implementation_agent")
    builder.add_conditional_edges("approved_node", route_next_strategy, ["implementation_agent", "save_memory"])
    builder.add_conditional_edges("skip_strategy_node", route_next_strategy, ["implementation_agent", "save_memory"])
    builder.add_edge("save_memory", END)

    return builder.compile(checkpointer=checkpointer, store=store)
