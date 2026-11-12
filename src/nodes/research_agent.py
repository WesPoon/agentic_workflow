from __future__ import annotations

from langgraph.prebuilt import ToolNode

from src.nodes.common import run_agent
from src.state import PipelineState
from src.tools.research_tools import RESEARCH_TOOLS
from src.prompts import render
from src.utils.prompt_utils import build_memory_context_block, format_paper_list


def _build_system_prompt(state: PipelineState) -> str:
    return render(
        "research/system.md",
        paper_list=format_paper_list(state.get("paper_paths", [])),
        memory_block=build_memory_context_block(state),
    )


def research_agent_node(state: PipelineState) -> dict:
    return run_agent(
        "research", state, _build_system_prompt, RESEARCH_TOOLS,
        kickoff=render("research/kickoff.md"),
        nudge=render("research/nudge.md"),
    )


research_tool_node = ToolNode(RESEARCH_TOOLS)
