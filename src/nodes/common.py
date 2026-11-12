"""Shared mechanics for the three agent nodes.

Each agent works in a *phase* keyed "<agent>:<strategy_index>:<revision>". A phase is one model
conversation:

- On phase entry the message channel is cleared, the system prompt is built ONCE and frozen in
  state, and the conversation starts with a HumanMessage.
- Within a phase the history is append-only and the system prompt and tool list never change.
  Claude Opus 5.5 binds each thinking block to the exact prefix that produced it (system, tools,
  every earlier message); editing or trimming that prefix returns a 400. Append-only also keeps
  the prompt cache warm.
- If the model stops without finishing (last message is an AIMessage without tool calls), a user
  nudge is appended, so a request never ends on an assistant turn.

Context is bounded by MAX_PHASE_STEPS, not by trimming. Current Opus models have 1M-token context.
"""
from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from src.config import get_settings
from src.llm import get_llm
from src.prompts import render

log = logging.getLogger(__name__)


_TRUNCATED_STOPS = {"length", "max_tokens"}


def was_truncated(msg: BaseMessage) -> bool:
    """True when the provider cut the reply off at the output-token limit.

    OpenAI-compatible APIs (DeepSeek, xAI) report finish_reason="length"; the Anthropic API
    stop_reason="max_tokens"; Bedrock Converse stopReason="max_tokens"."""
    md = getattr(msg, "response_metadata", None) or {}
    return any(str(md.get(k, "")).lower() in _TRUNCATED_STOPS for k in ("finish_reason", "stop_reason", "stopReason"))


def phase_key(agent: str, state: dict) -> str:
    return f"{agent}:{state.get('current_strategy_index', 0)}:{state.get('revision_count', 0)}"


def run_agent(agent: str, state: dict, build_system_prompt, tools: list, kickoff: str, nudge: str) -> dict:
    # Every agent step follows a tool step: mirror new artifacts before the next checkpoint is written,
    # so a reaped microVM never leaves a checkpoint pointing at files that exist only on its disk.
    if state.get("run_id"):
        from src.storage import persist_run

        persist_run(state["run_id"])

    key = phase_key(agent, state)
    update: dict = {"phase": key}

    if state.get("phase") != key:
        system_prompt = build_system_prompt(state)
        first = HumanMessage(content=kickoff)
        history: list[BaseMessage] = [first]
        new_messages: list[BaseMessage] = [RemoveMessage(id=REMOVE_ALL_MESSAGES), first]
        steps = 0
        update["phase_system"] = system_prompt
    else:
        system_prompt = state.get("phase_system") or build_system_prompt(state)
        history = list(state.get("messages", []))
        new_messages = []
        steps = state.get("phase_steps", 0)
        if not history:
            first = HumanMessage(content=kickoff)
            history, new_messages = [first], [first]
        elif isinstance(history[-1], AIMessage) and not history[-1].tool_calls:
            last_ai = history[-1]
            # A reply cut off mid tool call leaves unparseable calls in invalid_tool_calls. Providers
            # still see them as tool calls and reject the next request unless each one has a result.
            for bad in last_ai.invalid_tool_calls or []:
                if bad.get("id"):
                    t = ToolMessage(content=f"ERROR: the arguments of {bad.get('name') or 'this tool call'} were "
                                            "incomplete or invalid JSON, so it was not run. Send the complete call again.",
                                    tool_call_id=bad["id"], status="error")
                    history.append(t)
                    new_messages.append(t)
            if was_truncated(last_ai):
                log.warning("%s reply was cut off at LLM_MAX_TOKENS before a complete tool call; asking it to resume", agent)
                n = HumanMessage(content=render("shared/truncated.md"))
            else:
                n = HumanMessage(content=nudge)
            history.append(n)
            new_messages.append(n)

    llm = get_llm().bind_tools(tools)
    response = llm.invoke([SystemMessage(content=system_prompt)] + history)
    new_messages.append(response)
    log.info("%s step %d (%s): %d tool call(s)", agent, steps + 1, key, len(getattr(response, "tool_calls", []) or []))
    update.update({"messages": new_messages, "phase_steps": steps + 1})
    return update


def over_budget(state: dict) -> bool:
    return state.get("phase_steps", 0) >= get_settings().max_phase_steps


def last_has_tool_calls(state: dict) -> bool:
    msgs = state.get("messages", [])
    return bool(msgs) and isinstance(msgs[-1], AIMessage) and bool(msgs[-1].tool_calls)
