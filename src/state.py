from __future__ import annotations

import operator
from typing import Annotated, Literal, Optional

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class StrategySpec(TypedDict, total=False):
    name: str                 # safe identifier, used for file and branch names
    title: str                # heading as written in strategy_spec.md
    model: str                # free text from the spec's "model:" line, e.g. "XGBoost 10-class classifier"
    strategy_type: str        # cross_sectional_momentum | time_series_momentum | reversal | other
    implementable: bool
    exclusion_reason: str     # why implementable is "no" (empty otherwise)
    block: str                # the full markdown block for this strategy
    spec_hash: str


class BacktestResult(TypedDict, total=False):
    strategy_name: str
    revision: int
    attempt: int
    attempt_id: str
    commit_sha: str
    status: str
    oos_sharpe: Optional[float]
    deflated_sharpe_prob: Optional[float]


class ReviewFinding(TypedDict, total=False):
    strategy_name: str
    verdict: Literal["APPROVED", "NEEDS_REVISION"]
    attempt_id: str
    overfitting_flags: list[str]
    revision_notes: str
    iteration: int            # revision_count at the time of the review


class PipelineState(TypedDict, total=False):
    # Run metadata
    run_id: str
    paper_paths: list[str]

    # Outputs
    research_briefs: Annotated[list[str], operator.add]
    strategy_specs: list[StrategySpec]                     # implementable only; replaced on each write_strategy_spec
    excluded_strategies: list[StrategySpec]                # implementable: no, with exclusion_reason; replaced likewise
    spec_written: bool                                     # write_strategy_spec succeeded at least once
    backtest_results: Annotated[list[BacktestResult], operator.add]
    review_findings: Annotated[list[ReviewFinding], operator.add]
    skipped_strategies: Annotated[list[str], operator.add]

    # Conversation of the CURRENT phase only; cleared when a new phase starts
    messages: Annotated[list[AnyMessage], add_messages]

    # Control flow
    phase: str                 # "<agent>:<strategy_index>:<revision>"
    phase_steps: int           # LLM calls in the current phase
    phase_system: str          # system prompt frozen at phase start (thinking blocks are bound to it)
    current_strategy_index: int
    revision_count: int        # failed reviews of the current strategy so far

    # Loaded once at run start from the Store
    past_overfitting_patterns: list[str]
    past_approved_techniques: list[dict]
    past_rejected_techniques: list[dict]


def initial_state(thread_id: str, paper_paths: list[str], strategy_specs: list[StrategySpec] | None = None) -> PipelineState:
    return {
        "run_id": thread_id,
        "paper_paths": paper_paths,
        "research_briefs": [],
        "strategy_specs": strategy_specs or [],
        "excluded_strategies": [],
        "spec_written": bool(strategy_specs),
        "backtest_results": [],
        "review_findings": [],
        "skipped_strategies": [],
        "messages": [],
        "phase": "",
        "phase_steps": 0,
        "phase_system": "",
        "current_strategy_index": 0,
        "revision_count": 0,
        "past_overfitting_patterns": [],
        "past_approved_techniques": [],
        "past_rejected_techniques": [],
    }


def current_spec(state: dict) -> StrategySpec:
    idx = state.get("current_strategy_index", 0)
    specs = state.get("strategy_specs", []) or []
    return specs[idx] if idx < len(specs) else {}


def has_backtest_for_current(state: dict) -> bool:
    """True once run_backtest has produced an attempt for the current strategy at its current revision."""
    name = current_spec(state).get("name")
    rev = state.get("revision_count", 0)
    return any(r.get("strategy_name") == name and r.get("revision") == rev
               for r in state.get("backtest_results", []) or [])
