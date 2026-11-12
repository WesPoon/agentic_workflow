from __future__ import annotations

from langgraph.store.base import BaseStore

from src.state import PipelineState


def load_memory_node(state: PipelineState, store: BaseStore) -> dict:
    """
    Graph entry node. Queries the cross-thread Store and populates past_* fields.
    These are injected read-only into all downstream agent system prompts.
    """
    overfitting_items = store.search(("review", "overfitting_patterns"))
    past_patterns = [item.value.get("pattern", "") for item in overfitting_items]

    outcome_items = store.search(("review", "technique_outcomes"))
    approved = [i.value for i in outcome_items if i.value.get("verdict") == "APPROVED"]
    rejected = [i.value for i in outcome_items if i.value.get("verdict") != "APPROVED"]

    return {
        "past_overfitting_patterns": past_patterns,
        "past_approved_techniques": approved,
        "past_rejected_techniques": rejected,
    }
