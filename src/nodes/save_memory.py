from __future__ import annotations

from datetime import datetime, timezone

from langgraph.store.base import BaseStore

from src.state import PipelineState


def save_memory_node(state: PipelineState, store: BaseStore) -> dict:
    """Final graph node. Writes the run manifest to the Store. Individual overfitting patterns and
    technique outcomes are written by the review tools during the run."""
    findings = state.get("review_findings", []) or []
    approved = sorted({f["strategy_name"] for f in findings if f.get("verdict") == "APPROVED"})
    store.put(("runs", "manifests"), state["run_id"], {
        "run_id": state["run_id"],
        "paper_paths": state.get("paper_paths", []),
        "strategies_total": len(state.get("strategy_specs", []) or []),
        "strategies": [s.get("name") for s in state.get("strategy_specs", []) or []],
        "strategies_approved": approved,
        "strategies_skipped": state.get("skipped_strategies", []),
        "strategies_excluded": [{"name": x.get("name"), "reason": x.get("exclusion_reason")}
                                for x in state.get("excluded_strategies", []) or []],
        "backtest_attempts": len(state.get("backtest_results", []) or []),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return {}
