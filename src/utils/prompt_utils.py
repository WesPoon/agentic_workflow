from __future__ import annotations


def format_overfitting_patterns(patterns: list[str]) -> str:
    if not patterns:
        return "None detected yet — this is the first run."
    return "\n".join(f"  - {p}" for p in patterns)


def format_technique_outcomes(techniques: list[dict], verdict_filter: str) -> str:
    filtered = [t for t in techniques if t.get("verdict") == verdict_filter]
    if not filtered:
        return f"  No {verdict_filter.lower()} techniques recorded yet."
    lines = []
    for t in filtered:
        name = t.get("strategy_name", "unknown")
        tech = t.get("technique", "unknown")
        sharpe = t.get("oos_sharpe", "N/A")
        note = t.get("what_worked", t.get("revision_notes", ""))
        lines.append(f"  - [{tech}] {name} | OOS Sharpe: {sharpe} | {note}")
    return "\n".join(lines)


def build_memory_context_block(state: dict) -> str:
    """The 'Memory from past runs' block for every system prompt (wording: prompts/shared/memory.md)."""
    from src.prompts import render

    return render(
        "shared/memory.md",
        patterns=format_overfitting_patterns(state.get("past_overfitting_patterns", [])),
        approved=format_technique_outcomes(state.get("past_approved_techniques", []), "APPROVED"),
        rejected=format_technique_outcomes(state.get("past_rejected_techniques", []), "NEEDS_REVISION"),
    )


def format_paper_list(paths: list[str]) -> str:
    return "\n".join(f"  - {p}" for p in paths)
