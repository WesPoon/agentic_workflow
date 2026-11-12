from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedStore
from langgraph.store.base import BaseStore
from langgraph.types import Command

from src.state import StrategySpec
from src.tools.context import paths_from, thread_id_from

_STRATEGY_HEADING = re.compile(r"^##\s*Strategy:\s*(.+?)\s*$", re.MULTILINE)


def _field(block: str, name: str) -> str | None:
    """Value of a '- <name>: <value>' line in a spec block (case-insensitive), or None."""
    m = re.search(rf"^\s*[-*]?\s*\**{re.escape(name)}\**\s*:\s*(.*?)\s*$", block, re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else None


# Fields every implementable strategy must carry (see prompts/research/system.md).
REQUIRED_FOR_IMPLEMENTABLE = ("model", "strategy_type", "reference result", "implementation notes", "data requirements")


def _slug(text: str, limit: int = 60) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", text.strip().lower()).strip("_")
    return (s or "strategy")[:limit]


def parse_strategy_spec(content: str) -> list[StrategySpec]:
    """Split strategy_spec.md into one StrategySpec per '## Strategy: <name>' block.

    Returns every block, implementable or not; callers split on spec["implementable"]."""
    matches = list(_STRATEGY_HEADING.finditer(content))
    specs: list[StrategySpec] = []
    seen: set[str] = set()
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        block = content[m.start():end].strip()
        title = m.group(1).strip()
        name = _slug(title)
        base, n = name, 2
        while name in seen:
            name, n = f"{base}_{n}", n + 1
        seen.add(name)
        impl_raw = _field(block, "implementable") or ""
        verdict = re.match(r"(yes|no|true|false)\b\W*(.*)", impl_raw, re.IGNORECASE)
        implementable = bool(verdict) and verdict.group(1).lower() in ("yes", "true")
        specs.append({
            "name": name,
            "title": title,
            "model": _field(block, "model") or "",
            "strategy_type": (_field(block, "strategy_type") or "").lower(),
            "implementable": implementable,
            "exclusion_reason": "" if implementable else (verdict.group(2).strip() if verdict
                                                          else impl_raw or "no 'implementable:' line"),
            "block": block,
            "spec_hash": hashlib.sha256(block.encode("utf-8")).hexdigest()[:16],
        })
    return specs


def spec_problems(specs: list[StrategySpec]) -> list[str]:
    """Format violations that write_strategy_spec rejects, one line each."""
    problems = []
    for sp in specs:
        impl_raw = _field(sp["block"], "implementable")
        if impl_raw is None or not re.match(r"(yes|no|true|false)\b", impl_raw, re.IGNORECASE):
            problems.append(f"'{sp['title']}': missing or invalid '- implementable: yes | no — <reason>' line")
            continue
        if sp["implementable"]:
            missing = [f for f in REQUIRED_FOR_IMPLEMENTABLE if not _field(sp["block"], f)]
            if missing:
                problems.append(f"'{sp['title']}': implementable strategies need these lines: {', '.join(missing)}")
        elif not sp["exclusion_reason"].strip():
            problems.append(f"'{sp['title']}': give the reason after 'implementable: no —'")
    return problems


@tool
def read_pdf(file_path: str, config: RunnableConfig) -> str:
    """Read and extract text from one of the run's paper PDFs. Returns the full text content."""
    paths = paths_from(config)
    try:
        path = paths.resolve_inside(file_path)
    except PermissionError as e:
        return f"ERROR: {e}"
    if path.suffix.lower() != ".pdf":
        return "ERROR: read_pdf only reads .pdf files"
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            return "\n\n".join(p.extract_text() or "" for p in pdf.pages)
    except Exception:
        pass
    try:
        from pypdf import PdfReader

        return "\n\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    except Exception as e:
        return f"ERROR reading PDF: {e}"


@tool
def write_research_brief(
    title: str,
    content: str,
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Write a structured research brief to research/brief_<title>.md in the run directory."""
    path = paths_from(config).research / f"brief_{_slug(title)}.md"
    path.write_text(content, encoding="utf-8")
    return Command(update={
        "research_briefs": [str(path)],
        "messages": [ToolMessage(f"Brief written: {path}", tool_call_id=tool_call_id)],
    })


@tool
def write_strategy_spec(
    content: str,
    config: RunnableConfig,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Write the consolidated research/strategy_spec.md. Each strategy MUST start with a line
    '## Strategy: <name>' and carry '- implementable: yes | no — <reason>'. Only implementable
    strategies go on to implementation; the others stay in the file as a record.
    Calling this again replaces the previous spec."""
    def _reply(text: str, extra: dict | None = None) -> Command:
        return Command(update={**(extra or {}), "messages": [ToolMessage(text, tool_call_id=tool_call_id)]})

    specs = parse_strategy_spec(content)
    if not specs:
        return _reply("ERROR: no '## Strategy: <name>' headings found; spec not saved.")
    problems = spec_problems(specs)
    if problems:
        return _reply("ERROR: spec not saved. Fix these and call write_strategy_spec again with the full spec:\n- "
                      + "\n- ".join(problems))
    path = paths_from(config).research / "strategy_spec.md"
    path.write_text(content, encoding="utf-8")
    kept = [sp for sp in specs if sp["implementable"]]
    excluded = [sp for sp in specs if not sp["implementable"]]
    lines = [f"Spec written to {path}.",
             f"Implementable ({len(kept)}): {', '.join(sp['name'] for sp in kept) or 'none'}."]
    if excluded:
        lines.append("Excluded: " + "; ".join(f"{sp['name']} ({sp['exclusion_reason']})" for sp in excluded) + ".")
    if not kept:
        lines.append("No implementable strategies: the run will end after research.")
    return _reply(" ".join(lines), {"strategy_specs": kept, "excluded_strategies": excluded, "spec_written": True})


@tool
def query_past_signals(store: Annotated[BaseStore, InjectedStore()]) -> str:
    """Query the Store for signals extracted in past runs. Returns JSON list."""
    return json.dumps([i.value for i in store.search(("research", "extracted_signals"), limit=200)], indent=2, default=str)


@tool
def write_signal_to_memory(
    signal_name: str,
    paper_title: str,
    paper_year: int,
    quality_score: float,
    has_oos_validation: bool,
    data_snooping_risk: str,
    primary_params: dict,
    config: RunnableConfig,
    store: Annotated[BaseStore, InjectedStore()],
) -> str:
    """
    Persist an extracted signal to the Store so future runs can reference it.
    quality_score: 0.0–1.0 (1.0 = rigorous paper with OOS validation).
    data_snooping_risk: 'low' | 'medium' | 'high'.
    """
    store.put(("research", "extracted_signals"), _slug(signal_name, 50), {
        "signal_name": signal_name,
        "paper_title": paper_title,
        "paper_year": paper_year,
        "quality_score": quality_score,
        "has_oos_validation": has_oos_validation,
        "data_snooping_risk": data_snooping_risk,
        "primary_params": primary_params,
        "run_id": thread_id_from(config),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    return f"Signal '{signal_name}' saved to Store."


RESEARCH_TOOLS = [read_pdf, write_research_brief, write_strategy_spec, query_past_signals, write_signal_to_memory]
