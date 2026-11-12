# Pipeline prompts

These files are exactly what the LangGraph agents send to the model. Edit them here, not in Python.
(`agents/*.md` are design notes from the Claude Code era and are not loaded.)

Placeholders look like `{{name}}` and are filled from the run state by `src/prompts.py`. Rendering
fails loudly if a file uses a placeholder the code does not supply, so a typo cannot reach the model.
Single braces such as `{...}` are left alone.

| File | Sent as | Placeholders |
|---|---|---|
| `research/system.md` | research agent system prompt | `paper_list`, `memory_block` |
| `research/kickoff.md` | first user message of the research phase | — |
| `research/nudge.md` | user message when the agent stops before `write_strategy_spec` | — |
| `impl/system.md` | implementation agent system prompt | `name`, `spec_block`, `revision_block`, `memory_block` |
| `impl/revision.md` | inserted as `revision_block` after a failed review | `revision_count`, `verdict`, `flags`, `notes` |
| `impl/kickoff.md` | first user message, first attempt | `name` |
| `impl/kickoff_revision.md` | first user message after a failed review | `name`, `revision_count` |
| `impl/nudge.md` | user message when the agent stops mid-task | — |
| `impl/nudge_no_backtest.md` | user message when the agent stops before any backtest exists for this revision (review cannot start) | — |
| `review/system.md` | review agent system prompt | `strategy_name`, `revision_count`, `spec_block`, `attempts_block`, `memory_block` |
| `review/kickoff.md` | first user message of a review | `name` |
| `review/nudge.md` | user message when the review ends without a verdict | — |
| `shared/memory.md` | `memory_block` in every system prompt | `patterns`, `approved`, `rejected` |
| `shared/truncated.md` | user message when a reply was cut off at `LLM_MAX_TOKENS` before any tool call (any agent) | — |

The spec format (`## Strategy:` blocks with `implementable`, `model`, `strategy_type`, `Reference result`,
`Data requirements`, `Implementation notes`) is defined in `research/system.md` and enforced by
`write_strategy_spec` (`src/tools/research_tools.py`): a spec missing these lines is rejected with the list
of problems, and only `implementable: yes` strategies reach the implementation agent.

The system prompt is rendered once at the start of each agent phase and frozen for that phase
(Claude Opus 5.5 binds thinking to it). Prompt edits take effect at the next phase or run.
Lists inside placeholders (papers, attempts, memory entries) are formatted in
`src/utils/prompt_utils.py`.
