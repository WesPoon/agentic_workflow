"""Prompt files and the code that fills them must agree."""
from __future__ import annotations

import pytest

from src.prompts import placeholders, prompts_dir, render

# What the code supplies for each prompt file. Update together with the .md files.
SUPPLIED = {
    "research/system.md": {"paper_list", "memory_block"},
    "research/kickoff.md": set(),
    "research/nudge.md": set(),
    "impl/system.md": {"name", "spec_block", "revision_block", "memory_block", "backtest_timeout_min"},
    "impl/revision.md": {"revision_count", "verdict", "flags", "notes"},
    "impl/kickoff.md": {"name"},
    "impl/kickoff_revision.md": {"name", "revision_count"},
    "impl/nudge.md": set(),
    "impl/nudge_no_backtest.md": set(),
    "review/system.md": {"strategy_name", "revision_count", "spec_block", "attempts_block", "memory_block"},
    "review/kickoff.md": {"name"},
    "review/nudge.md": set(),
    "shared/memory.md": {"patterns", "approved", "rejected"},
    "shared/truncated.md": set(),
}


def test_every_prompt_file_is_known():
    files = {p.relative_to(prompts_dir()).as_posix() for p in prompts_dir().rglob("*.md") if p.name != "README.md"}
    assert files == set(SUPPLIED), f"unregistered or missing prompt files: {files ^ set(SUPPLIED)}"


@pytest.mark.parametrize("prompt_file,supplied", sorted(SUPPLIED.items()))
def test_placeholders_match_code(prompt_file, supplied):
    assert placeholders(prompt_file) <= supplied, f"{prompt_file} uses placeholders the code never fills"
    text = render(prompt_file, **{k: f"<{k}>" for k in supplied})
    assert "{{" not in text and text.strip()


def test_missing_placeholder_fails_loudly():
    with pytest.raises(KeyError, match="paper_list"):
        render("research/system.md", memory_block="x")


def test_single_braces_untouched():
    assert "params={...}" in render("impl/system.md", name="n", spec_block="", revision_block="", memory_block="", backtest_timeout_min=30)


def test_prompts_dir_override(tmp_path, monkeypatch):
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / "kickoff.md").write_text("custom kickoff\n")
    monkeypatch.setenv("PROMPTS_DIR", str(tmp_path))
    from src.prompts import clear_cache
    clear_cache()
    assert render("research/kickoff.md") == "custom kickoff"
    clear_cache()
