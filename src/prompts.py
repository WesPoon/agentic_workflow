"""Load and render the Markdown prompts in prompts/ (or PROMPTS_DIR).

Placeholders are `{{name}}`. Every placeholder in a file must be supplied, otherwise rendering
raises, so a typo in a prompt fails the run instead of reaching the model. Single braces are left
untouched, so code samples such as `params={...}` need no escaping. Trailing whitespace is trimmed.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


def prompts_dir() -> Path:
    from src.config import CODE_ROOT

    return Path(os.environ.get("PROMPTS_DIR") or CODE_ROOT / "prompts")


@lru_cache(maxsize=None)
def _load(path: str) -> str:
    return Path(path).read_text(encoding="utf-8").rstrip()


def placeholders(prompt_file: str) -> set[str]:
    return set(_PLACEHOLDER.findall(_load(str(prompts_dir() / prompt_file))))


def render(prompt_file: str, /, **values: object) -> str:
    """Render prompts/<prompt_file> (e.g. 'impl/system.md') with the given placeholder values."""
    path = prompts_dir() / prompt_file
    if not path.exists():
        raise FileNotFoundError(f"prompt file missing: {path}")
    text = _load(str(path))
    missing = set(_PLACEHOLDER.findall(text)) - values.keys()
    if missing:
        raise KeyError(f"{prompt_file} uses placeholder(s) the code does not supply: {sorted(missing)}")
    return _PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), text)


def clear_cache() -> None:
    _load.cache_clear()
