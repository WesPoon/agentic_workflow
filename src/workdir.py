"""Per-run working directory layout. Nothing is written into the source tree any more."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.config import get_settings

_NAME_RE = re.compile(r"^[A-Za-z0-9_]{1,64}$")


def safe_name(name: str) -> str:
    """Validate an LLM-supplied identifier used in file names and branch names."""
    if not _NAME_RE.fullmatch(name or ""):
        raise ValueError(
            f"invalid name {name!r}: use 1-64 characters from [A-Za-z0-9_] only"
        )
    return name


@dataclass(frozen=True)
class RunPaths:
    thread_id: str
    root: Path

    @property
    def papers(self) -> Path:
        return self.root / "papers"

    @property
    def research(self) -> Path:
        return self.root / "research"

    @property
    def strategies(self) -> Path:
        return self.root / "strategies"

    @property
    def backtests(self) -> Path:
        return self.root / "backtests"

    @property
    def reviews(self) -> Path:
        return self.root / "reviews"

    @property
    def jobs(self) -> Path:
        return self.root / "jobs"

    @property
    def status_file(self) -> Path:
        return self.root / "status.json"

    @property
    def attempts_csv(self) -> Path:
        return self.root / "attempts.csv"

    def ensure(self) -> "RunPaths":
        for d in (self.papers, self.research, self.strategies, self.backtests, self.reviews, self.jobs):
            d.mkdir(parents=True, exist_ok=True)
        return self

    def resolve_inside(self, user_path: str) -> Path:
        """Resolve a tool-supplied path and refuse anything outside this run dir or the code root.

        Relative paths are resolved against the run dir first, then the code root.
        """
        settings = get_settings()
        candidates: list[Path] = []
        p = Path(user_path).expanduser()
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(self.root / p)
            candidates.append(settings.code_root / p)
        allowed_roots = (self.root.resolve(), settings.code_root.resolve())
        for cand in candidates:
            resolved = cand.resolve()
            if any(resolved == r or r in resolved.parents for r in allowed_roots) and resolved.exists():
                return resolved
        raise PermissionError(
            f"path {user_path!r} is outside the run directory ({self.root}) and the code root"
        )


def run_paths(thread_id: str) -> RunPaths:
    return RunPaths(thread_id=thread_id, root=get_settings().run_dir(thread_id))
