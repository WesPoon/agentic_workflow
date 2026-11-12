"""Per-run local git history: one commit per backtest attempt (COMPUTE_BACKEND=local).

The run directory (runs/<thread_id>/) becomes its own git repository, so `git log` / `git diff`
there show exactly what changed between attempts, like the per-branch history on GitHub.
Failures never block a backtest: without git, the commit sha is just empty.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)
_IDENTITY = ["-c", "user.name=momentum-pipeline", "-c", "user.email=pipeline@localhost"]


def _git(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *_IDENTITY, *args], cwd=str(root), capture_output=True, text=True, timeout=60)


def snapshot(root: Path, files: list[Path], message: str) -> str:
    """Commit `files` (inside `root`) and return the commit sha, or "" if git is unavailable."""
    try:
        if not (root / ".git").exists():
            if _git(root, "init", "-q").returncode != 0:
                return ""
        rel = [str(Path(f).resolve().relative_to(root.resolve())) for f in files if Path(f).exists()]
        if not rel:
            return ""
        _git(root, "add", "--", *rel)
        commit = _git(root, "commit", "-q", "--allow-empty", "-m", message)
        if commit.returncode != 0:
            log.warning("git commit failed: %s", commit.stderr.strip()[:200])
            return ""
        return _git(root, "rev-parse", "HEAD").stdout.strip()
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        log.warning("git snapshot skipped: %s", e)
        return ""


def diff(root: Path, old: str, new: str, path: str) -> str:
    """Unified diff of one file between two commits ("" if unavailable)."""
    try:
        return _git(root, "diff", f"{old}..{new}", "--", path).stdout
    except (OSError, subprocess.SubprocessError):
        return ""
