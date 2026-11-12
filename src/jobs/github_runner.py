"""Submit a backtest attempt as ONE commit on a new branch via the GitHub Git Data API.

Sequence (no git binary, no clone):
  GET  git/ref/heads/<base>        → base commit sha
  GET  git/commits/<sha>           → base tree sha
  POST git/trees  (base_tree + files)
  POST git/commits (tree, parent = base)
  POST git/refs   refs/heads/<branch> → the commit
The branch is created last and points at a commit that already holds every file, so the
`on: push` workflow fires exactly once per attempt.
"""
from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger(__name__)


class GitHubRunner:
    def __init__(self, repo: str, base_branch: str = "main", api_url: str = "https://api.github.com", token: str | None = None):
        if not repo or "/" not in repo:
            raise ValueError("GITHUB_REPO must be '<owner>/<name>'")
        self.repo = repo
        self.base_branch = base_branch
        self.api = api_url.rstrip("/")
        self.token = token or os.environ.get("GITHUB_TOKEN", "")
        if not self.token:
            raise EnvironmentError("GITHUB_TOKEN is not set (load it from Secrets Manager via GITHUB_TOKEN_SECRET_ARN)")
        self.s = requests.Session()
        self.s.headers.update({
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })

    def _url(self, path: str) -> str:
        return f"{self.api}/repos/{self.repo}/{path.lstrip('/')}"

    def _get(self, path: str) -> dict:
        r = self.s.get(self._url(path), timeout=30)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict) -> dict:
        r = self.s.post(self._url(path), json=body, timeout=60)
        r.raise_for_status()
        return r.json()

    def submit(self, branch: str, files: dict[str, str], message: str) -> str:
        """Create `branch` with one commit containing `files` ({repo path: text}). Returns the commit sha."""
        base_sha = self._get(f"git/ref/heads/{self.base_branch}")["object"]["sha"]
        base_tree = self._get(f"git/commits/{base_sha}")["tree"]["sha"]
        tree = self._post("git/trees", {
            "base_tree": base_tree,
            "tree": [{"path": p, "mode": "100644", "type": "blob", "content": text} for p, text in files.items()],
        })
        commit = self._post("git/commits", {"message": message, "tree": tree["sha"], "parents": [base_sha]})
        self._post("git/refs", {"ref": f"refs/heads/{branch}", "sha": commit["sha"]})
        log.info("pushed %d file(s) to %s@%s", len(files), branch, commit["sha"][:8])
        return commit["sha"]
