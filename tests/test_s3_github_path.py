"""COMPUTE_BACKEND=github + STORAGE=s3 against a mocked S3. GitHub is replaced by a fake that,
like the Actions workflow on the EC2, checks out the branch files and runs the worker."""
from __future__ import annotations

import json
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "momentum-test-bucket"


@pytest.fixture()
def s3_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("STORAGE", "s3")
    monkeypatch.setenv("ARTIFACT_BUCKET", BUCKET)
    monkeypatch.setenv("COMPUTE_BACKEND", "github")
    monkeypatch.setenv("GITHUB_REPO", "owner/repo")
    monkeypatch.setenv("GITHUB_TOKEN", "x")
    monkeypatch.setenv("JOB_POLL_S", "0")
    monkeypatch.setenv("BACKTEST_TIMEOUT_S", "60")
    monkeypatch.setenv("WORK_DIR", str(tmp_path / "work"))
    from src.config import get_settings
    get_settings.cache_clear()
    import src.storage as storage
    storage._UPLOADED.clear()
    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        yield tmp_path
    get_settings.cache_clear()


class FakeActions:
    """Stands in for GitHubRunner: 'pushing' a branch runs the worker like the workflow would."""
    pushed: list = []

    def __init__(self, *a, **k):
        pass

    def submit(self, branch: str, files: dict[str, str], message: str) -> str:
        FakeActions.pushed.append((branch, sorted(files), message))
        checkout = Path(FakeActions.checkout_root) / branch.replace("/", "_")
        for rel, text in files.items():
            (checkout / rel).parent.mkdir(parents=True, exist_ok=True)
            (checkout / rel).write_text(text)
        job_file = next(p for p in files if p.startswith("jobs/"))
        job = json.loads((checkout / job_file).read_text())
        job["commit_sha"], job["branch"] = "abc123def456", branch
        job["strategy_path"] = str(checkout / job["strategy_relpath"])   # the worker resolves relpath vs cwd
        from src.jobs.worker import run
        run(job)
        return "abc123def456"


def test_github_backend_roundtrip(s3_env, monkeypatch):
    tmp_path = s3_env
    import src.jobs.github_runner as gr
    FakeActions.pushed, FakeActions.checkout_root = [], str(tmp_path / "checkouts")
    monkeypatch.setattr(gr, "GitHubRunner", FakeActions)

    from src.jobs.runner import run_job
    from src.storage import persist_run, read_status, write_status
    from src.workdir import run_paths

    paths = run_paths("thr_s3").ensure()
    strat = paths.strategies / "alpha.py"
    strat.write_text((ROOT / "tests" / "jobs" / "synthetic_strategy.py").read_text())

    out = json.loads(run_job({"job_id": "j1", "thread_id": "thr_s3", "strategy_name": "alpha", "revision": 0,
                              "attempt": 1, "params": {"seed": 2}, "change_note": "first"}, strat, paths))
    assert out["status"] == "ok", out
    assert out["commit_sha"] == "abc123def456" and out["branch"] == "run/thr_s3/alpha-r0-j1"
    branch, files, message = FakeActions.pushed[0]
    assert files == ["jobs/j1.json", "strategies/alpha.py"] and message == "alpha r0 a1: first"

    s3 = boto3.client("s3", region_name="us-east-1")
    keys = {o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET)["Contents"]}
    for name in ("result.json", "metrics.csv", "subperiods.csv", "returns.csv", "tearsheet.html"):
        assert f"runs/thr_s3/jobs/j1/{name}" in keys
    assert "runs/thr_s3/attempts.csv" in keys
    assert paths.attempts_csv.exists()                       # synced back to the agent's run dir
    assert (paths.backtests / "alpha_a1_metrics.csv").exists()

    # status + persist + hydrate (crash recovery onto a fresh disk)
    write_status("thr_s3", "running", node="impl_tools")
    assert read_status("thr_s3")["node"] == "impl_tools"
    persist_run("thr_s3")
    keys = {o["Key"] for o in s3.list_objects_v2(Bucket=BUCKET)["Contents"]}
    assert "runs/thr_s3/strategies/alpha.py" in keys

    s3.put_object(Bucket=BUCKET, Key="papers/p.pdf", Body=b"%PDF-1.4")
    import shutil
    shutil.rmtree(paths.root)
    from src.storage import hydrate_run
    local = hydrate_run("thr_s3", ["papers/p.pdf"])
    assert Path(local[0]).exists() and (paths.strategies / "alpha.py").exists()


def test_github_runner_rest_calls(monkeypatch):
    """One commit, branch created last: exactly one push event per attempt (no network)."""
    from src.jobs.github_runner import GitHubRunner

    calls = []

    class Resp:
        def __init__(self, body):
            self._b = body

        def json(self):
            return self._b

        def raise_for_status(self):
            pass

    gh = GitHubRunner("owner/repo", "main", token="t")

    def get(url, **kw):
        path = url.replace("https://api.github.com/repos/owner/repo/", "")
        calls.append(("GET", path))
        return Resp({"object": {"sha": "base"}} if path.startswith("git/ref/") else {"tree": {"sha": "tree0"}})

    def post(url, json=None, **kw):
        path = url.replace("https://api.github.com/repos/owner/repo/", "")
        calls.append(("POST", path, json))
        return Resp({"sha": {"git/trees": "tree1", "git/commits": "c1"}.get(path, "")})

    monkeypatch.setattr(gh.s, "get", get)
    monkeypatch.setattr(gh.s, "post", post)
    sha = gh.submit("run/t/a-r0-1", {"strategies/a.py": "x", "jobs/1.json": "{}"}, "msg")
    assert sha == "c1"
    assert [c[:2] for c in calls] == [("GET", "git/ref/heads/main"), ("GET", "git/commits/base"),
                                      ("POST", "git/trees"), ("POST", "git/commits"), ("POST", "git/refs")]
    assert {e["path"] for e in calls[2][2]["tree"]} == {"strategies/a.py", "jobs/1.json"}
    assert calls[3][2]["parents"] == ["base"] and calls[4][2] == {"ref": "refs/heads/run/t/a-r0-1", "sha": "c1"}
