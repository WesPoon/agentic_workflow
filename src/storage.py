"""Artifact storage: the run directory on disk, mirrored to S3 when STORAGE=s3.

S3 layout (all under ARTIFACT_PREFIX if set):
  papers/<file>.pdf                      inputs, uploaded by the user once
  runs/<thread_id>/status.json           run status for pollers
  runs/<thread_id>/attempts.csv          one row per backtest attempt
  runs/<thread_id>/{research,strategies,backtests,reviews}/...
  runs/<thread_id>/jobs/<job_id>/...     worker outputs (result.json, metrics.csv, tearsheet)
"""
from __future__ import annotations

import json
import os
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.config import get_settings
from src.workdir import RunPaths, run_paths

log = logging.getLogger(__name__)

SYNC_DIRS = ("research", "strategies", "backtests", "reviews")
SKIP_PARTS = ("cache", "__pycache__")


def _s3():
    import boto3

    return boto3.client("s3", region_name=get_settings().aws_region)


def s3_key(*parts: str) -> str:
    prefix = get_settings().artifact_prefix
    key = "/".join(p.strip("/") for p in parts if p)
    return f"{prefix}/{key}" if prefix else key


def run_prefix(thread_id: str, *parts: str) -> str:
    return s3_key("runs", thread_id, *parts)


# ----------------------------------------------------------------------------- S3 helpers

def s3_exists(key: str) -> bool:
    import botocore

    try:
        _s3().head_object(Bucket=get_settings().artifact_bucket, Key=key)
        return True
    except botocore.exceptions.ClientError as e:  # 404 / 403
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def s3_download_prefix(prefix: str, dest: Path) -> list[Path]:
    bucket = get_settings().artifact_bucket
    paginator = _s3().get_paginator("list_objects_v2")
    out: list[Path] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix.rstrip("/") + "/"):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix.rstrip("/")) + 1 :]
            if not rel:
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            _s3().download_file(bucket, obj["Key"], str(target))
            out.append(target)
    return out


_UPLOADED: dict[str, tuple[float, int]] = {}   # key -> (mtime, size) already uploaded by this process


def s3_upload_dir(src: Path, prefix: str) -> int:
    """Upload new or changed files only (per-process cache), so it is cheap to call often."""
    bucket = get_settings().artifact_bucket
    n = 0
    for path in src.rglob("*"):
        if not path.is_file() or any(part in SKIP_PARTS for part in path.relative_to(src).parts):
            continue
        key = f"{prefix.rstrip('/')}/{path.relative_to(src).as_posix()}"
        st = path.stat()
        sig = (st.st_mtime, st.st_size)
        if _UPLOADED.get(key) == sig:
            continue
        _s3().upload_file(str(path), bucket, key)
        _UPLOADED[key] = sig
        n += 1
    return n


def s3_upload_file(path: Path, key: str) -> None:
    _s3().upload_file(str(path), get_settings().artifact_bucket, key)


def s3_download_file(key: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _s3().download_file(get_settings().artifact_bucket, key, str(dest))
    return dest


def s3_put_text(key: str, text: str) -> None:
    _s3().put_object(Bucket=get_settings().artifact_bucket, Key=key, Body=text.encode("utf-8"))


def s3_get_text(key: str) -> str | None:
    if not s3_exists(key):
        return None
    body = _s3().get_object(Bucket=get_settings().artifact_bucket, Key=key)["Body"].read()
    return body.decode("utf-8")


# ----------------------------------------------------------------------------- run lifecycle

def hydrate_run(thread_id: str, paper_keys: list[str]) -> list[str]:
    """Prepare the run dir. Returns local paper paths.

    STORAGE=local: `paper_keys` are local paths; they are used in place.
    STORAGE=s3:    `paper_keys` are S3 keys (e.g. papers/x.pdf); they and any previous artifacts
                   of this run are downloaded into the run dir (crash recovery / resume).
    """
    s = get_settings()
    paths = run_paths(thread_id).ensure()
    if s.storage == "local":
        sources = [Path(p).expanduser().resolve() for p in paper_keys]
        missing = [str(p) for p in sources if not p.exists()]
        if missing:
            raise FileNotFoundError(f"papers not found: {missing}")
        # copy into the run dir so tools only ever read inside it, as with S3
        return [str(copy_into(src, paths.papers)) for src in sources]

    local_papers: list[str] = []
    for key in paper_keys:
        full_key = key if (s.artifact_prefix and key.startswith(s.artifact_prefix + "/")) else s3_key(key)
        dest = paths.papers / Path(key).name
        s3_download_file(full_key, dest)
        local_papers.append(str(dest))
    restored = s3_download_prefix(run_prefix(thread_id), paths.root)
    if restored:
        log.info("restored %d files for run %s from S3", len(restored), thread_id)
    return local_papers


def persist_run(thread_id: str) -> None:
    """Mirror the run dir's artifact folders and attempts.csv to S3. No-op for local storage."""
    s = get_settings()
    if s.storage != "s3":
        return
    paths = run_paths(thread_id)
    total = 0
    for name in SYNC_DIRS:
        d = getattr(paths, name)
        if d.exists():
            total += s3_upload_dir(d, run_prefix(thread_id, name))
    if paths.attempts_csv.exists():
        s3_upload_file(paths.attempts_csv, run_prefix(thread_id, "attempts.csv"))
    log.info("persisted %d files for run %s", total, thread_id)


def write_status(thread_id: str, status: str, **detail) -> dict:
    record = {
        "thread_id": thread_id,
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        **detail,
    }
    paths = run_paths(thread_id)
    paths.root.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, indent=2, default=str)
    paths.status_file.write_text(text, encoding="utf-8")
    if get_settings().storage == "s3":
        s3_put_text(run_prefix(thread_id, "status.json"), text)
    return record


def read_status(thread_id: str) -> dict:
    if get_settings().storage == "s3":
        text = s3_get_text(run_prefix(thread_id, "status.json"))
        if text is None:
            return {"thread_id": thread_id, "status": "unknown"}
        return json.loads(text)
    f = run_paths(thread_id).status_file
    if not f.exists():
        return {"thread_id": thread_id, "status": "unknown"}
    return json.loads(f.read_text(encoding="utf-8"))


def copy_into(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / src.name
    shutil.copy2(src, target)
    return target


__all__ = [
    "RunPaths", "run_paths", "hydrate_run", "persist_run", "write_status", "read_status",
    "s3_key", "run_prefix", "s3_exists", "s3_get_text", "s3_put_text", "s3_download_prefix",
    "s3_download_file", "s3_upload_file", "s3_upload_dir", "copy_into",
]
