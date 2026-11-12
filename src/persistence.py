"""Checkpointer and Store factories keyed on PERSISTENCE=memory|sqlite|dynamodb."""
from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from src.config import get_settings


def make_checkpointer() -> BaseCheckpointSaver:
    s = get_settings()
    if s.persistence == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()
    if s.persistence == "sqlite":
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        s.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(s.sqlite_path), check_same_thread=False)
        return SqliteSaver(conn)
    if s.persistence == "dynamodb":
        from langgraph_checkpoint_aws import DynamoDBSaver

        kwargs = dict(
            table_name=s.checkpoint_table,
            region_name=s.aws_region,
            ttl_seconds=s.checkpoint_ttl_days * 86400,
            enable_checkpoint_compression=True,
        )
        if s.artifact_bucket:
            kwargs["s3_offload_config"] = {"bucket_name": s.artifact_bucket}
        return DynamoDBSaver(**kwargs)
    raise ValueError(f"unknown PERSISTENCE {s.persistence!r}")


def make_store() -> BaseStore:
    s = get_settings()
    if s.persistence in ("memory", "sqlite"):
        # Local runs: an in-memory store, persisted to disk as JSON so `inspect` and the next
        # run can see it. Good enough for development; DynamoDB is the real thing.
        from src.local_store import JsonFileStore

        if s.persistence == "memory":
            from langgraph.store.memory import InMemoryStore

            return InMemoryStore()
        return JsonFileStore(s.sqlite_path.with_suffix(".store.json"))
    if s.persistence == "dynamodb":
        from src.ddb_store import SafeDynamoDBStore

        return SafeDynamoDBStore(table_name=s.store_table, region_name=s.aws_region)
    raise ValueError(f"unknown PERSISTENCE {s.persistence!r}")
