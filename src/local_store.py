"""An InMemoryStore that snapshots itself to a JSON file after every write.

Used only for PERSISTENCE=sqlite so cross-run memory survives locally without any service.

ToolNode runs parallel tool calls in threads, so several puts can land at once: writes are
serialized with a lock and go to a temp file that atomically replaces the snapshot, so a reader
never sees a half-written or interleaved file.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

from langgraph.store.base import Op, PutOp, Result
from langgraph.store.memory import InMemoryStore

log = logging.getLogger(__name__)


class JsonFileStore(InMemoryStore):
    def __init__(self, path: Path):
        super().__init__()
        self._path = Path(path)
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            return
        text = self._path.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Snapshots written before the lock fix could be interleaved: keep the first complete array.
            data, end = json.JSONDecoder().raw_decode(text.lstrip())
            log.warning("store file %s had trailing data after char %d; loaded the first snapshot only",
                        self._path, end)
        for entry in data:
            super().put(tuple(entry["namespace"]), entry["key"], entry["value"])

    def _dump(self) -> None:
        with self._lock:
            self._dump_locked()

    def _dump_locked(self) -> None:
        items = []
        for ns_prefix in self.list_namespaces():
            for item in super().search(ns_prefix, limit=10_000):
                if tuple(item.namespace) == tuple(ns_prefix):
                    items.append({"namespace": list(item.namespace), "key": item.key, "value": item.value})
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(f"{self._path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(items, indent=2, default=str), encoding="utf-8")
        os.replace(tmp, self._path)  # atomic on POSIX: readers see the old or the new file, never a mix

    def batch(self, ops: list[Op]) -> list[Result]:
        with self._lock:
            results = super().batch(ops)
            if any(isinstance(op, PutOp) for op in ops):
                self._dump_locked()
        return results
