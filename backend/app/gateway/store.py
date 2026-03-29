"""In-memory thread store for LangGraph Platform API compatibility."""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Any


@dataclass
class ThreadRecord:
    thread_id: str
    created_at: str
    updated_at: str
    metadata: dict[str, Any]
    status: str  # "idle", "busy", "interrupted", "error"
    values: dict[str, Any]
    _active_runs: int = 0

    def to_dict(self, select: list[str] | None = None) -> dict[str, Any]:
        result = {
            "thread_id": self.thread_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
            "status": self.status,
            "values": dict(self.values),
        }
        if select:
            # thread_id is always included
            return {k: v for k, v in result.items() if k in select or k == "thread_id"}
        return result


_DEFAULT_MAX_THREADS = 10_000


class ThreadStore:
    """Thread-safe in-memory store for thread metadata.

    Evicts the oldest (by updated_at) threads when capacity is reached.
    """

    def __init__(self, max_threads: int = _DEFAULT_MAX_THREADS):
        self._threads: dict[str, ThreadRecord] = {}
        self._lock = Lock()
        self._max_threads = max_threads

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()

    def _evict_if_needed(self) -> None:
        """Evict oldest idle threads when at capacity. Must hold self._lock."""
        while len(self._threads) >= self._max_threads:
            # Find oldest idle thread by updated_at
            oldest_id = None
            oldest_time = None
            for tid, rec in self._threads.items():
                if rec.status == "busy":
                    continue
                if oldest_time is None or rec.updated_at < oldest_time:
                    oldest_id = tid
                    oldest_time = rec.updated_at
            if oldest_id is None:
                break  # all threads are busy, can't evict
            del self._threads[oldest_id]

    def create(self, thread_id: str | None = None, metadata: dict | None = None) -> ThreadRecord:
        with self._lock:
            self._evict_if_needed()
            tid = thread_id or str(uuid.uuid4())
            now = self._now()
            record = ThreadRecord(
                thread_id=tid,
                created_at=now,
                updated_at=now,
                metadata=metadata or {},
                status="idle",
                values={},
            )
            self._threads[tid] = record
            return record

    def get(self, thread_id: str) -> ThreadRecord | None:
        return self._threads.get(thread_id)

    def get_or_create(self, thread_id: str) -> ThreadRecord:
        with self._lock:
            record = self._threads.get(thread_id)
            if record is not None:
                return record
            self._evict_if_needed()
            now = self._now()
            record = ThreadRecord(
                thread_id=thread_id,
                created_at=now,
                updated_at=now,
                metadata={},
                status="idle",
                values={},
            )
            self._threads[thread_id] = record
            return record

    def search(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        sort_by: str = "updated_at",
        sort_order: str = "desc",
        select: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            threads = list(self._threads.values())

        reverse = sort_order == "desc"
        threads.sort(key=lambda t: getattr(t, sort_by, ""), reverse=reverse)
        threads = threads[offset : offset + limit]
        return [t.to_dict(select=select) for t in threads]

    def delete(self, thread_id: str) -> bool:
        with self._lock:
            return self._threads.pop(thread_id, None) is not None

    def update_values(self, thread_id: str, values: dict[str, Any]) -> None:
        with self._lock:
            record = self._threads.get(thread_id)
            if record:
                record.values = {**record.values, **values}
                record.updated_at = self._now()

    def set_busy(self, thread_id: str) -> None:
        """Increment active run count and mark thread as busy."""
        with self._lock:
            record = self._threads.get(thread_id)
            if record:
                record._active_runs = getattr(record, "_active_runs", 0) + 1
                record.status = "busy"

    def set_idle(self, thread_id: str) -> None:
        """Decrement active run count; only mark idle when no runs remain."""
        with self._lock:
            record = self._threads.get(thread_id)
            if record:
                count = max(0, getattr(record, "_active_runs", 1) - 1)
                record._active_runs = count
                if count == 0:
                    record.status = "idle"
