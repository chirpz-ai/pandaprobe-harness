"""The append-only diagnostic journal — the harness's cross-run memory.

Every notable event (notice posted, notice acknowledged, rule added/retired,
reflection run, health transition, recovery) is one JSON line in
``<harness_root>/journal.jsonl``. The agent mines it through
``harness_journal``/``harness_reflect`` to spot recurring failure patterns
across runs ("boosting on past mistakes") and to judge rule effectiveness.

All methods are synchronous blocking I/O; async callers wrap them in
``asyncio.to_thread``.
"""

from __future__ import annotations

import threading
from collections.abc import Collection, Mapping
from datetime import UTC, datetime
from typing import Any

from ..config import HarnessConfig
from ._io import append_jsonl, read_jsonl

__all__ = ["Journal"]


class Journal:
    """Append-only JSONL event log under the harness workspace."""

    def __init__(self, config: HarnessConfig) -> None:
        self._path = config.journal_file
        self._lock = threading.Lock()

    def record(self, event: Mapping[str, Any]) -> dict[str, Any]:
        """Append one event, defaulting ``ts`` and requiring ``type``."""

        stored = dict(event)
        stored.setdefault("ts", datetime.now(UTC).isoformat())
        stored.setdefault("type", "unknown")
        with self._lock:
            append_jsonl(self._path, stored)
        return stored

    def recent(
        self, limit: int = 50, *, types: Collection[str] | None = None
    ) -> list[dict[str, Any]]:
        """The last ``limit`` events (chronological), optionally filtered by type."""

        events = read_jsonl(self._path)
        if types is not None:
            wanted = set(types)
            events = [e for e in events if e.get("type") in wanted]
        if limit > 0:
            events = events[-limit:]
        return events

    def notices_for(self, metric: str) -> list[dict[str, Any]]:
        """All ``notice`` events that involved ``metric``, chronological."""

        matches: list[dict[str, Any]] = []
        for event in read_jsonl(self._path):
            if event.get("type") != "notice":
                continue
            metrics = event.get("metrics")
            if not isinstance(metrics, list):
                continue
            names = {m.get("name") for m in metrics if isinstance(m, dict)}
            if metric in names:
                matches.append(event)
        return matches
