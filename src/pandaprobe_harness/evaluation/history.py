"""Local, persistent per-(session, metric) score history with EWMA state.

This store is the substrate for low-latency trend detection (see ``trends.py``):
each resolved turn appends one score and incrementally updates a fast and a slow
EWMA in **O(1)**, so the detector never needs to re-scan a window or make a
network call on the turn path. Persisted as a single atomically-written JSON
file under ``<harness_root>/state/`` so trend state survives process restarts.

All methods are synchronous blocking I/O; async callers wrap them in
``asyncio.to_thread`` (the hook does). Because ``asyncio.to_thread`` uses a
multi-worker thread pool (and one store instance is shared across sessions), the
in-memory cache and the on-disk file are guarded by a ``threading.Lock`` and
each persist writes to a unique temp file before ``os.replace`` — so concurrent
updates for different sessions cannot corrupt state or collide on the temp path.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..config import HarnessConfig

__all__ = ["EwmaState", "ScoreSample", "ScoreHistoryStore"]

# Cap retained per-key samples so the file cannot grow without bound.
_MAX_SAMPLES = 500


@dataclass(frozen=True, slots=True)
class EwmaState:
    """Incremental exponentially-weighted moving averages for one series."""

    fast: float
    slow: float
    count: int


@dataclass(frozen=True, slots=True)
class ScoreSample:
    value: float
    ts: str
    run_id: str | None = None


def _alpha(span: int) -> float:
    return 2.0 / (max(1, span) + 1.0)


class ScoreHistoryStore:
    """Persistent score series + EWMA state, keyed by ``session × metric``."""

    def __init__(self, config: HarnessConfig) -> None:
        self._config = config
        self._path = config.history_file
        self._alpha_fast = _alpha(config.ewma_fast_span)
        self._alpha_slow = _alpha(config.ewma_slow_span)
        self._data: dict[str, dict[str, Any]] | None = None
        self._lock = threading.Lock()

    # -- persistence ----------------------------------------------------------

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._data is not None:
            return self._data
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._data = raw if isinstance(raw, dict) else {}
        except (FileNotFoundError, ValueError):
            self._data = {}
        return self._data

    def _persist(self) -> None:
        data = self._load()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Unique temp name so concurrent persists never collide on one path.
        tmp = self._path.with_name(f"{self._path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self._path)

    @staticmethod
    def _key(session_id: str, metric: str) -> str:
        return f"{session_id}::{metric}"

    # -- API ------------------------------------------------------------------

    def record(
        self,
        session_id: str,
        metric: str,
        value: float,
        *,
        run_id: str | None = None,
        ts: str | None = None,
    ) -> EwmaState:
        """Append a score and incrementally update its EWMA state (O(1))."""

        with self._lock:
            data = self._load()
            key = self._key(session_id, metric)
            entry = data.setdefault(key, {"series": [], "ewma": None})

            prev = entry.get("ewma")
            if prev is None:
                state = EwmaState(fast=value, slow=value, count=1)
            else:
                fast = self._alpha_fast * value + (1.0 - self._alpha_fast) * prev["fast"]
                slow = self._alpha_slow * value + (1.0 - self._alpha_slow) * prev["slow"]
                state = EwmaState(fast=fast, slow=slow, count=int(prev["count"]) + 1)

            entry["ewma"] = {"fast": state.fast, "slow": state.slow, "count": state.count}
            series: list[dict[str, Any]] = entry["series"]
            series.append(
                {"value": value, "ts": ts or datetime.now(UTC).isoformat(), "run_id": run_id}
            )
            if len(series) > _MAX_SAMPLES:
                del series[: len(series) - _MAX_SAMPLES]

            self._persist()
            return state

    def ewma(self, session_id: str, metric: str) -> EwmaState | None:
        with self._lock:
            entry = self._load().get(self._key(session_id, metric))
            if not entry or entry.get("ewma") is None:
                return None
            e = entry["ewma"]
            return EwmaState(fast=e["fast"], slow=e["slow"], count=int(e["count"]))

    def series(self, session_id: str, metric: str) -> list[ScoreSample]:
        with self._lock:
            entry = self._load().get(self._key(session_id, metric))
            if not entry:
                return []
            return [
                ScoreSample(value=s["value"], ts=s["ts"], run_id=s.get("run_id"))
                for s in entry["series"]
            ]

    def values(self, session_id: str, metric: str) -> list[float]:
        return [s.value for s in self.series(session_id, metric)]
