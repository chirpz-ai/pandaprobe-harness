"""Local, persistent per-(session, metric) score history and gate state.

The trajectory gate's running peak and stall counter live beside each retained
score series under a ``"gate"`` key.

Persisted as a single atomically-written JSON file under
``<harness_root>/state/`` so state survives process restarts.

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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..config import HarnessConfig

__all__ = ["GateFold", "GateState", "ScoreSample", "ScoreHistoryStore"]

#: A gate fold: given the current state, return the next one. Run under the store
#: lock so it always sees fresh state.
GateFold = Callable[["GateState"], "GateState"]

# Cap retained per-key samples so the file cannot grow without bound.
_MAX_SAMPLES = 500


@dataclass(frozen=True, slots=True)
class GateState:
    """Trajectory-gate state for one series: the running peak and stall counter.

    Lives alongside the score series so the gate never needs a second file or
    a second lock.
    """

    peak: float | None = None
    turns_since_gain: int = 0

    def to_json(self) -> dict[str, Any]:
        return {"peak": self.peak, "turns_since_gain": self.turns_since_gain}

    @classmethod
    def from_json(cls, data: Any) -> GateState:
        if not isinstance(data, dict):
            return cls()
        peak = data.get("peak")
        turns = data.get("turns_since_gain")
        return cls(
            peak=float(peak) if isinstance(peak, (int, float)) else None,
            turns_since_gain=int(turns) if isinstance(turns, int) else 0,
        )


@dataclass(frozen=True, slots=True)
class ScoreSample:
    value: float
    ts: str
    run_id: str | None = None


class ScoreHistoryStore:
    """Persistent score series + gate state, keyed by ``session × metric``."""

    def __init__(self, config: HarnessConfig) -> None:
        self._config = config
        self._path = config.history_file
        self._data: dict[str, dict[str, Any]] | None = None
        self._lock = threading.Lock()

    # -- persistence ----------------------------------------------------------

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._data is not None:
            return self._data
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._data = {}
            if isinstance(raw, dict):
                for key, value in raw.items():
                    if not isinstance(value, dict):
                        continue
                    series = value.get("series")
                    entry: dict[str, Any] = {
                        "series": series if isinstance(series, list) else []
                    }
                    if isinstance(value.get("gate"), dict):
                        entry["gate"] = value["gate"]
                    self._data[str(key)] = entry
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

    def _append_locked(
        self,
        entry: dict[str, Any],
        value: float,
        *,
        run_id: str | None,
        ts: str | None,
    ) -> None:
        """Append one sample in place. Caller holds the lock and persists."""

        series: list[dict[str, Any]] = entry["series"]
        series.append(
            {"value": value, "ts": ts or datetime.now(UTC).isoformat(), "run_id": run_id}
        )
        if len(series) > _MAX_SAMPLES:
            del series[: len(series) - _MAX_SAMPLES]

    def record_gated(
        self,
        session_id: str,
        samples: Sequence[tuple[str, float, GateFold]],
    ) -> dict[str, GateState]:
        """Append samples and fold their gate state in **one** locked write.

        Each item is ``(metric, value, fold)``, where ``fold`` receives the *fresh*
        gate state read under the store lock — so two concurrent sessions folding
        their own trajectories cannot clobber each other's peak.

        Batching matters: the store persists by re-serializing the whole file, so
        the naive shape (append, then separately fold) costs two whole-file writes
        per metric — four per trace at Tier 1. This does one, for a whole trace.
        """

        results: dict[str, GateState] = {}
        with self._lock:
            data = self._load()
            for metric, value, fold in samples:
                key = self._key(session_id, metric)
                entry = data.setdefault(key, {"series": []})
                self._append_locked(entry, value, run_id=None, ts=None)
                results[metric] = fold(GateState.from_json(entry.get("gate")))
                entry["gate"] = results[metric].to_json()
            if samples:
                self._persist()
        return results

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
