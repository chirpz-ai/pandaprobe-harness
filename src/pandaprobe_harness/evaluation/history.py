"""Local, persistent per-(session, metric) score history with EWMA + gate state.

This store is the substrate for both local detectors:

* **trend** (``trends.py``) — each resolved turn appends one score and
  incrementally updates a fast and a slow EWMA in **O(1)**, so the detector never
  re-scans a window or makes a network call on the turn path.
* **trajectory** (``trajectory.py``) — the running peak and stall counter live in
  the same entry under a ``"gate"`` key, mutated through
  :meth:`ScoreHistoryStore.update_gate`.

Persisted as a single atomically-written JSON file under
``<harness_root>/state/`` so both survive process restarts.

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

__all__ = ["EwmaState", "GateFold", "GateState", "ScoreSample", "ScoreHistoryStore"]

#: A gate fold: given the current state, return the next one. Run under the store
#: lock so it always sees fresh state.
GateFold = Callable[["GateState"], "GateState"]

# Cap retained per-key samples so the file cannot grow without bound.
_MAX_SAMPLES = 500


@dataclass(frozen=True, slots=True)
class EwmaState:
    """Incremental exponentially-weighted moving averages for one series."""

    fast: float
    slow: float
    count: int


@dataclass(frozen=True, slots=True)
class GateState:
    """Trajectory-gate state for one series: the running peak and stall counter.

    Lives alongside the EWMA state in the same entry so the gate never needs a
    second file or a second lock.
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
            entry = self._load().setdefault(
                self._key(session_id, metric), {"series": [], "ewma": None}
            )
            state = self._append_locked(entry, value, run_id=run_id, ts=ts)
            self._persist()
            return state

    def _append_locked(
        self,
        entry: dict[str, Any],
        value: float,
        *,
        run_id: str | None,
        ts: str | None,
    ) -> EwmaState:
        """Fold one sample into ``entry`` in place. Caller holds the lock and persists."""

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
        return state

    def seed(
        self,
        session_id: str,
        metric: str,
        samples: Sequence[tuple[float, str, str | None]],
    ) -> None:
        """Bulk-insert backend samples ``(value, ts, run_id)`` for cold-start.

        Samples whose ``run_id`` is already present are skipped (idempotent
        hydration), the EWMA state is advanced for each new sample, and the
        file is persisted once.
        """

        with self._lock:
            data = self._load()
            key = self._key(session_id, metric)
            entry = data.setdefault(key, {"series": [], "ewma": None})
            series: list[dict[str, Any]] = entry["series"]
            known_runs = {s.get("run_id") for s in series if s.get("run_id")}

            inserted = False
            for value, ts, run_id in samples:
                if run_id and run_id in known_runs:
                    continue
                prev = entry.get("ewma")
                if prev is None:
                    state = EwmaState(fast=value, slow=value, count=1)
                else:
                    fast = self._alpha_fast * value + (1.0 - self._alpha_fast) * prev["fast"]
                    slow = self._alpha_slow * value + (1.0 - self._alpha_slow) * prev["slow"]
                    state = EwmaState(fast=fast, slow=slow, count=int(prev["count"]) + 1)
                entry["ewma"] = {"fast": state.fast, "slow": state.slow, "count": state.count}
                series.append(
                    {
                        "value": value,
                        "ts": ts or datetime.now(UTC).isoformat(),
                        "run_id": run_id,
                    }
                )
                if run_id:
                    known_runs.add(run_id)
                inserted = True
            if len(series) > _MAX_SAMPLES:
                del series[: len(series) - _MAX_SAMPLES]
            if inserted:
                self._persist()

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
                entry = data.setdefault(key, {"series": [], "ewma": None})
                self._append_locked(entry, value, run_id=None, ts=None)
                results[metric] = fold(GateState.from_json(entry.get("gate")))
                entry["gate"] = results[metric].to_json()
            if samples:
                self._persist()
        return results

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
