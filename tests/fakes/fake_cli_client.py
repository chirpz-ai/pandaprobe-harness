"""In-process fake implementing the ``CliClient`` Protocol.

Primary mock seam for the fast test suite — no subprocess, no network. Models
the real CLI's **session-scoped** evaluation surface:

* ``evals runs batch --target session --session-ids <id> --metrics <m1,m2>``
  hands out a ``run_id`` and remembers the metric set for that run;
* ``evals runs scores <run_id> --target session`` returns a bare list of
  ``SessionScoreResponse``-shaped dicts (value as a string, status ``SUCCESS``),
  with optional ``running_polls`` PENDING rounds to exercise the poll loop;
* per-metric scores can be **flipped between turns** (low → high) to drive the
  self-healing / trend scenarios;
* ``error_on_prefix`` raises typed ``CliError``s to exercise degrade paths;
* ``evals scores list`` / ``evals scores get`` / ``traces get`` are stubbed for
  history cold-start, agent diagnosis, and dump enrichment.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pandaprobe_harness.cli.client import CliResult
from pandaprobe_harness.cli.errors import CliAuthError, CliError, CliGeneralError


@dataclass
class _Run:
    metrics: list[str]
    poll_count: int = 0
    session_id: str | None = None


@dataclass
class FakeCliClient:
    """A scripted, stateful fake of the ``pandaprobe`` CLI (session-scoped)."""

    running_polls: int = 0
    # metric name -> score in [0,1]. Mutate between turns to drive self-heal/trend.
    metric_values: dict[str, float] = field(
        default_factory=lambda: {"agent_reliability": 0.9, "agent_consistency": 0.9}
    )
    # session id -> {metric: score} overriding `metric_values` for that session
    # (lets a *replayed* session score differently from the live one).
    session_metric_values: dict[str, dict[str, float]] = field(default_factory=dict)
    metric_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Per-metric terminal status override (e.g. "FAILED" → null value).
    metric_status: dict[str, str] = field(default_factory=dict)
    # Per-metric verbatim value string (e.g. "N/A"/"") to test non-numeric parsing.
    raw_metric_values: dict[str, str] = field(default_factory=dict)
    # argv-prefix tuple -> exception to raise (degrade-path testing).
    error_on_prefix: dict[tuple[str, ...], CliError] = field(default_factory=dict)
    # Empty-then-populated: emit empty score lists for this many batch runs first
    # (simulates trace-ingestion lag → evaluator retry/backoff).
    empty_runs: int = 0
    # Optional canned payload for `evals scores get`.
    scores_get_payload: dict[str, Any] | None = None
    # Optional canned series for `evals scores list` (history cold-start).
    scores_list_payload: list[dict[str, Any]] | None = None
    # Per-session canned series for `evals scores list --session-id <id>`
    # (backend trend hydration / harness_history).
    session_scores_list: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    # Optional canned payload for `traces spans`.
    traces_spans_payload: dict[str, Any] | None = None
    # Health-check knobs: `version` / `auth status` raise when flipped off.
    version_ok: bool = True
    auth_ok: bool = True
    # Concurrency instrumentation: each run() sleeps `latency_s` and the peak
    # number of simultaneously in-flight calls is recorded in `max_inflight`.
    latency_s: float = 0.0
    inflight: int = 0
    max_inflight: int = 0

    calls: list[tuple[str, ...]] = field(default_factory=list)
    _runs: dict[str, _Run] = field(default_factory=dict)
    _counter: int = 0
    _runs_created: int = 0

    # -- CliClient Protocol ---------------------------------------------------

    async def run(self, *args: str, timeout: float | None = None) -> CliResult:
        self.calls.append(args)
        self._maybe_raise(args)
        if args[:1] == ("version",) and not self.version_ok:
            raise CliGeneralError("fake: pandaprobe binary unavailable")
        if args[:2] == ("auth", "status") and not self.auth_ok:
            raise CliAuthError("fake: not authenticated")
        self.inflight += 1
        self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            if self.latency_s > 0:
                await asyncio.sleep(self.latency_s)
            payload = self._dispatch(args)
        finally:
            self.inflight -= 1
        return CliResult(args=args, exit_code=0, stdout=json.dumps(payload), stderr="")

    # -- helpers --------------------------------------------------------------

    def set_scores(self, **values: float) -> None:
        """Update current metric scores (e.g. after the agent self-heals)."""

        self.metric_values.update(values)

    def set_session_scores(self, session_id: str, **values: float) -> None:
        """Set per-session score overrides (e.g. for a replayed session)."""

        self.session_metric_values.setdefault(session_id, {}).update(values)

    @property
    def batch_calls(self) -> list[tuple[str, ...]]:
        return [c for c in self.calls if c[:3] == ("evals", "runs", "batch")]

    def _maybe_raise(self, args: Sequence[str]) -> None:
        for prefix, exc in self.error_on_prefix.items():
            if tuple(args[: len(prefix)]) == prefix:
                raise exc

    def _dispatch(self, args: Sequence[str]) -> Any:
        prefix = tuple(args[:3])
        if args[:1] == ("version",):
            return {"version": "v0.2.0-fake"}
        if args[:2] == ("auth", "status"):
            return {"authenticated": True}
        if prefix == ("evals", "runs", "batch"):
            return self._create_run(args)
        if prefix == ("evals", "runs", "scores"):
            return self._scores(args)
        if prefix == ("evals", "scores", "list"):
            session_id = _flag_value(args, "--session-id")
            if session_id and session_id in self.session_scores_list:
                return {"items": self.session_scores_list[session_id]}
            return {"items": self.scores_list_payload or []}
        if prefix == ("evals", "scores", "get"):
            return self.scores_get_payload or {
                "id": _positional(args, 3),
                "scores": [
                    {"name": m, "value": str(v), "status": "SUCCESS"}
                    for m, v in self.metric_values.items()
                ],
            }
        if prefix[:2] == ("traces", "spans"):
            return self.traces_spans_payload or {
                "trace_id": _positional(args, 2),
                "spans": [],
            }
        if prefix[:2] == ("traces", "get"):
            return {"trace_id": _positional(args, 2), "spans": []}
        if prefix[:2] == ("traces", "list"):
            return {"items": []}
        return {}

    def _create_run(self, args: Sequence[str]) -> dict[str, Any]:
        metrics_csv = _flag_value(args, "--metrics") or ""
        metrics = [m for m in metrics_csv.split(",") if m]
        self._counter += 1
        self._runs_created += 1
        run_id = f"run-session-{self._counter}"
        empty = self._runs_created <= self.empty_runs
        session_id = _flag_value(args, "--session-ids")
        self._runs[run_id] = _Run(metrics=[] if empty else metrics, session_id=session_id)
        return {"id": run_id, "status": "PENDING", "target_type": "SESSION"}

    def _scores(self, args: Sequence[str]) -> list[dict[str, Any]]:
        run_id = _positional(args, 3)
        run = self._runs.get(run_id or "")
        if run is None or not run.metrics:
            return []  # empty → non-terminal (lag) → evaluator retries/polls
        run.poll_count += 1
        if run.poll_count <= self.running_polls:
            return [{"name": m, "value": None, "status": "PENDING"} for m in run.metrics]
        return [self._score_record(m, session_id=run.session_id) for m in run.metrics]

    def _score_record(self, metric: str, *, session_id: str | None = None) -> dict[str, Any]:
        status = self.metric_status.get(metric, "SUCCESS")
        overrides = self.session_metric_values.get(session_id or "", {})
        value: str | None
        if metric in self.raw_metric_values:
            value = self.raw_metric_values[metric]  # verbatim (may be non-numeric)
        elif status.upper() in {"FAILED", "ERROR"}:
            value = None  # the backend returns a null value for failed scores
        elif metric in overrides:
            value = str(overrides[metric])  # per-session override (replayed sessions)
        else:
            value = str(self.metric_values.get(metric))
        return {
            "name": metric,
            "value": value,
            "status": status,
            "reason": f"score for {metric}",
            "metadata": self.metric_metadata.get(metric, {}),
        }


def _flag_value(args: Sequence[str], flag: str) -> str | None:
    for i, token in enumerate(args):
        if token == flag and i + 1 < len(args):
            return args[i + 1]
    return None


def _positional(args: Sequence[str], index: int) -> str | None:
    """The argument at ``index`` if it's not a flag, else None."""

    if index < len(args) and not args[index].startswith("--"):
        return args[index]
    return None
