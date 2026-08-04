"""In-process fake implementing the ``CliClient`` Protocol.

Primary mock seam for the fast test suite — no subprocess, no network. Models
both evaluation surfaces of the real CLI:

* ``evals runs batch --target <trace|session> <--trace-ids|--session-ids> <ids>
  --metrics <m1,m2>`` hands out a ``run_id`` and remembers the metric set, the
  target and the ids for that run;
* ``evals runs scores <run_id> --target <trace|session>`` returns
  score-shaped dicts (value as a *string*, status ``SUCCESS``, and ``trace_id``
  on the trace target — matching the real payload), with optional
  ``running_polls`` PENDING rounds to exercise the poll loop;
* ``traces list --session-id <id>`` models trace ingestion: a session not
  scripted in ``session_traces`` **grows by one trace per listing**, because in
  reality one turn end emits one trace and is followed by one listing;
* scores can be flipped between turns (``metric_values``), per session
  (``session_metric_values``) or per trace (``trace_metric_values`` — the way to
  script a trajectory for the gate: climbing, stalled or regressing);
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

_DEFAULT_SCORES: dict[str, float] = {
    # Session composites (the v1 trigger).
    "agent_reliability": 0.9,
    "agent_consistency": 0.9,
    # Trace metrics (the v2 tiers). Healthy by default so a test opts *into*
    # failure rather than out of it.
    "task_completion": 0.9,
    "coherence": 0.9,
    "tool_correctness": 0.9,
    "argument_correctness": 0.9,
    "plan_adherence": 0.9,
    "plan_quality": 0.9,
    "step_efficiency": 0.9,
}


@dataclass
class _Run:
    metrics: list[str]
    poll_count: int = 0
    session_id: str | None = None
    trace_ids: list[str] = field(default_factory=list)
    target: str = "session"


@dataclass
class FakeCliClient:
    """A scripted, stateful fake of the ``pandaprobe`` CLI (both targets)."""

    running_polls: int = 0
    # metric name -> score in [0,1]. Mutate between turns to drive self-heal/trend.
    metric_values: dict[str, float] = field(default_factory=lambda: dict(_DEFAULT_SCORES))
    # session id -> {metric: score} overriding `metric_values` for that session
    # (lets a *replayed* session score differently from the live one).
    session_metric_values: dict[str, dict[str, float]] = field(default_factory=dict)
    # trace id -> {metric: score}, the narrowest override. This is how a test
    # scripts a *trajectory*: give consecutive traces rising, flat or falling
    # values and let the gate decide.
    trace_metric_values: dict[str, dict[str, float]] = field(default_factory=dict)
    # session id -> explicit trace ids (oldest first). A session absent here
    # grows by one synthetic trace on every `traces list` call.
    session_traces: dict[str, list[str]] = field(default_factory=dict)
    # session id -> successive oldest-first snapshots returned by trace listing.
    # The last snapshot repeats after the script is exhausted.
    session_trace_listings: dict[str, list[list[str]]] = field(default_factory=dict)
    auto_traces: bool = True
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
    _auto_traces: dict[str, list[str]] = field(default_factory=dict)
    _trace_listing_counts: dict[str, int] = field(default_factory=dict)

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

    def script_trajectory(self, session_id: str, metric: str, values: Sequence[float]) -> None:
        """Give ``session_id`` one trace per value, each scoring ``metric`` at it.

        The way to drive the trajectory gate: pass a rising series for a healthy
        climb, a flat one for a stall, or a peak-then-drop for a regression.
        """

        ids = [f"{session_id}-tr{i + 1}" for i in range(len(values))]
        self.session_traces[session_id] = ids
        for trace_id, value in zip(ids, values, strict=True):
            self.set_trace_scores(trace_id, **{metric: value})

    def set_trace_scores(self, trace_id: str, **values: float) -> None:
        """Set per-trace score overrides (the narrowest scope)."""

        self.trace_metric_values.setdefault(trace_id, {}).update(values)

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
            return self._traces_list(args)
        return {}

    # -- trace ingestion ------------------------------------------------------

    def _traces_list(self, args: Sequence[str]) -> dict[str, Any]:
        """``traces list --session-id`` — newest-first, like the real CLI."""

        session_id = _flag_value(args, "--session-id")
        if not session_id:
            return {"items": [], "pagination": {"total": 0}}
        scripted = self.session_trace_listings.get(session_id)
        if scripted:
            count = self._trace_listing_counts.get(session_id, 0)
            ids = scripted[min(count, len(scripted) - 1)]
            self._trace_listing_counts[session_id] = count + 1
        else:
            ids = self.session_traces.get(session_id)
        if ids is None:
            # Not scripted: model one-trace-per-turn by appending on each listing.
            ids = self._auto_traces.setdefault(session_id, [])
            if self.auto_traces:
                ids.append(f"{session_id}-tr{len(ids) + 1}")
        oldest_first = [
            {
                "trace_id": trace_id,
                "name": "fake-turn",
                "status": "COMPLETED",
                "started_at": f"2026-01-01T00:{index // 60:02d}:{index % 60:02d}+00:00",
                "session_id": session_id,
            }
            for index, trace_id in enumerate(ids)
        ]
        items = list(reversed(oldest_first))  # the CLI is queried newest-first
        limit = _flag_value(args, "--limit")
        if limit and limit.isdigit():
            items = items[: int(limit)]
        return {"items": items, "pagination": {"total": len(ids), "limit": limit}}

    # -- eval runs ------------------------------------------------------------

    def _create_run(self, args: Sequence[str]) -> dict[str, Any]:
        metrics_csv = _flag_value(args, "--metrics") or ""
        metrics = [m for m in metrics_csv.split(",") if m]
        target = _flag_value(args, "--target") or "trace"
        self._counter += 1
        self._runs_created += 1
        run_id = f"run-{target}-{self._counter}"
        empty = self._runs_created <= self.empty_runs
        # A trace run may cover several ids at once (the batch path), so keep them
        # all — every score comes back tagged with the trace it belongs to.
        trace_csv = _flag_value(args, "--trace-ids") or ""
        self._runs[run_id] = _Run(
            metrics=[] if empty else metrics,
            session_id=_flag_value(args, "--session-ids"),
            trace_ids=[t for t in trace_csv.split(",") if t],
            target=target,
        )
        return {
            "id": run_id,
            "status": "PENDING",
            "target_type": target.upper(),
        }

    def _scores(self, args: Sequence[str]) -> list[dict[str, Any]]:
        run_id = _positional(args, 3)
        run = self._runs.get(run_id or "")
        if run is None or not run.metrics:
            return []  # empty → non-terminal (lag) → evaluator retries/polls
        run.poll_count += 1
        if run.poll_count <= self.running_polls:
            return [{"name": m, "value": None, "status": "PENDING"} for m in run.metrics]
        targets: list[str | None] = list(run.trace_ids) or [None]
        return [
            self._score_record(m, session_id=run.session_id, trace_id=trace_id)
            for trace_id in targets
            for m in run.metrics
        ]

    def _score_record(
        self,
        metric: str,
        *,
        session_id: str | None = None,
        trace_id: str | None = None,
    ) -> dict[str, Any]:
        status = self.metric_status.get(metric, "SUCCESS")
        # Narrowest override wins: trace, then session, then the global default.
        trace_overrides = self.trace_metric_values.get(trace_id or "", {})
        session_overrides = self.session_metric_values.get(session_id or "", {})
        value: str | None
        if metric in self.raw_metric_values:
            value = self.raw_metric_values[metric]  # verbatim (may be non-numeric)
        elif status.upper() in {"FAILED", "ERROR"}:
            value = None  # the backend returns a null value for failed scores
        elif metric in trace_overrides:
            value = str(trace_overrides[metric])
        elif metric in session_overrides:
            value = str(session_overrides[metric])
        else:
            value = str(self.metric_values.get(metric))
        record: dict[str, Any] = {
            "name": metric,
            "value": value,
            "status": status,
            "reason": f"score for {metric}",
            "metadata": self.metric_metadata.get(metric, {}),
        }
        if trace_id is not None:
            record["trace_id"] = trace_id
        return record


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
