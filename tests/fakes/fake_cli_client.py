"""In-process fake implementing the ``CliClient`` Protocol.

This is the primary mock seam for the fast test suite — no subprocess, no
network. It is a stateful dispatcher keyed on the argv prefix that:

* resolves trace IDs for ``traces list``,
* hands out ``run_id``s for ``evals runs batch`` (remembering which metric each
  run was for),
* simulates asynchronous polling — the first ``running_polls`` calls to
  ``evals runs scores`` report ``running`` before a terminal score,
* serves per-metric scores that the test can **flip between turns** (low →
  high) to drive the self-healing scenario,
* can raise typed ``CliError``s on a matching prefix to exercise degrade paths.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pandaprobe_harness.cli.client import CliResult
from pandaprobe_harness.cli.errors import CliError


@dataclass
class _Run:
    metric: str
    poll_count: int = 0


@dataclass
class FakeCliClient:
    """A scripted, stateful fake of the ``pandaprobe`` CLI."""

    trace_ids: list[str] = field(default_factory=lambda: ["trace-1"])
    running_polls: int = 0
    # metric name -> score in [0,1]. Mutate between turns to drive self-heal.
    metric_values: dict[str, float] = field(
        default_factory=lambda: {"agent_reliability": 0.9, "agent_consistency": 0.9}
    )
    metric_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    # argv-prefix tuple -> exception to raise (degrade-path testing).
    error_on_prefix: dict[tuple[str, ...], CliError] = field(default_factory=dict)
    # If set, `evals scores get <id>` returns this raw stdout (agent diagnosis).
    scores_get_payload: dict[str, Any] | None = None

    calls: list[tuple[str, ...]] = field(default_factory=list)
    _runs: dict[str, _Run] = field(default_factory=dict)
    _counter: int = 0

    # -- CliClient Protocol ---------------------------------------------------

    async def run(self, *args: str, timeout: float | None = None) -> CliResult:
        self.calls.append(args)
        self._maybe_raise(args)
        payload = self._dispatch(args)
        return CliResult(args=args, exit_code=0, stdout=json.dumps(payload), stderr="")

    # -- helpers --------------------------------------------------------------

    def set_scores(self, **values: float) -> None:
        """Update current metric scores (e.g. after the agent self-heals)."""

        self.metric_values.update(values)

    def _maybe_raise(self, args: Sequence[str]) -> None:
        for prefix, exc in self.error_on_prefix.items():
            if tuple(args[: len(prefix)]) == prefix:
                raise exc

    def _dispatch(self, args: Sequence[str]) -> Any:
        prefix = tuple(args[:3])
        if prefix[:2] == ("traces", "list"):
            return {"items": [{"trace_id": tid} for tid in self.trace_ids]}
        if prefix == ("evals", "runs", "batch"):
            return self._create_run(args)
        if prefix == ("evals", "runs", "scores"):
            return self._scores(args)
        if prefix[:2] == ("evals", "scores"):  # evals scores get <id>
            return self.scores_get_payload or {
                "trace_id": args[-1],
                "scores": [
                    {"name": m, "value": v, "status": "completed"}
                    for m, v in self.metric_values.items()
                ],
            }
        if prefix[:2] == ("traces", "get"):
            return {"trace_id": args[-1], "spans": []}
        return {}

    def _create_run(self, args: Sequence[str]) -> dict[str, Any]:
        metric = _flag_value(args, "--metrics") or "unknown"
        self._counter += 1
        run_id = f"run-{metric}-{self._counter}"
        self._runs[run_id] = _Run(metric=metric)
        return {"run_id": run_id, "status": "pending"}

    def _scores(self, args: Sequence[str]) -> dict[str, Any]:
        run_id = args[-1]
        run = self._runs.get(run_id)
        if run is None:
            return {"run_id": run_id, "scores": []}
        run.poll_count += 1
        if run.poll_count <= self.running_polls:
            return {
                "run_id": run_id,
                "scores": [{"name": run.metric, "value": None, "status": "running"}],
            }
        return {
            "run_id": run_id,
            "scores": [
                {
                    "name": run.metric,
                    "value": self.metric_values.get(run.metric),
                    "status": "completed",
                    "reason": f"score for {run.metric}",
                    "metadata": self.metric_metadata.get(run.metric, {}),
                }
            ],
        }


def _flag_value(args: Sequence[str], flag: str) -> str | None:
    for i, token in enumerate(args):
        if token == flag and i + 1 < len(args):
            return args[i + 1]
    return None
