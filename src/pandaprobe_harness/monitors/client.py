"""Evaluation-monitor management via the ``pandaprobe`` CLI.

Monitors are scheduled, recurring evaluation runs — useful for out-of-loop trend
watching that complements the in-loop EWMA detector. This client shells out to
the CLI's ``evals monitors`` command group through the shared ``CliClient`` seam,
so authentication is the CLI's responsibility (env / ``~/.pandaprobe/config.yaml``)
exactly like every other harness CLI call — no separate auth to manage.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..cli.client import CliClient

__all__ = ["MonitorClient", "MonitorResponse"]


@dataclass(frozen=True, slots=True)
class MonitorResponse:
    """A parsed view of the CLI's ``MonitorResponse`` JSON."""

    id: str
    name: str
    target_type: str
    metric_names: tuple[str, ...]
    cadence: str
    status: str
    filters: dict[str, Any] = field(default_factory=dict)
    sampling_rate: float = 1.0
    model: str | None = None
    only_if_changed: bool = True
    last_run_at: str | None = None
    last_run_id: str | None = None
    next_run_at: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> MonitorResponse:
        filters = payload.get("filters")
        return cls(
            id=str(payload.get("id", "")),
            name=str(payload.get("name", "")),
            target_type=str(payload.get("target_type", "")),
            metric_names=tuple(payload.get("metric_names") or []),
            cadence=str(payload.get("cadence", "")),
            status=str(payload.get("status", "")),
            filters=dict(filters) if isinstance(filters, Mapping) else {},
            sampling_rate=float(payload.get("sampling_rate", 1.0)),
            model=payload.get("model"),
            only_if_changed=bool(payload.get("only_if_changed", True)),
            last_run_at=payload.get("last_run_at"),
            last_run_id=payload.get("last_run_id"),
            next_run_at=payload.get("next_run_at"),
            raw=dict(payload),
        )


def _bool_flag(name: str, value: bool) -> str:
    # Cobra bool flags must use the `--flag=value` single-token form.
    return f"--{name}={'true' if value else 'false'}"


def _filter_args(filters: Mapping[str, Any] | None) -> list[str]:
    """Translate a filters mapping into ``evals monitors create`` filter flags."""

    if not filters:
        return []
    args: list[str] = []
    scalar = {
        "date_from": "--date-from",
        "date_to": "--date-to",
        "status": "--status",
        "session_id": "--session-id",
        "user_id": "--user-id",
        "filter_name": "--filter-name",
        "min_trace_count": "--min-trace-count",
    }
    for key, flag in scalar.items():
        value = filters.get(key)
        if value is not None:
            args += [flag, str(value)]
    tags = filters.get("tags")
    if tags:
        args += ["--tags", ",".join(str(t) for t in tags)]
    if filters.get("has_error") is not None:
        args += [_bool_flag("has-error", bool(filters["has_error"]))]
    return args


class MonitorClient:
    """Manage evaluation monitors through the ``pandaprobe`` CLI."""

    def __init__(self, cli: CliClient) -> None:
        self._cli = cli

    # -- create / update ------------------------------------------------------

    async def create(
        self,
        name: str,
        target: str,
        metrics: Sequence[str],
        *,
        cadence: str,
        sampling_rate: float | None = None,
        model: str | None = None,
        only_if_changed: bool | None = None,
        signal_weights: Mapping[str, float] | None = None,
        filters: Mapping[str, Any] | None = None,
    ) -> MonitorResponse:
        args = [
            "evals", "monitors", "create",
            "--target", target,
            "--name", name,
            "--metrics", ",".join(metrics),
            "--cadence", cadence,
        ]
        if sampling_rate is not None:
            args += ["--sampling-rate", str(sampling_rate)]
        if model is not None:
            args += ["--model", model]
        if only_if_changed is not None:
            args.append(_bool_flag("only-if-changed", only_if_changed))
        if signal_weights is not None:
            args += ["--signal-weights", json.dumps(dict(signal_weights))]
        args += _filter_args(filters)
        result = await self._cli.run(*args)
        return MonitorResponse.parse(result.json())

    async def update(
        self,
        monitor_id: str,
        *,
        name: str | None = None,
        metrics: Sequence[str] | None = None,
        cadence: str | None = None,
        sampling_rate: float | None = None,
        model: str | None = None,
        only_if_changed: bool | None = None,
        filters: Mapping[str, Any] | None = None,
        signal_weights: Mapping[str, float] | None = None,
    ) -> MonitorResponse:
        args = ["evals", "monitors", "update", monitor_id]
        if name is not None:
            args += ["--name", name]
        if metrics is not None:
            args += ["--metrics", ",".join(metrics)]
        if cadence is not None:
            args += ["--cadence", cadence]
        if sampling_rate is not None:
            args += ["--sampling-rate", str(sampling_rate)]
        if model is not None:
            args += ["--model", model]
        if only_if_changed is not None:
            args.append(_bool_flag("only-if-changed", only_if_changed))
        if filters is not None:
            args += ["--filters", json.dumps(dict(filters))]
        if signal_weights is not None:
            args += ["--signal-weights", json.dumps(dict(signal_weights))]
        result = await self._cli.run(*args)
        return MonitorResponse.parse(result.json())

    # -- read -----------------------------------------------------------------

    async def list_monitors(
        self, *, status: str | None = None, limit: int | None = None, offset: int | None = None
    ) -> list[MonitorResponse]:
        args = ["evals", "monitors", "list"]
        if status:
            args += ["--status", status]
        if limit is not None:
            args += ["--limit", str(limit)]
        if offset is not None:
            args += ["--offset", str(offset)]
        return [MonitorResponse.parse(item) for item in _items((await self._cli.run(*args)).json())]

    async def get(self, monitor_id: str) -> MonitorResponse:
        result = await self._cli.run("evals", "monitors", "get", monitor_id)
        return MonitorResponse.parse(result.json())

    async def runs(
        self, monitor_id: str, *, limit: int | None = None, offset: int | None = None
    ) -> list[dict[str, Any]]:
        args = ["evals", "monitors", "runs", monitor_id]
        if limit is not None:
            args += ["--limit", str(limit)]
        if offset is not None:
            args += ["--offset", str(offset)]
        return list(_items((await self._cli.run(*args)).json()))

    # -- lifecycle ------------------------------------------------------------

    async def pause(self, monitor_id: str) -> MonitorResponse:
        result = await self._cli.run("evals", "monitors", "pause", monitor_id)
        return MonitorResponse.parse(result.json())

    async def resume(self, monitor_id: str) -> MonitorResponse:
        result = await self._cli.run("evals", "monitors", "resume", monitor_id)
        return MonitorResponse.parse(result.json())

    async def trigger(self, monitor_id: str) -> dict[str, Any]:
        result = await self._cli.run("evals", "monitors", "trigger", monitor_id)
        payload = result.json()
        return dict(payload) if isinstance(payload, Mapping) else {}

    async def delete(self, monitor_id: str) -> None:
        await self._cli.run("evals", "monitors", "delete", monitor_id)


def _items(payload: Any) -> list[Any]:
    """Extract the items list from a CLI ``ListResult`` (``{items, pagination}``)."""

    if isinstance(payload, Mapping):
        items = payload.get("items", [])
        return list(items) if isinstance(items, Sequence) else []
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        return list(payload)
    return []
