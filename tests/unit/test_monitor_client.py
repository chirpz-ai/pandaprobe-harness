"""MonitorClient drives the CLI's `evals monitors` commands via the CliClient seam."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from pandaprobe_harness import MonitorClient, MonitorResponse
from pandaprobe_harness.cli.client import CliResult

_MONITOR = {
    "id": "m1",
    "project_id": "p1",
    "name": "n",
    "target_type": "SESSION",
    "metric_names": ["agent_reliability"],
    "filters": {},
    "sampling_rate": 1.0,
    "model": None,
    "cadence": "daily",
    "only_if_changed": True,
    "status": "ACTIVE",
    "last_run_at": None,
    "last_run_id": None,
    "next_run_at": "2026-06-26T00:00:00Z",
    "created_at": "2026-06-25T00:00:00Z",
    "updated_at": "2026-06-25T00:00:00Z",
}


class _Cli:
    """Records argv and returns a scripted JSON payload (by subcommand or fixed)."""

    def __init__(self, payload: Any | Callable[[Sequence[str]], Any]) -> None:
        self.calls: list[tuple[str, ...]] = []
        self._payload = payload

    async def run(self, *args: str, timeout: float | None = None) -> CliResult:
        self.calls.append(args)
        payload = self._payload(args) if callable(self._payload) else self._payload
        return CliResult(args=args, exit_code=0, stdout=json.dumps(payload), stderr="")


async def test_create_builds_correct_argv() -> None:
    cli = _Cli(_MONITOR)
    mc = MonitorClient(cli)
    monitor = await mc.create(
        "n", "session", ["agent_reliability", "agent_consistency"],
        cadence="daily", sampling_rate=0.5, only_if_changed=False,
        signal_weights={"confidence": 1.0}, filters={"has_error": True, "tags": ["x", "y"]},
    )
    assert isinstance(monitor, MonitorResponse)
    assert monitor.id == "m1" and monitor.status == "ACTIVE"
    argv = cli.calls[0]
    joined = " ".join(argv)
    assert argv[:3] == ("evals", "monitors", "create")
    assert "--target session" in joined
    assert "--name n" in joined
    assert "--metrics agent_reliability,agent_consistency" in joined
    assert "--cadence daily" in joined
    assert "--sampling-rate 0.5" in joined
    assert "--only-if-changed=false" in joined  # cobra bool single-token form
    assert "--signal-weights" in joined and "confidence" in joined
    assert "--has-error=true" in joined
    assert "--tags x,y" in joined


async def test_list_parses_listresult_items() -> None:
    cli = _Cli({"items": [_MONITOR], "pagination": {"total": 1, "limit": 50, "offset": 0}})
    monitors = await MonitorClient(cli).list_monitors(status="ACTIVE", limit=10)
    assert len(monitors) == 1 and monitors[0].id == "m1"
    joined = " ".join(cli.calls[0])
    assert "evals monitors list" in joined
    assert "--status ACTIVE" in joined and "--limit 10" in joined


async def test_get_pause_resume_paths() -> None:
    cli = _Cli(_MONITOR)
    mc = MonitorClient(cli)
    await mc.get("m1")
    await mc.pause("m1")
    await mc.resume("m1")
    assert cli.calls[0] == ("evals", "monitors", "get", "m1")
    assert cli.calls[1] == ("evals", "monitors", "pause", "m1")
    assert cli.calls[2] == ("evals", "monitors", "resume", "m1")


async def test_update_sends_only_changed_fields() -> None:
    cli = _Cli(_MONITOR)
    await MonitorClient(cli).update("m1", cadence="weekly", filters={"user_id": "u1"})
    argv = cli.calls[0]
    joined = " ".join(argv)
    assert argv[:4] == ("evals", "monitors", "update", "m1")
    assert "--cadence weekly" in joined
    assert "--filters" in joined  # raw JSON for partial update
    assert "--name" not in joined  # unchanged fields omitted


async def test_trigger_returns_eval_run_dict() -> None:
    cli = _Cli({"id": "run1", "status": "PENDING", "monitor_id": "m1"})
    run = await MonitorClient(cli).trigger("m1")
    assert run["id"] == "run1" and run["status"] == "PENDING"
    assert cli.calls[0] == ("evals", "monitors", "trigger", "m1")


async def test_runs_unwraps_items() -> None:
    cli = _Cli({"items": [{"id": "run1"}], "pagination": {"total": 1, "limit": 50, "offset": 0}})
    runs = await MonitorClient(cli).runs("m1", limit=5)
    assert runs == [{"id": "run1"}]
    assert "--limit 5" in " ".join(cli.calls[0])


async def test_delete_invokes_cli() -> None:
    cli = _Cli({"status": "deleted", "id": "m1"})
    await MonitorClient(cli).delete("m1")  # must not raise
    assert cli.calls[0] == ("evals", "monitors", "delete", "m1")
