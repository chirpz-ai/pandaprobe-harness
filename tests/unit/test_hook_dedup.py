from __future__ import annotations

from pathlib import Path

from pandaprobe_harness import HarnessConfig, HarnessFilesystem, RawLoopAdapter
from pandaprobe_harness.hook.core import PandaHarnessHook
from tests.fakes.fake_cli_client import FakeCliClient


def _hook(
    tmp_path: Path, cli: FakeCliClient, **cfgkw: object
) -> tuple[PandaHarnessHook, RawLoopAdapter]:
    cfg = HarnessConfig(
        harness_root=tmp_path / "h",
        poll_interval_s=0.0,
        poll_max_attempts=5,
        eval_retry_backoff_s=0.0,
        **cfgkw,  # type: ignore[arg-type]
    )
    fs = HarnessFilesystem(cfg)
    fs.provision()
    adapter = RawLoopAdapter()
    return PandaHarnessHook(adapter, cli, config=cfg, filesystem=fs), adapter


async def _turn(hook: PandaHarnessHook, adapter: RawLoopAdapter, sid: str, idx: int) -> list[str]:
    hook.on_turn_end(adapter.make_turn(sid, idx))
    await hook.drain_pending(sid)
    return adapter.consume_alerts()


async def test_breach_alerts_once_until_recovery(tmp_path: Path) -> None:
    cli = FakeCliClient(metric_values={"agent_reliability": 0.2, "agent_consistency": 0.2})
    hook, adapter = _hook(tmp_path, cli, enable_trend=False)

    assert len(await _turn(hook, adapter, "s", 1)) == 1  # first breach alerts
    assert await _turn(hook, adapter, "s", 2) == []  # same breach suppressed

    cli.set_scores(agent_reliability=0.9, agent_consistency=0.9)
    assert await _turn(hook, adapter, "s", 3) == []  # recovered → no alert

    cli.set_scores(agent_reliability=0.2, agent_consistency=0.2)
    assert len(await _turn(hook, adapter, "s", 4)) == 1  # re-breach re-alerts


async def test_cooldown_reinjects_after_n_turns(tmp_path: Path) -> None:
    cli = FakeCliClient(metric_values={"agent_reliability": 0.2, "agent_consistency": 0.2})
    hook, adapter = _hook(tmp_path, cli, enable_trend=False, alert_cooldown_turns=2)

    assert len(await _turn(hook, adapter, "s", 1)) == 1  # inject; cooldown=2
    assert await _turn(hook, adapter, "s", 2) == []  # cooldown 2→1
    assert await _turn(hook, adapter, "s", 3) == []  # cooldown 1→0
    assert len(await _turn(hook, adapter, "s", 4)) == 1  # cooldown elapsed → reinject
