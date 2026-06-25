from __future__ import annotations

from pandaprobe_harness import HarnessConfig, HarnessFilesystem, RawLoopAdapter
from pandaprobe_harness.hook.core import PandaHarnessHook
from tests.fakes.fake_cli_client import FakeCliClient


def _hook(
    cli: FakeCliClient, config: HarnessConfig, fs: HarnessFilesystem, adapter: RawLoopAdapter
) -> PandaHarnessHook:
    return PandaHarnessHook(adapter, cli, config=config, filesystem=fs)


async def test_breach_writes_dump_and_injects_alert(
    config: HarnessConfig, fs: HarnessFilesystem, adapter: RawLoopAdapter
) -> None:
    cli = FakeCliClient(metric_values={"agent_reliability": 0.2, "agent_consistency": 0.3})
    hook = _hook(cli, config, fs, adapter)

    hook.on_turn_end(adapter.make_turn("s-1", 1, action="charge"))
    report = await hook.drain_pending("s-1")

    assert report is not None and report.any_breach
    assert config.latest_eval_file.exists()
    dump = fs.read_latest_eval()
    assert dump["any_breach"] is True
    assert len(adapter.pending_alerts) == 1
    assert "SYSTEM ALERT" in adapter.pending_alerts[0]


async def test_no_breach_writes_nothing_and_no_alert(
    config: HarnessConfig, fs: HarnessFilesystem, adapter: RawLoopAdapter
) -> None:
    cli = FakeCliClient(metric_values={"agent_reliability": 0.9, "agent_consistency": 0.9})
    hook = _hook(cli, config, fs, adapter)

    hook.on_turn_end(adapter.make_turn("s-1", 1))
    report = await hook.drain_pending("s-1")

    assert report is not None and not report.any_breach
    assert not config.latest_eval_file.exists()
    assert adapter.pending_alerts == ()


async def test_drain_with_no_pending_returns_none(
    config: HarnessConfig, fs: HarnessFilesystem, adapter: RawLoopAdapter
) -> None:
    hook = _hook(FakeCliClient(), config, fs, adapter)
    assert await hook.drain_pending("unknown") is None


async def test_parse_failure_is_swallowed(
    config: HarnessConfig, fs: HarnessFilesystem, adapter: RawLoopAdapter
) -> None:
    hook = _hook(FakeCliClient(), config, fs, adapter)
    # Missing session_id -> parse_turn raises, but on_turn_end must not.
    hook.on_turn_end({"turn_index": 1})
    # nothing scheduled; drain is a no-op
    assert await hook.drain_pending("s-1") is None


async def test_drain_pops_task_so_second_drain_is_none(
    config: HarnessConfig, fs: HarnessFilesystem, adapter: RawLoopAdapter
) -> None:
    cli = FakeCliClient(metric_values={"agent_reliability": 0.9, "agent_consistency": 0.9})
    hook = _hook(cli, config, fs, adapter)
    hook.on_turn_end(adapter.make_turn("s-1", 1))
    assert await hook.drain_pending("s-1") is not None
    assert await hook.drain_pending("s-1") is None
