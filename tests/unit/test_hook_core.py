from __future__ import annotations

import asyncio
from pathlib import Path

from pandaprobe_harness import HarnessConfig, HarnessFilesystem, RawLoopAdapter
from pandaprobe_harness.cli.errors import CliError
from pandaprobe_harness.evaluation.metrics import EvalReport
from pandaprobe_harness.hook.core import PandaHarnessHook
from pandaprobe_harness.hook.turn import TurnContext
from tests.fakes.fake_cli_client import FakeCliClient


def _hook(
    cli: FakeCliClient, config: HarnessConfig, fs: HarnessFilesystem, adapter: RawLoopAdapter
) -> PandaHarnessHook:
    return PandaHarnessHook(adapter, cli, config=config, filesystem=fs)


def _make(
    tmp_path: Path, cli: FakeCliClient, **kw: object
) -> tuple[PandaHarnessHook, HarnessFilesystem, RawLoopAdapter]:
    cfg = HarnessConfig(
        harness_root=tmp_path / "h",
        poll_interval_s=0.0,
        poll_max_attempts=5,
        eval_retry_backoff_s=0.0,
        **kw,  # type: ignore[arg-type]
    )
    fs = HarnessFilesystem(cfg)
    fs.provision()
    adapter = RawLoopAdapter()
    return PandaHarnessHook(adapter, cli, config=cfg, filesystem=fs), fs, adapter


class _BlockingEvaluator:
    """Blocks until released — exercises the drain-timeout path."""

    def __init__(self) -> None:
        self.event = asyncio.Event()

    async def evaluate_turn(self, ctx: TurnContext) -> EvalReport:
        await self.event.wait()
        return EvalReport.from_scores(ctx.session_id, ctx.turn_index, [])


class _RaisingEvaluator:
    async def evaluate_turn(self, ctx: TurnContext) -> EvalReport:
        raise RuntimeError("unexpected eval failure")


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


async def test_enrichment_attaches_flagged_trace_detail(tmp_path: Path) -> None:
    cli = FakeCliClient(
        metric_values={"agent_reliability": 0.2, "agent_consistency": 0.2},
        metric_metadata={"agent_reliability": {"flagged_traces": ["trace-1"]}},
    )
    hook, fs, adapter = _make(tmp_path, cli, enrich_flagged_traces=True)
    hook.on_turn_end(adapter.make_turn("s", 1))
    await hook.drain_pending("s")

    dump = fs.read_latest_eval()
    assert dump["flagged_trace_detail"] == {"trace_id": "trace-1", "spans": []}
    assert any(c[:2] == ("traces", "get") and "--kind" in c for c in cli.calls)


async def test_enrichment_failure_still_writes_dump_and_alerts(tmp_path: Path) -> None:
    cli = FakeCliClient(
        metric_values={"agent_reliability": 0.2, "agent_consistency": 0.2},
        metric_metadata={"agent_reliability": {"flagged_traces": ["trace-1"]}},
        error_on_prefix={("traces", "get"): CliError("boom")},
    )
    hook, fs, adapter = _make(tmp_path, cli, enrich_flagged_traces=True)
    hook.on_turn_end(adapter.make_turn("s", 1))
    await hook.drain_pending("s")

    dump = fs.read_latest_eval()
    assert "flagged_trace_detail" not in dump  # enrichment is best-effort
    assert len(adapter.pending_alerts) == 1  # alert still injected


async def test_drain_timeout_keeps_task_for_later(tmp_path: Path) -> None:
    evaluator = _BlockingEvaluator()
    cfg = HarnessConfig(harness_root=tmp_path / "h", drain_timeout_s=0.02)
    fs = HarnessFilesystem(cfg)
    fs.provision()
    adapter = RawLoopAdapter()
    hook = PandaHarnessHook(
        adapter, FakeCliClient(), config=cfg, filesystem=fs, evaluator=evaluator  # type: ignore[arg-type]
    )

    hook.on_turn_end(adapter.make_turn("s", 1))
    assert await hook.drain_pending("s") is None  # not ready within budget
    assert "s" in hook._pending  # task retained for a later drain

    evaluator.event.set()
    assert await hook.drain_pending("s") is not None  # now resolves
    assert "s" not in hook._pending


async def test_drain_pops_on_unexpected_failure(tmp_path: Path) -> None:
    cfg = HarnessConfig(harness_root=tmp_path / "h", drain_timeout_s=1.0)
    fs = HarnessFilesystem(cfg)
    fs.provision()
    adapter = RawLoopAdapter()
    hook = PandaHarnessHook(
        adapter, FakeCliClient(), config=cfg, filesystem=fs, evaluator=_RaisingEvaluator()  # type: ignore[arg-type]
    )

    hook.on_turn_end(adapter.make_turn("s", 1))
    assert await hook.drain_pending("s") is None
    assert "s" not in hook._pending  # popped on failure
