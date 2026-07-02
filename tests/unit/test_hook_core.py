from __future__ import annotations

import asyncio
import os
from pathlib import Path

from pandaprobe_harness import HarnessConfig, HarnessFilesystem, Mailbox, RawLoopAdapter
from pandaprobe_harness.cli.errors import CliError
from pandaprobe_harness.evaluation.metrics import EvalReport
from pandaprobe_harness.hook.core import PandaHarnessHook
from pandaprobe_harness.hook.turn import TurnContext
from tests.fakes.fake_cli_client import FakeCliClient


def _hook(cli: FakeCliClient, config: HarnessConfig, fs: HarnessFilesystem) -> PandaHarnessHook:
    return PandaHarnessHook(cli, config=config, filesystem=fs)


def _make(
    tmp_path: Path, cli: FakeCliClient, **kw: object
) -> tuple[PandaHarnessHook, HarnessFilesystem, HarnessConfig]:
    cfg = HarnessConfig(
        harness_root=tmp_path / "h",
        poll_interval_s=0.0,
        poll_max_attempts=5,
        eval_retry_backoff_s=0.0,
        **kw,  # type: ignore[arg-type]
    )
    fs = HarnessFilesystem(cfg)
    fs.provision()
    return PandaHarnessHook(cli, config=cfg, filesystem=fs), fs, cfg


class _BlockingEvaluator:
    """Blocks until released — exercises the refresh-timeout path."""

    def __init__(self) -> None:
        self.event = asyncio.Event()

    async def evaluate_turn(self, ctx: TurnContext) -> EvalReport:
        await self.event.wait()
        return EvalReport.from_scores(ctx.session_id, ctx.turn_index, [])


class _RaisingEvaluator:
    async def evaluate_turn(self, ctx: TurnContext) -> EvalReport:
        raise RuntimeError("unexpected eval failure")


async def test_breach_writes_dump_and_posts_notice(
    config: HarnessConfig, fs: HarnessFilesystem, mailbox: Mailbox
) -> None:
    cli = FakeCliClient(metric_values={"agent_reliability": 0.2, "agent_consistency": 0.3})
    hook = _hook(cli, config, fs)

    hook.on_turn_end(RawLoopAdapter.make_turn("s-1", 1, action="charge"))
    report = await hook.refresh("s-1")

    assert report is not None and report.any_breach
    assert config.latest_eval_file.exists()
    dump = fs.read_latest_eval()
    assert dump["any_breach"] is True

    pending = mailbox.pending()
    assert len(pending) == 1
    notice = pending[0]
    assert notice.severity == "breach"
    assert notice.session_id == "s-1"
    assert {m.name for m in notice.metrics} == {"agent_reliability", "agent_consistency"}
    assert notice.dump_path and os.path.exists(notice.dump_path)
    assert "breach:agent_reliability" in notice.signatures

    # The full cycle is journaled.
    events = hook.journal.recent(types=("notice",))
    assert len(events) == 1 and events[0]["id"] == notice.id


async def test_no_breach_writes_nothing_and_no_notice(
    config: HarnessConfig, fs: HarnessFilesystem, mailbox: Mailbox
) -> None:
    cli = FakeCliClient(metric_values={"agent_reliability": 0.9, "agent_consistency": 0.9})
    hook = _hook(cli, config, fs)

    hook.on_turn_end(RawLoopAdapter.make_turn("s-1", 1))
    report = await hook.refresh("s-1")

    assert report is not None and not report.any_breach
    assert not config.latest_eval_file.exists()
    assert mailbox.pending() == []


async def test_refresh_with_no_pending_returns_none(
    config: HarnessConfig, fs: HarnessFilesystem
) -> None:
    hook = _hook(FakeCliClient(), config, fs)
    assert await hook.refresh("unknown") is None


async def test_parse_failure_is_swallowed(config: HarnessConfig, fs: HarnessFilesystem) -> None:
    hook = _hook(FakeCliClient(), config, fs)
    # Missing session_id -> the parser raises, but on_turn_end must not.
    hook.on_turn_end({"turn_index": 1})
    # nothing scheduled; refresh is a no-op
    assert await hook.refresh("s-1") is None


async def test_refresh_pops_task_so_second_refresh_is_none(
    config: HarnessConfig, fs: HarnessFilesystem
) -> None:
    cli = FakeCliClient(metric_values={"agent_reliability": 0.9, "agent_consistency": 0.9})
    hook = _hook(cli, config, fs)
    hook.on_turn_end(RawLoopAdapter.make_turn("s-1", 1))
    assert await hook.refresh("s-1") is not None
    assert await hook.refresh("s-1") is None


async def test_notice_posts_without_any_refresh(
    config: HarnessConfig, fs: HarnessFilesystem, mailbox: Mailbox
) -> None:
    """The pull model needs no drain barrier: the eval task posts on resolution."""

    cli = FakeCliClient(metric_values={"agent_reliability": 0.2, "agent_consistency": 0.2})
    hook = _hook(cli, config, fs)
    hook.on_turn_end(RawLoopAdapter.make_turn("s-1", 1))

    for _ in range(200):  # wait for the detached task, never calling refresh
        if mailbox.pending():
            break
        await asyncio.sleep(0.01)
    assert len(mailbox.pending()) == 1


async def test_enrichment_attaches_flagged_trace_detail(tmp_path: Path) -> None:
    cli = FakeCliClient(
        metric_values={"agent_reliability": 0.2, "agent_consistency": 0.2},
        metric_metadata={"agent_reliability": {"flagged_traces": ["trace-1"]}},
    )
    hook, fs, _cfg = _make(tmp_path, cli, enrich_flagged_traces=True)
    hook.on_turn_end(RawLoopAdapter.make_turn("s", 1))
    await hook.refresh("s")

    dump = fs.read_latest_eval()
    assert dump["flagged_trace_detail"] == {"trace_id": "trace-1", "spans": []}
    assert any(c[:2] == ("traces", "get") and "--kind" in c for c in cli.calls)


async def test_enrichment_failure_still_writes_dump_and_posts(tmp_path: Path) -> None:
    cli = FakeCliClient(
        metric_values={"agent_reliability": 0.2, "agent_consistency": 0.2},
        metric_metadata={"agent_reliability": {"flagged_traces": ["trace-1"]}},
        error_on_prefix={("traces", "get"): CliError("boom")},
    )
    hook, fs, cfg = _make(tmp_path, cli, enrich_flagged_traces=True)
    hook.on_turn_end(RawLoopAdapter.make_turn("s", 1))
    await hook.refresh("s")

    dump = fs.read_latest_eval()
    assert "flagged_trace_detail" not in dump  # enrichment is best-effort
    assert len(Mailbox(cfg).pending()) == 1  # notice still posted


async def test_refresh_timeout_keeps_task_for_later(tmp_path: Path) -> None:
    evaluator = _BlockingEvaluator()
    cfg = HarnessConfig(harness_root=tmp_path / "h", drain_timeout_s=0.02)
    fs = HarnessFilesystem(cfg)
    fs.provision()
    hook = PandaHarnessHook(
        FakeCliClient(), config=cfg, filesystem=fs, evaluator=evaluator  # type: ignore[arg-type]
    )

    hook.on_turn_end(RawLoopAdapter.make_turn("s", 1))
    assert await hook.refresh("s") is None  # not ready within budget
    assert "s" in hook._pending  # task retained, still running detached

    evaluator.event.set()
    assert await hook.refresh("s") is not None  # now resolves
    await asyncio.sleep(0)  # let the task's finally-cleanup run
    assert "s" not in hook._pending


async def test_eval_failure_is_contained_in_the_task(tmp_path: Path) -> None:
    cfg = HarnessConfig(harness_root=tmp_path / "h", drain_timeout_s=1.0)
    fs = HarnessFilesystem(cfg)
    fs.provision()
    hook = PandaHarnessHook(
        FakeCliClient(), config=cfg, filesystem=fs, evaluator=_RaisingEvaluator()  # type: ignore[arg-type]
    )

    hook.on_turn_end(RawLoopAdapter.make_turn("s", 1))
    assert await hook.refresh("s") is None  # the exception never escapes
    assert "s" not in hook._pending  # popped by the task's own cleanup


async def test_superseding_turn_cancels_prior_eval(tmp_path: Path) -> None:
    evaluator = _BlockingEvaluator()
    cfg = HarnessConfig(harness_root=tmp_path / "h", drain_timeout_s=0.5)
    fs = HarnessFilesystem(cfg)
    fs.provision()
    hook = PandaHarnessHook(
        FakeCliClient(), config=cfg, filesystem=fs, evaluator=evaluator  # type: ignore[arg-type]
    )

    hook.on_turn_end(RawLoopAdapter.make_turn("s", 1))
    first = hook._pending["s"]
    hook.on_turn_end(RawLoopAdapter.make_turn("s", 2))
    await asyncio.sleep(0)
    assert first.cancelled()
    # refresh joins the superseding task, not the cancelled one.
    evaluator.event.set()
    assert await hook.refresh("s") is not None
