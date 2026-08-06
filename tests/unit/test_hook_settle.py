"""The per-turn evaluation and managed-repair barrier."""

from __future__ import annotations

import asyncio
from pathlib import Path

from pandaprobe_harness import (
    Harness,
    HarnessConfig,
    HarnessFilesystem,
    Mailbox,
    PandaHarnessHook,
    RawLoopAdapter,
)
from pandaprobe_harness.evaluation.traces import TraceLocator
from pandaprobe_harness.workspace.evalset import EvalCase
from tests.fakes.fake_cli_client import FakeCliClient


class _BlockingEvaluator:
    """Stands in for a slow platform: resolves only when released."""

    def __init__(self) -> None:
        self.event = asyncio.Event()
        self.calls = 0
        # The hook shares its evaluator's locator so one seen-set governs both.
        self.locator = TraceLocator(FakeCliClient(), HarnessConfig())

    async def evaluate_traces(self, *args: object, **kwargs: object) -> list[object]:
        del args, kwargs
        self.calls += 1
        await self.event.wait()
        return []


def _cfg(tmp_path: Path, **kw: object) -> HarnessConfig:
    return HarnessConfig(
        harness_root=tmp_path / "h",
        poll_interval_s=0.0,
        poll_max_attempts=5,
        eval_retry_attempts=1,
        eval_retry_backoff_s=0.0,
        gate_window=2,
        repair_model="test/fake-repair",
        **{"rule_validation": True, **kw},  # type: ignore[arg-type]
    )


async def test_settle_waits_for_the_notice_to_be_posted(tmp_path: Path) -> None:
    """The point of the barrier: after it returns, the mailbox already reflects
    this turn — the agent cannot start its next turn blind to its own diagnosis."""

    cli = FakeCliClient(metric_values={"tool_correctness": 0.1})
    cli.script_trajectory("s", "task_completion", [0.2, 0.2, 0.2])
    cfg = _cfg(tmp_path)
    fs = HarnessFilesystem(cfg)
    fs.provision()
    hook = PandaHarnessHook(cli, config=cfg, filesystem=fs)

    hook.on_turn_end(RawLoopAdapter.make_turn("s", 1))
    result = await hook.settle("s")

    assert result.timed_out is False
    assert result.report is not None
    assert result.alerting
    assert len(Mailbox(cfg).pending()) == 1


async def test_settle_uses_the_barrier_budget_not_the_drain_budget(
    tmp_path: Path,
) -> None:
    """`drain_timeout_s` is a best-effort join; the barrier must be able to wait
    far longer than it."""

    evaluator = _BlockingEvaluator()
    # A drain budget too short to ever cover the eval, and a barrier budget
    # that comfortably does.
    cfg = _cfg(tmp_path, drain_timeout_s=0.01, barrier_timeout_s=5.0)
    fs = HarnessFilesystem(cfg)
    fs.provision()
    hook = PandaHarnessHook(
        FakeCliClient(), config=cfg, filesystem=fs, evaluator=evaluator  # type: ignore[arg-type]
    )

    hook.on_turn_end(RawLoopAdapter.make_turn("s", 1))
    assert await hook.refresh("s") is None  # the short drain budget gives up

    async def release() -> None:
        await asyncio.sleep(0.02)
        evaluator.event.set()

    releaser = asyncio.ensure_future(release())
    result = await hook.settle("s")
    await releaser

    assert result.report is not None  # the barrier waited it out
    assert result.timed_out is False


async def test_settle_reports_a_timeout_and_leaves_the_work_running(
    tmp_path: Path,
) -> None:
    evaluator = _BlockingEvaluator()
    cfg = _cfg(tmp_path, barrier_timeout_s=0.01)
    fs = HarnessFilesystem(cfg)
    fs.provision()
    hook = PandaHarnessHook(
        FakeCliClient(), config=cfg, filesystem=fs, evaluator=evaluator  # type: ignore[arg-type]
    )

    hook.on_turn_end(RawLoopAdapter.make_turn("s", 1))
    result = await hook.settle("s")

    assert result.timed_out is True
    assert result.report is None
    # A slow platform costs latency, not correctness: the eval is still going.
    assert "s" in hook._pending

    evaluator.event.set()
    assert await hook.refresh("s") is not None


async def test_settle_does_not_wait_for_the_validation_replay_round(
    tmp_path: Path,
) -> None:
    """A replay re-runs the developer's agent, which can need the very resource the
    current turn holds — an environment lock, a world, a container. Awaiting it
    inside a per-turn barrier deadlocks until the replay times out, so the barrier
    covers the diagnosis only; drain_validation handles the round at a phase
    boundary, where nothing is held."""

    started = asyncio.Event()
    release = asyncio.Event()

    async def replay(case: EvalCase, context: str) -> str:
        del case, context
        started.set()
        await release.wait()  # stands in for blocking on a held resource
        return "s-replay"

    cli = FakeCliClient(metric_values={"tool_correctness": 0.1})
    cli.script_trajectory("s", "task_completion", [0.2, 0.2, 0.2])
    cfg = _cfg(tmp_path, rule_validation=True, capture_eval_cases=True)
    harness = Harness.create(cfg, cli=cli, replay=replay)
    harness.rules.add(
        "check first", "why", metric="tool_correctness"
    )

    harness.on_turn_end(
        {"session_id": "s", "turn_index": 1, "end_state": {"task": "t"}}
    )
    # Completes despite the replay being wedged.
    result = await asyncio.wait_for(harness.settle("s"), timeout=5.0)

    assert result.report is not None
    assert harness.mailbox.pending() == []  # managed repair resolved the diagnosis
    assert result.repair is not None

    release.set()
    await harness.drain_validation()


async def test_settle_with_nothing_in_flight_is_a_no_op(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    fs = HarnessFilesystem(cfg)
    fs.provision()
    hook = PandaHarnessHook(FakeCliClient(), config=cfg, filesystem=fs)

    result = await hook.settle("never-seen")

    assert result.report is None
    assert result.timed_out is False


async def test_explicit_timeout_overrides_the_configured_budget(tmp_path: Path) -> None:
    evaluator = _BlockingEvaluator()
    cfg = _cfg(tmp_path, barrier_timeout_s=30.0)
    fs = HarnessFilesystem(cfg)
    fs.provision()
    hook = PandaHarnessHook(
        FakeCliClient(), config=cfg, filesystem=fs, evaluator=evaluator  # type: ignore[arg-type]
    )

    hook.on_turn_end(RawLoopAdapter.make_turn("s", 1))
    result = await hook.settle("s", timeout=0.01)

    assert result.timed_out is True
    evaluator.event.set()
    await hook.refresh("s")


async def test_turn_scope_can_settle(tmp_path: Path) -> None:
    cli = FakeCliClient(metric_values={"tool_correctness": 0.1})
    cli.script_trajectory("s", "task_completion", [0.2, 0.2, 0.2])
    harness = Harness.create(_cfg(tmp_path), cli=cli)

    async with harness.turn("s", settle=True):
        pass

    # No refresh call needed: the scope itself waited for the cycle.
    assert harness.mailbox.pending() == []
    assert len(harness.journal.recent(types=("repair_no_proposal",))) == 1


async def test_turn_scope_does_not_settle_by_default(tmp_path: Path) -> None:
    """Existing callers keep fire-and-forget semantics: the scope schedules the
    evaluation and returns, so nothing has been posted yet."""

    cli = FakeCliClient(metric_values={"tool_correctness": 0.1})
    cli.script_trajectory("s", "task_completion", [0.2, 0.2, 0.2])
    harness = Harness.create(_cfg(tmp_path), cli=cli)

    async with harness.turn("s"):
        pass

    assert harness.mailbox.pending() == []  # returned without waiting
    # Explicit settlement evaluates and lets managed repair resolve the notice.
    await harness.settle("s")
    assert harness.mailbox.pending() == []
    assert len(harness.journal.recent(types=("repair_no_proposal",))) == 1
