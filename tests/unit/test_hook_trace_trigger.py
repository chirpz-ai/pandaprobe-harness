"""The trace trigger end to end through the hook: gate → notice → eval case."""

from __future__ import annotations

from pathlib import Path

from pandaprobe_harness import (
    HarnessConfig,
    HarnessFilesystem,
    Journal,
    Mailbox,
    PandaHarnessHook,
    RawLoopAdapter,
    ScoreHistoryStore,
)
from pandaprobe_harness.workspace.evalset import EvalSet
from tests.fakes.fake_cli_client import FakeCliClient


def _hook(
    tmp_path: Path, cli: FakeCliClient, **kw: object
) -> tuple[PandaHarnessHook, HarnessConfig]:
    cfg = HarnessConfig(
        harness_root=tmp_path / "h",
        poll_interval_s=0.0,
        poll_max_attempts=5,
        eval_retry_attempts=1,
        eval_retry_backoff_s=0.0,
        drain_timeout_s=5.0,
        gate_window=2,
        rule_validation=False,
        **kw,  # type: ignore[arg-type]
    )
    fs = HarnessFilesystem(cfg)
    fs.provision()
    return PandaHarnessHook(cli, config=cfg, filesystem=fs), cfg


async def _turn(hook: PandaHarnessHook, session: str, index: int) -> None:
    hook.on_turn_end(RawLoopAdapter.make_turn(session, index))
    await hook.refresh(session)


async def test_healthy_climb_posts_no_notice(tmp_path: Path) -> None:
    cli = FakeCliClient()
    cli.script_trajectory("s", "task_completion", [0.2, 0.5, 0.9])
    hook, cfg = _hook(tmp_path, cli)

    await _turn(hook, "s", 1)

    assert Mailbox(cfg).pending() == []


async def test_tier1_only_fire_is_advisory_and_captures_no_eval_case(
    tmp_path: Path,
) -> None:
    """A stalled trajectory whose step metrics are clean is a `trend` notice: the
    agent is told it is not progressing, but nothing is recorded as a failure
    case, because no specific step was shown to be wrong."""

    cli = FakeCliClient()  # step metrics healthy
    cli.script_trajectory("s", "task_completion", [0.2, 0.2, 0.2])
    hook, cfg = _hook(tmp_path, cli, capture_eval_cases=True)

    await _turn(hook, "s", 1)

    (notice,) = Mailbox(cfg).pending()
    assert notice.severity == "trend"
    assert "stall:task_completion" in notice.signatures
    # Detection tier does not make the resulting lesson universally applicable.
    assert notice.scope_hint == "scoped"
    assert EvalSet(cfg, journal=Journal(cfg)).cases() == []


async def test_tier2_breach_is_a_breach_notice_and_captures_a_failure_case(
    tmp_path: Path,
) -> None:
    cli = FakeCliClient(metric_values={"tool_correctness": 0.1})
    cli.script_trajectory("s", "task_completion", [0.2, 0.2, 0.2])
    hook, cfg = _hook(tmp_path, cli, capture_eval_cases=True)

    hook.on_turn_end({"session_id": "s", "turn_index": 1, "end_state": {"task": "t"}})
    await hook.refresh("s")

    (notice,) = Mailbox(cfg).pending()
    assert notice.severity == "breach"
    assert "breach:tool_correctness" in notice.signatures
    # A surgical step-level breach is scoped, not global.
    assert notice.scope_hint == "scoped"

    (case,) = EvalSet(cfg, journal=Journal(cfg)).cases()
    # The baseline is the trace metric that triggered — the signal promotion will
    # later be judged on, not a session composite.
    assert "tool_correctness" in case.baseline_scores


async def test_notice_carries_the_reason_trace_id_and_tier(tmp_path: Path) -> None:
    """The judge's free-text `reason` is the raw material for a rule, and the
    trace id tells the agent exactly what to inspect."""

    cli = FakeCliClient(metric_values={"tool_correctness": 0.1})
    cli.script_trajectory("s", "task_completion", [0.2, 0.2, 0.2])
    hook, cfg = _hook(tmp_path, cli)

    await _turn(hook, "s", 1)

    (notice,) = Mailbox(cfg).pending()
    by_name = {m.name: m for m in notice.metrics}
    assert by_name["tool_correctness"].reason == "score for tool_correctness"
    assert by_name["tool_correctness"].trace_id == "s-tr3"
    assert by_name["tool_correctness"].tier == 2
    assert by_name["task_completion"].tier == 1
    # The trace the agent needs to look at is flagged.
    assert "s-tr3" in notice.flagged_traces


async def test_dump_reports_per_trace_scores(tmp_path: Path) -> None:
    cli = FakeCliClient(metric_values={"tool_correctness": 0.1})
    cli.script_trajectory("s", "task_completion", [0.2, 0.2, 0.2])
    hook, cfg = _hook(tmp_path, cli)

    await _turn(hook, "s", 1)

    dump = HarnessFilesystem(cfg).read_latest_eval()
    assert dump["gate_breached"] is True
    breakdown = dump["signal_breakdown"]
    # One entry per scored trace, each carrying that trace's metric values.
    assert set(breakdown) == {"s-tr1", "s-tr2", "s-tr3"}
    assert breakdown["s-tr1"]["task_completion"] == 0.2
    assert breakdown["s-tr3"]["tool_correctness"] == 0.1


async def test_the_gate_records_a_series_per_session(tmp_path: Path) -> None:
    """v1's trend machinery was inert because history held one sample per
    session. The trace trigger must accumulate a real series within a session."""

    cli = FakeCliClient()
    cli.script_trajectory("s", "task_completion", [0.2, 0.4, 0.6, 0.8])
    hook, cfg = _hook(tmp_path, cli)

    await _turn(hook, "s", 1)

    assert ScoreHistoryStore(cfg).values("s", "task_completion") == [0.2, 0.4, 0.6, 0.8]


async def test_verifier_outcome_reaches_the_mailbox(tmp_path: Path) -> None:
    cli = FakeCliClient()
    cli.script_trajectory("s", "task_completion", [0.9])  # healthy: gate stays shut
    cfg = HarnessConfig(
        harness_root=tmp_path / "h",
        poll_interval_s=0.0,
        poll_max_attempts=5,
        eval_retry_attempts=1,
        eval_retry_backoff_s=0.0,
        drain_timeout_s=5.0,
        rule_validation=False,
    )
    fs = HarnessFilesystem(cfg)
    fs.provision()
    hook = PandaHarnessHook(
        cli, config=cfg, filesystem=fs, verifier=lambda _sid, _state: 0.25
    )

    await _turn(hook, "s", 1)

    (notice,) = Mailbox(cfg).pending()
    assert notice.severity == "breach"
    assert "breach:outcome_correct" in notice.signatures
