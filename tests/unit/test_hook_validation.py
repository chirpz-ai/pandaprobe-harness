"""Hook-level tests for eval-case capture and (with the validation phase)
automatic candidate validation."""

from __future__ import annotations

from pathlib import Path

from pandaprobe_harness import Harness, HarnessConfig
from tests.fakes.fake_cli_client import FakeCliClient


def _failing_cli() -> FakeCliClient:
    return FakeCliClient(
        metric_values={"agent_reliability": 0.30, "agent_consistency": 0.40},
        metric_metadata={
            "agent_reliability": {"flagged_traces": ["trace-1"]},
        },
    )


def _config(tmp_path: Path, **overrides: object) -> HarnessConfig:
    return HarnessConfig(
        harness_root=tmp_path / "harness",
        poll_interval_s=0.0,
        poll_max_attempts=5,
        eval_retry_backoff_s=0.0,
        drain_timeout_s=5.0,
        **overrides,  # type: ignore[arg-type]
    )


async def test_breach_captures_replayable_eval_case(tmp_path: Path) -> None:
    cfg = _config(tmp_path, capture_eval_cases=True)
    harness = Harness.create(cfg, cli=_failing_cli())

    harness.on_turn_end(
        {"session_id": "s-1", "turn_index": 1, "end_state": {"task": "charge the payment"}}
    )
    await harness.refresh("s-1")

    (case,) = harness.evalset.cases()
    assert case.kind == "failure"
    assert case.session_id == "s-1"
    assert "breach:agent_reliability" in case.signature
    assert case.baseline_scores["agent_reliability"] == 0.3
    assert case.replay_input == {"task": "charge the payment"}
    assert case.replayable

    (event,) = harness.journal.recent(types=("evalset_capture",))
    assert event["case_id"] == case.id


async def test_capture_without_end_state_is_not_replayable(tmp_path: Path) -> None:
    cfg = _config(tmp_path, capture_eval_cases=True)
    harness = Harness.create(cfg, cli=_failing_cli())

    harness.on_turn_end({"session_id": "s-1", "turn_index": 1, "end_state": {}})
    await harness.refresh("s-1")

    (case,) = harness.evalset.cases()
    assert not case.replayable
    assert case.replay_input is None


async def test_capture_disabled_by_default(tmp_path: Path) -> None:
    harness = Harness.create(_config(tmp_path), cli=_failing_cli())

    harness.on_turn_end({"session_id": "s-1", "turn_index": 1, "end_state": {"task": "x"}})
    await harness.refresh("s-1")

    assert harness.mailbox.pending()  # the notice still posts
    assert harness.evalset.cases() == []


async def test_healthy_turns_capture_nothing(tmp_path: Path) -> None:
    cfg = _config(tmp_path, capture_eval_cases=True)
    harness = Harness.create(cfg, cli=FakeCliClient())

    harness.on_turn_end({"session_id": "s-1", "turn_index": 1, "end_state": {"task": "x"}})
    await harness.refresh("s-1")

    assert harness.evalset.cases() == []


async def test_validation_disabled_means_no_engine(tmp_path: Path) -> None:
    harness = Harness.create(_config(tmp_path, rule_validation=False), cli=FakeCliClient())
    assert await harness.validate_candidates() == []
    await harness.drain_validation()  # no-op, never raises


async def test_poisoned_validation_never_breaks_the_loop(tmp_path: Path) -> None:
    """A blown-up validation engine degrades to a log line; the report path,
    notice persistence, and refresh() all keep working."""

    cfg = _config(tmp_path, capture_eval_cases=True)
    harness = Harness.create(cfg, cli=_failing_cli())

    class _Boom:
        def observe_report(self, session_id: str, signatures: set[str]) -> None:
            raise RuntimeError("poisoned engine")

    harness.hook._validation = _Boom()  # type: ignore[assignment]

    harness.on_turn_end({"session_id": "s-1", "turn_index": 1, "end_state": {"t": 1}})
    report = await harness.refresh("s-1")

    assert report is not None  # the eval resolved despite the poison
    assert len(harness.mailbox.pending()) == 1  # the notice still posted


async def test_candidate_rule_promoted_automatically_from_live_turns(
    tmp_path: Path,
) -> None:
    """The hook cadence alone (no explicit validate_candidates call) drives a
    forward trial to promotion."""

    cfg = _config(tmp_path, rule_trial_min_sessions=2)
    harness = Harness.create(cfg, cli=FakeCliClient())
    added = await harness.toolset.call(
        "harness_rule_add",
        {"rule": "check before retrying", "rationale": "x", "metric": "agent_reliability"},
    )
    assert added["rule"]["status"] == "candidate"

    for session in ("s-a", "s-b"):
        harness.on_turn_end({"session_id": session, "turn_index": 1, "end_state": {}})
        await harness.refresh(session)
        await harness.drain_validation()

    (rule,) = harness.rules.active()
    assert rule.id == added["rule"]["id"]


async def test_advisory_trend_notice_captures_no_eval_case(tmp_path: Path) -> None:
    """Trend-severity notices are advisory — their baseline scores can sit
    ABOVE the threshold, so treating them as failure cases would pollute the
    eval-set and its proxy labels."""

    cfg = HarnessConfig(
        harness_root=tmp_path / "harness",
        poll_interval_s=0.0,
        poll_max_attempts=5,
        eval_retry_backoff_s=0.0,
        drain_timeout_s=5.0,
        capture_eval_cases=True,
        eval_consistency=False,  # isolate one declining metric
        trend_min_samples=4,
        trend_margin_cross=0.05,
    )
    cli = FakeCliClient(metric_values={"agent_reliability": 0.80})
    harness = Harness.create(cfg, cli=cli)

    session = "s-trend"
    for idx, score in enumerate((0.80, 0.74, 0.68, 0.62, 0.58, 0.55)):
        cli.set_scores(agent_reliability=score)
        harness.on_turn_end(
            {"session_id": session, "turn_index": idx, "end_state": {"task": "x"}}
        )
        await harness.refresh(session)

    (notice,) = harness.mailbox.pending()
    assert notice.severity == "trend"  # above threshold the whole way down
    assert harness.evalset.cases() == []
