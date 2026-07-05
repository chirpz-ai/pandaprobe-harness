"""End-to-end closed-loop self-healing (the v0.6 default semantics).

Proves the full evidence-before-trust loop, all automatic — no human:

  breach turn -> notice posted AND the session captured as a replayable
  eval case -> the agent pulls the notice and records a rule -> the rule
  lands as a CANDIDATE (in force, provisional) -> the harness replays the
  captured failure with the candidate in context, the replayed session
  scores healthy -> the candidate is PROMOTED to active -> the promoted
  rule re-enters the system context (no provisional section, no banner) ->
  a regression run over the eval-set (a win + the fixed failure) is clean.

A second scenario covers the automatic fallback: with no replay function
wired, a candidate is promoted by the forward trial after enough healthy
live sessions, with the fallback logged exactly once.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from pandaprobe_harness import Harness, HarnessConfig
from pandaprobe_harness.workspace.evalset import EvalCase
from pandaprobe_harness.workspace.rules import PROVISIONAL_HEADING
from tests.fakes.fake_cli_client import FakeCliClient
from tests.fakes.mock_agent import MITIGATION_RULE, MockLLMAgent

SESSION = "s-closed-1"


def _failing_cli() -> FakeCliClient:
    return FakeCliClient(
        metric_values={"agent_reliability": 0.30, "agent_consistency": 0.40},
        metric_metadata={
            "agent_reliability": {
                "flagged_traces": ["trace-1"],
                "per_trace_signals": {
                    "trace-1": {"loop_detection": 0.1, "tool_correctness": 0.2}
                },
            }
        },
    )


def _config(tmp_path: Path, **overrides: Any) -> HarnessConfig:
    return HarnessConfig(
        harness_root=tmp_path / "harness",
        poll_interval_s=0.0,
        poll_max_attempts=5,
        eval_retry_backoff_s=0.0,
        drain_timeout_s=5.0,
        capture_eval_cases=True,
        **overrides,
    )


async def test_closed_loop_replay_validates_promotes_and_stays_regression_clean(
    tmp_path: Path,
) -> None:
    cfg = _config(tmp_path)
    cli = _failing_cli()
    replay_calls: list[str] = []
    replay_contexts: list[str] = []

    async def replay(case: EvalCase, context: str) -> str:
        # Called for candidate validation (rule provisional) and again by the
        # regression run (rule promoted) — the rule is in force both times.
        assert MITIGATION_RULE[:30] in context
        replay_calls.append(case.id)
        replay_contexts.append(context)
        return "s-replay-1"

    harness = Harness.create(cfg, cli=cli, replay=replay)
    agent = MockLLMAgent(session_id=SESSION, toolset=harness.toolset)

    async def run_turn() -> None:
        raw = await agent.take_turn()
        if agent.healed:
            cli.set_scores(agent_reliability=0.92, agent_consistency=0.88)
        harness.on_turn_end(raw)
        await harness.refresh(SESSION)
        await harness.drain_validation()

    # --- Turn 1: the failure. Notice + replayable eval case. ----------------
    await run_turn()
    assert len(harness.mailbox.pending()) == 1
    (case,) = harness.evalset.cases()
    assert case.kind == "failure"
    assert "breach:agent_reliability" in case.signature
    assert case.replayable  # replay input captured from the turn's end_state
    assert case.baseline_scores["agent_reliability"] == 0.30

    # The replayed session will score healthy (the rule works).
    cli.set_session_scores(
        "s-replay-1", agent_reliability=0.92, agent_consistency=0.88
    )

    # --- Turn 2: agent records the rule -> candidate -> replay -> promoted --
    await run_turn()
    assert replay_calls == [case.id], "the candidate was validated by replay"
    # During validation the candidate was IN FORCE via the provisional section.
    assert PROVISIONAL_HEADING in replay_contexts[0]

    (rule,) = harness.rules.active()
    assert rule.id == agent.rule_ids[0]
    assert "breach:agent_reliability" in rule.tags  # derived from the notice
    assert harness.rules.candidates() == []

    # The agent saw the candidate state right after acking (pre-validation).
    assert agent.rule_status is not None
    assert agent.rule_status["status"] == "candidate"

    # The journal shows the full lifecycle, in order.
    events = harness.journal.recent(types=("notice", "rule_add", "rule_promote"))
    assert [e["type"] for e in events] == ["notice", "rule_add", "rule_promote"]
    assert events[1]["status"] == "candidate"
    assert events[2]["validator"] == "replay"
    assert "improved" in events[2]["reason"]

    # The promoted rule re-enters context as a validated rule; banner cleared.
    # Retrieval is on by default: a task hint matching the rule's derived tags
    # (breach:agent_reliability, payment vocabulary) keeps it in the top-k.
    context = harness.system_context(task_hint="charge a customer payment")
    assert "⚠ HARNESS" not in context
    assert MITIGATION_RULE[:30] in context
    assert PROVISIONAL_HEADING not in context

    # --- Turns 3-4: corrected behaviour, nothing new posts. -----------------
    await run_turn()
    await run_turn()
    assert harness.mailbox.pending() == []

    # --- Regression guard: protect the win, confirm the fix. ----------------
    win = harness.evalset.capture(
        session_id=SESSION,
        kind="win",
        signature=("healthy",),
        baseline_scores={"agent_reliability": 0.92, "agent_consistency": 0.88},
        replay_input={"action": "verified_payment_then_charge"},
    )
    assert win is not None

    report = await harness.run_regression()
    assert report.clean
    statuses = {result.case_id: result.status for result in report.results}
    assert statuses[case.id] == "improved"  # the old failure is fixed
    assert statuses[win.id] == "unchanged"  # the win is intact

    (regression_event,) = harness.journal.recent(types=("regression",))
    assert regression_event["clean"] is True
    assert regression_event["improved"] == 1


async def test_closed_loop_forward_trial_fallback_without_replay(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    cfg = _config(tmp_path, rule_trial_min_sessions=3)
    cli = FakeCliClient()  # healthy scores throughout the trial
    harness = Harness.create(cfg, cli=cli)  # no replay function wired

    added = await harness.toolset.call(
        "harness_rule_add",
        {
            "rule": "Verify the transaction status before retrying a charge.",
            "rationale": "reliability breaches from blind retries",
            "metric": "agent_reliability",
        },
    )
    rule_id = added["rule"]["id"]
    assert added["rule"]["status"] == "candidate"

    with caplog.at_level(logging.WARNING, logger="pandaprobe_harness.validation"):
        for idx in range(1, 4):
            session = f"s-trial-{idx}"
            harness.on_turn_end({"session_id": session, "turn_index": 1, "end_state": {}})
            await harness.refresh(session)
            await harness.drain_validation()

    status = await harness.toolset.call("harness_rule_status", {"rule_id": rule_id})
    assert status["ok"] is True
    assert status["lifecycle"]["status"] == "active"
    assert status["lifecycle"]["trial_rate"] == 0.0
    assert status["lifecycle"]["verdict"].startswith("promoted:")

    (promote_event,) = harness.journal.recent(types=("rule_promote",))
    assert promote_event["validator"] == "forward_trial"

    # The replay-less fallback is loud exactly once — never silent, never spam.
    fallback_logs = [
        record for record in caplog.records if "no replay function wired" in record.message
    ]
    assert len(fallback_logs) == 1
    (fallback_event,) = harness.journal.recent(types=("validation",))
    assert fallback_event["mode"] == "forward_trial"
