"""End-to-end pull-based self-healing scenario (v0.5-compat mode).

Simulates an agent stuck in an infinite-repetition / inconsistent-session
failure and verifies the full pull loop:

  failure turn -> eval resolves -> hook posts a diagnostic notice (NO
  injection) -> the system context shows the mailbox banner -> the agent
  pulls the notice with its harness toolset, inspects the flagged trace,
  records a structured rule with provenance, acknowledges the notice ->
  the banner clears -> subsequent turns pass and post nothing further ->
  the journal records the whole cycle, across runs.

These tests run with ``rule_validation=False`` / ``rule_retrieval=False`` —
the explicit v0.5-compatibility switches — so a rule enters ``active`` the
moment it is written. The default closed-loop behavior (candidate rules,
automatic validation, retrieval) is covered by ``e2e_closed_loop_test.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

from pandaprobe_harness import Harness, HarnessConfig
from tests.fakes.fake_cli_client import FakeCliClient
from tests.fakes.mock_agent import MITIGATION_RULE, MockLLMAgent

SESSION = "s-e2e-1"


def _compat_config(tmp_path: Path) -> HarnessConfig:
    """The v0.5-equivalent switches under test: add() -> active immediately."""

    return HarnessConfig(
        harness_root=tmp_path / "harness",
        poll_interval_s=0.0,
        poll_max_attempts=5,
        drain_timeout_s=5.0,
        rule_validation=False,
        rule_retrieval=False,
        gate_window=1,
    )


def _failing_cli() -> FakeCliClient:
    return FakeCliClient()


async def test_pull_loop_self_heals_and_converges(tmp_path: Path) -> None:
    config = _compat_config(tmp_path)
    cli = _failing_cli()
    harness = Harness.create(config, cli=cli)
    agent = MockLLMAgent(session_id=SESSION, toolset=harness.toolset)

    async def run_turn() -> None:
        raw = await agent.take_turn()
        scores = (
            {
                "task_completion": 0.92,
                "coherence": 0.88,
                "tool_correctness": 0.92,
                "argument_correctness": 0.88,
            }
            if agent.healed
            else {
                "task_completion": 0.30,
                "coherence": 0.40,
                "tool_correctness": 0.20,
                "argument_correctness": 0.20,
            }
        )
        cli.script_trace(SESSION, **scores)
        if len(cli.session_traces[SESSION]) == 1:
            cli.script_trace(SESSION, **scores)
        harness.on_turn_end(raw)
        await harness.refresh(SESSION)

    # --- Turn 1: the failure (identical repeated tool call) -----------------
    await run_turn()
    assert agent.actions == ["charge_payment"]

    # The eval resolved and the notice was posted with NO injection and NO
    # drain barrier — it is simply waiting in the mailbox.
    pending = harness.mailbox.pending()
    assert len(pending) == 1
    notice = pending[0]
    assert notice.severity == "breach"
    assert notice.flagged_traces == (f"{SESSION}-tr2",)
    assert notice.signal_breakdown[f"{SESSION}-tr2"] == {
        "task_completion": 0.3,
        "coherence": 0.4,
        "tool_correctness": 0.2,
        "argument_correctness": 0.2,
    }
    assert os.path.exists(notice.dump_path)
    assert config.latest_eval_file.exists()

    # The always-loaded system context now carries the banner + protocol.
    context = harness.system_context()
    assert "⚠ HARNESS: 1 pending diagnostic notice(s)" in context
    assert "max severity: breach" in context
    assert "harness_mailbox_list" in context

    # --- Turn 2: the agent pulls, diagnoses, records a rule, acknowledges ---
    await run_turn()
    assert agent.actions[-1] == "diagnose"

    ops = [name for name, _ in agent.tool_calls]
    assert ops[:2] == ["harness_mailbox_list", "harness_mailbox_list"] or ops[0] == (
        "harness_mailbox_list"
    )
    for expected in (
        "harness_mailbox_read",
        "harness_trace_inspect",
        "harness_journal",
        "harness_rule_add",
        "harness_mailbox_ack",
    ):
        assert expected in ops

    # The notice moved pending -> processed with its resolution.
    assert harness.mailbox.pending() == []
    processed = harness.mailbox.read(notice.id)
    assert processed is not None and processed.status == "acknowledged"
    assert processed.resolution is not None
    assert processed.resolution.rule_id == agent.rule_ids[0]

    # The structured rule carries provenance and reaches the rendered rules.
    # With rule_validation=False (the compat switch) it is active immediately.
    active = harness.rules.active()
    assert len(active) == 1
    assert active[0].status == "active"
    assert active[0].source_notice_id == notice.id
    assert active[0].metric in {"task_completion", "coherence"}
    # The rule body lands in its scope file; the skill root only indexes it.
    scope_file = config.rules_scope_file(active[0].scope)
    assert MITIGATION_RULE[:30] in scope_file.read_text(encoding="utf-8")

    # The banner cleared, and the learned rule is back within the agent's reach:
    # named in the References and one `harness_rules_read` away. That pull — not
    # an injection into the prompt — is what closes the loop in v2.
    context = harness.system_context()
    assert "⚠ HARNESS" not in context
    assert f"rules/{active[0].scope}.md" in context
    assert MITIGATION_RULE[:30] not in context
    fetched = await harness.toolset.call("harness_rules_read", {"scope": active[0].scope})
    assert MITIGATION_RULE[:30] in fetched["content"]

    # --- Turns 3-4: corrected behaviour, no further notices -----------------
    await run_turn()
    assert agent.actions[-1] == "verified_payment_then_charge"
    await run_turn()
    assert harness.mailbox.pending() == []

    # --- Convergence: the journal recorded the full cycle, exactly once -----
    notices = harness.journal.recent(types=("notice",))
    assert len(notices) == 1
    assert [e["type"] for e in harness.journal.recent(types=("rule_add",))] == ["rule_add"]
    assert [e["type"] for e in harness.journal.recent(types=("ack",))] == ["ack"]
    assert len(harness.journal.recent(types=("recovery",))) == 1
    assert agent.healed


async def test_tier_one_stall_posts_single_advisory_notice(tmp_path: Path) -> None:
    """A Tier-1 stall posts one advisory notice without a Tier-2 breach."""

    cfg = HarnessConfig(
        harness_root=tmp_path / "harness",
        poll_interval_s=0.0,
        poll_max_attempts=5,
        eval_retry_backoff_s=0.0,
        gate_window=2,
    )
    cli = FakeCliClient()
    cli.script_trajectory("s-stall", "task_completion", (0.30, 0.30, 0.30))
    harness = Harness.create(cfg, cli=cli)

    session = "s-stall"
    harness.on_turn_end({"session_id": session, "turn_index": 1, "end_state": {}})
    await harness.refresh(session)
    cli.script_trace(session, task_completion=0.30)
    harness.on_turn_end({"session_id": session, "turn_index": 2, "end_state": {}})
    await harness.refresh(session)

    pending = harness.mailbox.pending()
    assert len(pending) == 1, "exactly one notice despite a persistent stall"
    assert pending[0].severity == "trend"


async def test_rules_and_journal_persist_across_runs(tmp_path: Path) -> None:
    """Cross-run memory: a second harness over the same workspace sees the
    learned rules in its context, the journal spanning runs, and rule
    effectiveness computed against pre-rule notices."""

    config = _compat_config(tmp_path)
    # Run 1: breach -> self-heal.
    cli = _failing_cli()
    harness1 = Harness.create(config, cli=cli)
    agent = MockLLMAgent(session_id=SESSION, toolset=harness1.toolset)
    for _ in range(2):
        raw = await agent.take_turn()
        scores = (
            {
                "task_completion": 0.92,
                "coherence": 0.88,
                "tool_correctness": 0.92,
                "argument_correctness": 0.88,
            }
            if agent.healed
            else {
                "task_completion": 0.30,
                "coherence": 0.40,
                "tool_correctness": 0.20,
                "argument_correctness": 0.20,
            }
        )
        cli.script_trace(SESSION, **scores)
        if len(cli.session_traces[SESSION]) == 1:
            cli.script_trace(SESSION, **scores)
        harness1.on_turn_end(raw)
        await harness1.refresh(SESSION)
    assert agent.healed

    # Run 2: a fresh process over the same workspace. The learned rule survives
    # and is still reachable — the References index is regenerated from the store,
    # so a new process rediscovers the rule files without being told about them.
    harness2 = Harness.create(config, cli=FakeCliClient())
    assert "rules/scoped.md" in harness2.system_context()
    fetched = await harness2.toolset.call("harness_rules_read", {"scope": "scoped"})
    assert "payment tool twice" in fetched["content"]

    events = harness2.journal.recent(types=("notice", "rule_add", "ack"))
    assert {e["type"] for e in events} == {"notice", "rule_add", "ack"}

    effectiveness = harness2.rules.effectiveness()
    rule_id = harness2.rules.active()[0].id
    assert effectiveness[rule_id]["notices_before"] >= 1
    assert effectiveness[rule_id]["notices_after"] == 0


def test_fake_binary_is_executable() -> None:
    fake = Path(__file__).parent / "bin" / "fake_pandaprobe"
    assert fake.exists()
    assert os.access(fake, os.X_OK), "fake CLI must carry the executable bit"
    first_line = fake.read_text(encoding="utf-8").splitlines()[0]
    assert first_line.startswith("#!") and "python" in first_line
