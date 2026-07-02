"""Unit tests for the global notice circuit breaker.

Per-session dedup means each *new* breaching session posts one notice; the
breaker counts notices globally. When the cap is hit within the window it
escalates once to ``needs_human``, then suppresses further notices until a
recovery resets it.
"""

from __future__ import annotations

from pathlib import Path

from pandaprobe_harness import HarnessConfig, HarnessFilesystem, RawLoopAdapter
from pandaprobe_harness.hook.core import PandaHarnessHook
from tests.fakes.fake_cli_client import FakeCliClient

BREACHING = {"agent_reliability": 0.2, "agent_consistency": 0.2}
HEALTHY = {"agent_reliability": 0.9, "agent_consistency": 0.9}


def _make(tmp_path: Path, cli: FakeCliClient) -> PandaHarnessHook:
    cfg = HarnessConfig(
        harness_root=tmp_path / "h",
        poll_interval_s=0.0,
        poll_max_attempts=5,
        eval_retry_backoff_s=0.0,
        enable_trend=False,
        alert_cooldown_turns=0,
        circuit_breaker_max_notices=3,
        circuit_breaker_window_s=600.0,
    )
    fs = HarnessFilesystem(cfg)
    fs.provision()
    return PandaHarnessHook(cli, config=cfg, filesystem=fs)


async def _drive(hook: PandaHarnessHook, session_id: str, turn_index: int = 1) -> None:
    hook.on_turn_end(RawLoopAdapter.make_turn(session_id, turn_index))
    await hook.refresh(session_id)


async def test_breaker_escalates_suppresses_and_resets_on_recovery(
    tmp_path: Path,
) -> None:
    cli = FakeCliClient(metric_values=dict(BREACHING))
    hook = _make(tmp_path, cli)

    # --- Below the cap: each new session posts one plain breach notice. ------
    for sid in ("s1", "s2", "s3"):
        await _drive(hook, sid)
    pending = hook.mailbox.pending()
    assert len(pending) == 3
    assert all(notice.severity == "breach" for notice in pending)

    # --- Cap hit: exactly one more notice, escalated to needs_human. ---------
    await _drive(hook, "s4")
    pending = hook.mailbox.pending()
    assert len(pending) == 4
    escalations = [n for n in pending if n.severity == "needs_human"]
    assert len(escalations) == 1
    assert escalations[0].session_id == "s4"
    assert "rate" in escalations[0].summary
    assert len(hook.journal.recent(types=("notice",))) == 4

    # --- Tripped: further breaching sessions post nothing at all. ------------
    await _drive(hook, "s5")
    assert len(hook.mailbox.pending()) == 4
    assert len(hook.journal.recent(types=("notice",))) == 4

    # --- Recovery on a previously-alerting session resets the breaker. -------
    cli.set_scores(**HEALTHY)
    await _drive(hook, "s1", turn_index=2)
    recoveries = hook.journal.recent(types=("recovery",))
    assert len(recoveries) == 1
    assert recoveries[0]["session_id"] == "s1"
    assert len(hook.mailbox.pending()) == 4  # recovery itself posts nothing

    # --- Untripped: a fresh breach posts a plain notice again. ---------------
    cli.set_scores(**BREACHING)
    await _drive(hook, "s6")
    pending = hook.mailbox.pending()
    assert len(pending) == 5
    fresh = [n for n in pending if n.session_id == "s6"]
    assert len(fresh) == 1
    assert fresh[0].severity == "breach"
    assert len(hook.journal.recent(types=("notice",))) == 5
