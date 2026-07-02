"""Scale scenario: many breaching sessions, bounded concurrency, clean state.

Eight sessions each drive five breaching turn-rounds through one harness.
The global semaphore must cap concurrent CLI work, per-session dedup must
collapse five breaching rounds into one notice per session, and every shared
workspace artifact (journal, status.json, score history) must stay coherent
with no orphaned atomic-write temp files.
"""

from __future__ import annotations

import json
from pathlib import Path

from pandaprobe_harness import Harness, HarnessConfig, ScoreHistoryStore
from tests.fakes.fake_cli_client import FakeCliClient

SESSIONS = [f"s-{i}" for i in range(8)]
ROUNDS = 5


async def test_eight_sessions_bounded_concurrency_and_deduped_notices(
    tmp_path: Path,
) -> None:
    cfg = HarnessConfig(
        harness_root=tmp_path / "harness",
        poll_interval_s=0.0,
        poll_max_attempts=5,
        eval_retry_backoff_s=0.0,
        max_concurrent_evals=3,
        # All 8 sessions breach at once; disable the global circuit breaker so
        # each session's single notice reaches the mailbox (dedup is under test).
        circuit_breaker_max_notices=0,
    )
    cli = FakeCliClient(
        latency_s=0.01,
        metric_values={"agent_reliability": 0.2, "agent_consistency": 0.2},
    )
    harness = Harness.create(cfg, cli=cli)

    for round_index in range(ROUNDS):
        for session in SESSIONS:
            harness.on_turn_end(
                {"session_id": session, "turn_index": round_index, "end_state": {}}
            )
        await harness.refresh_all()

    # The global semaphore bounds concurrent CLI work across all sessions.
    assert cli.max_inflight <= 3

    # Dedup: exactly one breach notice per session despite 5 breaching rounds.
    pending = harness.mailbox.pending()
    assert len(pending) == 8
    assert {notice.session_id for notice in pending} == set(SESSIONS)
    assert all(notice.severity == "breach" for notice in pending)

    # Every journal line written under concurrency is a parseable typed event.
    events = harness.journal.recent(limit=0)
    assert events, "the journal recorded events"
    assert all(isinstance(event, dict) and "type" in event for event in events)
    raw_lines = [
        line
        for line in cfg.journal_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(raw_lines) == len(events)  # no corrupt/interleaved lines were skipped
    assert sum(1 for event in events if event.get("type") == "notice") == 8

    # status.json stays an accurate always-current summary.
    status = json.loads(cfg.mailbox_status_file.read_text(encoding="utf-8"))
    assert status["pending_count"] == 8

    # Trend history recorded scores for every session.
    history = ScoreHistoryStore(cfg)
    for session in SESSIONS:
        assert history.values(session, "agent_reliability")

    # Atomic writes: no orphaned temp files anywhere under the workspace root.
    assert list(cfg.harness_root.rglob("*.tmp")) == []
