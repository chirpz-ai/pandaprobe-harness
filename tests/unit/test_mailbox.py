"""Unit tests for the pull-model diagnostic mailbox."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from pandaprobe_harness import DiagnosticNotice, HarnessConfig, Mailbox, NoticeMetric
from pandaprobe_harness.workspace.mailbox import Resolution


def _notice(
    notice_id: str | None = None,
    *,
    severity: str = "breach",
    session_id: str = "s-1",
    turn_index: int = 1,
    **kwargs: Any,
) -> DiagnosticNotice:
    return DiagnosticNotice(
        id=notice_id or DiagnosticNotice.new_id(),
        created_at=datetime.now(UTC).isoformat(),
        session_id=session_id,
        turn_index=turn_index,
        severity=severity,  # type: ignore[arg-type]
        **kwargs,
    )


def test_post_then_pending_sorted_oldest_first_by_id(mailbox: Mailbox) -> None:
    ids = [
        "n-20260101T000000000003-cc",
        "n-20260101T000000000001-aa",
        "n-20260101T000000000002-bb",
    ]
    for nid in ids:
        mailbox.post(_notice(nid))
    assert [n.id for n in mailbox.pending()] == sorted(ids)


def test_read_finds_notice_in_pending_then_processed(mailbox: Mailbox) -> None:
    notice = _notice()
    mailbox.post(notice)

    found = mailbox.read(notice.id)
    assert found is not None and found.status == "pending"

    mailbox.acknowledge(notice.id)
    found = mailbox.read(notice.id)
    assert found is not None and found.status == "acknowledged"

    assert mailbox.read("n-does-not-exist") is None


def test_acknowledge_moves_file_and_sets_resolution(
    mailbox: Mailbox, config: HarnessConfig
) -> None:
    notice = _notice()
    mailbox.post(notice)

    acked = mailbox.acknowledge(notice.id, rule_id="r-1", note="mitigated")

    assert not (config.mailbox_pending_dir / f"{notice.id}.json").exists()
    processed_path = config.mailbox_processed_dir / f"{notice.id}.json"
    assert processed_path.exists()

    assert acked.status == "acknowledged"
    assert acked.resolution is not None
    assert acked.resolution.acked_at
    assert acked.resolution.rule_id == "r-1"
    assert acked.resolution.note == "mitigated"

    on_disk = json.loads(processed_path.read_text(encoding="utf-8"))
    assert on_disk["status"] == "acknowledged"
    assert on_disk["resolution"]["rule_id"] == "r-1"
    assert on_disk["resolution"]["note"] == "mitigated"


def test_double_acknowledge_raises_key_error(mailbox: Mailbox) -> None:
    notice = _notice()
    mailbox.post(notice)
    mailbox.acknowledge(notice.id)
    with pytest.raises(KeyError):
        mailbox.acknowledge(notice.id)


def test_status_pending_count_severity_ranking_and_latest_id(mailbox: Mailbox) -> None:
    mailbox.post(_notice("n-1", severity="trend"))
    assert mailbox.status().max_severity == "trend"

    mailbox.post(_notice("n-2", severity="relative"))
    assert mailbox.status().max_severity == "relative"

    mailbox.post(_notice("n-3", severity="breach"))
    assert mailbox.status().max_severity == "breach"

    mailbox.post(_notice("n-4", severity="needs_human"))
    status = mailbox.status()
    assert status.max_severity == "needs_human"
    assert status.pending_count == 4
    assert status.latest_id == "n-4"

    # A later, lower-severity notice does not lower the max.
    mailbox.post(_notice("n-5", severity="trend"))
    status = mailbox.status()
    assert status.max_severity == "needs_human"
    assert status.pending_count == 5
    assert status.latest_id == "n-5"


def test_status_recomputes_after_deleting_status_file(
    mailbox: Mailbox, config: HarnessConfig
) -> None:
    mailbox.post(_notice("n-1", severity="relative"))
    mailbox.post(_notice("n-2", severity="breach"))
    config.mailbox_status_file.unlink()

    status = mailbox.status()
    assert status.pending_count == 2
    assert status.max_severity == "breach"
    assert status.latest_id == "n-2"
    # The recomputed summary is persisted again.
    assert config.mailbox_status_file.exists()


def test_corrupt_pending_file_is_skipped(mailbox: Mailbox, config: HarnessConfig) -> None:
    mailbox.post(_notice("n-good"))
    (config.mailbox_pending_dir / "n-bad.json").write_text("{{{ not json", encoding="utf-8")
    assert [n.id for n in mailbox.pending()] == ["n-good"]


def test_no_tmp_residue_after_post_and_ack(mailbox: Mailbox, config: HarnessConfig) -> None:
    notice = _notice()
    mailbox.post(notice)
    mailbox.acknowledge(notice.id, rule_id="r-1")
    mailbox.status()
    assert list(config.mailbox_dir.rglob("*.tmp")) == []


def test_from_json_is_forgiving() -> None:
    notice = DiagnosticNotice.from_json({"severity": "catastrophic"})
    assert notice.severity == "breach"
    assert notice.id == ""
    assert notice.created_at == ""
    assert notice.session_id == ""
    assert notice.turn_index == 0
    assert notice.metrics == ()
    assert notice.flagged_traces == ()
    assert notice.signatures == ()
    assert notice.summary == ""
    assert notice.status == "pending"
    assert notice.resolution is None

    empty = DiagnosticNotice.from_json({})
    assert empty.severity == "breach"
    assert empty.metrics == ()


def test_to_json_from_json_round_trips_key_fields() -> None:
    original = DiagnosticNotice(
        id="n-roundtrip",
        created_at="2026-07-01T00:00:00+00:00",
        session_id="s-9",
        turn_index=3,
        severity="relative",
        metrics=(
            NoticeMetric(
                name="agent_reliability",
                value=0.4,
                threshold=0.5,
                reason="dropped below baseline",
                conditions=("relative",),
            ),
        ),
        flagged_traces=("t-1", "t-2"),
        signal_breakdown={"t-1": {"signal": "bad"}},
        dump_path="/tmp/dump.json",
        summary="reliability dropped",
        signatures=("relative:agent_reliability",),
        status="acknowledged",
        resolution=Resolution(
            acked_at="2026-07-01T01:00:00+00:00", rule_id="r-1", note="handled"
        ),
    )
    assert DiagnosticNotice.from_json(original.to_json()) == original


def test_new_id_unique_and_lexicographically_ordered() -> None:
    base = datetime(2026, 7, 1, tzinfo=UTC)
    ordered = [DiagnosticNotice.new_id(base + timedelta(milliseconds=i)) for i in range(10)]
    assert ordered == sorted(ordered)
    assert len(set(ordered)) == len(ordered)

    # Even with an identical timestamp, ids never collide.
    burst = {DiagnosticNotice.new_id(base) for _ in range(100)}
    assert len(burst) == 100
