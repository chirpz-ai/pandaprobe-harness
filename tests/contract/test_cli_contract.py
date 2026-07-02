"""Contract tests for the ``pandaprobe`` CLI payload shapes.

The offline half parses recorded fixtures (tests/contract/fixtures/) through
the same typed views the harness uses at runtime, pinning the shapes the code
depends on. The live half replays the same commands against a real CLI +
credentials; it is opt-in via ``PANDAPROBE_LIVE=1`` so the suite stays fully
offline by default.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from pandaprobe_harness.cli.models import RunCreated, RunScores, ScoreRecord
from pandaprobe_harness.cli.subprocess_client import SubprocessCliClient
from pandaprobe_harness.workspace.mailbox import DiagnosticNotice

FIXTURES = Path(__file__).parent / "fixtures"

LIVE = bool(os.environ.get("PANDAPROBE_LIVE"))

live = pytest.mark.skipif(
    not LIVE, reason="set PANDAPROBE_LIVE=1 with credentials for live contract tests"
)


def _load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# -- offline half: recorded fixtures through the typed views ---------------------


def test_run_created_parses_recorded_payload() -> None:
    payload = _load("run_created.json")
    run = RunCreated.parse(payload)
    assert run.run_id == "run-abc123"
    assert run.status == "PENDING"


def test_run_scores_terminal_parses_and_converts_values() -> None:
    payload = _load("run_scores_terminal.json")
    scores = RunScores.parse("run-x", payload)
    assert scores.run_id == "run-x"
    assert scores.is_terminal() is True

    reliability = scores.by_name("agent_reliability")
    assert reliability is not None
    # The CLI serializes values as strings; the view converts to float.
    assert reliability.value == pytest.approx(0.42)
    assert reliability.reason == "low tool correctness on trace tr-1"
    assert reliability.metadata["flagged_traces"] == ["tr-1"]
    assert reliability.metadata["per_trace_signals"]["tr-1"]["tool_correctness"] == 0.4
    assert reliability.metadata["aggregation"] == {"method": "min"}

    consistency = scores.by_name("agent_consistency")
    assert consistency is not None
    assert consistency.value == pytest.approx(0.55)
    assert consistency.reason is None
    assert consistency.metadata == {}


def test_score_record_tolerates_missing_optionals() -> None:
    # Backend session-score listings omit status/reason/metadata entirely.
    payload = _load("session_scores_list.json")
    items = payload["items"]
    assert len(items) == 3
    for item in items:
        record = ScoreRecord.parse(item)
        assert record.name in {"agent_reliability", "agent_consistency"}
        assert record.value is not None
        assert record.reason is None
        assert record.metadata == {}
    # Entirely bare records parse too, defaulting sensibly.
    bare = ScoreRecord.parse({})
    assert bare.name == ""
    assert bare.value is None
    assert bare.status == "pending"
    assert bare.is_terminal is False


def test_notice_round_trip_is_a_fixed_point() -> None:
    data = _load("notice.json")
    first = DiagnosticNotice.from_json(data)
    second = DiagnosticNotice.from_json(first.to_json())
    assert second.id == first.id == "n-20260630T121500123456-9f3a1c2b"
    assert second.severity == first.severity == "breach"
    assert second.metrics == first.metrics
    assert len(second.metrics) == 2
    assert second.resolution == first.resolution
    assert second.resolution is not None
    assert second.resolution.rule_id == "r-0001"


# -- live half: the same commands against a real CLI (opt-in) --------------------


@live
async def test_live_version_runs() -> None:
    result = await SubprocessCliClient().run("version")
    assert result.exit_code == 0


@live
async def test_live_auth_status_runs() -> None:
    # SubprocessCliClient raises a CliError subclass on non-zero exit.
    result = await SubprocessCliClient().run("auth", "status")
    assert result.exit_code == 0


@live
async def test_live_session_scores_list_parses() -> None:
    session_id = os.environ.get("PANDAPROBE_CONTRACT_SESSION")
    if not session_id:
        pytest.skip("set PANDAPROBE_CONTRACT_SESSION to a session id with scores")
    result = await SubprocessCliClient().run(
        "evals", "scores", "list", "--target", "session", "--session-id", session_id
    )
    payload = result.json()
    items = payload.get("items", []) if isinstance(payload, dict) else payload
    assert isinstance(items, list)
    for item in items:
        record = ScoreRecord.parse(item)
        assert record.name
