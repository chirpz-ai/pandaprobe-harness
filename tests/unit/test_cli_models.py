from __future__ import annotations

import pytest

from pandaprobe_harness.cli.models import RunCreated, RunScores, ScoreRecord


def test_run_created_parse_variants() -> None:
    assert RunCreated.parse({"run_id": "r1", "status": "pending"}).run_id == "r1"
    # tolerate 'id' alias and missing status
    rc = RunCreated.parse({"id": "r2"})
    assert rc.run_id == "r2"
    assert rc.status == "pending"


def test_run_created_requires_id() -> None:
    with pytest.raises(ValueError):
        RunCreated.parse({"status": "pending"})


def test_score_record_terminal_and_floats() -> None:
    rec = ScoreRecord.parse(
        {"name": "agent_reliability", "value": "0.3", "status": "SUCCESS"}
    )
    assert rec.value == 0.3
    assert rec.is_terminal

    running = ScoreRecord.parse({"name": "x", "value": None, "status": "PENDING"})
    assert running.value is None
    assert not running.is_terminal


def test_score_record_failed_is_terminal_with_null_value() -> None:
    rec = ScoreRecord.parse({"name": "agent_reliability", "value": None, "status": "FAILED"})
    assert rec.is_terminal  # FAILED is terminal — stops polling
    assert rec.value is None


def test_non_numeric_value_parses_to_none() -> None:
    assert ScoreRecord.parse({"name": "m", "value": "N/A", "status": "SUCCESS"}).value is None
    assert ScoreRecord.parse({"name": "m", "value": "", "status": "SUCCESS"}).value is None


def test_run_scores_terminal_logic() -> None:
    payload = {
        "run_id": "r1",
        "scores": [
            {"name": "agent_reliability", "value": 0.3, "status": "completed"},
            {"name": "agent_consistency", "value": None, "status": "running"},
        ],
    }
    scores = RunScores.parse("r1", payload)
    assert not scores.is_terminal()  # one still running
    assert scores.by_name("agent_reliability") is not None
    assert scores.by_name("missing") is None


def test_run_scores_empty_is_not_terminal() -> None:
    assert not RunScores.parse("r1", {"scores": []}).is_terminal()


def test_run_scores_accepts_bare_list() -> None:
    scores = RunScores.parse("r1", [{"name": "m", "value": 1.0, "status": "completed"}])
    assert scores.is_terminal()
    assert scores.run_id == "r1"


def test_run_scores_tolerates_extra_fields() -> None:
    rec = ScoreRecord.parse(
        {
            "name": "agent_reliability",
            "value": 0.9,
            "status": "completed",
            "metadata": {"flagged_traces": ["t-7"]},
            "unexpected": "ignored",
        }
    )
    assert rec.metadata["flagged_traces"] == ["t-7"]
