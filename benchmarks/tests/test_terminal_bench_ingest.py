"""Offline coverage for Harbor result ingestion and resume ordinals."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pandabench.providers.models import load_registry
from pandabench.results import RecordWriter, TrialRecord
from pandabench.runners.terminal_bench import (
    _error_record,
    _harbor_argv,
    _ingest_results,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _payload(
    *,
    task: str,
    trial_name: str,
    started_at: str,
    rewards: dict[str, float],
    turns: int,
    cost: float,
    exception: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "task_name": task,
        "trial_name": trial_name,
        "started_at": started_at,
        "finished_at": "2026-07-30T12:00:10+00:00",
        "verifier_result": {"rewards": rewards},
        "agent_result": {
            "n_input_tokens": 100 + turns,
            "n_output_tokens": 20 + turns,
            "cost_usd": cost,
            "metadata": {
                "arm": "harness",
                "seed": 3,
                "turns": turns,
                "stopped_reason": "final",
                "harness": {
                    "session_id": f"session-{trial_name}",
                    "reliability": None,
                    "consistency": None,
                    "breached": True,
                    "rules_active": 1,
                    "rules_candidate": 0,
                    "rules_retired": 0,
                    "notices": 2,
                    "scores": {"task_completion": 0.25},
                    "gate_breached": True,
                },
            },
        },
        "exception_info": exception,
    }


def _write_result(job_dir: Path, dirname: str, payload: dict[str, Any]) -> None:
    target = job_dir / dirname
    target.mkdir(parents=True)
    (target / "result.json").write_text(json.dumps(payload), encoding="utf-8")


def test_ingest_rewards_ordinals_usage_and_harness_metadata(tmp_path):
    model = load_registry(CONFIGS / "models.yaml").resolve("mock")
    job_dir = tmp_path / "raw" / "learning"
    # Directory names are intentionally opposite timestamp order. Harbor's suffix
    # is random, so started_at must assign stable attempt ordinals.
    _write_result(
        job_dir,
        "task-a__zzzzzzz",
        _payload(
            task="task-a",
            trial_name="task-a__zzzzzzz",
            started_at="2026-07-30T12:00:02+00:00",
            rewards={"reward": 1.0},
            turns=7,
            cost=0.40,
        ),
    )
    _write_result(
        job_dir,
        "task-a__aaaaaaa",
        _payload(
            task="task-a",
            trial_name="task-a__aaaaaaa",
            started_at="2026-07-30T12:00:01+00:00",
            rewards={"score": 0.0},
            turns=3,
            cost=0.20,
        ),
    )
    records_path = tmp_path / "records.jsonl"
    writer = RecordWriter(records_path)

    seen = _ingest_results(
        job_dir=job_dir,
        tasks=["task-a"],
        k=2,
        arm="harness",
        model=model,
        phase="learning",
        writer=writer,
        run_id="run-1",
        seed=3,
        benchmark="terminal_bench",
    )

    records = [
        TrialRecord.from_json(json.loads(line))
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    assert seen == {("task-a", 0), ("task-a", 1)}
    assert [record.trial for record in records] == [0, 1]
    assert records[0].native_metrics["rewards"] == {"score": 0.0}
    assert records[0].native_metrics["reward"] == 0.0  # first-value fallback
    assert records[0].passed is False
    assert records[0].turns == 3
    assert records[0].usage["cost_usd"] == pytest.approx(0.20)
    assert records[1].native_metrics["reward"] == 1.0
    assert records[1].passed is True
    assert records[1].turns == 7
    assert records[1].harness is not None
    assert records[1].harness["scores"] == {"task_completion": 0.25}


def test_ingest_exception_and_synthetic_error_record(tmp_path):
    model = load_registry(CONFIGS / "models.yaml").resolve("mock")
    job_dir = tmp_path / "raw" / "eval"
    _write_result(
        job_dir,
        "task-b__aaaaaaa",
        _payload(
            task="task-b",
            trial_name="task-b__aaaaaaa",
            started_at="2026-07-30T12:00:01+00:00",
            rewards={},
            turns=1,
            cost=0.01,
            exception={
                "exception_type": "RuntimeError",
                "exception_message": "container failed",
            },
        ),
    )
    records_path = tmp_path / "records.jsonl"
    writer = RecordWriter(records_path)
    _ingest_results(
        job_dir=job_dir,
        tasks=["task-b"],
        k=1,
        arm="harness",
        model=model,
        phase="eval",
        writer=writer,
        run_id="run-2",
        seed=3,
        benchmark="terminal_bench",
    )
    record = TrialRecord.from_json(json.loads(records_path.read_text(encoding="utf-8")))
    assert record.error == "RuntimeError: container failed"

    missing = _error_record(
        message="Harbor exited with status 2",
        run_id="run-2",
        benchmark="terminal_bench",
        task_id="task-c",
        arm="baseline",
        model=model,
        seed=3,
        trial=0,
        phase="eval",
        returncode=2,
    )
    assert missing.passed is False
    assert missing.error == "Harbor exited with status 2"
    assert missing.native_metrics == {"harbor_exit_code": 2}


def test_harbor_argv_is_serial_noninteractive_and_forwards_ablation(tmp_path):
    model = load_registry(CONFIGS / "models.yaml").resolve("mock")
    argv = _harbor_argv(
        dataset="terminal-bench-sample@2.0",
        tasks=["task-a", "task-b"],
        k=2,
        arm="harness",
        model=model,
        phase="learning",
        raw_dir=tmp_path / "raw",
        seed=3,
        backend=None,
        harness_root=tmp_path / "harness_root",
        max_turns=50,
        noval=True,
    )

    assert Path(argv[0]).name == "harbor"
    assert argv[1] == "run"
    assert argv[argv.index("-d") + 1] == "terminal-bench-sample@2.0"
    assert argv[argv.index("-n") + 1] == "1"
    assert "-y" in argv
    assert "noval=true" in argv
    assert "capture=true" in argv
    assert [argv[index + 1] for index, value in enumerate(argv) if value == "-i"] == [
        "task-a",
        "task-b",
    ]
