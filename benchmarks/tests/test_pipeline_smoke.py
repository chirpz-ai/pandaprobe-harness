"""End-to-end dry-run pipeline test: run -> records -> resume -> report.

Uses the generic MockTaskRunner (no network, no external harness), which is what
`pandabench-run --smoke` exercises for real."""

from __future__ import annotations

import csv
from pathlib import Path

from pandabench.agents.harness_wiring import HarnessWiring
from pandabench.config import load_study
from pandabench.providers.litellm_client import ChatClient
from pandabench.providers.models import ResolvedModel, load_registry
from pandabench.report import aggregate
from pandabench.runners.base import BenchmarkRunner, TaskOutcome
from pandabench.runners.mock import MockTaskRunner

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class RecordingMockTaskRunner(MockTaskRunner):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.session_ids: list[str] = []

    async def run_once(
        self,
        *,
        task_id: str,
        session_id: str,
        model: ResolvedModel,
        client: ChatClient,
        max_turns: int,
        wiring: HarnessWiring | None,
        preamble: str | None = None,
    ) -> TaskOutcome:
        self.session_ids.append(session_id)
        return await super().run_once(
            task_id=task_id, session_id=session_id, model=model, client=client,
            max_turns=max_turns, wiring=wiring, preamble=preamble,
        )


def _runner(
    tmp_path: Path, single: MockTaskRunner | None = None
) -> BenchmarkRunner:
    return BenchmarkRunner(
        single=single or MockTaskRunner("appworld"),
        study=load_study(CONFIGS / "study.yaml"),
        registry=load_registry(CONFIGS / "models.yaml"),
        run_root=tmp_path / "runs",
        repo_root=tmp_path,
        lock_path=tmp_path / "uv.lock",
    )


async def test_dry_run_pipeline_and_resume(tmp_path):
    run_dir = await _runner(tmp_path).run(
        arm="baseline", model_key="gemini-3.1-flash-lite", backend=None, seed=1,
        k=1, limit=2, dry_run=True, phases=("eval",),
    )
    records_file = run_dir / "records.jsonl"
    n_first = len(records_file.read_text().splitlines())
    assert n_first == 2
    assert (run_dir / "manifest.json").exists()

    # Resume: rerun with the same run_id -> every trial is skipped, no duplicates.
    await _runner(tmp_path).run(
        arm="baseline", model_key="gemini-3.1-flash-lite", backend=None, seed=1,
        k=1, limit=2, dry_run=True, phases=("eval",), run_id=run_dir.name,
    )
    assert len(records_file.read_text().splitlines()) == n_first

    # Report aggregates the run into the summary artifacts.
    summary = tmp_path / "summary"
    aggregate(tmp_path / "runs", summary)
    assert (summary / "headline.csv").read_text().strip() != ""
    assert (summary / "report.md").exists()
    assert (summary / "all_records.csv").exists()


async def test_both_arms_dry_run_pipeline(tmp_path):
    for arm in ("baseline", "harness"):
        run_dir = await _runner(tmp_path).run(
            arm=arm, model_key="gemini-3.1-flash-lite", backend=None, seed=1,
            k=1, limit=1, dry_run=True, phases=("learning", "eval"),
        )
        assert (run_dir / "records.jsonl").exists()


async def test_repeated_setup_gets_a_fresh_remote_session_namespace(tmp_path):
    first = RecordingMockTaskRunner("appworld")
    second = RecordingMockTaskRunner("appworld")

    run_dir = await _runner(tmp_path, first).run(
        arm="baseline", model_key="gemini-3.1-flash-lite", backend=None, seed=1,
        k=1, limit=1, dry_run=True, phases=("eval",), run_id="interrupted-run",
    )
    (run_dir / "records.jsonl").unlink()
    await _runner(tmp_path, second).run(
        arm="baseline", model_key="gemini-3.1-flash-lite", backend=None, seed=1,
        k=1, limit=1, dry_run=True, phases=("eval",), run_id="interrupted-run",
    )

    assert len(first.session_ids) == len(second.session_ids) == 1
    assert first.session_ids[0] != second.session_ids[0]


async def test_report_keeps_datasets_as_separate_benchmark_cells(tmp_path):
    for dataset in ("airline", "retail"):
        await _runner(tmp_path).run(
            arm="baseline", model_key="gemini-3.1-flash-lite", backend=None, seed=1,
            k=1, limit=1, dry_run=True, phases=("eval",),
            dataset_override=dataset, run_id=f"appworld-{dataset}",
        )

    summary = tmp_path / "summary"
    aggregate(tmp_path / "runs", summary)
    with (summary / "headline.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert {row["dataset"] for row in rows} == {"airline", "retail"}
    assert len(rows) == 2
