"""End-to-end dry-run pipeline test: run -> records -> resume -> report.

Uses the generic MockTaskRunner (no network, no external harness), which is what
`pandabench-run --smoke` exercises for real."""

from __future__ import annotations

from pathlib import Path

from pandabench.config import load_study
from pandabench.providers.models import load_registry
from pandabench.report import aggregate
from pandabench.runners.base import BenchmarkRunner
from pandabench.runners.mock import MockTaskRunner

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _runner(tmp_path: Path) -> BenchmarkRunner:
    return BenchmarkRunner(
        single=MockTaskRunner("appworld"),
        study=load_study(CONFIGS / "study.yaml"),
        registry=load_registry(CONFIGS / "models.yaml"),
        run_root=tmp_path / "runs",
        repo_root=tmp_path,
        lock_path=tmp_path / "uv.lock",
    )


async def test_dry_run_pipeline_and_resume(tmp_path):
    run_dir = await _runner(tmp_path).run(
        arm="baseline", model_key="gemini-2.5-flash", backend=None, seed=1,
        k=1, limit=2, dry_run=True, phases=("eval",),
    )
    records_file = run_dir / "records.jsonl"
    n_first = len(records_file.read_text().splitlines())
    assert n_first == 2
    assert (run_dir / "manifest.json").exists()

    # Resume: rerun with the same run_id -> every trial is skipped, no duplicates.
    await _runner(tmp_path).run(
        arm="baseline", model_key="gemini-2.5-flash", backend=None, seed=1,
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
            arm=arm, model_key="gemini-2.5-flash", backend=None, seed=1,
            k=1, limit=1, dry_run=True, phases=("learning", "eval"),
        )
        assert (run_dir / "records.jsonl").exists()
