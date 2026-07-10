"""Checkpoint 1 tooling: metric<->failure calibration.

Turns a run's benchmark pass/fail records into the ``pandaprobe-harness-calibrate``
label format, points the CLI at that run's archived harness workspace, and
records precision/recall/F1 + the recommended threshold into IMPLEMENTATION_NOTES.
The harness metrics were designed for production sessions; this verifies they
actually correlate with benchmark task failure before the full matrix is run
(docs/benchmark-study-brief.md §5.1).
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .harness_glue import make_session_id

logger = logging.getLogger("pandabench.checkpoints")

__all__ = ["records_to_labels", "run_calibration"]


def records_to_labels(
    records_path: Path, out_path: Path, *, benchmark: str, phase: str = "learning",
    arm: str = "harness",
) -> int:
    """Write a calibrate label file from records: ``failed = not passed``.

    Session ids are recomputed with the SAME ``make_session_id`` used at trace
    time, so the labels join to the platform's session scores.
    """

    labels: list[dict[str, Any]] = []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("benchmark") != benchmark or rec.get("phase") != phase or rec.get("arm") != arm:
            continue
        session_id = make_session_id(
            benchmark=rec["benchmark"], task_id=rec["task_id"], arm=rec["arm"],
            model_key=rec["model"], seed=rec["seed"], trial=rec["trial"],
        )
        labels.append({"session_id": session_id, "failed": not bool(rec["passed"])})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return len(labels)


def _latest_harness_run(runs_dir: Path, benchmark: str) -> Path | None:
    """Most recent run dir for this benchmark's harness arm with an archived workspace."""

    candidates: list[tuple[str, Path]] = []
    for manifest in runs_dir.glob("*/manifest.json"):
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if data.get("benchmark") == benchmark and data.get("arm") == "harness":
            candidates.append((str(data.get("started_at", "")), manifest.parent))
    if not candidates:
        return None
    return max(candidates, key=lambda c: c[0])[1]


def run_calibration(benchmark: str, runs_dir: Path) -> int:
    """Checkpoint 1 for a benchmark; append the report to IMPLEMENTATION_NOTES.md."""

    run_dir = _latest_harness_run(runs_dir, benchmark)
    if run_dir is None:
        logger.error("no harness run found for %s under %s", benchmark, runs_dir)
        return 1
    records = run_dir / "records.jsonl"
    workspace = run_dir / "harness"  # archived HARNESS_ROOT
    if not workspace.exists():
        logger.error("no archived harness workspace at %s (a real arm-B run is needed)", workspace)
        return 1

    labels_path = run_dir / "calibration_labels.json"
    n = records_to_labels(records, labels_path, benchmark=benchmark)
    logger.info("wrote %d labels to %s", n, labels_path)

    import os

    env = {**os.environ, "HARNESS_ROOT": str(workspace)}
    proc = subprocess.run(
        ["pandaprobe-harness-calibrate", "--labels", str(labels_path), "--json"],
        capture_output=True, text=True, env=env, check=False,
    )
    report = _parse(proc.stdout, proc.stderr, proc.returncode)
    _append_notes(benchmark, run_dir, report)
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok", False) else 1


def _parse(stdout: str, stderr: str, code: int) -> dict[str, Any]:
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": (stderr or stdout or "no output").strip()[:500]}
    if not isinstance(parsed, dict):
        return {"ok": False, "error": "unexpected calibrate output (not an object)"}
    result: dict[str, Any] = dict(parsed)
    result.setdefault("ok", code == 0)
    return result


def _append_notes(benchmark: str, run_dir: Path, report: dict[str, Any]) -> None:
    notes = run_dir.parents[2] / "IMPLEMENTATION_NOTES.md"
    stamp = datetime.now(UTC).isoformat()
    block = [
        f"\n### Checkpoint 1 — {benchmark} calibration ({stamp})",
        f"- run: `{run_dir.name}`",
        f"- ok: {report.get('ok')}",
    ]
    if report.get("ok"):
        for metric in report.get("metrics", []):
            labeled = metric.get("labeled") or {}
            block.append(
                f"- {metric.get('metric')}: precision={labeled.get('precision')} "
                f"recall={labeled.get('recall')} f1={labeled.get('f1')} "
                f"best_f1_threshold={labeled.get('best_f1_threshold')}"
            )
    else:
        block.append(f"- error: {report.get('error')}")
    try:
        with notes.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(block) + "\n")
    except OSError as exc:
        logger.warning("could not append to %s: %s", notes, exc)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run_calibration(sys.argv[1], Path(sys.argv[2])))
