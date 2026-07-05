"""Tests for the ``pandaprobe-harness-calibrate`` operator CLI (subprocess-level).

The fake ``pandaprobe`` binary only echoes argv (no scores), so these tests
seed the local stores — which doubles as coverage of the CLI-degrade path.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from pandaprobe_harness import HarnessConfig
from pandaprobe_harness.workspace.evalset import EvalSet

_MODULE = "pandaprobe_harness.calibration"


def _run(
    args: list[str], tmp_path: Path, fake_bin: Path
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "HARNESS_ROOT": str(tmp_path / "h"),
        "HARNESS_CLI_BINARY": str(fake_bin),
    }
    return subprocess.run(
        [sys.executable, "-m", _MODULE, *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def _seed_evalset(tmp_path: Path) -> None:
    config = HarnessConfig(harness_root=tmp_path / "h")
    evalset = EvalSet(config)
    evalset.capture(
        session_id="s-bad",
        signature=("breach:agent_reliability",),
        baseline_scores={"agent_reliability": 0.30, "agent_consistency": 0.40},
    )
    evalset.capture(
        session_id="s-good",
        kind="win",
        signature=("healthy",),
        baseline_scores={"agent_reliability": 0.90, "agent_consistency": 0.85},
    )


def test_no_scores_anywhere_is_error_exit_1(tmp_path: Path, fake_bin: Path) -> None:
    proc = _run([], tmp_path, fake_bin)
    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    assert "no session scores" in payload["error"]


def test_unlabeled_report_from_evalset_scores(tmp_path: Path, fake_bin: Path) -> None:
    _seed_evalset(tmp_path)
    proc = _run([], tmp_path, fake_bin)

    assert proc.returncode == 0
    assert "agent_reliability" in proc.stdout
    assert "histogram" in proc.stdout
    assert "precision" not in proc.stdout  # unlabeled


def test_from_evalset_proxy_labels_json_report(tmp_path: Path, fake_bin: Path) -> None:
    _seed_evalset(tmp_path)
    proc = _run(["--from-evalset", "--json"], tmp_path, fake_bin)

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["sources"] == ["evalset"]
    reliability = next(
        m for m in payload["metrics"] if m["metric"] == "agent_reliability"
    )
    labeled = reliability["labeled"]
    # s-bad (0.30, failed) breaches; s-good (0.90, ok) does not: perfect split.
    assert labeled["tp"] == 1 and labeled["tn"] == 1
    assert labeled["precision"] == 1.0 and labeled["recall"] == 1.0
    assert labeled["best_f1"] == 1.0


def test_explicit_labels_win_over_evalset(tmp_path: Path, fake_bin: Path) -> None:
    _seed_evalset(tmp_path)
    labels = tmp_path / "labels.json"
    # Deliberately inverted labels: the breaching session is marked ok.
    labels.write_text(json.dumps({"s-bad": False, "s-good": True}), encoding="utf-8")
    proc = _run(
        ["--labels", str(labels), "--from-evalset", "--json"], tmp_path, fake_bin
    )

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    reliability = next(
        m for m in payload["metrics"] if m["metric"] == "agent_reliability"
    )
    labeled = reliability["labeled"]
    assert labeled["tp"] == 0 and labeled["fp"] == 1 and labeled["fn"] == 1


def test_bad_labels_path_is_error_json_exit_1(tmp_path: Path, fake_bin: Path) -> None:
    _seed_evalset(tmp_path)
    proc = _run(["--labels", str(tmp_path / "missing.json")], tmp_path, fake_bin)
    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
