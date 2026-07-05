"""Tests for the ``pandaprobe-harness-eval`` operator CLI (subprocess-level)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from pandaprobe_harness import HarnessConfig
from pandaprobe_harness.workspace.evalset import EvalSet

_MODULE = "pandaprobe_harness.validation.regression"


def _run(
    args: list[str],
    tmp_path: Path,
    fake_bin: Path,
    *,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "HARNESS_ROOT": str(tmp_path / "h"),
        "HARNESS_CLI_BINARY": str(fake_bin),
        **(extra_env or {}),
    }
    return subprocess.run(
        [sys.executable, "-m", _MODULE, *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


def _seed_case(tmp_path: Path, *, replayable: bool) -> str:
    config = HarnessConfig(harness_root=tmp_path / "h")
    evalset = EvalSet(config)
    case = evalset.capture(
        session_id="s-1",
        signature=("breach:agent_reliability",),
        baseline_scores={"agent_reliability": 0.3},
        replay_input={"task": "charge"} if replayable else None,
    )
    assert case is not None
    return case.id


def test_list_cases(tmp_path: Path, fake_bin: Path) -> None:
    case_id = _seed_case(tmp_path, replayable=True)
    proc = _run(["--list"], tmp_path, fake_bin)

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert [case["id"] for case in payload["cases"]] == [case_id]
    assert payload["cases"][0]["replayable"] is True


def test_without_replay_reports_skips_and_exits_clean(
    tmp_path: Path, fake_bin: Path
) -> None:
    _seed_case(tmp_path, replayable=True)
    proc = _run([], tmp_path, fake_bin)

    assert proc.returncode == 0  # skips are honest, not regressions
    assert "skipped 1" in proc.stdout
    assert "CLEAN" in proc.stdout


def test_replay_flag_imports_and_runs(tmp_path: Path, fake_bin: Path) -> None:
    case_id = _seed_case(tmp_path, replayable=True)
    replay_dir = tmp_path / "replaymod"
    replay_dir.mkdir()
    sentinel = replay_dir / "invoked.txt"
    (replay_dir / "myreplay.py").write_text(
        "from pathlib import Path\n"
        "async def replay(case, context):\n"
        "    assert 'Harness Rules' in context\n"
        f"    Path({str(sentinel)!r}).write_text(case.id, encoding='utf-8')\n"
        "    return 's-replayed'\n",
        encoding="utf-8",
    )
    proc = _run(
        ["--replay", "myreplay:replay", "--json"],
        tmp_path,
        fake_bin,
        extra_env={"PYTHONPATH": str(replay_dir)},
    )

    # The fake binary only echoes argv, so scoring degrades and the case is
    # skipped — the sentinel proves the replay fn was imported AND invoked
    # in the subprocess (a run with replay=None would leave it absent).
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["clean"] is True
    assert sentinel.read_text(encoding="utf-8") == case_id
    (result,) = payload["results"]
    assert result["reason"].startswith("scoring") or "scores" in result["reason"]


def test_bad_replay_spec_is_error_json_exit_1(tmp_path: Path, fake_bin: Path) -> None:
    proc = _run(["--replay", "nope"], tmp_path, fake_bin)
    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    assert "MODULE" in payload["error"] or "nope" in payload["error"]
