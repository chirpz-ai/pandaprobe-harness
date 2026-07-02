"""Tests for the companion CLI (``python -m pandaprobe_harness.agent_tools``)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from pandaprobe_harness.agent_tools.companion import _parse_args, _usage
from pandaprobe_harness.agent_tools.toolset import OP_SCHEMAS


def _run(
    args: list[str], tmp_path: Path, fake_bin: Path
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "HARNESS_ROOT": str(tmp_path / "h"),
        "HARNESS_CLI_BINARY": str(fake_bin),
    }
    return subprocess.run(
        [sys.executable, "-m", "pandaprobe_harness.agent_tools", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )


# -- process-level behaviour --------------------------------------------------------


def test_no_args_prints_usage_with_all_ops(tmp_path: Path, fake_bin: Path) -> None:
    proc = _run([], tmp_path, fake_bin)

    assert proc.returncode == 0
    assert "usage:" in proc.stdout
    assert len(OP_SCHEMAS) == 9
    for name in OP_SCHEMAS:
        assert name.startswith("harness_")
        assert name in proc.stdout


def test_unknown_op_is_error_json_exit_1(tmp_path: Path, fake_bin: Path) -> None:
    proc = _run(["harness_bogus"], tmp_path, fake_bin)

    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    assert "harness_bogus" in payload["error"]


def test_mailbox_list_over_env_workspace(tmp_path: Path, fake_bin: Path) -> None:
    proc = _run(["harness_mailbox_list"], tmp_path, fake_bin)

    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["pending"] == []
    assert payload["status"]["pending_count"] == 0


def test_rule_add_is_idempotent_across_invocations(
    tmp_path: Path, fake_bin: Path
) -> None:
    args = [
        "harness_rule_add",
        "--rule",
        "never repeat a failing call",
        "--rationale",
        "demo",
    ]

    first = _run(args, tmp_path, fake_bin)
    assert first.returncode == 0
    rules_file = tmp_path / "h" / "rules.jsonl"
    assert rules_file.exists()

    second = _run(args, tmp_path, fake_bin)
    assert second.returncode == 0

    records = [
        json.loads(line)
        for line in rules_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len({r["id"] for r in records}) == 1


def test_bad_arg_shape_is_error_json_exit_1(tmp_path: Path, fake_bin: Path) -> None:
    proc = _run(["harness_mailbox_read", "positional"], tmp_path, fake_bin)

    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    assert payload["ok"] is False
    assert "positional" in payload["error"]


# -- helper functions ----------------------------------------------------------------


def test_parse_args_json_values() -> None:
    parsed = _parse_args(["--limit", "5"])
    assert parsed == {"limit": 5}
    assert isinstance(parsed["limit"], int)

    assert _parse_args(["--x", '{"a":1}']) == {"x": {"a": 1}}
    assert _parse_args(["--note", "hello"]) == {"note": "hello"}
    # ``--key-with-dashes`` maps to snake_case.
    assert _parse_args(["--rule-id", "r-abc"]) == {"rule_id": "r-abc"}


def test_parse_args_bad_shapes_return_error_strings() -> None:
    missing = _parse_args(["--limit"])
    assert isinstance(missing, str)
    assert "missing value" in missing

    not_a_flag = _parse_args(["oops"])
    assert isinstance(not_a_flag, str)
    assert "oops" in not_a_flag


def test_usage_lists_every_op_with_description() -> None:
    usage = _usage()
    assert usage.startswith("usage:")
    for name, meta in OP_SCHEMAS.items():
        assert name in usage
        assert str(meta["description"]).split()[0] in usage
