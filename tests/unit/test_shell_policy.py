from __future__ import annotations

from pathlib import Path

import pytest

from pandaprobe_harness.sandbox.policy import ShellPolicy, ShellPolicyError


def _policy(tmp_path: Path) -> ShellPolicy:
    return ShellPolicy(workdir=tmp_path)


def test_allowed_binary_passes(tmp_path: Path) -> None:
    _policy(tmp_path).validate(["pandaprobe", "version"])


def test_disallowed_binary_rejected(tmp_path: Path) -> None:
    with pytest.raises(ShellPolicyError):
        _policy(tmp_path).validate(["rm", "-rf", "/"])


def test_empty_command_rejected(tmp_path: Path) -> None:
    with pytest.raises(ShellPolicyError):
        _policy(tmp_path).validate([])


@pytest.mark.parametrize("meta", [";", "|", ">", "<", "&", "`", "$"])
def test_shell_metacharacters_rejected(tmp_path: Path, meta: str) -> None:
    with pytest.raises(ShellPolicyError):
        _policy(tmp_path).validate(["pandaprobe", f"traces{meta}list"])


def test_pipes_allowed_when_enabled(tmp_path: Path) -> None:
    policy = ShellPolicy(workdir=tmp_path, allow_pipes=True)
    # does not raise on metachars when pipes are explicitly allowed
    policy.validate(["pandaprobe", "traces|jq"])


def test_path_escape_rejected(tmp_path: Path) -> None:
    with pytest.raises(ShellPolicyError):
        _policy(tmp_path).validate(["cat", "/etc/passwd"])


def test_relative_escape_rejected(tmp_path: Path) -> None:
    with pytest.raises(ShellPolicyError):
        _policy(tmp_path).validate(["cat", "../../secret"])


def test_in_workdir_path_allowed(tmp_path: Path) -> None:
    target = tmp_path / "traces" / "latest_eval.json"
    _policy(tmp_path).validate(["cat", str(target)])
