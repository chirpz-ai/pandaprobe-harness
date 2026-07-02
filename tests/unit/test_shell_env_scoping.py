"""Credential scoping and deny rules for the restricted sandbox shell.

The policy must scrub credential-shaped environment variables for plain
binaries (``jq``/``cat``/``ls``), restore the PandaProbe auth variables only
for the platform binaries, and refuse credential/config surfaces outright.
The integration test proves the scoping end-to-end through a real subprocess.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
from pathlib import Path

import pytest

from pandaprobe_harness.sandbox.policy import ShellPolicy, ShellPolicyError
from pandaprobe_harness.sandbox.shell import RestrictedShellTool

BASE_ENV = {
    "PATH": "/usr/bin",
    "HOME": "/home/agent",
    "PANDAPROBE_API_KEY": "sk-panda",
    "PANDAPROBE_ENDPOINT": "https://api.example.test",
    "MY_SECRET": "hunter2",
    "GITHUB_TOKEN": "gh-token",
    "AWS_ACCESS_KEY_ID": "AKIA123",
}

SCRUBBED = ("PANDAPROBE_API_KEY", "MY_SECRET", "GITHUB_TOKEN", "AWS_ACCESS_KEY_ID")


# -- unit: environment scoping -------------------------------------------------


def test_scrubbed_env_removes_credentials_for_plain_binaries() -> None:
    env = ShellPolicy().scrubbed_env("jq", BASE_ENV)
    for name in SCRUBBED:
        assert name not in env
    assert env["PATH"] == "/usr/bin"
    assert env["HOME"] == "/home/agent"


@pytest.mark.parametrize("binary", ["pandaprobe", "pandaprobe-harness-agent"])
def test_scrubbed_env_restores_auth_vars_only_for_platform_binaries(binary: str) -> None:
    env = ShellPolicy().scrubbed_env(binary, BASE_ENV)
    assert env["PANDAPROBE_API_KEY"] == "sk-panda"
    assert env["PANDAPROBE_ENDPOINT"] == "https://api.example.test"
    assert "MY_SECRET" not in env  # non-PandaProbe credentials stay scrubbed


# -- unit: deny rules ------------------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["pandaprobe", "config", "show"],
        ["pandaprobe", "auth", "login"],
        ["pandaprobe", "evals", "scores", "get", "x", "--reveal-secrets"],
    ],
)
def test_denied_commands_raise(argv: list[str]) -> None:
    with pytest.raises(ShellPolicyError):
        ShellPolicy().validate(argv)


def test_allowed_diagnostic_command_passes() -> None:
    ShellPolicy().validate(["pandaprobe", "evals", "scores", "get", "x"])  # must not raise


def test_harness_agent_binary_is_allow_listed_by_default() -> None:
    assert "pandaprobe-harness-agent" in ShellPolicy().allowed_binaries


# -- integration: real subprocess sees the scoped environment --------------------


def _install_shim(fake_bin: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(fake_bin, target)
    target.chmod(target.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


async def test_subprocess_env_scoping_end_to_end(tmp_path: Path, fake_bin: Path) -> None:
    shim = tmp_path / "shim"
    _install_shim(fake_bin, shim / "pandaprobe")
    _install_shim(fake_bin, shim / "jq")

    env = {
        **os.environ,
        "PATH": f"{shim}{os.pathsep}{os.environ['PATH']}",
        "PANDAPROBE_API_KEY": "sk-secret",
        "FAKE_ECHO_ENV": "PANDAPROBE_API_KEY",
    }
    tool = RestrictedShellTool(ShellPolicy(workdir=tmp_path), env=env)

    # The platform binary gets its auth variable back.
    result = await tool("pandaprobe evals scores get x")
    assert result.ok
    assert json.loads(result.stdout)["env"]["PANDAPROBE_API_KEY"] == "sk-secret"

    # A plain binary never sees the credential (JSON null == scrubbed).
    result = await tool("jq somearg")
    assert result.ok
    assert json.loads(result.stdout)["env"]["PANDAPROBE_API_KEY"] is None

    # Denied surfaces are refused before any subprocess is spawned.
    with pytest.raises(ShellPolicyError):
        await tool("pandaprobe config show")
