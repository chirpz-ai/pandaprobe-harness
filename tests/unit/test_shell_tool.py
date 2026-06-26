from __future__ import annotations

from pathlib import Path

import pytest

from pandaprobe_harness.sandbox.policy import ShellPolicy, ShellPolicyError
from pandaprobe_harness.sandbox.shell import RestrictedShellTool


async def test_runs_allowed_cat(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    tool = RestrictedShellTool(ShellPolicy(workdir=tmp_path))
    result = await tool("cat note.txt")
    assert result.ok
    assert result.stdout.strip() == "hello"


async def test_runs_pandaprobe_via_path_shim(
    tmp_path: Path, pandaprobe_path: dict[str, str]
) -> None:
    tool = RestrictedShellTool(ShellPolicy(workdir=tmp_path), env=pandaprobe_path)
    result = await tool("pandaprobe evals scores get trace-1")
    assert result.ok
    assert "argv" in result.stdout


async def test_disallowed_command_raises(tmp_path: Path) -> None:
    tool = RestrictedShellTool(ShellPolicy(workdir=tmp_path))
    with pytest.raises(ShellPolicyError):
        await tool("rm -rf /")


def test_tool_schema_shape(tmp_path: Path) -> None:
    tool = RestrictedShellTool(ShellPolicy(workdir=tmp_path))
    schema = tool.tool_schema
    assert schema["name"] == "sandbox_shell"
    assert "command" in schema["input_schema"]["properties"]
    assert schema["input_schema"]["required"] == ["command"]
