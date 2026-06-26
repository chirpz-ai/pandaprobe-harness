"""the restricted shell-execution tool exposed to the agent.

The agent's toolbelt is augmented with this tool so it can call the
``pandaprobe`` CLI natively (and read its workspace) while being prevented from
running arbitrary destructive commands. Commands are parsed with ``shlex`` and
executed via ``asyncio.create_subprocess_exec`` — **never** ``shell=True``.
"""

from __future__ import annotations

import asyncio
import os
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .policy import ShellPolicy

__all__ = ["RestrictedShellTool", "ShellResult"]


@dataclass(frozen=True, slots=True)
class ShellResult:
    """The outcome of a sandboxed command, returned to the agent."""

    command: str
    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class RestrictedShellTool:
    """A policy-enforced async shell command runner."""

    def __init__(
        self,
        policy: ShellPolicy | None = None,
        *,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._policy = policy or ShellPolicy()
        self._env = dict(env) if env is not None else dict(os.environ)

    async def __call__(self, command: str) -> ShellResult:
        argv = shlex.split(command)
        self._policy.validate(argv)

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._policy.workdir),
            env=self._env,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), self._policy.timeout_s
            )
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return ShellResult(
                command=command,
                exit_code=124,
                stdout="",
                stderr=f"command timed out after {self._policy.timeout_s}s",
            )

        return ShellResult(
            command=command,
            exit_code=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
        )

    @property
    def tool_schema(self) -> dict[str, Any]:
        """A JSON schema describing this tool for LLM tool-use APIs."""

        return {
            "name": "sandbox_shell",
            "description": (
                "Execute a restricted shell command inside the diagnostic "
                "sandbox. Use it to run the `pandaprobe` CLI and read your "
                "workspace. Allowed binaries: "
                f"{sorted(self._policy.allowed_binaries)}."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The full command line, e.g. "
                        "'pandaprobe evals scores get trace-123'.",
                    }
                },
                "required": ["command"],
            },
        }
