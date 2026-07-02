"""Real ``CliClient`` backed by an out-of-process ``pandaprobe`` invocation.

Uses ``asyncio.create_subprocess_exec`` so CLI calls integrate natively with the
event loop and never block the agent's main runtime thread. Authentication is
deliberately *not* re-implemented here: the child process inherits the parent
environment, so ``PANDAPROBE_API_KEY`` / ``PANDAPROBE_PROJECT_NAME`` /
``PANDAPROBE_ENDPOINT`` env vars, ``~/.pandaprobe/config.yaml`` and a prior
``pandaprobe auth login`` all continue to work unchanged.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping, Sequence

from .client import CliResult
from .errors import CliGeneralError, CliTimeoutError, raise_for_exit_code

__all__ = ["SubprocessCliClient"]


class SubprocessCliClient:
    """Invoke the ``pandaprobe`` binary as an async subprocess."""

    def __init__(
        self,
        binary: str = "pandaprobe",
        *,
        default_timeout: float = 30.0,
        base_flags: Sequence[str] = ("--format", "json"),
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._binary = binary
        self._default_timeout = default_timeout
        self._base_flags = tuple(base_flags)
        # Default to inheriting the full process environment so CLI auth resolves.
        self._env = dict(env) if env is not None else dict(os.environ)

    async def run(self, *args: str, timeout: float | None = None) -> CliResult:
        argv = (*self._base_flags, *args)
        try:
            proc = await asyncio.create_subprocess_exec(
                self._binary,
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._env,
            )
        except OSError as exc:
            # A missing/unexecutable binary (FileNotFoundError, PermissionError)
            # must surface as a CliError so the harness degrades gracefully
            # rather than crashing the host loop with a raw OSError.
            result = CliResult(args=argv, exit_code=-1, stdout="", stderr=str(exc))
            raise CliGeneralError(
                f"failed to launch {self._binary!r}: {exc}", result=result
            ) from exc

        effective_timeout = self._default_timeout if timeout is None else timeout
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(), effective_timeout
            )
        except TimeoutError as exc:
            proc.kill()
            await proc.wait()
            result = CliResult(args=argv, exit_code=-1, stdout="", stderr="")
            raise CliTimeoutError(
                f"`{result.command_line}` timed out after {effective_timeout}s",
                result=result,
            ) from exc

        result = CliResult(
            args=argv,
            exit_code=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout_b.decode("utf-8", errors="replace"),
            stderr=stderr_b.decode("utf-8", errors="replace"),
        )
        raise_for_exit_code(result)
        return result
