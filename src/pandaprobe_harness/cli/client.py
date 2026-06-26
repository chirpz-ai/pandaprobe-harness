"""The CLI seam: the single, narrow boundary to the external ``pandaprobe`` binary.

Everything above this module (evaluator, hook) depends only on the ``CliClient``
Protocol — never on subprocess internals. This makes the real subprocess client
and the in-process test fake fully interchangeable.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .errors import CliOutputError

__all__ = ["CliResult", "CliClient"]


@dataclass(frozen=True, slots=True)
class CliResult:
    """The outcome of a single CLI invocation."""

    args: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str

    @property
    def command_line(self) -> str:
        """A shell-quoted rendering of the invocation, for diagnostics."""

        return "pandaprobe " + " ".join(shlex.quote(a) for a in self.args)

    def json(self) -> Any:
        """Parse ``stdout`` as JSON, raising ``CliOutputError`` on failure.

        Parsing is lazy (on demand) so that verbose/debug noise on stderr never
        interferes, and callers that only need exit codes pay nothing.
        """

        try:
            return json.loads(self.stdout)
        except (json.JSONDecodeError, ValueError) as exc:
            raise CliOutputError(
                f"failed to parse JSON from `{self.command_line}`: {exc}",
                result=self,
            ) from exc


@runtime_checkable
class CliClient(Protocol):
    """Abstract async client for the ``pandaprobe`` CLI."""

    async def run(self, *args: str, timeout: float | None = None) -> CliResult:
        """Invoke the CLI with ``args`` and return its result.

        Implementations raise a ``CliError`` subclass for non-zero exit codes
        and timeouts.
        """
        ...
