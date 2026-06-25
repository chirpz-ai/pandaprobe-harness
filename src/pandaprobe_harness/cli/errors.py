"""Typed error hierarchy for PandaProbe CLI invocations.

The CLI documents these process exit codes:

    0  success
    1  general error
    2  authentication error
    3  not found
    4  validation error
    5  other API error

``raise_for_exit_code`` maps a non-zero exit code to the corresponding typed
exception so callers can branch on failure mode (e.g. treat ``CliNotFoundError``
from eventual-consistency lag as recoverable).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .client import CliResult

__all__ = [
    "CliError",
    "CliGeneralError",
    "CliAuthError",
    "CliNotFoundError",
    "CliValidationError",
    "CliApiError",
    "CliTimeoutError",
    "CliOutputError",
    "raise_for_exit_code",
]


class CliError(RuntimeError):
    """Base class for all CLI invocation failures."""

    def __init__(self, message: str, *, result: CliResult | None = None) -> None:
        super().__init__(message)
        self.result = result
        self.exit_code = result.exit_code if result is not None else None


class CliGeneralError(CliError):
    """Exit code 1 — unspecified general error."""


class CliAuthError(CliError):
    """Exit code 2 — authentication/authorization failure."""


class CliNotFoundError(CliError):
    """Exit code 3 — requested resource does not exist (may be eventual lag)."""


class CliValidationError(CliError):
    """Exit code 4 — invalid arguments/input."""


class CliApiError(CliError):
    """Exit code 5 — remote API error."""


class CliTimeoutError(CliError):
    """The CLI process exceeded its timeout and was killed."""


class CliOutputError(CliError):
    """The CLI produced output that could not be parsed as JSON."""


_EXIT_MAP: dict[int, type[CliError]] = {
    1: CliGeneralError,
    2: CliAuthError,
    3: CliNotFoundError,
    4: CliValidationError,
    5: CliApiError,
}


def raise_for_exit_code(result: CliResult) -> None:
    """Raise the typed error matching ``result.exit_code``; no-op on success."""

    if result.exit_code == 0:
        return
    exc_type = _EXIT_MAP.get(result.exit_code, CliGeneralError)
    detail = result.stderr.strip() or result.stdout.strip() or "<no output>"
    raise exc_type(
        f"`{result.command_line}` exited with code {result.exit_code}: {detail}",
        result=result,
    )
