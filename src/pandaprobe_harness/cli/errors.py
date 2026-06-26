"""Typed error hierarchy for PandaProbe CLI invocations.

The real ``pandaprobe`` CLI (Go/Cobra) defines a stable exit-code contract
(``internal/exitcode/exitcode.go`` + ``internal/api/errors.go``):

    0  OK         — success
    1  General    — unexpected/general failure (network, decode, etc.)
    2  Auth       — authentication/authorization failure (HTTP 401, 403)
    3  NotFound   — resource not found (HTTP 404)
    4  Validation — client-side validation, or HTTP 400 / 422
    5  APIError   — other server-side error (other 4xx, 5xx)

``raise_for_exit_code`` maps the (deterministic) exit code to the matching typed
exception so callers can branch on failure mode — e.g. treat ``CliNotFoundError``
from eventual-consistency lag as recoverable. stderr text is used only as a
best-effort fallback when the exit code is unknown.
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
    """Exit code 1 — unexpected/general failure (network, decode, etc.)."""


class CliAuthError(CliError):
    """Exit code 2 — authentication/authorization failure (401, 403)."""


class CliNotFoundError(CliError):
    """Exit code 3 — resource not found (404); often eventual-consistency lag."""


class CliValidationError(CliError):
    """Exit code 4 — client-side validation, or 400/422 from the server."""


class CliApiError(CliError):
    """Exit code 5 — other server-side error (other 4xx, 5xx)."""


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

# Best-effort hints, used only to classify an UNKNOWN (non-contract) exit code.
_AUTH_HINTS = ("401", "403", "unauthor", "forbidden")
_NOTFOUND_HINTS = ("404", "not found")


def _fallback_for_unknown(result: CliResult) -> type[CliError]:
    text = f"{result.stderr}\n{result.stdout}".lower()
    if any(h in text for h in _AUTH_HINTS):
        return CliAuthError
    if any(h in text for h in _NOTFOUND_HINTS):
        return CliNotFoundError
    return CliGeneralError


def raise_for_exit_code(result: CliResult) -> None:
    """Raise the typed error matching ``result.exit_code``; no-op on success."""

    if result.exit_code == 0:
        return
    exc_type = _EXIT_MAP.get(result.exit_code) or _fallback_for_unknown(result)
    detail = result.stderr.strip() or result.stdout.strip() or "<no output>"
    raise exc_type(
        f"`{result.command_line}` exited with code {result.exit_code}: {detail}",
        result=result,
    )
