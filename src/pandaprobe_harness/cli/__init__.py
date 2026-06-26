"""PandaProbe CLI seam: the only boundary to the external ``pandaprobe`` binary."""

from .client import CliClient, CliResult
from .errors import (
    CliApiError,
    CliAuthError,
    CliError,
    CliGeneralError,
    CliNotFoundError,
    CliOutputError,
    CliTimeoutError,
    CliValidationError,
    raise_for_exit_code,
)
from .models import RunCreated, RunScores, ScoreRecord
from .subprocess_client import SubprocessCliClient

__all__ = [
    "CliClient",
    "CliResult",
    "SubprocessCliClient",
    "CliError",
    "CliGeneralError",
    "CliAuthError",
    "CliNotFoundError",
    "CliValidationError",
    "CliApiError",
    "CliTimeoutError",
    "CliOutputError",
    "raise_for_exit_code",
    "RunCreated",
    "RunScores",
    "ScoreRecord",
]
