from __future__ import annotations

import pytest

from pandaprobe_harness.cli.client import CliResult
from pandaprobe_harness.cli.errors import (
    CliApiError,
    CliAuthError,
    CliGeneralError,
    CliNotFoundError,
    CliValidationError,
    raise_for_exit_code,
)


def _result(exit_code: int, stderr: str = "boom") -> CliResult:
    return CliResult(args=("evals", "runs", "list"), exit_code=exit_code, stdout="", stderr=stderr)


def test_exit_zero_does_not_raise() -> None:
    raise_for_exit_code(_result(0))  # no exception


@pytest.mark.parametrize(
    ("code", "exc"),
    [
        (1, CliGeneralError),
        (2, CliAuthError),
        (3, CliNotFoundError),
        (4, CliValidationError),
        (5, CliApiError),
    ],
)
def test_exit_code_is_deterministic(code: int, exc: type[Exception]) -> None:
    # Classification is driven by the exit code, NOT by stderr text.
    with pytest.raises(exc) as info:
        raise_for_exit_code(_result(code, stderr="unrelated noise"))
    assert info.value.exit_code == code  # type: ignore[attr-defined]


def test_exit_code_ignores_misleading_stderr() -> None:
    # Exit 2 is always Auth even if stderr happens to mention "not found".
    with pytest.raises(CliAuthError):
        raise_for_exit_code(_result(2, "404 not found"))


def test_unknown_code_falls_back_to_stderr_hints() -> None:
    with pytest.raises(CliAuthError):
        raise_for_exit_code(_result(7, "HTTP 401 unauthorized"))
    with pytest.raises(CliNotFoundError):
        raise_for_exit_code(_result(7, "resource 404 not found"))
    with pytest.raises(CliGeneralError):
        raise_for_exit_code(_result(7, "mystery"))
