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


def _result(exit_code: int) -> CliResult:
    return CliResult(args=("traces", "list"), exit_code=exit_code, stdout="", stderr="err")


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
        (99, CliGeneralError),  # unknown -> general
    ],
)
def test_exit_code_mapping(code: int, exc: type[Exception]) -> None:
    with pytest.raises(exc) as info:
        raise_for_exit_code(_result(code))
    assert info.value.exit_code == code  # type: ignore[attr-defined]
