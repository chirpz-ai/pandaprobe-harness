from __future__ import annotations

from pathlib import Path

import pytest

from pandaprobe_harness.cli.errors import (
    CliAuthError,
    CliNotFoundError,
    CliOutputError,
    CliTimeoutError,
)
from pandaprobe_harness.cli.subprocess_client import SubprocessCliClient


def _client(fake_bin: Path, **kw) -> SubprocessCliClient:
    return SubprocessCliClient(str(fake_bin), **kw)


async def test_success_parses_json_and_injects_base_flags(fake_bin: Path) -> None:
    client = _client(fake_bin)
    result = await client.run("traces", "list")
    assert result.exit_code == 0
    payload = result.json()
    # base flags `--format json` are injected ahead of the subcommand
    assert payload["argv"][:2] == ["--format", "json"]
    assert payload["argv"][-2:] == ["traces", "list"]


async def test_stderr_noise_does_not_break_parsing(fake_bin: Path) -> None:
    client = _client(fake_bin)
    result = await client.run("version")
    assert result.stderr  # noise present
    assert result.json()["argv"]  # stdout still valid JSON


async def test_auth_exit_code_maps(fake_bin: Path) -> None:
    # Real CLI contract: exit 2 == Auth (401/403).
    client = _client(fake_bin, env={"FAKE_EXIT_CODE": "2", "FAKE_STDERR": "unauthorized"})
    with pytest.raises(CliAuthError) as info:
        await client.run("traces", "list")
    assert "unauthorized" in str(info.value)


async def test_not_found_exit_code_maps(fake_bin: Path) -> None:
    # Real CLI contract: exit 3 == NotFound (404).
    client = _client(fake_bin, env={"FAKE_EXIT_CODE": "3", "FAKE_STDERR": "not found"})
    with pytest.raises(CliNotFoundError):
        await client.run("traces", "get", "nope")


async def test_timeout_kills_child(fake_bin: Path) -> None:
    client = _client(fake_bin, env={"FAKE_SLEEP": "5"}, default_timeout=0.2)
    with pytest.raises(CliTimeoutError):
        await client.run("traces", "list")


async def test_bad_json_raises_output_error(fake_bin: Path) -> None:
    client = _client(fake_bin, env={"FAKE_BAD_JSON": "1"})
    result = await client.run("traces", "list")
    with pytest.raises(CliOutputError):
        result.json()


async def test_env_passthrough(fake_bin: Path) -> None:
    client = _client(
        fake_bin, env={"FAKE_ECHO_ENV": "PANDAPROBE_API_KEY", "PANDAPROBE_API_KEY": "sk_pp_x"}
    )
    payload = (await client.run("auth", "status")).json()
    assert payload["env"]["PANDAPROBE_API_KEY"] == "sk_pp_x"
