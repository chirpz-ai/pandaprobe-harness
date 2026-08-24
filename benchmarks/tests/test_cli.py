"""Offline tests for CLI launch-time safety checks."""

from __future__ import annotations

import pytest

from pandabench import cli


def test_live_run_rejects_bedrock_bearer_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "expired-test-token")

    with pytest.raises(SystemExit) as exc_info:
        cli.run_main(["--benchmark", "appworld", "--model", "gpt-oss-20b"])

    assert exc_info.value.code == 2
    assert "unsupported by PandaBench" in capsys.readouterr().err


def test_preflight_fails_even_when_another_provider_is_available(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "expired-test-token")
    monkeypatch.setenv("VERTEXAI_PROJECT", "test-project")
    monkeypatch.setattr(cli.shutil, "which", lambda _name: "/test/pandaprobe")
    monkeypatch.setattr(cli, "_docker_ok", lambda: True)
    monkeypatch.setattr(cli, "_ping", lambda _model: (True, "test ping"))

    assert cli.preflight() == 1
    output = capsys.readouterr().out
    assert "Bedrock bearer token" in output
    assert "preflight: FAIL" in output
