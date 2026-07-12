"""Offline tests for the provider layer: model resolution, param allowlist,
tool-call parsing, cost fallback, and the mock client. No network."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pandabench.providers.litellm_client import (
    LiteLLMClient,
    MockClient,
    Usage,
    _parse_response,
)
from pandabench.providers.models import load_registry, provider_of
from pandabench.providers.tracing import PandaTracer

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def registry():
    return load_registry(CONFIGS / "models.yaml")


# -- model resolution ---------------------------------------------------------


def test_single_backend_resolution(registry):
    m = registry.resolve("gemini-3.1-flash-lite")
    assert m.litellm_model == "vertex_ai/gemini-3.1-flash-lite"
    assert m.provider == "vertex"
    assert m.backend is None
    assert "temperature" in m.param_allowlist


def test_claude_defaults_to_vertex(registry):
    m = registry.resolve("claude-sonnet-5", env={})
    assert m.litellm_model == "vertex_ai/claude-sonnet-5"
    assert m.provider == "vertex"
    assert m.backend == "vertex_ai"
    assert "temperature" not in m.param_allowlist  # Claude 5 rejects it


def test_claude_backend_arg_overrides(registry):
    m = registry.resolve(
        "claude-sonnet-5", backend="anthropic", env={"CLAUDE_BACKEND": "vertex_ai"}
    )
    assert m.litellm_model == "anthropic/claude-sonnet-5"
    assert m.provider == "anthropic"
    assert m.backend == "anthropic"


def test_claude_env_overrides_default(registry):
    m = registry.resolve("claude-sonnet-5", env={"CLAUDE_BACKEND": "anthropic"})
    assert m.backend == "anthropic"


def test_unknown_model_raises(registry):
    with pytest.raises(KeyError):
        registry.resolve("nope-9")


def test_backend_on_single_backend_raises(registry):
    with pytest.raises(ValueError):
        registry.resolve("gemini-3.1-flash-lite", backend="anthropic")


def test_unknown_backend_raises(registry):
    with pytest.raises(ValueError):
        registry.resolve("claude-sonnet-5", backend="bedrock", env={})


def test_roles(registry):
    assert registry.role("user_simulator") == "gemini-3.1-flash-lite"
    assert registry.resolve(registry.role("dry_run")).is_mock is True
    with pytest.raises(KeyError):
        registry.role("does-not-exist")


def test_provider_of():
    assert provider_of("vertex_ai/claude-haiku-4-5") == "vertex"
    assert provider_of("openai/gpt-5.4-mini") == "openai"
    assert provider_of("something/weird") == "something"


# -- param allowlist ----------------------------------------------------------


def test_param_allowlist_drops_temperature_for_claude(registry):
    client = LiteLLMClient(tracer=PandaTracer.disabled())
    claude = registry.resolve("claude-sonnet-5", env={})
    params = client._call_params(claude, max_tokens=1000, extra={"temperature": 0.7})
    assert "temperature" not in params  # not allowlisted
    assert params["max_tokens"] == 1000


def test_param_allowlist_keeps_temperature_for_gemini(registry):
    client = LiteLLMClient(tracer=PandaTracer.disabled())
    gemini = registry.resolve("gemini-3.1-flash-lite")
    params = client._call_params(gemini, max_tokens=None, extra={"temperature": 0.5})
    assert params["temperature"] == 0.5
    assert "max_tokens" in params  # default applied


# -- response parsing ---------------------------------------------------------


class _FakeFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, id: str, name: str, arguments: str) -> None:
        self.id = id
        self.function = _FakeFunction(name, arguments)


class _FakeMessage:
    def __init__(self, content: Any, tool_calls: list[_FakeToolCall] | None) -> None:
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, message: _FakeMessage, finish_reason: str) -> None:
        self.message = message
        self.finish_reason = finish_reason


class _FakeUsage:
    def __init__(self, p: int, c: int) -> None:
        self.prompt_tokens = p
        self.completion_tokens = c


class _FakeResponse:
    def __init__(self, choice: _FakeChoice, usage: _FakeUsage) -> None:
        self.choices = [choice]
        self.usage = usage


def test_parse_tool_call_arguments_are_json(registry):
    resp = _FakeResponse(
        _FakeChoice(
            _FakeMessage(
                None,
                [_FakeToolCall("call_1", "execute", '{"code": "print(1)"}')],
            ),
            "tool_calls",
        ),
        _FakeUsage(100, 20),
    )
    model = registry.resolve("gemini-3.1-flash-lite")
    result = _parse_response(resp, model)
    assert len(result.tool_calls) == 1
    tc = result.tool_calls[0]
    assert tc.name == "execute"
    assert tc.arguments == {"code": "print(1)"}  # parsed dict, not a string
    assert result.assistant_message["tool_calls"][0]["function"]["name"] == "execute"
    assert result.usage.input_tokens == 100
    assert result.usage.output_tokens == 20
    # No real litellm price for the fake response -> price-table fallback.
    assert result.usage.cost_usd == pytest.approx(100 / 1e6 * 0.25 + 20 / 1e6 * 1.50)


def test_parse_malformed_tool_args_do_not_crash(registry):
    resp = _FakeResponse(
        _FakeChoice(_FakeMessage(None, [_FakeToolCall("c", "f", "not json")]), "tool_calls"),
        _FakeUsage(1, 1),
    )
    result = _parse_response(resp, registry.resolve("gemini-3.1-flash-lite"))
    assert result.tool_calls[0].arguments == {}  # degrades, never raises


def test_parse_plain_text_final(registry):
    resp = _FakeResponse(_FakeChoice(_FakeMessage("done", None), "stop"), _FakeUsage(5, 3))
    result = _parse_response(resp, registry.resolve("gemini-3.1-flash-lite"))
    assert result.tool_calls == []
    assert result.assistant_message == {"role": "assistant", "content": "done"}
    assert result.finish_reason == "stop"


# -- mock client + usage ------------------------------------------------------


async def test_mock_client_returns_final(registry):
    client = MockClient()
    result = await client.chat(
        model=registry.resolve("mock"), messages=[{"role": "user", "content": "hi"}]
    )
    assert result.tool_calls == []
    assert "done" in result.assistant_message["content"]
    assert client.calls[0]["n_messages"] == 1


def test_usage_addition():
    total = Usage(10, 5, 0.1) + Usage(3, 2, 0.05)
    assert total == Usage(13, 7, pytest.approx(0.15))
    assert total.to_dict()["input_tokens"] == 13
