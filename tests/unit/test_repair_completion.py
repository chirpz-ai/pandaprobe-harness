"""Official PandaProbe LiteLLM wrapper integration contract."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from pandaprobe_harness.repair.completion import PandaProbeLiteLLMCompletion

_REAL_COMPLETE = PandaProbeLiteLLMCompletion.complete


class FakeLiteLLM:
    def __init__(
        self,
        response: Any = None,
        error: Exception | None = None,
        *,
        supported_params: tuple[str, ...] = (),
    ) -> None:
        self.response = response
        self.error = error
        self.supported_params = supported_params
        self.calls: list[dict[str, Any]] = []

    def get_supported_openai_params(self, *, model: str) -> list[str]:
        del model
        return list(self.supported_params)

    async def acompletion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def _response() -> SimpleNamespace:
    calls = [
        SimpleNamespace(
            id="c1",
            function=SimpleNamespace(name="harness_notice_read", arguments='{"notice_id":"n"}'),
        ),
        SimpleNamespace(
            id="c2",
            function=SimpleNamespace(name="harness_rules_search", arguments='{"query":"x"}'),
        ),
    ]
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="diagnosis", tool_calls=calls))],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
        _hidden_params={"response_cost": 0.0125},
    )


@pytest.mark.parametrize(
    ("model", "supports_reasoning"),
    [
        ("openai/gpt-test", True),
        ("anthropic/claude-test", False),
        ("bedrock/anthropic.claude-test-v1:0", False),
        ("vertex_ai/gemini-test", True),
    ],
)
async def test_all_provider_identifiers_use_the_same_wrapped_path(
    model: str, supports_reasoning: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeLiteLLM(
        _response(),
        supported_params=("reasoning_effort",) if supports_reasoning else (),
    )
    wrapped: list[object] = []

    def wrap_litellm() -> FakeLiteLLM:
        wrapped.append(fake)
        return fake

    monkeypatch.setattr("pandaprobe.wrappers.litellm.wrap_litellm", wrap_litellm)
    result = await _REAL_COMPLETE(
        PandaProbeLiteLLMCompletion(),
        model=model,
        messages=[{"role": "user", "content": "repair"}],
        tools=[],
        max_tokens=123,
        temperature=None,
        reasoning_effort="none",
        timeout_s=9,
    )
    assert wrapped == [fake]
    assert fake.calls[0]["model"] == model
    assert "temperature" not in fake.calls[0]
    if supports_reasoning:
        assert fake.calls[0]["reasoning_effort"] == "none"
    else:
        assert "reasoning_effort" not in fake.calls[0]
    assert [call.name for call in result.tool_calls] == [
        "harness_notice_read",
        "harness_rules_search",
    ]
    assert result.tool_calls[0].arguments == '{"notice_id":"n"}'
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 7
    assert result.usage.cost == pytest.approx(0.0125)


async def test_completion_contributes_to_caller_owned_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeLiteLLM(_response(), supported_params=("reasoning_effort",))
    monkeypatch.setattr("pandaprobe.wrappers.litellm.wrap_litellm", lambda: fake)
    monkeypatch.setattr(
        "pandaprobe.start_trace",
        lambda *args, **kwargs: pytest.fail("completion transport must not own traces"),
    )
    await _REAL_COMPLETE(
        PandaProbeLiteLLMCompletion(),
        model="openai/test",
        messages=[{"role": "user", "content": "repair"}],
        tools=[],
        max_tokens=10,
        temperature=0.2,
        reasoning_effort="none",
        timeout_s=3,
    )
    assert fake.calls[0]["temperature"] == 0.2
    assert fake.calls[0]["reasoning_effort"] == "none"


async def test_wrapper_provider_exception_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeLiteLLM(error=TimeoutError("provider timeout"))
    monkeypatch.setattr("pandaprobe.wrappers.litellm.wrap_litellm", lambda: fake)
    with pytest.raises(TimeoutError):
        await _REAL_COMPLETE(
            PandaProbeLiteLLMCompletion(),
            model="vertex_ai/test",
            messages=[],
            tools=[],
            max_tokens=10,
            temperature=None,
            reasoning_effort=None,
            timeout_s=1,
        )
