from __future__ import annotations

import importlib.util

import pytest

import pandaprobe_harness.adapters._base as adapter_base
from pandaprobe_harness.adapters.claude_agent_sdk import ClaudeAgentSDKAdapter

_HAS_CLAUDE = (
    importlib.util.find_spec("claude_agent_sdk") is not None
    and importlib.util.find_spec("wrapt") is not None
)


def test_parse_turn_and_session(monkeypatch: pytest.MonkeyPatch) -> None:
    assert ClaudeAgentSDKAdapter(session_id="s").parse_turn({}).session_id == "s"
    with pytest.raises(ValueError):
        ClaudeAgentSDKAdapter().parse_turn({})
    monkeypatch.setattr(adapter_base, "current_session_id", lambda: "ctx")
    assert ClaudeAgentSDKAdapter().parse_turn({}).session_id == "ctx"


def test_adapter_exposes_no_injection_surface() -> None:
    adapter = ClaudeAgentSDKAdapter(session_id="s")
    for legacy in ("inject_alert", "consume_alerts", "inject_into_history", "prime_startup"):
        assert not hasattr(adapter, legacy)


def test_instrument_returns_false_when_dep_absent() -> None:
    if not _HAS_CLAUDE:
        assert ClaudeAgentSDKAdapter().instrument() is False
