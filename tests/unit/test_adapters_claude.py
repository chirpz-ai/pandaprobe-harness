from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

import pandaprobe_harness.adapters._base as adapter_base
from pandaprobe_harness.adapters.claude_agent_sdk import ClaudeAgentSDKAdapter

_HAS_CLAUDE = (
    importlib.util.find_spec("claude_agent_sdk") is not None
    and importlib.util.find_spec("wrapt") is not None
)


class _Hook:
    def startup_context(self) -> str:
        return "RULES: never double-charge"


def test_parse_turn_and_session(monkeypatch: pytest.MonkeyPatch) -> None:
    assert ClaudeAgentSDKAdapter(session_id="s").parse_turn({}).session_id == "s"
    with pytest.raises(ValueError):
        ClaudeAgentSDKAdapter().parse_turn({})
    monkeypatch.setattr(adapter_base, "current_session_id", lambda: "ctx")
    assert ClaudeAgentSDKAdapter().parse_turn({}).session_id == "ctx"


def test_inject_into_history_appends_system_messages() -> None:
    adapter = ClaudeAgentSDKAdapter(session_id="s")
    adapter.inject_alert("ALERT-1")
    adapter.inject_alert("ALERT-2")
    client = SimpleNamespace()  # no _pandaprobe_history yet
    n = adapter.inject_into_history(client)
    assert n == 2
    assert client._pandaprobe_history == [
        {"role": "system", "content": "ALERT-1"},
        {"role": "system", "content": "ALERT-2"},
    ]
    assert adapter.pending_alerts == ()  # drained


def test_prime_startup_inserts_rules_first() -> None:
    adapter = ClaudeAgentSDKAdapter(session_id="s")
    adapter.register(_Hook())
    client = SimpleNamespace(_pandaprobe_history=[{"role": "user", "content": "hi"}])
    adapter.prime_startup(client)
    assert client._pandaprobe_history[0] == {
        "role": "system",
        "content": "RULES: never double-charge",
    }


def test_instrument_returns_false_when_dep_absent() -> None:
    if not _HAS_CLAUDE:
        assert ClaudeAgentSDKAdapter().instrument() is False
