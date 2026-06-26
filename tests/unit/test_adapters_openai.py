from __future__ import annotations

import importlib.util

import pytest

import pandaprobe_harness.adapters._base as adapter_base
from pandaprobe_harness.adapters.openai_agents import OpenAIAgentsAdapter

_HAS_OPENAI_AGENTS = importlib.util.find_spec("agents") is not None


class _Hook:
    def startup_context(self) -> str:
        return "RULES: verify before charging"


def test_parse_turn_and_session(monkeypatch: pytest.MonkeyPatch) -> None:
    assert OpenAIAgentsAdapter(session_id="s").parse_turn({}).session_id == "s"
    with pytest.raises(ValueError):
        OpenAIAgentsAdapter().parse_turn({})
    monkeypatch.setattr(adapter_base, "current_session_id", lambda: "ctx")
    assert OpenAIAgentsAdapter().parse_turn({}).session_id == "ctx"


def test_consume_input_items_for_manual_injection() -> None:
    adapter = OpenAIAgentsAdapter(session_id="s")
    adapter.inject_alert("ALERT")
    assert adapter.consume_input_items() == [{"role": "system", "content": "ALERT"}]
    assert adapter.pending_alerts == ()  # drained


def test_startup_input_items() -> None:
    adapter = OpenAIAgentsAdapter(session_id="s")
    assert adapter.startup_input_items() == []  # no hook → no rules
    adapter.register(_Hook())
    assert adapter.startup_input_items() == [
        {"role": "system", "content": "RULES: verify before charging"}
    ]


def test_instrument_returns_false_when_dep_absent() -> None:
    if not _HAS_OPENAI_AGENTS:
        assert OpenAIAgentsAdapter().instrument() is False
