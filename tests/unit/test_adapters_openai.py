from __future__ import annotations

import importlib.util

import pytest

import pandaprobe_harness.adapters._base as adapter_base
from pandaprobe_harness.adapters.openai_agents import OpenAIAgentsAdapter

_HAS_OPENAI_AGENTS = importlib.util.find_spec("agents") is not None


def test_parse_turn_and_session(monkeypatch: pytest.MonkeyPatch) -> None:
    assert OpenAIAgentsAdapter(session_id="s").parse_turn({}).session_id == "s"
    with pytest.raises(ValueError):
        OpenAIAgentsAdapter().parse_turn({})
    monkeypatch.setattr(adapter_base, "current_session_id", lambda: "ctx")
    assert OpenAIAgentsAdapter().parse_turn({}).session_id == "ctx"


def test_adapter_exposes_no_injection_surface() -> None:
    adapter = OpenAIAgentsAdapter(session_id="s")
    for legacy in (
        "inject_alert",
        "consume_alerts",
        "consume_input_items",
        "startup_input_items",
    ):
        assert not hasattr(adapter, legacy)


def test_instrument_returns_false_when_dep_absent() -> None:
    if not _HAS_OPENAI_AGENTS:
        assert OpenAIAgentsAdapter().instrument() is False
