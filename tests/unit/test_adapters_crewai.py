from __future__ import annotations

import importlib.util

import pytest

import pandaprobe_harness.adapters._base as adapter_base
from pandaprobe_harness.adapters.crewai import CrewAIAdapter

_HAS_CREWAI = (
    importlib.util.find_spec("crewai") is not None
    and importlib.util.find_spec("wrapt") is not None
)


def test_parse_turn_resolves_crew_id() -> None:
    adapter = CrewAIAdapter()
    ctx = adapter.parse_turn({"crew_id": "c1", "turn_index": 3})
    assert ctx.session_id == "c1"
    assert ctx.turn_index == 3


def test_parse_turn_uses_constructor_session() -> None:
    assert CrewAIAdapter(session_id="sc").parse_turn({}).session_id == "sc"


def test_parse_turn_requires_a_session() -> None:
    with pytest.raises(ValueError):
        CrewAIAdapter().parse_turn({})


def test_session_bridge_from_contextvar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter_base, "current_session_id", lambda: "ctx-sid")
    assert CrewAIAdapter().parse_turn({}).session_id == "ctx-sid"


def test_inject_and_consume_context() -> None:
    adapter = CrewAIAdapter(session_id="s")
    adapter.inject_alert("X")
    assert adapter.pending_alerts == ("X",)
    assert adapter.consume_context() == ["X"]
    assert adapter.pending_alerts == ()


def test_startup_context_text_empty_without_hook() -> None:
    assert CrewAIAdapter(session_id="s").startup_context_text() == ""


def test_instrument_returns_false_when_dep_absent() -> None:
    if not _HAS_CREWAI:
        assert CrewAIAdapter().instrument() is False
