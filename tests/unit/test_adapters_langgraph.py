from __future__ import annotations

import importlib.util

import pytest

import pandaprobe_harness.adapters._base as adapter_base
from pandaprobe_harness.adapters.langgraph import LangGraphAdapter

_HAS_LANGCHAIN = importlib.util.find_spec("langchain_core") is not None


def test_parse_turn_from_mapping() -> None:
    adapter = LangGraphAdapter()
    ctx = adapter.parse_turn({"session_id": "s1", "turn_index": 2, "end_state": {"x": 1}})
    assert ctx.session_id == "s1"
    assert ctx.turn_index == 2
    assert ctx.end_state == {"x": 1}


def test_parse_turn_uses_constructor_session() -> None:
    adapter = LangGraphAdapter(session_id="sc")
    assert adapter.parse_turn({}).session_id == "sc"


def test_parse_turn_requires_a_session() -> None:
    with pytest.raises(ValueError):
        LangGraphAdapter().parse_turn({})


def test_session_bridge_from_contextvar(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(adapter_base, "current_session_id", lambda: "ctx-sid")
    assert LangGraphAdapter().parse_turn({}).session_id == "ctx-sid"


def test_adapter_exposes_no_injection_surface() -> None:
    adapter = LangGraphAdapter()
    for legacy in ("inject_alert", "consume_alerts", "consume_messages", "drain_into"):
        assert not hasattr(adapter, legacy)


def test_make_callback_respects_langchain_availability() -> None:
    adapter = LangGraphAdapter()
    if _HAS_LANGCHAIN:
        handler = adapter.make_callback()
        assert hasattr(handler, "on_chain_end")
    else:
        with pytest.raises(ImportError):
            adapter.make_callback()
