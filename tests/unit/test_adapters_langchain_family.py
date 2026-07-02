"""LangChain-family adapters (LangChain, DeepAgents) share the LangGraph base."""

from __future__ import annotations

import importlib.util

import pytest

import pandaprobe_harness.adapters._base as adapter_base
from pandaprobe_harness.adapters._langchain import LangChainCallbackAdapter
from pandaprobe_harness.adapters.deepagents import DeepAgentsAdapter
from pandaprobe_harness.adapters.langchain import LangChainAdapter
from pandaprobe_harness.adapters.langgraph import LangGraphAdapter

_HAS_LANGCHAIN = importlib.util.find_spec("langchain_core") is not None
_NEW = [LangChainAdapter, DeepAgentsAdapter]


def test_all_three_share_the_callback_base() -> None:
    for cls in (LangGraphAdapter, LangChainAdapter, DeepAgentsAdapter):
        assert issubclass(cls, LangChainCallbackAdapter)
    # Distinct pip-extra hints for ImportErrors.
    assert {LangGraphAdapter._extra, LangChainAdapter._extra, DeepAgentsAdapter._extra} == {
        "langgraph",
        "langchain",
        "deepagents",
    }


@pytest.mark.parametrize("cls", _NEW)
def test_parse_turn_and_session(cls: type[LangChainCallbackAdapter]) -> None:
    ctx = cls().parse_turn({"session_id": "s1", "turn_index": 4})
    assert ctx.session_id == "s1" and ctx.turn_index == 4
    assert cls(session_id="sc").parse_turn({}).session_id == "sc"
    with pytest.raises(ValueError):
        cls().parse_turn({})


@pytest.mark.parametrize("cls", _NEW)
def test_session_bridge(
    cls: type[LangChainCallbackAdapter], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(adapter_base, "current_session_id", lambda: "ctx-sid")
    assert cls().parse_turn({}).session_id == "ctx-sid"


@pytest.mark.parametrize("cls", _NEW)
def test_no_injection_surface_and_guarded_callback(
    cls: type[LangChainCallbackAdapter],
) -> None:
    adapter = cls()
    for legacy in ("inject_alert", "consume_alerts", "consume_messages", "startup_messages"):
        assert not hasattr(adapter, legacy)
    if _HAS_LANGCHAIN:
        assert hasattr(adapter.make_callback(), "on_chain_end")
    else:
        with pytest.raises(ImportError):
            adapter.make_callback()
