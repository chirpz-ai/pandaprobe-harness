"""Wiring smoke tests for the optional framework extras.

Each test gates on ``pytest.importorskip`` for its extra, so the file collects
(and mostly skips) on a core-only install while exercising the real native
registrations wherever the extra is present. ``as_anthropic_tools`` is pure
stdlib dicts, so it runs unconditionally.
"""

from __future__ import annotations

import pytest

from pandaprobe_harness import Harness, HarnessConfig, HarnessToolset
from pandaprobe_harness.agent_tools.native import as_anthropic_tools
from tests.fakes.fake_cli_client import FakeCliClient


def test_langchain_family(config: HarnessConfig, toolset: HarnessToolset) -> None:
    pytest.importorskip("langchain_core")
    from pandaprobe_harness.agent_tools.native import as_langchain_tools

    h = Harness.for_langgraph(session_id="s", config=config, cli=FakeCliClient())
    handler = h.adapter.make_callback()
    assert hasattr(handler, "on_chain_end")

    tools = as_langchain_tools(toolset)
    assert len(tools) == 9
    assert [t.name for t in tools] == [s.name for s in toolset.specs()]


def test_openai_function_tools(toolset: HarnessToolset) -> None:
    pytest.importorskip("agents")
    from pandaprobe_harness.agent_tools.native import as_openai_function_tools

    tools = as_openai_function_tools(toolset)
    assert len(tools) == 9
    for tool in tools:
        assert tool.name
        assert tool.params_json_schema
    assert [t.name for t in tools] == [s.name for s in toolset.specs()]


def test_crewai_instrument() -> None:
    pytest.importorskip("crewai")
    pytest.importorskip("wrapt")
    from pandaprobe_harness.adapters.crewai import CrewAIAdapter

    adapter = CrewAIAdapter(session_id="s")
    assert adapter.instrument() is True
    # Idempotent: a second call is a no-op that still reports success.
    assert adapter.instrument() is True


def test_claude_instrument() -> None:
    pytest.importorskip("claude_agent_sdk")
    pytest.importorskip("wrapt")
    from pandaprobe_harness.adapters.claude_agent_sdk import ClaudeAgentSDKAdapter

    adapter = ClaudeAgentSDKAdapter(session_id="s")
    assert adapter.instrument() is True
    assert adapter.instrument() is True


async def test_anthropic_tools(toolset: HarnessToolset) -> None:
    specs, dispatcher = as_anthropic_tools(toolset)
    assert len(specs) == 9
    for spec in specs:
        assert spec["name"]
        assert spec["description"]
        assert spec["input_schema"]
    result = await dispatcher("harness_mailbox_list", {})
    assert result["ok"] is True
