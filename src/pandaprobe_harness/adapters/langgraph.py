"""LangGraph adapter (optional ``langgraph`` extra).

LangGraph is instrumented via a LangChain callback handler: turn detection
fires on the root chain end. Self-healing is delivered through the workspace
mailbox + harness toolset (see :class:`LangChainCallbackAdapter`) — this
subclass only sets the extra name for dependency hints.

Wiring sketch::

    harness = Harness.for_langgraph(session_id=session_id)
    handler = harness.adapter.make_callback()
    tools = my_tools + as_langchain_tools(harness.toolset)
    system_prompt = harness.system_context() + BASE_PROMPT  # re-read each turn
    # each turn (inside `with pandaprobe.session(session_id):`):
    await graph.ainvoke(state, config={"callbacks": [handler],
                                        "configurable": {"thread_id": session_id}})
"""

from __future__ import annotations

from ._langchain import LangChainCallbackAdapter

__all__ = ["LangGraphAdapter"]


class LangGraphAdapter(LangChainCallbackAdapter):
    """Bridge ``PandaHarnessHook`` to a LangGraph (async) execution."""

    _extra = "langgraph"
