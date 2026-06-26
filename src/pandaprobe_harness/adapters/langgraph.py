"""LangGraph adapter (optional ``langgraph`` extra).

LangGraph is instrumented via a LangChain callback handler. Turn detection fires
on the root chain end; alert/rules injection is state-level (the developer merges
:meth:`consume_messages` / :meth:`startup_messages` into the next ``ainvoke``
input's ``messages``). See :class:`LangChainCallbackAdapter` for the full
contract — this subclass only sets the extra name for dependency hints.

Wiring sketch::

    adapter = LangGraphAdapter()
    hook = PandaHarnessHook(adapter, SubprocessCliClient(), config=cfg)
    adapter.register(hook)
    handler = adapter.make_callback()
    state = {"messages": adapter.startup_messages() + [user_message]}
    # each turn (inside `with pandaprobe.session(session_id):`):
    await hook.drain_pending(session_id)
    adapter.drain_into(state["messages"])
    await graph.ainvoke(state, config={"callbacks": [handler],
                                        "configurable": {"thread_id": session_id}})
"""

from __future__ import annotations

from ._langchain import LangChainCallbackAdapter

__all__ = ["LangGraphAdapter"]


class LangGraphAdapter(LangChainCallbackAdapter):
    """Bridge ``PandaHarnessHook`` to a LangGraph (async) execution."""

    _extra = "langgraph"
