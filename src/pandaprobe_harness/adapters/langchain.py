"""LangChain adapter (optional ``langchain`` extra).

Works with ``langchain.agents.create_agent`` agents and LCEL chains — anything
that accepts a LangChain callback via ``config={"callbacks": [...]}``. Turn
detection fires on the root chain end; alert/rules injection is state-level (the
developer merges :meth:`consume_messages` / :meth:`startup_messages` into the
next ``invoke`` input's ``messages``). See :class:`LangChainCallbackAdapter`.
"""

from __future__ import annotations

from ._langchain import LangChainCallbackAdapter

__all__ = ["LangChainAdapter"]


class LangChainAdapter(LangChainCallbackAdapter):
    """Bridge ``PandaHarnessHook`` to a LangChain agent/chain execution."""

    _extra = "langchain"
