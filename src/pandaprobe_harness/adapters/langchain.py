"""LangChain adapter (optional ``langchain`` extra).

Works with ``langchain.agents.create_agent`` agents and LCEL chains — anything
that accepts a LangChain callback via ``config={"callbacks": [...]}``. Turn
detection fires on the root chain end. See :class:`LangChainCallbackAdapter`.
"""

from __future__ import annotations

from ._langchain import LangChainCallbackAdapter

__all__ = ["LangChainAdapter"]


class LangChainAdapter(LangChainCallbackAdapter):
    """Bridge ``PandaHarnessHook`` to a LangChain agent/chain execution."""

    _extra = "langchain"
