"""DeepAgents adapter (optional ``deepagents`` extra).

DeepAgents (by LangChain) is instrumented via a LangChain callback handler, so
it shares the LangChain-family contract: turn detection on the root chain end.
See :class:`LangChainCallbackAdapter`.
"""

from __future__ import annotations

from ._langchain import LangChainCallbackAdapter

__all__ = ["DeepAgentsAdapter"]


class DeepAgentsAdapter(LangChainCallbackAdapter):
    """Bridge ``PandaHarnessHook`` to a DeepAgents execution."""

    _extra = "deepagents"
