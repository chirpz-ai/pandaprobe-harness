"""Shared base for LangChain-callback-based adapters.

LangGraph, LangChain, and DeepAgents all instrument via a LangChain
``AsyncCallbackHandler`` (the SDK's integrations are thin subclasses of one
shared callback): :meth:`make_callback` returns a handler that fires
``hook.on_turn_end`` on the *root* chain end (one turn = one root invocation);
session identity comes from the SDK session ``ContextVar``.

That is the adapter's entire job in the pull model. Self-healing context
(rules + protocol + mailbox banner) comes from ``Harness.system_context()``
and the agent's harness tools — nothing is spliced into the message state.

Concrete adapters subclass this and set ``_extra`` (the pip extra name used in
ImportError hints).
"""

from __future__ import annotations

from typing import Any

from ._base import BaseSinkAdapter

__all__ = ["LangChainCallbackAdapter"]


class LangChainCallbackAdapter(BaseSinkAdapter):
    """Base adapter for frameworks instrumented via a LangChain callback."""

    #: pip extra name surfaced in ImportError hints (overridden by subclasses).
    _extra = "langchain"

    # -- turn detection ------------------------------------------------------

    def make_callback(self) -> Any:
        """An async LangChain callback firing ``on_turn_end`` on root chain end."""

        try:
            from langchain_core.callbacks import AsyncCallbackHandler
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                f"{type(self).__name__}.make_callback requires langchain-core; "
                f"install the '{self._extra}' extra."
            ) from exc

        adapter = self

        class _HarnessTurnHandler(AsyncCallbackHandler):  # type: ignore[misc]
            async def on_chain_end(
                self, outputs: Any, *, run_id: Any, parent_run_id: Any = None, **_: Any
            ) -> None:
                # Only the root chain (no parent) delimits an agent turn.
                if parent_run_id is not None:
                    return
                adapter.notify_turn_end(end_state={"outputs": outputs})

        return _HarnessTurnHandler()
