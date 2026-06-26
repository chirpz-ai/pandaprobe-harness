"""Shared base for LangChain-callback-based adapters.

LangGraph, LangChain, and DeepAgents all instrument via a LangChain
``BaseCallbackHandler`` (the SDK's integrations are thin subclasses of one shared
callback). They share identical harness semantics:

* **turn detection** — :meth:`make_callback` returns an async LangChain callback
  that fires ``hook.on_turn_end`` on the *root* chain end (one turn = one root
  invocation); session identity comes from the SDK session ``ContextVar``.
* **alert / rules injection** — LangChain callbacks are read-only (they cannot
  mutate graph/agent state), so injection is state-level: pending alerts and the
  startup rules preamble are exposed as ``SystemMessage``s for the developer to
  merge into the next ``invoke``/``ainvoke`` input's ``messages``.

Concrete adapters subclass this and set ``_extra`` (the pip extra name used in
ImportError hints). The contract methods are importable without LangChain; the
``SystemMessage``/callback helpers raise an informative ``ImportError`` if the
relevant extra is missing.
"""

from __future__ import annotations

from collections.abc import MutableSequence
from typing import Any

from ._base import BaseSinkAdapter

__all__ = ["LangChainCallbackAdapter"]


class LangChainCallbackAdapter(BaseSinkAdapter):
    """Base adapter for frameworks instrumented via a LangChain callback."""

    #: pip extra name surfaced in ImportError hints (overridden by subclasses).
    _extra = "langchain"

    # -- alert / rules injection (state-level) -------------------------------

    def consume_messages(self) -> list[Any]:
        """Pending alerts as ``SystemMessage``s for the next invoke input."""

        system_message = self._system_message_cls()
        return [system_message(content=alert) for alert in self.consume_alerts()]

    def startup_messages(self) -> list[Any]:
        """The living harness rules as a leading ``SystemMessage`` (closes loop)."""

        preamble = self.startup_context_text()
        if not preamble:
            return []
        system_message = self._system_message_cls()
        return [system_message(content=preamble)]

    def drain_into(self, messages: MutableSequence[Any]) -> None:
        """Append pending alert ``SystemMessage``s into a state ``messages`` list."""

        for message in self.consume_messages():
            messages.append(message)

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

    # -- internals -----------------------------------------------------------

    def _system_message_cls(self) -> Any:
        try:
            from langchain_core.messages import SystemMessage
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError(
                f"{type(self).__name__} message helpers require langchain-core; "
                f"install the '{self._extra}' extra."
            ) from exc
        return SystemMessage
