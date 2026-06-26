"""Claude Agent SDK adapter (optional ``claude-agent-sdk`` extra).

Instrumented by monkey-patching ``ClaudeSDKClient.receive_response`` (the SDK uses
the same ``wrapt`` approach): a completed ``receive_response`` stream is one agent
turn, after which the hook fires. Session identity comes from the SDK session
``ContextVar``.

Unlike the LangChain/OpenAI integrations, the Claude SDK integration maintains a
mutable conversation history on the client (``client._pandaprobe_history``), which
gives a **real** injection surface: :meth:`inject_into_history` appends buffered
alerts as system messages that the next ``receive_response`` will see, and
:meth:`prime_startup` seeds the living rules once at startup. (This requires the
SDK's own Claude integration to be instrumented too, since it owns that history.)
"""

from __future__ import annotations

import logging
from typing import Any

from ._base import BaseSinkAdapter

__all__ = ["ClaudeAgentSDKAdapter"]

logger = logging.getLogger("pandaprobe_harness.adapters.claude_agent_sdk")


class ClaudeAgentSDKAdapter(BaseSinkAdapter):
    """Bridge ``PandaHarnessHook`` to a Claude Agent SDK client."""

    def __init__(self, *, session_id: str | None = None) -> None:
        super().__init__(session_id=session_id)
        self._instrumented = False

    # -- injection (via the SDK-maintained client history) -------------------

    @staticmethod
    def _history(client: Any) -> list[dict[str, Any]]:
        history = getattr(client, "_pandaprobe_history", None)
        if history is None:
            history = []
            client._pandaprobe_history = history
        return history

    def inject_into_history(self, client: Any) -> int:
        """Append buffered alerts as system messages to the client's history.

        Returns the number of alerts injected. Call between turns (after
        ``hook.drain_pending``) so the next ``receive_response`` sees them.
        """

        history = self._history(client)
        alerts = self.consume_alerts()
        for alert in alerts:
            history.append({"role": "system", "content": alert})
        return len(alerts)

    def prime_startup(self, client: Any) -> None:
        """Seed the living harness rules as a leading system message (once)."""

        preamble = self.startup_context_text()
        if preamble:
            self._history(client).insert(0, {"role": "system", "content": preamble})

    # -- turn detection ------------------------------------------------------

    def instrument(self) -> bool:
        """Monkey-patch ``receive_response`` to fire the hook at turn end.

        Idempotent. Returns ``False`` (and logs) if dependencies are missing.
        """

        try:
            import claude_agent_sdk  # noqa: F401
            from wrapt import wrap_function_wrapper
        except ImportError as exc:  # pragma: no cover - optional dep
            logger.warning("ClaudeAgentSDKAdapter.instrument: missing dependency — %s", exc)
            return False
        if self._instrumented:
            return True

        adapter = self

        def _wrap(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
            return adapter._instrument_stream(wrapped(*args, **kwargs))

        wrap_function_wrapper(
            "claude_agent_sdk", "ClaudeSDKClient.receive_response", _wrap
        )
        self._instrumented = True
        return True

    async def _instrument_stream(self, stream: Any) -> Any:
        """Delegate the async message stream, firing the hook when it completes."""

        try:
            async for message in stream:
                yield message
        finally:
            try:
                self.notify_turn_end()
            except RuntimeError:  # pragma: no cover - no running loop
                logger.debug("no running loop; on_turn_end skipped")
