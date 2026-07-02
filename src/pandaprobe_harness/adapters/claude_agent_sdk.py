"""Claude Agent SDK adapter (optional ``claude-agent-sdk`` extra).

Instrumented by monkey-patching ``ClaudeSDKClient.receive_response`` (the SDK uses
the same ``wrapt`` approach): a completed ``receive_response`` stream is one agent
turn, after which the hook fires. Session identity comes from the SDK session
``ContextVar``. Diagnostics reach the agent through the workspace mailbox +
harness toolset (register them via ``as_anthropic_tools``).
"""

from __future__ import annotations

import logging
from typing import Any

from ._base import BaseSinkAdapter

__all__ = ["ClaudeAgentSDKAdapter"]

logger = logging.getLogger("pandaprobe_harness.adapters.claude_agent_sdk")

# ``ClaudeSDKClient.receive_response`` is patched process-globally, so patch it
# once and route to the most recently instrumented adapter (last-wins), so
# rebuilding a harness never stacks wrappers or keeps a retired hook firing.
_patched = False
_active: ClaudeAgentSDKAdapter | None = None


class ClaudeAgentSDKAdapter(BaseSinkAdapter):
    """Bridge ``PandaHarnessHook`` to a Claude Agent SDK client."""

    def __init__(self, *, session_id: str | None = None) -> None:
        super().__init__(session_id=session_id)
        self._instrumented = False

    # -- turn detection ------------------------------------------------------

    def instrument(self) -> bool:
        """Monkey-patch ``receive_response`` to fire the hook at turn end.

        Idempotent and safe across many adapters: the global patch is applied
        once and always routes to the most recently instrumented adapter.
        Returns ``False`` (and logs) if dependencies are missing.
        """

        global _patched, _active
        try:
            import claude_agent_sdk  # noqa: F401
            from wrapt import wrap_function_wrapper
        except ImportError as exc:  # pragma: no cover - optional dep
            logger.warning("ClaudeAgentSDKAdapter.instrument: missing dependency — %s", exc)
            return False

        _active = self
        self._instrumented = True
        if _patched:
            return True

        def _wrap(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
            adapter = _active
            if adapter is None:
                return wrapped(*args, **kwargs)
            return adapter._instrument_stream(wrapped(*args, **kwargs))

        wrap_function_wrapper(
            "claude_agent_sdk", "ClaudeSDKClient.receive_response", _wrap
        )
        _patched = True
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
