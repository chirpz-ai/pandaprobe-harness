"""OpenAI Agents SDK adapter (optional ``openai-agents`` extra).

Instrumented via the OpenAI Agents SDK's first-class ``TracingProcessor``
interface: one ``Runner.run`` trace is one agent turn, and the hook fires on
trace end. Session identity comes from the SDK session ``ContextVar``.
Diagnostics reach the agent through the workspace mailbox + harness toolset
(register them as function tools via ``as_openai_function_tools``).
"""

from __future__ import annotations

import logging
from typing import Any

from ._base import BaseSinkAdapter

__all__ = ["OpenAIAgentsAdapter"]

logger = logging.getLogger("pandaprobe_harness.adapters.openai_agents")

# The trace processor is registered process-globally, so register it once and
# route to the most recently instrumented adapter (last-wins). Rebuilding a
# harness must not stack processors or keep a retired hook firing.
_registered = False
_active: OpenAIAgentsAdapter | None = None


class OpenAIAgentsAdapter(BaseSinkAdapter):
    """Bridge ``PandaHarnessHook`` to an OpenAI Agents SDK runner."""

    def __init__(self, *, session_id: str | None = None) -> None:
        super().__init__(session_id=session_id)
        self._instrumented = False

    # -- turn detection ------------------------------------------------------

    def instrument(self) -> bool:
        """Register a ``TracingProcessor`` that fires the hook on trace end.

        Idempotent and safe across many adapters: one processor is registered
        and always routes to the most recently instrumented adapter. Returns
        ``False`` (and logs) if dependencies are missing.
        """

        global _registered, _active
        try:
            from agents import tracing as agents_tracing
        except ImportError as exc:  # pragma: no cover - optional dep
            logger.warning("OpenAIAgentsAdapter.instrument: missing dependency — %s", exc)
            return False

        _active = self
        self._instrumented = True
        if _registered:
            return True

        class _HarnessProcessor(agents_tracing.TracingProcessor):  # type: ignore[misc]
            def on_trace_start(self, trace: Any) -> None:
                pass

            def on_trace_end(self, trace: Any) -> None:
                adapter = _active
                if adapter is None:
                    return
                try:
                    adapter.notify_turn_end(end_state={"trace": getattr(trace, "name", None)})
                except RuntimeError:  # pragma: no cover - no running loop
                    logger.debug("no running loop; on_turn_end skipped")

            def on_span_start(self, span: Any) -> None:
                pass

            def on_span_end(self, span: Any) -> None:
                pass

            def shutdown(self) -> None:
                pass

            def force_flush(self) -> None:
                pass

        agents_tracing.add_trace_processor(_HarnessProcessor())
        _registered = True
        return True
