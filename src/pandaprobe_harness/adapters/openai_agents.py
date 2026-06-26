"""OpenAI Agents SDK adapter (optional ``openai-agents`` extra).

Instrumented via the OpenAI Agents SDK's first-class ``TracingProcessor``
interface: one ``Runner.run`` trace is one agent turn, and the hook fires on
trace end. Session identity comes from the SDK session ``ContextVar``.

**Injection is observation-only.** The OpenAI Agents SDK abstracts away the
conversation state and exposes no in-flight injection point, so the harness
cannot transparently inject an alert. Instead, buffered alerts are exposed as
input items (:meth:`consume_input_items`) that the developer **prepends to the
next ``Runner.run`` input** themselves; :meth:`startup_input_items` does the same
for the living rules. This constraint is inherent to the framework, not the
adapter.
"""

from __future__ import annotations

import logging
from typing import Any

from ._base import BaseSinkAdapter

__all__ = ["OpenAIAgentsAdapter"]

logger = logging.getLogger("pandaprobe_harness.adapters.openai_agents")


class OpenAIAgentsAdapter(BaseSinkAdapter):
    """Bridge ``PandaHarnessHook`` to an OpenAI Agents SDK runner."""

    def __init__(self, *, session_id: str | None = None) -> None:
        super().__init__(session_id=session_id)
        self._instrumented = False

    # -- injection (manual; the SDK exposes no in-flight hook) ---------------

    def consume_input_items(self) -> list[dict[str, str]]:
        """Buffered alerts as input items to PREPEND to the next ``Runner.run`` input."""

        return [{"role": "system", "content": alert} for alert in self.consume_alerts()]

    def startup_input_items(self) -> list[dict[str, str]]:
        """The living harness rules as a leading input item (once at startup)."""

        preamble = self.startup_context_text()
        return [{"role": "system", "content": preamble}] if preamble else []

    # -- turn detection ------------------------------------------------------

    def instrument(self) -> bool:
        """Register a ``TracingProcessor`` that fires the hook on trace end.

        Idempotent. Returns ``False`` (and logs) if dependencies are missing.
        """

        try:
            from agents import tracing as agents_tracing
        except ImportError as exc:  # pragma: no cover - optional dep
            logger.warning("OpenAIAgentsAdapter.instrument: missing dependency — %s", exc)
            return False
        if self._instrumented:
            return True

        adapter = self

        class _HarnessProcessor(agents_tracing.TracingProcessor):  # type: ignore[misc]
            def on_trace_start(self, trace: Any) -> None:
                pass

            def on_trace_end(self, trace: Any) -> None:
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
        self._instrumented = True
        return True
