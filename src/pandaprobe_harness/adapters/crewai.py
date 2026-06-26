"""CrewAI adapter (optional ``crewai`` extra).

CrewAI is instrumented by monkey-patching ``Crew.kickoff`` (the SDK uses the same
``wrapt`` approach) so a completed crew run fires ``hook.on_turn_end``; session
identity comes from the SDK session ``ContextVar``.

Injection is honest about CrewAI's constraints: there is no mid-crew message
queue, so alerts and the startup rules preamble are exposed as text
(:meth:`consume_context`, :meth:`startup_context_text`) for the developer to feed
into the next ``kickoff(inputs=...)`` / task context.
"""

from __future__ import annotations

import logging
from typing import Any

from ._base import BaseSinkAdapter

__all__ = ["CrewAIAdapter"]

logger = logging.getLogger("pandaprobe_harness.adapters.crewai")


class CrewAIAdapter(BaseSinkAdapter):
    """Bridge ``PandaHarnessHook`` to a CrewAI run."""

    _id_keys = ("crew_id",)

    def __init__(self, *, session_id: str | None = None) -> None:
        super().__init__(session_id=session_id)
        self._instrumented = False

    def consume_context(self) -> list[str]:
        """Pop pending alerts to feed into the next ``kickoff`` inputs/context."""

        return self.consume_alerts()

    def instrument(self) -> bool:
        """Monkey-patch ``Crew.kickoff`` to fire the hook on completion.

        Idempotent. Returns ``False`` (and logs) if the optional dependencies
        are unavailable.
        """

        try:
            import crewai  # noqa: F401
            from wrapt import wrap_function_wrapper
        except ImportError as exc:  # pragma: no cover - optional dep
            logger.warning("CrewAIAdapter.instrument: missing dependency — %s", exc)
            return False
        if self._instrumented:
            return True

        adapter = self

        def _after_kickoff(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
            result = wrapped(*args, **kwargs)
            try:
                adapter.notify_turn_end(end_state={"result": str(result)})
            except RuntimeError:
                # No running event loop (sync kickoff) — the harness hook
                # requires an async context to schedule evaluation.
                logger.debug("no running loop; on_turn_end skipped")
            return result

        wrap_function_wrapper("crewai", "Crew.kickoff", _after_kickoff)
        self._instrumented = True
        return True
