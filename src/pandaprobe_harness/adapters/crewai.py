"""CrewAI adapter (optional ``crewai`` extra).

CrewAI is instrumented by monkey-patching ``Crew.kickoff`` (the SDK uses the same
``wrapt`` approach) so a completed crew run fires ``hook.on_turn_end``; session
identity comes from the SDK session ``ContextVar``. Diagnostics reach the agent
through the workspace mailbox + harness toolset, not through this adapter.
"""

from __future__ import annotations

import logging
from typing import Any

from ._base import BaseSinkAdapter

__all__ = ["CrewAIAdapter"]

logger = logging.getLogger("pandaprobe_harness.adapters.crewai")

# ``Crew.kickoff`` is a process-global symbol, so it must be patched at most
# once no matter how many adapters/harnesses are built. The single wrapper
# dispatches to whichever adapter registered most recently (last-wins), so
# rebuilding a harness never stacks wrappers or leaves a retired hook firing.
_patched = False
_active: CrewAIAdapter | None = None


class CrewAIAdapter(BaseSinkAdapter):
    """Bridge ``PandaHarnessHook`` to a CrewAI run."""

    _id_keys = ("crew_id",)

    def __init__(self, *, session_id: str | None = None) -> None:
        super().__init__(session_id=session_id)
        self._instrumented = False

    def instrument(self) -> bool:
        """Monkey-patch ``Crew.kickoff`` to fire the hook on completion.

        Idempotent and safe to call across many adapters: the global patch is
        applied once and always routes to the most recently instrumented
        adapter. Returns ``False`` (and logs) if the optional dependencies are
        unavailable.
        """

        global _patched, _active
        try:
            import crewai  # noqa: F401
            from wrapt import wrap_function_wrapper
        except ImportError as exc:  # pragma: no cover - optional dep
            logger.warning("CrewAIAdapter.instrument: missing dependency — %s", exc)
            return False

        _active = self
        self._instrumented = True
        if _patched:
            return True

        def _after_kickoff(wrapped: Any, instance: Any, args: Any, kwargs: Any) -> Any:
            result = wrapped(*args, **kwargs)
            adapter = _active
            if adapter is not None:
                try:
                    adapter.notify_turn_end(end_state={"result": str(result)})
                except RuntimeError:
                    # No running event loop (sync kickoff) — the harness hook
                    # requires an async context to schedule evaluation.
                    logger.debug("no running loop; on_turn_end skipped")
            return result

        wrap_function_wrapper("crewai", "Crew.kickoff", _after_kickoff)
        _patched = True
        return True
