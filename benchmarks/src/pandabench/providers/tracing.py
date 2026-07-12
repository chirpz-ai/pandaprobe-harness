"""PandaProbe tracing seam for LiteLLM calls — native wrapper (SDK >= 0.5).

PandaProbe 0.5 ships a native LiteLLM wrapper: ``wrap_litellm(litellm)``
monkey-patches module-level ``litellm.completion`` / ``acompletion`` so every
call automatically emits an LLM span (input messages, output, model, token
usage) — no manual span bookkeeping. We bind those auto-spans to the harness
session by running each call inside ``pandaprobe.session(session_id)``, whose
ContextVar the wrapped call inherits.

Verified against ``pandaprobe==0.5.0``:
- ``from pandaprobe.wrappers import wrap_litellm`` — idempotent, returns the module.
- ``pandaprobe.session(session_id)`` — context manager binding the session id.
- ``pandaprobe.get_client()`` — ``None`` when no client is configured (no
  ``PANDAPROBE_API_KEY`` / ``PANDAPROBE_PROJECT_NAME``), so tracing is guarded on
  client availability — arm A and offline tests (``disabled()``) never patch
  LiteLLM or open a session.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator

logger = logging.getLogger("pandabench.tracing")

__all__ = ["PandaTracer"]


def _client_available() -> bool:
    """True when the SDK can resolve/auto-init a client (creds present)."""

    try:
        import pandaprobe

        return pandaprobe.get_client() is not None
    except Exception as exc:  # pragma: no cover - defensive; SDK/network hiccup
        logger.debug("pandaprobe client unavailable: %s", exc)
        return False


def _patch_litellm() -> None:
    """Install the native LiteLLM wrapper (idempotent)."""

    try:
        import litellm
        from pandaprobe.wrappers import wrap_litellm

        wrap_litellm(litellm)
    except Exception as exc:  # pragma: no cover - never fail a run on telemetry setup
        logger.warning("wrap_litellm failed; LiteLLM calls will not be traced: %s", exc)


class PandaTracer:
    """Binds LiteLLM auto-spans to a harness session.

    Construct via :meth:`from_env` (enabled iff a PandaProbe client is available,
    which also installs the native wrapper once) or :meth:`disabled` to force a
    no-op (baseline arm, unit tests). The wrapper is the only thing that touches
    the SDK for tracing.
    """

    def __init__(self, *, enabled: bool) -> None:
        self._enabled = enabled
        if enabled:
            _patch_litellm()

    @classmethod
    def from_env(cls) -> PandaTracer:
        return cls(enabled=_client_available())

    @classmethod
    def disabled(cls) -> PandaTracer:
        return cls(enabled=False)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @contextlib.contextmanager
    def session(self, session_id: str | None) -> Iterator[None]:
        """Bind traces produced inside the block to ``session_id``.

        No-op when tracing is disabled or no session id is given. Never raises on
        a tracing error — a benchmark trial must not crash on telemetry.
        """

        if not self._enabled or not session_id:
            yield
            return
        try:
            import pandaprobe

            with pandaprobe.session(session_id):
                yield
        except Exception as exc:  # pragma: no cover - telemetry must never crash a trial
            logger.warning("pandaprobe session bind failed for %s: %s", session_id, exc)
            yield

    def flush(self, timeout: float = 30.0) -> None:
        """Block until buffered spans are sent (no-op when disabled).

        Called before the harness fires a session eval so scoring sees the full
        session, not one still sitting in the SDK's async send buffer.
        """

        if not self._enabled:
            return
        try:
            import pandaprobe

            pandaprobe.flush(timeout=timeout)
        except Exception as exc:  # pragma: no cover - telemetry must never crash a trial
            logger.warning("pandaprobe flush failed: %s", exc)
