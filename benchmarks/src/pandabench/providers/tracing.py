"""PandaProbe manual tracing seam for LiteLLM calls.

The PandaProbe SDK auto-instruments only the *native* provider clients
(``wrap_openai`` / ``wrap_anthropic`` / ``wrap_gemini``); it has no LiteLLM
wrapper. Since the entire study routes every model call through
``litellm.acompletion``, we must open a PandaProbe trace + LLM span *by hand*
around each call so the resulting spans land under the harness's session id —
otherwise the platform sees no traces for the session and the harness (arm B)
is inert.

Verified against ``pandaprobe==0.4.0``:
- ``pandaprobe.session(session_id)`` — sync context manager binding the session.
- ``pandaprobe.start_trace(name, *, session_id=, input=) -> TraceContext``.
- ``TraceContext.span(name, *, kind, model) -> SpanContext`` with
  ``set_input / set_output / set_model / set_model_parameters /
  set_token_usage(prompt_tokens=, completion_tokens=) / set_cost(total=) /
  set_error``.
- ``start_trace`` **raises** ``RuntimeError`` when no client is configured
  (no ``PANDAPROBE_API_KEY`` / ``PANDAPROBE_PROJECT_NAME`` and no ``init()``),
  so tracing is guarded on client availability — arm A and offline tests that
  set ``enabled=False`` never touch the SDK.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from typing import Any, Protocol

logger = logging.getLogger("pandabench.tracing")

__all__ = ["PandaTracer", "SpanRecorder"]


class SpanRecorder(Protocol):
    """The subset of the SDK ``SpanContext`` surface the client writes to.

    Mirrored by :class:`_NullSpan` so the client code path is identical whether
    or not tracing is active.
    """

    def set_output(self, output: Any) -> None: ...
    def set_model_parameters(self, params: dict[str, Any]) -> None: ...
    def set_token_usage(self, *, prompt_tokens: int = 0, completion_tokens: int = 0) -> None: ...
    def set_cost(self, *, total: float) -> None: ...
    def set_error(self, error: str) -> None: ...


class _NullSpan:
    """No-op recorder used when tracing is disabled (arm A / no creds / tests)."""

    def set_output(self, output: Any) -> None:  # noqa: D102
        pass

    def set_model_parameters(self, params: dict[str, Any]) -> None:  # noqa: D102
        pass

    def set_token_usage(self, *, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        pass

    def set_cost(self, *, total: float) -> None:  # noqa: D102
        pass

    def set_error(self, error: str) -> None:  # noqa: D102
        pass


def _client_available() -> bool:
    """True when the SDK can resolve/auto-init a client (creds present)."""

    try:
        import pandaprobe

        return pandaprobe.get_client() is not None
    except Exception as exc:  # pragma: no cover - defensive; SDK/network hiccup
        logger.debug("pandaprobe client unavailable: %s", exc)
        return False


class PandaTracer:
    """Emits one trace with a single LLM span per model call, bound to a session.

    Kept fully self-contained here so the LiteLLM wrapper is the only code that
    imports the SDK for tracing. Construct via :meth:`from_env` (enabled iff a
    PandaProbe client is available) or explicitly with ``enabled=False`` to
    force a no-op (baseline arm, unit tests).
    """

    def __init__(self, *, enabled: bool) -> None:
        self._enabled = enabled

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
    def llm_call(
        self,
        *,
        session_id: str | None,
        model: str,
        messages: list[dict[str, Any]],
        name: str = "litellm.acompletion",
    ) -> Iterator[SpanRecorder]:
        """Open a session-bound trace + LLM span for one completion call.

        Yields a :class:`SpanRecorder` the client writes the response/usage to.
        On any tracing error, degrades to a no-op recorder rather than failing
        the model call (a benchmark trial must never crash on telemetry).
        """

        if not self._enabled or not session_id:
            yield _NullSpan()
            return

        try:
            import pandaprobe
        except Exception:  # pragma: no cover
            yield _NullSpan()
            return

        try:
            with pandaprobe.session(session_id):
                trace_input = {"messages": _last_inbound(messages)}
                with pandaprobe.start_trace(
                    name="agent-turn", session_id=session_id, input=trace_input
                ) as trace:
                    span = trace.span(name, kind=pandaprobe.SpanKind.LLM, model=model)
                    with span:
                        span.set_input({"messages": messages})
                        recorder = _TracingRecorder(trace, span)
                        yield recorder
                    trace.set_output({"messages": recorder.output_messages})
        except Exception as exc:  # pragma: no cover - telemetry must never crash a trial
            logger.warning("pandaprobe tracing failed for session %s: %s", session_id, exc)
            yield _NullSpan()


class _TracingRecorder:
    """Adapts an SDK ``SpanContext`` to :class:`SpanRecorder`, remembering the
    assistant message so the enclosing trace output can mirror it."""

    def __init__(self, trace: Any, span: Any) -> None:
        self._trace = trace
        self._span = span
        self.output_messages: list[dict[str, Any]] = []

    def set_output(self, output: Any) -> None:
        self._span.set_output(output)
        if isinstance(output, dict) and isinstance(output.get("messages"), list):
            self.output_messages = output["messages"]

    def set_model_parameters(self, params: dict[str, Any]) -> None:
        self._span.set_model_parameters(params)

    def set_token_usage(self, *, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self._span.set_token_usage(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )

    def set_cost(self, *, total: float) -> None:
        self._span.set_cost(total=total)

    def set_error(self, error: str) -> None:
        self._span.set_error(error)


def _last_inbound(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The most recent non-assistant message, as the trace's turn input."""

    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            return [msg]
    return messages[-1:] if messages else []
