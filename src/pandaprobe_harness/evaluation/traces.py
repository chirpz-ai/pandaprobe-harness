"""Trace discovery — the seam the trace-level trigger needs.

v1 never enumerated traces: trace ids arrived only as ``flagged_traces`` metadata
on a *session* score, which is exactly the dependency v2 removes. The platform
does expose an enumeration API (``traces list --session-id``), so this module
wraps it.

Two properties are load-bearing:

* **Chronological order.** ``new_traces`` re-sorts oldest-first even though the
  CLI is queried newest-first, because the trajectory gate folds peak/stall state
  one trace at a time and the EWMA store is order-sensitive.
* **Once-only delivery.** A per-session seen-set means a trace is Tier-1 scored
  exactly once no matter how often the barrier runs. The set is bounded and
  FIFO-evicted, so a long-lived process handling many short sessions cannot grow
  without limit.

Trace ingestion lags turn end (the SDK flushes on a background thread), so an
empty first result is normal rather than terminal: the listing is retried with
the same transient/backoff policy the evaluator uses.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..cli.client import CliClient
from ..cli.errors import CliError, is_transient
from ..cli.models import unwrap_items
from ..config import HarnessConfig

__all__ = ["TraceRef", "TraceLocator"]

logger = logging.getLogger("pandaprobe_harness.evaluation")

# Bound per-session bookkeeping (mirrors the hook's own cap).
_MAX_TRACKED_SESSIONS = 4096
# Cap on remembered ids per session, so one very long session is bounded too.
_MAX_SEEN_PER_SESSION = 2000


@dataclass(frozen=True, slots=True)
class TraceRef:
    """A completed trace of one session, as returned by ``traces list``.

    Only the id is kept: ordering comes from the server-side sort (plus the
    reversal in :meth:`TraceLocator.new_traces`), and completeness from the
    ``--status COMPLETED`` filter, so the other columns would be carried unused.
    """

    trace_id: str

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> TraceRef | None:
        raw_id = payload.get("trace_id") or payload.get("id")
        return cls(trace_id=str(raw_id)) if raw_id else None


class TraceLocator:
    """Enumerates a session's completed traces, newest-first from the platform."""

    def __init__(self, cli: CliClient, config: HarnessConfig) -> None:
        self._cli = cli
        self._config = config
        # Insertion-ordered so eviction is FIFO; values keep per-session order
        # for the same reason.
        self._seen: dict[str, dict[str, None]] = {}

    # -- public API -----------------------------------------------------------

    async def new_traces(self, session_id: str) -> list[TraceRef]:
        """Completed traces for ``session_id`` not yet returned, oldest-first."""

        listed = await self._list(session_id)
        seen = self._seen.setdefault(session_id, {})
        if len(self._seen) > _MAX_TRACKED_SESSIONS:
            self._evict_oldest_session()

        # `listed` arrives newest-first; reverse to fold gate state in the order
        # the traces actually happened.
        fresh = [ref for ref in reversed(listed) if ref.trace_id not in seen]
        for ref in fresh:
            seen[ref.trace_id] = None
        if len(seen) > _MAX_SEEN_PER_SESSION:
            for stale in list(seen)[: len(seen) - _MAX_SEEN_PER_SESSION]:
                del seen[stale]
        return fresh

    async def last_trace(self, session_id: str) -> TraceRef | None:
        """The most recently started completed trace of ``session_id``."""

        listed = await self._list(session_id, limit=1)
        return listed[0] if listed else None

    def forget(self, session_id: str) -> None:
        """Drop a session's seen-set (used when its bookkeeping is evicted)."""

        self._seen.pop(session_id, None)

    # -- CLI ------------------------------------------------------------------

    async def _list(self, session_id: str, *, limit: int | None = None) -> list[TraceRef]:
        """``traces list`` for one session, newest-first. Never raises."""

        effective = limit if limit is not None else max(1, self._config.trace_list_limit)
        args = [
            "traces", "list",
            "--session-id", session_id,
            "--status", "COMPLETED",
            "--sort-by", "started_at",
            "--sort-order", "desc",
            "--limit", str(effective),
        ]
        attempts = max(1, self._config.eval_retry_attempts)
        # Only a session we have never seen a trace for can be suffering ingestion
        # lag. Once one has landed, an empty listing is the truth — retrying it
        # would sleep out the backoff budget on every healthy turn that simply has
        # nothing new to score.
        cold = not self._seen.get(session_id)
        for attempt in range(attempts):
            try:
                result = await self._cli.run(*args)
            except CliError as exc:
                if is_transient(exc) and attempt + 1 < attempts:
                    await self._backoff(attempt)
                    continue
                logger.warning(
                    "trace listing degraded for session=%s: %s", session_id, exc
                )
                return []
            refs = [
                ref
                for ref in (
                    TraceRef.parse(item) for item in unwrap_items(result.json(), "items", "traces")
                )
                if ref is not None
            ]
            if not refs and cold and attempt + 1 < attempts:
                await self._backoff(attempt)
                continue
            return refs
        return []

    async def _backoff(self, attempt: int) -> None:
        await asyncio.sleep(self._config.eval_retry_backoff_s * (attempt + 1))

    def _evict_oldest_session(self) -> None:
        try:
            oldest = next(iter(self._seen))
        except StopIteration:  # pragma: no cover - guarded by the caller
            return
        self._seen.pop(oldest, None)


