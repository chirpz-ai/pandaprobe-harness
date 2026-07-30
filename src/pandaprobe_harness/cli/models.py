"""Typed views over the JSON payloads the harness consumes from the CLI.

These are forgiving, stdlib-only parsers (no pydantic): unknown fields are
ignored and missing optional fields default sensibly, so a CLI payload that
gains fields over time will not break parsing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["RunCreated", "ScoreRecord", "RunScores", "unwrap_items"]


def unwrap_items(payload: Any, *keys: str) -> list[Mapping[str, Any]]:
    """The CLI's list payloads, unwrapped forgivingly.

    Accepts either a bare list of objects or an object wrapping them under one of
    ``keys`` (the CLI uses ``items`` for paginated listings and ``scores`` for eval
    runs). Non-object entries are dropped rather than raising, so a payload that
    gains fields or nests differently degrades instead of breaking.
    """

    raw: Sequence[Any]
    if isinstance(payload, Mapping):
        raw = next(
            (
                value
                for key in keys
                if isinstance(value := payload.get(key), Sequence)
                and not isinstance(value, (str, bytes))
            ),
            [],
        )
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        raw = payload
    else:
        raw = []
    return [item for item in raw if isinstance(item, Mapping)]

# Score statuses that mean the platform has finished computing (terminal).
# The backend uses SUCCESS / FAILED / PENDING; synonyms kept for tolerance.
_TERMINAL_STATUSES = frozenset(
    {"success", "completed", "complete", "succeeded", "failed", "error"}
)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True, slots=True)
class RunCreated:
    """Result of ``evals runs batch`` / ``evals runs create`` — an async handle."""

    run_id: str
    status: str

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> RunCreated:
        run_id = payload.get("run_id") or payload.get("id") or ""
        if not run_id:
            raise ValueError(f"CLI run payload missing run_id: {payload!r}")
        return cls(run_id=str(run_id), status=str(payload.get("status", "pending")))


@dataclass(frozen=True, slots=True)
class ScoreRecord:
    """A single metric score within an eval run."""

    name: str
    value: float | None
    status: str
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    #: The trace this score was computed against. Present on trace-target
    #: payloads; ``None`` for session-target scores.
    trace_id: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status.lower() in _TERMINAL_STATUSES

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> ScoreRecord:
        metadata = payload.get("metadata")
        trace_id = payload.get("trace_id") or payload.get("target_id")
        return cls(
            name=str(payload.get("name", "")),
            value=_as_float(payload.get("value")),
            status=str(payload.get("status", "pending")),
            reason=payload.get("reason"),
            metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
            trace_id=str(trace_id) if trace_id is not None else None,
        )


@dataclass(frozen=True, slots=True)
class RunScores:
    """Result of ``evals runs scores <run-id>`` — zero or more metric scores."""

    run_id: str
    scores: tuple[ScoreRecord, ...]

    def is_terminal(self) -> bool:
        """True once every score has reached a terminal status.

        An empty score list is treated as non-terminal (still computing).
        """

        return bool(self.scores) and all(s.is_terminal for s in self.scores)

    def by_name(self, name: str) -> ScoreRecord | None:
        for score in self.scores:
            if score.name == name:
                return score
        return None

    def for_trace(self, name: str, trace_id: str) -> ScoreRecord | None:
        """Look a score up by ``(name, trace_id)``.

        Needed whenever a run covers more than one trace: ``by_name`` returns the
        first match, which would silently mis-attribute a multi-trace batch.
        Records without a ``trace_id`` still match on name alone so a payload
        that omits the field degrades rather than disappearing.
        """

        for score in self.scores:
            if score.name == name and (score.trace_id in (trace_id, None)):
                return score
        return None

    @classmethod
    def parse(cls, run_id: str, payload: Any) -> RunScores:
        if isinstance(payload, Mapping):
            run_id = str(payload.get("run_id", run_id))
        scores = tuple(
            ScoreRecord.parse(item) for item in unwrap_items(payload, "scores", "items")
        )
        return cls(run_id=run_id, scores=scores)
