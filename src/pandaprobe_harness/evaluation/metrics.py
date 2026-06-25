"""Metric definitions and the structured evaluation report.

Scores returned by the platform are in ``[0.0, 1.0]`` where **higher is better**.
A metric is *breached* when its score is strictly below its configured threshold.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = ["Metric", "MetricScore", "EvalReport"]


class Metric(StrEnum):
    """Registry names of the metrics this harness evaluates."""

    RELIABILITY = "agent_reliability"
    CONSISTENCY = "agent_consistency"

    @property
    def target(self) -> str:
        """The CLI ``--target`` scope for this metric.

        ``agent_reliability`` (TRACER) is evaluated over a turn's traces;
        ``agent_consistency`` aggregates across the whole session.
        """

        return "trace" if self is Metric.RELIABILITY else "session"


@dataclass(frozen=True, slots=True)
class MetricScore:
    """A single metric's score against its threshold for one turn."""

    metric: Metric
    value: float | None
    threshold: float
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def breached(self) -> bool:
        """True only when a concrete score sits below the threshold.

        A ``None`` value (pending/unresolved/degraded) is never a breach.
        """

        return self.value is not None and self.value < self.threshold

    @property
    def pending(self) -> bool:
        return self.value is None

    def to_dump(self) -> dict[str, Any]:
        return {
            "metric": str(self.metric),
            "value": self.value,
            "threshold": self.threshold,
            "breached": self.breached,
            "pending": self.pending,
            "reason": self.reason,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class EvalReport:
    """The combined evaluation outcome for a single completed turn."""

    session_id: str
    turn_index: int
    scores: tuple[MetricScore, ...] = ()

    @property
    def any_breach(self) -> bool:
        return any(score.breached for score in self.scores)

    @property
    def breached_scores(self) -> tuple[MetricScore, ...]:
        return tuple(score for score in self.scores if score.breached)

    def to_dump(self) -> dict[str, Any]:
        """Serializable payload written to ``traces/latest_eval.json``."""

        return {
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "any_breach": self.any_breach,
            "scores": [score.to_dump() for score in self.scores],
        }

    @classmethod
    def from_scores(
        cls, session_id: str, turn_index: int, scores: Sequence[MetricScore]
    ) -> EvalReport:
        return cls(session_id=session_id, turn_index=turn_index, scores=tuple(scores))
