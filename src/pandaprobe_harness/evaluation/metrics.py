"""Metric definitions and the structured evaluation report.

Scores returned by the platform are in ``[0.0, 1.0]`` where **higher is better**.
A metric is *breached* when its score is strictly below its configured threshold.

``agent_reliability`` and ``agent_consistency`` are both **session-level** metrics
on the platform (registered via ``@register_session_metric``): they are evaluated
by ``session_id`` and aggregate the per-trace signals ``confidence``,
``coherence``, ``loop_detection`` and ``tool_correctness``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = ["Metric", "MetricScore", "EvalReport", "SIGNAL_NAMES"]

# The trace-level signals the platform aggregates into the session metrics.
SIGNAL_NAMES: tuple[str, ...] = (
    "confidence",
    "coherence",
    "loop_detection",
    "tool_correctness",
)


class Metric(StrEnum):
    """Registry names of the session metrics this harness evaluates."""

    RELIABILITY = "agent_reliability"
    CONSISTENCY = "agent_consistency"

    @property
    def target(self) -> str:
        """The CLI ``--target`` scope. Both metrics are session-scoped."""

        return "session"


@dataclass(frozen=True, slots=True)
class MetricScore:
    """A single metric's score against its threshold for one turn."""

    metric: Metric
    value: float | None
    threshold: float
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Set by the trend detector; an absolute breach is not the only trigger.
    trend_declining: bool = False
    relative_breach: bool = False
    # Soft corroborator (latest score in the low tail of its recent window);
    # alert-worthy but advisory — never escalates to a critical SYSTEM alert.
    percentile_breach: bool = False

    @property
    def breached(self) -> bool:
        """True only when a concrete score sits below the absolute threshold.

        A ``None`` value (pending/unresolved/degraded) is never a breach.
        """

        return self.value is not None and self.value < self.threshold

    @property
    def pending(self) -> bool:
        return self.value is None

    @property
    def alerting(self) -> bool:
        """Any condition warranting an alert: absolute, relative, trend, or percentile."""

        return (
            self.breached
            or self.relative_breach
            or self.trend_declining
            or self.percentile_breach
        )

    @property
    def flagged_traces(self) -> list[str]:
        raw = self.metadata.get("flagged_traces")
        return [str(t) for t in raw] if isinstance(raw, list) else []

    @property
    def per_trace_signals(self) -> dict[str, Any]:
        raw = self.metadata.get("per_trace_signals")
        return dict(raw) if isinstance(raw, dict) else {}

    @property
    def aggregation(self) -> dict[str, Any]:
        raw = self.metadata.get("aggregation")
        return dict(raw) if isinstance(raw, dict) else {}

    def to_dump(self) -> dict[str, Any]:
        return {
            "metric": str(self.metric),
            "value": self.value,
            "threshold": self.threshold,
            "breached": self.breached,
            "relative_breach": self.relative_breach,
            "trend_declining": self.trend_declining,
            "percentile_breach": self.percentile_breach,
            "pending": self.pending,
            "reason": self.reason,
            "flagged_traces": self.flagged_traces,
            "aggregation": self.aggregation,
            "per_trace_signals": self.per_trace_signals,
        }


@dataclass(frozen=True, slots=True)
class EvalReport:
    """The combined evaluation outcome for a single completed turn."""

    session_id: str
    turn_index: int
    scores: tuple[MetricScore, ...] = ()

    @property
    def any_breach(self) -> bool:
        """Absolute-threshold breach on any metric."""

        return any(score.breached for score in self.scores)

    @property
    def any_alert(self) -> bool:
        """Any alerting condition (absolute, relative, or trend)."""

        return any(score.alerting for score in self.scores)

    @property
    def breached_scores(self) -> tuple[MetricScore, ...]:
        return tuple(score for score in self.scores if score.breached)

    @property
    def alerting_scores(self) -> tuple[MetricScore, ...]:
        return tuple(score for score in self.scores if score.alerting)

    @property
    def flagged_traces(self) -> list[str]:
        """Union of flagged trace ids across all scores, de-duplicated, ordered."""

        seen: dict[str, None] = {}
        for score in self.scores:
            for trace_id in score.flagged_traces:
                seen.setdefault(trace_id, None)
        return list(seen)

    def signal_breakdown(self) -> dict[str, dict[str, Any]]:
        """Merge ``per_trace_signals`` across scores → ``{trace_id: {signal: ...}}``.

        Surfaces the four trace-level signals (``confidence``, ``coherence``,
        ``loop_detection``, ``tool_correctness``) the platform aggregated into
        the session metrics — no extra CLI calls, the data is already in the
        session score metadata.
        """

        merged: dict[str, dict[str, Any]] = {}
        for score in self.scores:
            for trace_id, signals in score.per_trace_signals.items():
                if isinstance(signals, dict):
                    merged.setdefault(trace_id, {}).update(signals)
        return merged

    def to_dump(self) -> dict[str, Any]:
        """Serializable payload written to ``traces/latest_eval.json``."""

        return {
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "any_breach": self.any_breach,
            "any_alert": self.any_alert,
            "flagged_traces": self.flagged_traces,
            "signal_breakdown": self.signal_breakdown(),
            "scores": [score.to_dump() for score in self.scores],
        }

    @classmethod
    def from_scores(
        cls, session_id: str, turn_index: int, scores: Sequence[MetricScore]
    ) -> EvalReport:
        return cls(session_id=session_id, turn_index=turn_index, scores=tuple(scores))
