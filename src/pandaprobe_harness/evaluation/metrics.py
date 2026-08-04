"""Metric definitions and the structured evaluation report.

Scores returned by the platform are in ``[0.0, 1.0]`` where **higher is better**.
A metric is *breached* when its score is strictly below its configured threshold.

The platform metrics are trace-scoped. The tiered trigger evaluates completed
traces directly, preserving the trajectory and step-level signals that session
aggregation obscures.

``outcome_correct`` is neither: it is the synthetic score produced locally by an
optional developer-supplied verifier, so its target is ``"local"``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "ABSOLUTE_BREACH_TIERS",
    "EvalReport",
    "Metric",
    "MetricScore",
    "TRACE_METRIC_NAMES",
]

#: Which tiers breach on a score's *absolute* value. The one place this policy
#: lives, so every consumer agrees on it:
#:
#: * **0** (verifier outcome) — yes, a low value is the finding.
#: * **1** — no. Tier 1 breaches on *trajectory* only: an agent three steps into a
#:   task has legitimately not completed it, so a low early ``task_completion`` is
#:   expected, not a fault. Its verdict is ``stalled`` / ``regressed``. Treating the
#:   floor as a breach is exactly the point-threshold behaviour v2 removes.
#: * **2** — yes. This is the surgical step-level diagnosis.
#: * **3** — no. Tier 3 only ever runs to *explain* a confirmed Tier-2 breach, so it
#:   must never be a breach source of its own.
ABSOLUTE_BREACH_TIERS: frozenset[int] = frozenset({0, 2})

# Exactly the metrics `evals metrics --target trace` reports as runnable.
# `loop_detection` is deliberately absent: it is not trace-runnable (HTTP 422)
# and only exists inside the session composites.
TRACE_METRIC_NAMES: frozenset[str] = frozenset(
    {
        "task_completion",
        "coherence",
        "tool_correctness",
        "argument_correctness",
        "plan_adherence",
        "plan_quality",
        "step_efficiency",
        "confidence",
    }
)


class Metric(StrEnum):
    """Registry names of the metrics this harness evaluates."""

    # Trace metrics.
    TASK_COMPLETION = "task_completion"
    COHERENCE = "coherence"
    TOOL_CORRECTNESS = "tool_correctness"
    ARGUMENT_CORRECTNESS = "argument_correctness"
    PLAN_ADHERENCE = "plan_adherence"
    PLAN_QUALITY = "plan_quality"
    STEP_EFFICIENCY = "step_efficiency"
    CONFIDENCE = "confidence"

    # Locally-computed outcome score from an optional developer verifier.
    OUTCOME = "outcome_correct"

    @property
    def target(self) -> str:
        """The CLI ``--target`` scope this metric is evaluated against."""

        if self is Metric.OUTCOME:
            return "local"
        return "trace"


@dataclass(frozen=True, slots=True)
class MetricScore:
    """A single metric's score against its threshold for one turn."""

    metric: Metric
    value: float | None
    threshold: float
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # The trace this score was computed against (trace-target metrics only).
    trace_id: str | None = None
    # Which tier produced this score: 1/2/3 for the trace tiers, 0 for the
    # a verifier's synthetic outcome score.
    tier: int = 0
    # Set by the trajectory gate on Tier-1 metrics. Neither is an absolute
    # breach: `stalled` means no gain toward the target across the window,
    # `regressed` means a drop from the running peak.
    stalled: bool = False
    regressed: bool = False

    @property
    def gate_breached(self) -> bool:
        """True when the trajectory gate fired for this score."""

        return self.stalled or self.regressed

    @property
    def below_threshold(self) -> bool:
        """The raw comparison: a concrete score strictly below its threshold.

        A ``None`` value (pending/unresolved/degraded) is never below.
        """

        return self.value is not None and self.value < self.threshold

    @property
    def breached(self) -> bool:
        """True when this score's absolute floor counts as a breach.

        Only the tiers in :data:`ABSOLUTE_BREACH_TIERS` breach on their absolute
        value; for the others ``below_threshold`` is diagnostic only (and is
        dumped as such).
        """

        return self.below_threshold and self.tier in ABSOLUTE_BREACH_TIERS

    @property
    def conditions(self) -> tuple[str, ...]:
        """Every alerting condition this score satisfies, in severity order.

        The single source of the condition vocabulary. Notice metrics render it
        directly and breach *signatures* are ``f"{condition}:{metric}"`` over it,
        so a new detector is added here once rather than in every consumer.
        """

        out: list[str] = []
        if self.breached:
            out.append("breach")
        if self.stalled:
            out.append("stall")
        if self.regressed:
            out.append("regression")
        return tuple(out)

    @property
    def pending(self) -> bool:
        return self.value is None

    @property
    def alerting(self) -> bool:
        """Any condition warranting an alert."""

        return bool(self.conditions)

    @property
    def flagged_traces(self) -> list[str]:
        raw = self.metadata.get("flagged_traces")
        return [str(t) for t in raw] if isinstance(raw, list) else []

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
            "below_threshold": self.below_threshold,
            "stalled": self.stalled,
            "regressed": self.regressed,
            "pending": self.pending,
            "reason": self.reason,
            "trace_id": self.trace_id,
            "tier": self.tier,
            "flagged_traces": self.flagged_traces,
            "aggregation": self.aggregation,
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
        """Any alerting condition (breach, stall, or regression)."""

        return any(score.alerting for score in self.scores)

    @property
    def breached_scores(self) -> tuple[MetricScore, ...]:
        return tuple(score for score in self.scores if score.breached)

    @property
    def alerting_scores(self) -> tuple[MetricScore, ...]:
        return tuple(score for score in self.scores if score.alerting)

    @property
    def flagged_traces(self) -> list[str]:
        """Trace ids implicated by this report, de-duplicated, order-preserving.

        A score can identify a trace through platform metadata or its own
        trace-scoped ``trace_id``. The latter is what makes a tiered notice point
        at the trace the agent needs to inspect.
        """

        seen: dict[str, None] = {}
        for score in self.scores:
            for trace_id in score.flagged_traces:
                seen.setdefault(trace_id, None)
            if score.trace_id and score.alerting:
                seen.setdefault(score.trace_id, None)
        return list(seen)

    @property
    def gate_breached(self) -> bool:
        """True when the trajectory gate fired on any score."""

        return any(score.gate_breached for score in self.scores)

    def scores_for_tier(self, tier: int) -> tuple[MetricScore, ...]:
        return tuple(score for score in self.scores if score.tier == tier)

    def signal_breakdown(self) -> dict[str, dict[str, Any]]:
        """Per-trace detail merged across scores → ``{trace_id: {name: ...}}``.

        Each resolved trace-scoped score is keyed by its own ``trace_id``. This
        costs no extra CLI calls.
        """

        merged: dict[str, dict[str, Any]] = {}
        for score in self.scores:
            if score.trace_id and score.value is not None:
                merged.setdefault(score.trace_id, {})[str(score.metric)] = score.value
        return merged

    def to_dump(self) -> dict[str, Any]:
        """Serializable payload written to ``traces/latest_eval.json``."""

        return {
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "any_breach": self.any_breach,
            "any_alert": self.any_alert,
            "gate_breached": self.gate_breached,
            "flagged_traces": self.flagged_traces,
            "signal_breakdown": self.signal_breakdown(),
            "scores": [score.to_dump() for score in self.scores],
        }

    @classmethod
    def from_scores(
        cls, session_id: str, turn_index: int, scores: Sequence[MetricScore]
    ) -> EvalReport:
        return cls(session_id=session_id, turn_index=turn_index, scores=tuple(scores))
