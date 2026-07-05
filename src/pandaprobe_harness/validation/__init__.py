"""Closing the loop: evidence before trust.

``regression`` replays the captured eval-set against the current rule set;
``validator`` (added with the candidate lifecycle) promotes or retires
candidate rules based on replay or forward-trial evidence. Both share the
developer-supplied :data:`~pandaprobe_harness.workspace.evalset.ReplayFn`
seam and score replayed sessions directly through the ``MetricEvaluator`` —
never through the live hook's turn pipeline.
"""

from __future__ import annotations

from .regression import CaseResult, CaseStatus, RegressionReport, run_regression
from .validator import (
    ForwardTrialValidator,
    ReplayValidator,
    RuleValidator,
    ValidationEngine,
    ValidationVerdict,
    VerdictOutcome,
)

__all__ = [
    "CaseResult",
    "CaseStatus",
    "ForwardTrialValidator",
    "RegressionReport",
    "ReplayValidator",
    "RuleValidator",
    "ValidationEngine",
    "ValidationVerdict",
    "VerdictOutcome",
    "run_regression",
]
