"""Per-turn metric evaluation against the PandaProbe platform."""

from .evaluator import MetricEvaluator
from .history import EwmaState, ScoreHistoryStore, ScoreSample
from .metrics import SIGNAL_NAMES, EvalReport, Metric, MetricScore
from .thresholds import is_breach
from .trends import TrendDetector, TrendVerdict

__all__ = [
    "MetricEvaluator",
    "EvalReport",
    "Metric",
    "MetricScore",
    "SIGNAL_NAMES",
    "is_breach",
    "ScoreHistoryStore",
    "ScoreSample",
    "EwmaState",
    "TrendDetector",
    "TrendVerdict",
]
