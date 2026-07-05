"""Metric-calibration demo — is a `breach` a real failure? Is 0.5 sane?

    python examples/calibration_demo.py

Everything the harness does — notices, candidate validation, regression
classification — keys off "score below threshold". This offline demo shows
how the operator checks that trigger against ground truth:

  1. a labeled run: precision/recall/F1 of the breach predicate at the
     configured threshold, plus a threshold sweep recommending the
     F1-maximizing threshold and the lowest threshold hitting a target
     precision;
  2. an unlabeled run: the score distribution, histogram, and inter-metric
     agreement — enough to pick a threshold sanely without labels.

In production the same analysis runs as `pandaprobe-harness-calibrate`
(scores come from the platform CLI / local history / the eval-set; labels
from a JSON/CSV file or `--from-evalset` proxy labels). Here the scores are
inline so the demo is fully self-contained.
"""

from __future__ import annotations

from pandaprobe_harness import HarnessConfig, calibrate

# Twelve sessions: reliability separates failures (~0.2-0.45) from healthy
# ones (~0.55-0.9) imperfectly — like real data.
SCORES: dict[str, dict[str, float]] = {
    "s-01": {"agent_reliability": 0.20, "agent_consistency": 0.35},
    "s-02": {"agent_reliability": 0.30, "agent_consistency": 0.45},
    "s-03": {"agent_reliability": 0.35, "agent_consistency": 0.30},
    "s-04": {"agent_reliability": 0.45, "agent_consistency": 0.60},  # missed at 0.4
    "s-05": {"agent_reliability": 0.55, "agent_consistency": 0.50},  # noisy healthy
    "s-06": {"agent_reliability": 0.60, "agent_consistency": 0.70},
    "s-07": {"agent_reliability": 0.65, "agent_consistency": 0.75},
    "s-08": {"agent_reliability": 0.70, "agent_consistency": 0.65},
    "s-09": {"agent_reliability": 0.75, "agent_consistency": 0.80},
    "s-10": {"agent_reliability": 0.80, "agent_consistency": 0.85},
    "s-11": {"agent_reliability": 0.85, "agent_consistency": 0.90},
    "s-12": {"agent_reliability": 0.90, "agent_consistency": 0.95},
}

# Ground truth: which sessions actually failed (operator-labeled).
LABELS: dict[str, bool] = {
    "s-01": True,
    "s-02": True,
    "s-03": True,
    "s-04": True,   # a real failure scoring just above some thresholds
    "s-05": False,  # a healthy session scoring low-ish
    **{f"s-{i:02d}": False for i in range(6, 13)},
}


def main() -> None:
    config = HarnessConfig()  # configured thresholds: 0.5 / 0.5

    print("=== labeled calibration (precision/recall/F1 + threshold sweep) ===\n")
    labeled = calibrate(SCORES, config=config, labels=LABELS, target_precision=0.9)
    print(labeled.render_text())

    print("\n\n=== unlabeled calibration (distribution + sweep + agreement) ===\n")
    unlabeled = calibrate(SCORES, config=config)
    print(unlabeled.render_text())

    reliability = next(m for m in labeled.metrics if m.metric == "agent_reliability")
    assert reliability.labeled is not None
    print(
        f"\nrecommendation: agent_reliability threshold {reliability.threshold:.2f} -> "
        f"F1 {reliability.labeled.f1:.2f}; best F1 {reliability.labeled.best_f1:.2f} "
        f"at {reliability.labeled.best_f1_threshold:.2f}"
    )


if __name__ == "__main__":
    main()
