"""Unit tests for the offline metric-calibration module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pandaprobe_harness import HarnessConfig
from pandaprobe_harness.calibration import (
    calibrate,
    collect_scores,
    labels_from_evalset,
    load_labels,
)
from pandaprobe_harness.workspace._io import atomic_write_json
from pandaprobe_harness.workspace.evalset import EvalSet
from tests.fakes.fake_cli_client import FakeCliClient

# 4 sessions at threshold 0.5: breach {s1, s2}; failed {s1, s3}.
# -> tp=1 (s1), fp=1 (s2), fn=1 (s3), tn=1 (s4): P = R = F1 = 0.5.
_SCORES = {
    "s1": {"agent_reliability": 0.30},
    "s2": {"agent_reliability": 0.40},
    "s3": {"agent_reliability": 0.70},
    "s4": {"agent_reliability": 0.80},
}
_LABELS = {"s1": True, "s2": False, "s3": True, "s4": False}


def test_labeled_confusion_at_configured_threshold() -> None:
    report = calibrate(_SCORES, config=HarnessConfig(), labels=_LABELS)

    (cal,) = report.metrics
    assert cal.metric == "agent_reliability"
    assert cal.threshold == 0.5
    labeled = cal.labeled
    assert labeled is not None
    assert (labeled.tp, labeled.fp, labeled.fn, labeled.tn) == (1, 1, 1, 1)
    assert labeled.precision == pytest.approx(0.5)
    assert labeled.recall == pytest.approx(0.5)
    assert labeled.f1 == pytest.approx(0.5)
    assert labeled.labeled_sessions == 4


def test_sweep_finds_best_f1_and_target_precision() -> None:
    # Failed sessions score 0.30/0.40, healthy ones 0.80/0.90: any threshold
    # in (0.40, 0.80] separates them perfectly. The sweep grid's first such
    # point is 0.45.
    scores = {
        "s1": {"agent_reliability": 0.30},
        "s2": {"agent_reliability": 0.40},
        "s3": {"agent_reliability": 0.80},
        "s4": {"agent_reliability": 0.90},
    }
    labels = {"s1": True, "s2": True, "s3": False, "s4": False}
    report = calibrate(scores, config=HarnessConfig(), labels=labels, target_precision=0.9)

    (cal,) = report.metrics
    labeled = cal.labeled
    assert labeled is not None
    assert labeled.best_f1 == pytest.approx(1.0)
    assert labeled.best_f1_threshold == pytest.approx(0.45)
    # Precision hits 1.0 as soon as any true breach is caught: 0.35 catches s1.
    assert labeled.target_precision_threshold == pytest.approx(0.35)

    by_threshold = {point.threshold: point for point in cal.sweep}
    assert by_threshold[0.45].f1 == pytest.approx(1.0)
    assert by_threshold[0.05].breach_count == 0
    assert by_threshold[0.95].breach_count == 4


def test_unlabeled_distribution_histogram_and_agreement() -> None:
    scores = {
        "s1": {"agent_reliability": 0.30, "agent_consistency": 0.40},
        "s2": {"agent_reliability": 0.90, "agent_consistency": 0.90},
        "s3": {"agent_reliability": 0.20, "agent_consistency": 0.80},
    }
    report = calibrate(scores, config=HarnessConfig())

    assert report.session_count == 3
    reliability = next(m for m in report.metrics if m.metric == "agent_reliability")
    assert reliability.labeled is None
    assert reliability.count == 3
    assert reliability.minimum == pytest.approx(0.2)
    assert reliability.maximum == pytest.approx(0.9)
    assert reliability.median == pytest.approx(0.3)
    assert sum(reliability.histogram) == 3
    assert reliability.histogram[2] == 1  # 0.20 lands in [0.2, 0.3)
    assert reliability.histogram[9] == 1  # 0.90 lands in the top bucket
    assert all(point.precision is None for point in reliability.sweep)

    # s1: both breach; s2: neither; s3: reliability breaches, consistency not.
    assert report.agreement == pytest.approx(2 / 3)


def test_calibrate_handles_empty_and_single_scores() -> None:
    report = calibrate({}, config=HarnessConfig())
    assert report.metrics == () and report.agreement is None

    single = calibrate({"s1": {"agent_reliability": 0.4}}, config=HarnessConfig())
    (cal,) = single.metrics
    assert cal.stdev is None  # needs >= 2 samples
    assert cal.mean == pytest.approx(0.4)


def test_load_labels_json_dict_list_and_csv(tmp_path: Path) -> None:
    as_dict = tmp_path / "labels.json"
    as_dict.write_text(json.dumps({"s1": True, "s2": False}), encoding="utf-8")
    assert load_labels(as_dict) == {"s1": True, "s2": False}

    as_list = tmp_path / "labels_list.json"
    as_list.write_text(
        json.dumps([{"session_id": "s1", "failed": "yes"}, {"session_id": "s2"}]),
        encoding="utf-8",
    )
    assert load_labels(as_list) == {"s1": True, "s2": False}

    as_csv = tmp_path / "labels.csv"
    as_csv.write_text("session_id,failed\ns1,true\ns2,0\n", encoding="utf-8")
    assert load_labels(as_csv) == {"s1": True, "s2": False}


def test_labels_from_evalset(tmp_path: Path) -> None:
    config = HarnessConfig(harness_root=tmp_path / "harness")
    evalset = EvalSet(config)
    evalset.capture(session_id="s-bad", signature=("breach:agent_reliability",))
    evalset.capture(session_id="s-good", kind="win", signature=("healthy",))

    assert labels_from_evalset(evalset) == {"s-bad": True, "s-good": False}


async def test_collect_scores_merges_with_cli_precedence(tmp_path: Path) -> None:
    config = HarnessConfig(harness_root=tmp_path / "harness")

    # Eval-set baseline (lowest precedence).
    evalset = EvalSet(config)
    evalset.capture(
        session_id="s1",
        signature=("breach:agent_reliability",),
        baseline_scores={"agent_reliability": 0.10},
    )
    # Local history store (middle precedence) — seeded on disk.
    atomic_write_json(
        config.history_file,
        {
            "s1::agent_reliability": {
                "series": [{"value": 0.20, "ts": "t", "run_id": None}],
                "ewma": None,
            },
            "s2::agent_consistency": {
                "series": [{"value": 0.60, "ts": "t", "run_id": None}],
                "ewma": None,
            },
        },
    )
    # Backend CLI (highest precedence).
    fake = FakeCliClient(
        scores_list_payload=[
            {"session_id": "s1", "name": "agent_reliability", "value": "0.30"},
            {"session_id": "s3", "metric": "agent_consistency", "value": 0.9},
            {"name": "agent_reliability", "value": 0.5},  # no session -> skipped
        ]
    )

    from pandaprobe_harness.evaluation.history import ScoreHistoryStore

    scores, sources = await collect_scores(
        fake, config, history=ScoreHistoryStore(config), evalset=evalset
    )

    assert sources == ("evalset", "history", "cli")
    assert scores["s1"]["agent_reliability"] == pytest.approx(0.30)  # cli wins
    assert scores["s2"]["agent_consistency"] == pytest.approx(0.60)
    assert scores["s3"]["agent_consistency"] == pytest.approx(0.90)


async def test_collect_scores_degrades_per_source(tmp_path: Path) -> None:
    from pandaprobe_harness.cli.errors import CliGeneralError

    config = HarnessConfig(harness_root=tmp_path / "harness")
    fake = FakeCliClient(
        error_on_prefix={("evals", "scores", "list"): CliGeneralError("down")}
    )
    scores, sources = await collect_scores(fake, config)
    assert scores == {} and sources == ()


def test_render_text_mentions_the_essentials() -> None:
    report = calibrate(_SCORES, config=HarnessConfig(), labels=_LABELS)
    text = report.render_text()
    assert "agent_reliability" in text
    assert "precision 0.50" in text
    assert "histogram" in text
    assert "sweep" in text
