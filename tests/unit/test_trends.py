from __future__ import annotations

from pathlib import Path

from pandaprobe_harness import HarnessConfig
from pandaprobe_harness.evaluation.history import ScoreHistoryStore
from pandaprobe_harness.evaluation.trends import TrendDetector


def _detector(tmp_path: Path, **kw: object) -> TrendDetector:
    cfg = HarnessConfig(harness_root=tmp_path / "h", trend_min_samples=4, **kw)  # type: ignore[arg-type]
    return TrendDetector(cfg, ScoreHistoryStore(cfg))


def test_declining_series_triggers_after_min_samples(tmp_path: Path) -> None:
    det = _detector(tmp_path)
    verdicts = [det.update("s", "agent_reliability", v) for v in (0.80, 0.72, 0.64, 0.56)]
    assert not any(v.declining for v in verdicts[:3])  # below min_samples
    assert verdicts[-1].declining


def test_flat_series_does_not_trigger(tmp_path: Path) -> None:
    det = _detector(tmp_path)
    verdicts = [det.update("s", "m", 0.8) for _ in range(6)]
    assert not any(v.declining for v in verdicts)


def test_rising_series_does_not_trigger(tmp_path: Path) -> None:
    det = _detector(tmp_path)
    verdicts = [det.update("s", "m", v) for v in (0.4, 0.5, 0.6, 0.7, 0.8)]
    assert not any(v.declining for v in verdicts)


def test_adaptive_relative_breach(tmp_path: Path) -> None:
    det = _detector(tmp_path, adaptive_threshold=True, adaptive_margin_drop=0.15)
    for v in (0.9, 0.9, 0.9, 0.9):
        det.update("s", "m", v)
    verdict = det.update("s", "m", 0.6)  # 0.6 < baseline(~0.9) - 0.15
    assert verdict.relative_breach


def test_relative_breach_off_by_default(tmp_path: Path) -> None:
    det = _detector(tmp_path)  # adaptive_threshold defaults False
    for v in (0.9, 0.9, 0.9, 0.9):
        det.update("s", "m", v)
    assert not det.update("s", "m", 0.6).relative_breach


def test_percentile_breach_when_enabled(tmp_path: Path) -> None:
    det = _detector(tmp_path, percentile_window=5, percentile_floor=0.25)
    for v in (0.9, 0.85, 0.8, 0.88):
        det.update("s", "m", v)
    verdict = det.update("s", "m", 0.5)  # well below the window's 25th percentile
    assert verdict.percentile_breach
