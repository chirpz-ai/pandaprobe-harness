"""Offline metric calibration: does a ``breach`` mean a real failure?

Everything downstream of the harness — notices, candidate validation,
regression classification — keys off "score below threshold". This module
gives the operator the evidence to trust (or retune) that trigger. It is an
out-of-band diagnostic, not a runtime gate, and keeps the healing loop fully
automatic.

With ground-truth labels (``session_id -> failed``, supplied as JSON/CSV or
proxied from the eval-set's failure/win kinds) it reports precision / recall
/ F1 of the breach predicate per metric, a confusion matrix, and a threshold
sweep with the F1-maximizing threshold and the lowest threshold hitting a
target precision. Without labels it reports the score distribution, a
histogram, the breach count at every candidate threshold, and inter-metric
agreement — enough to pick a threshold sanely.

Scores come from three sources, merged with precedence CLI > local history >
eval-set baselines; every source degrades gracefully. All statistics are
stdlib (``statistics`` + hand-rolled counts) — zero dependencies.

``main`` is the ``pandaprobe-harness-calibrate`` operator CLI over the
env-configured workspace (``HARNESS_*``).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

from .cli.client import CliClient
from .cli.errors import CliError
from .config import HarnessConfig
from .evaluation.history import ScoreHistoryStore
from .workspace._io import load_json
from .workspace.evalset import EvalSet

__all__ = [
    "CalibrationReport",
    "LabeledStats",
    "MetricCalibration",
    "ThresholdPoint",
    "calibrate",
    "collect_scores",
    "labels_from_evalset",
    "load_labels",
    "main",
]

logger = logging.getLogger("pandaprobe_harness.calibration")

#: Candidate thresholds swept per metric: 0.05, 0.10, ... 0.95.
_SWEEP_GRID: tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))
_HISTOGRAM_BUCKETS = 10
_TRUTHY = {"1", "true", "yes", "y"}


@dataclass(frozen=True, slots=True)
class ThresholdPoint:
    """One candidate threshold's behavior over the score set."""

    threshold: float
    breach_count: int
    precision: float | None = None  # labeled runs only
    recall: float | None = None
    f1: float | None = None
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "breach_count": self.breach_count,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
        }


@dataclass(frozen=True, slots=True)
class LabeledStats:
    """Breach-vs-label quality at the configured threshold, plus sweep picks."""

    labeled_sessions: int
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int
    best_f1_threshold: float
    best_f1: float
    target_precision: float
    target_precision_threshold: float | None

    def to_json(self) -> dict[str, Any]:
        return {
            "labeled_sessions": self.labeled_sessions,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "best_f1_threshold": self.best_f1_threshold,
            "best_f1": self.best_f1,
            "target_precision": self.target_precision,
            "target_precision_threshold": self.target_precision_threshold,
        }


@dataclass(frozen=True, slots=True)
class MetricCalibration:
    """One metric's calibration: distribution, sweep, and (optional) labels."""

    metric: str
    threshold: float  # the currently-configured breach threshold
    count: int
    minimum: float | None
    maximum: float | None
    mean: float | None
    median: float | None
    stdev: float | None
    histogram: tuple[int, ...]  # 10 fixed buckets over [0.0, 1.0)
    sweep: tuple[ThresholdPoint, ...]
    labeled: LabeledStats | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "threshold": self.threshold,
            "count": self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "median": self.median,
            "stdev": self.stdev,
            "histogram": list(self.histogram),
            "sweep": [point.to_json() for point in self.sweep],
            "labeled": self.labeled.to_json() if self.labeled is not None else None,
        }


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """The full calibration outcome across metrics."""

    generated_at: str
    sources: tuple[str, ...]  # subset of ("cli", "history", "evalset")
    session_count: int
    metrics: tuple[MetricCalibration, ...]
    agreement: float | None  # fraction of fully-scored sessions where all
    # metrics agree on breach/no-breach at their configured thresholds

    def to_json(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "sources": list(self.sources),
            "session_count": self.session_count,
            "metrics": [metric.to_json() for metric in self.metrics],
            "agreement": self.agreement,
        }

    def render_text(self) -> str:
        lines = [
            f"Calibration over {self.session_count} session(s) "
            f"(sources: {', '.join(self.sources) or 'none'})",
        ]
        for cal in self.metrics:
            lines.append("")
            lines.append(
                f"{cal.metric}  (configured threshold {cal.threshold:.2f}, "
                f"{cal.count} scores)"
            )
            if cal.count and cal.mean is not None:
                stdev_label = f"{cal.stdev:.2f}" if cal.stdev is not None else "n/a"
                lines.append(
                    f"  distribution: min {cal.minimum:.2f}  median {cal.median:.2f}  "
                    f"mean {cal.mean:.2f}  max {cal.maximum:.2f}  stdev {stdev_label}"
                )
                lines.append(
                    "  histogram [0.0..1.0): " + " ".join(str(n) for n in cal.histogram)
                )
            labeled = cal.labeled
            if labeled is not None:
                lines.append(
                    f"  at threshold {cal.threshold:.2f}: precision {labeled.precision:.2f}  "
                    f"recall {labeled.recall:.2f}  F1 {labeled.f1:.2f}  "
                    f"(tp {labeled.tp} fp {labeled.fp} fn {labeled.fn} tn {labeled.tn} "
                    f"over {labeled.labeled_sessions} labeled)"
                )
                lines.append(
                    f"  best F1 {labeled.best_f1:.2f} at threshold "
                    f"{labeled.best_f1_threshold:.2f}"
                )
                if labeled.target_precision_threshold is not None:
                    lines.append(
                        f"  precision ≥ {labeled.target_precision:.2f} first at "
                        f"threshold {labeled.target_precision_threshold:.2f}"
                    )
                else:
                    lines.append(
                        f"  precision ≥ {labeled.target_precision:.2f}: unreachable "
                        "on this data"
                    )
            lines.append("  sweep (threshold -> breaches):")
            row = "    " + "  ".join(
                f"{point.threshold:.2f}:{point.breach_count}" for point in cal.sweep
            )
            lines.append(row)
        if self.agreement is not None:
            lines.append("")
            lines.append(
                f"inter-metric agreement (breach verdicts identical): {self.agreement:.2f}"
            )
        return "\n".join(lines)


# -- core statistics (pure, sync, stdlib) --------------------------------------------


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _confusion(
    pairs: Sequence[tuple[float, bool]], threshold: float
) -> tuple[int, int, int, int]:
    """(tp, fp, fn, tn) for breach = value < threshold vs. failed labels."""

    tp = fp = fn = tn = 0
    for value, failed in pairs:
        breached = value < threshold
        if breached and failed:
            tp += 1
        elif breached and not failed:
            fp += 1
        elif failed:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def _calibrate_metric(
    metric: str,
    values: Mapping[str, float],  # session_id -> value
    *,
    threshold: float,
    labels: Mapping[str, bool] | None,
    target_precision: float,
) -> MetricCalibration:
    series = sorted(values.values())
    histogram = [0] * _HISTOGRAM_BUCKETS
    for value in series:
        bucket = min(_HISTOGRAM_BUCKETS - 1, max(0, int(value * _HISTOGRAM_BUCKETS)))
        histogram[bucket] += 1

    labeled_pairs: list[tuple[float, bool]] = []
    if labels is not None:
        labeled_pairs = [
            (value, labels[session])
            for session, value in values.items()
            if session in labels
        ]

    sweep: list[ThresholdPoint] = []
    for candidate in _SWEEP_GRID:
        breach_count = sum(1 for value in series if value < candidate)
        if labeled_pairs:
            tp, fp, fn, tn = _confusion(labeled_pairs, candidate)
            precision, recall, f1 = _prf(tp, fp, fn)
            sweep.append(
                ThresholdPoint(
                    threshold=candidate,
                    breach_count=breach_count,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    tp=tp,
                    fp=fp,
                    tn=tn,
                    fn=fn,
                )
            )
        else:
            sweep.append(ThresholdPoint(threshold=candidate, breach_count=breach_count))

    labeled_stats: LabeledStats | None = None
    if labeled_pairs:
        tp, fp, fn, tn = _confusion(labeled_pairs, threshold)
        precision, recall, f1 = _prf(tp, fp, fn)
        best = max(sweep, key=lambda point: (point.f1 or 0.0, -point.threshold))
        reaching_target = [
            point
            for point in sweep
            if point.precision is not None
            and point.precision >= target_precision
            and (point.recall or 0.0) > 0.0
        ]
        labeled_stats = LabeledStats(
            labeled_sessions=len(labeled_pairs),
            precision=precision,
            recall=recall,
            f1=f1,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            best_f1_threshold=best.threshold,
            best_f1=best.f1 or 0.0,
            target_precision=target_precision,
            target_precision_threshold=(
                min(point.threshold for point in reaching_target)
                if reaching_target
                else None
            ),
        )

    return MetricCalibration(
        metric=metric,
        threshold=threshold,
        count=len(series),
        minimum=series[0] if series else None,
        maximum=series[-1] if series else None,
        mean=mean(series) if series else None,
        median=median(series) if series else None,
        stdev=stdev(series) if len(series) >= 2 else None,
        histogram=tuple(histogram),
        sweep=tuple(sweep),
        labeled=labeled_stats,
    )


def calibrate(
    scores: Mapping[str, Mapping[str, float]],  # session_id -> {metric: value}
    *,
    config: HarnessConfig,
    labels: Mapping[str, bool] | None = None,
    target_precision: float = 0.9,
    sources: Sequence[str] = (),
) -> CalibrationReport:
    """Pure, offline calibration over collected session scores."""

    by_metric: dict[str, dict[str, float]] = {}
    for session_id, metric_values in scores.items():
        for metric, value in metric_values.items():
            by_metric.setdefault(metric, {})[session_id] = value

    metrics = tuple(
        _calibrate_metric(
            metric,
            values,
            threshold=config.threshold_for(metric),
            labels=labels,
            target_precision=target_precision,
        )
        for metric, values in sorted(by_metric.items())
    )

    agreement: float | None = None
    if len(by_metric) >= 2:
        fully_scored = [
            session_id
            for session_id, metric_values in scores.items()
            if all(metric in metric_values for metric in by_metric)
        ]
        if fully_scored:
            agreeing = 0
            for session_id in fully_scored:
                verdicts = {
                    scores[session_id][metric] < config.threshold_for(metric)
                    for metric in by_metric
                }
                if len(verdicts) == 1:
                    agreeing += 1
            agreement = agreeing / len(fully_scored)

    return CalibrationReport(
        generated_at=datetime.now(UTC).isoformat(),
        sources=tuple(sources),
        session_count=len(scores),
        metrics=metrics,
        agreement=agreement,
    )


# -- score collection (three degradable sources) --------------------------------------


def _parse_backend_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        raw = payload.get("items") or payload.get("scores") or []
        return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


async def collect_scores(
    cli: CliClient,
    config: HarnessConfig,
    *,
    history: ScoreHistoryStore | None = None,
    evalset: EvalSet | None = None,
) -> tuple[dict[str, dict[str, float]], tuple[str, ...]]:
    """Gather ``session -> {metric: value}`` from CLI > history > eval-set.

    Later (higher-precedence) sources overwrite earlier ones; each source
    degrades independently and the returned tuple names the ones that
    contributed.
    """

    merged: dict[str, dict[str, float]] = {}
    sources: list[str] = []

    if evalset is not None:
        try:
            cases = await asyncio.to_thread(evalset.cases)
        except Exception:  # noqa: BLE001 - a broken store is a missing source
            logger.debug("eval-set score collection degraded", exc_info=True)
            cases = []
        contributed = False
        for case in cases:
            for metric, value in case.baseline_scores.items():
                merged.setdefault(case.session_id, {})[metric] = value
                contributed = True
        if contributed:
            sources.append("evalset")

    if history is not None:
        data = await asyncio.to_thread(load_json, config.history_file)
        contributed = False
        for key, entry in (data or {}).items():
            history_session, sep, history_metric = str(key).partition("::")
            if not sep or not isinstance(entry, dict):
                continue
            series = entry.get("series")
            if not isinstance(series, list) or not series:
                continue
            last = series[-1]
            latest = last.get("value") if isinstance(last, dict) else None
            if isinstance(latest, (int, float)):
                merged.setdefault(history_session, {})[history_metric] = float(latest)
                contributed = True
        if contributed:
            sources.append("history")

    try:
        result = await cli.run("evals", "scores", "list", "--target", "session")
        items = _parse_backend_items(result.json())
    except CliError:
        logger.debug("backend score collection degraded", exc_info=True)
        items = []
    contributed = False
    for item in items:
        item_session = item.get("session_id") or item.get("session")
        item_metric = item.get("name") or item.get("metric")
        try:
            numeric = float(item.get("value"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if item_session and item_metric:
            merged.setdefault(str(item_session), {})[str(item_metric)] = numeric
            contributed = True
    if contributed:
        sources.append("cli")

    return merged, tuple(sources)


# -- labels ---------------------------------------------------------------------------


def _as_failed(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().casefold() in _TRUTHY


def load_labels(path: Path) -> dict[str, bool]:
    """Ground-truth labels: JSON ``{sid: bool}``, JSON list of
    ``{"session_id", "failed"}``, or CSV with ``session_id,failed`` columns."""

    if path.suffix.lower() == ".csv":
        labels: dict[str, bool] = {}
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                session_id = row.get("session_id") or row.get("session")
                if session_id:
                    labels[str(session_id)] = _as_failed(row.get("failed"))
        return labels
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return {str(session): _as_failed(failed) for session, failed in payload.items()}
    if isinstance(payload, list):
        labels = {}
        for item in payload:
            if isinstance(item, dict) and item.get("session_id"):
                labels[str(item["session_id"])] = _as_failed(item.get("failed"))
        return labels
    raise ValueError(f"unsupported label format in {path}")


def labels_from_evalset(evalset: EvalSet) -> dict[str, bool]:
    """Eval-case kinds as proxy labels: failure -> failed, win -> ok."""

    return {case.session_id: case.kind == "failure" for case in evalset.cases()}


# -- the pandaprobe-harness-calibrate operator CLI ------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pandaprobe-harness-calibrate",
        description=(
            "Offline calibration of the breach thresholds: precision/recall/F1 "
            "and a threshold sweep with labels; score distribution and sweep "
            "without. Reads scores from the platform CLI, the local history "
            "store, and the eval-set."
        ),
    )
    parser.add_argument(
        "--labels", type=Path, default=None, help="JSON/CSV ground-truth labels path"
    )
    parser.add_argument(
        "--from-evalset",
        action="store_true",
        help="use eval-set kinds (failure/win) as proxy labels",
    )
    parser.add_argument("--target-precision", type=float, default=0.9)
    parser.add_argument("--json", action="store_true", help="emit the full JSON report")
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        config = HarnessConfig.from_env()
        # Imported here to keep module import light for the pure-stats API.
        from .cli.subprocess_client import SubprocessCliClient

        cli = SubprocessCliClient(config.cli_binary, default_timeout=config.cli_timeout_s)
        history = ScoreHistoryStore(config)
        evalset = EvalSet(config)
        scores, sources = asyncio.run(
            collect_scores(cli, config, history=history, evalset=evalset)
        )
        if not scores:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "error": (
                            "no session scores found in any source (platform CLI, "
                            f"{config.history_file}, {config.evalset_dir}) — run some "
                            "evaluated sessions or enable capture_eval_cases first"
                        ),
                    }
                )
            )
            return 1
        labels: dict[str, bool] | None = None
        if args.labels is not None:
            labels = load_labels(args.labels)
        elif args.from_evalset:
            labels = labels_from_evalset(evalset)
        report = calibrate(
            scores,
            config=config,
            labels=labels,
            target_precision=args.target_precision,
            sources=sources,
        )
    except Exception as exc:  # noqa: BLE001 - a CLI must not traceback at the operator
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1

    if args.json:
        print(json.dumps(report.to_json(), indent=2, sort_keys=True, default=str))
    else:
        print(report.render_text())
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess tests
    raise SystemExit(main())
