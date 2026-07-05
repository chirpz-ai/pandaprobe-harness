"""Regression runs: replay the eval-set against the current rule set.

``run_regression`` re-runs every (or a sampled subset of the) captured eval
case through the developer-supplied :data:`ReplayFn` with the *current*
rendered rules in context, scores the replayed session via the
``MetricEvaluator``, and classifies each case ``improved`` / ``unchanged`` /
``regressed`` against its captured baseline. This is the periodic "did a new
rule break an old win" guard.

Replays run sequentially: each one re-runs the developer's agent, so
parallel replays would multiply LLM cost and can violate framework
thread-safety, while the platform-eval polling inside each case is already
async. Without a replay function the run degrades honestly — one clear
warning, every case reported ``skipped`` — and never raises.

``main`` is the ``pandaprobe-harness-eval`` operator CLI over the
env-configured workspace (``HARNESS_*``), mirroring the companion CLI's
subprocess-friendly JSON output.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from ..config import HarnessConfig
from ..evaluation.evaluator import MetricEvaluator
from ..hook.turn import TurnContext
from ..workspace.evalset import EvalCase, EvalSet, ReplayFn
from ..workspace.journal import Journal
from ..workspace.rules import RulesStore

__all__ = ["CaseResult", "CaseStatus", "RegressionReport", "main", "run_regression"]

logger = logging.getLogger("pandaprobe_harness.validation")

CaseStatus = Literal["improved", "unchanged", "regressed", "skipped"]

#: Cap on per-case results embedded in the journal's ``regression`` event.
_MAX_JOURNALED_RESULTS = 50


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class CaseResult:
    """One eval case's outcome in a regression run."""

    case_id: str
    kind: str
    status: CaseStatus
    replay_session_id: str | None = None
    baseline_scores: dict[str, float] | None = None
    replay_scores: dict[str, float] | None = None
    deltas: dict[str, float] | None = None
    reason: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "kind": self.kind,
            "status": self.status,
            "replay_session_id": self.replay_session_id,
            "baseline_scores": dict(self.baseline_scores or {}),
            "replay_scores": dict(self.replay_scores or {}),
            "deltas": dict(self.deltas or {}),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class RegressionReport:
    """Aggregate outcome of one regression run."""

    started_at: str
    finished_at: str
    total_cases: int
    replayed: int
    improved: int
    unchanged: int
    regressed: int
    skipped: int
    results: tuple[CaseResult, ...] = ()

    @property
    def clean(self) -> bool:
        """No case regressed (skips are reported, not failures)."""

        return self.regressed == 0

    def to_json(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_cases": self.total_cases,
            "replayed": self.replayed,
            "improved": self.improved,
            "unchanged": self.unchanged,
            "regressed": self.regressed,
            "skipped": self.skipped,
            "clean": self.clean,
            "results": [result.to_json() for result in self.results],
        }

    def render_text(self) -> str:
        lines = [
            f"Regression run: {self.total_cases} case(s), {self.replayed} replayed — "
            f"improved {self.improved}, unchanged {self.unchanged}, "
            f"regressed {self.regressed}, skipped {self.skipped}",
        ]
        for result in self.results:
            detail = ""
            if result.deltas:
                moved = ", ".join(
                    f"{metric} {delta:+.2f}" for metric, delta in sorted(result.deltas.items())
                )
                detail = f" ({moved})"
            elif result.reason:
                detail = f" ({result.reason})"
            session = f" -> {result.replay_session_id}" if result.replay_session_id else ""
            lines.append(f"  [{result.kind}] {result.case_id} {result.status}{detail}{session}")
        lines.append("CLEAN" if self.clean else "REGRESSIONS DETECTED")
        return "\n".join(lines)


def _classify(deltas: Mapping[str, float], config: HarnessConfig) -> CaseStatus:
    """Regression dominates improvement; small moves are noise (``unchanged``)."""

    if not deltas:
        return "unchanged"
    if any(delta <= -config.rule_regress_margin for delta in deltas.values()):
        return "regressed"
    if any(delta >= config.rule_promote_margin for delta in deltas.values()):
        return "improved"
    return "unchanged"


def _skipped(case: EvalCase, reason: str) -> CaseResult:
    return CaseResult(
        case_id=case.id,
        kind=case.kind,
        status="skipped",
        baseline_scores=dict(case.baseline_scores),
        reason=reason,
    )


async def replay_case(
    case: EvalCase,
    context: str,
    *,
    config: HarnessConfig,
    evaluator: MetricEvaluator,
    replay: ReplayFn,
) -> CaseResult:
    """Replay one case and classify it against its baseline. Never raises."""

    if not case.replayable:
        return _skipped(case, "no replay_input attached (harness_evalset_attach)")
    try:
        # Time-bounded: replays run sequentially, so one hung replay would
        # otherwise stall the whole regression run.
        new_session = str(
            await asyncio.wait_for(replay(case, context), config.replay_timeout_s)
        )
    except Exception as exc:  # noqa: BLE001 - a broken replay must not stop the run
        logger.warning("replay failed for case %s: %s", case.id, exc or type(exc).__name__)
        return _skipped(case, f"replay failed: {exc or type(exc).__name__}")
    try:
        report = await evaluator.evaluate_turn(
            TurnContext(session_id=new_session, turn_index=0)
        )
    except Exception as exc:  # noqa: BLE001 - degrade to a skip, never crash the run
        logger.warning("scoring replayed session %s failed: %s", new_session, exc)
        return _skipped(case, f"scoring failed: {exc}")
    replay_scores = {
        str(score.metric): score.value for score in report.scores if score.value is not None
    }
    if not replay_scores:
        return CaseResult(
            case_id=case.id,
            kind=case.kind,
            status="skipped",
            replay_session_id=new_session,
            baseline_scores=dict(case.baseline_scores),
            reason="replayed session produced no resolved scores",
        )
    deltas = {
        metric: value - case.baseline_scores[metric]
        for metric, value in replay_scores.items()
        if metric in case.baseline_scores
    }
    return CaseResult(
        case_id=case.id,
        kind=case.kind,
        status=_classify(deltas, config),
        replay_session_id=new_session,
        baseline_scores=dict(case.baseline_scores),
        replay_scores=replay_scores,
        deltas=deltas,
    )


async def run_regression(
    *,
    config: HarnessConfig,
    rules: RulesStore,
    evalset: EvalSet,
    evaluator: MetricEvaluator,
    journal: Journal | None = None,
    replay: ReplayFn | None = None,
    sample: int | None = None,
) -> RegressionReport:
    """Replay the eval-set against the current rules and report per-case drift.

    ``win`` cases run first (they are the regression guard); ``sample``
    overrides ``config.regression_sample`` (0 = all). Degrades — never
    raises — when no replay function is wired.
    """

    started = _utcnow_iso()
    cases = await asyncio.to_thread(evalset.cases)
    wins = sorted(
        (case for case in cases if case.kind == "win"),
        key=lambda c: (c.created_at, c.id),
        reverse=True,
    )
    failures = sorted(
        (case for case in cases if case.kind == "failure"),
        key=lambda c: (c.created_at, c.id),
        reverse=True,
    )
    ordered = wins + failures
    limit = config.regression_sample if sample is None else sample
    if limit > 0:
        ordered = ordered[:limit]

    results: list[CaseResult] = []
    if replay is None:
        logger.warning(
            "regression run cannot replay: no replay function wired (pass replay=... to "
            "Harness.create or --replay to pandaprobe-harness-eval); %s case(s) skipped",
            len(ordered),
        )
        results = [_skipped(case, "no replay function wired") for case in ordered]
    else:
        context = await asyncio.to_thread(rules.render_markdown)
        for case in ordered:
            results.append(
                await replay_case(
                    case, context, config=config, evaluator=evaluator, replay=replay
                )
            )

    counts = {status: 0 for status in ("improved", "unchanged", "regressed", "skipped")}
    for result in results:
        counts[result.status] += 1
    report = RegressionReport(
        started_at=started,
        finished_at=_utcnow_iso(),
        total_cases=len(ordered),
        replayed=len(ordered) - counts["skipped"],
        improved=counts["improved"],
        unchanged=counts["unchanged"],
        regressed=counts["regressed"],
        skipped=counts["skipped"],
        results=tuple(results),
    )
    if journal is not None:
        await asyncio.to_thread(
            journal.record,
            {
                "type": "regression",
                "total": report.total_cases,
                "replayed": report.replayed,
                "improved": report.improved,
                "unchanged": report.unchanged,
                "regressed": report.regressed,
                "skipped": report.skipped,
                "clean": report.clean,
                "results": [r.to_json() for r in report.results[:_MAX_JOURNALED_RESULTS]],
            },
        )
    return report


# -- the pandaprobe-harness-eval operator CLI -------------------------------------


def _resolve_replay(spec: str) -> ReplayFn:
    """Import ``pkg.module:attr`` and return it as the replay function."""

    module_name, _, attr = spec.partition(":")
    if not module_name or not attr:
        raise ValueError(f"--replay expects 'pkg.module:attr', got {spec!r}")
    module = importlib.import_module(module_name)
    fn = getattr(module, attr)
    if not callable(fn):
        raise TypeError(f"{spec!r} is not callable")
    return fn  # type: ignore[no-any-return]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pandaprobe-harness-eval",
        description=(
            "Replay the captured eval-set against the current rule set and report "
            "per-case improved/unchanged/regressed vs. baseline. Requires a replay "
            "function (--replay pkg.module:attr) to actually re-run the agent."
        ),
    )
    parser.add_argument(
        "--sample", type=int, default=None, help="replay only the first N cases (0 = all)"
    )
    parser.add_argument(
        "--replay",
        default=None,
        metavar="MODULE:ATTR",
        help="import path of an async (case, context) -> new_session_id replay function",
    )
    parser.add_argument("--json", action="store_true", help="emit the full JSON report")
    parser.add_argument(
        "--list", action="store_true", dest="list_cases", help="list eval cases and exit"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        config = HarnessConfig.from_env()
        journal = Journal(config)
        evalset = EvalSet(config, journal=journal)
        evalset.provision()
        if args.list_cases:
            cases = evalset.cases()
            print(
                json.dumps(
                    {"ok": True, "cases": [case.summary() for case in cases]},
                    indent=2,
                    sort_keys=True,
                    default=str,
                )
            )
            return 0
        replay = _resolve_replay(args.replay) if args.replay else None
        rules = RulesStore(config, journal=journal)
        # Imported here so `--list` works even where the CLI seam cannot.
        from ..cli.subprocess_client import SubprocessCliClient

        cli = SubprocessCliClient(config.cli_binary, default_timeout=config.cli_timeout_s)
        evaluator = MetricEvaluator(cli, config)
        report = asyncio.run(
            run_regression(
                config=config,
                rules=rules,
                evalset=evalset,
                evaluator=evaluator,
                journal=journal,
                replay=replay,
                sample=args.sample,
            )
        )
    except Exception as exc:  # noqa: BLE001 - a CLI must not traceback at the operator
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1

    if args.json:
        print(json.dumps(report.to_json(), indent=2, sort_keys=True, default=str))
    else:
        print(report.render_text())
    return 0 if report.clean else 1


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess tests
    raise SystemExit(main())
