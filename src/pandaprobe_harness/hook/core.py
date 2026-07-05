"""The lifecycle hook: turn end → evaluation → mailbox notice.

``PandaHarnessHook`` is non-blocking by design. ``on_turn_end`` applies the
cheap producing-side controls (budget, sampling, per-session rate limit),
supersedes any in-flight evaluation for the session, and schedules a detached
wrapper task. The wrapper — :meth:`_run_eval` — awaits the evaluation under a
global concurrency semaphore *and handles the resolved report itself*: it
feeds scores to the local EWMA trend detector, applies the dedup/cooldown
gate, and (when a breach/relative/trend condition fires) writes the telemetry
dump and posts a structured :class:`DiagnosticNotice` to the mailbox, where
the agent will *pull* it via its harness toolset.

Nothing is ever injected into the agent's input queue. Because handling lives
inside the wrapper task, no next-turn drain barrier is required for
correctness: evaluations resolve and post as soon as they finish. ``refresh``
remains as a bounded await for callers and tests, and exceptions cannot
vanish — every failure path is caught and logged inside the task.

``startup_context()`` returns the rendered rules + pull protocol + mailbox
banner for prepending to the agent's system prompt.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..cli.client import CliClient
from ..cli.errors import CliAuthError, CliError
from ..config import HarnessConfig
from ..evaluation.evaluator import MetricEvaluator
from ..evaluation.history import ScoreHistoryStore
from ..evaluation.metrics import EvalReport, MetricScore
from ..evaluation.trends import TrendDetector
from ..workspace.evalset import EvalSet, ReplayFn
from ..workspace.journal import Journal
from ..workspace.mailbox import DiagnosticNotice, Mailbox, NoticeMetric, Severity
from ..workspace.rules import RulesStore
from ..workspace.sanitize import sanitize_text
from .context import compose_system_preamble
from .turn import TurnContext, parse_turn_payload

if TYPE_CHECKING:
    from ..filesystem.layout import HarnessFilesystem
    from ..validation.validator import ValidationEngine, ValidationVerdict

__all__ = ["PandaHarnessHook"]

logger = logging.getLogger("pandaprobe_harness.hook")

# Bound per-session bookkeeping so a long-lived process handling many
# short-lived session ids cannot grow memory without limit.
_MAX_TRACKED_SESSIONS = 4096


@dataclass
class _SessionNoticeState:
    signatures: set[str] = field(default_factory=set)
    cooldown: int = 0


class PandaHarnessHook:
    """Pluggable, framework-agnostic turn-completion hook (pull model)."""

    def __init__(
        self,
        cli: CliClient,
        *,
        config: HarnessConfig | None = None,
        mailbox: Mailbox | None = None,
        journal: Journal | None = None,
        rules: RulesStore | None = None,
        filesystem: HarnessFilesystem | None = None,
        evaluator: MetricEvaluator | None = None,
        parser: Callable[[object], TurnContext] | None = None,
        history: ScoreHistoryStore | None = None,
        evalset: EvalSet | None = None,
        validation: ValidationEngine | None = None,
        replay: ReplayFn | None = None,
    ) -> None:
        self._cli = cli
        self._config = config or HarnessConfig()
        self._evaluator = evaluator or MetricEvaluator(cli, self._config)
        if filesystem is None:
            # Imported lazily to avoid a hard cycle at module import time.
            from ..filesystem.layout import HarnessFilesystem

            filesystem = HarnessFilesystem(self._config)
        self._filesystem = filesystem
        self._journal = journal or Journal(self._config)
        self._mailbox = mailbox or Mailbox(self._config)
        self._rules = rules or RulesStore(self._config, journal=self._journal)
        self._parser = parser or parse_turn_payload

        # The regression eval-set: breaching sessions are captured as replayable
        # failure cases when the knob is on; the validation engine also replays
        # matching cases to vet candidate rules.
        self._evalset = evalset
        if self._evalset is None and (
            self._config.capture_eval_cases or self._config.rule_validation
        ):
            self._evalset = EvalSet(self._config, journal=self._journal)
        #: Latest non-empty turn payload per session — the replay input an eval
        #: case needs. Facade turns send `end_state={}`, so this stays empty
        #: there (attach inputs explicitly via the eval-set instead).
        self._replay_inputs: dict[str, Any] = {}

        # Candidate-rule validation (evidence before trust). Imported lazily to
        # avoid a hard cycle at module import time (same as HarnessFilesystem).
        self._validation = validation
        if self._validation is None and self._config.rule_validation:
            from ..validation.validator import ValidationEngine

            assert self._evalset is not None  # built above when validation is on
            self._validation = ValidationEngine(
                config=self._config,
                rules=self._rules,
                evalset=self._evalset,
                evaluator=self._evaluator,
                journal=self._journal,
                replay=replay,
            )
        self._validation_tasks: set[asyncio.Task[None]] = set()

        # One store instance must be shared with any other reader (the store
        # memoizes its file cache), so the facade passes its instance in.
        self._history: ScoreHistoryStore | None = history
        if self._history is None and (
            self._config.enable_trend or self._config.hydrate_history_from_backend
        ):
            self._history = ScoreHistoryStore(self._config)
        self._detector: TrendDetector | None = None
        if self._config.enable_trend and self._history is not None:
            self._detector = TrendDetector(self._config, self._history)

        # Task tracking: per-session latest task (supersede + refresh) and a
        # strong-ref set so detached tasks are never garbage-collected early.
        self._pending: dict[str, asyncio.Task[EvalReport | None]] = {}
        self._tasks: set[asyncio.Task[EvalReport | None]] = set()
        self._journal_tasks: set[asyncio.Task[Any]] = set()

        # Notice dedup/cooldown (per session) and the global circuit breaker.
        self._notice_state: dict[str, _SessionNoticeState] = {}
        self._notice_times: deque[float] = deque()
        self._breaker_tripped = False

        # Producing-side controls.
        self._semaphore = asyncio.Semaphore(max(1, self._config.max_concurrent_evals))
        self._turn_counts: dict[str, int] = {}
        self._last_eval_at: dict[str, float] = {}
        self._evals_launched = 0
        self._budget_logged = False

        # Startup health check (memoized) + one-time backend hydration.
        self._health_lock = asyncio.Lock()
        self._health_checked = False
        self._degraded_reason: str | None = None
        self._hydrated: set[str] = set()

    # -- surface ---------------------------------------------------------------

    @property
    def mailbox(self) -> Mailbox:
        return self._mailbox

    @property
    def journal(self) -> Journal:
        return self._journal

    @property
    def rules(self) -> RulesStore:
        return self._rules

    def startup_context(self, *, task_hint: str | None = None) -> str:
        """Rules + pull protocol + mailbox banner, for the agent's system prompt."""

        return compose_system_preamble(self._rules, self._mailbox, task_hint=task_hint)

    # -- producing side (turn end) -------------------------------------------

    def on_turn_end(self, raw_turn: object) -> None:
        """Schedule evaluation for a completed turn. Returns immediately."""

        try:
            ctx = self._parser(raw_turn)
        except Exception:  # noqa: BLE001 - never break the host loop
            logger.exception("failed to parse turn; skipping evaluation")
            return

        if not self._admit(ctx):
            return

        # Remember the turn payload so a breach can be captured as a
        # *replayable* eval case. Stashed only for admitted turns: only
        # evaluated turns can produce a notice, and admitted sessions are the
        # ones whose bookkeeping (and thus this stash) gets evicted.
        if self._capture_enabled() and ctx.end_state:
            self._replay_inputs[ctx.session_id] = dict(ctx.end_state)

        # If a prior turn's eval is still in flight for this session, cancel it;
        # the newest turn supersedes it (avoid orphaning a detached task).
        prev = self._pending.get(ctx.session_id)
        if prev is not None and not prev.done():
            prev.cancel()

        task: asyncio.Task[EvalReport | None] = asyncio.ensure_future(self._run_eval(ctx))
        self._pending[ctx.session_id] = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        self._last_eval_at[ctx.session_id] = time.monotonic()
        self._evals_launched += 1

    def _admit(self, ctx: TurnContext) -> bool:
        """Budget → sampling → rate-limit gate. Skips are logged, never silent."""

        budget = self._config.max_evals_per_run
        if budget > 0 and self._evals_launched >= budget:
            if not self._budget_logged:
                self._budget_logged = True
                logger.warning(
                    "eval budget exhausted (%s launches); skipping further evaluations",
                    budget,
                )
                self._journal_soon(
                    {"type": "skip", "reason": "budget", "session_id": ctx.session_id}
                )
            else:
                logger.info("eval budget exhausted; skipped session=%s", ctx.session_id)
            return False

        count = self._turn_counts.get(ctx.session_id, 0) + 1
        self._turn_counts[ctx.session_id] = count
        if len(self._turn_counts) > _MAX_TRACKED_SESSIONS:
            self._evict_oldest_session()
        every = max(1, self._config.eval_sample_every)
        if (count - 1) % every != 0:
            logger.info(
                "sampling: skipped eval for session=%s turn %s (every %s turns)",
                ctx.session_id,
                count,
                every,
            )
            return False

        min_interval = self._config.session_min_eval_interval_s
        if min_interval > 0:
            last = self._last_eval_at.get(ctx.session_id)
            if last is not None and (time.monotonic() - last) < min_interval:
                logger.info(
                    "rate-limited: skipped eval for session=%s (< %.1fs since last)",
                    ctx.session_id,
                    min_interval,
                )
                return False
        return True

    def _evict_oldest_session(self) -> None:
        """Drop the earliest-seen session's bookkeeping (memory bound).

        A later turn from an evicted session simply restarts its sampling /
        rate-limit counters and may re-hydrate once — never a correctness bug,
        just a bounded reset.
        """

        try:
            oldest = next(iter(self._turn_counts))
        except StopIteration:  # pragma: no cover - guarded by the caller
            return
        self._turn_counts.pop(oldest, None)
        self._last_eval_at.pop(oldest, None)
        self._hydrated.discard(oldest)
        self._notice_state.pop(oldest, None)
        self._replay_inputs.pop(oldest, None)

    def _journal_soon(self, event: dict[str, Any]) -> None:
        """Best-effort, non-blocking journal write from the sync path."""

        try:
            task = asyncio.ensure_future(asyncio.to_thread(self._journal.record, event))
            self._journal_tasks.add(task)
            task.add_done_callback(self._journal_tasks.discard)
        except RuntimeError:  # pragma: no cover - no running loop
            logger.debug("no running loop; journal event dropped: %s", event.get("type"))

    # -- the wrapper task -------------------------------------------------------

    async def _run_eval(self, ctx: TurnContext) -> EvalReport | None:
        """Evaluate one turn and handle the result. Never raises (except cancel)."""

        try:
            if not await self._ensure_healthy():
                return None
            if (
                self._config.hydrate_history_from_backend
                and self._history is not None
                and ctx.session_id not in self._hydrated
            ):
                await self._hydrate(ctx.session_id)
            async with self._semaphore:
                report = await self._evaluator.evaluate_turn(ctx)
            return await self._handle_report(report)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - degrade gracefully, never lose the error
            logger.exception("eval pipeline failed for session=%s", ctx.session_id)
            return None
        finally:
            if self._pending.get(ctx.session_id) is asyncio.current_task():
                self._pending.pop(ctx.session_id, None)

    # -- consuming side (tests / explicit callers) -----------------------------

    async def refresh(self, session_id: str) -> EvalReport | None:
        """Await the in-flight eval for ``session_id``, bounded by the drain budget.

        Handling already happened inside the task; this is purely an awaitable
        join. On timeout the task keeps running detached.
        """

        task = self._pending.get(session_id)
        if task is None:
            return None
        try:
            return await asyncio.wait_for(asyncio.shield(task), self._config.drain_timeout_s)
        except TimeoutError:
            logger.info("eval for session=%s not ready within refresh budget", session_id)
            return None
        except asyncio.CancelledError:
            # Distinguish *our caller* being cancelled (must propagate) from the
            # shielded eval being superseded-cancelled (benign). If a cancellation
            # was requested on this task, honor it even when the eval also happens
            # to have been cancelled in the same window.
            current = asyncio.current_task()
            if current is not None and current.cancelling() > 0:
                raise
            if task.cancelled():
                return None  # superseded eval, not our caller's cancellation
            raise

    async def refresh_all(self) -> None:
        """Await every in-flight eval (bounded by the drain budget)."""

        tasks = [task for task in self._tasks if not task.done()]
        if tasks:
            await asyncio.wait(tasks, timeout=self._config.drain_timeout_s)

    # -- report handling ---------------------------------------------------------

    async def _handle_report(self, report: EvalReport) -> EvalReport:
        # 1. Feed every resolved score to the trend detector (records history +
        #    sets trend/relative flags). Runs in a thread (sync store I/O).
        if self._detector is not None:
            report = await asyncio.to_thread(self._apply_trends, report)

        # 2. Candidate-rule validation: every handled report (healthy or
        #    alerting) feeds the forward trials — the trial needs the
        #    denominator — and kicks one single-flight evaluation round.
        await self._observe_for_validation(report)

        # 3. Dedup / cooldown / recovery gate.
        post, recovered = self._should_notice(report)
        if recovered:
            self._breaker_tripped = False
            self._notice_times.clear()
            await asyncio.to_thread(
                self._journal.record,
                {"type": "recovery", "session_id": report.session_id},
            )
        if not post:
            return report

        # 4. Circuit breaker, dump, notice (+ eval-case capture inside the
        #    same thread hop).
        try:
            payload = await self._build_dump(report)
            notice = self._breaker_or_notice(report, payload)
            if notice is not None:
                replay_input = (
                    self._replay_inputs.get(report.session_id)
                    if self._capture_enabled()
                    else None
                )
                await asyncio.to_thread(self._persist_notice, notice, payload, replay_input)
        except Exception:  # noqa: BLE001 - never break the host loop
            logger.exception("failed to persist notice for session=%s", report.session_id)
        return report

    async def _observe_for_validation(self, report: EvalReport) -> None:
        if self._validation is None:
            return
        try:
            await asyncio.to_thread(
                self._validation.observe_report, report.session_id, self._signatures(report)
            )
            self._spawn_validation()
        except Exception:  # noqa: BLE001 - never break the host loop
            logger.exception(
                "candidate validation step failed for session=%s", report.session_id
            )

    def _spawn_validation(self) -> None:
        """Kick one detached candidate-evaluation task (single-flight)."""

        if any(not task.done() for task in self._validation_tasks):
            return
        task = asyncio.ensure_future(self._run_validation())
        self._validation_tasks.add(task)
        task.add_done_callback(self._validation_tasks.discard)

    async def _run_validation(self) -> None:
        assert self._validation is not None
        try:
            await self._validation.evaluate_candidates()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - detached task; degrade, never lose the error
            logger.exception("candidate evaluation failed")

    async def validate_candidates(self) -> list[ValidationVerdict]:
        """Explicitly run one candidate-evaluation round (no-op when disabled)."""

        if self._validation is None:
            return []
        return await self._validation.evaluate_candidates()

    async def drain_validation(self) -> None:
        """Await in-flight validation tasks (bounded by the drain budget)."""

        tasks = [task for task in self._validation_tasks if not task.done()]
        if tasks:
            await asyncio.wait(tasks, timeout=self._config.drain_timeout_s)

    def _breaker_or_notice(
        self, report: EvalReport, payload: dict[str, Any]
    ) -> DiagnosticNotice | None:
        """Apply the circuit breaker; return the notice to persist (or None)."""

        if self._config.observe_only:
            # Shadow mode: journal-only notice, no dump, no breaker accounting.
            return self._build_notice(report, severity=self._severity(report), dump_path="")

        max_notices = self._config.circuit_breaker_max_notices
        if max_notices > 0:
            now = time.monotonic()
            window = self._config.circuit_breaker_window_s
            while self._notice_times and now - self._notice_times[0] > window:
                self._notice_times.popleft()
            if self._breaker_tripped and not self._notice_times:
                self._breaker_tripped = False  # window drained
            if self._breaker_tripped:
                logger.info(
                    "circuit breaker tripped; notice suppressed for session=%s",
                    report.session_id,
                )
                return None
            if len(self._notice_times) >= max_notices:
                self._breaker_tripped = True
                logger.warning(
                    "circuit breaker: %s notices within %.0fs — escalating to needs_human",
                    len(self._notice_times),
                    window,
                )
                return self._build_notice(
                    report,
                    severity="needs_human",
                    dump_path="",
                    summary=(
                        f"notice rate exceeded ({len(self._notice_times)} notices in "
                        f"{window:.0f}s) — self-healing paused; human attention required"
                    ),
                )
            self._notice_times.append(now)

        notice_id = DiagnosticNotice.new_id()
        dump_path = str(self._config.traces_dir / f"{notice_id}.json")
        return self._build_notice(
            report,
            severity=self._severity(report),
            dump_path=dump_path,
            notice_id=notice_id,
        )

    def _persist_notice(
        self,
        notice: DiagnosticNotice,
        payload: dict[str, Any],
        replay_input: Any | None = None,
    ) -> None:
        """Blocking persistence step, run in one ``to_thread`` hop."""

        self._filesystem.write_latest_eval(payload)
        if not self._config.observe_only:
            if notice.dump_path:
                self._filesystem.write_trace_dump(notice.id, payload)
            self._mailbox.post(notice)
        self._journal.record(
            {"type": "notice", "observe_only": self._config.observe_only, **notice.to_json()}
        )
        # Only absolute breaches become failure cases: trend/relative/
        # percentile notices are advisory (their baseline scores can sit
        # above the threshold, which would pollute the eval-set and its
        # proxy labels), and `needs_human` is a rate alarm.
        if (
            self._capture_enabled()
            and self._evalset is not None
            and not self._config.observe_only
            and notice.severity == "breach"
        ):
            try:
                self._evalset.capture(
                    session_id=notice.session_id,
                    kind="failure",
                    signature=notice.signatures,
                    baseline_scores=_baseline_from_dump(payload),
                    replay_input=replay_input,
                    notes=notice.summary,
                )
            except Exception:  # noqa: BLE001 - the notice is already persisted
                logger.exception(
                    "failed to capture eval case for session=%s", notice.session_id
                )

    def _capture_enabled(self) -> bool:
        return self._config.capture_eval_cases and self._evalset is not None

    # -- trend application ----------------------------------------------------

    def _apply_trends(self, report: EvalReport) -> EvalReport:
        assert self._detector is not None
        updated = []
        for score in report.scores:
            if not score.pending and score.value is not None:
                verdict = self._detector.update(
                    report.session_id, str(score.metric), score.value
                )
                score = replace(
                    score,
                    trend_declining=verdict.declining,
                    relative_breach=score.relative_breach or verdict.relative_breach,
                    percentile_breach=verdict.percentile_breach,
                )
            updated.append(score)
        return replace(report, scores=tuple(updated))

    # -- notice decisioning -----------------------------------------------------

    @staticmethod
    def _signatures(report: EvalReport) -> set[str]:
        sigs: set[str] = set()
        for score in report.scores:
            metric = str(score.metric)
            if score.breached:
                sigs.add(f"breach:{metric}")
            if score.relative_breach:
                sigs.add(f"relative:{metric}")
            if score.trend_declining:
                sigs.add(f"trend:{metric}")
            if score.percentile_breach:
                sigs.add(f"percentile:{metric}")
        return sigs

    def _should_notice(self, report: EvalReport) -> tuple[bool, bool]:
        """Dedup/cooldown gate → ``(post, recovered)``. Mutates per-session state."""

        session_id = report.session_id
        current = self._signatures(report)
        if not current:
            recovered = session_id in self._notice_state
            self._notice_state.pop(session_id, None)  # recovery
            return False, recovered

        state = self._notice_state.get(session_id, _SessionNoticeState())
        new_conditions = current - state.signatures
        cooldown_turns = self._config.alert_cooldown_turns

        post = bool(new_conditions) or (
            current == state.signatures and cooldown_turns > 0 and state.cooldown <= 0
        )

        if post:
            self._notice_state[session_id] = _SessionNoticeState(
                signatures=set(current), cooldown=cooldown_turns
            )
        else:
            self._notice_state[session_id] = _SessionNoticeState(
                signatures=set(current), cooldown=max(0, state.cooldown - 1)
            )
        return post, False

    @staticmethod
    def _severity(report: EvalReport) -> Severity:
        if any(score.breached for score in report.scores):
            return "breach"
        if any(score.relative_breach for score in report.scores):
            return "relative"
        return "trend"

    # -- notice construction ------------------------------------------------------

    def _build_notice(
        self,
        report: EvalReport,
        *,
        severity: Severity,
        dump_path: str,
        notice_id: str | None = None,
        summary: str | None = None,
    ) -> DiagnosticNotice:
        max_len = self._config.sanitize_max_len
        metrics = tuple(
            NoticeMetric(
                name=str(score.metric),
                value=score.value,
                threshold=score.threshold,
                reason=sanitize_text(score.reason, max_len=max_len) or None,
                conditions=self._conditions(score),
            )
            for score in report.alerting_scores
        )
        return DiagnosticNotice(
            id=notice_id or DiagnosticNotice.new_id(),
            created_at=_utcnow_iso(),
            session_id=report.session_id,
            turn_index=report.turn_index,
            severity=severity,
            metrics=metrics,
            flagged_traces=tuple(report.flagged_traces),
            signal_breakdown=report.signal_breakdown(),
            dump_path=dump_path,
            summary=sanitize_text(summary or self._summarize(report), max_len=max_len),
            signatures=tuple(sorted(self._signatures(report))),
        )

    @staticmethod
    def _conditions(score: MetricScore) -> tuple[str, ...]:
        conditions: list[str] = []
        if score.breached:
            conditions.append("breach")
        if score.relative_breach:
            conditions.append("relative")
        if score.trend_declining:
            conditions.append("trend")
        if score.percentile_breach:
            conditions.append("percentile")
        return tuple(conditions)

    def _summarize(self, report: EvalReport) -> str:
        parts: list[str] = []
        for score in report.alerting_scores:
            value = f"{score.value:.2f}" if score.value is not None else "n/a"
            conds = "+".join(self._conditions(score)) or "ok"
            parts.append(f"{score.metric}={value} [{conds}, threshold {score.threshold:.2f}]")
        line = "; ".join(parts) if parts else "no alerting scores"
        if report.flagged_traces:
            line += f"; flagged traces: {', '.join(report.flagged_traces[:5])}"
        return line

    async def _build_dump(self, report: EvalReport) -> dict[str, Any]:
        payload = report.to_dump()
        if self._config.enrich_flagged_traces and report.flagged_traces:
            trace_id = report.flagged_traces[0]
            try:
                result = await self._cli.run("traces", "get", trace_id, "--kind", "TOOL")
                payload["flagged_trace_detail"] = result.json()
            except Exception:  # noqa: BLE001 - enrichment is best-effort
                logger.debug("flagged-trace enrichment failed for %s", trace_id)
        return payload

    # -- robustness ---------------------------------------------------------------

    async def check_health(self) -> bool:
        """Verify the CLI is reachable and authenticated (memoized).

        On failure the harness runs *degraded*: one clear warning, a journal
        event, and every subsequent evaluation is skipped — never a crash,
        never a silent no-op.
        """

        if self._health_checked:
            return self._degraded_reason is None
        async with self._health_lock:
            if self._health_checked:
                return self._degraded_reason is None
            reason: str | None = None
            try:
                await self._cli.run("version")
            except CliError as exc:
                reason = f"pandaprobe CLI unavailable: {exc}"
            except Exception as exc:  # noqa: BLE001 - health check must never crash
                reason = f"pandaprobe CLI probe failed: {exc}"
            if reason is None:
                try:
                    await self._cli.run("auth", "status")
                except CliAuthError as exc:
                    reason = f"pandaprobe CLI unauthenticated: {exc}"
                except Exception:  # noqa: BLE001 - inconclusive, not fatal
                    logger.debug("auth-status probe inconclusive; assuming healthy")
            self._health_checked = True
            self._degraded_reason = reason
            if reason is not None:
                logger.warning("harness degraded — evaluations disabled: %s", reason)
            try:
                await asyncio.to_thread(
                    self._journal.record,
                    {"type": "health", "ok": reason is None, "reason": reason},
                )
            except Exception:  # noqa: BLE001 - journaling is best-effort here
                logger.debug("failed to journal health event", exc_info=True)
            return reason is None

    async def _ensure_healthy(self) -> bool:
        if not self._config.health_check:
            return True
        return await self.check_health()

    # -- backend hydration (shared trend state at scale) ---------------------------

    async def _hydrate(self, session_id: str) -> None:
        """Seed local trend history from the backend, once per session."""

        self._hydrated.add(session_id)  # one attempt, even on failure
        assert self._history is not None
        try:
            result = await self._cli.run(
                "evals", "scores", "list", "--target", "session", "--session-id", session_id
            )
            payload = result.json()
        except CliError:
            logger.debug("history hydration degraded for session=%s", session_id)
            return

        items: list[Any]
        if isinstance(payload, dict):
            raw = payload.get("items") or payload.get("scores") or []
            items = raw if isinstance(raw, list) else []
        elif isinstance(payload, list):
            items = payload
        else:
            items = []

        by_metric: dict[str, list[tuple[float, str, str | None]]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("metric")
            value = item.get("value")
            try:
                numeric = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if not name:
                continue
            ts = str(item.get("created_at") or item.get("ts") or "")
            run_id = item.get("run_id") or item.get("id")
            by_metric.setdefault(str(name), []).append(
                (numeric, ts, str(run_id) if run_id is not None else None)
            )
        for metric, samples in by_metric.items():
            # EWMA folding is order-sensitive, so seed in chronological order.
            # The backend commonly returns scores newest-first; samples without
            # a timestamp sort to the front (treated as oldest).
            ordered = sorted(samples, key=lambda s: s[1])
            await asyncio.to_thread(self._history.seed, session_id, metric, ordered)


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def _baseline_from_dump(payload: dict[str, Any]) -> dict[str, float]:
    """Resolved metric values from an ``EvalReport.to_dump()`` payload."""

    baseline: dict[str, float] = {}
    scores = payload.get("scores")
    if not isinstance(scores, list):
        return baseline
    for score in scores:
        if not isinstance(score, dict):
            continue
        value = score.get("value")
        if isinstance(value, (int, float)):
            baseline[str(score.get("metric"))] = float(value)
    return baseline
