"""The lifecycle hook: task turn → evaluation → notice → managed repair.

``PandaHarnessHook`` is non-blocking by design. ``on_turn_end`` applies the
cheap producing-side controls (budget, sampling, per-session rate limit),
supersedes any in-flight evaluation for the session, and schedules a detached
wrapper task. The wrapper — :meth:`_run_eval` — awaits the evaluation under a
global concurrency semaphore *and handles the resolved report itself*: it
applies the dedup/cooldown gate and, when a breach, stall, or regression fires,
writes the telemetry dump and posts a structured :class:`DiagnosticNotice` to
the mailbox, where package-owned managed repair consumes it.

Nothing is ever injected into the agent's input queue. Because handling lives
inside the wrapper task, evaluations resolve and post as soon as they finish;
``refresh`` remains a bounded best-effort join for callers and tests, and
exceptions cannot vanish — every failure path is caught and logged inside the
task.

:meth:`PandaHarnessHook.settle` is the opt-in **per-turn barrier** that makes
repair take effect *within* a session: it blocks until the turn's evaluation has
landed, any notice is posted, and managed repair has completed or timed out.
Candidate validation stays detached so replay cannot deadlock a task-owned
environment.

``startup_context()`` returns only a stable read-only capability note.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ..cli.client import CliClient
from ..cli.errors import CliAuthError, CliError
from ..config import HarnessConfig
from ..evaluation.evaluator import MetricEvaluator
from ..evaluation.history import ScoreHistoryStore
from ..evaluation.history_source import HistorySource
from ..evaluation.metrics import EvalReport
from ..evaluation.traces import TraceLocator
from ..evaluation.trajectory import TrajectoryGate
from ..workspace.evalset import EvalSet, ReplayFn
from ..workspace.journal import Journal
from ..workspace.mailbox import DiagnosticNotice, Mailbox, NoticeMetric, Severity
from ..workspace.rules import GLOBAL_SCOPE, RESERVED_SCOPES, RulesStore
from ..workspace.sanitize import sanitize_text
from .context import compose_system_preamble
from .tiers import TierRunner, VerifierFn
from .turn import RuleScopeHint, TurnContext, parse_turn_payload

if TYPE_CHECKING:
    from ..filesystem.layout import HarnessFilesystem
    from ..repair.coordinator import ManagedRepairCoordinator
    from ..repair.models import RepairResult
    from ..validation.validator import ValidationEngine, ValidationVerdict

__all__ = ["PandaHarnessHook", "SettleResult"]

logger = logging.getLogger("pandaprobe_harness.hook")


@dataclass(frozen=True, slots=True)
class SettleResult:
    """Outcome of one per-turn barrier (:meth:`PandaHarnessHook.settle`)."""

    session_id: str
    report: EvalReport | None = None
    repair: RepairResult | None = None
    #: True when the barrier's budget expired with work still in flight. The work
    #: continues detached, so this is a latency signal, not an error.
    timed_out: bool = False

    @property
    def alerting(self) -> bool:
        return self.report is not None and self.report.any_alert


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
        history: HistorySource | None = None,
        evalset: EvalSet | None = None,
        validation: ValidationEngine | None = None,
        replay: ReplayFn | None = None,
        verifier: VerifierFn | None = None,
        locator: TraceLocator | None = None,
        managed_repair: ManagedRepairCoordinator | None = None,
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
        self._repair = managed_repair

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
        self._scope_hints: dict[tuple[str, int], tuple[RuleScopeHint, ...]] = {}

        # Candidate-rule validation (evidence before trust). Imported lazily to
        # avoid a hard cycle at module import time (same as HarnessFilesystem).
        self._validation = validation
        if (
            self._validation is None
            and self._config.rule_validation
            and not self._config.observe_only
        ):
            from ..validation.validator import ValidationEngine

            assert self._evalset is not None  # built above when validation is on
            self._validation = ValidationEngine(
                config=self._config,
                rules=self._rules,
                evalset=self._evalset,
                evaluator=self._evaluator,
                journal=self._journal,
                replay=replay,
                verifier=verifier,
            )
        self._validation_tasks: set[asyncio.Task[None]] = set()

        # One store instance must be shared with any other reader (the store
        # memoizes its file cache), so the facade passes its instance in.
        self._history: HistorySource = history or ScoreHistoryStore(self._config)

        # Trace discovery + trajectory gate + tier ladder.
        # Share the evaluator's locator so one seen-set governs both paths.
        self._locator = locator or self._evaluator.locator
        self._tiers = TierRunner(
            self._config,
            self._evaluator,
            self._locator,
            TrajectoryGate(self._config, self._history),
            verifier=verifier,
        )

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

        # Startup health check (memoized).
        self._health_lock = asyncio.Lock()
        self._health_checked = False
        self._degraded_reason: str | None = None

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

    @property
    def pending_sessions(self) -> tuple[str, ...]:
        """Sessions whose evaluation is still in flight.

        A read-only view for host-side phase barriers ("has everything landed
        before I archive the workspace?"), so callers need not reach into the
        task bookkeeping.
        """

        return tuple(
            session_id
            for session_id, task in self._pending.items()
            if not task.done()
        )

    def startup_context(self, session_id: str, *, task_hint: str | None = None) -> str:
        """Stable capability-only preamble for one task session."""

        return compose_system_preamble(
            self._rules, self._mailbox, session_id, task_hint=task_hint
        )

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

        if self._repair is not None:
            self._repair.remember_turn(ctx)

        if ctx.rule_scope_hints:
            self._scope_hints[(ctx.session_id, ctx.turn_index)] = ctx.rule_scope_hints
            for hint in ctx.rule_scope_hints:
                try:
                    self._rules.register_scope_metadata(hint.key, hint.description)
                except Exception:  # noqa: BLE001 - metadata must not break a task
                    logger.debug("failed to persist rule scope metadata", exc_info=True)

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
        rate-limit counters — never a correctness bug, just a bounded reset.
        """

        try:
            oldest = next(iter(self._turn_counts))
        except StopIteration:  # pragma: no cover - guarded by the caller
            return
        self._turn_counts.pop(oldest, None)
        self._last_eval_at.pop(oldest, None)
        self._notice_state.pop(oldest, None)
        self._replay_inputs.pop(oldest, None)
        self._scope_hints = {
            key: value for key, value in self._scope_hints.items() if key[0] != oldest
        }
        # The locator's seen-set is bounded on its own, but evict in step so a
        # long-lived process does not retain trace ids for forgotten sessions.
        self._locator.forget(oldest)

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
            async with self._semaphore:
                report = await self._tiers.run(ctx)
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

    async def settle(self, session_id: str, *, timeout: float | None = None) -> SettleResult:
        """Block until task evaluation and managed repair have landed.

        The per-turn barrier that makes healing *in-session*: it waits for the
        turn's evaluation to resolve, its report to be handled (trial observation
        recorded, eval case captured), any notice to be posted, and one bounded
        managed repair attempt to finish. The next task turn can therefore see a
        newly written provisional candidate without racing the repair.

        It runs on ``barrier_timeout_s`` — deliberately generous, and separate
        from ``drain_timeout_s``, which is only a best-effort join. On expiry the
        evaluation work stays detached. Repair timeouts are cancelled, journaled,
        and leave the notice recoverable; neither path fails the developer task.

        It deliberately does **not** wait for the candidate-validation *round*.
        That round replays a captured case through the developer's agent, which
        can need the very resource the current turn holds — an environment lock, a
        world, a container — so awaiting it inside a per-turn barrier can deadlock
        until the replay times out. Validation is single-flight and detached;
        :meth:`drain_validation` awaits it at a phase boundary, where nothing is
        held. Candidate *observation* is not deferred: it happens inside the eval
        task this barrier does await.
        """

        budget = self._config.barrier_timeout_s if timeout is None else timeout
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, budget)

        report = await self._await_eval(session_id, deadline)
        eval_timed_out = report is None and self._pending.get(session_id) is not None
        repair = None
        if not eval_timed_out and self._repair is not None:
            remaining = max(0.0, deadline - loop.time())
            repair = await self._repair.settle(
                session_id,
                timeout_s=min(remaining, max(0.0, self._config.repair_timeout_s)),
            )
        timed_out = eval_timed_out or (repair is not None and repair.status == "timed_out")
        return SettleResult(
            session_id=session_id,
            report=report,
            repair=repair,
            timed_out=timed_out,
        )

    async def _await_eval(self, session_id: str, deadline: float) -> EvalReport | None:
        """Join the session's in-flight eval within the shared deadline."""

        task = self._pending.get(session_id)
        if task is None:
            return None
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            return None
        try:
            return await asyncio.wait_for(asyncio.shield(task), remaining)
        except TimeoutError:
            logger.warning(
                "settle: eval for session=%s did not land within the barrier budget",
                session_id,
            )
            return None
        except asyncio.CancelledError:
            # Same discipline as `refresh`: our caller being cancelled must
            # propagate, but a superseded (cancelled) eval is benign.
            current = asyncio.current_task()
            if current is not None and current.cancelling() > 0:
                raise
            if task.cancelled():
                return None
            raise

    # -- report handling ---------------------------------------------------------

    async def _handle_report(self, report: EvalReport) -> EvalReport:
        # 1. Candidate-rule validation: every handled report (healthy or
        #    alerting) feeds the forward trials — the trial needs the
        #    denominator — and kicks one single-flight evaluation round.
        await self._observe_for_validation(report)

        # 2. Dedup / cooldown / recovery gate.
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

        # 3. Circuit breaker, dump, notice (+ eval-case capture inside the
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

    @property
    def validation_pending(self) -> int:
        """How many candidate-validation rounds are in flight right now.

        A phase boundary must be able to see this. Waiting only on
        ``pending_sessions`` (evaluations) lets a snapshot be taken while a round
        is still deciding candidates, which records them as undecided forever.
        """

        return sum(1 for task in self._validation_tasks if not task.done())

    async def drain_validation(self, *, timeout: float | None = None) -> bool:
        """Await in-flight validation tasks; return whether they finished.

        ``timeout`` defaults to ``drain_timeout_s``, which is a best-effort join
        (15s by default) — far too short for a replay-bound round. A caller at a
        phase boundary should pass its own, larger budget. The return value
        distinguishes "drained" from "gave up", which a ``None`` return could not.

        Tasks are never cancelled: on expiry they keep running detached, exactly
        as before, so nothing is lost — the caller simply knows it did not wait
        long enough.
        """

        budget = self._config.drain_timeout_s if timeout is None else timeout
        tasks = [task for task in self._validation_tasks if not task.done()]
        if not tasks:
            return True
        _, still_running = await asyncio.wait(tasks, timeout=max(0.0, budget))
        return not still_running

    async def settle_validation(self, *, timeout: float) -> bool:
        """Run validation to a standstill: no candidate left both undecided and
        decidable, or the budget expires. Returns whether it settled.

        The per-turn barrier deliberately does not wait for validation (see
        :meth:`settle`). This is the phase-boundary counterpart: call it where
        nothing is held, before snapshotting or reporting a ruleset, so a
        candidate that had already earned promotion is not frozen as provisional.
        """

        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(0.0, timeout)
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            if not await self.drain_validation(timeout=remaining):
                return False
            if self._validation is None:
                return True
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            try:
                # Bounded: a round can be replay-bound, and a replay can block on
                # a resource that never frees. On expiry the round keeps running
                # detached — the caller learns it did not settle, and no evidence
                # is discarded.
                verdicts = await asyncio.wait_for(
                    asyncio.shield(
                        asyncio.ensure_future(self._validation.evaluate_candidates())
                    ),
                    remaining,
                )
            except TimeoutError:
                return False
            except Exception:  # noqa: BLE001 - degrade, never break the boundary
                logger.exception("candidate evaluation failed during settlement")
                return False
            # A round that decided nothing has nothing left to decide: every
            # remaining candidate is pending on evidence this loop cannot create
            # (more live sessions, a replayable case). Looping again would only
            # re-spend the budget on the same verdicts.
            if not any(verdict.outcome != "pending" for verdict in verdicts):
                return True

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
                        f"{window:.0f}s) — managed repair paused; human attention required"
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
        # Only breach-severity notices become failure cases. Tier-1 trajectory
        # notices are advisory, and `needs_human` is a rate alarm.
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

    # -- notice decisioning -----------------------------------------------------

    @staticmethod
    def _signatures(report: EvalReport) -> set[str]:
        return {
            f"{condition}:{score.metric}"
            for score in report.scores
            for condition in score.conditions
        }

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
        # Both are derived views over a frozen report; bind once rather than
        # rebuilding them per use.
        alerting = report.alerting_scores
        metrics = tuple(
            NoticeMetric(
                name=str(score.metric),
                value=score.value,
                threshold=score.threshold,
                # The judge's free-text `reason` is evidence used by managed
                # repair, so it must survive into the notice.
                reason=sanitize_text(score.reason, max_len=max_len) or None,
                conditions=score.conditions,
                trace_id=score.trace_id,
                tier=score.tier,
            )
            for score in alerting
        )
        host_hints = self._scope_hints.get((report.session_id, report.turn_index), ())
        # A host hint only *recommends*; it never decides. Prefer a hint that
        # names a real topic over one that merely restates a reserved name, and
        # leave `recommended_scope` unset when no hint applies — managed repair
        # must see "no recommendation", not a recommendation of the default.
        recommended = next(
            (
                hint
                for hint in host_hints
                if hint.recommended and hint.key not in RESERVED_SCOPES
            ),
            next((hint for hint in host_hints if hint.recommended), None),
        )
        recommended_scope = recommended.key if recommended is not None else None
        applicability = (
            recommended.applicability if recommended is not None else "global"
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
            scope_hint=self._scope_hint(report),
            scope_hints=tuple(hint.to_json() for hint in host_hints),
            recommended_scope=recommended_scope,
            applicability_hint=applicability,
        )

    @staticmethod
    def _scope_hint(report: EvalReport) -> str:
        """The baseline scope carried on a notice: always the default.

        Deliberately independent of the report. Evaluation tier says where a
        failure was *detected*, not how widely its lesson applies, so inferring a
        scope from it would be a mechanical guess dressed up as a judgment. The
        real decision belongs to managed repair, which has the failure evidence;
        this is only the floor it starts from.
        """

        del report
        return GLOBAL_SCOPE

    def _summarize(self, report: EvalReport) -> str:
        parts: list[str] = []
        for score in report.alerting_scores:
            value = f"{score.value:.2f}" if score.value is not None else "n/a"
            conds = "+".join(score.conditions) or "ok"
            parts.append(f"{score.metric}={value} [{conds}, threshold {score.threshold:.2f}]")
        line = "; ".join(parts) if parts else "no alerting scores"
        flagged = report.flagged_traces
        if flagged:
            line += f"; flagged traces: {', '.join(flagged[:5])}"
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
