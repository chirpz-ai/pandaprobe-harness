"""Central configuration for the PandaProbe Harness.

``HarnessConfig`` is the single source of truth for filesystem paths, CLI
invocation tunables, metric thresholds, and trajectory-gate knobs. It is a
frozen dataclass so it can be shared freely across async tasks without risk of
mutation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["HarnessConfig"]

DEFAULT_THRESHOLD = 0.5

# -- v2 trace-level trigger ---------------------------------------------------
# Tier 1 is the always-on progress signal: `task_completion` is the outcome
# trajectory and `coherence` is an embedding-based (free) co-gate.
DEFAULT_TIER1_METRICS: tuple[str, ...] = ("task_completion", "coherence")
# Tier 2 is the surgical step-level diagnosis, run only once Tier 1 breaches.
DEFAULT_TIER2_METRICS: tuple[str, ...] = ("tool_correctness", "argument_correctness")
# Tier 3 only enriches the reasoning behind a confirmed Tier-2 breach; it is
# never a breach source of its own, and it is opt-in because every LLM-judge
# metric reads the whole trace (cost scales with metric count).
DEFAULT_TIER3_METRICS: tuple[str, ...] = (
    "plan_adherence",
    "plan_quality",
    "step_efficiency",
)
# NOTE: the trace metrics deliberately have no per-metric threshold table. They
# all sit at DEFAULT_THRESHOLD, and the platform reports its own `threshold` in
# each score's metadata (which the evaluator prefers when no local override
# exists), so a table of identical values would only be a second place to spell
# a metric name — and one that has already drifted once. Calibrate by setting
# `thresholds={...}` on the config; Checkpoint 1 records the chosen values.


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class HarnessConfig:
    """Immutable harness configuration.

    Path fields ``traces_dir``, ``rules_file``, ``rules_dir``,
    ``latest_eval_file``, ``state_dir``, ``history_file`` and the
    mailbox/journal/rules-store paths are derived from ``harness_root`` in
    ``__post_init__`` and should not be passed explicitly.
    """

    harness_root: Path = Path("/harness")

    # Derived paths (init=False; computed from harness_root).
    traces_dir: Path = field(init=False)
    rules_file: Path = field(init=False)
    rules_dir: Path = field(init=False)
    latest_eval_file: Path = field(init=False)
    state_dir: Path = field(init=False)
    history_file: Path = field(init=False)
    mailbox_dir: Path = field(init=False)
    mailbox_pending_dir: Path = field(init=False)
    mailbox_processed_dir: Path = field(init=False)
    mailbox_status_file: Path = field(init=False)
    journal_file: Path = field(init=False)
    rules_store_file: Path = field(init=False)
    evalset_dir: Path = field(init=False)

    # CLI invocation.
    cli_binary: str = "pandaprobe"
    cli_timeout_s: float = 30.0

    # Async eval-run polling (evals runs are asynchronous on the platform).
    poll_interval_s: float = 1.0
    poll_max_attempts: int = 20

    # Eventual-consistency retry: the SDK flushes traces on a background thread,
    # so a freshly-ended session may not be scorable immediately.
    eval_retry_attempts: int = 3
    eval_retry_backoff_s: float = 1.0

    # Bounded await-barrier drained at the start of the next turn.
    drain_timeout_s: float = 15.0
    # The per-turn self-heal barrier (``settle``) runs on its own, deliberately
    # generous budget: it must actually wait for the turn's evaluation to land,
    # where ``drain_timeout_s`` is only a best-effort join.
    barrier_timeout_s: float = 180.0

    # -- trace-level trigger ---------------------------------------------------
    trace_metrics_tier1: tuple[str, ...] = DEFAULT_TIER1_METRICS
    trace_metrics_tier2: tuple[str, ...] = DEFAULT_TIER2_METRICS
    trace_metrics_tier3: tuple[str, ...] = DEFAULT_TIER3_METRICS
    enable_tier3: bool = False
    # Max traces fetched per session when discovering newly-completed traces.
    trace_list_limit: int = 50

    # -- trajectory gate (the primary knob is ``gate_window``) -----------------
    # A Tier-1 metric breaches on a *trajectory*, never a single low value:
    # STALL (no gain toward the target for ``gate_window`` traces) or REGRESSION
    # (a drop of ``gate_drop`` from the running peak). Any gain of at least
    # ``gate_gain`` resets the window, so a healthy climb never breaches.
    gate_window: int = 5
    gate_target: float = 0.5
    gate_gain: float = 0.02
    gate_drop: float = 0.15

    # -- optional developer outcome verifier ----------------------------------
    # Threshold for the synthetic ``outcome_correct`` score a verifier emits.
    outcome_threshold: float = 0.9

    # -- metrics & thresholds -------------------------------------------------
    # Per-metric absolute thresholds (overrides the scalar defaults below).
    thresholds: dict[str, float] = field(default_factory=dict)
    concurrent_eval: bool = True  # retained for back-compat; one run now covers all metrics

    # -- noticing (the pull-model mailbox) ------------------------------------
    # Suppress re-posting an identical notice signature for this many turns.
    # 0 means "suppress until the condition recovers".
    alert_cooldown_turns: int = 0
    # Optionally enrich the dump with the worst flagged trace's tool spans.
    enrich_flagged_traces: bool = False
    # Shadow mode: evaluate + journal, but never post mailbox notices.
    observe_only: bool = False
    # Escalate to a single `needs_human` notice when this many notices are
    # posted within the window (0 disables the circuit breaker).
    circuit_breaker_max_notices: int = 5
    circuit_breaker_window_s: float = 600.0

    # -- cost / latency / sampling controls -----------------------------------
    # Evaluate every Nth turn per session (1 = every turn).
    eval_sample_every: int = 1
    # Minimum seconds between eval launches for one session (0 disables).
    session_min_eval_interval_s: float = 0.0
    # Global cap on concurrently-running evaluations across all sessions.
    max_concurrent_evals: int = 4
    # Hard budget of eval launches for this process (0 = unlimited). A cheap
    # cost proxy: each launch is one platform eval run.
    max_evals_per_run: int = 0

    # -- self-heal rules -------------------------------------------------------
    # Cap on concurrently-live structured rules (agent must retire to add).
    max_active_rules: int = 50
    # Length cap applied when sanitizing eval-derived free text.
    sanitize_max_len: int = 2000

    # -- rule validation (evidence before trust) -------------------------------
    # New rules start as candidates and are promoted to active only after a
    # validator (replay or forward trial) shows they help. False restores the
    # v0.5 behavior: rules enter `active` the moment they are written.
    rule_validation: bool = True
    # Forward trial: distinct sessions to observe before a verdict.
    rule_trial_min_sessions: int = 5
    # Minimum improvement of the targeted metric to promote a candidate.
    rule_promote_margin: float = 0.05
    # Maximum tolerated drop on any other case/metric before a candidate or a
    # regression-run case counts as regressed.
    rule_regress_margin: float = 0.05
    # Hard bound on one developer replay invocation; a hung replay degrades to
    # an inconclusive case instead of wedging validation/regression forever.
    replay_timeout_s: float = 300.0

    # -- regression eval-set ---------------------------------------------------
    # Capture breaching sessions as replayable eval cases (opt-in: stores
    # session-derived data under the workspace).
    capture_eval_cases: bool = False
    # Corpus cap; oldest failure cases are evicted first, wins never.
    eval_case_max: int = 200
    # Cases replayed per regression run (0 = all).
    regression_sample: int = 0

    # -- rule retrieval (relevance over volume) --------------------------------
    # Inject only global rules + the top-k rules relevant to the current
    # situation instead of every active rule. False restores v0.5 rendering.
    rule_retrieval: bool = True
    # How many tagged rules the retrieval keeps in the system context.
    rules_context_topk: int = 8

    # -- robustness ------------------------------------------------------------
    # Verify the CLI is present and authenticated before the first eval.
    health_check: bool = True

    def __post_init__(self) -> None:
        root = Path(self.harness_root)
        # object.__setattr__ is required to populate fields on a frozen dataclass.
        object.__setattr__(self, "harness_root", root)
        object.__setattr__(self, "traces_dir", root / "traces")
        object.__setattr__(self, "rules_file", root / "harness_rules.md")
        object.__setattr__(self, "rules_dir", root / "rules")
        object.__setattr__(self, "latest_eval_file", root / "traces" / "latest_eval.json")
        object.__setattr__(self, "state_dir", root / "state")
        object.__setattr__(self, "history_file", root / "state" / "score_history.json")
        object.__setattr__(self, "mailbox_dir", root / "mailbox")
        object.__setattr__(self, "mailbox_pending_dir", root / "mailbox" / "pending")
        object.__setattr__(self, "mailbox_processed_dir", root / "mailbox" / "processed")
        object.__setattr__(self, "mailbox_status_file", root / "mailbox" / "status.json")
        object.__setattr__(self, "journal_file", root / "journal.jsonl")
        object.__setattr__(self, "rules_store_file", root / "rules.jsonl")
        object.__setattr__(self, "evalset_dir", root / "evalset")

    # -- helpers --------------------------------------------------------------

    def threshold_for(self, metric: str) -> float:
        """Resolve the absolute breach threshold for a metric name."""

        if metric in self.thresholds:
            return self.thresholds[metric]
        if metric == "outcome_correct":
            return self.outcome_threshold
        return DEFAULT_THRESHOLD

    def tier_metrics(self, tier: int) -> tuple[str, ...]:
        """The trace metrics to run for ``tier`` (1, 2 or 3).

        Tier 3 resolves to an empty tuple while ``enable_tier3`` is off, so the
        caller never needs to special-case the flag.
        """

        if tier == 1:
            return tuple(self.trace_metrics_tier1)
        if tier == 2:
            return tuple(self.trace_metrics_tier2)
        if tier == 3:
            return tuple(self.trace_metrics_tier3) if self.enable_tier3 else ()
        return ()

    def replay_metrics(self) -> tuple[str, ...]:
        """The trace metrics a replayed session is graded on: Tier 1 plus Tier 2.

        Tier 1 carries the outcome trajectory and Tier 2 the step-level verdict, so
        between them they cover whatever a notice can have triggered on. Tier 3 is
        deliberately excluded — it never triggers a breach, so re-scoring it would
        only add cost.
        """

        return tuple(dict.fromkeys((*self.tier_metrics(1), *self.tier_metrics(2))))

    def rules_scope_file(self, scope: str) -> Path:
        """Path of the agent-facing rule file for ``scope``.

        ``scope`` is agent-supplied and becomes a filename, so callers must pass
        a value already normalized by ``workspace.rules.normalize_scope``.
        """

        return self.rules_dir / f"{scope}.md"

    @classmethod
    def from_env(cls, **overrides: object) -> HarnessConfig:
        """Build a config from ``HARNESS_*`` / ``PANDAPROBE_*`` environment vars.

        Explicit ``overrides`` take precedence over environment values.
        """

        values: dict[str, object] = {
            "harness_root": Path(os.environ.get("HARNESS_ROOT", "/harness")),
            "cli_binary": os.environ.get("HARNESS_CLI_BINARY", "pandaprobe"),
            "cli_timeout_s": _env_float("HARNESS_CLI_TIMEOUT_S", 30.0),
            "poll_interval_s": _env_float("HARNESS_POLL_INTERVAL_S", 1.0),
            "poll_max_attempts": _env_int("HARNESS_POLL_MAX_ATTEMPTS", 20),
            "eval_retry_attempts": _env_int("HARNESS_EVAL_RETRY_ATTEMPTS", 3),
            "eval_retry_backoff_s": _env_float("HARNESS_EVAL_RETRY_BACKOFF_S", 1.0),
            "drain_timeout_s": _env_float("HARNESS_DRAIN_TIMEOUT_S", 15.0),
            # The four primary v2 knobs. Everything else about the trace
            # trigger is defaulted on purpose — adopting the harness should
            # need almost no configuration.
            "barrier_timeout_s": _env_float("HARNESS_BARRIER_TIMEOUT_S", 180.0),
            "gate_window": _env_int("HARNESS_GATE_WINDOW", 5),
            "enable_tier3": _env_bool("HARNESS_ENABLE_TIER3", False),
            "concurrent_eval": _env_bool("HARNESS_CONCURRENT_EVAL", True),
            "alert_cooldown_turns": _env_int("HARNESS_ALERT_COOLDOWN_TURNS", 0),
            "enrich_flagged_traces": _env_bool("HARNESS_ENRICH_FLAGGED_TRACES", False),
            "observe_only": _env_bool("HARNESS_OBSERVE_ONLY", False),
            "circuit_breaker_max_notices": _env_int("HARNESS_CIRCUIT_BREAKER_MAX_NOTICES", 5),
            "circuit_breaker_window_s": _env_float("HARNESS_CIRCUIT_BREAKER_WINDOW_S", 600.0),
            "eval_sample_every": _env_int("HARNESS_EVAL_SAMPLE_EVERY", 1),
            "session_min_eval_interval_s": _env_float(
                "HARNESS_SESSION_MIN_EVAL_INTERVAL_S", 0.0
            ),
            "max_concurrent_evals": _env_int("HARNESS_MAX_CONCURRENT_EVALS", 4),
            "max_evals_per_run": _env_int("HARNESS_MAX_EVALS_PER_RUN", 0),
            "max_active_rules": _env_int("HARNESS_MAX_ACTIVE_RULES", 50),
            "sanitize_max_len": _env_int("HARNESS_SANITIZE_MAX_LEN", 2000),
            "rule_validation": _env_bool("HARNESS_RULE_VALIDATION", True),
            "rule_trial_min_sessions": _env_int("HARNESS_RULE_TRIAL_MIN_SESSIONS", 5),
            "rule_promote_margin": _env_float("HARNESS_RULE_PROMOTE_MARGIN", 0.05),
            "rule_regress_margin": _env_float("HARNESS_RULE_REGRESS_MARGIN", 0.05),
            "replay_timeout_s": _env_float("HARNESS_REPLAY_TIMEOUT_S", 300.0),
            "capture_eval_cases": _env_bool("HARNESS_CAPTURE_EVAL_CASES", False),
            "eval_case_max": _env_int("HARNESS_EVAL_CASE_MAX", 200),
            "regression_sample": _env_int("HARNESS_REGRESSION_SAMPLE", 0),
            "rule_retrieval": _env_bool("HARNESS_RULE_RETRIEVAL", True),
            "rules_context_topk": _env_int("HARNESS_RULES_CONTEXT_TOPK", 8),
            "health_check": _env_bool("HARNESS_HEALTH_CHECK", True),
        }
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]
