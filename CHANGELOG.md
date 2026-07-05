# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2026-07-03

The "closed loop" release. v0.5 detected failures, proposed rules, and applied
them — but never confirmed a rule actually helped. v0.6 closes the loop:
**evidence before trust** (a rule must prove itself before it is trusted),
**relevance over volume** (only the rules relevant to the current situation
enter the prompt), and **measure the foundation** (an offline calibration tool
for the breach thresholds everything keys off). All of it automatic — no human
in the healing loop.

### Changed

- **BREAKING (behavior)**: `harness_rule_add` now records a **candidate**
  rule, not an active one. Candidates are still injected into the system
  context (under a clearly-labeled "Provisional rules (under evaluation)"
  section — a rule must be in force to be measurable) and are promoted to
  `active` only after a validator shows they help: `ReplayValidator` (replays
  the captured failing scenario through a developer-supplied replay function
  and requires the targeted metric to improve past `rule_promote_margin` with
  no case regressing past `rule_regress_margin`) or, when no replay function
  is wired, `ForwardTrialValidator` (compares the signature's breach rate
  over the next `rule_trial_min_sessions` live sessions against the baseline
  captured at add time). Unfavorable candidates are retired with a journaled
  reason. Set `rule_validation=false` (`HARNESS_RULE_VALIDATION=false`) to
  restore the v0.5 add→active behavior.
- **Rule retrieval is task-conditioned by default**: the system preamble now
  renders global (untagged) rules plus the top-`rules_context_topk` rules
  lexically relevant to the pending notices and an optional
  `system_context(task_hint=...)` — not the full set. The rest stay reachable
  via `harness_rules_search` / `harness_rules_list`. Set
  `rule_retrieval=false` to restore v0.5 render-everything behavior.
- `harness_rule_retire` now retires candidates as well as active rules and
  accepts a journaled `reason`; the dedup/cap in `RulesStore.add` now count
  the whole live set (active + candidate).
- `harness_reflect` additionally returns `candidate_rules` and
  `recent_validations` (promote/retire outcomes) so the reflection cycle can
  learn which kinds of rules survive validation.

### Added

- **Rule lifecycle** (`candidate → active | retired`) with `Rule.tags`
  (auto-derived from the source notice's signatures, metrics, and signal
  names), `Rule.trial` (`TrialState` bookkeeping: baseline vs. trial breach
  rates, observed/breached sessions, replay attempts, verdict), and
  `RulesStore.promote()/update_trial()/live()/candidates()`.
- **Validation package** (`pandaprobe_harness.validation`): `RuleValidator`
  protocol, `ReplayValidator`, `ForwardTrialValidator`, and the
  `ValidationEngine` the hook drives automatically on every handled report
  (single-flight, never blocks or raises into the host loop). New journal
  events: `rule_promote`, `rule_retire` (with reason), `validation`
  (fallback announcement), `evalset_capture`, `regression`.
- **Replayable regression eval-set** (`EvalSet`, `<harness_root>/evalset/`):
  breaching sessions are captured as `failure` cases (opt-in via
  `capture_eval_cases`) with their signature, baseline scores, and — when the
  turn payload carries one — the replay input; known-good sessions can be
  captured as protected `win` cases (never auto-evicted; failures evict
  oldest-first at `eval_case_max`). The **`ReplayFn` seam**
  (`(case, system_context) -> new_session_id`) is how the harness re-runs
  the developer's agent; wire it via `Harness.create(..., replay=...)`.
- `harness.run_regression()` + the **`pandaprobe-harness-eval`** operator CLI:
  replay the eval-set (wins first) against the current rule set and classify
  each case improved/unchanged/regressed vs. baseline — the "did a new rule
  break an old win" guard. Without a replay function it degrades to one clear
  warning and all-skipped results, never a crash.
- **Metric calibration** (`pandaprobe_harness.calibration` + the
  **`pandaprobe-harness-calibrate`** operator CLI): with labels (JSON/CSV, or
  eval-set kinds via `--from-evalset`) — precision/recall/F1 of the breach
  predicate, a confusion matrix, and a threshold sweep with the
  F1-maximizing threshold and the lowest threshold hitting a target
  precision; without labels — score distribution, histogram, sweep, and
  inter-metric agreement. Stdlib-only, fully offline-testable.
- New toolset operations (9 → 14): `harness_rule_status`,
  `harness_rules_search`, `harness_rules_list`, `harness_evalset_list`,
  `harness_evalset_attach`.
- New facade surface: `Harness.create(..., replay=)` (and all `for_*`
  factories), `harness.evalset`, `harness.run_regression()`,
  `harness.validate_candidates()`, `harness.drain_validation()`,
  `harness.system_context(task_hint=...)`.
- Config knobs (all mirrored as `HARNESS_*` env vars): `rule_validation`,
  `rule_trial_min_sessions`, `rule_promote_margin`, `rule_regress_margin`,
  `replay_timeout_s`, `capture_eval_cases`, `eval_case_max`,
  `regression_sample`, `rule_retrieval`, `rules_context_topk`.

### Fixed

- `Rule.from_json` no longer coerces unknown statuses to `active` — a
  persisted `candidate` now round-trips instead of silently self-promoting
  across process restarts.

## [0.5.0] - 2026-07-01

The "pull model" release. The harness no longer pushes alerts into agent
transcripts; it posts structured `DiagnosticNotice`s to a filesystem mailbox
that the agent pulls from via tools, and it maintains a durable journal and a
structured self-heal rules store.

### Added

- Workspace substrate: `Mailbox` with `DiagnosticNotice` records
  (`mailbox/pending/*.json` → `mailbox/processed/`), an append-only `Journal`
  (`journal.jsonl`), and a `RulesStore` (`rules.jsonl`) with provenance,
  dedup, an active-rule cap, and per-rule effectiveness tracking.
- `HarnessToolset` exposing 9 agent-facing operations over the workspace, the
  `pandaprobe-harness-agent` companion CLI (sandbox-allow-listed), and native
  tool registrations for the supported frameworks.
- `Harness` facade with zero-adapter `turn()` / `run_turn()` entry points.
- Cost/latency controls: per-session eval sampling (`eval_sample_every`),
  per-session rate limiting (`session_min_eval_interval_s`), a global
  concurrency cap (`max_concurrent_evals`), and a hard eval budget
  (`max_evals_per_run`).
- `observe_only` shadow mode: evaluate and journal without posting notices.
- Circuit breaker that escalates to a single `needs_human` notice when too
  many notices fire within a window (`circuit_breaker_max_notices`).
- Startup health check (CLI presence + auth) with a degraded, journal-only
  mode when it fails (`health_check`).
- Backend history hydration (`HistorySource`) to seed local trend state once
  per session for horizontally-scaled agents
  (`hydrate_history_from_backend`).
- Sandbox hardening: environment-variable scoping and argv deny rules in the
  restricted shell policy.
- Sanitization trust boundary for eval-derived free text crossing into agent
  context.

### Security

- Mailbox rejects `notice_id`s that are not a single safe path component, so a
  crafted id (e.g. `../../state/score_history`) can no longer escape the
  mailbox directory to read, overwrite, or delete arbitrary workspace files.
- The restricted shell's path-escape guard now catches mid-path traversal
  (`state/../../etc/passwd`), not only tokens that begin with a separator.
- Argv deny rules match the subcommand words as an ordered subsequence, so a
  leading global option (`pandaprobe --format json config show`) no longer
  bypasses the `pandaprobe config` / `auth login` denials, and denied flags
  are matched with or without an `=value` suffix.

### Fixed

- A missing/unexecutable `pandaprobe` binary now surfaces as a `CliError`
  instead of a raw `OSError`, so the startup health check degrades gracefully
  (one warning + a journal `health` event) instead of crashing the host loop.
- `refresh()` no longer swallows the caller's own cancellation when the
  awaited evaluation was concurrently superseded.
- Backend history hydration seeds the EWMA in chronological order, preventing
  a spurious trend verdict on the first post-hydration turn.
- `harness_journal` clamps a non-positive/oversized `limit` instead of
  dumping the entire journal into the tool result.
- The companion CLI rejects a flag-shaped value (a forgotten argument) rather
  than silently persisting the next flag as data.
- Per-session bookkeeping and framework instrumentation are bounded/idempotent
  across many sessions and repeated `Harness.for_*` builds.
- `py.typed` marker and a GitHub Actions CI pipeline (lint, typecheck, tests,
  per-extra adapter matrix, sandbox image build) plus a scheduled live
  contract workflow.

### Changed

- Framework adapters are now pure turn-detectors; they no longer carry alert
  queues or injection surfaces.
- `PandaHarnessHook` constructor is `(cli, *, ...)` — the adapter argument is
  gone; wiring is keyword-only (`config`, `mailbox`, `journal`, `rules`,
  `filesystem`, `evaluator`, `parser`, `history`).
- `compose_system_preamble(rules, mailbox)` renders the startup preamble from
  the rules store and mailbox status instead of consuming queued alerts.
- `drain_pending()` is replaced by `refresh(session_id)` / `refresh_all()`,
  bounded await-barriers over in-flight evaluation tasks.
- `harness_rules.md` is now rendered from `rules.jsonl` (the structured store
  is the source of truth; the markdown file is a projection).

### Removed

**BREAKING** — the push-model alert-injection surface is gone:

- `FrameworkAdapter.inject_alert`
- `BaseSinkAdapter.inject_alert` / `pending_alerts` / `consume_alerts`
- `LangChainCallbackAdapter.consume_messages` / `startup_messages` /
  `drain_into`
- `CrewAIAdapter.consume_context`
- `ClaudeAgentSDKAdapter.inject_into_history` / `prime_startup`
- `OpenAIAgentsAdapter.consume_input_items` / `startup_input_items`
- `RawLoopAdapter` alert queue
- `hook/alert.py` (`build_system_alert`, `build_trend_alert`)
- `PandaHarnessHook.drain_pending`
- `HarnessFilesystem.append_rule`

## [0.4.0]

- Async, supersede-cancelling evaluation loop with EWMA trend detection,
  adaptive (relative) thresholds, and per-signature alert cooldowns.
- Single batched eval run per turn covering all active session metrics, with
  eventual-consistency retries and bounded run polling.
- Framework adapter suite: LangGraph, LangChain, DeepAgents, CrewAI, Claude
  Agent SDK, and OpenAI Agents.

## [0.3.0]

- Initial public harness: `pandaprobe` CLI subprocess seam, turn-end
  evaluation hook with absolute score thresholds, trace dumps under
  `traces/`, `harness_rules.md`, and the Dockerised diagnostic sandbox with a
  restricted shell.
