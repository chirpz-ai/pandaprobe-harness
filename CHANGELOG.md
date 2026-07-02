# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
