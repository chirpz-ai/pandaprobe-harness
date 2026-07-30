# PandaProbe Harness v2 — Refactor Plan

> A step-by-step implementation brief for a coding agent. Read Section 0–2 fully
> before writing any code. Do not begin changes until you have completed the
> mandatory codebase review in Section 2.

---

## 0. How to use this plan

1. **Read the whole plan first.** It has six changes (3 primary + 3 supporting) that touch overlapping modules; understand all of them before editing.
2. **Complete the mandatory codebase review (Section 2)** and confirm current behavior with a run before touching anything.
3. Implement changes in the order given (they build on each other).
4. **Verify every change** with the procedures in Section 9. A change is not "done" until it is verified working end-to-end, not just type-checked.
5. Respect the invariants in Section 10 — especially the product philosophy.

---

## 1. Project intro — what we are building

**PandaProbe Harness** is a universal, self-healing wrapper for LLM agents. It passively evaluates an agent's trajectories, detects when the agent is failing, and lets the **agent itself** write, test, promote, and retire natural-language *rules* that improve its own behavior over time. It is a general-purpose developer package — it must work for *any* agent, not any single benchmark.

**Why v2 (what's wrong with v1):**
- The trigger is **session-level** LLM-judged metrics (`agent_reliability`, `agent_consistency`). Empirically these are **degenerate**: they sit at ~0.2 for essentially every session because they are a worst-case/aggregate rollup over many per-trace signals — with enough traces there is always one bad trace, so the session score floors. The trigger therefore carries almost no signal (it "breaches" on nearly everything).
- Because the metric doesn't discriminate, **rule promotion is effectively random** (validation is scored on the same degenerate metric).
- The agent was given a large diagnostic toolset plus a "check your mailbox every turn" mandate, so a capable agent **spent its turns operating the harness instead of doing the task** (a measured regression).

**v2 thesis (three primary changes):**
1. **Trace-level, three-tier trigger with trajectory gating** — replace the flat session-level trigger with per-trace metrics that actually discriminate, organized into cost-aware tiers, gated on a *trajectory* (trend), not a point threshold.
2. **Agent-owned SKILL-pattern rules workspace** — a lightweight `harness_rules.md` "skill root" that holds only high-level instructions + a directory of references, and a new `rules/` subtree that holds all rule content (global + scoped) and the full candidate/active/retired lifecycle. The agent reads/writes via tools by its own intelligence; nothing is force-injected.
3. **Per-turn agent-await barrier** — a first-class synchronous barrier so the agent pauses each turn while the harness evaluates and the agent processes notices / writes rules, enabling **in-session** self-healing.

**Plus three supporting changes:** an optional developer **outcome-verifier hook** (gold trigger), **config minimalism** (one primary knob), and **validation/promotion re-pointed to the trace metrics**.

**Product philosophy (NON-NEGOTIABLE — see Section 10):** the agent owns the workspace; the agent decides what to read, when, and what rules to write, using harness tools driven by its own intelligence; the harness **never force-injects** rules or notices into the agent's system prompt. This mirrors the "agent skill" pattern: a lightweight root doc with references the agent pulls on demand.

---

## 2. Codebase structure & MANDATORY first step

**Repo:** `/Users/sina/Projects/PandaProbe/pandaprobe-harness`

- `src/pandaprobe_harness/` — **the package you will change.**
- `benchmarks/` — a consumer project (AppWorld / Terminal-Bench / τ²) used here as the **real-world verification harness**. See `benchmarks/RUNNING.md`.
- `tests/` — unit test suite. **Must stay green.**
- `examples/` — usage examples (check these reflect any public-API change).

**Package modules to review (all under `src/pandaprobe_harness/`):**

| Module | What it does (why you must understand it) |
|---|---|
| `harness.py` | Public facade: `Harness.create(...)`, `on_turn_end`, `refresh`, `system_context`, `toolset`, `rules`, turn scopes. All DI wiring lives here. |
| `hook/core.py` | The heart: `on_turn_end` → admission → detached `_run_eval` → `_handle_report` → trend detection → breach gate → notice. This is where the trigger logic lives. |
| `hook/context.py` | Builds the agent's startup/system context (the pull protocol + rules render). |
| `hook/turn.py` | `TurnContext` parsing (session_id, turn_index, end_state). |
| `evaluation/evaluator.py` | `MetricEvaluator`: **runs evals by spawning the `pandaprobe` CLI as a subprocess** (`evals runs batch` + poll `evals runs scores`). Currently hardwired to `--target session`. |
| `evaluation/metrics.py` | `Metric` enum, `MetricScore`, `EvalReport`, breach/alerting logic, `signal_breakdown`, dump shape. |
| `evaluation/thresholds.py` | Breach policy. |
| `config.py` | `HarnessConfig`: all knobs (metrics, thresholds, poll, trend, capture, validation, retrieval, paths) + `from_env`. |
| `workspace/rules.py` | `RulesStore` (rules.jsonl) + `render_markdown`/`sync_markdown` → `harness_rules.md`; retrieval `relevant(query,k)`; tagging `derive_notice_tags`. |
| `workspace/mailbox.py` | `DiagnosticNotice`, `NoticeMetric`, filesystem mailbox. |
| `workspace/evalset.py` | Eval-case capture + `ReplayFn` type. |
| `validation/validator.py` | `ReplayValidator` / `ForwardTrialValidator`: promotion/retirement, scored via `MetricEvaluator`. |
| `validation/regression.py` | Regression run over captured cases. |
| `filesystem/layout.py` | On-disk paths (traces/, rules files, mailbox dirs, journal, evalset, state). |
| `agent_tools/toolset.py` | The `harness_*` tools the agent calls (mailbox, trace inspect, history, journal, rule add/retire, ack). |
| `cli/subprocess_client.py`, `cli/client.py`, `cli/models.py` | The subprocess CLI client + JSON parsing. |
| `calibration.py` | Checkpoint-1 calibration (metric ↔ failure). |

**MANDATORY before any change — produce a short written map (to yourself) confirming you understand:**
1. The exact **eval invocation**: how `_run_eval` → `MetricEvaluator.evaluate_turn` shells out (`pandaprobe --format json evals runs batch --target session --session-ids <sid> --metrics ...`, then polls `evals runs scores <run_id> --target session`), and how results parse into `EvalReport`.
2. The **trigger/breach path**: `_handle_report` → `_apply_trends` (EWMA/percentile) → `_should_notice` → `_build_notice` → mailbox. What data a `DiagnosticNotice` carries (`NoticeMetric{name,value,threshold,reason,conditions}`, `flagged_traces`, `signal_breakdown`, `dump_path`).
3. The **rules artifact**: how `RulesStore.render_markdown`/`sync_markdown` writes `harness_rules.md` (template preamble **+ live rules**), and how `harness_rule_add` (toolset) → `RulesStore.add` → re-render works. Note that today the rules content and the instructions live in the **same** file.
4. The **retrieval**: `RulesStore.relevant(query,k)` (untagged = global/always; tagged = ranked by relevance) and `derive_notice_tags`.
5. The **validation**: `ReplayValidator` target metric, `baseline_scores` (currently session-level from the capture dump), `_improved` / promote/retire margins.
6. The **config surface** and which knobs are `from_env`-bound.

**Then confirm current behavior by reviewing an existing run — you do NOT need to execute any benchmark.** A completed harness-arm AppWorld run is on disk; read it to see the v1 loop's actual output:

```
benchmarks/results/runs/appworld_gemini-3.1-pro_harness_1_20260717-050505_old
```

Inspect its `harness_root/` to ground your understanding of current behavior:
- `harness_rules.md` — see that it currently holds **both** the instructions/protocol **and** the live rule bodies (this is exactly what Change 2 splits apart).
- `rules.jsonl` — the structured rule store and the candidate/active/retired transitions.
- `mailbox/` (`pending/`, `processed/`) and `traces/*.json` — the `DiagnosticNotice` payloads and eval dumps: confirm what data a notice carries and how much per-trace detail survives.
- `journal.jsonl` — the cross-run event log (notice / eval-case capture / rule_add / rule_retire / rule_promote counts) — shows the loop firing end to end.
- `state/score_history.json` — the per-session metric EWMA series the trend detector maintains.
- The run's `records.jsonl` and `manifest.json` — per-trial outcomes and the run config.

This static review is sufficient to understand the current session-level trigger, the notice/rule flow, and the single-file rules artifact. (Optionally, if credentials are configured, you may sanity-check the trace-eval CLI surface — `pandaprobe evals metrics --target trace` lists the 8 trace-runnable metrics — but this is not required to begin.)

**Verified facts you can rely on (already confirmed):**
- Trace-runnable metrics: `task_completion, tool_correctness, argument_correctness, step_efficiency, plan_adherence, plan_quality, coherence, confidence`. **`loop_detection` is NOT runnable at trace level** (HTTP 422) — it only exists inside the session composite.
- `coherence` is embedding-based (cheap/free); the rest are LLM judges. Each LLM-judge metric reads the whole target trace, so **cost scales with the number of metrics run, not which ones**.
- Score objects carry `value`, a rich `reason`, and `metadata` (threshold, per-metric fields). The `reason` field is high quality and is the raw material for rules.

---

## 3. PRIMARY CHANGE 1 — Trace-level, three-tier trigger with trajectory gating

**Goal:** Replace the degenerate session-level trigger with per-trace metrics organized into three cost-aware tiers, gated on a trajectory (trend), not a point threshold. This is the core of v2.

### 3.1 Target behavior

Per session, as traces arrive (all evals async):

- **Tier 1 — `task_completion` + `coherence`, EVERY trace.** The always-on progress/outcome signal and the gate. Breach is **trajectory-based**, never a single low value:
  - **STALL:** over a sliding window of `W` traces, `task_completion` shows no gain toward a target (plateaued below target).
  - **REGRESSION:** current value drops from the running peak by `δ_drop`.
  - **RESET-ON-GAIN:** any real improvement resets the stall window, so a healthy climbing session never breaches.
  - `coherence` gets the same trajectory treatment (sustained-low or declining EWMA); it's a free co-gate.
- **Tier 2 — `tool_correctness` + `argument_correctness`, LAST TRACE ONLY, only when Tier 1 breaches.** The last trace carries the full trajectory in its context and is the current state to fix. Tier-2 breach = a step metric below its calibrated threshold → this is the surgical breach that drives a scoped rule.
- **Tier 3 — `plan_adherence`, `plan_quality`, `step_efficiency`, LAST TRACE ONLY, only on a confirmed Tier-2 breach (opt-in).** Enriches reasoning for a better rule. Not a breach source of its own.
- **Drop `loop_detection`** (not trace-runnable) and **`confidence`** (value floors; reasoning redundant with `tool_correctness`/`task_completion`).
- Every breach produces a **notice carrying the triggering metric(s) `value` + `reason`** (the reason is essential for the agent's rule).

### 3.2 The trajectory gate (implement exactly this logic)

```
# per session, per Tier-1 metric (task_completion, coherence):
state = { peak: None, turns_since_gain: 0 }
on new value v at trace t:
    if state.peak is None or v >= state.peak + δ_gain:
        state.peak = v; state.turns_since_gain = 0        # progressing → never breach
    else:
        state.turns_since_gain += 1
    breach = (
        (state.turns_since_gain >= W and state.peak < target)   # STALL below target
        or (v < state.peak - δ_drop)                            # REGRESSION
    )
    if breach: open_gate(); reset the window/counter          # don't re-fire every trace
```
- `W` = cadence (default ~5), `target` (default ~0.5), `δ_gain`, `δ_drop` — all config (Section 7).
- **Reuse the existing trend machinery** (`enable_trend`, `ewma_fast_span`, `ewma_slow_span`, `trend_margin_cross`, `percentile_floor`, `trend_min_samples`) — repoint it from session composites to per-trace `task_completion`/`coherence`. `W`→`trend_min_samples`, `δ_drop`→`trend_margin_cross`, `target`→an absolute floor/`percentile_floor`.

### 3.3 Implementation steps

1. **`evaluation/evaluator.py` — add a trace-target eval path.** `_create_session_run`/`_poll_scores` are hardwired to `--target session`. Add a trace-target mode that runs `pandaprobe --format json evals runs batch --target trace --trace-ids <ids> --metrics <set>` and polls `evals runs scores <run_id> --target trace` (or `evals scores get <trace_id> --target trace`). Keep the retry/backoff and polling machinery. Make the target and metric set parameters.
2. **`evaluation/metrics.py` — trace-level metrics + per-trace scores.** Add the trace metrics to the `Metric` enum with `target == "trace"`. `MetricScore` already carries `value/threshold/reason/metadata`; ensure a per-trace score flows through `breached`/`alerting`/`_signatures` unchanged (the breach logic is already metric-generic — keep it that way).
3. **`hook/core.py` — three-tier orchestration + trajectory gate.**
   - Tier 1 runs every admitted trace (task_completion + coherence). Feed values into the trajectory gate (reuse `_apply_trends`, repointed).
   - On Tier-1 breach → run Tier 2 on the **last trace** of the session (fetch the current last trace id). On Tier-2 breach → optionally run Tier 3 on the last trace.
   - Build the notice from the **triggering metric(s)** with their `value` + `reason`. Ensure the reason reaches the notice/agent (see 3.4).
4. **`config.py`** — add tier metric sets, per-metric thresholds, cadence `W`, target, gains/drops, and a Tier-3 on/off flag (Section 7). Default session-composite trigger **off**; keep it available behind a flag for the ablation.
5. **Drop** `loop_detection`/`confidence` from the default tier sets (leave the enum entries if the platform still references them, but don't request them).

### 3.4 Route metric `reason` to the agent (needed for good rules)

The notice must carry the per-metric `reason` so the agent can write a specific rule. Extend `NoticeMetric` (`workspace/mailbox.py`) if needed and populate it in `_build_notice` from the triggering `MetricScore.reason` / `per_trace_signals`. **But do not inject this into the system prompt** — it lives in the mailbox, which the agent pulls via a tool (Section 4 / 5).

---

## 4. PRIMARY CHANGE 2 — Agent-owned SKILL-pattern rules workspace (`rules/` subtree)

**Goal:** Split the single `harness_rules.md` into (a) a **clean, lightweight skill root** with only high-level instructions + a directory of references, and (b) a new **`rules/` subtree** that holds all rule content and the full candidate/active/retired lifecycle. The agent owns and curates this workspace via tools; nothing is force-injected.

### 4.1 Target layout

```
<harness_root>/
  harness_rules.md            # SKILL ROOT — instructions, tool list, protocol, and a
                              #   "References" section (regenerated) listing the rule
                              #   files that currently exist. NO rule content here.
  rules/
    global.md                 # Tier-1 GLOBAL rules — always in play.
    scoped.md                 # Tier-2/3 SCOPED rules — the general CATCH-ALL default.
    <topic>.md                # OPTIONAL, AGENT-CREATED on demand when a topic
                              #   accumulates enough related rules to be worth splitting
                              #   out (e.g. rules/payments.md, rules/planning.md).
  rules.jsonl                 # structured store (now includes a `scope` field)
  mailbox/ , traces/ , state/ , evalset/ , journal.jsonl   # unchanged
```

- **Root `harness_rules.md`** stays small and stable: the self-heal protocol, the tool list, and a References section (like a skill's front-matter) regenerated from whatever rule files exist. It is fine for the agent to read this as its entry point; it must NOT contain the live rule bodies.
- **`rules/*.md`** hold the actual rules, each with an active section and a clearly-labeled provisional (candidate) section. These are what the agent reads on demand and appends to.

### 4.2 Scope model — do NOT derive a semantic label; use a general catch-all + agent-created files

Deriving a meaningful scope label mechanically is fragile and subjective (and for single-tool agents like AppWorld, deriving scope from tool-span names collapses to one file). So use two axes:

- **Coarse axis — deterministic, tier-based (the only mechanical decision):** Tier-1 breach → `global`; Tier-2/3 breach → `scoped`. This coarse tag is the *only* "hint" the notice carries.
- **Fine axis — agent-owned, opt-in:** scoped rules go to the single general **`rules/scoped.md`** catch-all by default. The **agent may create its own specific file** (`rules/<topic>.md`) when a cluster of related rules accumulates and is worth splitting out. Emergent, agent-driven organization — no upfront taxonomy, no harness-side derivation.
- **`scope` is a free-form string**, not an enum: `harness_rule_add(text, scope=...)`. Default it from the notice's tier — Tier-1 → `"global"`, Tier-2/3 → `"scoped"`. The agent can pass any label to create/target a specific file. Scope is **tool-agnostic**: a planning/reasoning lesson is just `scope="planning"` (or stays in `scoped.md`); nothing depends on tool spans.
- Retrieval stays **per-rule** (over `rules.jsonl`), not per-file: `global` rules are always eligible; scoped rules rank by relevance. The `.md` files are only the agent-facing organizational view, so a large `scoped.md` does not degrade rule *selection*. One promotion margin for all scopes for now (keep it simple).
- **Anti-bloat:** the skill root instructs the agent — *"write scoped rules to `rules/scoped.md`; if several related rules accumulate on one topic, create `rules/<topic>.md`, move them there, and it will be added to the References."* Splitting is agent-driven and happens only when worth it.

### 4.3 Implementation steps

1. **`filesystem/layout.py`** — add paths for the `rules/` subtree: the root skill file, `rules/global.md`, `rules/scoped.md`, and a resolver for an arbitrary `rules/<scope>.md`. Keep `rules.jsonl` as the structured store path.
2. **`workspace/rules.py`** —
   - Add a **free-string `scope` field** to the rule record (default `"global"`/`"scoped"`; any agent-supplied label allowed), persisted in `rules.jsonl`.
   - **Split rendering**: instead of `sync_markdown` writing the whole thing into `harness_rules.md`, render (a) the **skill root** (instructions + tool list + a generated **References** section listing the rule files that exist) and (b) **one `.md` per distinct `scope` value** under `rules/` (`global.md`, `scoped.md`, plus any agent-created `<topic>.md`), each with active + provisional sections.
   - Update `add`/`retire` to route a rule to `rules/<scope>.md` and re-render only that file + the root References listing. A new `scope` value simply creates a new file. Candidate/active/retired transitions live entirely in the subtree.
   - Keep/extend `relevant(query,k)` (per-rule) so `global` is always eligible and scoped rules rank by relevance to the current task.
3. **`agent_tools/toolset.py`** — the agent still writes via a tool; add a **free-string `scope`** parameter to `harness_rule_add` (default filled from the notice's tier: `global` for Tier-1, `scoped` for Tier-2/3; any label creates/targets a file), and ensure the read tools can fetch a specific rule file / the references. Keep the toolset **agent-driven and pull-based** — do not remove the agent's ability to read its mailbox, notices, rules, and references. (You may prune genuinely redundant tools, but the agent must retain: read mailbox/notices, read rules & references, add rule, retire rule, ack.)
4. **`hook/context.py`** — the startup/system context becomes the **lightweight skill root only** (protocol + tools + references directory). It must **not** dump the live rule bodies. The agent pulls rule content via tools/references.
5. **Migration note:** because rule content moves out of `harness_rules.md`, update any code/tests that assumed rules render into that file. Ensure a fresh `harness_root` initializes the `rules/` subtree.

---

## 5. PRIMARY CHANGE 3 — Per-turn agent-await barrier (in-session self-healing)

**Goal:** A first-class synchronous barrier so the agent pauses **every turn** while the harness finishes evaluating that turn's trace(s), posts any notice, and gives the agent the chance (per the skill protocol) to pull its mailbox and write/read rules — so a rule learned mid-session helps the **rest of the same session**.

**Why every turn (confirmed):** Tier 1 runs on every trace and can breach on any trace; evals are async. If the barrier only fired at gate-open points, the agent could run past the breach turn before the (slower) eval detects it. A per-turn barrier prevents the agent from outrunning the harness.

### 5.1 Implementation steps

1. **`hook/core.py` / `harness.py`** — add a first-class **synchronous "await self-heal" method** (e.g., `await harness.settle(session_id)` or an awaitable turn scope) that blocks until: the turn's eval task completes, `_handle_report` runs, any notice is posted, and validation observation is recorded. Reuse the existing `refresh`/`drain_validation` primitives internally, but make this a clean blocking call with a **generous, configurable timeout** (not the short `drain_timeout_s`).
2. The barrier is the natural point where the agent, following the skill protocol, **pulls its mailbox and processes notices** (agent-driven, not injected).
3. Expose it so a host loop calls it once per turn when the harness is enabled. Update the `benchmarks/` glue (`benchmarks/src/pandabench/runners/base.py` `_run_trial`) to call the barrier every turn in the harness arm (replacing the current bounded `refresh` call).
4. Accept the latency cost (longer wall-clock per task) as the deliberate trade for in-session healing. Make the barrier timeout a config knob.

---

## 6. SUPPORTING CHANGE A — Optional outcome-verifier hook (Layer 1)

**Goal:** Let a developer who knows "success" (a golden dataset / benchmark evaluator / business rule) plug in a callback that becomes a **gold trigger and gold promotion criterion**. Off by default; the trace-tier trigger is the universal default.

### 6.1 Implementation steps

1. **`harness.py`** — add a `verifier` parameter to `Harness.create(...)`, injected the same way `ReplayFn` is (DI through to the hook). Signature: a callable `verify(session_id, end_state|trace) -> float|bool` (score in [0,1], higher = better, or pass/fail).
2. **`hook/core.py`** — in `_run_eval` (or just before `_handle_report`), when a verifier is present, run it and **emit a synthetic `MetricScore`** (e.g. `metric="outcome_correct", value=<score>, threshold=<τ>`). Because everything downstream keys off `EvalReport.scores`, this automatically produces a breach → notice → eval-case capture → validation observation with no further wiring.
3. When present, the verifier is authoritative for the **session outcome** and for **promotion** (see Section 8).
4. For real-world verification, wire AppWorld `/evaluate` into this hook from the `benchmarks/` side.

---

## 7. SUPPORTING CHANGE B — Config minimalism

**Goal:** Adopting the harness should need almost no configuration. The primary knob is the **cadence / trend sensitivity `W`**; everything else has sensible defaults.

### 7.1 Implementation steps

1. **`config.py`** — add, with defaults:
   - `trace_metrics_tier1` (default `["task_completion","coherence"]`), `trace_metrics_tier2` (`["tool_correctness","argument_correctness"]`), `trace_metrics_tier3` (`["plan_adherence","plan_quality","step_efficiency"]`), `enable_tier3` (default off/opt-in).
   - Trajectory gate: `gate_window` (`W`, default 5), `gate_target` (0.5), `gate_gain` (`δ_gain`), `gate_drop` (`δ_drop`).
   - `trigger_mode` (`"trace"` default | `"session"` legacy) so the old behavior remains available for the ablation.
   - `barrier_timeout_s` (generous; for the await barrier).
   - Per-metric `thresholds` (already exists) — provide calibrated defaults.
2. Bind the **main knobs** (`gate_window`, `trigger_mode`, `barrier_timeout_s`, `enable_tier3`) via `from_env` and document them. Keep the rest defaulted.
3. Keep the public `Harness.create(...)` surface minimal — a developer should be able to enable v2 with defaults + optionally set `W` and/or a `verifier`.

---

## 8. SUPPORTING CHANGE C — Validation/promotion re-pointed to trace metrics

**Goal:** Promotion must be gated on a signal that discriminates. Today `ReplayValidator` compares the degenerate **session** metrics; re-point it to the **triggering trace metric** (or the verifier).

### 8.1 Implementation steps

1. **`validation/validator.py`** — the replay re-scores the new session; change the baseline/target comparison to use the **triggering trace metric** (e.g., `task_completion` / `tool_correctness`) captured at eval-case time, or the **verifier** score when present. Promote iff the targeted metric improves by ≥ `rule_promote_margin` on failure cases and no case regresses by ≥ `rule_regress_margin`.
2. **`hook/core.py`** — `_baseline_from_dump` (or wherever `baseline_scores` are captured) must store the **trace-level triggering metric** value at capture time, not the session composite.
3. **`validation/regression.py`** — mirror the same metric change in the regression run and the operator CLI.
4. Keep a single promotion margin across scopes for now.

---

## 9. Verification (a change is not done until verified)

Verify **each** change both with unit tests and with a real end-to-end run. Do not rely on type-checking alone.

### 9.1 Unit tests (`tests/`)
- Keep the existing suite **green** throughout.
- Add tests for: the trace-target evaluator path (mock the CLI client); the trajectory gate (stall/regression/reset-on-gain sequences, including "healthy climb never breaches"); the three-tier escalation (Tier 2 only on Tier-1 breach; Tier 3 only on Tier-2 breach; last-trace-only); the `rules/` subtree rendering (root stays clean; global vs scoped files; candidate/active/retired transitions land in the subtree); the await barrier (blocks until settle); the verifier hook (synthetic score → breach → notice → capture); validation re-pointed to the trace metric.

### 9.2 Real end-to-end run (via `benchmarks/`)
Use `benchmarks/RUNNING.md`. Run a small AppWorld harness-arm run (e.g. `LIMIT=5`) with a real model and confirm, by inspecting the run's `harness_root/` and the PandaProbe project:
1. **Trace-level evals fire** (Tier 1 every trace; Tier 2 on the last trace only when Tier 1 breaches; Tier 3 only on Tier-2 breach). Confirm via the CLI that trace scores exist and via the journal that the tiers escalated as designed.
2. **The trajectory gate behaves** — a healthy climbing session does not breach; a stalled/regressing one does.
3. **Notices carry the metric `reason`**, and the agent writes rules **into the `rules/` subtree** (global vs scoped), while **`harness_rules.md` stays clean** (instructions + references only).
4. **The per-turn barrier blocks** — confirm the agent does not advance past a turn before its eval + notice processing completes, and that a rule written mid-session is **used later in the same session**.
5. **Validation/promotion** compares the trace metric (inspect the validation records / journal).
6. **Verifier hook** — run once with AppWorld `/evaluate` wired in and confirm it drives breaches/promotion.

### 9.3 CLI sanity (already available)
```bash
pandaprobe evals metrics --target trace
pandaprobe evals runs batch --target trace --trace-ids <id> --metrics task_completion,tool_correctness
pandaprobe evals runs scores <run_id> --target trace
```

---

## 10. Invariants & constraints (do not violate)

1. **Product philosophy — the agent owns the workspace.** The agent decides what to read, when, and what rules to write, via harness tools driven by its own intelligence. **Never force-inject rules or notices into the agent's system prompt.** The skill root may be the agent's entry point, but rule *content* is pulled on demand from the `rules/` subtree.
2. **Universal, not benchmark-specific.** The trace-tier trigger (esp. `task_completion`) must work for any agent with no domain code. Domain-specific success belongs only in the optional verifier hook.
3. **Keep the public API stable where possible.** `benchmarks/src/pandabench/harness_glue.py` and `runners/base.py` consume `Harness.create`, `on_turn_end`, `refresh`, `system_context`, `toolset`, `rules`, `build_harness_config`. Prefer additive/back-compatible changes; if a signature must change, update the `benchmarks/` consumer and `examples/` in the same PR.
4. **Do not re-introduce per-arm PandaProbe project overrides.** Traces and evals must use the configured `PANDAPROBE_PROJECT_NAME` uniformly.
5. **Legacy trigger stays available** behind `trigger_mode="session"` for the ablation; the default is `"trace"`.
6. **Cost/latency:** Tier-2/3 run on the **last trace only**; each LLM-judge metric reads the whole trace, so cost scales with metric count — keep the tier sets minimal by default.
7. **`loop_detection` and `confidence` are out** of the default tiers (`loop_detection` is not trace-runnable at all).

---

## 11. Suggested implementation order

1. Codebase review + confirm current behavior (Section 2).
2. Change 1 (trace-target evaluator + tiers + trajectory gate) — the foundation.
3. Change 2 (`rules/` subtree + skill root split + scope).
4. Change 3 (per-turn await barrier).
5. Supporting C (validation re-pointed) — depends on 1.
6. Supporting A (verifier hook) and B (config minimalism).
7. Full verification (Section 9) — unit + real run.
