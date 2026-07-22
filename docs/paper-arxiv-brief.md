# Task Brief — Draft the arXiv Technical Paper on PandaProbe Harness

> **You are a technical-writing agent with full read access to this repository.** Your job is to
> produce a complete, compilable **LaTeX draft** of a technical paper presenting **PandaProbe
> Harness** — an open-source self-healing envelope for LLM agents — to the technical / open-source
> community, together with the benchmark study built to evaluate it.
>
> This is an **arXiv preprint**, not an anonymous conference submission. It is **branded and
> attributed**: name the project, the company, and the authors. The goal is **technical branding +
> clear communication of a novel idea to engineers**, not blind-review theory. Favor systems
> clarity, concrete API/usage, architecture, and reproducible benchmarks over heavy formal proofs.
>
> Read this entire brief first. Then read the code (§3) before writing the mechanism section — your
> description must reflect what the code actually does.

## 0. This is the *branded twin* of the anonymized paper

A sibling brief (`docs/paper-draft-brief.md`) produces an **anonymized, math-heavy AAAI** version
of this same work. **This brief is different on purpose** — do not just copy that framing:

| Dimension | AAAI brief (the other one) | **This arXiv brief** |
|---|---|---|
| Names | Fully anonymized; product/company/author scrubbed | **Named**: "PandaProbe Harness", "PandaProbe", "Chirpz AI", authors |
| Audience | Blind AI-research reviewers | Engineers, agent builders, OSS community |
| Emphasis | Formalism, hypothesis-testing framing, proofs-of-rigor | **Architecture, API, design decisions, usage, reproducibility** |
| Math | As much as possible, the spine of the paper | **Enough to be precise**, not the spine — define the key formulas, skip the theory-paper apparatus |
| Tone | Conservative academic | Confident, clear, practitioner-facing (still credible and honest) |
| Artifacts | Placeholder tables only | Placeholder result tables **+** real code snippets, install/run commands, links |

If you have access to `docs/paper-draft-brief.md`, skim it to avoid duplicating prose, but write
this paper fresh in the branded, technical register.

## 1. Branding, attribution, and honesty (read first)

1. **Name everything openly.** Use **"PandaProbe Harness"** as the system name throughout (you may
   introduce a short form like "the Harness" after first use). The observability/evaluation
   platform it depends on is **PandaProbe** — name it, and describe the harness as a companion to
   the PandaProbe platform/SDK/CLI. Company: **Chirpz AI**.
2. **Byline / attribution.** Author line: **`Sina Tayebati, Chirpz AI`**.
   `% TODO(author): confirm exact author name spelling, add any co-authors, ORCID, and contact
   email before posting.` Do **not** invent co-authors or credentials — if unsure, leave the
   placeholder comment. Affiliation: **Chirpz AI**. It is fine to include the project links
   (PyPI page for `pandaprobe-harness`, the docs site `https://docs.pandaprobe.com/harness/...`,
   and the GitHub repository) — verify exact URLs from `README.md`; do not fabricate a URL.
3. **This is a preprint, so brand confidently — but never fabricate.** No invented benchmark
   numbers (see §5), no overclaiming beyond what the code and the (pending) results support. It is
   a strength, not a weakness, to be precise about what is implemented vs. measured; engineers trust
   honest technical writing. Where the benchmark integrations differ in verification maturity, say
   so plainly (see `benchmarks/IMPLEMENTATION_NOTES.md`).
4. **arXiv target: `cs.AI` primary.** (Reasonable cross-lists to mention in the README/TODO:
   `cs.SE`, `cs.MA`, `cs.LG`.) Match arXiv/preprint conventions: title, named authors +
   affiliation, abstract, no blind-review anonymization.

## 2. What PandaProbe Harness is (understand, then re-derive from code)

**PandaProbe Harness** is a Python package (zero runtime dependencies in its core) that wraps an
*arbitrary* LLM agent loop and makes it **self-healing**: the agent detects its own quality
degradations, diagnoses them, writes its own corrective operating rules, and **proves** those rules
before trusting them — fully automatically, no human in the healing loop. It is a companion to the
**PandaProbe** observability platform, which scores agent sessions on quality metrics; the harness
reaches the platform exclusively through the `pandaprobe` **CLI** (never a private API), which keeps
the core dependency-free and the trust boundary clean.

The loop, in four stages (this is your Figure 1 and the mechanism section's backbone):

1. **Evaluate / Detect** — after each turn (or task), a **detached, non-blocking** background task
   asks PandaProbe for the session's metrics (`agent_reliability`, `agent_consistency` ∈ [0,1]).
   Scores under a threshold (default 0.5) are **breaches**; declining **trends** are also caught.
   This never blocks or raises into the host agent loop — a first-class design constraint.
2. **Notice (pull model)** — breaches/trends post a structured **diagnostic notice** to a
   filesystem **mailbox** inside a workspace (`HARNESS_ROOT`). **Nothing is injected into the
   agent's conversation.** The agent *pulls* notices on its own initiative — a deliberate design
   choice that keeps the harness non-intrusive and framework-agnostic.
3. **Heal** — a standing protocol in the agent's system prompt (via `harness.system_context()`)
   tells the agent how to pull a notice, inspect its own flagged execution traces, and record a
   mitigation **rule** using a **14-operation self-diagnostic toolset** exposed to the agent.
4. **Validate (evidence gate)** — a new rule enters as a **candidate** (unproven, but still shown
   to the agent as "provisional"). It is promoted to **active** only on evidence — by **replaying**
   the captured failure under the new rule, or by a **forward trial** over subsequent live sessions
   — and **retired** if it regresses. Active rules are **retrieved** (task-conditioned, top-k) into
   the prompt on future runs; a replayable **eval-set** + `run_regression()` guard past wins against
   regressions.

Framing to lead with for a technical audience: **PandaProbe Harness closes the self-heal loop** —
prior "self-correcting" agents trust their own reflections immediately and within a single episode;
the Harness makes self-authored rules **cross-episode, evidence-gated, and regression-guarded**, and
does it as a drop-in envelope around any loop (raw loops, LangGraph, LangChain, DeepAgents, CrewAI,
Claude Agent SDK, OpenAI Agents — via `Harness.for_*` adapters). Emphasize the engineering
properties that make it adoptable: zero core deps, never blocks the host loop, CLI-only platform
access, framework adapters, a filesystem workspace you can inspect, and offline/credential-free
examples.

## 3. Where to read in the codebase (do this before the mechanism section)

The repo root is the PandaProbe Harness package. **Do not modify any code** — read only. Study the
**package** (`src/`) for the mechanism and API, and the **benchmark suite** (`benchmarks/` +
`docs/`) for the evaluation.

### 3.1 The package — `src/pandaprobe_harness/`

- `harness.py` — the **public facade / API** you will show in code snippets: `Harness.create(
  config, replay=...)`, the `for_langgraph()/for_langchain()/for_deepagents()/for_crewai()/
  for_claude_agent_sdk()/for_openai_agents()` adapters, `system_context(task_hint=...)`, `turn()`,
  `on_turn_end()`, `refresh()`, `run_regression()`, `validate_candidates()`, `drain_validation()`,
  and the `toolset`/`mailbox`/`journal`/`rules`/`evalset` accessors.
- `workspace/rules.py` — the **rule lifecycle**: `RuleStatus ∈ {candidate, active, retired}`,
  `TrialState`, `promote`/`retire`, `live()` vs `candidates()`, the task-conditioned **retrieval
  scorer**, and `render_markdown()` (how rules become prompt text, incl. the "provisional rules"
  section for candidates).
- `hook/context.py` — `compose_system_preamble(rules, mailbox, *, task_hint=...)` and the retrieval
  tokenizer/score. (How the system-prompt preamble is built.)
- `validation/validator.py` — `ReplayValidator`, `ForwardTrialValidator`, `ValidationEngine`. The
  promote/retire logic and the default margins/min-sessions. (Describe the decision rules precisely;
  see §5 for the minimum math.)
- `validation/regression.py` — `run_regression`: eval-set replay + improved/unchanged/regressed
  classification (the regression guard / anti-forgetting mechanism, exposed as the
  `pandaprobe-harness-eval` CLI).
- `workspace/evalset.py`, `workspace/mailbox.py`, `workspace/journal.py`, `filesystem/layout.py` —
  the **workspace substrate**: eval cases + `ReplayFn`, the notice mailbox, the append-only journal,
  and the on-disk layout under `HARNESS_ROOT`. Good material for an "anatomy of the workspace"
  subsection and a directory-tree figure.
- `evaluation/` (`evaluator.py`, `metrics.py`, `thresholds.py`, `trends.py`) — how session metrics
  are pulled and scored, what a breach is (`value < threshold`), and how trends are detected.
- `agent_tools/toolset.py` (+ `native.py`, `companion.py`) — the **14 self-diagnostic tools** and
  the exporters `as_anthropic_tools()` / `as_openai_function_tools()` / `as_langchain_tools()`.
  (The agent's healing action space — worth a table.)
- `cli/` — the `pandaprobe` CLI seam (`SubprocessCliClient`): how the harness talks to the platform
  without a runtime dependency. (Emphasize this as a design decision.)
- `calibration.py` — the `pandaprobe-harness-calibrate` tool (threshold precision/recall/F1 sweep).
- `config.py` — every knob + default (`rule_trial_min_sessions=5`, `rule_promote_margin=0.05`,
  `rule_regress_margin=0.05`, `rules_context_topk=8`, `capture_eval_cases`, `rule_validation`,
  `rule_retrieval`, `HARNESS_ROOT`, ...) and the `HARNESS_*` env mirrors. (Your configuration
  table.)
- `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, and `examples/` (`closed_loop_self_heal.py`,
  `offline_self_heal.py`, `calibration_demo.py`, the per-framework `*_agent.py` demos) — the
  narrative, install/quickstart, and runnable end-to-end demos. **These are your source for the
  real API snippets, install commands, and the offline-demo callout.** (Unlike the AAAI brief, here
  you *keep* the names and links.)

### 3.2 The benchmark suite — `benchmarks/` and `docs/`

- `docs/benchmark-study-brief.md` — the full study design: arms (**baseline** vs **harness**, plus
  a no-validation ablation), learning→frozen-eval phases, `pass@1`/`pass^k`, seeds, McNemar +
  bootstrap, task subsets, the two methodological checkpoints, and the token-overhead confound.
- `benchmarks/README.md`, `benchmarks/RUNNING.md` — arms, the exact `make`/CLI run commands (great
  for a "reproduce it yourself" subsection), outputs, and result artifacts.
- `benchmarks/IMPLEMENTATION_NOTES.md` — pinned versions, integration recipes, deviations, and
  **verification status** (which benchmarks are end-to-end verified vs. adapter-implemented). Report
  this honestly in the paper.
- The three benchmarks: **AppWorld** (own loop; TGC/SGC + no-collateral-damage tests),
  **Terminal-Bench 2.x via Harbor** (custom `BaseAgent`, `--agent-import-path`), **τ²-bench**
  (custom agent + LLM user simulator; native `pass^k`). Explain why each was chosen (each stresses
  a different reliability facet).
- `benchmarks/configs/study.yaml`, `configs/models.yaml` — the model matrix, task subsets, seeds,
  `k`. Report model tiers/providers generically-but-named (Claude/GPT/Gemini tiers are fine to name;
  do **not** paste API keys, project ids, or account identifiers — those aren't branding, they're
  secrets).
- `benchmarks/src/pandabench/metrics.py` — the exact estimators (`pass@1`, `pass^k`, McNemar,
  bootstrap). `results.py` — the per-trial record schema incl. harness telemetry. `report.py` —
  what `make report` emits. (Ground the evaluation section here.)

Do not run anything. Reading + writing only.

## 4. Paper structure (required)

A named arXiv preprint (single- or two-column; see §8). Sections:

1. **Abstract** — the reliability problem, PandaProbe Harness as a self-healing envelope for any
   agent loop, its four-stage evidence-gated loop, the open-source availability, and the
   three-benchmark A/B evaluation. One-line placeholder for the headline result.
2. **Introduction** — motivate the gap between agent *capability* and *reliability* (the `pass^k`
   gap); why existing self-correction is unvalidated and within-episode; introduce PandaProbe
   Harness and its design principles (non-blocking, pull-model, zero-core-deps, CLI-only,
   framework-agnostic, evidence-gated). Clear contributions list: the self-heal architecture; the
   candidate→active evidence gate; the framework-agnostic integration surface; the open benchmark
   suite; (findings — mark forthcoming). Mention availability (PyPI/docs/GitHub) early.
3. **Background and Related Work** — practitioner-facing but credible: self-refine/reflection
   (Reflexion, Self-Refine, CRITIC), agent memory / experiential learning (MemGPT, Voyager,
   generative agents), verification / self-consistency / LLM-as-judge, guardrails & agent
   observability, RAG/context engineering, and the three benchmarks. Make the delta explicit:
   PandaProbe Harness is cross-episode, evidence-gated, regression-guarded, and drop-in. Use
   `\cite` with real entries where known, placeholders otherwise.
4. **System Design** *(core section — architecture-forward)* — see §5 for required subsections and
   the minimum formalism. This section should read like excellent systems/engineering writing:
   architecture diagram, the workspace anatomy, the pull model, the toolset, the adapters, the
   design decisions and their rationale, and real API/code snippets from `examples/`.
5. **Implementation & Usage** — install (`pip install pandaprobe-harness`), the quickstart snippet
   (from `README.md`/`examples/`), the `for_*` adapters, the operator CLIs
   (`pandaprobe-harness-eval`, `-calibrate`), configuration via `HARNESS_*`, and the offline
   credential-free examples. Keep snippets real and minimal.
6. **Benchmark Study** — a technical rendering of `docs/benchmark-study-brief.md`: the three
   benchmarks and why, the two arms, learning/frozen phases, model matrix, `k`, seeds, the exact
   metrics/estimators, and the "reproduce it yourself" commands from `benchmarks/README.md`.
7. **Results** — **placeholder tables/figures** with the analysis scaffolding around them (§7):
   - **Main results** — headline table (benchmark × model: baseline vs harness `pass@1`, `pass^k`,
     Δ, bootstrap CI, McNemar p) — placeholder cells.
   - **Learning dynamics** — arm-B learning curve + rules promoted over time — placeholder figure.
   - **Harness overhead & telemetry** — rules promoted/retired, notices, breach rate, token/latency
     overhead vs. baseline — placeholder table.
   - **Ablations** — validation on/off, retrieval on/off, replay-vs-forward-trial-only. Explain what
     each isolates. Placeholder tables.
   - **Limitations & honest caveats** — statistical power at the chosen subset sizes; dependence on
     the PandaProbe breach signal tracking task failure (reference the calibration checkpoint);
     token-overhead confound; benchmark-integration verification maturity; nondeterminism; scope of
     domains tested. Frame as engineering honesty, not weakness.
8. **Conclusion and Future Work** — recap; roadmap (richer rule representations, multi-metric
   validation, cross-agent rule sharing, more adapters/benchmarks, community contributions). A short
   call to try/contribute (with links) is appropriate for a branded preprint.
9. **References**.
10. *(Optional)* short **Availability / Reproducibility** statement near the end: version, license
    (MIT), PyPI/docs/repo links, and how to reproduce the study.

## 5. System Design section — required content + minimum math

Lead with **architecture and design decisions**, use **just enough formalism** to be precise. Do
not build the full theory-paper apparatus (that's the AAAI twin's job) — but the following must be
stated precisely and correctly, grounded in §3.1:

- **Metrics & breach.** Session metrics `m(s) ∈ [0,1]^2` (reliability, consistency); a breach on
  metric `j` is `m_j(s) < τ_j` (default `τ = 0.5`); note the trend-based signal too. Keep it a
  clear definition, not a theorem.
- **The self-heal loop** — a labeled architecture description (Detect→Notice→Heal→Validate), with
  the **pull model** and the **non-blocking** property called out as design decisions. An
  `algorithm` pseudocode block for the loop is encouraged.
- **Task-conditioned rule retrieval** — state the scoring rule from the code
  (`score(r,Q) = (2·|Q∩tag| + |Q∩text|)/max(1,|Q|)`, global rules always included, top-k tagged,
  `k = rules_context_topk`) as a concise formula + a sentence of intuition. Verify against the code.
- **The evidence gate** — the two validators, stated as decision rules (this is the novel core;
  give it real precision but keep it readable):
  - **Forward trial:** baseline breach rate `p_0` (with the conservative `p_0=1` empty-window
    convention) vs. trial breach rate `p̂` over `n = rule_trial_min_sessions` sessions; **promote**
    iff `p̂ = 0` or `p̂ ≤ p_0 − δ_promote`, else **retire** (`δ_promote = rule_promote_margin =
    0.05`).
  - **Replay:** re-run captured failure cases under the new rule via the `ReplayFn`; **promote** iff
    some failure case improves by ≥ `δ_promote` on the target metric **and** no case (failure or
    win) regresses by more than `δ_regress` on any metric; else **retire**. Present as
    improvement-with-regression-guard.
  - **Regression guard:** `run_regression` replays the whole eval-set and classifies each case;
    describe it as the anti-forgetting safety net.
- **Evaluation metrics** — define `pass@1` and `pass^k = (1/|T|)Σ_t Π_i X_{t,i}` and the central
  hypothesis (the Harness lifts `pass^k` — reliability — more than `pass@1`), plus the paired tests
  (McNemar, bootstrap CI on the delta). Match `benchmarks/src/pandabench/metrics.py`.
- **Workspace anatomy** — a subsection (and/or a directory-tree listing) of `HARNESS_ROOT`: mailbox,
  rules, eval-set, journal, traces, state. Engineers love seeing the on-disk shape they can inspect.
- **Design decisions** — a short subsection enumerating and justifying: zero core dependencies;
  never block/raise into the host loop; CLI-only platform access; pull model over injection;
  candidate-first (provisional) rules; framework adapters. This is prime technical-branding
  material — make the reasoning crisp.

Number the few key equations; add a small notation note. Add `% source: src/...` comments next to
non-obvious formal statements.

## 6. Figures (at least two; placeholders that compile)

Create `figures/`. Reference figures with `\includegraphics` to placeholder files you generate, and
write a detailed caption + an in-source comment describing what the author should draw:

1. **Figure 1 — Architecture / self-heal loop.** Detect→Notice→Heal→Validate, the workspace
   (mailbox/rules/eval-set/journal), the CLI seam to the PandaProbe platform, the pull-model
   boundary, and candidate→active→retired. The paper's anchor diagram. (You may also add a small
   **workspace directory-tree** figure.)
2. **Figure 2 — Results.** `pass@1` vs `pass^k`, baseline vs harness across benchmarks, or the arm-B
   learning curve. Placeholder plot with axes/expected-shape described in the caption.

Prefer self-contained TikZ or a generated blank PDF/PNG so `pdflatex` succeeds with no external
assets. (Unlike the AAAI paper, a discreet project/company name in a figure is fine — but keep
figures clean and legible, no marketing clutter.)

## 7. Results scaffolding (write prose, placeholder the numbers)

Same rule as the AAAI twin: **no fabricated numbers.** Write the interpreting prose for each table
(what it compares, what a good result looks like, how significance is read) and leave numeric cells
as `TODO`/`--`. Include the table skeletons: headline (benchmark×model → pass@1/pass^k/Δ/CI/p),
telemetry/overhead, and ablations (full / no-validation / no-retrieval / replay-only /
forward-trial-only). Add a sentence in each that numbers are pending completing runs. **Real code
snippets and real run commands are encouraged elsewhere; only the *measured results* are
placeholders.**

## 8. Output layout & LaTeX conventions

Put everything in a new top-level **`paper-arxiv/`** directory (isolated; touch nothing else in the
repo, and do not collide with the AAAI paper's `paper/` directory):

```
paper-arxiv/
├── main.tex              # the paper (named authors + affiliation; NOT anonymized)
├── references.bib        # real + clearly-marked placeholder entries
├── figures/
│   ├── architecture.*    # placeholder (compilable)
│   └── results.*         # placeholder (compilable)
└── README.md             # build steps (pdflatex/bibtex), arXiv-submission notes, and a TODO list
                          #   (fill tables, swap figures, complete citations, confirm byline,
                          #    set arXiv categories cs.AI [+ cs.SE/cs.MA/cs.LG cross-lists])
```

- **Document class:** a clean preprint style is appropriate — either the widely-used `arxiv`
  preprint template (a styled `article`) or a standard two-column `article`. Pick one that compiles
  with no external downloads; if you reference a template `.sty` that isn't present, fall back to
  plain `article` and leave a `% TODO` note. **Named** title block: title, `Sina Tayebati` +
  `Chirpz AI` affiliation (with the confirm-me `% TODO(author)` comment), and the project links.
- Packages: `amsmath`, `amssymb`, `booktabs`, `graphicx`, `hyperref`, `listings` or `minted`-style
  code formatting (use `listings` to avoid `-shell-escape`), `algorithm`/`algorithmic`,
  `xcolor`. Format code snippets with a real code environment (they are a selling point here).
- Length: a complete technical paper (~8–12 pages incl. code listings is fine for arXiv — no strict
  page limit). Real prose in every section; use visible `% TODO(author): ...` for anything deferred.
- Keep API snippets accurate to the current code — copy from `examples/`/`README.md` and trim; add
  `% source:` comments. Don't show an API that doesn't exist.

## 9. Acceptance criteria

1. `cd paper-arxiv && pdflatex main && bibtex main && pdflatex main && pdflatex main` produces a PDF
   with no missing-file/undefined-reference errors (placeholder-citation warnings are fine).
2. Full structure (§4) present; System Design carries the §5 content (architecture + the minimum
   formalism + at least one pseudocode block + a design-decisions subsection); Results has ablations
   + limitations with placeholder tables.
3. At least two compilable placeholder figures (§6).
4. **Branded and attributed**: "PandaProbe Harness", "PandaProbe", "Chirpz AI", and the
   `Sina Tayebati, Chirpz AI` byline (with the confirm-me TODO) all present; real project links
   verified from `README.md`; **no secrets** (API keys, project ids, account identifiers) anywhere.
5. No fabricated quantitative results; every measured-result cell is a placeholder. (Real code
   snippets, install/run commands, and config values are welcome and encouraged.)
6. `references.bib` present; all `\cite` keys resolve; `paper-arxiv/README.md` lists remaining
   author TODOs and arXiv-submission notes.
7. Nothing outside `paper-arxiv/` is modified.

## 10. Suggested order of work

1. Read §3.1/§3.2; collect the real API surface, config defaults, formulas, run commands, and links.
2. Scaffold `paper-arxiv/` (named title block, packages, section skeleton, placeholder figures +
   bib, `listings` setup) and confirm it compiles empty.
3. Write §4 System Design (architecture, workspace anatomy, evidence gate, design decisions) with
   real code snippets and the minimum formalism — the technical heart.
4. Write §5 Implementation & Usage (install/quickstart/adapters/CLIs) and §6 Benchmark Study +
   reproduce-it commands.
5. Write §7 Results scaffolding with placeholder tables; §2/§3 intro + related work with citations.
6. Abstract, conclusion + call-to-try, availability statement, compile clean, write
   `paper-arxiv/README.md`.

Write like a strong engineer explaining a genuinely novel system to peers: precise, honest about
what's measured vs. implemented, generous with real code and design rationale, and proud of the
idea without overclaiming.
