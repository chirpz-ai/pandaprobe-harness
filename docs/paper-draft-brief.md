# Task Brief — Draft an AAAI Research Paper on the Self-Healing Agent Harness

> **You are a research-writing agent with full read access to this repository.** Your job is to
> produce a complete, compilable **LaTeX first draft** of a technical research paper about the
> self-healing agent harness implemented in this repo, plus the benchmark study built to evaluate
> it. The paper targets **AAAI** — a top-tier, highly competitive AI venue — so the framing must be
> rigorously academic and math-forward, even though the artifact is a practical software package.
>
> Read this entire brief before writing. Then read the code (§3) before writing the technical
> section — your formalism must reflect what the code actually does, not a guess.

## 1. Hard constraints (read first — violations sink the draft)

1. **Anonymized submission.** AAAI reviewing is double-blind. **Never** write the names
   `PandaProbe`, `pandaprobe`, `pandaprobe-harness`, `Chirpz`, `Chirpz AI`, the PyPI package name,
   any `github.com/chirpz-ai/...` URL, `docs.pandaprobe.com`, `cli.pandaprobe.com`, or any author,
   company, or personal name. Scrub all of them. Do not cite the package's own README/docs as a
   source, and do not reveal the artifact's provenance anywhere (text, comments, bib, figure files).
   - Give the system a **neutral research name** and use it consistently. Pick one you like
     (suggestions below) — something crisp and memorable, not a company brand.
   - Genericize the observability substrate: the platform that scores sessions becomes
     **"a session-level agent-evaluation service"** (or "the evaluation oracle" `E`). The CLI seam
     becomes "a command-line interface to the evaluation service." The two metrics
     (`agent_reliability`, `agent_consistency`) may keep those *descriptive* names — they reveal
     nothing — but introduce them as generic session-quality metrics, not product features.
2. **No fabricated results.** The benchmark runs are still in progress. **Do not invent any
   numbers.** Every results table and every reported figure of merit must be a **placeholder**
   (e.g. `\num{--}` / `TODO` cells, or `\cellcolor` TBD markers) with a caption and column
   structure ready for the author to fill. State clearly in the results section that numbers are
   forthcoming. It is acceptable — encouraged — to write the *analysis scaffolding* (what each
   table shows, what a positive result would look like, how significance is computed) around empty
   tables.
3. **Compilable output.** The draft must compile with `pdflatex` (+ `bibtex`/`biber`) out of the
   box, with placeholder figures and a placeholder `.bib`. No missing-file errors.
4. **Math-forward, but correct.** Introduce formal notation and use it consistently. Every formula
   must be grounded in the actual implementation (§3, §5) — do not hand-wave a formalism the code
   doesn't support. When you state a constant (a margin, a threshold, a top-k), take it from the
   code and cite the file in a comment.
5. **Citations.** Use `\cite{key}` throughout. Where you know the real work, add a proper BibTeX
   entry. Where you don't, add a **clearly-marked placeholder** entry (`@misc{placeholder-reflexion,
   note={TODO: cite Reflexion (Shinn et al. 2023)}}`) so the author can complete it. Never leave a
   claim that needs a citation uncited — insert a placeholder instead.

## 2. What the system is (for your understanding — then re-derive from code)

The artifact is a **harness**: a runtime envelope that wraps an *arbitrary* LLM agent loop and
turns it into a **self-healing agent** — one that detects its own quality degradations, diagnoses
them, writes its own corrective operating rules, and *proves* those rules before trusting them, all
with no human in the loop.

The loop has four stages:

1. **Detect** — after each agent turn (or task), a background, non-blocking process asks a
   session-level evaluation service for the session's quality metrics (reliability, consistency ∈
   [0,1]); scores below a threshold are **breaches**.
2. **Notice** — breaches and declining trends are posted as structured **diagnostic notices** to a
   filesystem *mailbox* in a workspace. Crucially, **nothing is injected into the agent's
   conversation** — the agent *pulls* notices on its own initiative (a "pull model").
3. **Heal** — a standing protocol in the agent's system prompt tells it how to pull a notice,
   inspect its own flagged execution traces, and record a mitigation **rule** via a self-diagnostic
   toolset.
4. **Validate** — a new rule enters as a **candidate** (unproven). It is promoted to **active**
   only on evidence — either by **replaying** the captured failure scenario under the new rule, or
   by a **forward trial** over subsequent live sessions — and retired if it regresses. Active rules
   are retrieved into the prompt on future runs; a replayable **eval-set** guards past wins against
   regressions (a defense against catastrophic forgetting of learned behavior).

The intellectual core worth emphasizing for a research audience: this is **closed-loop, evidence-
gated, self-supervised policy adaptation at inference time** — the agent modifies its own operating
context, and each modification must pass a statistical/empirical test before it is trusted. Frame
the self-heal cycle as an **operator on the agent's policy-inducing context**, and rule validation
as **sequential hypothesis testing / paired empirical improvement testing**.

**Suggested anonymous names** (pick one, use consistently): `SHIELD` (Self-Healing Inference-time
Evidence-gated Loop for Degradation), `AEGIS`, `SENTINEL`, `SAGE` (Self-Adapting Gated Envelope),
or simply "the Harness." Choose one; define it in the intro; never deviate.

**Suggested titles** (pick or improve — informative, a little trendy, no product name):
- "Self-Healing Agents: Evidence-Gated Inference-Time Rule Learning for Reliable LLM Agents"
- "Closing the Loop on Agent Reliability: A Self-Healing Harness with Validated Self-Authored Rules"
- "Heal Thyself: Evidence-Gated Self-Correction for Long-Horizon LLM Agents"
- "From `pass@1` to `pass^k`: A Self-Healing Harness for Reliable Agentic Execution"

## 3. Where to read in the codebase (do this before the technical section)

The repo root is the harness package. **Do not modify any code** — you only read it. Two things to
study: the **package** (`src/`) for the mechanism and its math, and the **benchmark suite**
(`benchmarks/` + `docs/`) for the experimental design.

### 3.1 The harness package — `src/pandaprobe_harness/`

Read these to ground the technical section (the parenthetical is what to extract):

- `harness.py` — the facade / public API: `Harness.create(config, replay=...)`, `system_context(
  task_hint=...)`, `turn()`, `on_turn_end()`, `refresh()`, `run_regression()`,
  `validate_candidates()`, `drain_validation()`. (The system's control surface — your Figure 1
  should mirror this flow.)
- `workspace/rules.py` — the **rule lifecycle**: `RuleStatus ∈ {candidate, active, retired}`,
  `TrialState` (baseline vs. trial breach counts), `promote`/`retire`, `live()` vs `candidates()`,
  and the **retrieval scorer** + `render_markdown(query=...)`. (Rule algebra + retrieval formula.)
- `hook/context.py` — `compose_system_preamble(rules, mailbox, *, task_hint=...)` and the tokenizer
  + scoring for task-conditioned retrieval. (The context-composition operator `ρ`.)
- `validation/validator.py` — `ReplayValidator`, `ForwardTrialValidator`, `ValidationEngine`. This
  is the **heart of the math**: the forward-trial baseline/trial breach-rate comparison with the
  promote/regress margins, and the replay-based paired improvement-with-regression-guard test.
  (Extract the exact promote/retire predicates and default margins.)
- `validation/regression.py` — `run_regression`: replays the eval-set, classifies each case
  improved/unchanged/regressed. (The catastrophic-forgetting guard.)
- `workspace/evalset.py` — `EvalCase`, `EvalSet`, `ReplayFn = Callable[[EvalCase, str],
  Awaitable[str]]`. (How failures are captured for replay.)
- `evaluation/` — `evaluator.py`, `metrics.py`, `thresholds.py`, `trends.py`: how session metrics
  are read, what a breach is (`value < threshold`, default threshold 0.5), and how declining
  *trends* (not just point breaches) are detected. (Definitions of `m(s)`, breach indicator,
  trend signal.)
- `calibration.py` — the offline threshold-calibration tool (precision/recall/F1 + threshold
  sweep). (Relevant to the "does the breach signal correlate with task failure?" methodology.)
- `agent_tools/toolset.py` — the 14 self-diagnostic operations the agent uses to heal (read
  notices, inspect scores/traces, add/search/inspect rules, list/attach eval cases). (The healing
  action space.)
- `config.py` — every tunable knob and its default (`rule_trial_min_sessions=5`,
  `rule_promote_margin=0.05`, `rule_regress_margin=0.05`, `rules_context_topk=8`,
  `capture_eval_cases`, `rule_validation`, `rule_retrieval`, ...). (Your hyperparameter table.)
- `README.md`, `CHANGELOG.md`, `examples/` (`closed_loop_self_heal.py`, `offline_self_heal.py`,
  `calibration_demo.py`) — narrative + runnable end-to-end illustrations of the loop. (Good for
  building intuition and for a worked-example figure.) **Remember to genericize everything you
  lift from these — the README uses the real product name.**

### 3.2 The benchmark suite — `benchmarks/` and `docs/`

- `docs/benchmark-study-brief.md` — **the authoritative study design**: arms (baseline / harness /
  optional no-validation ablation), learning→frozen-eval phases, `pass@1` / `pass^k`, seeds,
  McNemar + bootstrap statistics, task subsets, the two required checkpoints (metric↔failure
  calibration; rule-promotion gate), and the token-overhead confound. **Your experimental-setup
  section should be a rigorous academic rendering of this file.**
- `benchmarks/README.md`, `benchmarks/RUNNING.md` — arms, outputs, run mechanics, result artifacts
  (`records.jsonl` schema, `headline.csv`, `harness_telemetry.csv`).
- `benchmarks/IMPLEMENTATION_NOTES.md` — pinned versions, and the real integration recipes +
  deviations. Extract: the three benchmarks are **AppWorld** (own loop, library/HTTP; TGC/SGC +
  no-collateral-damage tests), **Terminal-Bench 2.x via Harbor** (custom `BaseAgent`,
  `--agent-import-path`, per-attempt rewards), and **τ²-bench** (custom agent subclass against its
  orchestrator + LLM user simulator; native `pass^k`). Also note which integrations were
  end-to-end verified vs. adapter-only at build time (be honest about this in Limitations).
- `benchmarks/configs/study.yaml`, `benchmarks/configs/models.yaml` — the exact **model matrix**,
  task subsets, seeds, and k. Read these for the real configuration; **do not** copy any real
  provider/account identifiers into the paper — report models generically (e.g. "a frontier
  proprietary model and a smaller model from each of three providers") unless the specific model
  names are non-identifying and useful (model names like a Claude/GPT/Gemini tier are fine; API
  keys, project ids, org names are not).
- `benchmarks/src/pandabench/metrics.py` — the exact estimators: `pass@1`, `pass^k`, McNemar
  (exact + chi-square), paired delta, bootstrap CI. **Lift the estimator definitions from here** so
  the paper's statistics section matches the code.
- `benchmarks/src/pandabench/results.py` — the per-trial record schema (what is logged per
  task-trial, including harness telemetry: rules active/candidate, notices, breach flags). Useful
  for describing what the study measures.

Do not run the benchmarks or the package. This is a reading-and-writing task.

## 4. Paper structure (required)

Produce a single-column-to-AAAI-two-column paper with these sections. Use `\section`/`\subsection`.

1. **Abstract** — problem (agent reliability degrades over long horizons / across trials), the
   idea (an evidence-gated self-healing harness that learns validated operating rules at inference
   time), the evaluation (three agent benchmarks, `pass@1` vs `pass^k`, baseline vs harness), and a
   one-line placeholder for the headline result ("the harness improves `pass^k` by [TODO]").
2. **Introduction** — motivate agent *reliability* as distinct from *capability*; the `pass^k`
   reliability gap; why open-loop self-correction (prompt-only reflection) is insufficient; the
   central contribution: a closed, **evidence-gated** loop where self-authored rules must pass an
   empirical test before adoption, plus a regression guard against forgetting. End with an explicit
   contributions list (the harness formalism; the validation mechanism; the three-benchmark A/B
   protocol; findings — mark findings as forthcoming).
3. **Background and Related Work** — position against: self-refinement / reflection (Reflexion,
   Self-Refine, CRITIC), verification & self-consistency, agent memory (MemGPT, generative agents,
   experiential/skill memory à la Voyager), retrieval-augmented prompting, LLM-as-judge and
   process/outcome reward models, guardrails/observability for agents, constitutional/rule-based
   steering, and test-time adaptation/compute. Make the delta explicit: prior self-correction is
   mostly *unvalidated* and *within-episode*; this work is *cross-episode*, *evidence-gated*, and
   *guarded against regression*. Cite the three benchmarks here or in §5. Use `\cite` + placeholders
   liberally.
4. **The Self-Healing Harness** *(core section — heaviest math)* — see §5 for the required
   formalism and subsections.
5. **Experimental Setup** — a rigorous rendering of `docs/benchmark-study-brief.md`: the three
   benchmarks and why each was chosen (reliability metrics fit), the arms, the learning/frozen
   phases, the model matrix, seeds, `k`, the estimators (§3.2), the two methodological checkpoints,
   and the token-overhead confound. Placeholder table for the configuration matrix.
6. **Results and Discussion** — subsections with **placeholder tables/figures** and the analysis
   scaffolding around them (§7). Must include:
   - **6.x Main results** — headline table (benchmark × model: baseline vs harness `pass@1` and
     `pass^k`, deltas, bootstrap CIs, McNemar p) — placeholder cells.
   - **6.x Learning dynamics** — the arm-B learning curve (pass rate vs. task index in the learning
     phase) + rule-promotion counts over time — placeholder figure/table.
   - **6.x Harness telemetry / cost** — rules promoted/retired, notices, breach rates, and the
     token/latency overhead of the harness vs. baseline — placeholder table.
   - **6.x Ablations** — at minimum: (a) validation on vs. off (the `rule_validation=false`
     ablation — rules trusted immediately); (b) retrieval on vs. off; (c) replay-validation vs.
     forward-trial-only. Explain what each isolates and the expected direction. Placeholder tables.
   - **6.x Limitations and Threats to Validity** — modest statistical power at the chosen subset
     sizes; dependence on the evaluation service's breach signal correlating with task failure
     (reference the calibration checkpoint); the preamble/tool token-overhead confound;
     benchmark-integration coverage (which benchmarks were fully vs. partially verified); potential
     learning-phase leakage; nondeterminism (current models reject temperature control, so trial
     variance is intrinsic); generality beyond the tested domains.
7. **Conclusion and Future Work** — recap the contribution; future directions (richer rule
   representations, multi-metric/vector-valued validation, cross-agent rule sharing, theoretical
   regret/convergence analysis of the heal loop, human-in-the-loop variants, online threshold
   adaptation).
8. **References** — via `\bibliography`.

## 5. Required mathematical formalism (verify each piece against §3.1)

Build a consistent formal spine. The following is a **suggested, code-grounded** formalism —
**verify every predicate and constant against the actual source** and adjust if the code differs.

- **Agent, session, evaluator.** An agent is a policy `π` producing actions over a session `s`.
  A session-level evaluator `E` maps a session to a metric vector `m(s) ∈ [0,1]^d` (here `d=2`:
  reliability, consistency). For metric `j` with threshold `τ_j` (default `0.5`), the **breach
  indicator** is `b_j(s) = 𝟙[m_j(s) < τ_j]`. Optionally formalize a **trend** breach from
  `evaluation/trends.py` (a declining sequence over recent sessions), not just a point breach.
- **Context-composition operator (retrieval).** The agent's prompt-inducing context is
  `c(R, q) = c_0 ⊕ ρ(R, q)`, where `R` is the active rule set, `q` a task query, and `ρ` selects +
  renders rules. Formalize the **retrieval score** from `hook/context.py`/`rules.py`: with query
  token set `Q`, rule tag-tokens `T_r`, text-tokens `X_r`,
  `score(r, Q) = (2·|Q ∩ T_r| + |Q ∩ (X_r \ T_r)|) / max(1, |Q|)`;
  select all untagged ("global") rules plus the top-`k` tagged rules by score
  (`k = rules_context_topk`, default 8), ties broken by recency. (Confirm the exact weights/tokens
  in code.)
- **The self-heal operator.** Define the loop as a map `H` on the rule set:
  `R_{t+1} = H(R_t, 𝔅_t)` where `𝔅_t` are observed breaches at step `t`. Decompose `H` into
  *propose* (agent authors a candidate `r` from a diagnostic notice) and *gate* (validate `r`).
- **Rule lifecycle as evidence-gating.** A candidate `r` is admitted to `R` (active) only if it
  passes a validator. Formalize both, from `validation/validator.py`:
  - **Forward-trial test.** Let `p_0` be the baseline breach rate for the rule's metric-scoped
    signature family, estimated over a pre-rule window (`baseline_breached_sessions /
    baseline_sessions`, with the conservative convention `p_0 = 1` when the window is empty). After
    admitting `r`, observe `n = rule_trial_min_sessions` (default 5) subsequent sessions; let `p̂`
    be the trial breach rate. **Promote** iff `p̂ = 0` or `p̂ ≤ p_0 − δ_promote`
    (`δ_promote = rule_promote_margin`, default 0.05); otherwise **retire**. Present this as a
    one-sided test on a difference of breach proportions with a decision margin, and discuss its
    conservativeness.
  - **Replay test.** For captured failure cases `{(s_i, x_i)}` with baseline scores `m(s_i)`,
    re-run each under context `c(R ∪ {r}, ·)` (via the developer-supplied `ReplayFn`) to obtain
    replay scores `m'(s_i)`. **Promote** iff (improvement) `∃` failure case `i` with
    `m'_{target}(s_i) ≥ m_{target}(s_i) + δ_promote` **and** (regression guard, over failure *and*
    win cases, all metrics) `∀ i, ∀ j: m'_j(s_i) ≥ m_j(s_i) − δ_regress`
    (`δ_regress = rule_regress_margin`, default 0.05); otherwise **retire** (with an
    inconclusive→pending path after bounded attempts). Present as a **paired improvement test with
    a multi-metric non-regression constraint**.
- **Regression guard / anti-forgetting.** `run_regression` replays the whole eval-set corpus and
  classifies each case improved/unchanged/regressed by the same margins; formalize the guarantee it
  provides ("no admitted rule set regresses a previously-passing case beyond `δ_regress`") and its
  limits.
- **Evaluation metrics (from `benchmarks/src/pandabench/metrics.py`).** For task `t` with `k`
  i.i.d. trials and pass indicators `X_{t,i} ∈ {0,1}`:
  `pass@1 = (1/|T|) Σ_t X_{t,1}` (or the trial-averaged variant the code uses — check),
  `pass^k = (1/|T|) Σ_t Π_{i=1}^{k} X_{t,i}` (all `k` trials pass). State the central hypothesis:
  the harness raises `pass^k` (reliability) more than `pass@1` (peak capability) — i.e. it reduces
  trial-to-trial variance. Formalize the **paired significance test**: McNemar on discordant task
  pairs `(b=1,h=0)` vs `(b=0,h=1)`, and a **bootstrap CI** on `Δ = pass_H − pass_B`. Match the
  exact estimator forms in the code.

Define a **notation table** early in §4. Number all key equations. Keep symbols consistent
throughout (including §6).

## 6. Figures (at least two; placeholders)

Create a `figures/` directory. Reference figures with `\includegraphics` pointing at placeholder
files you generate, so the document compiles, and write a **detailed caption + an in-source comment
describing exactly what the author should draw**:

1. **Figure 1 — System overview / the self-heal loop.** A schematic of Detect → Notice → Heal →
   Validate (candidate→active→retired), the workspace (mailbox, rules, eval-set, journal), the
   pull-model boundary (the agent pulls; nothing is injected), and the evidence gate (replay /
   forward-trial). This is the paper's anchor diagram.
2. **Figure 2 — Results.** Either the `pass@1`-vs-`pass^k` baseline-vs-harness deltas across
   benchmarks, or the arm-B learning curve (pass rate + cumulative promoted rules vs. task index).
   Placeholder plot; caption describes axes and the expected shape.

For placeholders, prefer a self-contained TikZ box or a generated blank PDF/PNG (e.g. a framed
rectangle with the figure title) so `pdflatex` succeeds with no external assets. Do **not** embed
any identifying logo or URL in a figure.

## 7. Results-section scaffolding (write the prose, placeholder the numbers)

For each results subsection: write the paragraph that *interprets* the table (what comparison it
shows, what a positive/negative result implies for the hypothesis, how significance is read), and
leave the numeric cells as `TODO`/`--`. Example table skeletons to include (fill columns, empty
cells):

- Headline: rows = (benchmark × model); columns = `pass@1` (B), `pass@1` (H), `Δ`, `pass^k` (B),
  `pass^k` (H), `Δ`, bootstrap 95% CI, McNemar `p`.
- Telemetry: rows = run; columns = rules proposed / promoted / retired, notices, breach rate,
  mean rules-in-context, added tokens/turn, latency overhead.
- Ablations: rows = (full harness / no-validation / no-retrieval / replay-only / forward-trial-
  only); columns = `pass@1`, `pass^k`, Δ vs full.

Add a sentence in each that the numbers are pending the completing benchmark runs.

## 8. Output layout & LaTeX conventions

Put everything in a new top-level **`paper/`** directory (isolated, like `benchmarks/`; touch
nothing else in the repo):

```
paper/
├── main.tex              # the paper
├── references.bib        # real + clearly-marked placeholder entries
├── figures/
│   ├── overview.*        # placeholder (compilable)
│   └── results.*         # placeholder (compilable)
└── README.md             # how to build (pdflatex/bibtex), and a TODO list for the author
                          #   (fill tables, swap figures, complete placeholder citations, drop in
                          #    the official AAAI .sty, choose final title/system name)
```

- **Document class:** target the **AAAI author-kit** two-column format. If the official
  `aaai2026.sty`/`aaai.sty` is not present, either fetch/verify it or fall back to a standard
  two-column `article` setup that mirrors AAAI (10pt, two columns, letterpaper) and leave a clearly
  marked `% TODO: replace with official AAAI style file` note. The draft must compile *now*; the
  official style can be dropped in later. Follow AAAI rules you can honor: no page numbers in the
  final format is handled by their sty; **anonymize** (no author block — use "Anonymous
  Submission"); include an Abstract; references via BibTeX.
- Use `amsmath`, `amssymb`, `booktabs` (tables), `graphicx`, `hyperref` (with `hidelinks`),
  `algorithm`/`algorithmic` (an "Self-Heal Loop" and a "Rule Validation" pseudocode block are
  strongly encouraged), `siunitx` (optional, for `\num{--}` placeholder cells).
- Keep it a *first draft*: aim for a complete arc, roughly the length of a full AAAI paper (~7–8
  pages of body ex-references), with real prose in every section — not an outline. Where you must
  defer, use a visible `% TODO(author): ...` comment, not silent omission.
- Every claim of fact about the mechanism should be traceable to a file you read; add
  `% source: src/...` comments next to non-obvious formal statements so the author can verify.

## 9. Acceptance criteria

1. `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main` produces a PDF with
   no missing-file/undefined-reference errors (undefined-citation warnings from placeholder bib
   keys are acceptable and expected).
2. Full required structure (§4) present, with the core technical section carrying the §5 formalism
   (notation table, numbered equations, at least one pseudocode block), and Results containing
   ablations + limitations subsections with placeholder tables.
3. At least two figures referenced and compiling as placeholders (§6).
4. Zero occurrences of any banned identifying string (§1); a single consistent anonymous system
   name; the evaluation service genericized.
5. No fabricated quantitative results anywhere; every numeric result cell is a placeholder.
6. `references.bib` present; all `\cite` keys resolve to an entry (real or clearly-marked
   placeholder); a `paper/README.md` lists the author's remaining TODOs.
7. Nothing outside `paper/` is modified.

## 10. Suggested order of work

1. Read §3.1 and §3.2 source files; build your notation + a private map of formulas/constants.
2. Scaffold `paper/` (class, packages, section skeleton, placeholder figures + bib) and confirm it
   compiles empty.
3. Write §4 (core) with the formalism and pseudocode — this is where reviewers will look hardest.
4. Write §5 (setup) from the study brief; §3-related figures.
5. Write §6 scaffolding with placeholder tables; §2/§3 intro + related work with citations.
6. Abstract, conclusion, final anonymization sweep (grep for banned strings), compile clean, write
   `paper/README.md`.

Be rigorous, be honest about what is not yet measured, and make the math the spine of the paper.
