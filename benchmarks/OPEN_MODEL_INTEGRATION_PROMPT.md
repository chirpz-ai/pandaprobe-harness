# Task: integrate open-weight models (AWS Bedrock) into the PandaBench A/B suite

## 1. Mission

Add a set of open-weight models to this benchmark suite so they can be run as first-class
study models, exactly like the existing closed models. The models are served by **AWS
Bedrock**. Twelve candidates were requested; **nine qualify and three do not** — the
research and live verification is already done and is given to you in §5. Your job is to
integrate the nine, and to prove with small live runs that they actually work end to end
in the benchmark, not just in an isolated API call.

Read §5 before doing any model research of your own. Do not re-derive it; it was verified
against LiteLLM's model map, `aws bedrock list-foundation-models` in the target account,
and live tool-calling calls through this repo's own client.

---

## 2. What this repository is

Root: `pandaprobe-harness/`. Two things live here:

- `src/pandaprobe_harness/` — the **product**: a self-healing runtime-oversight harness for
  agents. It watches an agent's trace, detects degradation, writes candidate rules, and
  validates them. **Out of scope. Do not modify.** The benchmark consumes it as the PyPI
  pin `pandaprobe-harness==0.9.0`; keep that pin.
- `benchmarks/` — the **`pandabench` package**: an A/B study harness that measures whether
  the product improves agent reliability on third-party agent benchmarks. This is where you
  work.

### The experiment shape

Every run is one `(benchmark, dataset, model, arm, seed, k)` cell. There are two arms:

- **`baseline`** — the task agent runs alone.
- **`harness`** — the same agent with the harness live for the whole dataset in one
  continuous pass (no learning/eval split, no frozen ruleset).

A pair of runs that differ only in `arm` is a matched comparison. `k` trials per task
(default 4) give repeatability metrics. Results land in
`benchmarks/results/runs/<run_id>/` as `manifest.json` + `records.jsonl`, and
`make report` aggregates them.

### Three benchmarks, all tool-driven

| Benchmark | Agent's tool surface | Runner |
|---|---|---|
| AppWorld | a single `execute(code)` Python tool; env runs out-of-process | `runners/appworld.py` |
| τ²-bench | domain tools (airline / retail / telecom) + a simulated user | `runners/tau2.py` |
| Terminal-Bench 2.x | `bash` inside a Docker sandbox, driven via Harbor | `runners/terminal_bench.py` |

**Every one of them is driven by tool calls.** This matters enormously for model
selection — see §4.

---

## 3. How model integration works today

The design intent, which you should preserve: **adding or switching a model is a config
change, not a code change.**

### `benchmarks/configs/models.yaml` — the single source of routing truth

Each key is a study model identifier; the value resolves to a LiteLLM model string. Two
shapes exist today:

- **Single-backend** (Gemini, OpenAI): a `litellm_model` string.
- **Dual-backend** (Claude): a `backends` map plus a `default_backend`, switchable per run
  via `--backend`, the `CLAUDE_BACKEND` env var, or the default.

Per-model fields you will care about:

- `provider_family` — coarse label.
- `param_allowlist` — a gate on which **caller-supplied** sampler params LiteLLM may
  forward. Anything not listed is dropped. The study deliberately fixes Claude to
  `max_tokens` only.
- `default_params` — provider-**required** constants sent on every call (e.g.
  `reasoning_effort: none` for GPT-5.6, which otherwise 400s when function tools are
  combined with reasoning). Distinct from the allowlist: these are not the caller's choice.
  A caller value of the same name still wins.
- `price_per_mtok` — a `{input, output}` fallback used only when
  `litellm.completion_cost` has no price for the model.
- `roles:` at the bottom binds named roles: `user_simulator` (the τ² simulated user, held
  fixed across arms), `dry_run` (the mock pseudo-model), `smoke`.

The file's header comments document the routing constraints and why each exists. Read them
— they encode real operational history (e.g. Claude uses Bedrock *global inference
profiles* because the bare foundation-model IDs reject on-demand invocation).

### `benchmarks/src/pandabench/providers/models.py`

`load_registry()` parses the YAML into `ModelRegistry`; `resolve(key, backend=...)` returns
a frozen `ResolvedModel` carrying the resolved LiteLLM string, a coarse `provider` label,
the allowlist, the default params, and the price fallback. Two behaviours worth knowing:

- Backend precedence for dual-backend models: explicit arg → `CLAUDE_BACKEND` env → spec
  default. `CLAUDE_BACKEND` is consulted **only** for models that declare `backends`, so it
  cannot leak into single-backend models.
- Passing `--backend` to a **single-backend** model raises. Keep that in mind when deciding
  the shape of the new entries and when writing run commands.

### `benchmarks/src/pandabench/providers/litellm_client.py`

The **only** place LiteLLM is touched. One async `chat()`: OpenAI-format messages + tool
schemas in; a normalized assistant message, parsed tool calls, and usage/cost out. Every
model call in the study goes through it — both arms, all three benchmarks, the τ² user
simulator, and every replay — so tool semantics, accounting, retries and tracing stay
identical everywhere. `MockClient` is the network-free path used by `--dry-run` and unit
tests.

### Surfaces that select a model

- CLI: `--model <key>`, `--backend <name>`, `--arm`, `--benchmark`, `--seed`, `--k`,
  `--limit`, `--dataset`, `--max-turns`, `--run-id`, `--dry-run`.
- `make run BENCHMARK= ARM= MODEL= SEED= [BACKEND=] [K=] [LIMIT=] [DATASET=] [MAXTURNS=] [RUN_ID=]`
- `make preflight` — validates Docker, CLIs, credentials, and does a small ping per
  provider. Note its Bedrock check is currently *described* as being for "the
  Claude/bedrock backend"; after this change Bedrock is no longer Claude-only, so review
  whether that check and its wording still tell the truth.
- `make smoke` — 2 tasks × 1 trial × both arms × a cheap model across all benchmarks.
- `make check` — ruff + mypy (strict) + offline unit tests. Must stay green.

### Tests

`benchmarks/tests/test_providers.py` is the registry's test: resolution for single- and
dual-backend models, backend precedence, the raise-paths, roles, allowlist filtering, and
tool-argument parsing. Other test modules resolve models incidentally. All offline — no
network.

### Credentials (already configured, verified live)

`benchmarks/.env` carries `AWS_PROFILE_NAME=poweruser` and `AWS_REGION=us-west-2`. LiteLLM
reads **`AWS_PROFILE_NAME`** (not `AWS_PROFILE`). The profile is an SSO profile and
auto-refreshes, so it survives a multi-hour run; the preflight forces a credential resolve
so a dead session fails fast rather than mid-run. `AWS_BEARER_TOKEN_BEDROCK` must stay
**unset** — it silently overrides the profile and expires mid-run. Target account is
`446677925779`, region `us-west-2`, where all the models below are enabled.

---

## 4. The hard requirement: native structured tool calling

This is the gate that decides whether a model is usable at all.

`litellm_client.py` normalizes a response by reading the provider's **structured**
`tool_calls` field — `id`, `function.name`, `function.arguments` — and JSON-parsing the
arguments into a dict. Its docstring states the invariant plainly: *the loop must never
string-match model output.*

Consequence: a model that cannot emit native OpenAI-shape tool calls is **not integrable**
here. It will return prose, the loop will read zero tool calls, treat the turn as a final
answer, and every trial will fail for a reason that has nothing to do with the harness
being measured. That would silently corrupt the A/B comparison, which is the whole point of
the suite.

So the two stated requirements — *OpenAI-format compatible* and *LiteLLM-supported* —
reduce to one testable question: **does a live call through
`pandabench.providers.litellm_client` with a tool schema come back with a parsed
`ToolCall`?** That is the test that was run for §5, and it is the test that must hold for
anything you add.

---

## 5. Verified model research — use this, do not redo it

Method: LiteLLM's `model_cost` map (LiteLLM **1.91.1**, the pinned version) for
support/pricing metadata; `aws bedrock list-foundation-models` in account `446677925779` /
`us-west-2` for real availability and inference type; then a **live** tool-calling call per
model through this repo's own `LiteLLMClient` with a one-function schema, plus a multi-turn
`role:"tool"` round-trip on a vendor-diverse subset.

### 5a. QUALIFIED — integrate these nine

All nine are `ON_DEMAND` in `us-west-2` (except Llama 4 Scout, see the note), route via
LiteLLM's `bedrock_converse` path, and **returned a correctly parsed tool call in a live
call**. Prices are LiteLLM's, USD per 1M tokens.

| Requested name | LiteLLM / Bedrock model id | $ in | $ out | ctx in | max out |
|---|---|---|---|---|---|
| gpt-oss-120b | `bedrock/openai.gpt-oss-120b-1:0` | 0.15 | 0.60 | 128k | 128k |
| gpt-oss-20b | `bedrock/openai.gpt-oss-20b-1:0` | 0.07 | 0.30 | 128k | 128k |
| Qwen3 32B | `bedrock/qwen.qwen3-32b-v1:0` | 0.15 | 0.60 | 131k | 16k |
| Qwen3 Coder 30B A3B | `bedrock/qwen.qwen3-coder-30b-a3b-v1:0` | 0.15 | 0.60 | 262k | 131k |
| Qwen3 235B A22B 2507 | `bedrock/qwen.qwen3-235b-a22b-2507-v1:0` | 0.22 | 0.88 | 262k | 131k |
| Qwen3 Next 80B A3B | `bedrock/qwen.qwen3-next-80b-a3b` | 0.15 | 1.20 | 128k | 8k |
| NVIDIA Nemotron 3 Super 120B | `bedrock/nvidia.nemotron-super-3-120b` | 0.15 | 0.65 | 256k | 32k |
| Kimi K2.5 | `bedrock/moonshotai.kimi-k2.5` | 0.60 | 3.00 | 262k | 262k |
| Llama 4 Scout 17B | `bedrock/us.meta.llama4-scout-17b-instruct-v1:0` | 0.17 | 0.66 | 128k | 4k |

**Llama 4 Scout — the `us.` prefix is mandatory.** Bedrock lists it as
`INFERENCE_PROFILE` only, with no `ON_DEMAND`. The bare id
`bedrock/meta.llama4-scout-17b-instruct-v1:0` was tried and fails with *"Invocation of model
ID … with on-demand throughput isn't supported"*. The cross-region inference profile
`us.meta.llama4-scout-17b-instruct-v1:0` works. This is the same class of problem the
existing Claude entries solve with `global.anthropic.*` profiles, and models.yaml's header
already documents that pattern.

**Multi-turn verified.** A full round-trip — user → assistant tool call → `role:"tool"`
result → assistant — succeeded on `gpt-oss-120b`, `kimi-k2.5`, `llama4-scout` and
`nemotron-super-3-120b`. Bedrock's Converse message-ordering strictness is not a blocker
for these.

### 5b. DISQUALIFIED — do not integrate these three

| Requested name | Bedrock model id | Verified failure |
|---|---|---|
| Mixtral 8×7B | `mistral.mixtral-8x7b-instruct-v0:1` | LiteLLM raises `UnsupportedParamsError: bedrock does not support parameters: ['tools']`. No tool calling at all. |
| Gemma 3 27B | `google.gemma-3-27b-it` | Same `UnsupportedParamsError` on `['tools']`. LiteLLM metadata also reports no function-calling support. |
| Magistral Small 2509 | `mistral.magistral-small-2509` | Subtler and worse. It *accepts* a `tools` param without error, then returns the raw Mistral tool token **as text content**: `[TOOL_CALLS]execute{"command": "ls -l"}`, with `finish_reason='stop'` and zero structured tool calls. A system-prompt nudge did not change it, and `tool_choice` is also rejected by Bedrock for this model. |

The Magistral case deserves a note in whatever you write up: it fails *silently*. Nothing
errors, so a run would complete and report near-total task failure that looks like a model
or harness result rather than a plumbing defect. It would only become integrable behind a
text-format tool-call parser, which contradicts the client's explicit "never string-match
model output" invariant. Treat it as out of scope unless the repo owner decides otherwise.

### 5c. A requested route that does not exist

The request asked for **gpt-oss-120b and gpt-oss-20b via the OpenAI API**. That is not
available: OpenAI does not serve the open-weight gpt-oss models on its first-party API, and
LiteLLM has no `openai/gpt-oss-*` entry (checked across the whole model map — every
gpt-oss entry belongs to some other provider). Both models **are** verified working on
Bedrock at the ids above, which also keeps them consistent with the rest of this batch and
with the credentials already configured.

If a non-Bedrock path for gpt-oss is genuinely wanted later, LiteLLM does support several
OpenAI-compatible hosts for these weights — `groq/openai/gpt-oss-120b`,
`together_ai/openai/gpt-oss-120b`, `fireworks_ai/gpt-oss-120b`, `cerebras/gpt-oss-120b`,
all with function-calling support — but each needs its own API key and its own pricing
entry. Surface that as a decision for the repo owner; default to Bedrock.

---

## 6. What to implement

1. **Register the nine qualified models** so they are selectable by `--model` / `MODEL=`
   and behave like existing entries: resolvable, correctly priced, correctly labelled in
   run manifests and reports.
2. **Choose each model's param policy deliberately and say why.** Relevant fact: LiteLLM
   reports no `supports_temperature` metadata for any of the nine, and Bedrock rejected
   `tool_choice` for at least one Mistral model. The conservative precedent in this repo is
   Claude's minimal `max_tokens`-only allowlist. If any model *requires* a constant to work
   with function tools — the GPT-5.6 `reasoning_effort: none` situation — that belongs in
   `default_params`, not the allowlist. Several of these are reasoning/thinking-capable, so
   check whether reasoning output interferes with content or tool-call parsing.
3. **Respect the differing output ceilings.** Three of the nine cap below the others
   (Llama 4 Scout 4k, Qwen3 Next 8k, Qwen3 32B 16k) and the client's default max-token
   budget is 4096. Make sure nothing silently truncates an agent turn.
4. **Keep the config-not-code principle.** If something cannot be expressed in
   `models.yaml` and genuinely needs code, that is a signal worth reporting, not routing
   around.
5. **Update the operator-facing docs** that enumerate available models or describe provider
   routing — `benchmarks/README.md`, `benchmarks/RUNNING.md`, `IMPLEMENTATION_NOTES.md`,
   and models.yaml's own header — including the fact that Bedrock is no longer Claude-only.
   Re-check the preflight's Bedrock wording against that.
6. **Extend the offline tests** to cover the new entries at the level the existing registry
   tests work at. `make check` must pass: ruff clean, mypy strict clean, tests green.

---

## 7. Live verification — required, and the real deliverable

An isolated API probe is not sufficient evidence; §5 already has those and they do not
prove the benchmark works. For **every** model you integrate, demonstrate it inside the
actual pipeline:

- Start with the cheapest, smallest thing that exercises the whole path, then widen. Very
  small task and trial limits are the point — this is a plumbing check, not a study run.
- Cover **both arms**. The `harness` arm exercises the product's tool surface, per-turn
  evaluation barrier, and rule lifecycle, and can fail in ways `baseline` cannot.
- Cover **more than one benchmark**, because the tool surfaces differ substantially
  (AppWorld's single Python `execute` vs Terminal-Bench's sandboxed `bash` via Harbor vs
  τ²'s domain tools plus a simulated user). Terminal-Bench needs Docker and Harbor;
  AppWorld needs its isolated env.
- **Confirm real tool use in the records, not just absence of errors.** A run that
  "completes" with zero tool calls per trial is the exact silent-failure mode described in
  §4 and §5b. Check trials are recorded without `error`, that turns contain tool calls, and
  that token/cost accounting is populated and plausible.
- Confirm the `harness` arm actually did something — that it produced rules rather than
  degenerating into baseline-plus-overhead.
- Confirm `make report` ingests the new runs and attributes them to the right model.
- Confirm `--dry-run` still works fully offline for all three benchmarks and needs no
  credentials.

Report per model: what you ran, what came back, and whether it is study-ready. **If a model
qualifies at the API layer but misbehaves in the pipeline, say so plainly rather than
tuning until it passes** — a model that needs special handling to look functional is a
finding about that model, and the suite's value depends on that being stated.

Keep spend small and report it. The whole verification batch should cost very little: these
models are 4–35× cheaper per token than the study's current closed models, the cheapest
being gpt-oss-20b at $0.07/$0.30 per 1M tokens and the priciest Kimi K2.5 at $0.60/$3.00.

---

## 8. Constraints

- **Do not modify `src/pandaprobe_harness/`** at the repo root. Keep the
  `pandaprobe-harness==0.9.0` PyPI pin; do not add a local source override.
- **Do not touch existing results.** Everything under `benchmarks/results/runs/` is real
  experimental data backing a paper draft. Never edit, delete, or regenerate a record.
  Your verification runs must be new run directories, clearly disposable, and you should
  say which ones they are so they can be removed.
- **Do not change existing models' behaviour.** The current closed models and the
  `user_simulator` / `dry_run` / `smoke` roles are load-bearing for in-flight comparisons.
  Additive changes only.
- **Do not remove existing provider support or credentials.** In particular
  `ANTHROPIC_API_KEY` stays — it is the non-Bedrock path for Claude and someone depends on
  it. Same for the Vertex and OpenAI paths.
- **`--dry-run` must stay fully offline** for all three benchmarks.
- **No commits, tags, pushes, or version bumps.** Leave the work in the tree.
- Do not set `AWS_BEARER_TOKEN_BEDROCK`.
- Do not start long or expensive study runs. Small verification runs only.

## 9. Deliverables

1. The nine models integrated and selectable, with `make check` green.
2. Docs updated to match reality.
3. A short written report: per-model live-verification result and study-readiness; total
   spend; the disposable run directories you created; the gpt-oss routing decision from
   §5c; and anything you found that contradicts this brief — including anything in §5, which
   was verified on one day against one account and one LiteLLM version and could drift.
