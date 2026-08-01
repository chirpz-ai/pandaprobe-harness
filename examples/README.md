# Examples

Runnable demonstrations of the **pull model** (the harness evaluates each
completed turn, posts diagnostic notices to a workspace **mailbox**, and the
agent *pulls* them with its own harness tools — nothing is ever injected into
the agent's message queue) and the v0.6 **closed loop** (a recorded rule is a
*candidate* until replay or forward-trial evidence promotes it; the eval-set
guards old wins; calibration measures the breach thresholds).

| Example                            | Extra required                         | Credentials needed                          |
| ---------------------------------- | -------------------------------------- | ------------------------------------------- |
| `demo/demo_self_heal.py`           | `pandaprobe` + `litellm`               | PandaProbe CLI/SDK + selected model         |
| `misc/offline_self_heal.py`        | none (core install only)               | none — fully offline                        |
| `misc/closed_loop_self_heal.py`    | none (core install only)               | none — fully offline                        |
| `misc/calibration_demo.py`         | none (core install only)               | none — fully offline                        |
| `misc/langgraph_agent.py`          | `pandaprobe-harness[langgraph]`        | `pandaprobe` CLI auth + model API key       |
| `misc/openai_agents_agent.py`      | `pandaprobe-harness[openai-agents]`    | `pandaprobe` CLI auth + `OPENAI_API_KEY`    |
| `misc/claude_agent_sdk_agent.py`   | `pandaprobe-harness[claude-agent-sdk]` | `pandaprobe` CLI auth + `ANTHROPIC_API_KEY` |
| `misc/crewai_agent.py`             | `pandaprobe-harness[crewai]`           | `pandaprobe` CLI auth + model API key       |

## Running

The offline demo needs nothing beyond the core package:

```bash
uv run python examples/misc/offline_self_heal.py
# or: python examples/misc/offline_self_heal.py  (with the package installed)
```

The framework examples are documented **sketches**: install the extra shown in
the table, authenticate the `pandaprobe` CLI (`pandaprobe auth login`), export
your model provider's API key, then run the script. Each exits with an install
hint if the extra is missing.

```bash
pip install 'pandaprobe-harness[langgraph]'
python examples/misc/langgraph_agent.py
```

## YC video: real two-run self-heal

`demo/demo_self_heal.py` is the short, real-network demo: run 1 repeats a failed
refund action until the trajectory gate fires; the model reads the resulting
PandaProbe notice and writes a candidate rule; the harness replays the captured
failure and promotes the rule only on measured improvement; run 2 reads the
proven global rule and completes the same task. The final terminal block uses
only captured scores and journal events.

It requires an authenticated `pandaprobe` CLI, SDK tracing credentials
(`PANDAPROBE_API_KEY` and `PANDAPROBE_PROJECT_NAME`), and credentials for the
selected LiteLLM model. The default is `openai/gpt-5.6-terra`; override it with
`DEMO_MODEL` or `--model`. GPT-5.6 calls automatically use
`reasoning_effort="none"` because function tools reject its default effort.

```bash
pip install pandaprobe-harness==0.7.0 pandaprobe litellm
export OPENAI_API_KEY=... PANDAPROBE_API_KEY=... PANDAPROBE_PROJECT_NAME=...
python examples/demo/demo_self_heal.py --reset
```

The default workspace is `examples/demo/workspace/`, regardless of the current
working directory. It is ignored by Git. Reuse `--reset` for each filmed take;
the script prints the auditable rule, journal, and score-history paths when it
finishes. The short-horizon gate window, calibrated completion/tool thresholds,
and faster polling are explicitly documented in the file as demo tuning.

## What the offline demo proves

`misc/offline_self_heal.py` drives the complete acceptance flow against a scripted
in-process `CliClient` and a throwaway temp workspace — no network, no real
`pandaprobe` binary:

1. **trace** — the agent repeats an identical tool call (the seeded failure);
2. **eval** — the turn-end hook scores every trace of the turn on Tier 1
   (`task_completion` / `coherence`); the trajectory is flat below target so the
   gate fires a STALL, and Tier 2 (`tool_correctness` /
   `argument_correctness`) then confirms the step-level breach on the last trace;
3. **notice** — a structured `DiagnosticNotice` (flagged trace + per-trace
   signal breakdown + dump) is posted to `mailbox/pending/`, and the
   `⚠ HARNESS` banner appears in `harness.system_context()`;
4. **pull** — the agent works the mailbox with its harness toolset:
   `harness_mailbox_list` → `harness_mailbox_read` → `harness_trace_inspect`
   → `harness_rules_read`;
5. **rule** — it records a permanent mitigation rule with provenance
   (`harness_rule_add`), which lands in `rules/<scope>.md` and is listed in the
   References section of `harness_rules.md`, so the agent can pull it back;
6. **ack** — it acknowledges the notice (`harness_mailbox_ack`), clearing the
   banner;
7. **recovery** — the corrected behaviour scores healthy, no new notice is
   posted, and the journal records the whole cycle in order:
   `health → notice → rule_add → ack → recovery`.

## What the closed-loop demo adds

`misc/closed_loop_self_heal.py` extends the same scenario with the v0.6 loop,
wiring a toy **replay function** (`Harness.create(..., replay=...)`):

1. the breach additionally captures the session as a **replayable eval case**
   (`capture_eval_cases=True`);
2. the agent's rule lands as a **candidate** (rendered under "Provisional
   rules (under evaluation)" — in force, but unproven);
3. the harness automatically **replays the captured failure** with the
   candidate in context; the replayed session scores healthy, so the rule is
   **promoted** (`journal: rule_promote`, `validator: replay`);
4. a protected `win` case is captured and `harness.run_regression()` replays
   the corpus against the current rules: the failure is `improved`, the win
   `unchanged`, the report `CLEAN`.

`misc/calibration_demo.py` shows the offline threshold check
(`pandaprobe-harness-calibrate` as a library call): precision/recall/F1 and a
threshold sweep with labels; distribution/histogram/agreement without.

Every framework example is the same loop with real turn detection wired by a
`Harness.for_<framework>()` factory; the self-diagnostic tools are delivered
either natively (`as_langchain_tools`, `as_openai_function_tools`,
`as_anthropic_tools`) or through the sandboxed companion CLI
(`pandaprobe-harness-agent` via the `RestrictedShellTool`, shown in the
CrewAI example).
