# Examples

Runnable demonstrations of the **pull model** (the harness evaluates each
completed turn, posts diagnostic notices to a workspace **mailbox**, and the
agent *pulls* them with its own harness tools — nothing is ever injected into
the agent's message queue) and the v0.6 **closed loop** (a recorded rule is a
*candidate* until replay or forward-trial evidence promotes it; the eval-set
guards old wins; calibration measures the breach thresholds).

| Example                     | Extra required                            | Credentials needed                        |
| --------------------------- | ----------------------------------------- | ----------------------------------------- |
| `offline_self_heal.py`      | none (core install only)                  | none — fully offline                      |
| `closed_loop_self_heal.py`  | none (core install only)                  | none — fully offline                      |
| `calibration_demo.py`       | none (core install only)                  | none — fully offline                      |
| `langgraph_agent.py`        | `pandaprobe-harness[langgraph]`           | `pandaprobe` CLI auth + model API key     |
| `openai_agents_agent.py`    | `pandaprobe-harness[openai-agents]`       | `pandaprobe` CLI auth + `OPENAI_API_KEY`  |
| `claude_agent_sdk_agent.py` | `pandaprobe-harness[claude-agent-sdk]`    | `pandaprobe` CLI auth + `ANTHROPIC_API_KEY` |
| `crewai_agent.py`           | `pandaprobe-harness[crewai]`              | `pandaprobe` CLI auth + model API key     |

## Running

The offline demo needs nothing beyond the core package:

```bash
uv run python examples/offline_self_heal.py
# or: python examples/offline_self_heal.py  (with pandaprobe-harness installed)
```

The framework examples are documented **sketches**: install the extra shown in
the table, authenticate the `pandaprobe` CLI (`pandaprobe auth login`), export
your model provider's API key, then run the script. Each exits with an install
hint if the extra is missing.

```bash
pip install 'pandaprobe-harness[langgraph]'
python examples/langgraph_agent.py
```

## What the offline demo proves

`offline_self_heal.py` drives the complete acceptance flow against a scripted
in-process `CliClient` and a throwaway temp workspace — no network, no real
`pandaprobe` binary:

1. **trace** — the agent repeats an identical tool call (the seeded failure);
2. **eval** — the turn-end hook runs the session metrics
   (`agent_reliability` / `agent_consistency`) and they breach;
3. **notice** — a structured `DiagnosticNotice` (flagged trace + per-trace
   signal breakdown + dump) is posted to `mailbox/pending/`, and the
   `⚠ HARNESS` banner appears in `harness.system_context()`;
4. **pull** — the agent works the mailbox with its harness toolset:
   `harness_mailbox_list` → `harness_mailbox_read` → `harness_trace_inspect`
   → `harness_journal`;
5. **rule** — it records a permanent mitigation rule with provenance
   (`harness_rule_add`), which lands in `harness_rules.md` and re-enters the
   system context;
6. **ack** — it acknowledges the notice (`harness_mailbox_ack`), clearing the
   banner;
7. **recovery** — the corrected behaviour scores healthy, no new notice is
   posted, and the journal records the whole cycle in order:
   `health → notice → rule_add → ack → recovery`.

## What the closed-loop demo adds

`closed_loop_self_heal.py` extends the same scenario with the v0.6 loop,
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

`calibration_demo.py` shows the offline threshold check
(`pandaprobe-harness-calibrate` as a library call): precision/recall/F1 and a
threshold sweep with labels; distribution/histogram/agreement without.

Every framework example is the same loop with real turn detection wired by a
`Harness.for_<framework>()` factory; the self-diagnostic tools are delivered
either natively (`as_langchain_tools`, `as_openai_function_tools`,
`as_anthropic_tools`) or through the sandboxed companion CLI
(`pandaprobe-harness-agent` via the `RestrictedShellTool`, shown in the
CrewAI example).
