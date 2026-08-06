# Examples

These examples use the new ownership boundary: the developer owns the task
agent and its domain loop, while PandaProbe evaluates task traces and runs the
separate managed repair agent. The task agent receives bounded guidance and,
optionally, four read-only rule tools.

| Example | Purpose | Credentials |
| --- | --- | --- |
| `misc/offline_self_heal.py` | failure → managed candidate → same-session next turn | none |
| `misc/closed_loop_self_heal.py` | candidate → replay validation → promotion | none |
| `misc/calibration_demo.py` | offline threshold calibration | none |
| `demo/demo_self_heal.py` | real SDK/CLI/model integration sketch | PandaProbe + provider |
| framework examples | framework-owned task loops and read-only rule delivery | framework/provider |

Run the offline examples from the repository root:

```bash
uv run python examples/misc/offline_self_heal.py
uv run python examples/misc/closed_loop_self_heal.py
uv run python examples/misc/calibration_demo.py
```

The two repair examples use a deterministic fake of the PandaProbe/LiteLLM
completion seam. The package still owns the prompt, conversation, tool loop,
notice lifecycle, and workspace writes; the fake replaces only the network
completion transport.

The required host sequence is visible in each example:

1. build and run the developer-owned task agent;
2. flush/export its task trace;
3. call `on_turn_end` with the task session and end-state descriptor;
4. await `settle(session_id)` before another task turn;
5. rebuild task context with `system_context(session_id, task_hint=...)`;
6. keep replay validation outside any non-reentrant task environment lock.

The managed repair model is configured through `HarnessConfig.repair_model` or
`HARNESS_REPAIR_MODEL`. OpenAI, Anthropic API, Bedrock Anthropic, and Vertex
Gemini identifiers all flow through PandaProbe's official LiteLLM wrapper.
