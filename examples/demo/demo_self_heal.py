"""Real managed-repair integration sketch; requires explicit credentials/models.

The developer still owns ``task_turn``. PandaProbe owns only instrumentation,
evaluation, managed repair, and learned-guidance delivery.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandaprobe
from pandaprobe.wrappers.litellm import wrap_litellm

from pandaprobe_harness import Harness, HarnessConfig


async def task_turn(
    model: str, system: str, user_input: str, harness: Harness
) -> str:
    """A tiny developer-owned task agent with optional read-only rule tools."""

    litellm = wrap_litellm()
    tools = [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.input_schema,
            },
        }
        for spec in harness.task_tools.specs()
    ]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_input},
    ]
    for _ in range(4):
        response = await litellm.acompletion(model=model, messages=messages, tools=tools)
        message = response.choices[0].message
        calls = message.tool_calls or []
        if not calls:
            return str(message.content or "")
        messages.append(message.model_dump(exclude_none=True))
        for call in calls:
            arguments = json.loads(call.function.arguments or "{}")
            result = await harness.task_tools.call(call.function.name, arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "name": call.function.name,
                    "content": json.dumps(result),
                }
            )
    raise RuntimeError("task-agent turn limit reached")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--workspace", type=Path, default=Path("examples/demo/workspace"))
    args = parser.parse_args()

    task_model = os.environ.get("DEMO_TASK_MODEL")
    repair_model = os.environ.get("HARNESS_REPAIR_MODEL")
    if not task_model or not repair_model:
        raise SystemExit("Set DEMO_TASK_MODEL and HARNESS_REPAIR_MODEL explicitly.")

    session_id = f"demo-task-{uuid4()}"
    harness = Harness.create(
        HarnessConfig(
            harness_root=args.workspace.resolve(),
            repair_model=repair_model,
            trace_repair_agent=True,
            capture_eval_cases=True,
        )
    )
    context = harness.system_context(session_id, task_hint=args.prompt)
    with pandaprobe.session(session_id):
        answer = await task_turn(task_model, context, args.prompt, harness)
    pandaprobe.flush()

    harness.on_turn_end(
        {
            "session_id": session_id,
            "turn_index": 1,
            "end_state": {"prompt": args.prompt, "answer": answer},
        }
    )
    settlement = await harness.settle(session_id)
    print(answer)
    if settlement.repair is not None:
        print(json.dumps(settlement.repair.to_json(), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
