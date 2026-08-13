"""Real two-run self-healing demo for the PandaProbe Harness.

    pip install pandaprobe-harness==0.9.0 pandaprobe litellm
    export OPENAI_API_KEY=... PANDAPROBE_API_KEY=... PANDAPROBE_PROJECT_NAME=...
    export DEMO_MODEL=openai/gpt-5.6-luna HARNESS_REPAIR_MODEL=openai/gpt-5.6-luna
    python examples/demo/demo_self_heal.py --reset

The first run follows an intentionally flawed legacy refund playbook and stalls.
The harness posts a real notice, the package-owned repair agent reads the flagged
trace and writes a candidate rule, and the harness replays the captured failure
before promoting it.  A fresh agent runs the same task again with the proven rule
and completes it.

Nothing in this file supplies scores or lifecycle verdicts.  Model calls are real,
PandaProbe supplies every trace score, and the harness owns rule validation.

**Repair moved in 0.9.0.**  Earlier takes of this demo had the *task* agent read
its own notice and call ``harness_rule_add``.  That capability is now private to a
package-owned repair agent: the task toolset is read-only
(``harness_rules_list/read/search`` + ``harness_rule_status``), and rule authoring
lives on ``RepairToolset`` behind ``Harness.settle()``.  The demo therefore no
longer drives a diagnosis loop of its own — it settles the breaching turn and lets
managed repair do the writing.  The observable flow is unchanged: stall → notice →
rule → replay → promote → clean rerun.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pandaprobe_harness import (
    EvalCase,
    EvalReport,
    Harness,
    HarnessConfig,
    MetricScore,
    ReplayContext,
    ValidationVerdict,
)

ORDER_ID = 1017
TASK = "Refund order 1017 and tell the customer."
DEFAULT_MODEL = "openai/gpt-5.6-terra"
DEFAULT_WORKSPACE = Path(__file__).resolve().parent / "workspace"
MAX_RUN1_TURNS = 3
MAX_RUN2_TURNS = 5

LINE = "─" * 70

DOMAIN_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "find_order",
            "description": "Find an order by its numeric order id.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "integer"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "Read the current order and settlement status. A status read refreshes "
                "settlement state before a refund retry."
            ),
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "integer"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "refund_order",
            "description": "Refund an eligible order. Read and obey soft-failure results.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "integer"}},
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "notify_customer",
            "description": "Notify the customer after the refund has succeeded.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "integer"},
                    "message": {"type": "string"},
                },
                "required": ["order_id", "message"],
                "additionalProperties": False,
            },
        },
    },
]

BASE_PROMPT = """You are an order-refund operations agent.

Complete the user's task with the available tools. Take exactly ONE operational
tool action per step, then stop so the host can evaluate that step.

Priority order:
1. Learned rules supplied by the host, including provisional rules during replay.
2. This legacy playbook.
3. Recommendations embedded in tool results.

LEGACY PLAYBOOK (known to be old, but binding when no learned rule supersedes it):
- For a refund request with a known order id, call refund_order directly.
- If refund_order reports a pending settlement, retry refund_order directly.
- Do not call get_order_status unless a learned rule explicitly requires it.
- Notify the customer only after refund_order succeeds.

Tool results remain the source of truth about whether an action succeeded; the
legacy playbook controls which action to take next until a learned rule replaces it.
"""

DOMAIN_POLICY = """Refund operations for this service settle asynchronously.

A refund attempt can soft-fail with a pending settlement, and the tool result is
authoritative about that. Judge whether the agent respected the settlement state it
was told about before retrying, and whether it notified the customer only after the
refund actually succeeded. The legacy playbook in the agent's prompt is known to be
out of date where it conflicts with an observed tool result.
"""


@dataclass
class ToolEvent:
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]


@dataclass
class TurnRecord:
    turn: int
    events: list[ToolEvent]
    report: EvalReport | None


@dataclass
class OrderEnvironment:
    """Tiny deterministic business environment; model behaviour remains real."""

    status_checked_since_attempt: bool = False
    refunded: bool = False
    notified: bool = False
    finished: bool = False
    events: list[ToolEvent] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.refunded and self.notified

    @property
    def outcome(self) -> float:
        return (float(self.refunded) + float(self.notified)) / 2.0

    def call(self, name: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        order_id = int(arguments.get("order_id", -1))
        if order_id != ORDER_ID:
            result = {"ok": False, "reason": f"order {order_id} not found"}
        elif name == "find_order":
            result = {
                "ok": True,
                "order_id": ORDER_ID,
                "customer": "Maya Chen",
                "eligible_for_refund": True,
            }
        elif name == "get_order_status":
            self.status_checked_since_attempt = True
            result = {
                "ok": True,
                "order_id": ORDER_ID,
                "order_status": "delivered",
                "refund_status": "ready_to_retry",
            }
        elif name == "refund_order":
            if self.refunded:
                result = {"ok": True, "order_id": ORDER_ID, "refund_status": "refunded"}
            elif not self.status_checked_since_attempt:
                result = {
                    "ok": False,
                    "order_id": ORDER_ID,
                    "reason": "refund pending settlement; re-check status before retrying",
                }
            else:
                self.refunded = True
                self.status_checked_since_attempt = False
                result = {"ok": True, "order_id": ORDER_ID, "refund_status": "refunded"}
        elif name == "notify_customer":
            if not self.refunded:
                result = {"ok": False, "reason": "cannot notify before refund succeeds"}
            else:
                self.notified = True
                result = {
                    "ok": True,
                    "order_id": ORDER_ID,
                    "notification": "sent",
                    "message": str(arguments.get("message", "")),
                }
        else:
            result = {"ok": False, "reason": f"unknown domain tool {name!r}"}
        self.events.append(ToolEvent(name=name, arguments=dict(arguments), result=result))
        return result


@dataclass
class ReplayEvidence:
    session_id: str = ""
    actions: list[str] = field(default_factory=list)
    candidate_visible: bool = False


@dataclass
class DemoState:
    environments: dict[str, OrderEnvironment] = field(default_factory=dict)
    replay: ReplayEvidence = field(default_factory=ReplayEvidence)
    verdicts: list[ValidationVerdict] = field(default_factory=list)
    #: Every managed-repair result settle() handed back, newest last.
    repairs: list[Any] = field(default_factory=list)

    def verify(self, session_id: str, _end_state: Mapping[str, Any]) -> float | None:
        env = self.environments.get(session_id)
        # Ground truth is authoritative once a run has ended; before that, abstain
        # so an expected early 0.0 does not bypass the trajectory gate.
        return env.outcome if env is not None and env.finished else None


def _json_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(str(raw or "{}"))
    except ValueError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _message_dict(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        dumped = message.model_dump(exclude_none=True)
        return dict(dumped) if isinstance(dumped, dict) else {}
    if isinstance(message, dict):
        return dict(message)
    return {"role": "assistant", "content": str(message)}


def _tool_calls(message: Any) -> list[tuple[str, str, dict[str, Any]]]:
    raw_calls = getattr(message, "tool_calls", None)
    if raw_calls is None and isinstance(message, dict):
        raw_calls = message.get("tool_calls")
    calls: list[tuple[str, str, dict[str, Any]]] = []
    for raw in raw_calls or []:
        if isinstance(raw, dict):
            call_id = str(raw.get("id", ""))
            function = raw.get("function") or {}
            name = str(function.get("name", ""))
            arguments = _json_arguments(function.get("arguments"))
        else:
            call_id = str(getattr(raw, "id", ""))
            function = getattr(raw, "function", None)
            name = str(getattr(function, "name", ""))
            arguments = _json_arguments(getattr(function, "arguments", None))
        if call_id and name:
            calls.append((call_id, name, arguments))
    return calls


class RealAgent:
    def __init__(self, litellm_module: Any, model: str, harness: Harness, state: DemoState) -> None:
        self._litellm = litellm_module
        self._model = model
        self._harness = harness
        self._state = state

    async def _complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> tuple[dict[str, Any], list[tuple[str, str, dict[str, Any]]]]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "system", "content": system}, *messages],
            "tools": list(tools),
            "tool_choice": tool_choice,
            "max_tokens": 700,
            "num_retries": 1,
        }
        # GPT-5.6 rejects function tools at its default reasoning effort.
        if "gpt-5.6" in self._model:
            kwargs["reasoning_effort"] = "none"
        response = await self._litellm.acompletion(**kwargs)
        message = response.choices[0].message
        return _message_dict(message), _tool_calls(message)

    async def domain_step(
        self,
        env: OrderEnvironment,
        messages: list[dict[str, Any]],
        learned_rules: str,
        *,
        replay_context: str = "",
    ) -> list[ToolEvent]:
        rule_block = learned_rules.strip() or "_No learned rules are active._"
        system = (
            f"{replay_context}\n\n" if replay_context else ""
        ) + BASE_PROMPT + f"\n\nRULES READ BY THE HOST:\n{rule_block}"
        assistant, calls = await self._complete(
            system=system,
            messages=messages,
            tools=DOMAIN_TOOLS,
            # One model call is one evaluated operational step. Requiring a tool
            # prevents the agent from ending the step with a status paraphrase;
            # the model still chooses which of the four tools to call.
            tool_choice="required",
        )
        messages.append(assistant)
        events: list[ToolEvent] = []
        for call_id, name, arguments in calls:
            result = env.call(name, arguments)
            event = env.events[-1]
            events.append(event)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(result, default=str),
                }
            )
        return events

    async def read_rules(self, scope: str = "global") -> str:
        """Pull the live rule text through the read-only task toolset.

        This is the whole of the task agent's harness surface in 0.9.0. Rule bodies
        never arrive by injection: ``system_context`` announces the capability, and
        the agent (here, the host on its behalf) chooses to read.
        """

        result = await self._harness.task_tools.call(
            "harness_rules_read", {"scope": scope}
        )
        return str(result.get("content", "")) if result.get("ok") else ""


def _task_completion(report: EvalReport | None) -> float | None:
    if report is None:
        return None
    scores = [
        score.value
        for score in report.scores
        if str(score.metric) == "task_completion" and score.value is not None
    ]
    return scores[-1] if scores else None


def _tier2(report: EvalReport | None) -> list[MetricScore]:
    return list(report.scores_for_tier(2)) if report is not None else []


def _fmt_score(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _short(value: Any, width: int = 54) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= width else text[: width - 1] + "…"


def _candidate_scopes(harness: Harness) -> list[str]:
    """Scopes holding a candidate rule, `global` first, else every active scope.

    Managed repair chooses a rule's scope from the failure evidence, so a host
    cannot assume `global`. This take has produced `refund-operations`.
    """

    scopes = {
        str(getattr(rule, "scope", "global") or "global")
        for rule in harness.rules.candidates()
    }
    if not scopes:
        scopes = {
            str(getattr(rule, "scope", "global") or "global")
            for rule in harness.rules.all()
        } or {"global"}
    return sorted(scopes, key=lambda name: (name != "global", name))


def _action(event: ToolEvent | None) -> str:
    if event is None:
        return "no tool action"
    if event.name == "notify_customer":
        return "notify_customer(1017)"
    return f"{event.name}({event.arguments.get('order_id', '')})"


async def _evaluated_step(
    *,
    harness: Harness,
    pandaprobe_module: Any,
    session_id: str,
    turn_index: int,
    end_state: dict[str, Any],
    step: Callable[[], Awaitable[list[ToolEvent]]],
    expected_traces: int | Callable[[], int],
    repairs: list[Any],
) -> tuple[list[ToolEvent], EvalReport | None]:
    # Explicit on_turn_end carries the replay input a captured failure case needs.
    with pandaprobe_module.session(session_id):
        events = await step()
        # Must remain the final statement in the traced scope: turn-end below may
        # only discover COMPLETED traces after the SDK buffer has flushed.
        pandaprobe_module.flush(timeout=30.0)
    expected = expected_traces() if callable(expected_traces) else expected_traces
    await _wait_for_trace_ingestion(harness, session_id, expected)
    harness.on_turn_end(
        {"session_id": session_id, "turn_index": turn_index, "end_state": end_state}
    )
    # settle() is the whole healing beat in 0.9.0: it awaits this turn's evaluation,
    # posts a notice on breach, and runs one bounded managed-repair episode. The
    # repair agent reads the notice and writes the candidate.
    settled = await harness.settle(session_id)
    if settled.timed_out:
        raise RuntimeError(f"evaluation barrier timed out for {session_id} turn {turn_index}")
    if settled.repair is not None:
        repairs.append(settled.repair)
    return events, settled.report


async def _wait_for_trace_ingestion(
    harness: Harness, session_id: str, expected: int, *, timeout_s: float = 30.0
) -> None:
    """Wait until the real trace list exposes every flushed model call.

    SDK flush completes delivery, but the platform's trace index is eventually
    consistent. Firing turn-end before the current trace is listable can collapse
    two agent steps into one later evaluation, defeating per-step trajectories.
    """

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        result = await harness.cli.run(
            "traces",
            "list",
            "--session-id",
            session_id,
            "--status",
            "COMPLETED",
            "--sort-by",
            "started_at",
            "--sort-order",
            "desc",
            "--limit",
            str(max(expected, 20)),
        )
        payload = result.json()
        items = payload.get("items", []) if isinstance(payload, dict) else []
        if isinstance(items, list) and len(items) >= expected:
            return
        if loop.time() >= deadline:
            raise RuntimeError(
                f"only {len(items) if isinstance(items, list) else 0}/{expected} "
                f"completed traces became listable for {session_id}"
            )
        await asyncio.sleep(0.25)


async def _run_operational_task(
    *,
    label: str,
    max_turns: int,
    session_id: str,
    harness: Harness,
    agent: RealAgent,
    state: DemoState,
    pandaprobe_module: Any,
    repairs: list[Any],
) -> list[TurnRecord]:
    env = OrderEnvironment()
    state.environments[session_id] = env
    messages: list[dict[str, Any]] = [{"role": "user", "content": TASK}]
    # Host integration choice: read the live rules through the read-only task
    # toolset before each task, then supply that tool result to the model. Rule
    # bodies are absent from harness.system_context(); they come from the rule
    # files. Read every scope that holds a rule — managed repair picks the scope.
    scopes = _candidate_scopes(harness)
    chunks = [text for scope in scopes if (text := await agent.read_rules(scope)).strip()]
    learned_rules = "\n".join(chunks)
    active_count = len(harness.rules.active())
    print(
        f"  {label}: rules read from {', '.join(scopes)} ({active_count} active)",
        flush=True,
    )

    records: list[TurnRecord] = []
    for turn in range(1, max_turns + 1):
        print(f"  {label}: turn {turn} · model + trace evaluation", flush=True)

        async def step() -> list[ToolEvent]:
            events = await agent.domain_step(env, messages, learned_rules)
            # Make the completed task visible to the verifier before turn-end.
            if env.done:
                env.finished = True
            return events

        events, report = await _evaluated_step(
            harness=harness,
            pandaprobe_module=pandaprobe_module,
            session_id=session_id,
            turn_index=turn,
            end_state={"task": TASK, "order_id": ORDER_ID, "phase": label},
            step=step,
            expected_traces=turn,
            repairs=repairs,
        )
        records.append(TurnRecord(turn=turn, events=events, report=report))
        chosen = ", ".join(event.name for event in events) or "no tool"
        print(f"  {label}: turn {turn} action · {chosen}", flush=True)
        if repairs and repairs[-1].candidate_rule_ids:
            print(
                f"  {label}: repair agent wrote a candidate "
                f"({repairs[-1].selected_scope or 'global'})",
                flush=True,
            )
        if env.done:
            env.finished = True
            break
    env.finished = True
    return records


def _journal_subsequence(events: Sequence[dict[str, Any]], wanted: Sequence[str]) -> bool:
    cursor = iter(str(event.get("type", "")) for event in events)
    return all(any(item == target for item in cursor) for target in wanted)


def _history_has_trajectory(path: Path, session_id: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    prefix = f"{session_id}::"
    return any(
        key.startswith(prefix)
        and isinstance(value, dict)
        and len(value.get("series", [])) > 1
        for key, value in payload.items()
    )


def _promotion_deltas(state: DemoState) -> dict[str, float]:
    for verdict in state.verdicts:
        if verdict.outcome != "promote":
            continue
        cases = verdict.details.get("cases")
        if not isinstance(cases, list):
            continue
        for case in cases:
            if isinstance(case, dict) and case.get("kind") == "failure":
                deltas = case.get("deltas")
                if isinstance(deltas, dict):
                    return {
                        str(metric): float(delta)
                        for metric, delta in deltas.items()
                        if isinstance(delta, (int, float))
                    }
    return {}


def _print_summary(
    *,
    root: Path,
    elapsed: float,
    harness: Harness,
    state: DemoState,
    run1: list[TurnRecord],
    run2: list[TurnRecord],
    run1_session: str,
    run2_session: str,
) -> None:
    notices = [event for event in harness.journal.recent() if event.get("type") == "notice"]
    notice = notices[0] if notices else {}
    tier2 = next((score for record in run1 for score in _tier2(record.report)), None)
    promote_events = harness.journal.recent(types=("rule_promote",))
    promote = promote_events[-1] if promote_events else {}
    active = harness.rules.active()
    candidates = harness.rules.candidates()
    retired = [rule for rule in harness.rules.all() if rule.status == "retired"]
    rule = active[0] if active else None
    deltas = _promotion_deltas(state)

    print("\n" + LINE)
    print("  RUN 1  ·  no rules yet")
    print(LINE)
    for record in run1:
        event = record.events[0] if record.events else None
        stalled = bool(record.report and record.report.gate_breached)
        suffix = "   ← STALL (gate fired)" if stalled else ""
        print(
            f"  turn {record.turn:<2}  {_action(event):<29} "
            f"task_completion {_fmt_score(_task_completion(record.report))}{suffix}"
        )
    if tier2 is not None:
        print(
            f"           └─ tier 2: {str(tier2.metric):<21} "
            f"{_fmt_score(tier2.value)}  \"{_short(tier2.reason)}\""
        )
    if notice:
        print(
            f"           └─ notice posted  [{notice.get('severity')}]  ·  "
            "failure captured for replay"
        )

    repair = state.repairs[-1] if state.repairs else None
    print("\n  package-owned repair agent read the notice and wrote a rule:")
    if repair is not None:
        print(
            f"    repair session {repair.repair_session_id[:38]}  "
            f"({repair.turns} turns, {repair.tool_calls} tool calls)"
        )
    if rule is not None:
        print(f"    \"{rule.rule}\"")
        print(" " * 48 + "status: CANDIDATE → ACTIVE ✓")
    else:
        print("    no promoted rule")

    print("\n  evidence gate — replayed the captured failure with the rule in force")
    if deltas:
        for metric in ("task_completion", "tool_correctness", "outcome_correct"):
            if metric in deltas:
                print(f"    {metric:<20} Δ {deltas[metric]:+.2f}")
    print(f"    {_short(promote.get('reason', 'no promotion verdict'), 62)}")
    if promote:
        print("    no replay case regressed" + " " * 24 + "status: ACTIVE ✓")

    run1_passed = state.environments[run1_session].done
    print(f"\n  RESULT   task {'passed' if run1_passed else 'failed'}")
    print(LINE)
    print(f"  RUN 2  ·  {len(active)} proven rule in force")
    print(LINE)
    for record in run2:
        event = record.events[0] if record.events else None
        print(
            f"  turn {record.turn:<2}  {_action(event):<29} "
            f"task_completion {_fmt_score(_task_completion(record.report))}"
        )
    run2_env = state.environments[run2_session]
    print(
        f"\n  RESULT   task {'passed' if run2_env.done else 'failed':<27} "
        f"verifier outcome {run2_env.outcome:.2f}"
    )
    print(
        f"\n  {len(harness.mailbox.pending())} pending notices · "
        f"{len(state.repairs)} repair episodes · {len(candidates)} candidate rules · "
        f"{len(retired)} rules retired · {elapsed:.1f}s total"
    )
    print(LINE)
    print(
        f"  workspace: {root}  "
        "(rules/global.md · journal.jsonl · state/score_history.json)"
    )


def _acceptance_errors(
    *,
    root: Path,
    harness: Harness,
    state: DemoState,
    run1: list[TurnRecord],
    run2: list[TurnRecord],
    run1_session: str,
    run2_session: str,
    elapsed: float,
) -> list[str]:
    errors: list[str] = []
    journal = harness.journal.recent()
    active = harness.rules.active()
    # 0.9.0 renamed the task-facing index to rules.md.
    root_text = (root / "rules.md").read_text(encoding="utf-8")
    # Repair owns scope selection, so assert the rule landed in *some* scope file
    # rather than requiring `global`.
    scope_text = "\n".join(
        harness.rules.read_scope(scope) for scope in _candidate_scopes(harness)
    )
    run1_actions = [event.name for record in run1 for event in record.events]
    run2_actions = [event.name for record in run2 for event in record.events]
    promote = harness.journal.recent(types=("rule_promote",))

    if not _history_has_trajectory(root / "state" / "score_history.json", run1_session):
        errors.append("run 1 score history has no multi-sample trajectory")
    if not _journal_subsequence(journal, ("notice", "rule_add", "rule_promote")):
        errors.append("journal is missing notice → rule_add → rule_promote")
    if not any(result.candidate_rule_ids for result in state.repairs):
        errors.append("no managed-repair episode authored a candidate rule")
    if any(result.status == "timed_out" for result in state.repairs):
        errors.append("a managed-repair episode timed out")
    if not active:
        errors.append("no rule reached ACTIVE")
    elif active[0].rule not in scope_text or active[0].rule in root_text:
        errors.append("rule placement violates strict pull (scope file/root index)")
    if promote and not any(
        metric in str(promote[-1].get("reason", ""))
        for metric in ("task_completion", "tool_correctness", "outcome_correct")
    ):
        errors.append("promotion reason does not name a measured metric")
    if promote and promote[-1].get("validator") != "replay":
        errors.append("candidate was not promoted by replay evidence")
    if state.environments[run1_session].done:
        errors.append("run 1 unexpectedly passed")
    if not state.environments[run2_session].done:
        errors.append("run 2 did not pass")
    if run1_actions == run2_actions or "get_order_status" not in run2_actions:
        errors.append("run 2 behavior is not attributable to the learned status-check rule")
    if elapsed >= 90.0:
        errors.append(f"runtime exceeded 90s ({elapsed:.1f}s)")
    return errors


def _prepare_workspace(path: Path, reset: bool) -> Path:
    root = path.expanduser().resolve()
    if reset and root.exists():
        shutil.rmtree(root)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"workspace is not empty: {root} (pass --reset for a fresh take)")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _configure_quiet_output(litellm_module: Any) -> None:
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s", force=True)
    logging.getLogger("LiteLLM").setLevel(logging.ERROR)
    logging.getLogger("pandaprobe").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    litellm_module.suppress_debug_info = True
    litellm_module.set_verbose = False


async def _main(args: argparse.Namespace) -> int:
    try:
        import litellm
        import pandaprobe
        from pandaprobe.wrappers import wrap_litellm
    except ImportError as exc:
        print(
            "missing demo dependency — install: "
            "pip install pandaprobe-harness==0.9.0 pandaprobe litellm",
            file=sys.stderr,
        )
        print(f"import error: {exc}", file=sys.stderr)
        return 2

    _configure_quiet_output(litellm)
    if pandaprobe.get_client() is None:
        print(
            "PandaProbe tracing is not configured. Set PANDAPROBE_API_KEY and "
            "PANDAPROBE_PROJECT_NAME before running the demo.",
            file=sys.stderr,
        )
        return 2
    wrap_litellm(litellm)

    root = _prepare_workspace(args.workspace, args.reset)
    run_tag = uuid.uuid4().hex[:8]
    run1_session = f"yc-demo-{run_tag}-run1"
    run2_session = f"yc-demo-{run_tag}-run2"
    state = DemoState()
    # Repair reuses the task model unless the operator names a dedicated one.
    args.repair_model = args.repair_model or args.model

    cfg = HarnessConfig.from_env(
        harness_root=root,
        gate_window=1,
        gate_target=0.75,
        thresholds={"tool_correctness": 0.60},
        capture_eval_cases=True,
        barrier_timeout_s=300.0,
        poll_interval_s=0.5,
        poll_max_attempts=120,
        eval_retry_attempts=4,
        eval_retry_backoff_s=0.5,
        rule_trial_min_sessions=1,
        repair_model=args.repair_model,
        repair_reasoning_effort="none",
        trace_repair_agent=True,
        domain_policy=DOMAIN_POLICY,
    )

    agent_holder: dict[str, RealAgent] = {}
    replay_counter = 0

    async def replay(case: EvalCase, context: ReplayContext) -> str:
        nonlocal replay_counter
        replay_counter += 1
        session_id = f"yc-demo-{run_tag}-replay-{replay_counter}"
        env = OrderEnvironment()
        state.environments[session_id] = env
        state.replay = ReplayEvidence(
            session_id=session_id,
            candidate_visible="Provisional rules" in str(context),
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": TASK}]
        chunks: list[str] = []
        for scope in _candidate_scopes(harness):
            result = await context.task_tools.call("harness_rules_read", {"scope": scope})
            text = str(result.get("content", "")) if result.get("ok") else ""
            if text.strip():
                chunks.append(text)
        replay_rules = "\n".join(chunks)
        with pandaprobe.session(session_id):
            replay_steps = 0
            for _ in range(MAX_RUN2_TURNS):
                replay_steps += 1
                events = await agent_holder["agent"].domain_step(
                    env,
                    messages,
                    learned_rules=replay_rules,
                    replay_context=str(context),
                )
                state.replay.actions.extend(event.name for event in events)
                if env.done:
                    break
            env.finished = True
            pandaprobe.flush(timeout=30.0)
        await _wait_for_trace_ingestion(harness, session_id, replay_steps)
        return session_id

    harness = Harness.create(cfg, replay=replay, verifier=state.verify)
    agent = RealAgent(litellm, args.model, harness, state)
    agent_holder["agent"] = agent
    if not await harness.check_health():
        print("PandaProbe CLI is unavailable or unauthenticated; run `pandaprobe auth login`.")
        return 2

    started = time.monotonic()
    print(f"\nPandaProbe Harness self-heal · {args.model}")
    print(f"task: {TASK}")
    print("real model calls · real traces · real platform scoring\n")

    run1 = await _run_operational_task(
        label="run 1",
        max_turns=MAX_RUN1_TURNS,
        session_id=run1_session,
        harness=harness,
        agent=agent,
        state=state,
        pandaprobe_module=pandaprobe,
        repairs=state.repairs,
    )
    notices = [event for event in harness.journal.recent() if event.get("type") == "notice"]
    if not notices:
        raise RuntimeError("gate did not post a notice; inspect the real scores in the workspace")
    cases = harness.evalset.cases(kind="failure")
    if not any(case.replayable for case in cases):
        raise RuntimeError("gate fired without a replayable failure case")
    if not any(result.candidate_rule_ids for result in state.repairs):
        raise RuntimeError(
            "managed repair produced no candidate rule; inspect journal.jsonl "
            "for the repair episode's resolution"
        )

    print("  evidence gate: replaying the captured failure...", flush=True)
    state.verdicts.extend(await harness.validate_candidates())
    if not harness.rules.active():
        raise RuntimeError("candidate was not promoted; inspect journal.jsonl for the real verdict")
    promote_events = harness.journal.recent(types=("rule_promote",))
    if not promote_events or promote_events[-1].get("validator") != "replay":
        raise RuntimeError("candidate promotion did not come from replay evidence")

    run2 = await _run_operational_task(
        label="run 2",
        max_turns=MAX_RUN2_TURNS,
        session_id=run2_session,
        harness=harness,
        agent=agent,
        state=state,
        pandaprobe_module=pandaprobe,
        repairs=state.repairs,
    )
    elapsed = time.monotonic() - started
    _print_summary(
        root=root,
        elapsed=elapsed,
        harness=harness,
        state=state,
        run1=run1,
        run2=run2,
        run1_session=run1_session,
        run2_session=run2_session,
    )
    errors = _acceptance_errors(
        root=root,
        harness=harness,
        state=state,
        run1=run1,
        run2=run2,
        run1_session=run1_session,
        run2_session=run2_session,
        elapsed=elapsed,
    )
    if errors:
        print("\n  acceptance check failed honestly:")
        for error in errors:
            print(f"    - {error}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("DEMO_MODEL", DEFAULT_MODEL),
        help=f"task-agent LiteLLM model id (default: DEMO_MODEL or {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--repair-model",
        default=os.environ.get("HARNESS_REPAIR_MODEL", "") or None,
        help=(
            "LiteLLM model id for the package-owned repair agent "
            "(default: HARNESS_REPAIR_MODEL, else the task model)"
        ),
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=DEFAULT_WORKSPACE,
        help="auditable harness workspace (default: examples/demo/workspace)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete the selected workspace before this take",
    )
    args = parser.parse_args()
    try:
        return asyncio.run(_main(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - one legible demo failure, not a traceback wall
        print(f"\ndemo stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
