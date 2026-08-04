"""Closed-loop self-healing demo — evidence before trust, no network, no extras.

    python examples/misc/closed_loop_self_heal.py

Extends the offline pull-loop demo with the v0.6 closed loop. A toy
``ReplayFn`` stands in for "re-run my agent on a captured scenario":

  turn 1: the seeded failure -> eval breach -> a DiagnosticNotice posts AND
          the session is captured as a *replayable eval case* (its input came
          from the turn payload);
  turn 2: the agent pulls the notice and records a mitigation rule -> the
          rule lands as a CANDIDATE (in force, but provisional) -> the
          harness automatically replays the captured failure with the
          candidate in context -> the replayed session scores healthy ->
          the candidate is PROMOTED to active;
  then:   a protected `win` case is captured and `harness.run_regression()`
          confirms the fix stuck and nothing regressed.

Everything runs against a throwaway temp workspace and an in-process
``CliClient`` — the same seams the real ``pandaprobe`` binary plugs into.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from pandaprobe_harness import CliResult, EvalCase, Harness, HarnessConfig

SESSION = "s-demo-1"
REPLAY_SESSION = "s-replay-1"
MITIGATION_RULE = (
    "Never call the payment tool twice without first verifying the "
    "transaction status identifier."
)


class ScriptedCliClient:
    """In-process trace evaluator with session-isolated trajectories."""

    def __init__(self) -> None:
        self.healthy_sessions = {REPLAY_SESSION}
        self._runs = 0
        self._runs_by_id: dict[str, tuple[list[str], list[str]]] = {}
        self._traces: dict[str, list[str]] = {}
        self._trace_sessions: dict[str, str] = {}

    def emit_traces(self, session_id: str, count: int) -> None:
        traces = self._traces.setdefault(session_id, [])
        for _ in range(count):
            trace_id = f"{session_id}-trace-{len(traces) + 1}"
            traces.append(trace_id)
            self._trace_sessions[trace_id] = session_id

    async def run(self, *args: str, timeout: float | None = None) -> CliResult:
        payload = self._dispatch(args)
        return CliResult(args=args, exit_code=0, stdout=json.dumps(payload), stderr="")

    def _dispatch(self, args: tuple[str, ...]) -> Any:
        if args[:1] == ("version",):
            return {"version": "v0.7.0-demo"}
        if args[:2] == ("auth", "status"):
            return {"authenticated": True}
        if args[:2] == ("traces", "list"):
            session_id = _flag(args, "--session-id") or ""
            traces = self._traces.get(session_id, [])
            return {
                "items": [
                    {
                        "trace_id": trace_id,
                        "status": "COMPLETED",
                        "started_at": f"2026-01-01T00:00:{index:02d}+00:00",
                    }
                    for index, trace_id in reversed(list(enumerate(traces)))
                ]
            }
        if args[:3] == ("evals", "runs", "batch"):
            self._runs += 1
            run_id = f"run-{self._runs}"
            self._runs_by_id[run_id] = (
                [t for t in (_flag(args, "--trace-ids") or "").split(",") if t],
                [m for m in (_flag(args, "--metrics") or "").split(",") if m],
            )
            return {"id": run_id, "status": "PENDING"}
        if args[:3] == ("evals", "runs", "scores"):
            trace_ids, metrics = self._runs_by_id.get(args[3], ([], []))
            return [self._score(metric, trace_id) for trace_id in trace_ids for metric in metrics]
        if args[:3] == ("evals", "scores", "get"):
            return {"id": args[3], "scores": [self._score("tool_correctness", args[3])]}
        if args[:2] == ("traces", "spans"):
            spans = [{"kind": "TOOL", "name": "charge_payment", "input": {"amount": 42}}] * 2
            return {"trace_id": args[2], "spans": spans}
        if args[:2] == ("traces", "get"):
            return {"trace_id": args[2], "status": "OK", "span_count": 2}
        return {}

    def _score(self, metric: str, trace_id: str) -> dict[str, Any]:
        session_id = self._trace_sessions.get(trace_id, "")
        healthy = session_id in self.healthy_sessions
        values = (
            {
                "task_completion": 0.92,
                "coherence": 0.88,
                "tool_correctness": 0.92,
                "argument_correctness": 0.88,
            }
            if healthy
            else {
                "task_completion": 0.30,
                "coherence": 0.40,
                "tool_correctness": 0.20,
                "argument_correctness": 0.20,
            }
        )
        return {
            "name": metric,
            "trace_id": trace_id,
            "value": str(values.get(metric, 0.9)),
            "status": "SUCCESS",
            "reason": "ok" if healthy else "identical repeated tool call",
            "metadata": {"threshold": 0.5},
        }


def _flag(args: tuple[str, ...], name: str) -> str | None:
    for index, token in enumerate(args):
        if token == name and index + 1 < len(args):
            return args[index + 1]
    return None


async def replay(case: EvalCase, context: str) -> str:
    """The developer-supplied replay seam: re-run the agent on the case's
    input under ``context`` and return the NEW session id it produced."""

    provisional = "### Provisional rules (under evaluation)" in context
    print(
        f"   [replay] re-running case {case.id} "
        f"(input={case.replay_input}, candidate in force: {provisional})"
    )
    return REPLAY_SESSION


class PullAgent:
    """Scripted pull-protocol agent (see offline_self_heal.py for the basics)."""

    def __init__(self, harness: Harness, client: ScriptedCliClient) -> None:
        self._harness = harness
        self._client = client
        self.healed = False
        self.rule: dict[str, Any] = {}

    async def take_turn(self) -> str:
        listing = await self._harness.toolset.call("harness_mailbox_list", {})
        pending = listing.get("pending", []) if listing.get("ok") else []
        if pending and not self.healed:
            for notice in pending:
                await self._diagnose_and_heal(notice)
            return "diagnose"
        return "verified_payment_then_charge" if self.healed else "charge_payment"

    async def _diagnose_and_heal(self, summary: dict[str, Any]) -> None:
        toolset = self._harness.toolset
        notice_id = str(summary["id"])
        read = await toolset.call("harness_mailbox_read", {"notice_id": notice_id})
        notice = read.get("notice", {}) if read.get("ok") else {}
        for trace_id in notice.get("flagged_traces") or []:
            await toolset.call("harness_trace_inspect", {"trace_id": str(trace_id)})
        metrics = notice.get("metrics") or [{}]
        added = await toolset.call(
            "harness_rule_add",
            {
                "rule": MITIGATION_RULE,
                "rationale": "Repeated identical payment call flagged by the reliability eval.",
                "notice_id": notice_id,
                "metric": metrics[0].get("name"),
            },
        )
        self.rule = added.get("rule", {}) if added.get("ok") else {}
        print(
            f"   [agent] recorded rule {self.rule.get('id')} "
            f"(status={self.rule.get('status')}, tags={self.rule.get('tags')})"
        )
        await toolset.call(
            "harness_mailbox_ack",
            {"notice_id": notice_id, "rule_id": self.rule.get("id"), "note": "mitigated"},
        )
        self.healed = True
        # The fix is live for the agent's own session from here on.
        self._client.healthy_sessions.add(SESSION)


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="pandaprobe-closed-loop-")) / "harness"
    cfg = HarnessConfig(
        harness_root=root,
        poll_interval_s=0.0,
        poll_max_attempts=3,
        eval_retry_backoff_s=0.0,
        capture_eval_cases=True,  # breaches become replayable eval cases
        gate_window=1,
    )
    client = ScriptedCliClient()
    # The replayed session will score healthy: the rule demonstrably helps.
    client.emit_traces(REPLAY_SESSION, 1)
    harness = Harness.create(cfg, cli=client, replay=replay)
    agent = PullAgent(harness, client)
    print(f"workspace: {root}")

    # --- Turn 1: failure -> notice + captured eval case ---------------------
    print("\n[turn 1] the seeded failure:")
    action = await agent.take_turn()
    client.emit_traces(SESSION, 2)
    harness.on_turn_end({"session_id": SESSION, "turn_index": 1, "end_state": {"action": action}})
    await harness.refresh(SESSION)
    await harness.drain_validation()
    (case,) = harness.evalset.cases()
    print(f"[turn 1] notice posted; eval case captured: {case.id} "
          f"(signature={list(case.signature)}, replayable={case.replayable})")

    # --- Turn 2: candidate rule -> automatic replay validation -> promotion --
    print("\n[turn 2] the agent heals itself; the harness validates the rule:")
    action = await agent.take_turn()
    client.emit_traces(SESSION, 1)
    harness.on_turn_end({"session_id": SESSION, "turn_index": 2, "end_state": {"action": action}})
    await harness.refresh(SESSION)
    await harness.drain_validation()  # join the validation round

    status = await harness.toolset.call(
        "harness_rule_status", {"rule_id": str(agent.rule.get("id"))}
    )
    lifecycle = status.get("lifecycle", {})
    print(f"[turn 2] rule {agent.rule.get('id')} is now: {lifecycle.get('status')}")
    (promote,) = harness.journal.recent(types=("rule_promote",))
    print(f"[turn 2] promoted by {promote['validator']}: {promote['reason']}")

    context = harness.system_context()
    scope = str(agent.rule.get("scope", "scoped"))
    assert f"rules/{scope}.md" in context
    fetched = await harness.toolset.call("harness_rules_read", {"scope": scope})
    assert "payment tool twice" in fetched["content"]
    assert "### Provisional rules" not in fetched["content"]
    print("[turn 2] the validated rule is referenced and no longer provisional")

    # --- Regression guard: protect a win, confirm the fix -------------------
    print("\n[regression] protecting a win and replaying the eval-set:")
    harness.evalset.capture(
        session_id=SESSION,
        kind="win",
        signature=("healthy",),
        baseline_scores={"task_completion": 0.92, "coherence": 0.88},
        replay_input={"action": "verified_payment_then_charge"},
    )
    report = await harness.run_regression()
    print(report.render_text())
    assert report.clean and report.improved == 1

    types = [event["type"] for event in harness.journal.recent()]
    print("\njournal event types (in order):", " -> ".join(types))


if __name__ == "__main__":
    asyncio.run(main())
