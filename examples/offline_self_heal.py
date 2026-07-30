"""Offline self-healing demo — the complete pull loop, no network, no extras.

    python examples/offline_self_heal.py

A scripted CLI stand-in serves per-trace evaluation scores, and a scripted agent
follows the standing pull protocol. Three turns show the full acceptance flow:

  turn 1: a repeated identical tool call (the seeded failure). Tier 1 scores every
          trace of the turn; the trajectory is flat and below target, so the gate
          fires a STALL, Tier 2 then diagnoses the last trace and confirms a
          step-level breach -> a DiagnosticNotice lands in the mailbox. The
          per-turn barrier (``settle``) means this has all happened before the
          agent takes another step;
  turn 2: the agent sees the banner, pulls the notice, inspects the flagged
          trace, reads the rules already in force, records a mitigation rule with
          provenance, acknowledges. The rule lands in ``rules/<scope>.md`` and the
          skill root *references* it — the rule text is never injected;
  turn 3: healthy scores, no new notice, the recovery is journaled.

Everything runs against a throwaway temp workspace and an in-process
``CliClient`` — the same seams the real ``pandaprobe`` binary plugs into.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any

from pandaprobe_harness import CliResult, EvalReport, Harness, HarnessConfig

SESSION = "s-demo-1"
MITIGATION_RULE = (
    "Never call the payment tool twice without first verifying the "
    "transaction status identifier."
)


class ScriptedCliClient:
    """Minimal in-process stand-in for the ``pandaprobe`` binary (CliClient).

    Models the trace-target surface the v2 trigger uses: a session's traces are
    enumerable, and each trace can be scored on its own. Until the agent heals
    (``self.healed``) the scripted trajectory is **flat and below target** — which
    is what the trajectory gate flags. A climbing trajectory would not.
    """

    def __init__(self) -> None:
        self.healed = False  # flipped by the agent once its mitigation rule is in place
        self._runs = 0
        self._runs_by_id: dict[str, tuple[list[str], list[str]]] = {}
        self._traces: list[str] = []

    def emit_traces(self, count: int) -> None:
        """Simulate a turn emitting ``count`` traces (one per model call)."""

        start = len(self._traces)
        self._traces += [f"trace-{start + i + 1}" for i in range(count)]

    async def run(self, *args: str, timeout: float | None = None) -> CliResult:
        payload = self._dispatch(args)
        return CliResult(args=args, exit_code=0, stdout=json.dumps(payload), stderr="")

    def _dispatch(self, args: tuple[str, ...]) -> Any:
        if args[:1] == ("version",):
            return {"version": "v0.7.0-demo"}
        if args[:2] == ("auth", "status"):
            return {"authenticated": True}
        if args[:2] == ("traces", "list"):  # trace discovery, newest first
            items = [
                {"trace_id": tid, "status": "COMPLETED",
                 "started_at": f"2026-01-01T00:00:{i:02d}+00:00"}
                for i, tid in enumerate(self._traces)
            ]
            return {"items": list(reversed(items))}
        if args[:3] == ("evals", "runs", "batch"):  # async eval run handle
            self._runs += 1
            run_id = f"run-{self._runs}"
            # A run may cover several traces at once (Tier 1 batches the turn).
            self._runs_by_id[run_id] = (
                [t for t in (_flag(args, "--trace-ids") or "").split(",") if t],
                [m for m in (_flag(args, "--metrics") or "").split(",") if m],
            )
            return {"id": run_id, "status": "PENDING"}
        if args[:3] == ("evals", "runs", "scores"):  # terminal on first poll
            trace_ids, metrics = self._runs_by_id.get(args[3], ([], []))
            return [self._score(m, t) for t in trace_ids for m in metrics]
        if args[:3] == ("evals", "scores", "get"):  # every score of one trace
            return {"id": args[3], "scores": [self._score("tool_correctness", str(args[3]))]}
        if args[:2] == ("traces", "spans"):  # the flagged trace's (repeated) TOOL spans
            spans = [{"kind": "TOOL", "name": "charge_payment", "input": {"amount": 42}}] * 2
            return {"trace_id": args[2], "spans": spans}
        if args[:2] == ("traces", "get"):
            return {"trace_id": args[2], "status": "OK", "span_count": 2}
        return {}

    def _score(self, metric: str, trace_id: str) -> dict[str, Any]:
        healthy = {"task_completion": 0.95, "coherence": 0.9, "tool_correctness": 0.9,
                   "argument_correctness": 0.9}
        # Flat at 0.2: the agent is busy but getting no closer to done.
        failing = {"task_completion": 0.2, "coherence": 0.55, "tool_correctness": 0.2,
                   "argument_correctness": 0.6}
        table = healthy if self.healed else failing
        reason = (
            "the same payment call is repeated without checking the result"
            if not self.healed
            else "the agent verified state before acting"
        )
        return {
            "name": metric,
            "trace_id": trace_id,
            "value": str(table.get(metric, 0.9)),
            "status": "SUCCESS",
            "reason": reason,
            "metadata": {"threshold": 0.5},
        }


def _flag(args: tuple[str, ...], name: str) -> str | None:
    for i, token in enumerate(args):
        if token == name and i + 1 < len(args):
            return args[i + 1]
    return None


class PullAgent:
    """A scripted agent following the standing pull protocol.

    Checks the mailbox each turn; on a pending notice it reads it, inspects
    the flagged trace, consults the journal, records a rule, and acknowledges.
    """

    def __init__(self, harness: Harness, client: ScriptedCliClient) -> None:
        self._toolset = harness.toolset
        self._client = client
        self.healed = False
        self.rule: dict[str, Any] = {}

    async def take_turn(self) -> str:
        listing = await self._call("harness_mailbox_list", {})
        pending = listing.get("pending", []) if listing.get("ok") else []
        if pending and not self.healed:
            for notice in pending:
                await self._diagnose_and_heal(notice)
            return "diagnose"
        return "verified_payment_then_charge" if self.healed else "charge_payment"

    async def _diagnose_and_heal(self, summary: dict[str, Any]) -> None:
        notice_id = str(summary["id"])
        read = await self._call("harness_mailbox_read", {"notice_id": notice_id})
        notice = read.get("notice", {}) if read.get("ok") else {}
        for trace_id in notice.get("flagged_traces") or []:  # inspect the evidence
            await self._call("harness_trace_inspect", {"trace_id": str(trace_id)})
        # Read the rules already in force before writing a new one.
        await self._call("harness_rules_read", {"scope": "global"})
        metrics = notice.get("metrics") or [{}]
        added = await self._call(
            "harness_rule_add",
            {
                "rule": MITIGATION_RULE,
                "rationale": "Repeated identical payment call flagged by the reliability eval.",
                "notice_id": notice_id,  # provenance: the notice that motivated the rule
                "metric": metrics[0].get("name"),
                # Omitted `scope` defaults to the notice's own hint: `global` for a
                # trajectory fire, `scoped` for a surgical step-level breach.
            },
        )
        self.rule = added.get("rule", {}) if added.get("ok") else {}
        await self._call(
            "harness_mailbox_ack",
            {"notice_id": notice_id, "rule_id": self.rule.get("id"), "note": "mitigated"},
        )
        self.healed = True
        self._client.healed = True  # the fix is live: subsequent evals score high

    async def _call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        result = await self._toolset.call(name, args)
        print(f"   - {name} -> {self._describe(name, result)}")
        return result

    @staticmethod
    def _describe(name: str, result: dict[str, Any]) -> str:
        if not result.get("ok"):
            return f"ERROR: {result.get('error')}"
        if name == "harness_mailbox_list":
            return f"{len(result['pending'])} pending notice(s)"
        if name == "harness_mailbox_read":
            dump = result.get("dump") or {}
            return (
                f"notice {result['notice']['id']} (severity={result['notice']['severity']}); "
                f"dump loaded ({len(dump.get('scores', []))} scores)"
            )
        if name == "harness_trace_inspect":
            spans = (result.get("tool_spans") or {}).get("spans", [])
            return f"trace {result['trace_id']}: {len(spans)} TOOL span(s) + trace scores fetched"
        if name == "harness_rules_read":
            return f"read {result['path']} ({len(result['content'].splitlines())} lines)"
        if name == "harness_rule_add":
            return f"rule {result['rule']['id']} recorded"
        if name == "harness_mailbox_ack":
            rule_id = (result["notice"].get("resolution") or {}).get("rule_id")
            return f"notice acknowledged (linked rule {rule_id})"
        return "ok"


def _scores_line(report: EvalReport | None) -> str:
    """One line per tier, so the escalation is visible: Tier 1 ran on every trace,
    Tier 2 only on the last one."""

    assert report is not None, "the scripted eval always resolves within the barrier budget"
    parts: list[str] = []
    for tier in (1, 2, 3):
        scores = report.scores_for_tier(tier)
        if not scores:
            continue
        traces = {score.trace_id for score in scores}
        rendered = ", ".join(
            f"{score.metric}={score.value:.2f}"
            for score in scores
            if score.value is not None
        )
        parts.append(f"tier{tier} over {len(traces)} trace(s): {rendered}")
    return " | ".join(parts) or "no scores"


def _epilogue(harness: Harness, cfg: HarnessConfig) -> None:
    """Cross-run memory: the journal recorded the whole cycle; the rule persists."""

    types = [event["type"] for event in harness.journal.recent()]
    print("\njournal event types (in order):", " -> ".join(types))
    tail = cfg.rules_file.read_text(encoding="utf-8").strip().splitlines()[-3:]
    print(f"tail of {cfg.rules_file.name}:")
    for line in tail:
        print(f"   {line}")


async def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="pandaprobe-harness-demo-")) / "harness"
    cfg = HarnessConfig(
        harness_root=root,
        poll_interval_s=0.0,
        poll_max_attempts=3,
        eval_retry_backoff_s=0.0,
        eval_retry_attempts=1,
        # Two flat traces are enough to close the stall window in a short demo.
        gate_window=2,
    )
    client = ScriptedCliClient()
    harness = Harness.create(cfg, cli=client)  # provisions the workspace, wires the hook
    agent = PullAgent(harness, client)
    print(f"workspace: {root}")

    # --- Turn 1: the failure -> eval -> NOTICE posted (pull model: no injection)
    print("\n[turn 1] failing action:")
    action = await agent.take_turn()
    print(f"[turn 1] agent action: {action} (identical repeated tool call — the seeded failure)")
    client.emit_traces(3)  # the turn's model calls produced three traces
    harness.on_turn_end({"session_id": SESSION, "turn_index": 1, "end_state": {"action": action}})
    # `settle` is the per-turn barrier: it blocks until the tiers have run and any
    # notice is posted, so the agent cannot outrun its own diagnosis.
    print(f"[turn 1] eval resolved: {_scores_line((await harness.settle(SESSION)).report)}")
    notice = harness.mailbox.pending()[0]
    print(f"[turn 1] NOTICE posted: id={notice.id} severity={notice.severity}")
    print(f"         summary: {notice.summary}")
    banner = next(ln for ln in harness.system_context().splitlines() if "⚠ HARNESS" in ln)
    print(f"[turn 1] system-context banner: {banner}")

    # --- Turn 2: the agent pulls, diagnoses, records a rule, acknowledges
    print("\n[turn 2] the agent pulls its diagnostics and heals itself:")
    action = await agent.take_turn()
    scope = str(agent.rule.get("scope", "global"))
    print(f"[turn 2] new rule: {agent.rule.get('id')}: {agent.rule.get('rule')!r}")
    print(f"[turn 2] filed under: rules/{scope}.md")
    context = harness.system_context()
    assert "⚠ HARNESS" not in context
    # Strict pull: the root indexes the rule file, it never inlines the rule.
    assert f"rules/{scope}.md" in context and "payment tool twice" not in context
    print("[turn 2] banner cleared; the root now references the rule file (no rule text in it)")
    client.emit_traces(1)
    harness.on_turn_end({"session_id": SESSION, "turn_index": 2, "end_state": {"action": action}})
    await harness.settle(SESSION)  # healthy eval -> the hook journals the recovery

    # --- Turn 3: corrected behaviour, healthy scores, no new notice
    print("\n[turn 3] corrected behaviour:")
    action = await agent.take_turn()
    print(f"[turn 3] agent action: {action}")
    client.emit_traces(1)
    harness.on_turn_end({"session_id": SESSION, "turn_index": 3, "end_state": {"action": action}})
    print(
        "[turn 3] eval resolved healthy: "
        f"{_scores_line((await harness.settle(SESSION)).report)}"
    )
    print(f"[turn 3] no new notice: {len(harness.mailbox.pending())} pending")
    print(f"[turn 3] recovery journaled: {len(harness.journal.recent(types=('recovery',)))} event")

    _epilogue(harness, cfg)


if __name__ == "__main__":
    asyncio.run(main())
