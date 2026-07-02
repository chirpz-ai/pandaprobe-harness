"""Offline self-healing demo — the complete pull loop, no network, no extras.

    python examples/offline_self_heal.py

A scripted CLI stand-in serves failing evaluation scores, and a scripted agent
follows the standing pull protocol. Three turns show the full acceptance flow:

  turn 1: a repeated identical tool call (the seeded failure) -> eval breach ->
          a DiagnosticNotice lands in the workspace mailbox (nothing injected);
  turn 2: the agent sees the banner, pulls the notice, inspects the flagged
          trace, records a mitigation rule with provenance, acknowledges;
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

    Serves low scores until the agent heals (``self.healed``), then high ones.
    """

    def __init__(self) -> None:
        self.healed = False  # flipped by the agent once its mitigation rule is in place
        self._runs = 0

    async def run(self, *args: str, timeout: float | None = None) -> CliResult:
        payload = self._dispatch(args)
        return CliResult(args=args, exit_code=0, stdout=json.dumps(payload), stderr="")

    def _dispatch(self, args: tuple[str, ...]) -> Any:
        if args[:1] == ("version",):
            return {"version": "v0.5.0-demo"}
        if args[:2] == ("auth", "status"):
            return {"authenticated": True}
        if args[:3] == ("evals", "runs", "batch"):  # async eval run handle
            self._runs += 1
            return {"id": f"run-{self._runs}", "status": "PENDING"}
        if args[:3] == ("evals", "runs", "scores"):  # terminal on first poll
            return self._session_scores()
        if args[:3] == ("evals", "scores", "get"):  # trace-level scores (diagnosis)
            return {"id": args[3], "scores": [{"name": "loop_detection", "value": "0.1"}]}
        if args[:2] == ("traces", "spans"):  # the flagged trace's (repeated) TOOL spans
            spans = [{"kind": "TOOL", "name": "charge_payment", "input": {"amount": 42}}] * 2
            return {"trace_id": args[2], "spans": spans}
        if args[:2] == ("traces", "get"):
            return {"trace_id": args[2], "status": "OK", "span_count": 2}
        return {}

    def _session_scores(self) -> list[dict[str, Any]]:
        reliability, consistency = (0.92, 0.88) if self.healed else (0.30, 0.40)
        flagged: dict[str, Any] = {}
        if not self.healed:  # session-metric metadata carries the per-trace evidence
            flagged = {
                "flagged_traces": ["trace-1"],
                "per_trace_signals": {"trace-1": {"loop_detection": 0.1, "tool_correctness": 0.2}},
            }
        reasons = ("identical repeated tool call detected", "inconsistent session state")
        if self.healed:
            reasons = ("ok", "ok")
        return [
            {"name": "agent_reliability", "value": str(reliability), "status": "SUCCESS",
             "reason": reasons[0], "metadata": flagged},
            {"name": "agent_consistency", "value": str(consistency), "status": "SUCCESS",
             "reason": reasons[1], "metadata": {}},
        ]


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
        await self._call("harness_journal", {"limit": 10})  # any recurring pattern?
        metrics = notice.get("metrics") or [{}]
        added = await self._call(
            "harness_rule_add",
            {
                "rule": MITIGATION_RULE,
                "rationale": "Repeated identical payment call flagged by the reliability eval.",
                "notice_id": notice_id,  # provenance: the notice that motivated the rule
                "metric": metrics[0].get("name"),
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
        if name == "harness_journal":
            return f"{len(result['events'])} recent journal event(s)"
        if name == "harness_rule_add":
            return f"rule {result['rule']['id']} recorded"
        if name == "harness_mailbox_ack":
            rule_id = (result["notice"].get("resolution") or {}).get("rule_id")
            return f"notice acknowledged (linked rule {rule_id})"
        return "ok"


def _scores_line(report: EvalReport | None) -> str:
    assert report is not None, "the scripted eval always resolves within the refresh budget"
    return ", ".join(f"{score.metric}={score.value:.2f}" for score in report.scores)


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
        harness_root=root, poll_interval_s=0.0, poll_max_attempts=3, eval_retry_backoff_s=0.0
    )
    client = ScriptedCliClient()
    harness = Harness.create(cfg, cli=client)  # provisions the workspace, wires the hook
    agent = PullAgent(harness, client)
    print(f"workspace: {root}")

    # --- Turn 1: the failure -> eval -> NOTICE posted (pull model: no injection)
    print("\n[turn 1] failing action:")
    action = await agent.take_turn()
    print(f"[turn 1] agent action: {action} (identical repeated tool call — the seeded failure)")
    harness.on_turn_end({"session_id": SESSION, "turn_index": 1, "end_state": {"action": action}})
    print(f"[turn 1] eval resolved: {_scores_line(await harness.refresh(SESSION))}")
    notice = harness.mailbox.pending()[0]
    print(f"[turn 1] NOTICE posted: id={notice.id} severity={notice.severity}")
    print(f"         summary: {notice.summary}")
    banner = next(ln for ln in harness.system_context().splitlines() if "⚠ HARNESS" in ln)
    print(f"[turn 1] system-context banner: {banner}")

    # --- Turn 2: the agent pulls, diagnoses, records a rule, acknowledges
    print("\n[turn 2] the agent pulls its diagnostics and heals itself:")
    action = await agent.take_turn()
    print(f"[turn 2] new rule: {agent.rule.get('id')}: {agent.rule.get('rule')!r}")
    context = harness.system_context()
    assert "⚠ HARNESS" not in context and "payment tool twice" in context
    print("[turn 2] banner cleared: system context has no '⚠ HARNESS' line; rule re-enters it")
    harness.on_turn_end({"session_id": SESSION, "turn_index": 2, "end_state": {"action": action}})
    await harness.refresh(SESSION)  # healthy eval -> the hook journals the recovery

    # --- Turn 3: corrected behaviour, healthy scores, no new notice
    print("\n[turn 3] corrected behaviour:")
    action = await agent.take_turn()
    print(f"[turn 3] agent action: {action}")
    harness.on_turn_end({"session_id": SESSION, "turn_index": 3, "end_state": {"action": action}})
    print(f"[turn 3] eval resolved healthy: {_scores_line(await harness.refresh(SESSION))}")
    print(f"[turn 3] no new notice: {len(harness.mailbox.pending())} pending")
    print(f"[turn 3] recovery journaled: {len(harness.journal.recent(types=('recovery',)))} event")

    _epilogue(harness, cfg)


if __name__ == "__main__":
    asyncio.run(main())
