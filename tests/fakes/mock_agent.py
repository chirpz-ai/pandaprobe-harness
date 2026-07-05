"""A scripted mock agent that drives the pull-based self-healing loop.

Behaviour (the standing pull protocol, scripted):

* At the start of every turn the agent checks its mailbox
  (``harness_mailbox_list``).
* When notices are pending and it has not yet healed, it works through each
  one: read the notice + dump, inspect the first flagged trace, consult the
  journal, record a mitigation rule with provenance, and acknowledge the
  notice linking the rule. It emits a ``diagnose`` turn.
* Otherwise it emits the failure action (an *identical repeated* tool call —
  the infinite-repetition failure the harness should catch) until healed, and
  a distinct corrected action afterwards.

Every tool call and action is recorded for assertions.
"""

from __future__ import annotations

from typing import Any

from pandaprobe_harness import HarnessToolset
from pandaprobe_harness.adapters.raw_loop import RawLoopAdapter

MITIGATION_RULE = (
    "Never call the payment tool twice without first verifying the "
    "transaction status identifier."
)


class MockLLMAgent:
    def __init__(self, *, session_id: str, toolset: HarnessToolset) -> None:
        self._session_id = session_id
        self._toolset = toolset

        self.turn_index = 0
        self.healed = False
        self.actions: list[str] = []
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.rule_ids: list[str] = []
        self.rule_status: dict[str, Any] | None = None

    async def take_turn(self) -> dict[str, Any]:
        self.turn_index += 1

        # The standing protocol: check the mailbox at the start of each turn.
        listing = await self._call("harness_mailbox_list", {})
        pending = listing.get("pending", []) if listing.get("ok") else []

        if pending and not self.healed:
            for notice in pending:
                await self._diagnose_and_heal(notice)
            action = "diagnose"
        elif self.healed:
            action = "verified_payment_then_charge"  # corrected, non-repeating
        else:
            action = "charge_payment"  # repeated identical call (the failure)

        self.actions.append(action)
        return RawLoopAdapter.make_turn(self._session_id, self.turn_index, action=action)

    async def _diagnose_and_heal(self, notice_summary: dict[str, Any]) -> None:
        notice_id = str(notice_summary["id"])
        # 1. Read the notice in full, including the trace dump.
        read = await self._call("harness_mailbox_read", {"notice_id": notice_id})
        notice = read.get("notice", {}) if read.get("ok") else {}
        # 2. Inspect the first flagged trace via the platform.
        flagged = notice.get("flagged_traces") or []
        if flagged:
            await self._call("harness_trace_inspect", {"trace_id": str(flagged[0])})
        # 3. Consult the cross-run journal for recurring patterns.
        await self._call("harness_journal", {"limit": 10})
        # 4. Record a permanent mitigation rule with provenance.
        metrics = notice.get("metrics") or []
        metric = str(metrics[0]["name"]) if metrics else None
        added = await self._call(
            "harness_rule_add",
            {
                "rule": MITIGATION_RULE,
                "rationale": "Repeated identical payment call flagged by reliability eval.",
                "notice_id": notice_id,
                "metric": metric,
            },
        )
        rule_id = added.get("rule", {}).get("id") if added.get("ok") else None
        if rule_id:
            self.rule_ids.append(str(rule_id))
        # 5. Acknowledge the notice, linking the mitigation rule.
        await self._call(
            "harness_mailbox_ack",
            {"notice_id": notice_id, "rule_id": rule_id, "note": "mitigated"},
        )
        # 6. Check the rule's lifecycle (candidate rules are validated by the
        #    harness before they are trusted).
        if rule_id:
            status = await self._call("harness_rule_status", {"rule_id": rule_id})
            if status.get("ok"):
                self.rule_status = status.get("lifecycle")
        self.healed = True

    async def _call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.tool_calls.append((name, args))
        return await self._toolset.call(name, args)
