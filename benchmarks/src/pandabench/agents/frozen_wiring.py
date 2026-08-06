"""Read-only agent wiring backed only by a frozen benchmark rules snapshot."""

from __future__ import annotations

import re
from typing import Any

from ..frozen_rules import FrozenRulesSnapshot

__all__ = ["FrozenEvalWiring"]

_TOOL_DETAILS: dict[str, tuple[str, dict[str, Any]]] = {
    "harness_rules_read": (
        "Read the frozen active and provisional rules in one scope.",
        {
            "type": "object",
            "properties": {"scope": {"type": "string"}},
            "required": ["scope"],
        },
    ),
    "harness_rules_search": (
        "Search the frozen ruleset by keyword relevance.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                "status": {"type": "string", "enum": ["active", "candidate", "retired"]},
            },
            "required": ["query"],
        },
    ),
    "harness_rules_list": (
        "List frozen rules, optionally filtered by lifecycle status.",
        {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["active", "candidate", "retired"]}
            },
        },
    ),
    "harness_rule_status": (
        "Read one frozen rule's lifecycle status and preserved trial state.",
        {
            "type": "object",
            "properties": {"rule_id": {"type": "string"}},
            "required": ["rule_id"],
        },
    ),
}


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", value.lower()))


class FrozenEvalWiring:
    """A no-Harness wiring surface for fixed-rule evaluation."""

    def __init__(self, snapshot: FrozenRulesSnapshot) -> None:
        self.snapshot = snapshot
        self._tools = [
            {
                "type": "function",
                "function": {"name": name, "description": detail[0], "parameters": detail[1]},
            }
            for name, detail in _TOOL_DETAILS.items()
        ]

    @property
    def settles_turns(self) -> bool:
        return False

    def system_preamble(self) -> str:
        lines = [
            "PANDAPROBE LEARNING RULES — FROZEN READ-ONLY EVALUATION",
            "Learning is complete. Apply the fixed ruleset indexed below; "
            "it cannot change during eval.",
            "Rule bodies remain available through the read-only harness_rules_* tools.",
            "",
            "References:",
        ]
        scopes = self.live_rule_scopes()
        if not scopes:
            lines.append("- No learning rules were available at the frozen boundary.")
        for scope in scopes:
            rules = [rule for rule in self.snapshot.live_rules if rule.get("scope") == scope]
            active = sum(rule.get("status") == "active" for rule in rules)
            candidate = sum(rule.get("status") == "candidate" for rule in rules)
            suffix = f"{active} active"
            if candidate:
                suffix += f", {candidate} provisional"
            lines.append(f"- rules/{scope}.md — {suffix}")
        return "\n".join(lines)

    def harness_tools(self) -> list[dict[str, Any]]:
        return list(self._tools)

    def pending_notice_ids(self, *, session_id: str | None = None) -> tuple[str, ...]:
        del session_id
        return ()

    def live_rule_scopes(self) -> tuple[str, ...]:
        scopes = {str(rule.get("scope") or "global") for rule in self.snapshot.live_rules}
        ordered: list[str] = [scope for scope in ("global", "scoped") if scope in scopes]
        ordered.extend(sorted(scope for scope in scopes if scope not in ("global", "scoped")))
        return tuple(ordered)

    def is_harness_tool(self, name: str) -> bool:
        # Route even a hallucinated mutation through the safe rejecting dispatcher.
        return name.startswith("harness_")

    async def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "harness_rules_read":
            return self._read(args)
        if name == "harness_rules_search":
            return self._search(args)
        if name == "harness_rules_list":
            return self._list(args)
        if name == "harness_rule_status":
            return self._status(args)
        return {
            "ok": False,
            "error": f"tool {name!r} is unavailable in frozen read-only evaluation",
        }

    async def settle_turn(self, turn_index: int) -> None:
        """Compatibility no-op; eval callers intentionally do not invoke it."""

        del turn_index

    def _read(self, args: dict[str, Any]) -> dict[str, Any]:
        scope = str(args.get("scope") or "global")
        rules = [rule for rule in self.snapshot.live_rules if rule.get("scope") == scope]
        active = [rule for rule in rules if rule.get("status") == "active"]
        candidates = [rule for rule in rules if rule.get("status") == "candidate"]
        lines = [f"# Rules — {scope}", ""]
        if not rules:
            lines.append("_No frozen live rules in this scope._")
        for rule in active:
            lines.append(f"- [{rule.get('id')}] {rule.get('rule', '')}")
        if candidates:
            lines.extend(["", "## Provisional candidates", ""])
            for rule in candidates:
                lines.append(f"- [candidate {rule.get('id')}] {rule.get('rule', '')}")
        return {
            "ok": True,
            "scope": scope,
            "path": f"rules/{scope}.md",
            "content": "\n".join(lines) + "\n",
        }

    def _search(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "")
        try:
            limit = max(1, min(50, int(args.get("limit", 10))))
        except (TypeError, ValueError):
            limit = 10
        requested = args.get("status")
        statuses = {str(requested)} if requested in ("active", "candidate", "retired") else {
            "active",
            "candidate",
        }
        query_tokens = _tokens(query)
        ranked: list[tuple[int, str, dict[str, Any]]] = []
        for rule in self.snapshot.rules:
            if rule.get("status") not in statuses:
                continue
            haystack = " ".join(
                str(rule.get(key) or "") for key in ("rule", "rationale", "scope", "metric")
            )
            haystack += " " + " ".join(str(tag) for tag in rule.get("tags") or [])
            score = len(query_tokens & _tokens(haystack))
            ranked.append((score, str(rule.get("created_at", "")), rule))
        ranked.sort(key=lambda item: (item[0], item[1], str(item[2].get("id", ""))), reverse=True)
        return {
            "ok": True,
            "rules": [{**rule, "score": score} for score, _, rule in ranked[:limit]],
        }

    def _list(self, args: dict[str, Any]) -> dict[str, Any]:
        requested = args.get("status")
        rules = list(self.snapshot.rules)
        if requested in ("active", "candidate", "retired"):
            rules = [rule for rule in rules if rule.get("status") == requested]
        return {"ok": True, "rules": rules}

    def _status(self, args: dict[str, Any]) -> dict[str, Any]:
        rule_id = str(args.get("rule_id") or "")
        for rule in self.snapshot.rules:
            if str(rule.get("id")) == rule_id:
                return {
                    "ok": True,
                    "rule": rule,
                    "lifecycle": {
                        "status": rule.get("status"),
                        "trial": rule.get("trial"),
                    },
                }
        return {"ok": False, "error": f"no frozen rule {rule_id!r}"}
