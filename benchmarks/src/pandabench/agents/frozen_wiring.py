"""Read-only agent wiring backed only by a frozen benchmark rules snapshot.

**Currently unwired**: no run serves a frozen ruleset now that the harness is live
throughout. See :mod:`pandabench.frozen_rules` for why it is kept.
"""

from __future__ import annotations

import importlib.resources
import re
from typing import Any

# Frozen eval reimplements the read surface over a snapshot rather than a live
# store, but scope identity is the harness's to define — importing it keeps the
# two from drifting on defaults or on what counts as a safe name.
from pandaprobe_harness.workspace.scopes import (
    GLOBAL_SCOPE,
    normalize_scope_description,
    validate_scope,
)

from ..frozen_rules import FrozenRulesSnapshot

__all__ = ["FrozenEvalWiring"]

_TOOL_DETAILS: dict[str, tuple[str, dict[str, Any]]] = {
    "harness_rules_read": (
        "Read the frozen active and provisional rules in one scope.",
        {
            "type": "object",
            "properties": {"scope": {"type": "string"}},
            "required": [],
        },
    ),
    "harness_rules_search": (
        "Search the frozen ruleset by keyword relevance.",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["query"],
        },
    ),
    "harness_rules_list": (
        "Load the canonical rules.md guide and its compact frozen live-scope index.",
        {"type": "object", "properties": {}},
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
        return (
            "Frozen learned rules are available through PandaProbe's read-only "
            "tools. Call harness_rules_list to load rules.md and inspect "
            "relevant rule scopes; harness_rules_read, harness_rules_search, and "
            "harness_rule_status read them on demand. PandaProbe does not "
            "automatically insert rule contents."
        )

    def harness_tools(self) -> list[dict[str, Any]]:
        return list(self._tools)

    def pending_notice_ids(self, *, session_id: str | None = None) -> tuple[str, ...]:
        del session_id
        return ()

    def live_rule_scopes(self) -> tuple[str, ...]:
        scopes = {
            str(rule.get("scope") or GLOBAL_SCOPE) for rule in self.snapshot.live_rules
        }
        ordered: list[str] = [GLOBAL_SCOPE] if GLOBAL_SCOPE in scopes else []
        ordered.extend(sorted(scope for scope in scopes if scope != GLOBAL_SCOPE))
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
        try:
            scope = (
                validate_scope(args["scope"])
                if args.get("scope") is not None
                else GLOBAL_SCOPE
            )
        except ValueError:
            return {"ok": False, "error": "invalid rule scope"}
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
        statuses = {"active", "candidate"}
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
            "rules": [
                {
                    "id": rule.get("id"),
                    "scope": rule.get("scope") or GLOBAL_SCOPE,
                    "status": rule.get("status"),
                    "snippet": _snippet(str(rule.get("rule") or "")),
                    "score": score,
                }
                for score, _, rule in ranked[:limit]
            ],
        }

    def _list(self, args: dict[str, Any]) -> dict[str, Any]:
        del args
        scopes: list[dict[str, Any]] = []
        lines = [_guide_head(), ""]
        for index, scope in enumerate(self.live_rule_scopes()):
            rules = [rule for rule in self.snapshot.live_rules if rule.get("scope") == scope]
            active = sum(rule.get("status") == "active" for rule in rules)
            provisional = sum(rule.get("status") == "candidate" for rule in rules)
            description = _scope_description(scope)
            if index:
                lines.append("")
            lines.extend(
                [
                    f"- [`{scope}`](rules/{scope}.md)",
                    f"  {description}",
                    f"  {active} active, {provisional} provisional",
                ]
            )
            scopes.append(
                {
                    "scope": scope,
                    "path": f"rules/{scope}.md",
                    "description": description,
                    "active": active,
                    "provisional": provisional,
                }
            )
        if not scopes:
            lines.append("_No learned rules are available yet._")
        return {
            "ok": True,
            "path": "rules.md",
            "content": "\n".join(lines) + "\n",
            "scopes": scopes,
        }

    def _status(self, args: dict[str, Any]) -> dict[str, Any]:
        rule_id = str(args.get("rule_id") or "")
        for rule in self.snapshot.rules:
            if str(rule.get("id")) == rule_id:
                return {
                    "ok": True,
                    "rule": {
                        "id": rule.get("id"),
                        "scope": rule.get("scope") or GLOBAL_SCOPE,
                        "status": rule.get("status"),
                        "applicability": rule.get("applicability"),
                    },
                    "lifecycle": {
                        "status": rule.get("status"),
                        "trial": rule.get("trial"),
                    },
                }
        return {"ok": False, "error": f"no frozen rule {rule_id!r}"}


def _snippet(value: str, *, limit: int = 240) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"


def _scope_description(scope: str) -> str:
    return normalize_scope_description(None, scope=scope)


def _guide_head() -> str:
    """Load the installed package's canonical SKILL-style guide."""

    marker = "<!-- REFERENCES — generated by the harness; do not edit below -->"
    resource = (
        importlib.resources.files("pandaprobe_harness.filesystem.templates")
        / "rules.md"
    )
    template = resource.read_text(encoding="utf-8")
    before, found, _ = template.partition(marker)
    return before + marker if found else template.rstrip()
