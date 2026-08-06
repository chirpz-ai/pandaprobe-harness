"""Capability-separated task and managed-repair workspace tools."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

from ..cli.client import CliClient
from ..cli.errors import CliError
from ..config import HarnessConfig
from ..workspace._io import load_json
from ..workspace.journal import Journal
from ..workspace.mailbox import Mailbox
from ..workspace.rules import RulesStore, RuleStatus, derive_notice_tags, normalize_scope
from ..workspace.sanitize import is_sensitive_key, sanitize_text
from .spec import ToolSpec

__all__ = ["REPAIR_OP_SCHEMAS", "TASK_OP_SCHEMAS", "RepairToolset", "TaskToolset"]

logger = logging.getLogger("pandaprobe_harness.agent_tools")

TASK_OP_SCHEMAS: dict[str, dict[str, Any]] = {
    "harness_rules_read": {
        "description": "Read one learned-rule scope. Defaults to global.",
        "input_schema": {
            "type": "object",
            "properties": {"scope": {"type": "string"}},
            "required": [],
        },
    },
    "harness_rules_search": {
        "description": "Search learned rules by lexical relevance across scopes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
                "status": {
                    "type": "string",
                    "enum": ["candidate", "active", "retired"],
                },
            },
            "required": ["query"],
        },
    },
    "harness_rules_list": {
        "description": "List rules by lifecycle status, or all rules when omitted.",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["candidate", "active", "retired"],
                }
            },
            "required": [],
        },
    },
    "harness_rule_status": {
        "description": "Read one rule's lifecycle and validation state.",
        "input_schema": {
            "type": "object",
            "properties": {"rule_id": {"type": "string"}},
            "required": ["rule_id"],
        },
    },
}

REPAIR_OP_SCHEMAS: dict[str, dict[str, Any]] = {
    "harness_notice_read": {
        "description": "Read the assigned diagnostic notice and its bounded dump.",
        "input_schema": {
            "type": "object",
            "properties": {"notice_id": {"type": "string"}},
            "required": ["notice_id"],
        },
    },
    "harness_trace_inspect": {
        "description": "Inspect one trace explicitly named by the assigned notice.",
        "input_schema": {
            "type": "object",
            "properties": {"trace_id": {"type": "string"}},
            "required": ["trace_id"],
        },
    },
    **TASK_OP_SCHEMAS,
    "harness_rule_add": {
        "description": "Add concise guidance as a candidate linked to the assigned notice.",
        "input_schema": {
            "type": "object",
            "properties": {
                "rule": {"type": "string"},
                "rationale": {"type": "string"},
                "metric": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "scope": {"type": "string"},
            },
            "required": ["rule", "rationale"],
        },
    },
    "harness_rule_retire": {
        "description": "Retire an obsolete active rule; candidates remain validator-owned.",
        "input_schema": {
            "type": "object",
            "properties": {
                "rule_id": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["rule_id", "reason"],
        },
    },
    "harness_notice_ack": {
        "description": "Acknowledge the assigned notice after adding its candidate.",
        "input_schema": {
            "type": "object",
            "properties": {
                "notice_id": {"type": "string"},
                "rule_id": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["notice_id", "rule_id"],
        },
    },
    "harness_notice_resolve": {
        "description": "Resolve as duplicate/already-covered or no actionable proposal.",
        "input_schema": {
            "type": "object",
            "properties": {
                "notice_id": {"type": "string"},
                "resolution": {
                    "type": "string",
                    "enum": ["duplicate", "no_proposal"],
                },
                "existing_rule_id": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["notice_id", "resolution", "note"],
        },
    },
}


class _RuleReads:
    def __init__(self, *, config: HarnessConfig, rules: RulesStore) -> None:
        self._config = config
        self._rules = rules

    async def rules_read(self, args: Mapping[str, Any]) -> dict[str, Any]:
        scope = normalize_scope(str(args["scope"]) if args.get("scope") is not None else None)
        content = await asyncio.to_thread(self._rules.read_scope, scope)
        return {"ok": True, "scope": scope, "path": f"rules/{scope}.md", "content": content}

    async def rules_search(self, args: Mapping[str, Any]) -> dict[str, Any]:
        query = str(args["query"])
        try:
            limit = int(args.get("limit", 10))
        except (TypeError, ValueError):
            limit = 10
        limit = min(50, max(1, limit))
        raw_status = args.get("status")
        statuses: tuple[RuleStatus, ...] = (
            (raw_status,)
            if raw_status in {"candidate", "active", "retired"}
            else ("active", "candidate")
        )
        results = await asyncio.to_thread(
            self._rules.search, query, limit=limit, statuses=statuses
        )
        return {
            "ok": True,
            "rules": [{**rule.to_json(), "score": score} for rule, score in results],
        }

    async def rules_list(self, args: Mapping[str, Any]) -> dict[str, Any]:
        raw_status = args.get("status")
        rules = await asyncio.to_thread(self._rules.all)
        if raw_status in {"candidate", "active", "retired"}:
            rules = [rule for rule in rules if rule.status == raw_status]
        return {"ok": True, "rules": [rule.to_json() for rule in rules]}

    async def rule_status(self, args: Mapping[str, Any]) -> dict[str, Any]:
        rule_id = str(args["rule_id"])
        rules = await asyncio.to_thread(self._rules.all)
        for rule in rules:
            if rule.id != rule_id:
                continue
            lifecycle: dict[str, Any] = {"status": rule.status}
            if rule.trial is not None:
                lifecycle.update(
                    {
                        "baseline_rate": rule.trial.baseline_rate,
                        "trial_rate": rule.trial.trial_rate,
                        "sessions_observed": len(rule.trial.observed_sessions),
                        "sessions_needed": self._config.rule_trial_min_sessions,
                        "replay_attempts": rule.trial.replay_attempts,
                        "verdict": rule.trial.verdict,
                    }
                )
            return {"ok": True, "rule": rule.to_json(), "lifecycle": lifecycle}
        return {"ok": False, "error": f"no rule {rule_id!r}"}


class TaskToolset(_RuleReads):
    """Read-only capability surface safe to attach to a task agent."""

    def __init__(self, *, config: HarnessConfig, rules: RulesStore) -> None:
        super().__init__(config=config, rules=rules)
        self._specs = _specs(
            TASK_OP_SCHEMAS,
            {
                "harness_rules_read": self.rules_read,
                "harness_rules_search": self.rules_search,
                "harness_rules_list": self.rules_list,
                "harness_rule_status": self.rule_status,
            },
        )

    def specs(self) -> tuple[ToolSpec, ...]:
        return self._specs

    async def call(self, name: str, args: Mapping[str, Any]) -> dict[str, Any]:
        return await _dispatch(self._specs, name, args)


class RepairToolset(_RuleReads):
    """One-notice administrative surface used only by managed repair."""

    def __init__(
        self,
        *,
        config: HarnessConfig,
        cli: CliClient,
        mailbox: Mailbox,
        journal: Journal,
        rules: RulesStore,
        notice_id: str,
        allowed_trace_ids: tuple[str, ...],
    ) -> None:
        super().__init__(config=config, rules=rules)
        self._cli = cli
        self._mailbox = mailbox
        self._journal = journal
        self._notice_id = notice_id
        self._allowed_trace_ids = frozenset(allowed_trace_ids)
        self._candidate_ids: list[str] = []
        self._resolution: str | None = None
        self._existing_rule_id: str | None = None
        handlers = {
            "harness_notice_read": self.notice_read,
            "harness_trace_inspect": self.trace_inspect,
            "harness_rules_read": self.rules_read,
            "harness_rules_search": self.rules_search,
            "harness_rules_list": self.rules_list,
            "harness_rule_status": self.rule_status,
            "harness_rule_add": self.rule_add,
            "harness_rule_retire": self.rule_retire,
            "harness_notice_ack": self.notice_ack,
            "harness_notice_resolve": self.notice_resolve,
        }
        self._specs = _specs(REPAIR_OP_SCHEMAS, handlers)

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(self._candidate_ids)

    @property
    def resolution(self) -> str | None:
        return self._resolution

    @property
    def existing_rule_id(self) -> str | None:
        return self._existing_rule_id

    @property
    def resolved(self) -> bool:
        return self._resolution is not None

    def specs(self) -> tuple[ToolSpec, ...]:
        return self._specs

    async def call(self, name: str, args: Mapping[str, Any]) -> dict[str, Any]:
        return await _dispatch(self._specs, name, args)

    def _assigned(self, args: Mapping[str, Any]) -> dict[str, Any] | None:
        supplied = str(args.get("notice_id", ""))
        if supplied != self._notice_id:
            return {"ok": False, "error": "repair may access only its assigned notice"}
        return None

    async def notice_read(self, args: Mapping[str, Any]) -> dict[str, Any]:
        if error := self._assigned(args):
            return error
        notice = await asyncio.to_thread(self._mailbox.read, self._notice_id)
        if notice is None:
            return {"ok": False, "error": f"no notice {self._notice_id!r}"}
        dump = None
        if notice.dump_path:
            dump = await asyncio.to_thread(load_json, Path(notice.dump_path))
        return {
            "ok": True,
            "notice": notice.to_json(),
            "dump": _bounded_evidence(dump, max_len=self._config.sanitize_max_len),
        }

    async def trace_inspect(self, args: Mapping[str, Any]) -> dict[str, Any]:
        trace_id = str(args["trace_id"])
        if trace_id not in self._allowed_trace_ids:
            return {"ok": False, "error": "trace is outside the assigned notice"}
        trace = await self._cli_json("traces", "get", trace_id)
        tool_spans = await self._cli_json("traces", "spans", trace_id, "--kind", "TOOL")
        scores = await self._cli_json(
            "evals", "scores", "get", trace_id, "--target", "trace"
        )
        return {
            "ok": True,
            "trace_id": trace_id,
            "trace": _bounded_evidence(trace, max_len=self._config.sanitize_max_len),
            "tool_spans": _bounded_evidence(
                tool_spans, max_len=self._config.sanitize_max_len
            ),
            "scores": _bounded_evidence(scores, max_len=self._config.sanitize_max_len),
        }

    async def _cli_json(self, *argv: str) -> Any:
        try:
            result = await self._cli.run(*argv)
            return result.json()
        except CliError:
            logger.debug("repair CLI call %s degraded", argv, exc_info=True)
            return None

    async def rule_add(self, args: Mapping[str, Any]) -> dict[str, Any]:
        before = {rule.id for rule in await asyncio.to_thread(self._rules.live)}
        notice = await asyncio.to_thread(self._mailbox.read, self._notice_id)
        tags = (
            [str(tag) for tag in args.get("tags", [])]
            if isinstance(args.get("tags"), list)
            else []
        )
        derived = derive_notice_tags(notice) if notice is not None else ()
        scope = args.get("scope") or (notice.scope_hint if notice is not None else "global")
        rule = await asyncio.to_thread(
            self._rules.add,
            str(args["rule"]),
            str(args["rationale"]),
            source_notice_id=self._notice_id,
            metric=str(args["metric"]) if args.get("metric") is not None else None,
            tags=(*tags, *derived),
            scope=normalize_scope(str(scope)),
        )
        created = rule.id not in before
        if created and rule.id not in self._candidate_ids:
            self._candidate_ids.append(rule.id)
        return {"ok": True, "created": created, "rule": rule.to_json()}

    async def rule_retire(self, args: Mapping[str, Any]) -> dict[str, Any]:
        rule_id = str(args["rule_id"])
        existing = next(
            (rule for rule in await asyncio.to_thread(self._rules.all) if rule.id == rule_id),
            None,
        )
        if existing is None or existing.status != "active":
            return {"ok": False, "error": "managed repair may retire active rules only"}
        try:
            rule = await asyncio.to_thread(
                self._rules.retire,
                rule_id,
                reason=sanitize_text(
                    str(args["reason"]), max_len=self._config.sanitize_max_len
                ),
            )
        except KeyError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "rule": rule.to_json()}

    async def notice_ack(self, args: Mapping[str, Any]) -> dict[str, Any]:
        if error := self._assigned(args):
            return error
        rule_id = str(args["rule_id"])
        if rule_id not in self._candidate_ids:
            return {"ok": False, "error": "ack requires a candidate created in this repair"}
        return await self._ack("candidate", rule_id, str(args.get("note") or "candidate added"))

    async def notice_resolve(self, args: Mapping[str, Any]) -> dict[str, Any]:
        if error := self._assigned(args):
            return error
        resolution = str(args["resolution"])
        if resolution not in {"duplicate", "no_proposal"}:
            return {"ok": False, "error": "invalid repair resolution"}
        existing_rule_id = (
            str(args["existing_rule_id"]) if args.get("existing_rule_id") is not None else None
        )
        if resolution == "duplicate":
            if not existing_rule_id:
                return {"ok": False, "error": "duplicate resolution requires existing_rule_id"}
            existing = next(
                (
                    rule
                    for rule in await asyncio.to_thread(self._rules.live)
                    if rule.id == existing_rule_id
                ),
                None,
            )
            if existing is None:
                return {"ok": False, "error": "duplicate must reference a live rule"}
        self._existing_rule_id = existing_rule_id
        kind = cast(Literal["duplicate", "no_proposal"], resolution)
        return await self._ack(kind, existing_rule_id, str(args["note"]))

    async def _ack(
        self,
        kind: Literal["candidate", "duplicate", "no_proposal"],
        rule_id: str | None,
        note: str,
    ) -> dict[str, Any]:
        try:
            notice = await asyncio.to_thread(
                self._mailbox.acknowledge,
                self._notice_id,
                rule_id=rule_id,
                note=sanitize_text(note, max_len=self._config.sanitize_max_len),
                kind=kind,
            )
        except KeyError as exc:
            return {"ok": False, "error": str(exc)}
        self._resolution = kind
        await asyncio.to_thread(
            self._journal.record,
            {
                "type": "repair_notice_resolved",
                "notice_id": self._notice_id,
                "session_id": notice.session_id,
                "resolution": kind,
                "rule_id": rule_id,
            },
        )
        return {"ok": True, "notice": notice.to_json()}


def _specs(
    schemas: Mapping[str, Mapping[str, Any]], handlers: Mapping[str, Any]
) -> tuple[ToolSpec, ...]:
    return tuple(
        ToolSpec(
            name=name,
            description=str(meta["description"]),
            input_schema=dict(meta["input_schema"]),
            handler=handlers[name],
        )
        for name, meta in schemas.items()
    )


async def _dispatch(
    specs: tuple[ToolSpec, ...], name: str, args: Mapping[str, Any]
) -> dict[str, Any]:
    for spec in specs:
        if spec.name != name:
            continue
        try:
            return await spec.handler(args)
        except Exception as exc:  # noqa: BLE001 - tool errors never escape an agent loop
            logger.debug("tool %s failed", name, exc_info=True)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": False, "error": f"unsupported capability {name!r}"}


def _bounded_evidence(value: Any, *, max_len: int) -> Any:
    def scrub(item: Any) -> Any:
        if isinstance(item, dict):
            return {
                str(key): "[redacted]" if is_sensitive_key(key) else scrub(child)
                for key, child in item.items()
            }
        if isinstance(item, (list, tuple)):
            return [scrub(child) for child in item[:100]]
        if item is None or isinstance(item, (str, int, float, bool)):
            return item
        return str(item)

    scrubbed = scrub(value)
    encoded = json.dumps(scrubbed, sort_keys=True, default=str)
    return scrubbed if len(encoded) <= max_len else {"summary": encoded[:max_len]}
