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
from ..workspace.rules import (
    RulesStore,
    derive_notice_tags,
    normalize_scope,
    validate_scope,
)
from ..workspace.sanitize import is_sensitive_key, sanitize_text
from .spec import ToolSpec

__all__ = ["REPAIR_OP_SCHEMAS", "TASK_OP_SCHEMAS", "RepairToolset", "TaskToolset"]

logger = logging.getLogger("pandaprobe_harness.agent_tools")

TASK_OP_SCHEMAS: dict[str, dict[str, Any]] = {
    "harness_rules_read": {
        "description": "Read one learned-rule scope. Defaults to scoped.",
        "input_schema": {
            "type": "object",
            "properties": {"scope": {"type": "string"}},
            "required": [],
        },
    },
    "harness_rules_search": {
        "description": "Search live learned rules and return bounded references/snippets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    "harness_rules_list": {
        "description": "Load the canonical harness_guide.md and its compact live-scope index.",
        "input_schema": {
            "type": "object",
            "properties": {},
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
                "applicability": {
                    "type": "string",
                    "enum": ["global", "topical", "task"],
                },
            },
            "required": ["rule", "rationale"],
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
                    "enum": [
                        "duplicate", "already_covered", "no_proposal", "unactionable"
                    ],
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
        scope = (
            validate_scope(args["scope"])
            if args.get("scope") is not None
            else "scoped"
        )
        content = await asyncio.to_thread(self._rules.read_scope, scope)
        return {"ok": True, "scope": scope, "path": f"rules/{scope}.md", "content": content}

    async def rules_search(self, args: Mapping[str, Any]) -> dict[str, Any]:
        query = str(args["query"])
        try:
            limit = int(args.get("limit", 10))
        except (TypeError, ValueError):
            limit = 10
        limit = min(50, max(1, limit))
        results = await asyncio.to_thread(
            self._rules.search, query, limit=limit, statuses=("active", "candidate")
        )
        return {
            "ok": True,
            "rules": [
                {
                    "id": rule.id,
                    "scope": rule.scope,
                    "status": rule.status,
                    "snippet": _snippet(rule.rule),
                    "score": score,
                }
                for rule, score in results
            ],
        }

    async def rules_list(self, args: Mapping[str, Any]) -> dict[str, Any]:
        del args
        content, scopes = await asyncio.gather(
            asyncio.to_thread(self._rules.render_root),
            asyncio.to_thread(self._rules.scope_index),
        )
        return {
            "ok": True,
            "path": "harness_guide.md",
            "content": content,
            "scopes": scopes,
        }

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
            return {
                "ok": True,
                "rule": {
                    "id": rule.id,
                    "scope": rule.scope,
                    "status": rule.status,
                    "applicability": rule.applicability,
                },
                "lifecycle": lifecycle,
            }
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
    """One-episode administrative surface used only by managed repair."""

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
        notice_ids: tuple[str, ...] = (),
        episode_id: str = "",
        recommended_scope: str = "scoped",
        scope_hints: tuple[dict[str, Any], ...] = (),
        generic_scopes: tuple[str, ...] = (),
    ) -> None:
        super().__init__(config=config, rules=rules)
        self._cli = cli
        self._mailbox = mailbox
        self._journal = journal
        self._notice_id = notice_id
        self._notice_ids = notice_ids or (notice_id,)
        self._episode_id = episode_id
        self._recommended_scope = normalize_scope(recommended_scope)
        self._scope_hints = scope_hints
        self._generic_scopes = frozenset(normalize_scope(value) for value in generic_scopes)
        self._allowed_trace_ids = frozenset(allowed_trace_ids)
        self._candidate_ids: list[str] = []
        self._resolution: str | None = None
        self._existing_rule_id: str | None = None
        self._considered_rule_ids: list[str] = []
        self._selected_scope: str | None = None
        self._suppression_reason: str | None = None
        handlers = {
            "harness_notice_read": self.notice_read,
            "harness_trace_inspect": self.trace_inspect,
            "harness_rules_read": self.rules_read,
            "harness_rules_search": self.rules_search,
            "harness_rules_list": self.rules_list,
            "harness_rule_status": self.rule_status,
            "harness_rule_add": self.rule_add,
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
    def considered_rule_ids(self) -> tuple[str, ...]:
        return tuple(self._considered_rule_ids)

    @property
    def selected_scope(self) -> str | None:
        return self._selected_scope

    @property
    def suppression_reason(self) -> str | None:
        return self._suppression_reason

    @property
    def resolved(self) -> bool:
        return self._resolution is not None

    def specs(self) -> tuple[ToolSpec, ...]:
        return self._specs

    async def call(self, name: str, args: Mapping[str, Any]) -> dict[str, Any]:
        return await _dispatch(self._specs, name, args)

    def _assigned(self, args: Mapping[str, Any]) -> dict[str, Any] | None:
        supplied = str(args.get("notice_id", ""))
        if supplied not in self._notice_ids:
            return {"ok": False, "error": "repair may access only its assigned episode"}
        return None

    async def rules_search(self, args: Mapping[str, Any]) -> dict[str, Any]:
        result = await super().rules_search(args)
        self._remember_considered(result)
        return result

    async def rules_read(self, args: Mapping[str, Any]) -> dict[str, Any]:
        result = await super().rules_read(args)
        if result.get("ok"):
            scope = str(result.get("scope") or "")
            rules = await asyncio.to_thread(self._rules.by_scope, scope)
            self._remember_ids(rule.id for rule in rules)
        return result

    def _remember_considered(self, result: Mapping[str, Any]) -> None:
        raw_rules = result.get("rules")
        if not isinstance(raw_rules, list):
            return
        self._remember_ids(
            str(rule.get("id"))
            for rule in raw_rules
            if isinstance(rule, dict) and rule.get("id")
        )

    def _remember_ids(self, values: Any) -> None:
        for value in values:
            if value not in self._considered_rule_ids:
                self._considered_rule_ids.append(value)

    async def notice_read(self, args: Mapping[str, Any]) -> dict[str, Any]:
        if error := self._assigned(args):
            return error
        notice_id = str(args["notice_id"])
        notice = await asyncio.to_thread(self._mailbox.read, notice_id)
        if notice is None:
            return {"ok": False, "error": f"no notice {notice_id!r}"}
        dump = None
        if notice.dump_path:
            dump = await asyncio.to_thread(load_json, Path(notice.dump_path))
        return {
            "ok": True,
            "notice": notice.to_json(),
            "episode_id": self._episode_id,
            "notice_ids": list(self._notice_ids),
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
        if self._candidate_ids:
            return {"ok": False, "error": "a repair episode may create at most one candidate"}
        notices = [
            notice
            for notice in await asyncio.gather(
                *(asyncio.to_thread(self._mailbox.read, value) for value in self._notice_ids)
            )
            if notice is not None
        ]
        tags = (
            [str(tag) for tag in args.get("tags", [])]
            if isinstance(args.get("tags"), list)
            else []
        )
        derived = tuple(
            dict.fromkeys(tag for notice in notices for tag in derive_notice_tags(notice))
        )
        failure_signatures = tuple(
            dict.fromkeys(signature for notice in notices for signature in notice.signatures)
        )
        requested_scope = normalize_scope(
            str(args.get("scope") or self._recommended_scope)
        )
        if (
            requested_scope in self._generic_scopes
            and self._recommended_scope not in {"global", "scoped"}
        ):
            requested_scope = self._recommended_scope
        applicability = str(
            args.get("applicability")
            or (notices[0].applicability_hint if notices else "topical")
        )
        if applicability == "global":
            requested_scope = "global"
        self._selected_scope = requested_scope
        scope_description = next(
            (
                str(hint.get("description") or "")
                for hint in self._scope_hints
                if normalize_scope(str(hint.get("key") or "")) == requested_scope
            ),
            None,
        )
        considered = await asyncio.to_thread(
            self._rules.covering_rules,
            str(args["rule"]),
            scope=requested_scope,
            tags=(*tags, *derived),
            failure_signatures=failure_signatures,
        )
        self._remember_ids(rule.id for rule, _ in considered)
        if considered:
            existing, reason = considered[0]
            self._existing_rule_id = existing.id
            self._suppression_reason = reason
            resolution = "already_covered" if existing.status == "candidate" else "duplicate"
            return {
                "ok": True,
                "created": False,
                "suppressed": True,
                "suppression_reason": reason,
                "recommended_resolution": resolution,
                "existing_rule": {
                    "id": existing.id,
                    "scope": existing.scope,
                    "status": existing.status,
                },
            }
        rule, created = await asyncio.to_thread(
            self._rules.add_with_result,
            str(args["rule"]),
            str(args["rationale"]),
            source_notice_id=self._notice_id,
            metric=str(args["metric"]) if args.get("metric") is not None else None,
            tags=(*tags, *derived),
            scope=requested_scope,
            applicability=(
                cast(
                    Literal["global", "topical", "task"], applicability
                )
                if applicability in {"global", "topical", "task"}
                else None
            ),
            failure_signatures=failure_signatures,
            scope_description=scope_description,
            suppress_similar=True,
        )
        if created and rule.id not in self._candidate_ids:
            self._candidate_ids.append(rule.id)
        elif not created:
            self._existing_rule_id = rule.id
            self._suppression_reason = "concurrent_or_exact_duplicate"
            self._remember_ids((rule.id,))
            resolution = "already_covered" if rule.status == "candidate" else "duplicate"
            return {
                "ok": True,
                "created": False,
                "suppressed": True,
                "suppression_reason": self._suppression_reason,
                "recommended_resolution": resolution,
                "existing_rule": {
                    "id": rule.id,
                    "scope": rule.scope,
                    "status": rule.status,
                },
            }
        return {"ok": True, "created": True, "rule": rule.to_json()}

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
        if resolution not in {"duplicate", "already_covered", "no_proposal", "unactionable"}:
            return {"ok": False, "error": "invalid repair resolution"}
        existing_rule_id = (
            str(args["existing_rule_id"]) if args.get("existing_rule_id") is not None else None
        )
        if resolution in {"duplicate", "already_covered"}:
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
                return {"ok": False, "error": "coverage resolution must reference a live rule"}
            if resolution == "duplicate" and existing.status != "active":
                return {"ok": False, "error": "duplicate must reference an active rule"}
            if resolution == "already_covered" and existing.status != "candidate":
                return {"ok": False, "error": "already_covered must reference a candidate"}
        self._existing_rule_id = existing_rule_id
        kind = cast(
            Literal["duplicate", "already_covered", "no_proposal", "unactionable"],
            resolution,
        )
        return await self._ack(kind, existing_rule_id, str(args["note"]))

    async def _ack(
        self,
        kind: Literal[
            "candidate", "duplicate", "already_covered", "no_proposal", "unactionable"
        ],
        rule_id: str | None,
        note: str,
    ) -> dict[str, Any]:
        try:
            notices = await asyncio.to_thread(
                self._mailbox.acknowledge_many,
                self._notice_ids,
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
                "repair_episode_id": self._episode_id,
                "notice_id": self._notice_id,
                "notice_ids": list(self._notice_ids),
                "session_id": notices[0].session_id,
                "resolution": kind,
                "rule_id": rule_id,
                "recommended_scope": self._recommended_scope,
                "selected_scope": self._selected_scope,
                "considered_rule_ids": list(self._considered_rule_ids),
                "candidate_suppression_reason": self._suppression_reason,
            },
        )
        return {"ok": True, "notices": [notice.to_json() for notice in notices]}


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


def _snippet(value: str, *, limit: int = 240) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= limit else compact[: limit - 1].rstrip() + "…"
