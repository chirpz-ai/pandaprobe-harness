"""Structured self-heal rules with provenance, dedup, cap, and effectiveness.

Rules live as an append-only JSONL store at ``<harness_root>/rules.jsonl``
(the latest record per rule id wins, so retiring a rule appends an updated
record rather than rewriting the file). The prompt-facing
``harness_rules.md`` is a *rendered artifact*: :meth:`RulesStore.sync_markdown`
regenerates it from the packaged template plus the active rules, so learned
mitigations re-enter the agent's context on every run with their rationale
and source notice attached.

This addresses rule rot: duplicate rules are collapsed on normalized text, an
active-rule cap forces agent-driven compaction (retire before add), and
:meth:`RulesStore.effectiveness` gives the reflection cycle before/after
notice counts per rule from the journal.

All methods are synchronous blocking I/O; async callers wrap them in
``asyncio.to_thread``.
"""

from __future__ import annotations

import importlib.resources
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from ..config import HarnessConfig
from ._io import append_jsonl, atomic_write_text, read_jsonl
from .journal import Journal
from .sanitize import sanitize_text

__all__ = ["Rule", "RulesCapError", "RulesStore"]

_TEMPLATE_PACKAGE = "pandaprobe_harness.filesystem.templates"
_TEMPLATE_NAME = "harness_rules.md"
#: Everything in the template up to (and including) this line is preserved by
#: the renderer; the active rules are rendered below it.
RULES_MARKER = "<!-- ACTIVE RULES — managed by the harness; use the harness rule tools -->"


class RulesCapError(RuntimeError):
    """Raised when adding a rule would exceed the active-rule cap."""


@dataclass(frozen=True, slots=True)
class Rule:
    """One learned operating rule, with provenance."""

    id: str
    created_at: str
    rule: str
    rationale: str
    source_notice_id: str | None = None
    metric: str | None = None
    status: Literal["active", "retired"] = "active"

    @staticmethod
    def new_id() -> str:
        return f"r-{uuid.uuid4().hex[:10]}"

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "created_at": self.created_at,
            "rule": self.rule,
            "rationale": self.rationale,
            "source_notice_id": self.source_notice_id,
            "metric": self.metric,
            "status": self.status,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Rule:
        return cls(
            id=str(data.get("id", "")),
            created_at=str(data.get("created_at", "")),
            rule=str(data.get("rule", "")),
            rationale=str(data.get("rationale", "")),
            source_notice_id=(
                str(data["source_notice_id"])
                if data.get("source_notice_id") is not None
                else None
            ),
            metric=str(data["metric"]) if data.get("metric") is not None else None,
            status="retired" if data.get("status") == "retired" else "active",
        )


class RulesStore:
    """Structured rules + rendered ``harness_rules.md``."""

    def __init__(self, config: HarnessConfig, *, journal: Journal | None = None) -> None:
        self._config = config
        self._journal = journal
        self._lock = threading.Lock()

    # -- reads ------------------------------------------------------------------

    def all(self) -> list[Rule]:
        """Latest record per rule id, ordered by creation time."""

        latest: dict[str, Rule] = {}
        for record in read_jsonl(self._config.rules_store_file):
            rule = Rule.from_json(record)
            if rule.id:
                latest[rule.id] = rule
        return sorted(latest.values(), key=lambda r: (r.created_at, r.id))

    def active(self) -> list[Rule]:
        return [rule for rule in self.all() if rule.status == "active"]

    # -- writes -----------------------------------------------------------------

    def add(
        self,
        rule: str,
        rationale: str,
        *,
        source_notice_id: str | None = None,
        metric: str | None = None,
    ) -> Rule:
        """Record a new rule; idempotent on normalized text; capped.

        Raises ``RulesCapError`` at the active-rule cap — the agent must
        retire a rule first (agent-driven compaction, no silent eviction).
        """

        max_len = self._config.sanitize_max_len
        clean_rule = sanitize_text(rule, max_len=max_len).strip()
        clean_rationale = sanitize_text(rationale, max_len=max_len).strip()
        if not clean_rule:
            raise ValueError("rule text must not be empty")

        with self._lock:
            active = self.active()
            normalized = self._normalize(clean_rule)
            for existing in active:
                if self._normalize(existing.rule) == normalized:
                    return existing
            cap = self._config.max_active_rules
            if cap > 0 and len(active) >= cap:
                raise RulesCapError(
                    f"active-rule cap ({cap}) reached; retire a rule first "
                    "(harness_rule_retire)"
                )
            entry = Rule(
                id=Rule.new_id(),
                created_at=datetime.now(UTC).isoformat(),
                rule=clean_rule,
                rationale=clean_rationale,
                source_notice_id=source_notice_id,
                metric=metric,
            )
            append_jsonl(self._config.rules_store_file, entry.to_json())
            self._sync_markdown_locked()
        if self._journal is not None:
            self._journal.record({"type": "rule_add", **entry.to_json()})
        return entry

    def retire(self, rule_id: str) -> Rule:
        """Retire an active rule. Raises ``KeyError`` if not active."""

        with self._lock:
            current = {rule.id: rule for rule in self.all()}
            rule = current.get(rule_id)
            if rule is None or rule.status != "active":
                raise KeyError(f"rule {rule_id!r} is not an active rule")
            retired = Rule(
                id=rule.id,
                created_at=rule.created_at,
                rule=rule.rule,
                rationale=rule.rationale,
                source_notice_id=rule.source_notice_id,
                metric=rule.metric,
                status="retired",
            )
            append_jsonl(self._config.rules_store_file, retired.to_json())
            self._sync_markdown_locked()
        if self._journal is not None:
            self._journal.record({"type": "rule_retire", "id": rule_id})
        return retired

    # -- rendering ----------------------------------------------------------------

    def render_markdown(self) -> str:
        """The full ``harness_rules.md`` content: template preamble + active rules."""

        template = self._template()
        marker_at = template.find(RULES_MARKER)
        if marker_at >= 0:
            head = template[: marker_at + len(RULES_MARKER)]
        else:  # template without a marker: append at the end
            head = template.rstrip() + "\n\n" + RULES_MARKER
        lines = [head, ""]
        active = self.active()
        if not active:
            lines.append("_No learned rules yet._")
        for rule in active:
            provenance = f", from notice {rule.source_notice_id}" if rule.source_notice_id else ""
            lines.append(f"- **{rule.id}**: {rule.rule}")
            lines.append(f"  - rationale: {rule.rationale} (added {rule.created_at}{provenance})")
        return "\n".join(lines) + "\n"

    def sync_markdown(self) -> None:
        """Atomically regenerate the prompt-facing ``harness_rules.md``."""

        with self._lock:
            self._sync_markdown_locked()

    def _sync_markdown_locked(self) -> None:
        atomic_write_text(self._config.rules_file, self.render_markdown())

    @staticmethod
    def _template() -> str:
        resource = importlib.resources.files(_TEMPLATE_PACKAGE) / _TEMPLATE_NAME
        return resource.read_text(encoding="utf-8")

    # -- analysis -----------------------------------------------------------------

    def effectiveness(self) -> dict[str, dict[str, Any]]:
        """Per-rule notice counts before/after the rule was added.

        Computed from the journal's ``notice`` events for the rule's metric
        (all notices when the rule has no metric). Raw counts — the agent's
        reflection cycle interprets them.
        """

        if self._journal is None:
            return {}
        all_notices = self._journal.recent(limit=0, types=("notice",))
        result: dict[str, dict[str, Any]] = {}
        for rule in self.all():
            if rule.metric:
                events = self._journal.notices_for(rule.metric)
            else:
                events = all_notices
            before = sum(1 for e in events if str(e.get("ts", "")) < rule.created_at)
            after = sum(1 for e in events if str(e.get("ts", "")) >= rule.created_at)
            result[rule.id] = {
                "metric": rule.metric,
                "status": rule.status,
                "created_at": rule.created_at,
                "notices_before": before,
                "notices_after": after,
            }
        return result

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.casefold().split()).rstrip(".!;,")
