"""Structured self-heal rules with provenance, lifecycle, dedup, cap, and effectiveness.

Rules live as an append-only JSONL store at ``<harness_root>/rules.jsonl``
(the latest record per rule id wins, so retiring a rule appends an updated
record rather than rewriting the file). The prompt-facing
``harness_rules.md`` is a *rendered artifact*: :meth:`RulesStore.sync_markdown`
regenerates it from the packaged template plus the live rules, so learned
mitigations re-enter the agent's context on every run with their rationale
and source notice attached.

Lifecycle (evidence before trust)::

    candidate ──(validated: metric improved)──▶ active
        │
        └────────(invalidated: no improvement / regressed)──▶ retired
    active ──(agent or regression run retires it)──▶ retired

When ``config.rule_validation`` is on, :meth:`RulesStore.add` records a
**candidate**: it is rendered (clearly labeled as provisional, so it is in
force and therefore measurable) but only a validator verdict promotes it to
``active``. When the flag is off, rules enter ``active`` immediately (the
v0.5 behavior).

This addresses rule rot: duplicate rules are collapsed on normalized text, a
live-rule cap forces agent-driven compaction (retire before add), and
:meth:`RulesStore.effectiveness` gives the reflection cycle before/after
notice counts per rule from the journal.

All methods are synchronous blocking I/O; async callers wrap them in
``asyncio.to_thread``.
"""

from __future__ import annotations

import importlib.resources
import re
import threading
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Literal

from ..config import HarnessConfig
from ._io import append_jsonl, atomic_write_text, read_jsonl
from .journal import Journal
from .mailbox import DiagnosticNotice
from .sanitize import sanitize_text

__all__ = [
    "Rule",
    "RuleStatus",
    "RulesCapError",
    "RulesStore",
    "TrialState",
    "derive_notice_tags",
]

_TEMPLATE_PACKAGE = "pandaprobe_harness.filesystem.templates"
_TEMPLATE_NAME = "harness_rules.md"
#: Everything in the template up to (and including) this line is preserved by
#: the renderer; the active rules are rendered below it.
RULES_MARKER = "<!-- ACTIVE RULES — managed by the harness; use the harness rule tools -->"
#: Heading that separates unproven candidate rules from validated active ones.
PROVISIONAL_HEADING = "### Provisional rules (under evaluation)"
_PROVISIONAL_NOTE = (
    "_The following candidate rules are in force but not yet validated. Treat\n"
    "them as tentative: apply them, but prefer validated rules when they\n"
    "conflict._"
)

RuleStatus = Literal["candidate", "active", "retired"]

#: Journal `notice` events scanned to estimate a candidate's pre-trial baseline.
_BASELINE_WINDOW = 200
#: Bounds applied to rule tags (both derived and agent-supplied).
_TAG_MAX_COUNT = 16
_TAG_MAX_LEN = 48


class RulesCapError(RuntimeError):
    """Raised when adding a rule would exceed the live-rule cap."""


def _as_status(value: object) -> RuleStatus:
    """Forgiving status parse; unknown values degrade to ``active``.

    ``candidate`` must round-trip: a persisted candidate that silently read
    back as ``active`` would self-promote across process restarts.
    """

    if value == "candidate":
        return "candidate"
    if value == "retired":
        return "retired"
    return "active"


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def _clean_tags(values: Iterable[str]) -> tuple[str, ...]:
    """Normalize tags: casefold, strip, dedup (order kept), bounded count/length."""

    tags: list[str] = []
    for value in values:
        cleaned = value.strip().casefold()[:_TAG_MAX_LEN]
        if cleaned and cleaned not in tags:
            tags.append(cleaned)
        if len(tags) >= _TAG_MAX_COUNT:
            break
    return tuple(tags)


def derive_notice_tags(notice: DiagnosticNotice) -> tuple[str, ...]:
    """Retrieval tags for a rule learned from ``notice``.

    Signatures, metric names, per-trace signal names, and the severity — the
    lexical hooks a later query (pending-notice signatures + task hint) can
    match against.
    """

    raw: list[str] = list(notice.signatures)
    raw += [metric.name for metric in notice.metrics]
    for signals in notice.signal_breakdown.values():
        raw += [str(name) for name in signals]
    raw.append(notice.severity)
    return _clean_tags(raw)


#: Lexical retrieval tokenizer: word-ish tokens, casefolded, ≥2 chars. ":"
#: splits signatures ("breach:agent_reliability" → breach + agent_reliability)
#: while "_" keeps metric names whole.
_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(t for t in _TOKEN_RE.findall(text.casefold()) if len(t) >= 2)


def _relevance(rule: Rule, query_tokens: frozenset[str]) -> float:
    """Weighted token overlap, normalized by query size.

    Tag hits count double (tags are curated retrieval hooks); rule/rationale/
    metric text hits count once. A BM25-style IDF adds nothing over a corpus
    capped at ``max_active_rules`` (~50), so plain overlap keeps this stdlib.
    """

    if not query_tokens:
        return 0.0
    tag_tokens: frozenset[str] = frozenset()
    for tag in rule.tags:
        tag_tokens |= _tokenize(tag)
    text_tokens = (
        _tokenize(rule.rule) | _tokenize(rule.rationale) | _tokenize(rule.metric or "")
    )
    tag_hits = len(query_tokens & tag_tokens)
    text_hits = len(query_tokens & (text_tokens - tag_tokens))
    return (2.0 * tag_hits + float(text_hits)) / len(query_tokens)


def _rank(rules: list[Rule], scores: dict[str, float]) -> list[Rule]:
    """Order by score desc, then recency desc, then id asc (stable sorts)."""

    ranked = sorted(rules, key=lambda r: r.id)
    ranked.sort(key=lambda r: r.created_at, reverse=True)
    ranked.sort(key=lambda r: scores[r.id], reverse=True)
    return ranked


def _matches_metric_family(signatures: Sequence[str], metric: str | None) -> bool:
    """Whether any signature belongs to the metric's condition family.

    Signatures look like ``breach:agent_reliability``; the family is "any
    condition on this metric". With no metric, any signature at all matches.
    """

    if metric is None:
        return bool(signatures)
    suffix = f":{metric}"
    return any(signature.endswith(suffix) for signature in signatures)


@dataclass(frozen=True, slots=True)
class TrialState:
    """Validation bookkeeping for a candidate rule (forgiving JSON).

    The baseline is the pre-candidate breach rate for the rule's metric
    family, estimated from recent journal notices at add time; the trial
    counts distinct sessions observed (and breached) while the candidate is
    in force.
    """

    baseline_breached_sessions: int = 0
    baseline_sessions: int = 0
    baseline_window: int = 0
    trial_started_at: str = ""
    observed_sessions: tuple[str, ...] = ()
    breached_sessions: tuple[str, ...] = ()
    replay_attempts: int = 0
    verdict: str = ""

    @property
    def baseline_rate(self) -> float:
        """Pre-candidate breach rate; 1.0 when nothing was journaled (the rule
        was authored in response to a live failure — assume it was firing)."""

        if self.baseline_sessions <= 0:
            return 1.0
        return self.baseline_breached_sessions / self.baseline_sessions

    @property
    def trial_rate(self) -> float:
        if not self.observed_sessions:
            return 0.0
        return len(self.breached_sessions) / len(self.observed_sessions)

    def to_json(self) -> dict[str, Any]:
        return {
            "baseline_breached_sessions": self.baseline_breached_sessions,
            "baseline_sessions": self.baseline_sessions,
            "baseline_window": self.baseline_window,
            "trial_started_at": self.trial_started_at,
            "observed_sessions": list(self.observed_sessions),
            "breached_sessions": list(self.breached_sessions),
            "observed_breaches": len(self.breached_sessions),
            "replay_attempts": self.replay_attempts,
            "verdict": self.verdict,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> TrialState:
        def _int(key: str) -> int:
            raw = data.get(key)
            return raw if isinstance(raw, int) else 0

        def _str_tuple(key: str) -> tuple[str, ...]:
            raw = data.get(key)
            return tuple(str(item) for item in raw) if isinstance(raw, list) else ()

        return cls(
            baseline_breached_sessions=_int("baseline_breached_sessions"),
            baseline_sessions=_int("baseline_sessions"),
            baseline_window=_int("baseline_window"),
            trial_started_at=str(data.get("trial_started_at", "")),
            observed_sessions=_str_tuple("observed_sessions"),
            breached_sessions=_str_tuple("breached_sessions"),
            replay_attempts=_int("replay_attempts"),
            verdict=str(data.get("verdict", "")),
        )


@dataclass(frozen=True, slots=True)
class Rule:
    """One learned operating rule, with provenance and lifecycle."""

    id: str
    created_at: str
    rule: str
    rationale: str
    source_notice_id: str | None = None
    metric: str | None = None
    status: RuleStatus = "active"
    tags: tuple[str, ...] = ()
    trial: TrialState | None = None

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
            "tags": list(self.tags),
            "trial": self.trial.to_json() if self.trial is not None else None,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Rule:
        tags = data.get("tags")
        trial = data.get("trial")
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
            status=_as_status(data.get("status")),
            tags=tuple(str(t) for t in tags) if isinstance(tags, list) else (),
            trial=TrialState.from_json(trial) if isinstance(trial, dict) else None,
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

    def candidates(self) -> list[Rule]:
        return [rule for rule in self.all() if rule.status == "candidate"]

    def live(self) -> list[Rule]:
        """Rules currently in force: validated actives plus provisional candidates."""

        return [rule for rule in self.all() if rule.status in ("active", "candidate")]

    # -- retrieval ----------------------------------------------------------------

    def relevant(self, query: str | None, k: int) -> list[Rule]:
        """Active rules for the context: all globals + the top-``k`` tagged.

        Untagged rules are global (always eligible, exempt from ``k``). Tagged
        rules rank by lexical relevance to ``query``; with no token overlap
        the ranking degrades to recency, so a fresh query never hides the
        whole rule set. ``query=None`` means "no signal" — everything renders.
        """

        active = self.active()
        if query is None:
            return active
        query_tokens = _tokenize(query)
        global_rules = [rule for rule in active if not rule.tags]
        tagged = [rule for rule in active if rule.tags]
        scores = {rule.id: _relevance(rule, query_tokens) for rule in tagged}
        return global_rules + _rank(tagged, scores)[: max(0, k)]

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        statuses: Sequence[RuleStatus] = ("active",),
    ) -> list[tuple[Rule, float]]:
        """Rank rules of the given statuses by relevance to ``query``.

        Pure ranking — no global-rule special-casing — so everything beyond
        the context top-k stays reachable on demand.
        """

        wanted = set(statuses)
        pool = [rule for rule in self.all() if rule.status in wanted]
        query_tokens = _tokenize(query)
        scores = {rule.id: _relevance(rule, query_tokens) for rule in pool}
        return [(rule, scores[rule.id]) for rule in _rank(pool, scores)[: max(0, limit)]]

    # -- writes -----------------------------------------------------------------

    def add(
        self,
        rule: str,
        rationale: str,
        *,
        source_notice_id: str | None = None,
        metric: str | None = None,
        tags: Sequence[str] = (),
    ) -> Rule:
        """Record a new rule; idempotent on normalized text; capped.

        With ``rule_validation`` on, the rule enters as a **candidate** (with
        its pre-trial baseline captured from the journal); otherwise it is
        ``active`` immediately. Raises ``RulesCapError`` at the live-rule cap
        — the agent must retire a rule first (agent-driven compaction, no
        silent eviction).
        """

        max_len = self._config.sanitize_max_len
        clean_rule = sanitize_text(rule, max_len=max_len).strip()
        clean_rationale = sanitize_text(rationale, max_len=max_len).strip()
        if not clean_rule:
            raise ValueError("rule text must not be empty")

        status: RuleStatus = "candidate" if self._config.rule_validation else "active"
        # The baseline scans the journal; do it before taking the store lock.
        trial = self._capture_baseline(metric) if status == "candidate" else None

        with self._lock:
            live = self.live()
            normalized = self._normalize(clean_rule)
            for existing in live:
                if self._normalize(existing.rule) == normalized:
                    return existing
            cap = self._config.max_active_rules
            if cap > 0 and len(live) >= cap:
                raise RulesCapError(
                    f"live-rule cap ({cap}) reached; retire a rule first "
                    "(harness_rule_retire)"
                )
            entry = Rule(
                id=Rule.new_id(),
                created_at=_utcnow_iso(),
                rule=clean_rule,
                rationale=clean_rationale,
                source_notice_id=source_notice_id,
                metric=metric,
                status=status,
                tags=_clean_tags(tags),
                trial=trial,
            )
            append_jsonl(self._config.rules_store_file, entry.to_json())
            self._sync_markdown_locked()
        if self._journal is not None:
            self._journal.record({"type": "rule_add", **entry.to_json()})
        return entry

    def retire(
        self,
        rule_id: str,
        *,
        reason: str | None = None,
        trial: TrialState | None = None,
    ) -> Rule:
        """Retire a live (active or candidate) rule. Raises ``KeyError`` otherwise.

        ``trial`` lets a validator persist its verdict-stamped bookkeeping on
        the retired record, so ``harness_rule_status`` can explain why.
        """

        with self._lock:
            current = {rule.id: rule for rule in self.all()}
            rule = current.get(rule_id)
            if rule is None or rule.status == "retired":
                raise KeyError(f"rule {rule_id!r} is not a live rule")
            retired = replace(
                rule, status="retired", trial=trial if trial is not None else rule.trial
            )
            append_jsonl(self._config.rules_store_file, retired.to_json())
            self._sync_markdown_locked()
        if self._journal is not None:
            event: dict[str, Any] = {"type": "rule_retire", "id": rule_id}
            if reason:
                event["reason"] = reason
            self._journal.record(event)
        return retired

    def promote(
        self,
        rule_id: str,
        *,
        reason: str = "",
        validator: str = "",
        trial: TrialState | None = None,
    ) -> Rule:
        """Promote a candidate to ``active``. Raises ``KeyError`` unless a candidate."""

        with self._lock:
            current = {rule.id: rule for rule in self.all()}
            rule = current.get(rule_id)
            if rule is None or rule.status != "candidate":
                raise KeyError(f"rule {rule_id!r} is not a candidate rule")
            promoted = replace(
                rule, status="active", trial=trial if trial is not None else rule.trial
            )
            append_jsonl(self._config.rules_store_file, promoted.to_json())
            self._sync_markdown_locked()
        if self._journal is not None:
            self._journal.record(
                {
                    "type": "rule_promote",
                    "reason": reason,
                    "validator": validator,
                    **promoted.to_json(),
                }
            )
        return promoted

    def update_trial(
        self, rule_id: str, mutate: Callable[[TrialState], TrialState]
    ) -> Rule:
        """Atomically update a candidate's trial bookkeeping (no journal event).

        ``mutate`` receives the FRESH trial state read under the store lock
        and returns the new one — a read-modify-write over a caller-held
        snapshot would let concurrent observers silently drop trial evidence.
        Returning the same object signals "no change" (nothing is appended).
        Trial updates are not journaled: one per observed session would drown
        the cross-run memory in bookkeeping noise.
        """

        with self._lock:
            current = {rule.id: rule for rule in self.all()}
            rule = current.get(rule_id)
            if rule is None or rule.status != "candidate":
                raise KeyError(f"rule {rule_id!r} is not a candidate rule")
            existing = rule.trial if rule.trial is not None else TrialState()
            trial = mutate(existing)
            if trial == existing and rule.trial is not None:
                return rule
            updated = replace(rule, trial=trial)
            append_jsonl(self._config.rules_store_file, updated.to_json())
        return updated

    def _capture_baseline(self, metric: str | None) -> TrialState:
        """Pre-candidate breach rate for the metric family, from the journal.

        The denominator is every session that appears in the recent ``notice``
        *and* ``recovery`` events — an approximation biased toward sessions
        that had incidents (fully-healthy sessions never journal either
        event), which biases the baseline rate HIGH. That makes the forward
        trial lenient about promotion, never about retirement; replay remains
        the strong evidence path.
        """

        started = _utcnow_iso()
        if self._journal is None:
            return TrialState(trial_started_at=started)
        events = self._journal.recent(
            limit=_BASELINE_WINDOW, types=("notice", "recovery")
        )
        sessions: set[str] = set()
        breached: set[str] = set()
        for event in events:
            session_id = str(event.get("session_id") or "")
            if not session_id:
                continue
            sessions.add(session_id)
            if event.get("type") != "notice":
                continue
            raw = event.get("signatures")
            signatures = [str(s) for s in raw] if isinstance(raw, list) else []
            if _matches_metric_family(signatures, metric):
                breached.add(session_id)
        return TrialState(
            baseline_breached_sessions=len(breached),
            baseline_sessions=len(sessions),
            baseline_window=len(events),
            trial_started_at=started,
        )

    # -- rendering ----------------------------------------------------------------

    def render_markdown(self, *, query: str | None = None) -> str:
        """The ``harness_rules.md`` content: template preamble + live rules.

        Validated ``active`` rules render first; ``candidate`` rules render
        under a clearly-labeled provisional section — they must be in force
        to be measurable, but every reader can see they are unproven.

        With ``rule_retrieval`` on and a ``query``, only global rules plus
        the top-k relevant tagged rules render (a note points at the rest);
        candidates always render in full — retrieval must never starve a
        trial. The on-disk artifact (``sync_markdown``) is always the full
        render.
        """

        template = self._template()
        marker_at = template.find(RULES_MARKER)
        if marker_at >= 0:
            head = template[: marker_at + len(RULES_MARKER)]
        else:  # template without a marker: append at the end
            head = template.rstrip() + "\n\n" + RULES_MARKER
        lines = [head, ""]
        active = self.active()
        selected = active
        if self._config.rule_retrieval and query is not None:
            selected = self.relevant(query, self._config.rules_context_topk)
        omitted = len(active) - len(selected)
        candidates = self.candidates()
        if not active and not candidates:
            lines.append("_No learned rules yet._")
        for rule in selected:
            lines.extend(self._rule_bullet(rule))
        if omitted > 0:
            lines.append("")
            lines.append(
                f"_({omitted} more active rule(s) available — use "
                "harness_rules_search / harness_rules_list.)_"
            )
        if candidates:
            if selected or omitted:
                lines.append("")
            lines.append(PROVISIONAL_HEADING)
            lines.append("")
            lines.append(_PROVISIONAL_NOTE)
            lines.append("")
            for rule in candidates:
                lines.extend(self._rule_bullet(rule, provisional=True))
        return "\n".join(lines) + "\n"

    def _rule_bullet(self, rule: Rule, *, provisional: bool = False) -> list[str]:
        provenance = f", from notice {rule.source_notice_id}" if rule.source_notice_id else ""
        trial_note = ""
        if provisional and rule.trial is not None:
            observed = len(rule.trial.observed_sessions)
            needed = self._config.rule_trial_min_sessions
            trial_note = f"; trial: {observed}/{needed} sessions observed"
        label = f"**{rule.id}** (candidate)" if provisional else f"**{rule.id}**"
        return [
            f"- {label}: {rule.rule}",
            f"  - rationale: {rule.rationale} (added {rule.created_at}{provenance}{trial_note})",
        ]

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
