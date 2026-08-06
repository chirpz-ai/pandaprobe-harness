"""Bounded read-only learned guidance for the developer's task agent."""

from __future__ import annotations

import logging

from ..workspace.mailbox import Mailbox
from ..workspace.rules import GLOBAL_SCOPE, Rule, RulesStore

__all__ = ["compose_system_preamble"]

logger = logging.getLogger("pandaprobe_harness.hook")

_HEADER = "===================== PANDAPROBE HARNESS ==========================="
_FOOTER = "===================================================================="


def compose_system_preamble(
    rules: RulesStore,
    mailbox: Mailbox,
    session_id: str,
    *,
    task_hint: str | None = None,
) -> str:
    """Render session-relevant live guidance without administrative instructions."""

    try:
        root = rules.render_root().strip()
        selected = _select(rules, mailbox, session_id, task_hint)
    except Exception:  # noqa: BLE001 - context assembly must never break a task
        logger.debug("failed to render learned guidance", exc_info=True)
        root = "PandaProbe supplies read-only learned guidance for this task."
        selected = []

    guidance = ["Relevant learned guidance:", ""]
    if not selected:
        guidance.append("- No relevant live guidance is available yet.")
    else:
        for rule, turn_index in selected:
            if rule.status == "candidate":
                after = f", added after turn {turn_index}" if turn_index is not None else ""
                guidance.extend(
                    [f"- [candidate {rule.id}{after}]", f"  {rule.rule}"]
                )
            else:
                guidance.extend([f"- [active {rule.id}]", f"  {rule.rule}"])
    return f"{_HEADER}\n{root}\n\n" + "\n".join(guidance) + f"\n{_FOOTER}\n"


def _select(
    rules: RulesStore,
    mailbox: Mailbox,
    session_id: str,
    task_hint: str | None,
) -> list[tuple[Rule, int | None]]:
    live = rules.live()
    source_turns: dict[str, int] = {}
    current_candidates: list[Rule] = []
    for rule in live:
        if rule.status != "candidate" or rule.source_notice_id is None:
            continue
        notice = mailbox.read(rule.source_notice_id)
        if notice is None or notice.session_id != session_id:
            continue
        current_candidates.append(rule)
        source_turns[rule.id] = notice.turn_index
    current_candidates.reverse()  # newest current-session guidance first

    topk = max(0, rules.config.rules_context_topk)
    if task_hint:
        ranked = [
            rule
            for rule, _ in rules.search(
                task_hint, limit=max(1, topk), statuses=("active", "candidate")
            )
        ]
    else:
        global_active = [
            rule for rule in live if rule.status == "active" and rule.scope == GLOBAL_SCOPE
        ]
        remaining = [rule for rule in live if rule not in global_active]
        ranked = global_active + list(reversed(remaining))

    limit = max(1, topk) if current_candidates else topk
    selected: list[Rule] = []
    for rule in [*current_candidates, *ranked]:
        if len(selected) >= limit:
            break
        if rule in selected:
            continue
        selected.append(rule)
    return [(rule, source_turns.get(rule.id)) for rule in selected]
