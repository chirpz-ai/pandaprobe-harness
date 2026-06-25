"""Builds the System Alert injected into the agent's next-turn message queue."""

from __future__ import annotations

from ..config import HarnessConfig
from ..evaluation.metrics import EvalReport

__all__ = ["build_system_alert"]


def _format_breaches(report: EvalReport) -> str:
    lines = []
    for score in report.breached_scores:
        value = "n/a" if score.value is None else f"{score.value:.2f}"
        lines.append(f"  - {score.metric} = {value} (threshold {score.threshold:.2f})")
    return "\n".join(lines)


def build_system_alert(report: EvalReport, config: HarnessConfig) -> str:
    """Render a highly visible, actionable alert for a breached evaluation.

    The text names the exact dump file, the breached metrics and scores, and the
    exact diagnostic CLI commands — so a downstream agent has a deterministic,
    unambiguous remediation path: read the dump, inspect via the CLI, reason
    about the failure, then append a permanent rule to the living rules file.
    """

    dump_path = config.latest_eval_file
    rules_path = config.rules_file
    breaches = _format_breaches(report)

    return f"""\
================================ SYSTEM ALERT ================================
High risk or behavioral deviation detected via platform metrics
(agent_reliability / agent_consistency breach).

Breached metrics for this turn (session={report.session_id}, turn={report.turn_index}):
{breaches}

A detailed diagnostic trace dump has been written to your workspace at:
  {dump_path}

You MUST, BEFORE executing subsequent user steps:
  1. Read the dump:        cat {dump_path}
  2. Inspect what went wrong via the PandaProbe CLI, e.g.:
       pandaprobe evals scores get <trace-id>
       pandaprobe traces get <trace-id>
       pandaprobe traces spans <trace-id> --kind TOOL
  3. Reason about your failure trajectory (looping or tool-alignment error).
  4. Record a permanent mitigation directive by appending a new rule to:
       {rules_path}
     so this failure mode is prevented on every future run.
=============================================================================
"""
