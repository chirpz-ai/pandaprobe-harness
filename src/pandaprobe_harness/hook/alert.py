"""Builds the System / Trend alerts injected into the agent's next turn.

Two flavors, chosen by the hook from the active conditions:

* ``build_system_alert`` — **critical**: an absolute or relative threshold breach.
* ``build_trend_alert`` — **advisory**: a gradual decline (EWMA crossover) while
  still above the absolute floor.

Both name the diagnostic dump, the flagged traces, and the exact ``pandaprobe``
commands so a downstream agent has a deterministic remediation path.
"""

from __future__ import annotations

from ..config import HarnessConfig
from ..evaluation.metrics import EvalReport, MetricScore

__all__ = ["build_system_alert", "build_trend_alert"]


def _conditions(score: MetricScore) -> list[str]:
    out: list[str] = []
    if score.breached:
        value = "n/a" if score.value is None else f"{score.value:.2f}"
        out.append(f"absolute breach (score {value} < threshold {score.threshold:.2f})")
    if score.relative_breach:
        out.append("relative drop below session baseline")
    if score.trend_declining:
        out.append("declining trend over recent turns")
    if score.percentile_breach:
        out.append("score in the low tail of its recent window")
    return out


def _format_scores(report: EvalReport) -> str:
    lines: list[str] = []
    for score in report.alerting_scores:
        conds = ", ".join(_conditions(score))
        lines.append(f"  - {score.metric}: {conds}")
    return "\n".join(lines)


def _flagged_line(report: EvalReport) -> str:
    if not report.flagged_traces:
        return ""
    ids = ", ".join(report.flagged_traces)
    return f"\nFlagged traces (highest risk): {ids}\n"


def _inspect_commands(report: EvalReport) -> str:
    trace = report.flagged_traces[0] if report.flagged_traces else "<trace-id>"
    return (
        f"       pandaprobe evals scores get {trace} --target trace\n"
        f"       pandaprobe traces spans {trace} --kind TOOL\n"
        f"       pandaprobe evals scores list --target session "
        f"--session-id {report.session_id}"
    )


def build_system_alert(report: EvalReport, config: HarnessConfig) -> str:
    """Render a critical alert for an absolute/relative breach."""

    return f"""\
================================ SYSTEM ALERT ================================
High risk or behavioral deviation detected via platform metrics
(session={report.session_id}, turn={report.turn_index}).

Conditions:
{_format_scores(report)}
{_flagged_line(report)}
A detailed diagnostic trace dump has been written to your workspace at:
  {config.latest_eval_file}

You MUST, BEFORE executing subsequent user steps:
  1. Read the dump:        cat {config.latest_eval_file}
  2. Inspect what went wrong via the PandaProbe CLI, e.g.:
{_inspect_commands(report)}
  3. Reason about your failure trajectory (looping or tool-alignment error).
  4. Record a permanent mitigation directive by appending a new rule to:
       {config.rules_file}
     so this failure mode is prevented on every future run.
=============================================================================
"""


def build_trend_alert(report: EvalReport, config: HarnessConfig) -> str:
    """Render an advisory alert for a gradual decline (no absolute breach yet)."""

    return f"""\
================================ TREND ALERT ================================
Behavioral DRIFT detected: a metric is gradually declining across recent turns
even though it has not yet crossed its absolute threshold
(session={report.session_id}, turn={report.turn_index}).

Conditions:
{_format_scores(report)}
{_flagged_line(report)}
A diagnostic dump has been written to:
  {config.latest_eval_file}

Before continuing, you SHOULD:
  1. Review the recent score trajectory:
       pandaprobe evals scores list --target session --session-id {report.session_id}
  2. Identify what is eroding your performance and consider a preventive rule in:
       {config.rules_file}
============================================================================
"""
