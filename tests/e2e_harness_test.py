"""End-to-end self-healing scenario.

Simulates an agent stuck in an infinite-repetition / inconsistent-session
failure, and verifies the full harness loop:

  failure turn -> hook detects metric breach -> dumps latest_eval.json ->
  injects SYSTEM ALERT -> agent uses its sandbox shell to query the CLI ->
  appends a mitigation rule to harness_rules.md -> subsequent turns pass with
  no further alerts.
"""

from __future__ import annotations

import os
from pathlib import Path

from pandaprobe_harness import (
    HarnessConfig,
    HarnessFilesystem,
    PandaHarnessHook,
    RawLoopAdapter,
    RestrictedShellTool,
    ShellPolicy,
)
from tests.fakes.fake_cli_client import FakeCliClient
from tests.fakes.mock_agent import MockLLMAgent

SESSION = "s-e2e-1"


async def test_self_healing_loop_converges(
    config: HarnessConfig, pandaprobe_path: dict[str, str]
) -> None:
    # --- Provision the diagnostic filesystem (Component 3) ------------------
    fs = HarnessFilesystem(config)
    fs.provision()
    assert config.traces_dir.is_dir()
    assert config.rules_file.exists()
    assert "(self-heal)" not in fs.read_rules()  # no learned rules yet

    # --- Wire the harness ----------------------------------------------------
    cli = FakeCliClient(
        metric_values={"agent_reliability": 0.30, "agent_consistency": 0.40},
        metric_metadata={
            "agent_reliability": {
                "flagged_traces": ["trace-1"],
                "per_trace_signals": {"trace-1": {"loop_detection": 0.1, "tool_correctness": 0.2}},
            }
        },
    )
    adapter = RawLoopAdapter()
    hook = PandaHarnessHook(adapter, cli, config=config, filesystem=fs)
    adapter.register(hook)

    shell = RestrictedShellTool(
        ShellPolicy(workdir=config.harness_root), env=pandaprobe_path
    )
    agent = MockLLMAgent(session_id=SESSION, shell=shell, filesystem=fs, config=config)

    alerts_injected = 0

    async def run_turn() -> None:
        """One iteration: drain prior eval, feed alerts, act, schedule eval."""
        nonlocal alerts_injected
        await hook.drain_pending(SESSION)
        alerts_injected += len(adapter.pending_alerts)
        alerts = adapter.consume_alerts()
        raw = await agent.take_turn(alerts)
        # Once the agent has healed, its corrected behaviour is reflected by
        # improved platform scores on the next evaluation.
        if agent.healed:
            cli.set_scores(agent_reliability=0.92, agent_consistency=0.88)
        hook.on_turn_end(raw)

    # --- Turn 1: the failure (identical repeated tool call) -----------------
    await run_turn()
    assert agent.actions == ["charge_payment"]

    # --- Turn 2: alert from turn 1 is drained + injected, agent self-heals --
    await run_turn()
    # The breach dump was written with both breached metrics + flagged traces.
    dump = fs.read_latest_eval()
    assert dump["any_breach"] is True
    assert {s["metric"] for s in dump["scores"]} == {
        "agent_reliability",
        "agent_consistency",
    }
    reliability = next(s for s in dump["scores"] if s["metric"] == "agent_reliability")
    assert reliability["flagged_traces"] == ["trace-1"]
    assert dump["flagged_traces"] == ["trace-1"]
    # The four trace-level signals are surfaced in the dump (no extra CLI call).
    assert dump["signal_breakdown"]["trace-1"] == {"loop_detection": 0.1, "tool_correctness": 0.2}

    # The agent used its restricted shell to inspect the dump and the CLI.
    assert any("cat" in c and "latest_eval.json" in c for c in agent.shell_commands)
    assert any(c.startswith("pandaprobe evals scores get") for c in agent.shell_commands)

    # It recorded a permanent mitigation rule.
    rules = fs.read_rules()
    assert "(self-heal)" in rules
    assert "payment tool twice" in rules
    assert agent.actions[-1] == "diagnose"

    # The learned rule is now re-injectable into the agent's startup context,
    # closing the self-healing loop (it takes effect on every subsequent run).
    preamble = hook.startup_context()
    assert "payment tool twice" in preamble
    assert "PANDAPROBE HARNESS RULES" in preamble

    # --- Turn 3: corrected action, evaluation now passes --------------------
    await run_turn()
    assert agent.actions[-1] == "verified_payment_then_charge"

    # --- Turn 4: drain turn-3 eval; confirm no further alert ----------------
    await run_turn()

    # --- Convergence assertions ---------------------------------------------
    assert alerts_injected == 1, "exactly one alert should have been injected"
    assert rules.count("(self-heal)") == 1, "rules file grew by exactly one rule"
    assert agent.healed
    # final eval did not breach
    final = await hook.drain_pending(SESSION)
    assert final is None or not final.any_breach


async def test_trend_alert_fires_on_gradual_decline(tmp_path: Path) -> None:
    """A metric that drifts down over turns — never crossing the absolute floor —
    must still raise a single advisory TREND alert once the EWMA crosses over."""

    cfg = HarnessConfig(
        harness_root=tmp_path / "harness",
        poll_interval_s=0.0,
        poll_max_attempts=5,
        eval_retry_backoff_s=0.0,
        eval_consistency=False,  # isolate a single metric series
        trend_min_samples=4,
        trend_margin_cross=0.05,
    )
    fs = HarnessFilesystem(cfg)
    fs.provision()
    cli = FakeCliClient(metric_values={"agent_reliability": 0.80})
    adapter = RawLoopAdapter()
    hook = PandaHarnessHook(adapter, cli, config=cfg, filesystem=fs)

    session = "s-trend"
    trend_alerts = 0
    system_alerts = 0
    # All scores stay >= 0.5 (no absolute breach) but trend downward, and the
    # decline PERSISTS for several turns past the crossover — so this also proves
    # the trend dedup gate (one alert, not one per declining turn).
    for idx, score in enumerate((0.80, 0.74, 0.68, 0.62, 0.58, 0.55)):
        cli.set_scores(agent_reliability=score)
        hook.on_turn_end(adapter.make_turn(session, idx))
        await hook.drain_pending(session)
        for alert in adapter.consume_alerts():
            trend_alerts += "TREND ALERT" in alert
            system_alerts += "SYSTEM ALERT" in alert

    assert system_alerts == 0, "no absolute breach should occur (all scores >= 0.5)"
    assert trend_alerts == 1, "exactly one advisory trend alert despite a persistent decline"


def test_fake_binary_is_executable() -> None:
    fake = Path(__file__).parent / "bin" / "fake_pandaprobe"
    assert fake.exists()
    assert os.access(fake, os.X_OK), "fake CLI must carry the executable bit"
    first_line = fake.read_text(encoding="utf-8").splitlines()[0]
    assert first_line.startswith("#!") and "python" in first_line
