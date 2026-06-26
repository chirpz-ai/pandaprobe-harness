"""Hook-level trend wiring: alert-flavor selection and trend dedup."""

from __future__ import annotations

from pathlib import Path

from pandaprobe_harness import HarnessConfig, HarnessFilesystem, RawLoopAdapter
from pandaprobe_harness.hook.core import PandaHarnessHook
from tests.fakes.fake_cli_client import FakeCliClient


def _hook(
    tmp_path: Path, cli: FakeCliClient, **kw: object
) -> tuple[PandaHarnessHook, RawLoopAdapter]:
    cfg = HarnessConfig(
        harness_root=tmp_path / "h",
        poll_interval_s=0.0,
        poll_max_attempts=5,
        eval_retry_backoff_s=0.0,
        eval_consistency=False,  # single metric series for deterministic EWMA
        trend_min_samples=4,
        **kw,  # type: ignore[arg-type]
    )
    fs = HarnessFilesystem(cfg)
    fs.provision()
    adapter = RawLoopAdapter()
    return PandaHarnessHook(adapter, cli, config=cfg, filesystem=fs), adapter


async def _drive(
    hook: PandaHarnessHook, adapter: RawLoopAdapter, cli: FakeCliClient, scores: list[float]
) -> tuple[int, int]:
    """One turn per score; return (system_alert_count, trend_alert_count)."""

    system = trend = 0
    for idx, score in enumerate(scores):
        cli.set_scores(agent_reliability=score)
        hook.on_turn_end(adapter.make_turn("s", idx))
        await hook.drain_pending("s")
        for alert in adapter.consume_alerts():
            system += "SYSTEM ALERT" in alert
            trend += "TREND ALERT" in alert
    return system, trend


async def test_adaptive_relative_breach_routes_to_system_alert(tmp_path: Path) -> None:
    cli = FakeCliClient(metric_values={"agent_reliability": 0.9})
    hook, adapter = _hook(tmp_path, cli, adaptive_threshold=True, adaptive_margin_drop=0.15)
    # All scores >= 0.5 (no absolute breach), but the last drops far below baseline.
    system, _ = await _drive(hook, adapter, cli, [0.9, 0.9, 0.9, 0.9, 0.6])
    assert system == 1  # a relative breach is critical → SYSTEM alert


async def test_percentile_breach_routes_to_trend_not_system(tmp_path: Path) -> None:
    cli = FakeCliClient(metric_values={"agent_reliability": 0.9})
    hook, adapter = _hook(tmp_path, cli, percentile_window=5, percentile_floor=0.25)
    system, trend = await _drive(hook, adapter, cli, [0.9, 0.85, 0.8, 0.88, 0.6])
    assert system == 0  # percentile is advisory — never escalates to SYSTEM
    assert trend >= 1


async def test_trend_dedup_across_multiple_post_crossover_turns(tmp_path: Path) -> None:
    cli = FakeCliClient(metric_values={"agent_reliability": 0.9})
    hook, adapter = _hook(tmp_path, cli)
    # Continuous decline, all >= 0.5; the trend persists for several turns after
    # the crossover but must alert only ONCE (dedup), not every turn.
    system, trend = await _drive(hook, adapter, cli, [0.80, 0.74, 0.68, 0.62, 0.58, 0.55])
    assert system == 0
    assert trend == 1
