"""Hook-level trend wiring: notice-severity selection and trend dedup."""

from __future__ import annotations

from pathlib import Path

from pandaprobe_harness import HarnessConfig, HarnessFilesystem, Mailbox, RawLoopAdapter
from pandaprobe_harness.hook.core import PandaHarnessHook
from tests.fakes.fake_cli_client import FakeCliClient


def _hook(
    tmp_path: Path, cli: FakeCliClient, **kw: object
) -> tuple[PandaHarnessHook, Mailbox]:
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
    return PandaHarnessHook(cli, config=cfg, filesystem=fs), Mailbox(cfg)


async def _drive(
    hook: PandaHarnessHook, mailbox: Mailbox, cli: FakeCliClient, scores: list[float]
) -> tuple[int, int]:
    """One turn per score; return (critical_count, trend_count) of posted notices."""

    seen: set[str] = set()
    critical = trend = 0
    for idx, score in enumerate(scores):
        cli.set_scores(agent_reliability=score)
        hook.on_turn_end(RawLoopAdapter.make_turn("s", idx))
        await hook.refresh("s")
        for notice in mailbox.pending():
            if notice.id in seen:
                continue
            seen.add(notice.id)
            if notice.severity in {"breach", "relative"}:
                critical += 1
            elif notice.severity == "trend":
                trend += 1
    return critical, trend


async def test_adaptive_relative_breach_is_critical_severity(tmp_path: Path) -> None:
    cli = FakeCliClient(metric_values={"agent_reliability": 0.9})
    hook, mailbox = _hook(tmp_path, cli, adaptive_threshold=True, adaptive_margin_drop=0.15)
    # All scores >= 0.5 (no absolute breach), but the last drops far below baseline.
    critical, _ = await _drive(hook, mailbox, cli, [0.9, 0.9, 0.9, 0.9, 0.6])
    assert critical == 1  # a relative breach is critical severity


async def test_percentile_breach_is_advisory_trend_severity(tmp_path: Path) -> None:
    cli = FakeCliClient(metric_values={"agent_reliability": 0.9})
    hook, mailbox = _hook(tmp_path, cli, percentile_window=5, percentile_floor=0.25)
    critical, trend = await _drive(hook, mailbox, cli, [0.9, 0.85, 0.8, 0.88, 0.6])
    assert critical == 0  # percentile is advisory — never critical
    assert trend >= 1


async def test_trend_dedup_across_multiple_post_crossover_turns(tmp_path: Path) -> None:
    cli = FakeCliClient(metric_values={"agent_reliability": 0.9})
    hook, mailbox = _hook(tmp_path, cli)
    # Continuous decline, all >= 0.5; the trend persists for several turns after
    # the crossover but must post only ONE notice (dedup), not one per turn.
    critical, trend = await _drive(hook, mailbox, cli, [0.80, 0.74, 0.68, 0.62, 0.58, 0.55])
    assert critical == 0
    assert trend == 1
