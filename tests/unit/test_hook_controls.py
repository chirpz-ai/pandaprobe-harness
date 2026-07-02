"""Unit tests for the hook's cost/latency controls.

Sampling, per-session rate limiting, the process-wide eval budget, the global
concurrency semaphore, and observe-only (shadow) mode.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pandaprobe_harness import HarnessConfig, HarnessFilesystem, RawLoopAdapter
from pandaprobe_harness.hook.core import PandaHarnessHook
from tests.fakes.fake_cli_client import FakeCliClient

BREACHING = {"agent_reliability": 0.2, "agent_consistency": 0.2}


def _make(
    tmp_path: Path, cli: FakeCliClient, **kw: object
) -> tuple[PandaHarnessHook, HarnessConfig]:
    cfg = HarnessConfig(
        harness_root=tmp_path / "h",
        poll_interval_s=0.0,
        poll_max_attempts=5,
        eval_retry_backoff_s=0.0,
        **kw,  # type: ignore[arg-type]
    )
    fs = HarnessFilesystem(cfg)
    fs.provision()
    return PandaHarnessHook(cli, config=cfg, filesystem=fs), cfg


async def test_sampling_admits_every_nth_turn(tmp_path: Path) -> None:
    cli = FakeCliClient()
    hook, _cfg = _make(tmp_path, cli, eval_sample_every=3)

    for i in range(1, 8):
        hook.on_turn_end(RawLoopAdapter.make_turn("s", i))
        await hook.refresh("s")

    # Turns 1, 4, and 7 are admitted; the rest are sampled out.
    assert len(cli.batch_calls) == 3


async def test_session_rate_limit_skips_back_to_back_turns(tmp_path: Path) -> None:
    cli = FakeCliClient()
    hook, _cfg = _make(tmp_path, cli, session_min_eval_interval_s=60.0)

    hook.on_turn_end(RawLoopAdapter.make_turn("s", 1))
    await hook.refresh("s")
    hook.on_turn_end(RawLoopAdapter.make_turn("s", 2))
    await hook.refresh("s")

    assert len(cli.batch_calls) == 1


async def test_budget_caps_launches_and_journals_the_skip(tmp_path: Path) -> None:
    cli = FakeCliClient()
    hook, _cfg = _make(tmp_path, cli, max_evals_per_run=2)

    for sid in ("s-a", "s-b", "s-c"):
        hook.on_turn_end(RawLoopAdapter.make_turn(sid, 1))
        await hook.refresh(sid)

    assert len(cli.batch_calls) == 2

    # The skip is journaled from a detached task; give it a beat.
    await asyncio.sleep(0.05)
    skips = hook.journal.recent(types=("skip",))
    assert len(skips) == 1
    assert skips[0]["reason"] == "budget"
    assert skips[0]["session_id"] == "s-c"


async def test_semaphore_bounds_concurrent_evals(tmp_path: Path) -> None:
    cli = FakeCliClient(latency_s=0.02)
    hook, _cfg = _make(tmp_path, cli, max_concurrent_evals=2)

    for i in range(6):
        hook.on_turn_end(RawLoopAdapter.make_turn(f"s-{i}", 1))
    await hook.refresh_all()

    assert len(cli.batch_calls) == 6
    assert cli.max_inflight <= 2


async def test_observe_only_journals_without_posting(tmp_path: Path) -> None:
    cli = FakeCliClient(metric_values=dict(BREACHING))
    hook, cfg = _make(tmp_path, cli, observe_only=True)

    hook.on_turn_end(RawLoopAdapter.make_turn("s", 1))
    report = await hook.refresh("s")

    assert report is not None and report.any_breach
    # Shadow mode: no mailbox post, no per-notice dump — journal + latest only.
    assert hook.mailbox.pending() == []
    notices = hook.journal.recent(types=("notice",))
    assert len(notices) == 1
    assert notices[0]["observe_only"] is True
    assert cfg.latest_eval_file.exists()
    assert [p.name for p in cfg.traces_dir.iterdir()] == ["latest_eval.json"]
