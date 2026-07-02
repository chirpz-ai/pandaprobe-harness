from __future__ import annotations

from pathlib import Path

from pandaprobe_harness import HarnessConfig, HarnessFilesystem, Mailbox, RawLoopAdapter
from pandaprobe_harness.hook.core import PandaHarnessHook
from tests.fakes.fake_cli_client import FakeCliClient


def _hook(
    tmp_path: Path, cli: FakeCliClient, **cfgkw: object
) -> tuple[PandaHarnessHook, Mailbox]:
    cfg = HarnessConfig(
        harness_root=tmp_path / "h",
        poll_interval_s=0.0,
        poll_max_attempts=5,
        eval_retry_backoff_s=0.0,
        **cfgkw,  # type: ignore[arg-type]
    )
    fs = HarnessFilesystem(cfg)
    fs.provision()
    return PandaHarnessHook(cli, config=cfg, filesystem=fs), Mailbox(cfg)


async def _turn(hook: PandaHarnessHook, mailbox: Mailbox, sid: str, idx: int) -> int:
    """One turn; returns how many NEW notices were posted by it."""

    before = len(mailbox.pending())
    hook.on_turn_end(RawLoopAdapter.make_turn(sid, idx))
    await hook.refresh(sid)
    return len(mailbox.pending()) - before


async def test_breach_notices_once_until_recovery(tmp_path: Path) -> None:
    cli = FakeCliClient(metric_values={"agent_reliability": 0.2, "agent_consistency": 0.2})
    hook, mailbox = _hook(tmp_path, cli, enable_trend=False)

    assert await _turn(hook, mailbox, "s", 1) == 1  # first breach posts
    assert await _turn(hook, mailbox, "s", 2) == 0  # same breach suppressed

    cli.set_scores(agent_reliability=0.9, agent_consistency=0.9)
    assert await _turn(hook, mailbox, "s", 3) == 0  # recovered → no notice
    recoveries = hook.journal.recent(types=("recovery",))
    assert len(recoveries) == 1  # recovery is journaled

    cli.set_scores(agent_reliability=0.2, agent_consistency=0.2)
    assert await _turn(hook, mailbox, "s", 4) == 1  # re-breach re-posts


async def test_cooldown_reposts_after_n_turns(tmp_path: Path) -> None:
    cli = FakeCliClient(metric_values={"agent_reliability": 0.2, "agent_consistency": 0.2})
    hook, mailbox = _hook(tmp_path, cli, enable_trend=False, alert_cooldown_turns=2)

    assert await _turn(hook, mailbox, "s", 1) == 1  # post; cooldown=2
    assert await _turn(hook, mailbox, "s", 2) == 0  # cooldown 2→1
    assert await _turn(hook, mailbox, "s", 3) == 0  # cooldown 1→0
    assert await _turn(hook, mailbox, "s", 4) == 1  # cooldown elapsed → repost
