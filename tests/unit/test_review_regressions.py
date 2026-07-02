"""Regression tests for defects found by the v0.5 adversarial review.

Each test pins a specific confirmed finding so it cannot silently return.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from pandaprobe_harness import (
    HarnessConfig,
    HarnessFilesystem,
    Mailbox,
    RawLoopAdapter,
    ShellPolicy,
    SubprocessCliClient,
)
from pandaprobe_harness.agent_tools.companion import _parse_args
from pandaprobe_harness.cli.errors import CliError
from pandaprobe_harness.evaluation.history import ScoreHistoryStore
from pandaprobe_harness.hook.core import PandaHarnessHook
from pandaprobe_harness.sandbox.policy import ShellPolicyError
from pandaprobe_harness.workspace.mailbox import DiagnosticNotice
from tests.fakes.fake_cli_client import FakeCliClient

# -- H1/H9: missing binary surfaces as CliError, not raw OSError -------------


async def test_missing_binary_raises_clierror_not_oserror() -> None:
    client = SubprocessCliClient(binary="definitely-not-a-real-binary-xyz")
    with pytest.raises(CliError):  # not FileNotFoundError
        await client.run("version")


async def test_degraded_mode_engages_when_binary_missing(tmp_path: Path) -> None:
    cfg = HarnessConfig(harness_root=tmp_path / "h", poll_interval_s=0.0, eval_retry_backoff_s=0.0)
    fs = HarnessFilesystem(cfg)
    fs.provision()
    cli = SubprocessCliClient(binary="definitely-not-a-real-binary-xyz")
    hook = PandaHarnessHook(cli, config=cfg, filesystem=fs)

    assert await hook.check_health() is False  # returns, does not raise
    hook.on_turn_end(RawLoopAdapter.make_turn("s", 1))
    assert await hook.refresh("s") is None
    assert Mailbox(cfg).pending() == []
    health = hook.journal.recent(types=("health",))
    assert len(health) == 1 and health[0]["ok"] is False


# -- H2: mailbox notice-id path traversal ------------------------------------


def test_mailbox_rejects_traversal_notice_id(tmp_path: Path) -> None:
    cfg = HarnessConfig(harness_root=tmp_path / "h")
    fs = HarnessFilesystem(cfg)
    fs.provision()
    mailbox = Mailbox(cfg)
    mailbox.provision()

    # A real file the traversal would target.
    victim = cfg.state_dir / "score_history.json"
    victim.write_text(json.dumps({"keep": "me"}), encoding="utf-8")

    for evil in ("../../state/score_history", "..", "a/b", "/etc/passwd", "x" * 200):
        assert mailbox.read(evil) is None
        with pytest.raises(KeyError):
            mailbox.acknowledge(evil)

    assert json.loads(victim.read_text(encoding="utf-8")) == {"keep": "me"}


# -- H3/H7/H14: shell path-escape catches mid-path traversal -----------------


def test_shell_policy_blocks_midpath_traversal() -> None:
    policy = ShellPolicy(workdir=Path("/harness"))
    for escape in (
        "state/../../../etc/passwd",
        "a/../../secret",
        "../outside",
        "/etc/passwd",
    ):
        with pytest.raises(ShellPolicyError):
            policy.validate(["cat", escape])
    # An in-workspace relative path is still allowed.
    policy.validate(["cat", "traces/latest_eval.json"])
    policy.validate(["ls", "mailbox"])


# -- H8: argv-prefix denial cannot be bypassed with a leading global flag ----


def test_shell_policy_denials_survive_flag_insertion() -> None:
    policy = ShellPolicy(workdir=Path("/harness"))
    with pytest.raises(ShellPolicyError):
        policy.validate(["pandaprobe", "--format", "json", "config", "show"])
    with pytest.raises(ShellPolicyError):
        policy.validate(["pandaprobe", "config", "show"])
    with pytest.raises(ShellPolicyError):
        policy.validate(["pandaprobe", "evals", "scores", "get", "x", "--reveal-secrets=1"])
    # A legitimate command still passes.
    policy.validate(["pandaprobe", "evals", "scores", "get", "x"])


# -- M2: backend hydration seeds EWMA in chronological order -----------------


async def test_hydration_orders_samples_by_timestamp(tmp_path: Path) -> None:
    cfg = HarnessConfig(
        harness_root=tmp_path / "h",
        poll_interval_s=0.0,
        poll_max_attempts=3,
        eval_retry_backoff_s=0.0,
        hydrate_history_from_backend=True,
    )
    fs = HarnessFilesystem(cfg)
    fs.provision()
    # Backend returns newest-first; true chronology is 0.3 -> 0.6 -> 0.9.
    cli = FakeCliClient(
        metric_values={"agent_reliability": 0.9, "agent_consistency": 0.9},
        session_scores_list={
            "s": [
                {"name": "agent_reliability", "value": "0.9",
                 "created_at": "2026-06-03T00:00:00Z", "run_id": "r3"},
                {"name": "agent_reliability", "value": "0.6",
                 "created_at": "2026-06-02T00:00:00Z", "run_id": "r2"},
                {"name": "agent_reliability", "value": "0.3",
                 "created_at": "2026-06-01T00:00:00Z", "run_id": "r1"},
            ]
        },
    )
    hook = PandaHarnessHook(cli, config=cfg, filesystem=fs)
    hook.on_turn_end(RawLoopAdapter.make_turn("s", 1))
    await hook.refresh("s")

    # The seeded prefix must be chronological (ascending), not the CLI order.
    values = ScoreHistoryStore(cfg).values("s", "agent_reliability")
    assert values[:3] == [0.3, 0.6, 0.9]


# -- L2: companion CLI rejects a forgotten value that eats the next flag -----


def test_parse_args_rejects_flag_shaped_value() -> None:
    parsed = _parse_args(["--rule", "always cite", "--rationale", "--metric"])
    assert isinstance(parsed, str) and "missing value" in parsed
    # A well-formed pair still parses, with JSON coercion.
    ok = _parse_args(["--limit", "5", "--rule", "text"])
    assert ok == {"limit": 5, "rule": "text"}


# -- L3: per-session bookkeeping is bounded ----------------------------------


async def test_per_session_bookkeeping_is_bounded(tmp_path: Path) -> None:
    cfg = HarnessConfig(
        harness_root=tmp_path / "h",
        poll_interval_s=0.0,
        eval_retry_backoff_s=0.0,
        health_check=False,
        eval_sample_every=10_000,  # admit turn 1 only; keep it cheap
    )
    fs = HarnessFilesystem(cfg)
    fs.provision()
    hook = PandaHarnessHook(FakeCliClient(), config=cfg, filesystem=fs)
    from pandaprobe_harness.hook import core as core_mod

    for i in range(core_mod._MAX_TRACKED_SESSIONS + 50):
        hook.on_turn_end(RawLoopAdapter.make_turn(f"s-{i}", 1))
    await asyncio.sleep(0)
    assert len(hook._turn_counts) <= core_mod._MAX_TRACKED_SESSIONS


# -- ensure the mailbox still accepts a normally-generated id ----------------


def test_mailbox_accepts_generated_id(tmp_path: Path) -> None:
    cfg = HarnessConfig(harness_root=tmp_path / "h")
    mailbox = Mailbox(cfg)
    mailbox.provision()
    notice = DiagnosticNotice.from_json(
        {
            "id": DiagnosticNotice.new_id(),
            "created_at": "2026-01-01T00:00:00+00:00",
            "session_id": "s",
            "turn_index": 1,
            "severity": "breach",
        }
    )
    mailbox.post(notice)
    assert mailbox.read(notice.id) is not None
    mailbox.acknowledge(notice.id, rule_id="r-1")
    assert mailbox.read(notice.id).status == "acknowledged"  # type: ignore[union-attr]
