"""Unit tests for the ``Harness`` facade: provisioning, turn helpers, factories.

Covers ``Harness.create`` workspace assembly, the ``turn()`` context-manager /
decorator forms (including exceptional exit), ``run_turn``, the per-framework
factories (without the optional extras installed), and the degraded mode when
the CLI is unavailable.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import pytest

from pandaprobe_harness import Harness, HarnessConfig
from pandaprobe_harness.adapters.crewai import CrewAIAdapter
from pandaprobe_harness.adapters.langgraph import LangGraphAdapter
from tests.fakes.fake_cli_client import FakeCliClient

BREACHING = {"agent_reliability": 0.2, "agent_consistency": 0.2}


def _cfg(tmp_path: Path, name: str) -> HarnessConfig:
    return HarnessConfig(
        harness_root=tmp_path / name,
        poll_interval_s=0.0,
        poll_max_attempts=5,
        eval_retry_backoff_s=0.0,
    )


def test_create_provisions_workspace_and_context(
    config: HarnessConfig, harness: Harness
) -> None:
    assert config.mailbox_pending_dir.is_dir()
    assert config.mailbox_processed_dir.is_dir()
    assert config.traces_dir.is_dir()
    assert config.state_dir.is_dir()
    assert config.rules_file.is_file()
    assert config.rules_file.read_text(encoding="utf-8").strip()

    context = harness.system_context()
    assert "PANDAPROBE HARNESS RULES" in context
    assert "harness_mailbox_list" in context


async def test_turn_context_manager_fires_one_eval(
    harness: Harness, fake_cli: FakeCliClient
) -> None:
    async with harness.turn("s"):
        pass
    await harness.refresh("s")
    assert len(fake_cli.batch_calls) == 1


async def test_turn_context_manager_fires_on_exceptional_exit(
    harness: Harness, fake_cli: FakeCliClient
) -> None:
    with pytest.raises(RuntimeError):
        async with harness.turn("s"):
            raise RuntimeError("agent step blew up")
    await harness.refresh("s")
    assert len(fake_cli.batch_calls) == 1


async def test_turn_decorator_fires_per_call(
    harness: Harness, fake_cli: FakeCliClient
) -> None:
    async def step(value: int) -> int:
        return value * 2

    fn = harness.turn("s")(step)

    assert await fn(2) == 4
    await harness.refresh("s")
    assert await fn(3) == 6
    await harness.refresh("s")

    assert len(fake_cli.batch_calls) == 2


async def test_run_turn_returns_value_and_fires_turn(
    harness: Harness, fake_cli: FakeCliClient
) -> None:
    async def step(value: int) -> int:
        return value + 1

    result = await harness.run_turn("s", step, 41)
    assert result == 42
    await harness.refresh("s")
    assert len(fake_cli.batch_calls) == 1


def test_for_langgraph_wires_the_adapter(tmp_path: Path) -> None:
    harness = Harness.for_langgraph(
        session_id="x", config=_cfg(tmp_path, "lg"), cli=FakeCliClient()
    )
    assert isinstance(harness.adapter, LangGraphAdapter)


def test_for_crewai_builds_without_the_crewai_dep(tmp_path: Path) -> None:
    harness = Harness.for_crewai(
        session_id="x", config=_cfg(tmp_path, "crew"), cli=FakeCliClient()
    )
    assert isinstance(harness.adapter, CrewAIAdapter)
    # The optional dependency is absent: instrument() degrades to False, but
    # the harness itself is fully assembled and usable.
    assert harness.adapter.instrument() is False
    assert harness.mailbox.pending() == []
    assert "PANDAPROBE HARNESS RULES" in harness.system_context()


async def test_degraded_mode_skips_evals_with_one_warning(
    config: HarnessConfig, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger="pandaprobe_harness.hook")
    cli = FakeCliClient(version_ok=False, metric_values=dict(BREACHING))
    harness = Harness.create(config, cli=cli)

    harness.on_turn_end({"session_id": "s", "turn_index": 1, "end_state": {}})
    assert await harness.refresh("s") is None

    assert harness.mailbox.pending() == []
    assert cli.batch_calls == []

    # The health event is journaled from the (possibly concurrent) check task.
    for _ in range(200):
        if harness.journal.recent(types=("health",)):
            break
        await asyncio.sleep(0.01)
    health = harness.journal.recent(types=("health",))
    assert len(health) == 1
    assert health[0]["ok"] is False

    def degraded_warnings() -> int:
        return sum(
            1
            for record in caplog.records
            if record.name == "pandaprobe_harness.hook" and "degraded" in record.getMessage()
        )

    assert degraded_warnings() == 1

    # A second turn stays silent: memoized health, no new warning or event.
    harness.on_turn_end({"session_id": "s", "turn_index": 2, "end_state": {}})
    assert await harness.refresh("s") is None
    await asyncio.sleep(0.05)
    assert degraded_warnings() == 1
    assert len(harness.journal.recent(types=("health",))) == 1
    assert cli.batch_calls == []
