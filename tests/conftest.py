"""Shared pytest fixtures for the harness test suite."""

from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest

from pandaprobe_harness import (
    Harness,
    HarnessConfig,
    HarnessFilesystem,
    HarnessToolset,
    Journal,
    Mailbox,
    RawLoopAdapter,
    RulesStore,
    ScoreHistoryStore,
)
from pandaprobe_harness.workspace.evalset import EvalSet
from tests.fakes.fake_cli_client import FakeCliClient

FAKE_BIN = Path(__file__).parent / "bin" / "fake_pandaprobe"


@pytest.fixture
def config(tmp_path: Path) -> HarnessConfig:
    """A HarnessConfig rooted at a temp dir with fast polling for tests."""

    return HarnessConfig(
        harness_root=tmp_path / "harness",
        poll_interval_s=0.0,
        poll_max_attempts=5,
        drain_timeout_s=5.0,
    )


@pytest.fixture
def fs(config: HarnessConfig) -> HarnessFilesystem:
    """A provisioned diagnostic filesystem."""

    filesystem = HarnessFilesystem(config)
    filesystem.provision()
    return filesystem


@pytest.fixture
def fake_cli() -> FakeCliClient:
    return FakeCliClient()


@pytest.fixture
def adapter() -> RawLoopAdapter:
    return RawLoopAdapter()


@pytest.fixture
def fake_bin() -> Path:
    return FAKE_BIN


@pytest.fixture
def mailbox(config: HarnessConfig) -> Mailbox:
    box = Mailbox(config)
    box.provision()
    return box


@pytest.fixture
def journal(config: HarnessConfig) -> Journal:
    return Journal(config)


@pytest.fixture
def rules(config: HarnessConfig, journal: Journal) -> RulesStore:
    return RulesStore(config, journal=journal)


@pytest.fixture
def evalset(config: HarnessConfig, journal: Journal) -> EvalSet:
    store = EvalSet(config, journal=journal)
    store.provision()
    return store


@pytest.fixture
def toolset(
    config: HarnessConfig,
    fake_cli: FakeCliClient,
    mailbox: Mailbox,
    journal: Journal,
    rules: RulesStore,
    evalset: EvalSet,
) -> HarnessToolset:
    return HarnessToolset(
        config=config,
        cli=fake_cli,
        mailbox=mailbox,
        journal=journal,
        rules=rules,
        history=ScoreHistoryStore(config),
        evalset=evalset,
    )


@pytest.fixture
def harness(config: HarnessConfig, fake_cli: FakeCliClient) -> Harness:
    """A fully-assembled offline harness over the fake CLI."""

    return Harness.create(config, cli=fake_cli)


@pytest.fixture
def pandaprobe_path(tmp_path: Path) -> dict[str, str]:
    """An env exposing the fake binary on PATH under the name ``pandaprobe``.

    Lets the agent's RestrictedShellTool invoke ``pandaprobe ...`` offline.
    """

    bin_dir = tmp_path / "shim-bin"
    bin_dir.mkdir()
    shim = bin_dir / "pandaprobe"
    shim.write_text(FAKE_BIN.read_text(encoding="utf-8"), encoding="utf-8")
    shim.chmod(shim.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    return env


@pytest.fixture
def harness_data_dir() -> Iterator[None]:
    """Placeholder for symmetry; tests use tmp_path-based roots."""

    yield None
