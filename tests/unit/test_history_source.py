"""Tests for the pluggable trajectory-history protocol."""

from __future__ import annotations

from pandaprobe_harness import HarnessConfig, ScoreHistoryStore
from pandaprobe_harness.evaluation.history_source import HistorySource


def test_store_satisfies_the_history_source_protocol(config: HarnessConfig) -> None:
    assert isinstance(ScoreHistoryStore(config), HistorySource)
