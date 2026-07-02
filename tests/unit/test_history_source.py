"""Score-history seeding semantics and one-time-per-session backend hydration.

``ScoreHistoryStore.seed`` is the cold-start seam for horizontally-scaled
agents: bulk-inserted backend samples must be idempotent by ``run_id`` and
advance the EWMA state exactly like live recording. The hook hydrates a
session's history from ``evals scores list`` at most once, before any local
per-turn scores are appended.
"""

from __future__ import annotations

from pathlib import Path

from pandaprobe_harness import HarnessConfig, HarnessFilesystem, ScoreHistoryStore
from pandaprobe_harness.evaluation.history_source import HistorySource
from pandaprobe_harness.hook.core import PandaHarnessHook
from tests.fakes.fake_cli_client import FakeCliClient

SEEDED: list[tuple[float, str, str | None]] = [(0.8, "t1", "r1"), (0.7, "t2", "r2")]


# -- unit: seed semantics ---------------------------------------------------------


def test_seed_inserts_samples_and_advances_ewma(config: HarnessConfig) -> None:
    store = ScoreHistoryStore(config)
    store.seed("s", "agent_reliability", SEEDED)

    assert store.values("s", "agent_reliability") == [0.8, 0.7]
    state = store.ewma("s", "agent_reliability")
    assert state is not None and state.count == 2


def test_seed_is_idempotent_for_known_run_ids(config: HarnessConfig) -> None:
    store = ScoreHistoryStore(config)
    store.seed("s", "agent_reliability", SEEDED)
    store.seed("s", "agent_reliability", SEEDED)  # same run_ids again: no growth

    assert store.values("s", "agent_reliability") == [0.8, 0.7]
    state = store.ewma("s", "agent_reliability")
    assert state is not None and state.count == 2


def test_seed_mix_inserts_only_new_run_ids(config: HarnessConfig) -> None:
    store = ScoreHistoryStore(config)
    store.seed("s", "agent_reliability", SEEDED)
    store.seed("s", "agent_reliability", [(0.7, "t2", "r2"), (0.6, "t3", "r3")])

    assert store.values("s", "agent_reliability") == [0.8, 0.7, 0.6]
    state = store.ewma("s", "agent_reliability")
    assert state is not None and state.count == 3


def test_store_satisfies_the_history_source_protocol(config: HarnessConfig) -> None:
    assert isinstance(ScoreHistoryStore(config), HistorySource)


# -- integration: the hook hydrates once per session ------------------------------


def _declining_items() -> list[dict[str, str]]:
    return [
        {
            "name": "agent_reliability",
            "value": str(value),
            "created_at": f"2026-06-0{i}T00:00:00Z",
            "run_id": f"r{i}",
        }
        for i, value in enumerate((0.9, 0.8, 0.7, 0.6), start=1)
    ]


async def test_hook_hydrates_backend_history_once_per_session(
    tmp_path: Path, fake_cli: FakeCliClient
) -> None:
    cfg = HarnessConfig(
        harness_root=tmp_path / "harness",
        poll_interval_s=0.0,
        poll_max_attempts=5,
        eval_retry_backoff_s=0.0,
        hydrate_history_from_backend=True,
    )
    fs = HarnessFilesystem(cfg)
    fs.provision()
    fake_cli.session_scores_list = {"s": _declining_items()}
    fake_cli.set_scores(agent_reliability=0.85, agent_consistency=0.9)
    hook = PandaHarnessHook(fake_cli, config=cfg, filesystem=fs)

    for turn_index in (1, 2):
        hook.on_turn_end({"session_id": "s", "turn_index": turn_index, "end_state": {}})
        await hook.refresh("s")

    # One hydration call total, despite two evaluated turns.
    list_calls = [c for c in fake_cli.calls if c[:3] == ("evals", "scores", "list")]
    assert len(list_calls) == 1

    # Seeded backend samples come first, then the per-turn recorded scores.
    values = ScoreHistoryStore(cfg).values("s", "agent_reliability")
    assert values[:4] == [0.9, 0.8, 0.7, 0.6]
    assert values[4:] == [0.85, 0.85]
