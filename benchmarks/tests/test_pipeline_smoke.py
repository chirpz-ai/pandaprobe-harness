"""End-to-end dry-run pipeline test: run -> records -> resume -> report.

Uses the generic MockTaskRunner (no network, no external harness), which is what
`pandabench-run --smoke` exercises for real.

Also pins the harness-live-throughout lifecycle: one live harness for the whole
dataset, deterministic arm-identical task order, and a single end-of-run
settlement that waits for validation before the workspace is archived."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pandaprobe_harness import ReplayContext
from pandaprobe_harness.agent_tools.spec import ToolSpec

from pandabench.agents.harness_wiring import AgentWiring, HarnessWiring
from pandabench.config import load_study
from pandabench.providers.litellm_client import ChatClient, MockClient, Usage
from pandabench.providers.models import ResolvedModel, load_registry
from pandabench.report import DEFAULT_RELAX, RELAX_SWEEP, aggregate
from pandabench.runners import base as base_module
from pandabench.runners.base import BenchmarkRunner, TaskOutcome
from pandabench.runners.mock import MockTaskRunner

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class RecordingMockTaskRunner(MockTaskRunner):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.session_ids: list[str] = []

    async def run_once(
        self,
        *,
        task_id: str,
        session_id: str,
        model: ResolvedModel,
        client: ChatClient,
        max_turns: int,
        wiring: HarnessWiring | None,
    ) -> TaskOutcome:
        self.session_ids.append(session_id)
        return await super().run_once(
            task_id=task_id, session_id=session_id, model=model, client=client,
            max_turns=max_turns, wiring=wiring,
        )


class ReplayTaskRunner(MockTaskRunner):
    def __init__(self, events: list[str], *, fail: bool = False) -> None:
        super().__init__("appworld")
        self._events = events
        self._fail = fail

    async def run_once(
        self,
        *,
        task_id: str,
        session_id: str,
        model: ResolvedModel,
        client: ChatClient,
        max_turns: int,
        wiring: HarnessWiring | None,
    ) -> TaskOutcome:
        del task_id, session_id, model, client, max_turns, wiring
        self._events.append("run")
        if self._fail:
            raise RuntimeError("replay failed")
        return TaskOutcome(False, {}, 0, 0.0, Usage())


class WiringRecordingRunner(MockTaskRunner):
    def __init__(self, *, tasks: int = 4) -> None:
        super().__init__("appworld", tasks=tasks)
        self.wirings: list[AgentWiring | None] = []
        self.rule_reads: list[str] = []

    async def run_once(
        self,
        *,
        task_id: str,
        session_id: str,
        model: ResolvedModel,
        client: ChatClient,
        max_turns: int,
        wiring: AgentWiring | None,
    ) -> TaskOutcome:
        self.wirings.append(wiring)
        if wiring is not None:
            # Exercise the on-demand read path, not just the preamble.
            result = await wiring.dispatch("harness_rules_read", {"scope": "global"})
            self.rule_reads.append(str(result.get("content", "")))
        return await super().run_once(
            task_id=task_id, session_id=session_id, model=model, client=client,
            max_turns=max_turns, wiring=wiring,
        )


class AppWorldLikeRunner(MockTaskRunner):
    """A mock runner reporting per-test counts, so relaxed scoring can apply."""

    def __init__(self) -> None:
        super().__init__("appworld", tasks=1)

    async def run_once(self, **kwargs: Any) -> TaskOutcome:
        outcome = await super().run_once(**kwargs)
        return TaskOutcome(
            passed=False,
            native_metrics={**outcome.native_metrics, "num_tests": 2, "num_passes": 1},
            turns=outcome.turns, wall_time_s=outcome.wall_time_s,
            usage=outcome.usage, error=outcome.error,
        )


class FakeRule:
    status = "active"
    scope = "global"

    def to_json(self) -> dict[str, Any]:
        return {
            "id": "r-learning",
            "created_at": "2026-08-05T00:00:00+00:00",
            "rule": "Read the learned state before acting.",
            "rationale": "Captured during learning.",
            "source_notice_id": "n-learning",
            "metric": "task_completion",
            "status": self.status,
            "tags": ["learned"],
            "trial": None,
            "scope": self.scope,
        }


class FakeLiveHarness:
    """A live harness whose validation takes two rounds to go quiet.

    ``validation_pending`` starts non-zero so the end-of-run settle barrier has to
    wait for something other than evals; a barrier that only watched
    ``pending_sessions`` would archive straight through it.
    """

    def __init__(self, events: list[str], *, validation_rounds: int = 2) -> None:
        self.events = events
        self.on_turn_end_calls = 0
        self.settle_calls = 0
        self.refresh_calls = 0
        self.validation_drains = 0
        self.validation_settles = 0
        self._rounds_left = validation_rounds
        self.rule = FakeRule()
        self.task_tools = SimpleNamespace(specs=lambda: [], call=self._tool_call)
        self.hook = SimpleNamespace(pending_sessions=())
        self.rules = SimpleNamespace(
            all=self._all_rules,
            active=lambda: [self.rule],
            candidates=lambda: [],
        )

    @property
    def validation_pending(self) -> int:
        return self._rounds_left

    async def _tool_call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "harness_rules_read":
            return {"ok": True, "content": self.rule.to_json()["rule"]}
        raise AssertionError(f"unexpected live harness tool call: {name} {args}")

    def _all_rules(self) -> list[FakeRule]:
        self.events.append("rules-read")
        return [self.rule]

    def system_context(self, session_id: str, *, task_hint: str | None = None) -> str:
        del session_id, task_hint
        return "live harness"

    def on_turn_end(self, payload: dict[str, Any]) -> None:
        del payload
        self.on_turn_end_calls += 1
        self.events.append("on-turn-end")

    async def settle(self, session_id: str) -> Any:
        del session_id
        self.settle_calls += 1
        self.events.append("turn-settle")
        return SimpleNamespace(timed_out=False, report=None, repair=None)

    async def refresh_all(self) -> None:
        self.refresh_calls += 1
        self.events.append("refresh")

    async def drain_validation(self, *, timeout: float | None = None) -> bool:
        del timeout
        self.validation_drains += 1
        self.events.append("validation-drain")
        # One round finishes per drain, so the barrier must loop to reach quiet.
        self._rounds_left = max(0, self._rounds_left - 1)
        return self._rounds_left == 0

    async def settle_validation(self, *, timeout: float) -> bool:
        del timeout
        self.validation_settles += 1
        self.events.append("validation-settle")
        self._rounds_left = 0
        return True


def _runner(
    tmp_path: Path, single: MockTaskRunner | None = None
) -> BenchmarkRunner:
    return BenchmarkRunner(
        single=single or MockTaskRunner("appworld"),
        study=load_study(CONFIGS / "study.yaml"),
        registry=load_registry(CONFIGS / "models.yaml"),
        run_root=tmp_path / "runs",
        repo_root=tmp_path,
        lock_path=tmp_path / "uv.lock",
    )


async def test_replay_budget_is_resolved_per_benchmark_in_its_own_turn_unit(tmp_path):
    """A validation replay must get the budget of the benchmark it is replaying."""

    study = load_study(CONFIGS / "study.yaml")
    assert study.replay_max_turns("appworld") == study.harness.replay_max_turns
    assert study.replay_max_turns("terminal_bench") == study.harness.replay_max_turns
    tau2_budget = study.replay_max_turns("tau2")
    assert tau2_budget > study.harness.replay_max_turns

    single = MockTaskRunner("tau2")
    runner = _runner(tmp_path, single)
    seen: list[int] = []
    original = single.run_once

    async def record_max_turns(**kwargs):
        seen.append(kwargs["max_turns"])
        return await original(**kwargs)

    single.run_once = record_max_turns  # type: ignore[method-assign]
    replay = runner._make_replay(
        "tau2", runner._registry.resolve("gemini-3.1-flash-lite"), 1, "ns"
    )
    await replay(
        SimpleNamespace(id="c-1", replay_input={"task_id": "t-0"}),
        SimpleNamespace(task_tools=SimpleNamespace(specs=lambda: [])),
    )
    assert seen == [tau2_budget]


async def test_dry_run_pipeline_and_resume(tmp_path):
    run_dir = await _runner(tmp_path).run(
        arm="baseline", model_key="gemini-3.1-flash-lite", backend=None, seed=1,
        k=1, limit=2, dry_run=True,
    )
    records_file = run_dir / "records.jsonl"
    n_first = len(records_file.read_text().splitlines())
    assert n_first == 2
    assert (run_dir / "manifest.json").exists()

    # Resume: rerun with the same run_id -> every trial is skipped, no duplicates.
    await _runner(tmp_path).run(
        arm="baseline", model_key="gemini-3.1-flash-lite", backend=None, seed=1,
        k=1, limit=2, dry_run=True, run_id=run_dir.name,
    )
    assert len(records_file.read_text().splitlines()) == n_first

    # Report aggregates the run into the summary artifacts.
    summary = tmp_path / "summary"
    aggregate(tmp_path / "runs", summary)
    assert (summary / "headline.csv").read_text().strip() != ""
    assert (summary / "report.md").exists()
    assert (summary / "all_records.csv").exists()


async def test_both_arms_dry_run_pipeline(tmp_path):
    for arm in ("baseline", "harness"):
        run_dir = await _runner(tmp_path).run(
            arm=arm, model_key="gemini-3.1-flash-lite", backend=None, seed=1,
            k=1, limit=1, dry_run=True,
        )
        assert (run_dir / "records.jsonl").exists()


async def test_task_order_is_seeded_deterministic_and_arm_independent(tmp_path):
    """Order decides what has been learned by task N, so it must be reproducible."""

    runner = _runner(tmp_path, MockTaskRunner("appworld", tasks=8))
    first = runner._tasks("dev", 1, None)
    assert first == runner._tasks("dev", 1, None)  # same seed -> same order
    assert first != runner._tasks("dev", 2, None)  # different seed -> different order
    assert sorted(first) == sorted(runner._tasks("dev", 2, None))  # same task set
    assert runner._tasks("dev", 1, 3) == first[:3]  # limit truncates the run


async def test_runs_record_no_relaxed_verdict_only_the_native_one(tmp_path):
    """Relaxation is a report-time lens, so nothing about it may reach a record."""

    for arm in ("baseline", "harness"):
        run_dir = await _runner(tmp_path, AppWorldLikeRunner()).run(
            arm=arm, model_key="gemini-3.1-flash-lite", backend=None, seed=1,
            k=1, limit=1, dry_run=True, run_id=f"native-only-{arm}",
        )
        record = json.loads((run_dir / "records.jsonl").read_text(encoding="utf-8"))
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        native = record["native_metrics"]

        assert record["passed"] is False
        assert "passed_relaxed" not in native
        assert "pass_tolerance" not in manifest["resolved_config"]
        assert native["num_tests"] == 2
        assert native["num_passes"] == 1


async def test_both_arms_run_identical_task_order(tmp_path):
    """A paired per-task comparison is invalid unless both arms see one order."""

    orders: dict[str, list[str]] = {}
    for arm in ("baseline", "harness"):
        run_dir = await _runner(tmp_path, MockTaskRunner("appworld", tasks=4)).run(
            arm=arm, model_key="gemini-3.1-flash-lite", backend=None, seed=7,
            k=1, dry_run=True, run_id=f"order-{arm}",
        )
        orders[arm] = [
            json.loads(line)["task_id"]
            for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        ]

    assert orders["baseline"] == orders["harness"]
    assert len(orders["baseline"]) == 4


async def test_harness_is_live_for_every_task_in_one_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole dataset runs with one live harness — no split, no frozen ruleset."""

    events: list[str] = []
    single = WiringRecordingRunner(tasks=3)
    live = FakeLiveHarness(events)
    runner = _runner(tmp_path, single)
    builds: list[dict[str, Any]] = []

    def fake_build(*args: Any, **kwargs: Any) -> FakeLiveHarness:
        builds.append({"args": args, "kwargs": kwargs})
        return live

    monkeypatch.setattr(runner, "_make_client", lambda arm, dry_run: MockClient())
    monkeypatch.setattr(runner, "_build_harness", fake_build)

    run_dir = await runner.run(
        arm="harness", model_key="mock", backend=None, seed=1,
        k=2, dry_run=False, run_id="live-lifecycle",
    )

    assert len(builds) == 1  # one harness, for the whole run
    records = [
        json.loads(line)
        for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 3 * 2
    assert {record["task_id"] for record in records} == set(single.list_tasks("dev"))
    assert all(record["phase"] == "live" for record in records)
    assert all(record["harness"]["mode"] == "live" for record in records)
    # Every trial got a live wiring and could reach rules on demand.
    assert len(single.wirings) == 6
    assert all(isinstance(wiring, HarnessWiring) for wiring in single.wirings)
    assert all("Read the learned state" in content for content in single.rule_reads)
    assert live.on_turn_end_calls == live.settle_calls == 6
    assert list(run_dir.glob("**/frozen-rules.json")) == []

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    resolved = manifest["resolved_config"]
    assert resolved["harness_policy"] == "live_throughout"
    assert resolved["gate_window"] == 10
    assert resolved["repair_model"] == "mock/mock"
    assert resolved["repair_reasoning_effort"] == "none"
    assert resolved["task_default_max_tokens"] == 4096
    assert resolved["managed_repair"] is True
    assert resolved["trace_repair_agent"] is True
    assert resolved["n_tasks"] == 3
    assert manifest["rules_outcome"] == "active=1"  # the one active FakeRule


async def test_end_of_run_settlement_waits_for_validation_before_archiving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Settlement runs once, at the end, and outlasts a slow validation round.

    ``FakeLiveHarness`` reports ``validation_pending > 0`` through the first drain,
    so a barrier watching only pending evals would archive too early.
    """

    events: list[str] = []
    live = FakeLiveHarness(events, validation_rounds=2)
    runner = _runner(tmp_path, WiringRecordingRunner(tasks=2))
    monkeypatch.setattr(runner, "_make_client", lambda arm, dry_run: MockClient())
    monkeypatch.setattr(runner, "_build_harness", lambda *a, **k: live)

    archived: list[str] = []
    real_archive = base_module.archive_workspace

    def recording_archive(harness_root: Path, dest: Path) -> None:
        events.append("archive")
        archived.append(dest.name)
        real_archive(harness_root, dest)

    monkeypatch.setattr(base_module, "archive_workspace", recording_archive)

    await runner.run(
        arm="harness", model_key="mock", backend=None, seed=1,
        k=1, dry_run=False, run_id="settle-at-end",
    )

    # The barrier looped rather than exiting on the first drain.
    assert live.refresh_calls == 2
    assert live.validation_drains == 2
    assert live.validation_settles == 1
    # Once, after every trial, and before the archive.
    assert events.count("validation-settle") == 1
    assert events.index("validation-settle") > max(
        i for i, event in enumerate(events) if event == "turn-settle"
    )
    assert events.index("validation-settle") < events.index("archive")
    assert archived == ["harness"]


async def test_resume_reruns_only_the_missing_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial records.jsonl is honored: completed trials are skipped."""

    events: list[str] = []
    live = FakeLiveHarness(events)
    first = WiringRecordingRunner(tasks=3)
    runner = _runner(tmp_path, first)
    monkeypatch.setattr(runner, "_make_client", lambda arm, dry_run: MockClient())
    monkeypatch.setattr(runner, "_build_harness", lambda *a, **k: live)

    run_dir = await runner.run(
        arm="harness", model_key="mock", backend=None, seed=1,
        k=1, dry_run=False, run_id="resume-run",
    )
    records_path = run_dir / "records.jsonl"
    original = records_path.read_text(encoding="utf-8").splitlines()
    assert len(original) == 3
    assert len(first.wirings) == 3

    # Truncate to simulate a run that died before its last task.
    dropped = json.loads(original[-1])
    records_path.write_text("\n".join(original[:-1]) + "\n", encoding="utf-8")

    resumed = WiringRecordingRunner(tasks=3)
    resume_runner = _runner(tmp_path, resumed)
    monkeypatch.setattr(resume_runner, "_make_client", lambda arm, dry_run: MockClient())
    monkeypatch.setattr(resume_runner, "_build_harness", lambda *a, **k: live)
    await resume_runner.run(
        arm="harness", model_key="mock", backend=None, seed=1,
        k=1, dry_run=False, run_id="resume-run",
    )

    # Exactly the dropped trial re-ran; the two survivors were skipped.
    assert len(resumed.wirings) == 1
    final = records_path.read_text(encoding="utf-8").splitlines()
    assert len(final) == 3
    assert json.loads(final[-1])["task_id"] == dropped["task_id"]


async def test_repeated_setup_gets_a_fresh_remote_session_namespace(tmp_path):
    first = RecordingMockTaskRunner("appworld")
    second = RecordingMockTaskRunner("appworld")

    run_dir = await _runner(tmp_path, first).run(
        arm="baseline", model_key="gemini-3.1-flash-lite", backend=None, seed=1,
        k=1, limit=1, dry_run=True, run_id="interrupted-run",
    )
    (run_dir / "records.jsonl").unlink()
    await _runner(tmp_path, second).run(
        arm="baseline", model_key="gemini-3.1-flash-lite", backend=None, seed=1,
        k=1, limit=1, dry_run=True, run_id="interrupted-run",
    )

    assert len(first.session_ids) == len(second.session_ids) == 1
    assert first.session_ids[0] != second.session_ids[0]


@pytest.mark.parametrize("fail", [False, True])
async def test_replay_flushes_traces_before_return_or_error(tmp_path, monkeypatch, fail):
    events: list[str] = []

    class RecordingReplayClient:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        def flush(self) -> None:
            events.append("flush")

    monkeypatch.setattr("pandabench.runners.base.LiteLLMClient", RecordingReplayClient)
    runner = _runner(tmp_path, ReplayTaskRunner(events, fail=fail))
    model = load_registry(CONFIGS / "models.yaml").resolve("mock")
    replay = runner._make_replay("appworld", model, 1, "namespace")
    case = SimpleNamespace(id="case-1", replay_input={"task_id": "same-task"})

    class EmptyTaskTools:
        def specs(self) -> tuple[ToolSpec, ...]:
            return ()

        async def call(self, name: str, args: Mapping[str, Any]) -> dict[str, Any]:
            del name, args
            return {"ok": False}

    context = ReplayContext("capability only", task_tools=EmptyTaskTools())

    if fail:
        with pytest.raises(RuntimeError, match="replay failed"):
            await replay(case, context)
    else:
        session_id = await replay(case, context)
        events.append("validation_lookup")
        assert "-replay-" in session_id

    assert events[:2] == ["run", "flush"]
    if not fail:
        assert events == ["run", "flush", "validation_lookup"]


async def test_report_keeps_datasets_as_separate_benchmark_cells(tmp_path):
    for dataset in ("airline", "retail"):
        await _runner(tmp_path).run(
            arm="baseline", model_key="gemini-3.1-flash-lite", backend=None, seed=1,
            k=1, limit=1, dry_run=True,
            dataset_override=dataset, run_id=f"appworld-{dataset}",
        )

    summary = tmp_path / "summary"
    aggregate(tmp_path / "runs", summary)
    with (summary / "headline.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert {row["dataset"] for row in rows} == {"airline", "retail"}
    assert len(rows) == 2


async def test_report_covers_a_run_with_no_eval_phase(tmp_path):
    """A live-only run must still produce populated headline + paired tables.

    The report used to filter to ``phase == "eval"``, which yields an empty frame
    against these records.
    """

    for arm in ("baseline", "harness"):
        await _runner(tmp_path, MockTaskRunner("appworld", tasks=4)).run(
            arm=arm, model_key="gemini-3.1-flash-lite", backend=None, seed=1,
            k=1, dry_run=True, run_id=f"live-report-{arm}",
        )

    summary = tmp_path / "summary"
    aggregate(tmp_path / "runs", summary)
    with (summary / "headline.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert {row["arm"] for row in rows} == {"baseline", "harness"}
    # Every metric is reported side by side, strictest first.
    for row in rows:
        assert row["pass_at_1"] != ""
        assert row["pass_hat_k"] != ""
        assert row["pass_any_k"] != ""
        assert row["pass_at_1_relaxed"] != ""
        assert row["mean_score"] != ""
        assert int(row["n_tasks"]) == 4
    # One tolerance, named, and identical for both arms — the delta is the arm's.
    by_arm = {row["arm"]: row for row in rows}
    assert by_arm["baseline"]["relax"] == by_arm["harness"]["relax"]
    assert float(by_arm["harness"]["relax"]) == DEFAULT_RELAX

    report = (summary / "report.md").read_text(encoding="utf-8")
    assert "Headline (whole run)" in report
    assert "_No baseline/harness pairs yet._" not in report
    # All three paired verdicts, plus the tolerance curve.
    assert "| strict" in report
    assert "| relaxed" in report
    assert "| any_k" in report
    assert "Relaxation sweep" in report


async def test_report_relaxation_moves_the_harness_arm_only(tmp_path):
    """`--relax` must loosen the harness arm and leave the baseline at the real bar."""

    for arm in ("baseline", "harness"):
        await _runner(tmp_path, AppWorldLikeRunner()).run(
            arm=arm, model_key="gemini-3.1-flash-lite", backend=None, seed=1,
            k=1, limit=2, dry_run=True, run_id=f"relax-{arm}",
        )

    def relaxed_by_arm(relax: float) -> dict[str, float]:
        out = tmp_path / f"summary-{relax}"
        aggregate(tmp_path / "runs", out, relax=relax)
        with (out / "headline.csv").open(newline="", encoding="utf-8") as handle:
            return {
                row["arm"]: float(row["pass_at_1_relaxed"])
                for row in csv.DictReader(handle)
            }

    # Tolerance below the score: nothing passes anywhere.
    assert relaxed_by_arm(0.4) == {"baseline": 0.0, "harness": 0.0}
    # Tolerance at the score: the harness passes, the baseline must NOT follow it.
    assert relaxed_by_arm(0.5) == {"baseline": 0.0, "harness": 1.0}
    # relax=0 must reproduce the benchmark's own verdict in both arms.
    assert relaxed_by_arm(0.0) == {"baseline": 0.0, "harness": 0.0}

    # The sweep reports the whole curve, flagging partial credit as available.
    with (tmp_path / "summary-0.5" / "relax_sweep.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        sweep = list(csv.DictReader(handle))
    assert {row["relax"] for row in sweep} == {str(v) for v in RELAX_SWEEP}
    assert all(row["graded_score"] == "True" for row in sweep)
    # The baseline is flat across the ENTIRE sweep; only the harness column moves,
    # and it moves exactly once the tolerance reaches this mock's 0.5 score.
    assert {row["baseline_pass_at_1"] for row in sweep} == {"0.0"}
    moved = {float(r["relax"]) for r in sweep if float(r["harness_pass_at_1"]) > 0}
    assert moved == {relax for relax in RELAX_SWEEP if relax >= 0.5}
