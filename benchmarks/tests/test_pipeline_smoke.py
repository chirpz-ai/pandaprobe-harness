"""End-to-end dry-run pipeline test: run -> records -> resume -> report.

Uses the generic MockTaskRunner (no network, no external harness), which is what
`pandabench-run --smoke` exercises for real."""

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

from pandabench.agents.frozen_wiring import FrozenEvalWiring
from pandabench.agents.harness_wiring import AgentWiring, HarnessWiring
from pandabench.config import load_study
from pandabench.providers.litellm_client import ChatClient, MockClient, Usage
from pandabench.providers.models import ResolvedModel, load_registry
from pandabench.report import aggregate
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
    def __init__(self) -> None:
        super().__init__("appworld")
        self.wirings: list[AgentWiring | None] = []
        self.frozen_rule_reads: list[str] = []

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
        if isinstance(wiring, FrozenEvalWiring):
            result = await wiring.dispatch("harness_rules_read", {"scope": "global"})
            self.frozen_rule_reads.append(str(result["content"]))
        return await super().run_once(
            task_id=task_id, session_id=session_id, model=model, client=client,
            max_turns=max_turns, wiring=wiring,
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
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.on_turn_end_calls = 0
        self.settle_calls = 0
        self.refresh_calls = 0
        self.validation_drains = 0
        self.rule = FakeRule()
        self.task_tools = SimpleNamespace(specs=lambda: [], call=self._tool_call)
        self.hook = SimpleNamespace(pending_sessions=())
        self.rules = SimpleNamespace(
            all=self._all_rules,
            active=lambda: [self.rule],
            candidates=lambda: [],
        )

    async def _tool_call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError(f"unexpected live harness tool call: {name} {args}")

    def _all_rules(self) -> list[FakeRule]:
        self.events.append("rules-read")
        return [self.rule]

    def system_context(self, session_id: str, *, task_hint: str | None = None) -> str:
        del session_id, task_hint
        return "live learning harness"

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

    async def drain_validation(self) -> None:
        self.validation_drains += 1
        self.events.append("validation-drain")


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


async def test_dry_run_pipeline_and_resume(tmp_path):
    run_dir = await _runner(tmp_path).run(
        arm="baseline", model_key="gemini-3.1-flash-lite", backend=None, seed=1,
        k=1, limit=2, dry_run=True, phases=("eval",),
    )
    records_file = run_dir / "records.jsonl"
    n_first = len(records_file.read_text().splitlines())
    assert n_first == 2
    assert (run_dir / "manifest.json").exists()

    # Resume: rerun with the same run_id -> every trial is skipped, no duplicates.
    await _runner(tmp_path).run(
        arm="baseline", model_key="gemini-3.1-flash-lite", backend=None, seed=1,
        k=1, limit=2, dry_run=True, phases=("eval",), run_id=run_dir.name,
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
            k=1, limit=1, dry_run=True, phases=("learning", "eval"),
        )
        assert (run_dir / "records.jsonl").exists()


async def test_live_learning_freezes_once_and_eval_never_builds_or_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    single = WiringRecordingRunner()
    live = FakeLiveHarness(events)
    runner = _runner(tmp_path, single)
    builds: list[str] = []

    def fake_build(*args: Any, **kwargs: Any) -> FakeLiveHarness:
        phase = str(args[1] if len(args) > 1 else kwargs["phase"])
        builds.append(phase)
        return live

    monkeypatch.setattr(runner, "_make_client", lambda arm, dry_run: MockClient())
    monkeypatch.setattr(runner, "_build_harness", fake_build)

    run_dir = await runner.run(
        arm="harness", model_key="mock", backend=None, seed=1,
        k=2, limit=1, dry_run=False, phases=("learning", "eval"),
        run_id="frozen-lifecycle",
    )

    assert builds == ["learning"]
    assert live.on_turn_end_calls == live.settle_calls == 2
    assert live.refresh_calls == 1
    assert live.validation_drains == 2
    assert max(i for i, event in enumerate(events) if event == "validation-drain") < max(
        i for i, event in enumerate(events) if event == "rules-read"
    )
    assert all(isinstance(wiring, HarnessWiring) for wiring in single.wirings[:2])
    assert all(isinstance(wiring, FrozenEvalWiring) for wiring in single.wirings[2:])
    assert all("Read the learned state" in content for content in single.frozen_rule_reads)

    records = [
        json.loads(line)
        for line in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    eval_records = [record for record in records if record["phase"] == "eval"]
    assert len({record["harness"]["ruleset_hash"] for record in eval_records}) == 1
    assert all(record["harness"]["mode"] == "frozen_eval" for record in eval_records)
    assert all(record["harness"]["notices"] == 0 for record in eval_records)
    assert all(record["harness"]["scores"] == {} for record in eval_records)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    resolved = manifest["resolved_config"]
    assert resolved["gate_window"] == 10
    assert resolved["repair_model"] == "mock/mock"
    assert resolved["repair_reasoning_effort"] == "none"
    assert resolved["managed_repair"] is True
    assert resolved["trace_repair_agent"] is True
    assert resolved["eval_policy"] == "frozen_rules"
    assert resolved["trace_eval_during_eval"] is False
    assert resolved["ruleset_hash"] == eval_records[0]["harness"]["ruleset_hash"]


async def test_eval_only_harness_uses_explicit_empty_snapshot_without_live_harness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    single = WiringRecordingRunner()
    runner = _runner(tmp_path, single)
    monkeypatch.setattr(runner, "_make_client", lambda arm, dry_run: MockClient())

    def forbidden_build(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(f"eval built a live harness: {args} {kwargs}")

    monkeypatch.setattr(runner, "_build_harness", forbidden_build)
    run_dir = await runner.run(
        arm="harness", model_key="mock", backend=None, seed=1,
        k=1, limit=1, dry_run=False, phases=("eval",), run_id="empty-eval",
    )

    snapshot = json.loads((run_dir / "frozen-rules.json").read_text(encoding="utf-8"))
    record = json.loads((run_dir / "records.jsonl").read_text(encoding="utf-8"))
    assert snapshot["rules"] == []
    assert snapshot["sha256"] == record["harness"]["ruleset_hash"]
    assert record["harness"]["rules_active"] == 0
    assert isinstance(single.wirings[0], FrozenEvalWiring)

    # Resume must trust the existing verified snapshot, not a later mutation in
    # the live workspace. Remove only the synthetic record to force one eval slot
    # to execute again in this temporary test run.
    live_root = run_dir / "harness_root"
    live_root.mkdir(parents=True, exist_ok=True)
    (live_root / "rules.jsonl").write_text('{"id":"late-live-rule"}\n', encoding="utf-8")
    (run_dir / "records.jsonl").unlink()
    resumed = WiringRecordingRunner()
    resume_runner = _runner(tmp_path, resumed)
    monkeypatch.setattr(resume_runner, "_make_client", lambda arm, dry_run: MockClient())
    monkeypatch.setattr(resume_runner, "_build_harness", forbidden_build)
    await resume_runner.run(
        arm="harness", model_key="mock", backend=None, seed=1,
        k=1, limit=1, dry_run=False, phases=("eval",), run_id="empty-eval",
    )
    resumed_record = json.loads((run_dir / "records.jsonl").read_text(encoding="utf-8"))
    assert resumed_record["harness"]["ruleset_hash"] == snapshot["sha256"]
    assert resumed_record["harness"]["rules_active"] == 0


async def test_repeated_setup_gets_a_fresh_remote_session_namespace(tmp_path):
    first = RecordingMockTaskRunner("appworld")
    second = RecordingMockTaskRunner("appworld")

    run_dir = await _runner(tmp_path, first).run(
        arm="baseline", model_key="gemini-3.1-flash-lite", backend=None, seed=1,
        k=1, limit=1, dry_run=True, phases=("eval",), run_id="interrupted-run",
    )
    (run_dir / "records.jsonl").unlink()
    await _runner(tmp_path, second).run(
        arm="baseline", model_key="gemini-3.1-flash-lite", backend=None, seed=1,
        k=1, limit=1, dry_run=True, phases=("eval",), run_id="interrupted-run",
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
            k=1, limit=1, dry_run=True, phases=("eval",),
            dataset_override=dataset, run_id=f"appworld-{dataset}",
        )

    summary = tmp_path / "summary"
    aggregate(tmp_path / "runs", summary)
    with (summary / "headline.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert {row["dataset"] for row in rows} == {"airline", "retail"}
    assert len(rows) == 2
