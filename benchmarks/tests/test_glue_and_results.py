"""Offline tests for session-id stability, the record schema, and the
load-bearing arm-B capture path (on_turn_end end_state -> replayable EvalCase).
Uses a fake ``pandaprobe`` CLI client — no network, no real platform."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pandaprobe_harness import CliResult, Harness, HarnessConfig

from pandabench.agents.harness_wiring import HarnessWiring
from pandabench.agents.loop import run_agent_loop
from pandabench.checkpoints import records_to_labels
from pandabench.harness_glue import (
    make_session_id,
    make_verifier_fn,
    new_session_namespace,
    sanitize_component,
)
from pandabench.providers.litellm_client import MockClient
from pandabench.providers.models import load_registry
from pandabench.results import TrialRecord, collect_harness_telemetry, resume_key

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


# -- session ids --------------------------------------------------------------


def test_sanitize_component():
    assert sanitize_component("Retail/Task 12") == "retail-task-12"
    assert sanitize_component("  A B  ") == "a-b"
    assert sanitize_component("82e2fac_1") == "82e2fac_1"  # underscores are valid
    assert sanitize_component("!!!") == "x"


def test_session_id_stable_within_one_invocation():
    namespace = "0123456789abcdef0123456789abcdef"
    kw = dict(
        session_namespace=namespace, benchmark="appworld", task_id="82e2fac_1",
        arm="harness", model_key="claude-sonnet-4-6", seed=1, trial=0,
        phase="learning",
    )
    a = make_session_id(**kw)
    b = make_session_id(**kw)
    assert a == b == (
        "appworld-82e2fac_1-harness-claude-sonnet-4-6-s1-learning-t0-"
        "r0123456789abcdef0123456789abcdef"
    )


def test_session_ids_change_across_invocations_and_phases():
    common = dict(
        benchmark="tau2", task_id="37", arm="harness", model_key="claude-sonnet-4-6",
        seed=1, trial=0,
    )
    first_namespace = new_session_namespace()
    second_namespace = new_session_namespace()

    learning = make_session_id(
        session_namespace=first_namespace, phase="learning", **common
    )
    repeated_run = make_session_id(
        session_namespace=second_namespace, phase="learning", **common
    )
    eval_session = make_session_id(
        session_namespace=first_namespace, phase="eval", **common
    )

    assert first_namespace != second_namespace
    assert len({learning, repeated_run, eval_session}) == 3


def test_session_id_respects_platform_length_limit_without_losing_namespace():
    namespace = "f" * 32
    session_id = make_session_id(
        session_namespace=namespace, benchmark="terminal_bench", task_id="t" * 400,
        arm="harness", model_key="m" * 100, seed=1, trial=0, phase="learning",
    )
    same_prefix = make_session_id(
        session_namespace=namespace, benchmark="terminal_bench", task_id=("t" * 399) + "x",
        arm="harness", model_key="m" * 100, seed=1, trial=0, phase="learning",
    )
    assert len(session_id) == 255
    assert session_id.endswith(f"-r{namespace}")
    assert session_id != same_prefix


# -- record schema ------------------------------------------------------------


def test_trial_record_round_trip():
    rec = TrialRecord(
        run_id="r1", benchmark="appworld", task_id="t1", arm="harness",
        model="gemini-3.1-flash-lite", provider="vertex", backend=None,
        resolved_model="vertex_ai/gemini-3.1-flash-lite", seed=1, trial=0, phase="eval",
        passed=True, native_metrics={"tgc": 1.0}, turns=3, wall_time_s=12.5,
        usage={"input_tokens": 100, "output_tokens": 20, "cost_usd": 0.01},
        harness={"session_id": "s", "rules_active": 2}, error=None,
    )
    restored = TrialRecord.from_json(json.loads(json.dumps(rec.to_json())))
    assert restored == rec
    assert restored.resume_key == resume_key(
        "appworld", "t1", "harness", "gemini-3.1-flash-lite", None, 1, 0, "eval"
    )


def test_resume_key_normalizes_backend():
    assert resume_key("b", "t", "a", "m", None, 1, 0, "eval")[4] == ""
    assert resume_key("b", "t", "a", "m", "vertex_ai", 1, 0, "eval")[4] == "vertex_ai"


def test_calibration_uses_recorded_session_id(tmp_path):
    records = tmp_path / "records.jsonl"
    labels = tmp_path / "labels.json"
    current = {
        "schema_version": 2, "benchmark": "tau2", "task_id": "37", "arm": "harness",
        "model": "claude-sonnet-4-6", "seed": 1, "trial": 0, "phase": "learning",
        "passed": True, "harness": {"session_id": "actual-namespaced-session"},
    }
    records.write_text(json.dumps(current) + "\n", encoding="utf-8")

    assert records_to_labels(records, labels, benchmark="tau2") == 1
    assert json.loads(labels.read_text(encoding="utf-8")) == [
        {"session_id": "actual-namespaced-session", "failed": False},
    ]


def test_outcome_verifier_forwards_task_and_session_id():
    outcomes = {("same-task", "session-a"): 0.25, ("same-task", "session-b"): 1.0}
    verifier = make_verifier_fn(outcome_for=lambda task, session: outcomes.get((task, session)))

    assert verifier("session-a", {"task_id": "same-task"}) == 0.25
    assert verifier("session-b", {"task_id": "same-task"}) == 1.0
    assert verifier("unknown", {"task_id": "same-task"}) is None


# -- the arm-B capture path (fake CLI) ----------------------------------------


class FakeCli:
    """In-process ``pandaprobe`` stand-in serving the trace-target surface.

    Scores a flat, below-target trajectory, which is what the trajectory gate
    flags: three traces at ``task_completion=0.2`` stall the window, and Tier 2
    then confirms a step-level breach on the last trace.
    """

    def __init__(self) -> None:
        self._runs: dict[str, tuple[list[str], list[str]]] = {}
        self._n = 0
        self.traces = ["tr-1", "tr-2", "tr-3"]

    async def run(self, *args: str, timeout: float | None = None) -> CliResult:
        payload = self._dispatch(args)
        return CliResult(args=args, exit_code=0, stdout=json.dumps(payload), stderr="")

    def _dispatch(self, args: tuple[str, ...]) -> Any:
        if args[:1] == ("version",):
            return {"version": "v-test"}
        if args[:2] == ("auth", "status"):
            return {"authenticated": True}
        if args[:2] == ("traces", "list"):
            return {
                "items": [
                    {"trace_id": tid, "status": "COMPLETED",
                     "started_at": f"2026-01-01T00:00:{i:02d}+00:00"}
                    for i, tid in reversed(list(enumerate(self.traces)))
                ]
            }
        if args[:3] == ("evals", "runs", "batch"):
            self._n += 1
            run_id = f"run-{self._n}"
            # Tier 1 batches the whole turn, so a run may cover several traces.
            self._runs[run_id] = (
                args[args.index("--trace-ids") + 1].split(","),
                args[args.index("--metrics") + 1].split(","),
            )
            return {"id": run_id, "status": "PENDING"}
        if args[:3] == ("evals", "runs", "scores"):
            trace_ids, metrics = self._runs.get(args[3], ([], []))
            return [
                {"name": m, "trace_id": t, "value": "0.20", "status": "SUCCESS",
                 "reason": "breaching", "metadata": {"threshold": 0.5}}
                for t in trace_ids
                for m in metrics
                if m
            ]
        return {}


async def test_on_turn_end_capture_yields_replayable_eval_case(tmp_path):
    cfg = HarnessConfig(
        harness_root=tmp_path / "hroot",
        capture_eval_cases=True,
        poll_interval_s=0.0,
        poll_max_attempts=3,
        eval_retry_backoff_s=0.0,
        eval_retry_attempts=1,
        health_check=False,
        rule_trial_min_sessions=1,
        gate_window=2,  # two flat traces close the stall window in this short test
    )
    # `build_harness` deliberately has no `cli=` seam (it always drives the real
    # binary), so this test assembles the harness directly over one shared fake.
    # The subject here is the *glue* — wiring, session ids, capture, telemetry.
    harness = Harness.create(cfg, cli=FakeCli())

    registry = load_registry(CONFIGS / "models.yaml")
    model = registry.resolve("mock")
    session_id = make_session_id(
        session_namespace="test-namespace", benchmark="appworld", task_id="t1",
        arm="harness", model_key="mock", seed=1, trial=0, phase="learning",
    )
    descriptor = {"benchmark": "appworld", "task_id": "t1", "arm": "harness",
                  "model_key": "mock", "backend": None, "seed": 1, "trial": 0}
    wiring = HarnessWiring(
        harness=harness, benchmark="appworld", task_id="t1", capture=True,
        replay_descriptor=descriptor,
        # The loop settles each turn through the wiring; MockClient has no
        # flush, so none is passed.
        session_id=session_id,
    )

    result = await run_agent_loop(
        client=MockClient(),
        model=model,
        session_id=session_id,
        system_prompt="You are an agent.",
        tools=[],
        tool_executor=_noop_executor,
        initial_messages=[{"role": "user", "content": "do the task"}],
        max_turns=4,
        wiring=wiring,
    )
    assert result.stopped_reason == "final"
    assert result.turns == 1

    # Runner-side lifecycle, mirroring runners/base.py: the loop settles each turn
    # that continues, and the runner settles the final turn — taking the report
    # from that call, because the barrier consumes the pending evaluation.
    settled = await wiring.settle_turn(1)
    assert settled is not None
    report = settled.report

    assert report is not None
    # Tier 1 stalled on the flat trajectory, Tier 2 then confirmed the breach —
    # so this is a `breach`-severity finding and becomes a failure case.
    assert report.gate_breached is True
    assert report.any_breach is True
    assert {s.trace_id for s in report.scores_for_tier(1)} == {"tr-1", "tr-2", "tr-3"}
    assert {s.trace_id for s in report.scores_for_tier(2)} == {"tr-3"}

    cases = harness.evalset.cases()
    assert len(cases) == 1
    case = cases[0]
    assert case.replayable is True
    assert case.replay_input["task_id"] == "t1"
    assert case.replay_input["benchmark"] == "appworld"

    telemetry = collect_harness_telemetry(harness, session_id, report)
    assert telemetry.breached is True


async def test_settling_a_turn_twice_returns_the_first_diagnosis(tmp_path):
    """At the turn cap the loop's last turn IS the trial's last, so the loop and
    the runner can both settle the same index. The second call must not re-evaluate:
    the traces have already been delivered, so a fresh report would carry none of
    the tier scores, and a caller recording telemetry from it would write that
    emptiness over the real diagnosis."""

    cfg = HarnessConfig(
        harness_root=tmp_path / "hroot",
        poll_interval_s=0.0,
        poll_max_attempts=3,
        eval_retry_backoff_s=0.0,
        eval_retry_attempts=1,
        health_check=False,
        gate_window=2,
    )
    cli = FakeCli()
    harness = Harness.create(cfg, cli=cli)
    wiring = HarnessWiring(
        harness=harness, benchmark="appworld", task_id="t1", capture=True,
        replay_descriptor={}, session_id="s-dup",
    )

    first = await wiring.settle_turn(7)
    assert first is not None and first.report is not None
    tier1 = {s.trace_id for s in first.report.scores_for_tier(1)}
    assert tier1 == {"tr-1", "tr-2", "tr-3"}

    runs_before = cli._n
    again = await wiring.settle_turn(7)

    assert again is first  # same answer, not a fresh (impoverished) one
    assert cli._n == runs_before  # and no second platform eval run
    assert wiring.turns_settled == 1

    # A genuinely new turn still settles.
    assert await wiring.settle_turn(8) is not first


async def _noop_executor(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True}
