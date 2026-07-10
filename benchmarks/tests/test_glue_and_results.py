"""Offline tests for session-id stability, the record schema, and the
load-bearing arm-B capture path (on_turn_end end_state -> replayable EvalCase).
Uses a fake ``pandaprobe`` CLI client — no network, no real platform."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pandaprobe_harness import CliResult, HarnessConfig

from pandabench.agents.harness_wiring import HarnessWiring
from pandabench.agents.loop import run_agent_loop
from pandabench.harness_glue import (
    build_harness,
    make_session_id,
    project_name_for,
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


def test_session_id_stable_and_recomputable():
    kw = dict(benchmark="appworld", task_id="82e2fac_1", arm="harness",
              model_key="claude-sonnet-4-6", seed=1, trial=0)
    a = make_session_id(**kw)
    b = make_session_id(**kw)
    assert a == b == "appworld-82e2fac_1-harness-claude-sonnet-4-6-1-t0"


def test_project_name():
    assert project_name_for("tau2", "baseline") == "bench-tau2-baseline"


# -- record schema ------------------------------------------------------------


def test_trial_record_round_trip():
    rec = TrialRecord(
        run_id="r1", benchmark="appworld", task_id="t1", arm="harness",
        model="gemini-2.5-flash", provider="vertex", backend=None,
        resolved_model="vertex_ai/gemini-2.5-flash", seed=1, trial=0, phase="eval",
        passed=True, native_metrics={"tgc": 1.0}, turns=3, wall_time_s=12.5,
        usage={"input_tokens": 100, "output_tokens": 20, "cost_usd": 0.01},
        harness={"session_id": "s", "rules_active": 2}, error=None,
    )
    restored = TrialRecord.from_json(json.loads(json.dumps(rec.to_json())))
    assert restored == rec
    assert restored.resume_key == resume_key(
        "appworld", "t1", "harness", "gemini-2.5-flash", None, 1, 0, "eval"
    )


def test_resume_key_normalizes_backend():
    assert resume_key("b", "t", "a", "m", None, 1, 0, "eval")[4] == ""
    assert resume_key("b", "t", "a", "m", "vertex_ai", 1, 0, "eval")[4] == "vertex_ai"


# -- the arm-B capture path (fake CLI) ----------------------------------------


class FakeCli:
    """In-process ``pandaprobe`` stand-in returning breaching session scores."""

    def __init__(self) -> None:
        self._runs: dict[str, str] = {}
        self._n = 0

    async def run(self, *args: str, timeout: float | None = None) -> CliResult:
        payload = self._dispatch(args)
        return CliResult(args=args, exit_code=0, stdout=json.dumps(payload), stderr="")

    def _dispatch(self, args: tuple[str, ...]) -> Any:
        if args[:1] == ("version",):
            return {"version": "v-test"}
        if args[:2] == ("auth", "status"):
            return {"authenticated": True}
        if args[:3] == ("evals", "runs", "batch"):
            self._n += 1
            run_id = f"run-{self._n}"
            self._runs[run_id] = args[args.index("--session-ids") + 1]
            return {"id": run_id, "status": "PENDING"}
        if args[:3] == ("evals", "runs", "scores"):
            return [
                {"name": "agent_reliability", "value": "0.30", "status": "SUCCESS",
                 "reason": "breaching", "metadata": {"flagged_traces": ["tr-1"]}},
                {"name": "agent_consistency", "value": "0.40", "status": "SUCCESS",
                 "reason": "breaching", "metadata": {}},
            ]
        return {}


async def test_on_turn_end_capture_yields_replayable_eval_case(tmp_path):
    cfg = HarnessConfig(
        harness_root=tmp_path / "hroot",
        capture_eval_cases=True,
        poll_interval_s=0.0,
        poll_max_attempts=3,
        eval_retry_backoff_s=0.0,
        health_check=False,
        rule_trial_min_sessions=1,
    )
    harness = build_harness(cfg=cfg)
    # Swap in the fake CLI (build_harness uses the real binary by default).
    harness._cli = FakeCli()  # type: ignore[attr-defined]
    harness._evaluator._cli = FakeCli()  # type: ignore[attr-defined]
    harness._hook._cli = FakeCli()  # type: ignore[attr-defined]

    registry = load_registry(CONFIGS / "models.yaml")
    model = registry.resolve("mock")
    session_id = make_session_id(
        benchmark="appworld", task_id="t1", arm="harness",
        model_key="mock", seed=1, trial=0,
    )
    descriptor = {"benchmark": "appworld", "task_id": "t1", "arm": "harness",
                  "model_key": "mock", "backend": None, "seed": 1, "trial": 0}
    wiring = HarnessWiring(
        harness=harness, benchmark="appworld", task_id="t1", capture=True,
        replay_descriptor=descriptor,
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
        task_hint="do the task",
    )
    assert result.stopped_reason == "final"
    assert result.turns == 1

    # Runner-side session lifecycle (once per trial), mirroring runners/base.py.
    harness.on_turn_end(
        {"session_id": session_id, "turn_index": 1, "end_state": wiring.end_state()}
    )
    report = await harness.refresh(session_id)
    await harness.drain_validation()

    assert report is not None and report.any_breach is True

    cases = harness.evalset.cases()
    assert len(cases) == 1
    case = cases[0]
    assert case.replayable is True
    assert case.replay_input["task_id"] == "t1"
    assert case.replay_input["benchmark"] == "appworld"

    telemetry = collect_harness_telemetry(harness, session_id, report)
    assert telemetry.reliability == 0.30
    assert telemetry.breached is True


async def _noop_executor(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"ok": True}
