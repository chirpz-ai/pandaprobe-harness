from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pandas as pd
import pytest

from pandabench.agents.frozen_wiring import FrozenEvalWiring
from pandabench.frozen_rules import FrozenRulesSnapshot
from pandabench.providers.litellm_client import ChatResult, MockClient, ToolCall, Usage
from pandabench.providers.models import load_registry
from pandabench.report import (
    _harbor_test_counts,
    _parse_pytest_counts,
    _recovered_tests,
    _relaxed,
    _score,
    _score_is_graded,
)
from pandabench.runners.appworld import AppWorldRunner, _app_scope_hints, _experiment_name
from pandabench.runners.appworld_env import (
    AppWorldServer,
    EvalResult,
    HttpAppWorldEnv,
    TaskInfo,
    _failure_requirements,
)
from pandabench.runners.tau2 import Tau2Runner, _safe_task_workflow

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


class SequenceAppWorldEnv:
    def __init__(self, verdicts: list[EvalResult]) -> None:
        self.verdicts = list(verdicts)
        self.experiments: list[tuple[str, str]] = []

    def list_task_ids(self, dataset: str) -> list[str]:
        del dataset
        return ["same-task"]

    def initialize(self, task_id: str, *, experiment_name: str) -> TaskInfo:
        self.experiments.append((task_id, experiment_name))
        return TaskInfo(task_id, "Do the task", {}, None)

    def api_docs(self) -> str:
        return "docs"

    def execute(self, task_id: str, code: str) -> str:
        del task_id, code
        return "ok"

    def evaluate(self, task_id: str) -> EvalResult:
        del task_id
        return self.verdicts.pop(0)

    def close(self, task_id: str) -> None:
        del task_id


def _verdict(passes: int, tests: int = 4) -> EvalResult:
    return EvalResult(passes == tests, tests, passes, 1, {})


class RecordingToolsClient(MockClient):
    def __init__(self, *, scripted: list[ChatResult]) -> None:
        super().__init__(scripted=scripted)
        self.tool_names: list[list[str]] = []
        self.message_batches: list[list[dict[str, Any]]] = []

    async def chat(self, **kwargs: Any) -> ChatResult:
        self.tool_names.append([
            str((schema.get("function") or {}).get("name", ""))
            for schema in kwargs.get("tools") or []
        ])
        self.message_batches.append(list(kwargs["messages"]))
        return await super().chat(**kwargs)


class NoSettleFrozenWiring(FrozenEvalWiring):
    async def settle_turn(self, turn_index: int) -> None:
        raise AssertionError(f"frozen AppWorld eval settled turn {turn_index}")


def test_score_is_one_scale_across_benchmarks_and_flags_partial_credit() -> None:
    """The report's relaxation rests on one comparable 0-1 score per trial."""

    # AppWorld: the passing-test fraction, whatever the verdict says.
    assert _score({"num_tests": 4, "num_passes": 3, "pass_ratio": 0.75}, False) == 0.75
    assert _score({"num_tests": 4, "num_passes": 4, "pass_ratio": 1.0}, True) == 1.0
    # tau2 / Terminal-Bench: reward, which is binary in practice.
    assert _score({"reward": 1.0}, True) == 1.0
    assert _score({"reward": 0.0}, False) == 0.0
    # Neither signal: fall back to the verdict so the column is never missing.
    assert _score({}, True) == 1.0
    assert _score({}, False) == 0.0
    # A bool reward must not be mistaken for a numeric score.
    assert _score({"reward": True, "pass_ratio": 0.25}, False) == 0.25

    assert _score_is_graded({"num_tests": 4, "num_passes": 3}) is True
    assert _score_is_graded({"reward": 0.0}) is False
    # A single-test task carries no partial credit either.
    assert _score_is_graded({"num_tests": 1, "num_passes": 0}) is False


def _relax_frame(scores: list[float], arm: str) -> pd.DataFrame:
    """A minimal frame for `_relaxed`: score + the arm gate + the native verdict."""

    return pd.DataFrame(
        {
            "score": scores,
            "arm": [arm] * len(scores),
            # The benchmark's own verdict: a perfect score and nothing less.
            "passed": [s >= 1.0 for s in scores],
        }
    )


def test_relaxation_is_a_fraction_so_it_means_the_same_at_every_task_size() -> None:
    """Why the tolerance is fractional rather than an absolute missed-test count."""

    scores = [
        2 / 2, 1 / 2,      # 2-test task: perfect, one short
        10 / 10, 9 / 10,   # 10-test task: perfect, one short
        20 / 20, 19 / 20,  # 20-test task: perfect, one short
    ]
    harness = _relax_frame(scores, "harness")
    # At 0.10, one missed test is forgiven on the 10- and 20-test tasks but not on
    # the 2-test one, where a single test is half the task.
    assert list(_relaxed(harness, 0.10)) == [True, False, True, True, True, True]
    # At 0.0 only a perfect score passes, on every task size.
    assert list(_relaxed(harness, 0.0)) == [True, False, True, False, True, False]
    # Float division must not lose a boundary: 9/10 is exactly at 1 - 0.10.
    assert bool(_relaxed(_relax_frame([9 / 10], "harness"), 0.10).iloc[0]) is True


def test_tau2_reward_components_recover_partial_credit() -> None:
    """tau2's `reward` throws away detail its own evaluator computed."""

    def episode(db: float, communicate: float) -> dict[str, Any]:
        return {
            # tau2's own reward: min() of the components.
            "reward": min(db, communicate),
            "reward_breakdown": {
                "RewardType.DB": db,
                "RewardType.COMMUNICATE": communicate,
            },
        }

    both, comm, neither = episode(1.0, 1.0), episode(0.0, 1.0), episode(0.0, 0.0)

    assert _score(both, True) == 1.0
    assert _score(comm, False) == 0.5      # was 0.0 via `reward`
    assert _score(neither, False) == 0.0
    assert _score_is_graded(comm) is True

    # A lone component is no finer than `reward`, so it must not count as graded.
    single = {"reward": 0.0, "reward_breakdown": {"RewardType.DB": 0.0}}
    assert _score(single, False) == 0.0
    assert _score_is_graded(single) is False
    # No breakdown at all: fall through to `reward`.
    assert _score({"reward": 1.0}, True) == 1.0
    assert _score_is_graded({"reward": 1.0}) is False


def test_harbor_per_test_counts_are_recovered_from_the_archived_verifier_log(
    tmp_path: Path,
) -> None:
    """Terminal-Bench's binary reward is a threshold, not the absence of a signal."""

    log = tmp_path / "raw" / "eval" / "build-ext__ABC" / "verifier" / "test-stdout.txt"
    log.parent.mkdir(parents=True)
    log.write_text(
        # Real shape: build chatter, the session, then a LATER banner from a reinstall.
        "Get:1 http://deb.debian.org/debian bookworm InRelease [151 kB]\n"
        "============================= test session starts ==============================\n"
        "collected 11 items\n"
        "../tests/test_outputs.py .......F...                                     [100%]\n"
        "========================= 1 failed, 10 passed in 3.46s =========================\n",
        encoding="utf-8",
    )
    counts = _harbor_test_counts(tmp_path)
    assert counts == {"build-ext__ABC": (10, 11)}

    native = {"reward": 0.0, "harbor_trial_name": "build-ext__ABC"}
    tests = _recovered_tests(native, counts)
    assert tests == (10, 11)
    # Harbor said 0.0; the log says 10 of 11.
    assert _score(native, False, tests) == pytest.approx(10 / 11)
    assert _score_is_graded(native, tests) is True
    # Unknown trial name -> no recovery, and Harbor's own reward stands unchanged.
    assert _recovered_tests({"harbor_trial_name": "other"}, counts) is None
    assert _score(native, False, None) == 0.0
    assert _score_is_graded(native, None) is False

    # A run with no raw/ tree (AppWorld, tau2) yields nothing and must not raise.
    assert _harbor_test_counts(tmp_path / "nonexistent") == {}


def test_verifier_log_parsing_survives_real_world_noise(tmp_path: Path) -> None:
    """Only a genuine pytest tally counts; anything else must yield None, not a guess."""

    def parsed(text: str) -> tuple[int, int] | None:
        path = tmp_path / "log.txt"
        path.write_text(text, encoding="utf-8")
        return _parse_pytest_counts(path)

    assert parsed("==== 3 passed in 1.0s ====") == (3, 3)
    assert parsed("==== 2 failed, 3 passed in 1.0s ====") == (3, 5)
    assert parsed("==== 1 error, 1 passed in 1.0s ====") == (1, 2)
    # Skipped is not signal about the agent, so it leaves the denominator alone.
    assert parsed("==== 1 passed, 4 skipped in 1.0s ====") == (1, 1)
    # The LAST real tally wins: verifiers reinstall and print more banners after.
    assert parsed(
        "==== 5 passed in 1.0s ====\n==== 1 failed, 1 passed in 2.0s ====\n"
    ) == (1, 2)
    # A banner without a duration is a section header, not a tally.
    assert parsed("==== FAILURES ====\n==== short test summary info ====") is None
    assert parsed("no pytest here at all") is None
    # Everything skipped: no gradable unit, so no score rather than a 0/0 crash.
    assert parsed("==== 4 skipped in 1.0s ====") is None


def test_relaxation_never_reaches_the_baseline_arm() -> None:
    """The baseline is held to the benchmark's own verdict at any tolerance."""

    scores = [1.0, 0.9, 0.5]
    for relax in (0.0, 0.1, 0.5, 0.9):
        baseline = _relaxed(_relax_frame(scores, "baseline"), relax)
        # Identical to the native verdict, whatever the tolerance.
        assert list(baseline) == [True, False, False]

    # The same scores in the harness arm do move with the tolerance.
    assert list(_relaxed(_relax_frame(scores, "harness"), 0.5)) == [True, True, True]


async def test_appworld_records_which_tests_failed_bounded() -> None:
    """Persist the failing tests' identities, capped, and never their tracebacks."""

    raw_failures = [
        {"requirement": f"  assert thing {i} matches\n ", "trace": "x" * 5000, "label": None}
        for i in range(20)
    ]
    failures, truncated = _failure_requirements(raw_failures)
    env = SequenceAppWorldEnv([
        EvalResult(False, 24, 4, 1, {}, failures, truncated)
    ])
    runner = AppWorldRunner(env)
    model = load_registry(CONFIGS / "models.yaml").resolve("mock")

    outcome = await runner.run_once(
        task_id="same-task", session_id="s1", model=model, client=MockClient(),
        max_turns=2, wiring=None,
    )

    failing = outcome.native_metrics["failing_tests"]
    assert failing[0] == "assert thing 0 matches"  # whitespace collapsed
    assert len(failing) == 12  # capped in count
    assert outcome.native_metrics["failing_tests_truncated"] is True
    assert all(len(text) <= 240 for text in failing)
    assert "x" * 100 not in json.dumps(outcome.native_metrics)  # no traceback


def test_http_evaluate_parses_the_real_payload_shape() -> None:
    """Pin the shape AppWorld's ``/evaluate`` returns.

    Verified against ``appworld==0.1.3.post1``: ``TestTracker.to_dict`` wrapped in
    ``{"output": ...}``, with ``num_passes`` derivable only by counting ``passes``.
    """

    payload = {
        "output": {
            "success": False,
            "difficulty": 2,
            "num_tests": 3,
            "passes": [
                {"requirement": "assert answers match.", "label": None},
                {"requirement": "assert model changes match.", "label": "no_op_fail"},
            ],
            "failures": [
                {
                    "requirement": "assert added receiver ids match.",
                    "trace": "```python\n...\n```\nAssertionError: nope",
                    "label": "no_op_fail",
                }
            ],
        }
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    env = HttpAppWorldEnv(
        "http://appworld.test", appworld_root=Path("/nonexistent"), transport=transport
    )

    verdict = env.evaluate("task-1")

    assert verdict.success is False
    assert verdict.num_tests == 3
    assert verdict.num_passes == 2  # counted from `passes`
    assert verdict.difficulty == 2
    assert verdict.failures == ("assert added receiver ids match.",)
    assert verdict.failures_truncated is False
    # The traceback is reachable in `raw` for debugging but is not what we persist.
    assert "AssertionError" not in json.dumps(list(verdict.failures))


def test_failure_requirements_bounds_length_and_tolerates_junk() -> None:
    long_requirement = "y" * 400
    bounded, truncated = _failure_requirements(
        [{"requirement": long_requirement, "trace": "t"}]
    )
    assert len(bounded[0]) == 240
    assert bounded[0].endswith("…")
    assert truncated is True

    # A short list is kept verbatim and reported untruncated.
    assert _failure_requirements([{"requirement": "one"}]) == (("one",), False)
    # Malformed payloads degrade to empty rather than raising during grading.
    assert _failure_requirements(None) == ((), False)
    assert _failure_requirements(["not-a-dict", {"requirement": ""}]) == ((), False)


async def test_appworld_outcomes_and_experiments_are_session_scoped() -> None:
    env = SequenceAppWorldEnv([_verdict(1), _verdict(4), _verdict(3)])
    runner = AppWorldRunner(env)
    model = load_registry(CONFIGS / "models.yaml").resolve("mock")

    await runner.run_once(
        task_id="same-task", session_id="session-a", model=model, client=MockClient(),
        max_turns=2, wiring=None,
    )
    await runner.run_once(
        task_id="same-task", session_id="session-b", model=model, client=MockClient(),
        max_turns=2, wiring=None,
    )
    await runner.run_once(
        task_id="same-task", session_id="replay-session", model=model, client=MockClient(),
        max_turns=2, wiring=None,
    )

    assert runner.outcome_for("same-task", "session-a") == 0.25
    assert runner.outcome_for("same-task", "session-b") == 1.0
    assert runner.outcome_for("same-task", "replay-session") == 0.75
    assert runner.outcome_for("same-task", "unknown") is None
    assert len({experiment for _, experiment in env.experiments}) == 3
    assert _experiment_name("pandabench", "session-a") == _experiment_name(
        "pandabench", "session-a"
    )
    assert _experiment_name("unsafe name!", "session-a").startswith("unsafe-name-")


async def test_appworld_frozen_eval_reads_rules_and_only_runs_native_grading() -> None:
    env = SequenceAppWorldEnv([_verdict(4)])
    runner = AppWorldRunner(env)
    model = load_registry(CONFIGS / "models.yaml").resolve("mock")
    snapshot = FrozenRulesSnapshot.create(
        [{
            "id": "r-learned",
            "created_at": "2026-08-05T00:00:00+00:00",
            "rule": "Read the order before refunding it.",
            "rationale": "Learned from the training split.",
            "source_notice_id": "n-1",
            "metric": "task_completion",
            "status": "active",
            "tags": ["order", "refund"],
            "trial": None,
            "scope": "global",
        }],
        created_at="2026-08-05T01:00:00+00:00",
    )
    wiring = NoSettleFrozenWiring(snapshot)
    client = RecordingToolsClient(
        scripted=[
            ChatResult(
                assistant_message={
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "read-1", "type": "function",
                        "function": {
                            "name": "harness_rules_read",
                            "arguments": '{"scope":"global"}',
                        },
                    }],
                },
                tool_calls=[ToolCall("read-1", "harness_rules_read", {"scope": "global"})],
                usage=Usage(), finish_reason="tool_calls", resolved_model="mock",
            ),
            ChatResult(
                assistant_message={"role": "assistant", "content": "done"},
                tool_calls=[], usage=Usage(), finish_reason="stop", resolved_model="mock",
            ),
        ]
    )

    before = snapshot.to_dict()
    outcome = await runner.run_once(
        task_id="same-task", session_id="frozen-session", model=model,
        client=client, max_turns=3, wiring=wiring,
    )

    assert outcome.passed is True
    assert outcome.native_metrics["pass_ratio"] == 1.0
    assert runner.outcome_for("same-task", "frozen-session") == 1.0
    assert "Read the order before refunding" in client.message_batches[1][-1]["content"]
    assert set(client.tool_names[0]) == {
        "execute", "harness_rules_read", "harness_rules_search",
        "harness_rules_list", "harness_rule_status",
    }
    assert all("harness_rule_add" not in tools for tools in client.tool_names)
    assert wiring.pending_notice_ids() == ()
    assert snapshot.to_dict() == before


def test_tau2_outcomes_are_session_scoped() -> None:
    runner = Tau2Runner()
    runner._outcomes.update({"session-a": 0.25, "session-b": 1.0, "replay-session": 0.75})

    assert runner.outcome_for("same-task", "session-a") == 0.25
    assert runner.outcome_for("same-task", "session-b") == 1.0
    assert runner.outcome_for("same-task", "replay-session") == 0.75
    assert runner.outcome_for("same-task", "unknown") is None


def test_appworld_and_tau2_expose_semantic_scope_hints_without_task_ids() -> None:
    app_hints = _app_scope_hints(
        "Send a Venmo reminder, then update my Spotify playlist.",
        "- spotify: search, create_playlist\n- venmo: send_reminder\n- supervisor: show",
    )
    assert [hint.key for hint in app_hints] == ["venmo", "spotify"]
    assert app_hints[0].recommended is True
    assert all("workflows" in hint.description.casefold() for hint in app_hints)

    tau = Tau2Runner(domain="airline")
    domain_hint = tau.rule_scope_hints("opaque-task-id")
    assert domain_hint[0].key == "airline"
    assert domain_hint[0].recommended is True
    assert _safe_task_workflow(
        SimpleNamespace(metadata={"workflow": "change-flight"})
    ) == "change-flight"


def test_http_500_logs_bounded_response_body(caplog: pytest.LogCaptureFixture) -> None:
    traceback = "Traceback: appworld exploded"
    omitted = "SHOULD_NOT_BE_LOGGED"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text=traceback + ("x" * 5000) + omitted, request=request)

    env = HttpAppWorldEnv(
        "http://appworld.test",
        appworld_root=Path("."),
        transport=httpx.MockTransport(handler),
    )
    with caplog.at_level(logging.ERROR, logger="pandabench.appworld"):
        with pytest.raises(httpx.HTTPStatusError):
            env.execute("task-1", "print('secret request body')")
    env.aclose()

    assert "endpoint=/execute" in caplog.text
    assert "status=500" in caplog.text
    assert traceback in caplog.text
    assert omitted not in caplog.text
    assert "secret request body" not in caplog.text


class RecoveringAppWorldEnv(SequenceAppWorldEnv):
    def __init__(self, events: list[str]) -> None:
        super().__init__([_verdict(4), _verdict(4)])
        self.events = events
        self.execute_calls = 0

    def initialize(self, task_id: str, *, experiment_name: str) -> TaskInfo:
        self.events.append("initialize")
        return super().initialize(task_id, experiment_name=experiment_name)

    def execute(self, task_id: str, code: str) -> str:
        del task_id, code
        self.execute_calls += 1
        self.events.append("execute")
        request = httpx.Request("POST", "http://appworld.test/execute")
        response = httpx.Response(500, request=request, text="server traceback")
        raise httpx.HTTPStatusError("server error", request=request, response=response)

    def evaluate(self, task_id: str) -> EvalResult:
        self.events.append("evaluate")
        return super().evaluate(task_id)

    def close(self, task_id: str) -> None:
        del task_id
        self.events.append("close")


class RecordingServer:
    owns_process = True

    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.restarts = 0

    def restart(self) -> str:
        self.restarts += 1
        self.events.append("restart")
        return "http://127.0.0.1:9000"

    def stop(self) -> None:
        self.events.append("stop")


def _execute_then_finish_client() -> MockClient:
    return MockClient(
        scripted=[
            ChatResult(
                assistant_message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "execute", "arguments": '{"code":"x"}'},
                    }],
                },
                tool_calls=[ToolCall("call-1", "execute", {"code": "x"})],
                usage=Usage(),
                finish_reason="tool_calls",
                resolved_model="mock",
            ),
            ChatResult(
                assistant_message={"role": "assistant", "content": "done"},
                tool_calls=[],
                usage=Usage(),
                finish_reason="stop",
                resolved_model="mock",
            ),
        ]
    )


async def test_owned_server_restarts_after_5xx_without_retrying_execute() -> None:
    events: list[str] = []
    env = RecoveringAppWorldEnv(events)
    server = RecordingServer(events)
    runner = AppWorldRunner(env, server=server)  # type: ignore[arg-type]
    model = load_registry(CONFIGS / "models.yaml").resolve("mock")

    first = await runner.run_once(
        task_id="same-task", session_id="session-a", model=model,
        client=_execute_then_finish_client(), max_turns=2, wiring=None,
    )

    assert first.passed is False
    assert first.error is not None and first.error.startswith("execute:")
    assert runner.outcome_for("same-task", "session-a") is None
    assert env.execute_calls == 1
    assert server.restarts == 1
    assert events.index("close") < events.index("restart")

    second = await runner.run_once(
        task_id="same-task", session_id="session-b", model=model, client=MockClient(),
        max_turns=2, wiring=None,
    )
    assert second.passed is True
    assert events[-3:] == ["initialize", "evaluate", "close"]
    assert server.restarts == 1


class FailingEvaluationEnv(SequenceAppWorldEnv):
    def __init__(self, events: list[str]) -> None:
        super().__init__([])
        self.events = events

    def evaluate(self, task_id: str) -> EvalResult:
        del task_id
        self.events.append("evaluate")
        request = httpx.Request("POST", "http://appworld.test/evaluate")
        response = httpx.Response(500, request=request, text="evaluation traceback")
        raise httpx.HTTPStatusError("server error", request=request, response=response)

    def close(self, task_id: str) -> None:
        del task_id
        self.events.append("close")


async def test_evaluation_5xx_preserves_eval_error_and_restarts_after_close() -> None:
    events: list[str] = []
    server = RecordingServer(events)
    runner = AppWorldRunner(FailingEvaluationEnv(events), server=server)  # type: ignore[arg-type]
    model = load_registry(CONFIGS / "models.yaml").resolve("mock")

    outcome = await runner.run_once(
        task_id="same-task", session_id="session-a", model=model, client=MockClient(),
        max_turns=2, wiring=None,
    )

    assert outcome.passed is False
    assert "eval_error" in outcome.native_metrics
    assert outcome.error is not None and outcome.error.startswith("evaluate:")
    assert events == ["evaluate", "close", "restart"]


class FakeProcess:
    def __init__(self) -> None:
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float) -> int:
        del timeout
        return 0

    def kill(self) -> None:
        raise AssertionError("healthy fake process should terminate normally")


def test_owned_server_appends_combined_output_and_closes_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "appworld"
    log_path = tmp_path / "logs" / "server.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("earlier run\n", encoding="utf-8")
    captured: dict[str, Any] = {}
    process = FakeProcess()

    monkeypatch.setenv("APPWORLD_ROOT", str(root))
    monkeypatch.setenv("PANDABENCH_APPWORLD_PYTHON", "/fake/bin/python")
    monkeypatch.setenv("PANDABENCH_APPWORLD_LOG", str(log_path))
    monkeypatch.delenv("PANDABENCH_APPWORLD_URL", raising=False)

    def fake_popen(argv: list[str], **kwargs: Any) -> FakeProcess:
        captured.update({"argv": argv, **kwargs})
        return process

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(AppWorldServer, "_await_health", lambda self: None)

    server = AppWorldServer()
    server.start()
    handle = captured["stdout"]
    assert captured["stderr"] is subprocess.STDOUT
    assert handle is captured["stdout"]
    assert not handle.closed
    server.stop()

    assert process.terminated is True
    assert handle.closed
    assert log_path.read_text(encoding="utf-8") == "earlier run\n"


def test_external_server_5xx_logs_operator_restart_warning(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = "http://appworld.example:9000"
    monkeypatch.setenv("PANDABENCH_APPWORLD_URL", url)

    server = AppWorldServer()
    with caplog.at_level(logging.WARNING, logger="pandabench.appworld"):
        assert server.restart() == url

    assert server.owns_process is False
    assert "externally managed" in caplog.text
    assert "restart it before the next trial" in caplog.text
