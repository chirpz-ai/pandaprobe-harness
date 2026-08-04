from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest

from pandabench.providers.litellm_client import ChatResult, MockClient, ToolCall, Usage
from pandabench.providers.models import load_registry
from pandabench.runners.appworld import AppWorldRunner, _experiment_name
from pandabench.runners.appworld_env import (
    AppWorldServer,
    EvalResult,
    HttpAppWorldEnv,
    TaskInfo,
)
from pandabench.runners.tau2 import Tau2Runner

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


def test_tau2_outcomes_are_session_scoped() -> None:
    runner = Tau2Runner()
    runner._outcomes.update({"session-a": 0.25, "session-b": 1.0, "replay-session": 0.75})

    assert runner.outcome_for("same-task", "session-a") == 0.25
    assert runner.outcome_for("same-task", "session-b") == 1.0
    assert runner.outcome_for("same-task", "replay-session") == 0.75
    assert runner.outcome_for("same-task", "unknown") is None


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
