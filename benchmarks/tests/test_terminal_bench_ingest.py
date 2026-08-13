"""Offline coverage for Harbor result ingestion and resume ordinals."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pandabench.adapters import harbor_agent
from pandabench.providers.litellm_client import ChatResult, MockClient, ToolCall, Usage
from pandabench.providers.models import load_registry
from pandabench.results import RecordWriter, TrialRecord
from pandabench.runners.terminal_bench import (
    _error_record,
    _harbor_argv,
    _ingest_results,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _payload(
    *,
    task: str,
    trial_name: str,
    started_at: str,
    rewards: dict[str, float],
    turns: int,
    cost: float,
    exception: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "task_name": task,
        "trial_name": trial_name,
        "started_at": started_at,
        "finished_at": "2026-07-30T12:00:10+00:00",
        "verifier_result": {"rewards": rewards},
        "agent_result": {
            "n_input_tokens": 100 + turns,
            "n_output_tokens": 20 + turns,
            "cost_usd": cost,
            "metadata": {
                "arm": "harness",
                "seed": 3,
                "turns": turns,
                "stopped_reason": "final",
                "harness": {
                    "session_id": f"session-{trial_name}",
                    "breached": True,
                    "rules_active": 1,
                    "rules_candidate": 0,
                    "rules_retired": 0,
                    "notices": 2,
                    "scores": {"task_completion": 0.25},
                    "gate_breached": True,
                },
            },
        },
        "exception_info": exception,
    }


def test_terminal_scope_hint_uses_safe_harbor_metadata_not_task_id() -> None:
    hints = harbor_agent._terminal_scope_hints(
        SimpleNamespace(metadata={"task_family": "package-build"})
    )
    assert len(hints) == 1
    assert hints[0].key == "package-build"
    assert hints[0].recommended is True
    assert "terminal workflows" in hints[0].description
    assert harbor_agent._terminal_scope_hints(SimpleNamespace(metadata={})) == ()


def _write_result(job_dir: Path, dirname: str, payload: dict[str, Any]) -> None:
    target = job_dir / dirname
    target.mkdir(parents=True)
    (target / "result.json").write_text(json.dumps(payload), encoding="utf-8")


def test_ingest_rewards_ordinals_usage_and_harness_metadata(tmp_path):
    model = load_registry(CONFIGS / "models.yaml").resolve("mock")
    job_dir = tmp_path / "raw" / "live"
    # Directory names are intentionally opposite timestamp order. Harbor's suffix
    # is random, so started_at must assign stable attempt ordinals.
    _write_result(
        job_dir,
        "task-a__zzzzzzz",
        _payload(
            task="task-a",
            trial_name="task-a__zzzzzzz",
            started_at="2026-07-30T12:00:02+00:00",
            rewards={"reward": 1.0},
            turns=7,
            cost=0.40,
        ),
    )
    _write_result(
        job_dir,
        "task-a__aaaaaaa",
        _payload(
            task="task-a",
            trial_name="task-a__aaaaaaa",
            started_at="2026-07-30T12:00:01+00:00",
            rewards={"score": 0.0},
            turns=3,
            cost=0.20,
        ),
    )
    records_path = tmp_path / "records.jsonl"
    writer = RecordWriter(records_path)

    seen = _ingest_results(
        job_dir=job_dir,
        tasks=["task-a"],
        k=2,
        arm="harness",
        model=model,
        phase="live",
        writer=writer,
        run_id="run-1",
        seed=3,
        benchmark="terminal_bench",
    )

    records = [
        TrialRecord.from_json(json.loads(line))
        for line in records_path.read_text(encoding="utf-8").splitlines()
    ]
    assert seen == {("task-a", 0), ("task-a", 1)}
    assert [record.trial for record in records] == [0, 1]
    assert records[0].native_metrics["rewards"] == {"score": 0.0}
    assert records[0].native_metrics["reward"] == 0.0  # first-value fallback
    assert records[0].passed is False
    assert records[0].turns == 3
    assert records[0].usage["cost_usd"] == pytest.approx(0.20)
    assert records[1].native_metrics["reward"] == 1.0
    assert records[1].passed is True
    assert records[1].turns == 7
    assert records[1].harness is not None
    assert records[1].harness["scores"] == {"task_completion": 0.25}


def test_ingest_exception_and_synthetic_error_record(tmp_path):
    model = load_registry(CONFIGS / "models.yaml").resolve("mock")
    job_dir = tmp_path / "raw" / "live"
    _write_result(
        job_dir,
        "task-b__aaaaaaa",
        _payload(
            task="task-b",
            trial_name="task-b__aaaaaaa",
            started_at="2026-07-30T12:00:01+00:00",
            rewards={},
            turns=1,
            cost=0.01,
            exception={
                "exception_type": "RuntimeError",
                "exception_message": "container failed",
            },
        ),
    )
    records_path = tmp_path / "records.jsonl"
    writer = RecordWriter(records_path)
    _ingest_results(
        job_dir=job_dir,
        tasks=["task-b"],
        k=1,
        arm="harness",
        model=model,
        phase="live",
        writer=writer,
        run_id="run-2",
        seed=3,
        benchmark="terminal_bench",
    )
    record = TrialRecord.from_json(json.loads(records_path.read_text(encoding="utf-8")))
    assert record.error == "RuntimeError: container failed"

    missing = _error_record(
        message="Harbor exited with status 2",
        run_id="run-2",
        benchmark="terminal_bench",
        task_id="task-c",
        arm="baseline",
        model=model,
        seed=3,
        trial=0,
        phase="live",
        returncode=2,
    )
    assert missing.passed is False
    assert missing.error == "Harbor exited with status 2"
    assert missing.native_metrics == {"harbor_exit_code": 2}


def test_harbor_harness_argv_is_serial_and_requests_managed_harness(tmp_path):
    model = load_registry(CONFIGS / "models.yaml").resolve("mock")
    argv = _harbor_argv(
        dataset="terminal-bench-sample@2.0",
        tasks=["task-a", "task-b"],
        k=2,
        arm="harness",
        model=model,
        phase="live",
        raw_dir=tmp_path / "raw",
        seed=3,
        backend=None,
        harness_root=tmp_path / "harness_root",
        max_turns=50,
        session_namespace="test-namespace",
    )

    assert Path(argv[0]).name == "harbor"
    assert argv[1] == "run"
    assert argv[argv.index("-d") + 1] == "terminal-bench-sample@2.0"
    assert argv[argv.index("-n") + 1] == "1"
    assert "-y" in argv
    # The harness arm captures for the whole job; there is no frozen eval to flag.
    assert "capture=true" in argv
    assert "phase=live" in argv
    assert not any(arg.startswith("frozen") for arg in argv)
    assert "session_namespace=test-namespace" in argv
    assert [argv[index + 1] for index, value in enumerate(argv) if value == "-i"] == [
        "task-a",
        "task-b",
    ]


def test_harbor_baseline_argv_never_captures(tmp_path):
    model = load_registry(CONFIGS / "models.yaml").resolve("mock")
    argv = _harbor_argv(
        dataset="terminal-bench-sample@2.0", tasks=["task-a"], k=1,
        arm="baseline", model=model, phase="live", raw_dir=tmp_path / "raw",
        seed=1, backend=None, harness_root=tmp_path / "harness_root",
        max_turns=20, session_namespace="test-namespace",
    )

    assert "capture=false" in argv
    assert "arm=baseline" in argv


class RecordingHarborClient(MockClient):
    def __init__(self) -> None:
        super().__init__(scripted=[
            ChatResult(
                assistant_message={
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "read", "type": "function",
                        "function": {
                            "name": "harness_rules_read",
                            "arguments": '{"scope":"global"}',
                        },
                    }],
                },
                tool_calls=[ToolCall("read", "harness_rules_read", {"scope": "global"})],
                usage=Usage(), finish_reason="tool_calls", resolved_model="mock",
            ),
            ChatResult(
                assistant_message={"role": "assistant", "content": "done"},
                tool_calls=[], usage=Usage(), finish_reason="stop", resolved_model="mock",
            ),
        ])
        self.tool_names: list[list[str]] = []
        self.messages: list[list[dict[str, Any]]] = []
        self.flushes = 0

    async def chat(self, **kwargs: Any) -> ChatResult:
        self.tool_names.append([
            str((tool.get("function") or {}).get("name", ""))
            for tool in kwargs.get("tools") or []
        ])
        self.messages.append(list(kwargs["messages"]))
        return await super().chat(**kwargs)

    def flush(self) -> None:
        self.flushes += 1


async def test_baseline_harbor_agent_builds_no_harness_and_offers_no_rule_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Arm A stays a plain tool-calling loop, and still flushes its traces."""

    client = RecordingHarborClient()
    monkeypatch.setattr(harbor_agent, "LiteLLMClient", lambda **kwargs: client)
    monkeypatch.setattr(
        harbor_agent, "build_harness",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("built live Harness")),
    )
    monkeypatch.setattr(
        harbor_agent.PandaTracer, "from_env", classmethod(lambda cls: cls.disabled()),
    )
    logs_dir = tmp_path / "task-a__trial" / "agent"
    logs_dir.mkdir(parents=True)
    agent = harbor_agent.PandaBenchAgent(
        logs_dir,
        arm="baseline",
        seed=1,
        model_key="mock",
        phase="live",
        harness_root=str(tmp_path / "live-root"),
        max_turns=3,
    )
    context = SimpleNamespace()
    environment = SimpleNamespace(exec=None)

    await agent.run("complete the task", environment, context)

    assert agent._harness is None
    assert all(tools == ["bash"] for tools in client.tool_names)
    assert client.flushes == 1
    assert context.metadata["harness"] is None
