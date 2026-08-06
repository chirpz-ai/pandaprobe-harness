"""Persisted 0.8 workspace artifacts remain readable after the API break."""

from __future__ import annotations

import json
from pathlib import Path

from pandaprobe_harness import HarnessConfig, ScoreHistoryStore
from pandaprobe_harness.workspace.evalset import EvalSet
from pandaprobe_harness.workspace.journal import Journal
from pandaprobe_harness.workspace.mailbox import Mailbox
from pandaprobe_harness.workspace.rules import RulesStore


def test_representative_080_workspace_loads(tmp_path: Path) -> None:
    config = HarnessConfig(harness_root=tmp_path / "h", repair_model="test/fake")
    config.rules_store_file.parent.mkdir(parents=True)
    config.rules_store_file.write_text(
        json.dumps(
            {
                "id": "r-080",
                "created_at": "2026-07-31T00:00:00+00:00",
                "rule": "Paginate every result page.",
                "rationale": "Totals were incomplete.",
                "source_notice_id": "n-080",
                "metric": "task_completion",
                "status": "candidate",
                "tags": ["pagination"],
                "scope": "scoped",
                "trial": {"observed_sessions": ["old-session"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config.mailbox_processed_dir.mkdir(parents=True)
    (config.mailbox_processed_dir / "n-080.json").write_text(
        json.dumps(
            {
                "id": "n-080",
                "created_at": "2026-07-31T00:00:00+00:00",
                "session_id": "old-session",
                "turn_index": 2,
                "severity": "breach",
                "summary": "pagination failure",
                "status": "acknowledged",
                "resolution": {
                    "acked_at": "2026-07-31T00:01:00+00:00",
                    "rule_id": "r-080",
                    "note": "legacy acknowledgement",
                },
            }
        ),
        encoding="utf-8",
    )
    config.evalset_dir.mkdir(parents=True)
    (config.evalset_dir / "c-080.json").write_text(
        json.dumps(
            {
                "id": "c-080",
                "created_at": "2026-07-31T00:00:00+00:00",
                "session_id": "old-session",
                "kind": "failure",
                "signature": ["breach:tool_correctness"],
                "baseline_scores": {"tool_correctness": 0.2},
                "replay_input": {"task": "old task"},
            }
        ),
        encoding="utf-8",
    )
    config.journal_file.write_text(
        '{"type":"notice","session_id":"old-session"}\n', encoding="utf-8"
    )
    config.history_file.parent.mkdir(parents=True, exist_ok=True)
    config.history_file.write_text(
        json.dumps(
            {
                "old-session::task_completion": {
                    "series": [
                        {
                            "value": 0.2,
                            "ts": "2026-07-31T00:00:00+00:00",
                            "run_id": "run-old",
                        }
                    ],
                    "gate": {"peak": 0.2, "turns_since_gain": 1},
                }
            }
        ),
        encoding="utf-8",
    )

    rule = RulesStore(config).all()[0]
    assert rule.id == "r-080" and rule.status == "candidate"
    notice = Mailbox(config).read("n-080")
    assert notice is not None and notice.resolution is not None
    assert notice.resolution.kind == "legacy"
    assert EvalSet(config).cases()[0].id == "c-080"
    assert Journal(config).recent()[0]["session_id"] == "old-session"
    assert ScoreHistoryStore(config).values("old-session", "task_completion") == [0.2]
