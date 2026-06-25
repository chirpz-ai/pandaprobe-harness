from __future__ import annotations

import pytest

from pandaprobe_harness.adapters.raw_loop import RawLoopAdapter


def test_parse_turn_extracts_fields() -> None:
    adapter = RawLoopAdapter()
    raw = adapter.make_turn("s-1", 3, action="charge")
    ctx = adapter.parse_turn(raw)
    assert ctx.session_id == "s-1"
    assert ctx.turn_index == 3
    assert ctx.end_state == {"action": "charge"}


def test_parse_turn_requires_session_id() -> None:
    adapter = RawLoopAdapter()
    with pytest.raises(ValueError):
        adapter.parse_turn({"turn_index": 1})


def test_parse_turn_rejects_non_mapping() -> None:
    adapter = RawLoopAdapter()
    with pytest.raises(TypeError):
        adapter.parse_turn(["not", "a", "mapping"])


def test_inject_and_consume_alerts() -> None:
    adapter = RawLoopAdapter()
    adapter.inject_alert("ALERT-1")
    adapter.inject_alert("ALERT-2")
    assert adapter.pending_alerts == ("ALERT-1", "ALERT-2")
    consumed = adapter.consume_alerts()
    assert consumed == ["ALERT-1", "ALERT-2"]
    # consuming drains the queue
    assert adapter.pending_alerts == ()
    assert adapter.consume_alerts() == []
