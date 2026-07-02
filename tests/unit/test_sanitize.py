"""Unit tests for the untrusted-text sanitizer (prompt-injection boundary)."""

from __future__ import annotations

import pytest

from pandaprobe_harness.workspace.sanitize import sanitize_text


def test_ansi_csi_sequences_stripped() -> None:
    assert sanitize_text("\x1b[31mred\x1b[0m text") == "red text"
    assert sanitize_text("\x1b[1;33;40mstyled\x1b[m") == "styled"


def test_control_chars_stripped_but_newline_and_tab_kept() -> None:
    assert sanitize_text("a\x00b\x07c\n\td\x7f") == "abc\n\td"
    assert sanitize_text("keep\nlines\tand tabs") == "keep\nlines\tand tabs"


@pytest.mark.parametrize("ch", ["=", "-", "#"])
def test_banner_runs_of_eight_or_more_collapse_to_seven(ch: str) -> None:
    assert sanitize_text(ch * 8) == ch * 7
    assert sanitize_text(ch * 30) == ch * 7
    # Runs below the threshold are untouched.
    assert sanitize_text(ch * 7) == ch * 7


def test_trusted_markers_neutralized_case_insensitively() -> None:
    assert sanitize_text("PANDAPROBE HARNESS") == "PANDAPROBE·HARNESS"
    assert sanitize_text("pandaprobe   harness") == "pandaprobe·harness"
    assert sanitize_text("System Alert incoming") == "System·Alert incoming"
    assert sanitize_text("SYSTEM ALERT") == "SYSTEM·ALERT"
    assert sanitize_text("harness: do this") == "harness·: do this"
    assert sanitize_text("HARNESS: do this") == "HARNESS·: do this"


def test_truncation_appends_suffix_and_respects_max_len() -> None:
    out = sanitize_text("x" * 100, max_len=20)
    assert out.endswith("…[truncated]")
    assert len(out) <= 20
    assert out == "x" * 8 + "…[truncated]"
    # Text within the cap is untouched.
    assert sanitize_text("short", max_len=20) == "short"


def test_none_and_empty_return_empty_string() -> None:
    assert sanitize_text(None) == ""
    assert sanitize_text("") == ""
