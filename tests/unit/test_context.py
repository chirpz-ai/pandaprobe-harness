from __future__ import annotations

from pathlib import Path

from pandaprobe_harness import HarnessConfig, HarnessFilesystem, compose_system_preamble


def test_preamble_contains_appended_rule(tmp_path: Path) -> None:
    cfg = HarnessConfig(harness_root=tmp_path / "h")
    fs = HarnessFilesystem(cfg)
    fs.provision()
    fs.append_rule("never double-charge a payment", source="self-heal")
    preamble = compose_system_preamble(fs)
    assert "PANDAPROBE HARNESS RULES" in preamble
    assert "never double-charge a payment" in preamble


def test_preamble_empty_when_no_rules_file(tmp_path: Path) -> None:
    cfg = HarnessConfig(harness_root=tmp_path / "h")
    fs = HarnessFilesystem(cfg)  # not provisioned → no rules file
    assert compose_system_preamble(fs) == ""
