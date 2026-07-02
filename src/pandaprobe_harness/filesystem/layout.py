"""the diagnostic filesystem provisioner.

Manages the persistent ``/harness`` workspace the agent reads and rewrites to
self-heal:

- ``harness_rules.md`` — the living rules (seeded from the template; rendered
  by ``workspace.RulesStore`` once structured rules exist).
- ``traces/latest_eval.json`` — the most recent eval dump (written atomically).
- ``traces/<notice_id>.json`` — one immutable dump per diagnostic notice.

All methods here perform blocking I/O; async callers wrap them in
``asyncio.to_thread``.
"""

from __future__ import annotations

import importlib.resources
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..config import HarnessConfig
from ..workspace._io import atomic_write_json, load_json

__all__ = ["HarnessFilesystem"]

_TEMPLATE_PACKAGE = "pandaprobe_harness.filesystem.templates"
_TEMPLATE_NAME = "harness_rules.md"


class HarnessFilesystem:
    """Provisions and maintains the ``/harness`` diagnostic workspace."""

    def __init__(self, config: HarnessConfig) -> None:
        self._config = config

    # -- provisioning ---------------------------------------------------------

    def provision(self) -> None:
        """Create the ``/harness`` tree and seed ``harness_rules.md`` if absent.

        Idempotent: an existing rules file (which may have accumulated learned
        mitigations) is never overwritten.
        """

        cfg = self._config
        cfg.harness_root.mkdir(parents=True, exist_ok=True)
        cfg.traces_dir.mkdir(parents=True, exist_ok=True)
        cfg.state_dir.mkdir(parents=True, exist_ok=True)
        if not cfg.rules_file.exists():
            cfg.rules_file.write_text(self._default_rules_template(), encoding="utf-8")

    @staticmethod
    def _default_rules_template() -> str:
        resource = importlib.resources.files(_TEMPLATE_PACKAGE) / _TEMPLATE_NAME
        return resource.read_text(encoding="utf-8")

    # -- rules ----------------------------------------------------------------

    def read_rules(self) -> str:
        return self._config.rules_file.read_text(encoding="utf-8")

    # -- eval dumps -----------------------------------------------------------

    def write_latest_eval(self, payload: Mapping[str, Any]) -> None:
        """Atomically write the latest eval dump.

        Writes to a unique temp file in the same directory then ``os.replace``
        so a concurrent reader (the agent) never observes a half-written file
        and concurrent writers never collide on the temp path.
        """

        atomic_write_json(self._config.latest_eval_file, dict(payload))

    def read_latest_eval(self) -> dict[str, Any]:
        data = load_json(self._config.latest_eval_file)
        if data is None:
            raise FileNotFoundError(str(self._config.latest_eval_file))
        return data

    def write_trace_dump(self, name: str, payload: Mapping[str, Any]) -> Path:
        """Atomically write an immutable per-notice dump under ``traces/``."""

        target = self._config.traces_dir / f"{name}.json"
        atomic_write_json(target, dict(payload))
        return target

    def read_trace_dump(self, name: str) -> dict[str, Any] | None:
        return load_json(self._config.traces_dir / f"{name}.json")
