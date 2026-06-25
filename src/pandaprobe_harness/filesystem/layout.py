"""the diagnostic filesystem provisioner.

Manages the persistent ``/harness`` workspace the agent reads and rewrites to
self-heal:

- ``harness_rules.md`` — the living rules (never clobbered once it exists).
- ``traces/latest_eval.json`` — the most recent failure dump (written atomically).

All methods here perform blocking I/O; async callers wrap them in
``asyncio.to_thread``.
"""

from __future__ import annotations

import importlib.resources
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from ..config import HarnessConfig

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
        if not cfg.rules_file.exists():
            cfg.rules_file.write_text(self._default_rules_template(), encoding="utf-8")

    @staticmethod
    def _default_rules_template() -> str:
        resource = importlib.resources.files(_TEMPLATE_PACKAGE) / _TEMPLATE_NAME
        return resource.read_text(encoding="utf-8")

    # -- rules ----------------------------------------------------------------

    def read_rules(self) -> str:
        return self._config.rules_file.read_text(encoding="utf-8")

    def append_rule(
        self,
        rule: str,
        *,
        source: str = "self-heal",
        timestamp: datetime | None = None,
    ) -> None:
        """Append a timestamped, attributed mitigation rule to the rules file.

        This is the permanent self-healing artifact: a learned directive that
        will be read into context on every subsequent run.
        """

        when = (timestamp or datetime.now(UTC)).isoformat()
        block = f"\n- _[{when}] ({source})_ {rule.strip()}\n"
        with self._config.rules_file.open("a", encoding="utf-8") as handle:
            handle.write(block)

    # -- eval dumps -----------------------------------------------------------

    def write_latest_eval(self, payload: Mapping[str, Any]) -> None:
        """Atomically write the latest eval dump.

        Writes to a temp file in the same directory then ``os.replace`` so a
        concurrent reader (the agent) never observes a half-written file.
        """

        target = self._config.latest_eval_file
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, target)

    def read_latest_eval(self) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(
            self._config.latest_eval_file.read_text(encoding="utf-8")
        )
        return data
