"""Immutable, benchmark-owned ruleset captured at the learning/eval boundary."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["FROZEN_RULES_FILENAME", "FrozenRulesSnapshot"]

FROZEN_RULES_FILENAME = "frozen-rules.json"
_SCHEMA_VERSION = 1


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _rule_json(rule: Any) -> dict[str, Any]:
    if isinstance(rule, Mapping):
        value = dict(rule)
    else:
        to_json = getattr(rule, "to_json", None)
        if not callable(to_json):
            raise TypeError("frozen rules must be mappings or expose to_json()")
        value = to_json()
    if not isinstance(value, dict):
        raise TypeError("rule to_json() must return an object")
    # A canonical round-trip both validates JSON compatibility and severs every
    # reference to the mutable live workspace objects.
    copied = json.loads(_canonical(value))
    if not isinstance(copied, dict):  # pragma: no cover - guarded by value above
        raise TypeError("rule must serialize to an object")
    return copied


@dataclass(frozen=True, slots=True)
class FrozenRulesSnapshot:
    """An immutable ruleset whose digest covers all persisted snapshot fields."""

    created_at: str
    active_count: int
    candidate_count: int
    retired_count: int
    sha256: str
    _rule_records: tuple[str, ...]
    schema_version: int = _SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        rules: Iterable[Any],
        *,
        created_at: str | None = None,
    ) -> FrozenRulesSnapshot:
        records = [_rule_json(rule) for rule in rules]
        records.sort(
            key=lambda rule: (
                str(rule.get("created_at", "")),
                str(rule.get("id", "")),
                _canonical(rule),
            )
        )
        active = sum(rule.get("status") == "active" for rule in records)
        candidate = sum(rule.get("status") == "candidate" for rule in records)
        retired = sum(rule.get("status") == "retired" for rule in records)
        timestamp = created_at or datetime.now(UTC).isoformat()
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "created_at": timestamp,
            "active_count": active,
            "candidate_count": candidate,
            "retired_count": retired,
            "rules": records,
        }
        digest = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
        return cls(
            created_at=timestamp,
            active_count=active,
            candidate_count=candidate,
            retired_count=retired,
            sha256=digest,
            _rule_records=tuple(_canonical(rule) for rule in records),
        )

    @property
    def rules(self) -> tuple[dict[str, Any], ...]:
        """Detached rule copies; callers cannot mutate the frozen backing state."""

        return tuple(json.loads(record) for record in self._rule_records)

    @property
    def live_rules(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            rule for rule in self.rules if rule.get("status") in ("active", "candidate")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "active_count": self.active_count,
            "candidate_count": self.candidate_count,
            "retired_count": self.retired_count,
            "rules": list(self.rules),
            "sha256": self.sha256,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> FrozenRulesSnapshot:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"could not read frozen rules snapshot {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"frozen rules snapshot {path} must contain a JSON object")
        if raw.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(
                f"unsupported frozen rules schema in {path}: {raw.get('schema_version')!r}"
            )
        rules = raw.get("rules")
        if not isinstance(rules, list) or not all(isinstance(rule, dict) for rule in rules):
            raise ValueError(f"frozen rules snapshot {path} has invalid rules")
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "created_at": str(raw.get("created_at", "")),
            "active_count": raw.get("active_count"),
            "candidate_count": raw.get("candidate_count"),
            "retired_count": raw.get("retired_count", 0),
            "rules": rules,
        }
        expected = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
        recorded = str(raw.get("sha256", ""))
        if recorded != expected:
            raise ValueError(
                f"frozen rules snapshot hash mismatch for {path}: "
                f"recorded={recorded or '<missing>'} expected={expected}"
            )
        actual_counts = {
            "active_count": sum(rule.get("status") == "active" for rule in rules),
            "candidate_count": sum(rule.get("status") == "candidate" for rule in rules),
            "retired_count": sum(rule.get("status") == "retired" for rule in rules),
        }
        for name, actual in actual_counts.items():
            if payload[name] != actual:
                raise ValueError(
                    f"frozen rules snapshot count mismatch for {path}: "
                    f"{name}={payload[name]!r}, expected {actual}"
                )
        return cls(
            created_at=str(payload["created_at"]),
            active_count=actual_counts["active_count"],
            candidate_count=actual_counts["candidate_count"],
            retired_count=actual_counts["retired_count"],
            sha256=recorded,
            _rule_records=tuple(_canonical(rule) for rule in rules),
        )
