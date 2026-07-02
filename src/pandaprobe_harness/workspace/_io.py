"""Shared blocking-I/O helpers for the workspace stores.

All writes follow the collision-safe atomic pattern established by
``evaluation/history.py``: a unique temp file (pid + uuid) in the target's
directory followed by ``os.replace``, so concurrent writers from the
``asyncio.to_thread`` worker pool never collide on the temp path and a
concurrent reader never observes a half-written file. JSONL reads are
forgiving: unparseable or non-object lines are skipped rather than raised.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "append_jsonl",
    "atomic_write_json",
    "atomic_write_text",
    "load_json",
    "read_jsonl",
]


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically write ``content`` to ``path`` (unique temp + ``os.replace``)."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    """Atomically write ``payload`` as pretty JSON to ``path``."""

    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True))


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Append one JSON object as a single line to ``path``."""

    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(record), sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank/corrupt/non-object lines."""

    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    records: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def load_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON object file; ``None`` when missing/corrupt/non-object."""

    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None
