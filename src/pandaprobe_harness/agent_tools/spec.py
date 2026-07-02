"""The framework-agnostic tool-spec value type."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["ToolHandler", "ToolSpec"]

#: Every harness tool handler takes one argument mapping and returns a JSON-
#: serializable result dict — a deliberately uniform shape so the companion
#: CLI and every native registration helper can dispatch identically.
ToolHandler = Callable[[Mapping[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One agent-facing harness operation: name, schema, and async handler."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler
