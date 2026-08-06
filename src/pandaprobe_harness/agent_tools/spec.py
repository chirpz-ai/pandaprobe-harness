"""The framework-agnostic tool-spec value type."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = ["ToolDispatcher", "ToolHandler", "ToolSpec"]

#: Every harness tool handler takes one argument mapping and returns a JSON-
#: serializable result dict so every native registration helper can dispatch
#: identically.
ToolHandler = Callable[[Mapping[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One agent-facing harness operation: name, schema, and async handler."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler


class ToolDispatcher(Protocol):
    """Common registration and dispatch surface for framework adapters."""

    def specs(self) -> tuple[ToolSpec, ...]: ...

    async def call(self, name: str, args: Mapping[str, Any]) -> dict[str, Any]: ...
