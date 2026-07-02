"""Native tool registration for function-calling frameworks (optional extras).

Each helper turns the toolset's uniform specs into the shape a framework
expects. Imports are guarded so the core stays dependency-free;
``as_anthropic_tools`` needs no guard because Anthropic tool specs are plain
dicts dispatched by the caller.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from .spec import ToolSpec
from .toolset import HarnessToolset

__all__ = [
    "as_anthropic_tools",
    "as_langchain_tools",
    "as_openai_function_tools",
]


def as_anthropic_tools(
    toolset: HarnessToolset,
) -> tuple[list[dict[str, Any]], Callable[[str, Mapping[str, Any]], Awaitable[dict[str, Any]]]]:
    """Anthropic-style tool dicts plus an async dispatcher.

    The caller registers the dicts in its tool list and routes matching
    ``tool_use`` blocks through the dispatcher.
    """

    specs = [
        {"name": s.name, "description": s.description, "input_schema": s.input_schema}
        for s in toolset.specs()
    ]
    return specs, toolset.call


def as_langchain_tools(toolset: HarnessToolset) -> list[Any]:
    """LangChain ``StructuredTool``s (requires a ``langchain``-family extra)."""

    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "as_langchain_tools requires langchain-core; install a langchain-family "
            "extra, e.g. pip install 'pandaprobe-harness[langchain]'."
        ) from exc

    def _make(spec: ToolSpec) -> Any:
        async def _invoke(**kwargs: Any) -> dict[str, Any]:
            return await toolset.call(spec.name, kwargs)

        return StructuredTool(
            name=spec.name,
            description=spec.description,
            args_schema=spec.input_schema,
            coroutine=_invoke,
        )

    return [_make(spec) for spec in toolset.specs()]


def as_openai_function_tools(toolset: HarnessToolset) -> list[Any]:
    """OpenAI Agents SDK ``FunctionTool``s (requires the ``openai-agents`` extra)."""

    try:
        from agents import FunctionTool
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError(
            "as_openai_function_tools requires the openai-agents extra: "
            "pip install 'pandaprobe-harness[openai-agents]'."
        ) from exc

    def _make(spec: ToolSpec) -> Any:
        async def _invoke(_ctx: Any, args_json: str) -> str:
            try:
                args = json.loads(args_json) if args_json else {}
            except ValueError:
                args = {}
            result = await toolset.call(spec.name, args if isinstance(args, dict) else {})
            return json.dumps(result, default=str)

        # The Agents SDK requires additionalProperties to be explicit.
        schema = dict(spec.input_schema)
        schema.setdefault("additionalProperties", False)
        # Keep the harness schema as-authored: strict mode would rewrite every
        # property into `required` with no null type, making genuinely optional
        # params (rule_id, note, metric, session_id, …) mandatory for the model.
        return FunctionTool(
            name=spec.name,
            description=spec.description,
            params_json_schema=schema,
            on_invoke_tool=_invoke,
            strict_json_schema=False,
        )

    return [_make(spec) for spec in toolset.specs()]
