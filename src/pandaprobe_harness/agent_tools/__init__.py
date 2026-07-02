"""The agent-facing harness toolset (native tools + companion CLI)."""

from __future__ import annotations

from .companion import build_toolset_from_env, main
from .native import as_anthropic_tools, as_langchain_tools, as_openai_function_tools
from .spec import ToolHandler, ToolSpec
from .toolset import OP_SCHEMAS, HarnessToolset

__all__ = [
    "OP_SCHEMAS",
    "HarnessToolset",
    "ToolHandler",
    "ToolSpec",
    "as_anthropic_tools",
    "as_langchain_tools",
    "as_openai_function_tools",
    "build_toolset_from_env",
    "main",
]
