"""Framework adapters bridging ``PandaHarnessHook`` to agent runtimes.

Only :class:`RawLoopAdapter` and the :class:`FrameworkAdapter` protocol are
imported eagerly; the real-framework adapters are imported lazily so their
optional third-party dependencies are not required for the core install.
"""

from .protocol import FrameworkAdapter
from .raw_loop import RawLoopAdapter

__all__ = [
    "FrameworkAdapter",
    "RawLoopAdapter",
    "LangGraphAdapter",
    "LangChainAdapter",
    "DeepAgentsAdapter",
    "CrewAIAdapter",
    "ClaudeAgentSDKAdapter",
    "OpenAIAgentsAdapter",
]

_LAZY = {
    "LangGraphAdapter": (".langgraph", "LangGraphAdapter"),
    "LangChainAdapter": (".langchain", "LangChainAdapter"),
    "DeepAgentsAdapter": (".deepagents", "DeepAgentsAdapter"),
    "CrewAIAdapter": (".crewai", "CrewAIAdapter"),
    "ClaudeAgentSDKAdapter": (".claude_agent_sdk", "ClaudeAgentSDKAdapter"),
    "OpenAIAgentsAdapter": (".openai_agents", "OpenAIAgentsAdapter"),
}


def __getattr__(name: str) -> object:  # pragma: no cover - thin lazy import
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(target[0], __name__)
    return getattr(module, target[1])
