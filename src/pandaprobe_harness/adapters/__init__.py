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
    "CrewAIAdapter",
    "AutoGenAdapter",
]


def __getattr__(name: str) -> object:  # pragma: no cover - thin lazy import
    if name == "LangGraphAdapter":
        from .langgraph import LangGraphAdapter

        return LangGraphAdapter
    if name == "CrewAIAdapter":
        from .crewai import CrewAIAdapter

        return CrewAIAdapter
    if name == "AutoGenAdapter":
        from .autogen import AutoGenAdapter

        return AutoGenAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
