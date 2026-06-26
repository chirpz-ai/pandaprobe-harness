"""Shared session bridge for framework adapters.

All adapters resolve the current ``session_id`` from the PandaProbe SDK's session
``ContextVar`` (``pandaprobe.tracing.session.get_current_session_id``) so the
harness and the SDK traces agree on session identity. Import-guarded: if the SDK
is not installed, returns ``None`` and the adapter falls back to a constructor
value.
"""

from __future__ import annotations

__all__ = ["current_session_id"]


def current_session_id() -> str | None:
    try:
        from pandaprobe.tracing.session import get_current_session_id
    except ImportError:
        return None
    session_id: str | None = get_current_session_id()
    return session_id
