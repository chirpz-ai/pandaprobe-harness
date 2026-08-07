"""Scope identity: the two reserved names, normalization, and validation.

A scope is the key of the ``rules/<scope>.md`` reference a rule is filed under.
Two names are reserved and mean something specific:

``global``
    The default. Broadly reusable guidance that is not tied to one task,
    workflow, application, tool, or domain.
``scoped``
    The fallback for guidance that *is* specific but for which no meaningful
    stable name can be determined.

Every other name is open: managed repair may choose any concise topic — an
application, a workflow, a domain — and PandaProbe creates the file. There is no
required prefix or naming format; normalization exists purely so a scope is safe
to use as one path component.

This lives apart from :mod:`.rules` so the mailbox (which ``rules`` imports) and
the turn hook can share one definition instead of re-spelling the literals.
"""

from __future__ import annotations

import re

from .sanitize import sanitize_text

__all__ = [
    "GLOBAL_SCOPE",
    "SCOPED_SCOPE",
    "RESERVED_SCOPES",
    "normalize_scope",
    "normalize_scope_description",
    "validate_scope",
]

#: The default scope: broadly reusable, not tied to one context.
GLOBAL_SCOPE = "global"
#: The fallback for specific guidance with no meaningful stable name.
SCOPED_SCOPE = "scoped"
#: The two names with reserved meaning; every other scope key is open.
RESERVED_SCOPES = frozenset({GLOBAL_SCOPE, SCOPED_SCOPE})

#: A scope becomes a filename, so it must be exactly one safe path component.
_SAFE_SCOPE = re.compile(r"\A[a-z0-9][a-z0-9._-]{0,47}\Z")
_SCOPE_SEPARATORS = re.compile(r"[^a-z0-9._-]+")
_SCOPE_DESCRIPTION_MAX_LEN = 160
_SCOPE_DESCRIPTION_WS = re.compile(r"\s+")


def normalize_scope(value: str | None) -> str:
    """Slugify a supplied scope into one safe path component.

    Scope is deliberately free-form — managed repair organizes guidance — but it
    is also used as a filename, so it is slugified and bounded here solely for
    path safety. There is no semantic prefix format. Empty or unusable input
    selects :data:`GLOBAL_SCOPE`, the default; ``scoped`` must be chosen.
    """

    if value is None or not value.strip():
        return GLOBAL_SCOPE
    slug = _SCOPE_SEPARATORS.sub("-", value.strip().casefold()).strip("-._")[:48]
    if not slug or slug in {".", ".."} or not _SAFE_SCOPE.fullmatch(slug):
        return GLOBAL_SCOPE
    return slug


def normalize_scope_or_none(value: str | None) -> str | None:
    """Normalize a scope, or return ``None`` when the input cannot name one.

    Distinguishes "no scope was supplied" from "the default applies", which the
    repair path needs: an absent host recommendation must not read as an
    explicit choice of :data:`GLOBAL_SCOPE`.
    """

    if value is None or not value.strip():
        return None
    slug = _SCOPE_SEPARATORS.sub("-", value.strip().casefold()).strip("-._")[:48]
    if not slug or slug in {".", ".."} or not _SAFE_SCOPE.fullmatch(slug):
        return None
    return slug


def validate_scope(value: object) -> str:
    """Validate an agent-supplied scope without silently rewriting a path.

    Managed repair may normalize host/model metadata before persistence. Task
    reads are stricter: a caller must name one canonical scope component, so
    traversal, absolute paths, separators, and ambiguous aliases fail closed.
    """

    if not isinstance(value, str) or not value:
        raise ValueError("scope must be a non-empty canonical identifier")
    if value in {".", ".."} or not _SAFE_SCOPE.fullmatch(value):
        raise ValueError("invalid rule scope")
    if normalize_scope(value) != value:
        raise ValueError("scope must already be normalized")
    return value


def normalize_scope_description(value: str | None, *, scope: str) -> str:
    """Return one bounded metadata-only sentence for an index entry."""

    text = _SCOPE_DESCRIPTION_WS.sub(" ", sanitize_text(value or "", max_len=256)).strip()
    text = text[:_SCOPE_DESCRIPTION_MAX_LEN].rstrip(" ,;:")
    if text:
        return text if text.endswith((".", "!", "?")) else text + "."
    if scope == GLOBAL_SCOPE:
        return "Cross-domain execution and verification guidance."
    if scope == SCOPED_SCOPE:
        return "Narrow task-specific execution and verification guidance."
    label = scope.replace("_", " ").replace("-", " ").strip().title() or "Topical"
    return f"{label} workflows."
