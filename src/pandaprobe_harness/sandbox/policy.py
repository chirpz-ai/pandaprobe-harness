"""Command policy for the restricted sandbox shell.

The policy enforces three guarantees on a parsed ``argv``:

1. The invoked binary is on the allow-list.
2. No shell metacharacters / pipes / redirects are present (unless explicitly
   permitted).
3. No path argument escapes the configured working directory.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ShellPolicy", "ShellPolicyError"]

# Characters that imply shell interpretation we never want from a single argv.
_SHELL_METACHARS = frozenset(";&|><`$\n")


class ShellPolicyError(RuntimeError):
    """Raised when a command violates the shell policy."""


@dataclass(frozen=True, slots=True)
class ShellPolicy:
    """Allow-list-based policy for sandboxed command execution."""

    allowed_binaries: frozenset[str] = frozenset({"pandaprobe", "cat", "ls", "jq"})
    allow_pipes: bool = False
    workdir: Path = field(default_factory=lambda: Path("/harness"))
    timeout_s: float = 30.0

    def validate(self, argv: Sequence[str]) -> None:
        """Raise ``ShellPolicyError`` if ``argv`` is not permitted."""

        if not argv:
            raise ShellPolicyError("empty command")

        binary = argv[0]
        if binary not in self.allowed_binaries:
            raise ShellPolicyError(
                f"binary {binary!r} is not allow-listed "
                f"(allowed: {sorted(self.allowed_binaries)})"
            )

        if not self.allow_pipes:
            for token in argv:
                bad = _SHELL_METACHARS.intersection(token)
                if bad:
                    raise ShellPolicyError(
                        f"shell metacharacter(s) {sorted(bad)} not permitted in {token!r}"
                    )

        for token in argv[1:]:
            self._check_path_escape(token)

    def _check_path_escape(self, token: str) -> None:
        # Only scrutinize tokens that look like filesystem paths.
        if not (token.startswith(("/", "./", "../")) or token == ".."):
            return
        workdir = self.workdir.resolve()
        if token.startswith("/"):
            candidate = Path(token).resolve()
        else:
            candidate = (workdir / token).resolve()
        if candidate != workdir and workdir not in candidate.parents:
            raise ShellPolicyError(f"path {token!r} escapes workdir {workdir}")
