"""Command policy for the restricted sandbox shell.

The policy enforces five guarantees on a parsed ``argv``:

1. The invoked binary is on the allow-list.
2. The argv does not match a denied prefix (e.g. ``pandaprobe config``) and
   carries no denied flag (e.g. ``--reveal-secrets``).
3. No shell metacharacters / pipes / redirects are present (unless explicitly
   permitted).
4. No path argument escapes the configured working directory.
5. The subprocess environment is *scoped*: credential-shaped variables are
   scrubbed, and the auth variables the ``pandaprobe`` CLI needs are restored
   only for the binaries that legitimately require them — never for ``cat``/
   ``ls``/``jq`` (whose ``env`` builtin would otherwise leak them).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["ShellPolicy", "ShellPolicyError"]

# Characters that imply shell interpretation we never want from a single argv.
_SHELL_METACHARS = frozenset(";&|><`$\n")

#: Substrings that mark an environment variable as credential-shaped.
_SENSITIVE_MARKERS = (
    "API_KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "CREDENTIAL",
    "ACCESS_KEY",
    "PRIVATE",
)

#: The auth variables the PandaProbe CLI resolves from its environment.
_PANDAPROBE_AUTH_VARS = frozenset(
    {"PANDAPROBE_API_KEY", "PANDAPROBE_PROJECT_NAME", "PANDAPROBE_ENDPOINT", "PANDAPROBE_CONFIG"}
)


def _is_sensitive(name: str) -> bool:
    upper = name.upper()
    if upper.startswith("PANDAPROBE_"):
        return True
    return any(marker in upper for marker in _SENSITIVE_MARKERS)


def _is_ordered_subsequence(needle: Sequence[str], haystack: Sequence[str]) -> bool:
    """True if every token of ``needle`` appears in ``haystack`` in order."""

    if not needle:
        return True
    it = iter(haystack)
    return all(any(token == want for token in it) for want in needle)


class ShellPolicyError(RuntimeError):
    """Raised when a command violates the shell policy."""


@dataclass(frozen=True, slots=True)
class ShellPolicy:
    """Allow-list-based policy for sandboxed command execution."""

    allowed_binaries: frozenset[str] = frozenset(
        {"pandaprobe", "pandaprobe-harness-agent", "cat", "ls", "jq"}
    )
    allow_pipes: bool = False
    workdir: Path = field(default_factory=lambda: Path("/harness"))
    timeout_s: float = 30.0
    #: argv prefixes that are refused even for allow-listed binaries. The
    #: defaults keep the agent away from credential/config surfaces.
    denied_argv_prefixes: tuple[tuple[str, ...], ...] = (
        ("pandaprobe", "config"),
        ("pandaprobe", "auth", "login"),
        ("pandaprobe", "auth", "logout"),
    )
    #: Flags that are refused anywhere in the argv.
    denied_flags: frozenset[str] = frozenset({"--reveal-secrets"})
    #: Per-binary environment passthrough: which scrubbed (sensitive) variables
    #: are restored for which binary. Only binaries that talk to the platform
    #: get the auth variables.
    env_passthrough: dict[str, frozenset[str]] = field(
        default_factory=lambda: {
            "pandaprobe": _PANDAPROBE_AUTH_VARS,
            "pandaprobe-harness-agent": _PANDAPROBE_AUTH_VARS,
        }
    )

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

        # Deny when the binary matches and the denied subcommand words appear,
        # in order, anywhere in the arguments. Matching an ordered subsequence
        # (rather than an exact positional prefix) closes the bypass where a
        # global option is inserted before the subcommand, e.g.
        # `pandaprobe --format json config show` slipping past
        # `("pandaprobe", "config")`. Erring toward over-denial on these
        # credential/config surfaces is intentional.
        for prefix in self.denied_argv_prefixes:
            if prefix and prefix[0] == binary and _is_ordered_subsequence(prefix[1:], argv[1:]):
                raise ShellPolicyError(f"command {' '.join(prefix)!r} is denied by policy")
        for token in argv:
            # Compare the flag name before any `=value`, so `--reveal-secrets=1`
            # is denied just like `--reveal-secrets`.
            flag_name = token.split("=", 1)[0]
            if token in self.denied_flags or flag_name in self.denied_flags:
                raise ShellPolicyError(f"flag {token!r} is denied by policy")

        if not self.allow_pipes:
            for token in argv:
                bad = _SHELL_METACHARS.intersection(token)
                if bad:
                    raise ShellPolicyError(
                        f"shell metacharacter(s) {sorted(bad)} not permitted in {token!r}"
                    )

        for token in argv[1:]:
            self._check_path_escape(token)

    def scrubbed_env(self, argv0: str, base: Mapping[str, str]) -> dict[str, str]:
        """The scoped environment for one command.

        Credential-shaped variables are removed from ``base``; the ones listed
        in :attr:`env_passthrough` for ``argv0`` are restored.
        """

        passthrough = self.env_passthrough.get(argv0, frozenset())
        return {
            name: value
            for name, value in base.items()
            if not _is_sensitive(name) or name in passthrough
        }

    def _check_path_escape(self, token: str) -> None:
        # Skip pure option flags; scrutinize everything else that could name a
        # path. A token need not *start* with a separator to escape the
        # workdir — `state/../../etc/passwd` traverses out mid-path too.
        if not token or token.startswith("-"):
            return
        looks_like_path = (
            "/" in token
            or token in {".", ".."}
            or ".." in token.split("/")
        )
        if not looks_like_path:
            return
        workdir = self.workdir.resolve()
        if token.startswith("/"):
            candidate = Path(token).resolve()
        else:
            candidate = (workdir / token).resolve()
        if candidate != workdir and workdir not in candidate.parents:
            raise ShellPolicyError(f"path {token!r} escapes workdir {workdir}")
