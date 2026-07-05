"""The companion CLI: harness tools for sandboxed / framework-less agents.

Installed as the ``pandaprobe-harness-agent`` console script (also runnable as
``python -m pandaprobe_harness.agent_tools``), this is the second delivery
channel for the toolset: a pure-stdlib binary a *sandboxed* agent can invoke
through the ``RestrictedShellTool`` with zero framework coupling::

    pandaprobe-harness-agent harness_mailbox_list
    pandaprobe-harness-agent harness_rule_add --rule "..." --rationale "..."

Arguments are ``--key value`` pairs; each value is parsed as JSON when
possible, else taken as a string. The result envelope is printed as JSON;
the exit code is 0 when ``ok`` is true, 1 otherwise.

The workspace is resolved from ``HARNESS_*`` environment variables and the
platform is reached through the ``pandaprobe`` binary (``HARNESS_CLI_BINARY``)
— never the REST API directly.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Sequence
from typing import Any

from ..cli.subprocess_client import SubprocessCliClient
from ..config import HarnessConfig
from ..evaluation.history import ScoreHistoryStore
from ..workspace.evalset import EvalSet
from ..workspace.journal import Journal
from ..workspace.mailbox import Mailbox
from ..workspace.rules import RulesStore
from .toolset import OP_SCHEMAS, HarnessToolset

__all__ = ["build_toolset_from_env", "main"]


def build_toolset_from_env() -> HarnessToolset:
    """Assemble a toolset over the env-configured workspace and CLI."""

    config = HarnessConfig.from_env()
    mailbox = Mailbox(config)
    mailbox.provision()
    journal = Journal(config)
    rules = RulesStore(config, journal=journal)
    history = ScoreHistoryStore(config)
    evalset = EvalSet(config, journal=journal)
    evalset.provision()
    cli = SubprocessCliClient(config.cli_binary, default_timeout=config.cli_timeout_s)
    return HarnessToolset(
        config=config,
        cli=cli,
        mailbox=mailbox,
        journal=journal,
        rules=rules,
        history=history,
        evalset=evalset,
    )


def _usage() -> str:
    lines = [
        "usage: pandaprobe-harness-agent <operation> [--key value ...]",
        "",
        "Self-diagnostic harness operations (values parsed as JSON when possible):",
        "",
    ]
    for name, meta in OP_SCHEMAS.items():
        schema = meta["input_schema"]
        params = ", ".join(schema.get("properties", {}))
        lines.append(f"  {name}({params})")
        lines.append(f"      {meta['description']}")
    return "\n".join(lines)


def _parse_args(tokens: Sequence[str]) -> dict[str, Any] | str:
    """``--key value`` pairs → mapping; returns an error string on bad shape."""

    args: dict[str, Any] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--") or len(token) <= 2:
            return f"expected --key, got {token!r}"
        if index + 1 >= len(tokens):
            return f"missing value for {token!r}"
        raw = tokens[index + 1]
        if raw.startswith("--"):
            # A forgotten value would otherwise silently consume the next flag
            # as data (e.g. `--rationale --metric` -> rationale="--metric").
            return f"missing value for {token!r} (got flag {raw!r})"
        try:
            value: Any = json.loads(raw)
        except ValueError:
            value = raw
        args[token[2:].replace("-", "_")] = value
        index += 2
    return args


def main(argv: Sequence[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    if not tokens or tokens[0] in {"-h", "--help", "help"}:
        print(_usage())
        return 0

    op = tokens[0]
    if op not in OP_SCHEMAS:
        print(json.dumps({"ok": False, "error": f"unknown operation {op!r}"}))
        return 1
    parsed = _parse_args(tokens[1:])
    if isinstance(parsed, str):
        print(json.dumps({"ok": False, "error": parsed}))
        return 1

    try:
        toolset = build_toolset_from_env()
        result = asyncio.run(toolset.call(op, parsed))
    except Exception as exc:  # noqa: BLE001 - a CLI must not traceback at the agent
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("ok") else 1
