"""Central configuration for the PandaProbe Harness.

``HarnessConfig`` is the single source of truth for filesystem paths, CLI
invocation tunables, metric thresholds, and evaluation feature flags. It is a
frozen dataclass so it can be shared freely across async tasks without risk of
mutation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["HarnessConfig"]


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return default if raw is None else float(raw)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return default if raw is None else int(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class HarnessConfig:
    """Immutable harness configuration.

    Path fields ``traces_dir``, ``rules_file`` and ``latest_eval_file`` are
    derived from ``harness_root`` in ``__post_init__`` and should not be passed
    explicitly.
    """

    harness_root: Path = Path("/harness")

    # Derived paths (init=False; computed from harness_root).
    traces_dir: Path = field(init=False)
    rules_file: Path = field(init=False)
    latest_eval_file: Path = field(init=False)

    # CLI invocation.
    cli_binary: str = "pandaprobe"
    cli_timeout_s: float = 30.0

    # Async eval-run polling (evals runs are asynchronous on the platform).
    poll_interval_s: float = 1.0
    poll_max_attempts: int = 20

    # Bounded await-barrier drained at the start of the next turn.
    drain_timeout_s: float = 15.0

    # Metric thresholds. Scores are 0.0-1.0, higher is better; a score strictly
    # below the threshold is a breach.
    reliability_threshold: float = 0.5
    consistency_threshold: float = 0.5

    # Selective / concurrent evaluation flags.
    eval_reliability: bool = True
    eval_consistency: bool = True
    concurrent_eval: bool = True

    def __post_init__(self) -> None:
        root = Path(self.harness_root)
        # object.__setattr__ is required to populate fields on a frozen dataclass.
        object.__setattr__(self, "harness_root", root)
        object.__setattr__(self, "traces_dir", root / "traces")
        object.__setattr__(self, "rules_file", root / "harness_rules.md")
        object.__setattr__(self, "latest_eval_file", root / "traces" / "latest_eval.json")

    @classmethod
    def from_env(cls, **overrides: object) -> HarnessConfig:
        """Build a config from ``HARNESS_*`` environment variables.

        Explicit ``overrides`` take precedence over environment values.
        """

        values: dict[str, object] = {
            "harness_root": Path(os.environ.get("HARNESS_ROOT", "/harness")),
            "cli_binary": os.environ.get("HARNESS_CLI_BINARY", "pandaprobe"),
            "cli_timeout_s": _env_float("HARNESS_CLI_TIMEOUT_S", 30.0),
            "poll_interval_s": _env_float("HARNESS_POLL_INTERVAL_S", 1.0),
            "poll_max_attempts": _env_int("HARNESS_POLL_MAX_ATTEMPTS", 20),
            "drain_timeout_s": _env_float("HARNESS_DRAIN_TIMEOUT_S", 15.0),
            "reliability_threshold": _env_float("HARNESS_RELIABILITY_THRESHOLD", 0.5),
            "consistency_threshold": _env_float("HARNESS_CONSISTENCY_THRESHOLD", 0.5),
            "eval_reliability": _env_bool("HARNESS_EVAL_RELIABILITY", True),
            "eval_consistency": _env_bool("HARNESS_EVAL_CONSISTENCY", True),
            "concurrent_eval": _env_bool("HARNESS_CONCURRENT_EVAL", True),
        }
        values.update(overrides)
        return cls(**values)  # type: ignore[arg-type]
