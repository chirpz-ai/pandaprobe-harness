"""Model registry: resolve a study model key to a LiteLLM model string.

The study's provider-routing constraints (Gemini via Vertex; OpenAI via OpenAI's
API; open-weight models via Bedrock; Claude switchable across AWS Bedrock and
Anthropic API) are encoded entirely in ``configs/models.yaml`` — switching a
model or backend is a config change, never a code change. This module turns a
config key + optional backend into a fully-resolved :class:`ResolvedModel` that
the LiteLLM wrapper consumes.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = ["ModelRegistry", "ResolvedModel", "load_registry", "provider_of"]

# LiteLLM model-string prefix -> the coarse provider label recorded in results.
_PROVIDER_BY_PREFIX = {
    "vertex_ai": "vertex",
    "anthropic": "anthropic",
    "openai": "openai",
    "gemini": "gemini",
    "azure": "azure",
    "bedrock": "bedrock",
}


def provider_of(litellm_model: str) -> str:
    """Coarse provider label from a LiteLLM model string's prefix."""

    prefix = litellm_model.split("/", 1)[0]
    return _PROVIDER_BY_PREFIX.get(prefix, prefix)


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    """A model key resolved to a concrete LiteLLM call target."""

    key: str
    """The models.yaml key, e.g. ``claude-sonnet-5`` — recorded in results."""
    litellm_model: str
    """The LiteLLM string, e.g. ``bedrock/global.anthropic.claude-sonnet-5``."""
    provider: str
    """Coarse provider label, e.g. ``bedrock`` / ``anthropic`` / ``openai``."""
    backend: str | None
    """The chosen backend for dual-backend models, else ``None``."""
    param_allowlist: frozenset[str]
    """Sampler/params LiteLLM may forward; everything else is dropped."""
    default_params: dict[str, Any]
    """Provider-required constants sent on **every** call to this model, e.g.
    ``reasoning_effort: none`` for GPT-5.6 (which otherwise 400s when function
    tools are combined with reasoning on ``/v1/chat/completions``).

    Distinct from ``param_allowlist``, which gates *caller-supplied* sampler
    knobs: these are not the caller's choice, they are what the model requires.
    A caller-supplied value of the same name wins, so a run can still override."""
    price_per_mtok: dict[str, float] | None
    """Fallback USD price per 1M tokens ``{input, output}`` when
    ``litellm.completion_cost`` lacks a price."""
    default_max_tokens: int | None = None
    """Per-model task/replay output budget, or the client's default when unset."""
    repair_overrides: dict[str, Any] = field(default_factory=dict)
    """Managed-repair settings used when this task model is also the repair model."""
    is_mock: bool = False
    """True for the dry-run pseudo-model (no real API calls)."""

    def cost_from_usage(self, input_tokens: int, output_tokens: int) -> float | None:
        """Fallback cost from the price table; ``None`` if no table configured."""

        if not self.price_per_mtok:
            return None
        return (
            input_tokens / 1_000_000 * self.price_per_mtok.get("input", 0.0)
            + output_tokens / 1_000_000 * self.price_per_mtok.get("output", 0.0)
        )


@dataclass(frozen=True, slots=True)
class _ModelSpec:
    """Raw models.yaml entry (one model key)."""

    key: str
    provider_family: str
    litellm_model: str | None
    backends: dict[str, str]
    default_backend: str | None
    param_allowlist: frozenset[str]
    default_params: dict[str, Any]
    price_per_mtok: dict[str, float] | None
    default_max_tokens: int | None
    repair_overrides: dict[str, Any]
    is_mock: bool


class ModelRegistry:
    """Parsed ``models.yaml``: model specs plus named operational roles."""

    def __init__(self, specs: Mapping[str, _ModelSpec], roles: Mapping[str, str]) -> None:
        self._specs = dict(specs)
        self._roles = dict(roles)

    def keys(self) -> list[str]:
        return list(self._specs)

    def role(self, name: str) -> str:
        """The model key bound to a role (for example ``dry_run`` or ``smoke``)."""

        try:
            return self._roles[name]
        except KeyError:
            raise KeyError(
                f"no model role {name!r} in models.yaml (roles: {list(self._roles)})"
            ) from None

    def resolve(
        self,
        key: str,
        *,
        backend: str | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ResolvedModel:
        """Resolve a model key (+ optional backend override) to a call target.

        Backend precedence for dual-backend models: explicit ``backend`` arg >
        ``CLAUDE_BACKEND`` env > the spec's ``default_backend``.
        """

        env = os.environ if env is None else env
        try:
            spec = self._specs[key]
        except KeyError:
            raise KeyError(f"unknown model key {key!r} (known: {list(self._specs)})") from None

        if spec.backends:
            chosen = backend or env.get("CLAUDE_BACKEND") or spec.default_backend
            if chosen is None:
                raise ValueError(
                    f"model {key!r} has multiple backends but no default; pass --backend"
                )
            if chosen not in spec.backends:
                raise ValueError(
                    f"model {key!r} has no backend {chosen!r} (available: {list(spec.backends)})"
                )
            litellm_model = spec.backends[chosen]
            resolved_backend: str | None = chosen
        else:
            if spec.litellm_model is None:
                raise ValueError(f"model {key!r} declares neither litellm_model nor backends")
            if backend is not None and not spec.is_mock:
                raise ValueError(
                    f"model {key!r} is single-backend; --backend {backend!r} not applicable"
                )
            litellm_model = spec.litellm_model
            resolved_backend = None

        return ResolvedModel(
            key=key,
            litellm_model=litellm_model,
            provider=provider_of(litellm_model),
            backend=resolved_backend,
            param_allowlist=spec.param_allowlist,
            default_params=dict(spec.default_params),
            price_per_mtok=spec.price_per_mtok,
            default_max_tokens=spec.default_max_tokens,
            repair_overrides=dict(spec.repair_overrides),
            is_mock=spec.is_mock,
        )


def _parse_spec(key: str, raw: Mapping[str, Any]) -> _ModelSpec:
    backends = dict(raw.get("backends") or {})
    price = raw.get("price_per_mtok")
    default_max_tokens_raw = raw.get("default_max_tokens")
    default_max_tokens = (
        int(default_max_tokens_raw) if default_max_tokens_raw is not None else None
    )
    if default_max_tokens is not None and default_max_tokens <= 0:
        raise ValueError(f"model {key!r}: default_max_tokens must be positive")

    repair_raw = raw.get("repair_overrides") or {}
    if not isinstance(repair_raw, Mapping):
        raise ValueError(f"model {key!r}: repair_overrides must be a mapping")
    allowed_repair_keys = {
        "timeout_s",
        "max_turns",
        "max_tokens",
        "temperature",
        "reasoning_effort",
    }
    unknown_repair_keys = set(repair_raw) - allowed_repair_keys
    if unknown_repair_keys:
        raise ValueError(
            f"model {key!r}: unknown repair_overrides keys: "
            f"{sorted(unknown_repair_keys)}"
        )
    repair_overrides = {str(k): v for k, v in repair_raw.items()}
    for name in ("max_turns", "max_tokens"):
        value = repair_overrides.get(name)
        if value is not None and (isinstance(value, bool) or int(value) <= 0):
            raise ValueError(f"model {key!r}: repair_overrides.{name} must be positive")
        if value is not None:
            repair_overrides[name] = int(value)
    if repair_overrides.get("timeout_s") is not None:
        timeout_s = float(repair_overrides["timeout_s"])
        if timeout_s <= 0:
            raise ValueError(
                f"model {key!r}: repair_overrides.timeout_s must be positive"
            )
        repair_overrides["timeout_s"] = timeout_s
    if repair_overrides.get("temperature") is not None:
        repair_overrides["temperature"] = float(repair_overrides["temperature"])
    if repair_overrides.get("reasoning_effort") is not None:
        repair_overrides["reasoning_effort"] = str(
            repair_overrides["reasoning_effort"]
        )
    return _ModelSpec(
        key=key,
        provider_family=str(raw.get("provider_family", "")),
        litellm_model=raw.get("litellm_model"),
        backends={str(k): str(v) for k, v in backends.items()},
        default_backend=raw.get("default_backend"),
        param_allowlist=frozenset(raw.get("param_allowlist") or ()),
        default_params=dict(raw.get("default_params") or {}),
        price_per_mtok={str(k): float(v) for k, v in price.items()} if price else None,
        default_max_tokens=default_max_tokens,
        repair_overrides=repair_overrides,
        is_mock=bool(raw.get("mock", False)),
    )


def load_registry(path: str | Path) -> ModelRegistry:
    """Load and validate ``models.yaml``."""

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    models_raw = data.get("models") or {}
    if not isinstance(models_raw, dict):
        raise ValueError(f"{path}: 'models' must be a mapping")
    specs = {str(key): _parse_spec(str(key), raw) for key, raw in models_raw.items()}
    roles = {str(k): str(v) for k, v in (data.get("roles") or {}).items()}
    return ModelRegistry(specs, roles)
