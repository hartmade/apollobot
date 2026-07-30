"""Server-owned AI model routes available to Frontier investigations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True, slots=True)
class ModelRoute:
    model_id: str
    provider_tag: str
    billing_provider: str
    cached_input_cost_per_million: float
    input_cost_per_million: float
    output_cost_per_million: float
    max_output_tokens: int = 4096
    reasoning_effort: str = "low"


DEFAULT_MODEL_ID = "openai/gpt-oss-120b"
MODEL_CATALOG_VERSION = "frontier-model-catalog/2026-07-30"

_MODEL_CATALOG = {
    DEFAULT_MODEL_ID: ModelRoute(
        model_id=DEFAULT_MODEL_ID,
        provider_tag="groq",
        billing_provider="openrouter/groq",
        cached_input_cost_per_million=0.075,
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
    ),
    "deepseek/deepseek-v4-flash": ModelRoute(
        model_id="deepseek/deepseek-v4-flash",
        provider_tag="deepinfra/fp4",
        billing_provider="openrouter/deepinfra/fp4",
        cached_input_cost_per_million=0.018,
        input_cost_per_million=0.09,
        output_cost_per_million=0.18,
    ),
    "xiaomi/mimo-v2.5": ModelRoute(
        model_id="xiaomi/mimo-v2.5",
        provider_tag="parasail/fp8",
        billing_provider="openrouter/parasail/fp8",
        cached_input_cost_per_million=0.05,
        input_cost_per_million=0.14,
        output_cost_per_million=0.28,
    ),
    "nvidia/nemotron-3-super-120b-a12b": ModelRoute(
        model_id="nvidia/nemotron-3-super-120b-a12b",
        provider_tag="digitalocean",
        billing_provider="openrouter/digitalocean",
        cached_input_cost_per_million=0.06,
        input_cost_per_million=0.21,
        output_cost_per_million=0.455,
    ),
    "qwen/qwen3.6-35b-a3b": ModelRoute(
        model_id="qwen/qwen3.6-35b-a3b",
        provider_tag="deepinfra/fp8",
        billing_provider="openrouter/deepinfra/fp8",
        cached_input_cost_per_million=0.0,
        input_cost_per_million=0.10,
        output_cost_per_million=0.95,
    ),
    "minimax/minimax-m3": ModelRoute(
        model_id="minimax/minimax-m3",
        provider_tag="parasail/fp8",
        billing_provider="openrouter/parasail/fp8",
        cached_input_cost_per_million=0.06,
        input_cost_per_million=0.30,
        output_cost_per_million=1.20,
    ),
    "deepseek/deepseek-v4-pro": ModelRoute(
        model_id="deepseek/deepseek-v4-pro",
        provider_tag="baidu/fp8",
        billing_provider="openrouter/baidu/fp8",
        cached_input_cost_per_million=0.0518,
        input_cost_per_million=0.6253,
        output_cost_per_million=1.2506,
    ),
    "moonshotai/kimi-k2.7-code": ModelRoute(
        model_id="moonshotai/kimi-k2.7-code",
        provider_tag="coreweave/int4",
        billing_provider="openrouter/coreweave/int4",
        cached_input_cost_per_million=0.15,
        input_cost_per_million=0.71,
        output_cost_per_million=3.50,
    ),
    "z-ai/glm-5.2": ModelRoute(
        model_id="z-ai/glm-5.2",
        provider_tag="coreweave/fp4",
        billing_provider="openrouter/coreweave/fp4",
        cached_input_cost_per_million=0.14,
        input_cost_per_million=0.76,
        output_cost_per_million=2.42,
    ),
    "moonshotai/kimi-k3": ModelRoute(
        model_id="moonshotai/kimi-k3",
        provider_tag="moonshotai/mxfp4",
        billing_provider="openrouter/moonshotai/mxfp4",
        cached_input_cost_per_million=0.30,
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
    ),
    "anthropic/claude-opus-5": ModelRoute(
        model_id="anthropic/claude-opus-5",
        provider_tag="anthropic",
        billing_provider="openrouter/anthropic",
        cached_input_cost_per_million=0.50,
        input_cost_per_million=5.00,
        output_cost_per_million=25.00,
    ),
    "openai/gpt-5.6-sol": ModelRoute(
        model_id="openai/gpt-5.6-sol",
        provider_tag="openai",
        billing_provider="openrouter/openai",
        cached_input_cost_per_million=0.50,
        input_cost_per_million=5.00,
        output_cost_per_million=30.00,
    ),
    "openai/gpt-5.6-terra": ModelRoute(
        model_id="openai/gpt-5.6-terra",
        provider_tag="openai",
        billing_provider="openrouter/openai",
        cached_input_cost_per_million=0.20,
        input_cost_per_million=2.00,
        output_cost_per_million=12.00,
    ),
    "openai/gpt-5.6-luna": ModelRoute(
        model_id="openai/gpt-5.6-luna",
        provider_tag="openai",
        billing_provider="openrouter/openai",
        cached_input_cost_per_million=0.02,
        input_cost_per_million=0.20,
        output_cost_per_million=1.20,
    ),
}

MODEL_CATALOG: Mapping[str, ModelRoute] = MappingProxyType(_MODEL_CATALOG)


def resolve_model_route(
    model_id: object = None,
    requested_provider_tag: object = None,
) -> ModelRoute:
    """Resolve an allowlisted route without accepting browser-owned routing data."""
    selected_id = DEFAULT_MODEL_ID if model_id is None else model_id
    if not isinstance(selected_id, str) or selected_id not in MODEL_CATALOG:
        raise ValueError("Selected AI model is not supported")
    route = MODEL_CATALOG[selected_id]
    if requested_provider_tag is not None:
        if (
            not isinstance(requested_provider_tag, str)
            or requested_provider_tag != route.provider_tag
        ):
            raise ValueError("Selected AI model and provider route do not match")
    return route


__all__ = [
    "DEFAULT_MODEL_ID",
    "MODEL_CATALOG",
    "MODEL_CATALOG_VERSION",
    "ModelRoute",
    "resolve_model_route",
]
