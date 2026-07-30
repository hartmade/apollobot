from __future__ import annotations

from types import SimpleNamespace

import pytest

from apollobot.agents import OpenAIProvider, token_cost_usd
from apollobot.service.model_catalog import MODEL_CATALOG, resolve_model_route


EXPECTED_ROUTES = {
    "openai/gpt-oss-120b": "groq",
    "deepseek/deepseek-v4-flash": "deepinfra/fp4",
    "xiaomi/mimo-v2.5": "parasail/fp8",
    "nvidia/nemotron-3-super-120b-a12b": "digitalocean",
    "qwen/qwen3.6-35b-a3b": "deepinfra/fp8",
    "minimax/minimax-m3": "parasail/fp8",
    "deepseek/deepseek-v4-pro": "baidu/fp8",
    "moonshotai/kimi-k2.7-code": "coreweave/int4",
    "z-ai/glm-5.2": "coreweave/fp4",
    "moonshotai/kimi-k3": "moonshotai/mxfp4",
    "anthropic/claude-opus-5": "anthropic",
    "openai/gpt-5.6-sol": "openai",
    "openai/gpt-5.6-terra": "openai",
    "openai/gpt-5.6-luna": "openai",
}


@pytest.mark.parametrize(("model_id", "provider_tag"), EXPECTED_ROUTES.items())
def test_catalog_accepts_only_pinned_model_provider_pairs(
    model_id: str, provider_tag: str
) -> None:
    route = resolve_model_route(model_id, provider_tag)
    assert route.model_id == model_id
    assert route.provider_tag == provider_tag
    assert route.input_cost_per_million > 0
    assert route.output_cost_per_million > 0

    with pytest.raises(ValueError, match="do not match"):
        resolve_model_route(model_id, f"{provider_tag}-untrusted")


def test_catalog_contains_exactly_the_integrated_routes() -> None:
    assert set(MODEL_CATALOG) == set(EXPECTED_ROUTES)


def test_token_cost_uses_pinned_openrouter_rates_without_double_charging_cache() -> None:
    cost = token_cost_usd(
        26,
        19,
        8,
        input_cost_per_million=0.09,
        output_cost_per_million=0.18,
        cached_input_cost_per_million=0.018,
    )

    assert cost == pytest.approx(0.000005184)


def test_token_cost_matches_groq_gpt_oss_probe_invoice() -> None:
    cost = token_cost_usd(
        138,
        155,
        0,
        input_cost_per_million=0.15,
        output_cost_per_million=0.60,
        cached_input_cost_per_million=0.075,
    )

    assert cost == pytest.approx(0.0001137)


def test_token_cost_bounds_invalid_cached_token_count() -> None:
    cost = token_cost_usd(
        10,
        5,
        50,
        input_cost_per_million=0.09,
        output_cost_per_million=0.18,
        cached_input_cost_per_million=0.018,
    )

    assert cost == pytest.approx(0.00000108)


@pytest.mark.asyncio
async def test_selected_route_pins_provider_and_price_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        async def create(self, **request: object) -> SimpleNamespace:
            captured.update(request)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="ready"))],
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=5,
                    prompt_tokens_details=SimpleNamespace(cached_tokens=2),
                ),
            )

    class FakeClient:
        def __init__(self, **_kwargs: object) -> None:
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr("openai.AsyncOpenAI", FakeClient)
    route = resolve_model_route("deepseek/deepseek-v4-flash", "deepinfra/fp4")
    provider = OpenAIProvider(
        api_key="test-key",
        model=route.model_id,
        base_url="https://openrouter.ai/api/v1",
        provider_name=route.billing_provider,
        provider_tag=route.provider_tag,
        data_collection="deny",
        reasoning_effort=route.reasoning_effort,
        reasoning_exclude=True,
        input_cost_per_million=route.input_cost_per_million,
        cached_input_cost_per_million=route.cached_input_cost_per_million,
        output_cost_per_million=route.output_cost_per_million,
        max_tokens=route.max_output_tokens,
    )

    response = await provider.complete([{"role": "user", "content": "ready"}])
    routing = captured["extra_body"]["provider"]  # type: ignore[index]
    assert captured["model"] == "deepseek/deepseek-v4-flash"
    assert routing["only"] == ["deepinfra/fp4"]
    assert routing["allow_fallbacks"] is False
    assert routing["data_collection"] == "deny"
    assert routing["max_price"] == {"prompt": 0.09, "completion": 0.18}
    assert response.provider == "openrouter/deepinfra/fp4"
    assert response.model == "deepseek/deepseek-v4-flash"
