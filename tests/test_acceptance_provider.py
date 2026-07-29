"""Acceptance-only deterministic model boundary."""

import pytest

from apollobot.agents import AcceptanceProvider, create_llm
from apollobot.core import APIConfig


def test_acceptance_provider_is_unavailable_by_default(monkeypatch):
    monkeypatch.delenv("APOLLOBOT_ENV", raising=False)
    assert APIConfig(default_provider="acceptance").get_key() == ""
    with pytest.raises(ValueError, match="disabled outside acceptance"):
        create_llm("acceptance", "")


@pytest.mark.asyncio
async def test_acceptance_provider_returns_an_executable_plan(monkeypatch):
    monkeypatch.setenv("APOLLOBOT_ENV", "acceptance")
    provider = AcceptanceProvider()
    response = await provider.complete_json(
        [{"role": "user", "content": "# Research Objective\nVerify the workflow"}],
        system="You are a senior research scientist planning a computational study.",
    )
    assert response["analysis_steps"][0]["method"] == "deterministic_fixture"
    assert response["literature_queries"] == []
    assert response["risks"]


@pytest.mark.asyncio
async def test_acceptance_provider_generated_code_is_bounded(monkeypatch):
    monkeypatch.setenv("APOLLOBOT_ENV", "acceptance")
    provider = AcceptanceProvider()
    response = await provider.complete(
        [
            {
                "role": "user",
                "content": "Generate Python code. Use preregistered random seed 4242.",
            }
        ],
        system="You are a computational scientist writing analysis code.",
    )
    assert "subprocess" not in response.text
    assert "'fixture': True" in response.text
    assert "'random_seed': 4242" in response.text


@pytest.mark.asyncio
async def test_acceptance_provider_returns_complete_submission_review(monkeypatch):
    monkeypatch.setenv("APOLLOBOT_ENV", "acceptance")
    provider = AcceptanceProvider()

    base_review = await provider.complete_json(
        [{"role": "user", "content": "Review this scientific manuscript."}],
        system="You are an expert peer reviewer evaluating a computational research paper.",
    )
    scoring = await provider.complete_json(
        [{"role": "user", "content": "Score this manuscript on each dimension."}],
        system="You are a senior journal editor scoring a manuscript for peer review.",
    )

    assert base_review["overall_verdict"] == "accept"
    assert scoring["recommendation"] == "accept"
    assert [item["dimension"] for item in scoring["scores"]] == [
        "statistical_rigor",
        "methodological_soundness",
        "reproducibility",
        "novelty",
        "clarity",
    ]
