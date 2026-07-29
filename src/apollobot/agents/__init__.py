"""
Agent layer for ApolloBot — LLM providers and reasoning components.

Provides:
- LLMResponse: Response from an LLM call
- LLMProvider: Abstract base class for LLM providers
- AnthropicProvider: Claude implementation
- OpenAIProvider: OpenAI implementation
- create_llm: Factory function to instantiate providers
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    """Response from an LLM API call."""

    text: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    cached_input_tokens: int = 0


def token_cost_usd(
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int,
    *,
    input_cost_per_million: float,
    output_cost_per_million: float,
    cached_input_cost_per_million: float,
) -> float:
    """Calculate provider cost without charging cached prompt tokens twice."""
    cached = min(max(0, cached_input_tokens), max(0, input_tokens))
    uncached = max(0, input_tokens) - cached
    return (
        uncached * input_cost_per_million
        + cached * cached_input_cost_per_million
        + max(0, output_tokens) * output_cost_per_million
    ) / 1_000_000


def environment_price(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError as error:
        raise RuntimeError(f"{name} must be numeric") from error
    if value < 0 or value > 10_000:
        raise RuntimeError(f"{name} must be between 0 and 10000")
    return value


def environment_token_limit(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer") from error
    if value < 128 or value > 1_000_000:
        raise RuntimeError(f"{name} must be between 128 and 1000000")
    return value


def checked_price(value: float, label: str) -> float:
    if not isinstance(value, (int, float)) or value < 0 or value > 10_000:
        raise RuntimeError(f"{label} must be between 0 and 10000")
    return float(value)


def checked_token_limit(value: int, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 128 or value > 1_000_000:
        raise RuntimeError(f"{label} must be between 128 and 1000000")
    return value


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @staticmethod
    def _clean_text(text: str) -> str:
        """Strip <think>...</think> reasoning blocks from LLM output."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @abstractmethod
    async def complete(self, messages: list[dict[str, str]], system: str = "") -> LLMResponse:
        """Generate a completion from the LLM."""
        ...

    @staticmethod
    def _fix_json(text: str) -> str:
        """Apply common JSON fixes for non-standard LLM output."""
        # Strip /* block comments */
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        # Strip // line comments only at start of lines (safe — won't hit URLs)
        text = re.sub(r"^\s*//[^\n]*\n?", "", text, flags=re.MULTILINE)
        # Strip // comments after JSON structural tokens (not inside strings)
        text = re.sub(r"(?<=[,\]\}\d])\s*//[^\n]*", "", text)
        # Fix trailing commas before } or ] (common LLM output issue)
        text = re.sub(r",\s*([}\]])", r"\1", text)
        return text

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any]:
        """Extract and parse JSON from LLM output, handling common issues."""
        text = raw.strip()
        # Strip <think>...</think> blocks (e.g. from reasoning models)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # Strip <output>...</output> and similar XML wrapper tags
        text = re.sub(r"<(?:output|response|result|json|answer)>", "", text, flags=re.IGNORECASE)
        text = re.sub(r"</(?:output|response|result|json|answer)>", "", text, flags=re.IGNORECASE)
        # Handle markdown code blocks (possibly with language tag)
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        # Strip any leading prose before the first {
        first_brace = text.find("{")
        if first_brace > 0 and first_brace < 500:
            # Check if everything before { is non-JSON prose
            prefix = text[:first_brace].strip()
            if prefix and not prefix.startswith("["):
                text = text[first_brace:]
        text = text.strip()

        # Try parsing after basic fixes
        fixed = LLMProvider._fix_json(text)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # Fallback: find the outermost { ... } block via brace matching
        start = text.find("{")
        if start != -1:
            depth = 0
            in_str = False
            escape = False
            for i in range(start, len(text)):
                c = text[i]
                if escape:
                    escape = False
                    continue
                if c == "\\":
                    escape = True
                    continue
                if c == '"' and not escape:
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        block = text[start : i + 1]
                        block = LLMProvider._fix_json(block)
                        try:
                            return json.loads(block)
                        except json.JSONDecodeError:
                            break

            # Try aggressive cleanup on the extracted region
            region = text[start:]
            # Find last } in the region
            last_brace = region.rfind("}")
            if last_brace != -1:
                block = region[: last_brace + 1]
                block = LLMProvider._fix_json(block)
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    pass

            # Last resort: replace single quotes with double quotes
            try:
                fixed = re.sub(r"'([^']*)'(?=\s*:)", r'"\1"', region)
                fixed = LLMProvider._fix_json(fixed)
                return json.loads(fixed)
            except (json.JSONDecodeError, IndexError):
                pass

        raise json.JSONDecodeError(
            f"Could not parse JSON from LLM output (length={len(raw)})",
            raw[:200],
            0,
        )

    async def complete_json(
        self, messages: list[dict[str, str]], system: str = "", retries: int = 2
    ) -> dict[str, Any]:
        """Generate a completion and parse as JSON, with retry on parse failure."""
        json_system = (
            f"{system}\n\n" if system else ""
        ) + "Return only one valid JSON object with no markdown or commentary."
        last_error = None
        for attempt in range(1 + retries):
            if attempt == 0:
                response = await self.complete(messages, json_system)
            else:
                # Retry with error feedback appended
                retry_messages = messages + [
                    {"role": "assistant", "content": response.text},
                    {
                        "role": "user",
                        "content": (
                            f"Your response could not be parsed as JSON: {last_error}\n"
                            "Please return ONLY valid JSON with no comments, no trailing "
                            "commas, and double-quoted keys. No markdown fences."
                        ),
                    },
                ]
                response = await self.complete(retry_messages, json_system)
            try:
                return self._extract_json(response.text)
            except json.JSONDecodeError as e:
                last_error = str(e)
                continue
        raise json.JSONDecodeError(
            f"Failed to parse JSON after {1 + retries} attempts: {last_error}",
            response.text[:200],
            0,
        )


class AnthropicProvider(LLMProvider):
    """Claude implementation of LLMProvider."""

    # Cost per million tokens (approximate, Claude Sonnet 3.5)
    INPUT_COST_PER_M = 3.0
    OUTPUT_COST_PER_M = 15.0

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        import anthropic

        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    async def complete(self, messages: list[dict[str, str]], system: str = "") -> LLMResponse:
        response = await self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system or "You are a helpful research assistant.",
            messages=messages,
        )

        raw_text = response.content[0].text if response.content else ""
        text = self._clean_text(raw_text)
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        cost = (input_tokens / 1_000_000) * self.INPUT_COST_PER_M + (
            output_tokens / 1_000_000
        ) * self.OUTPUT_COST_PER_M

        return LLMResponse(
            text=text,
            provider="anthropic",
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )


class OpenAIProvider(LLMProvider):
    """OpenAI implementation of LLMProvider."""

    # Cost per million tokens (approximate, GPT-4o)
    INPUT_COST_PER_M = 5.0
    CACHED_INPUT_COST_PER_M = 5.0
    OUTPUT_COST_PER_M = 15.0

    def __init__(
        self,
        api_key: str,
        model: str = "",
        base_url: str = "",
        *,
        provider_name: str | None = None,
        provider_tag: str | None = None,
        data_collection: str | None = None,
        reasoning_effort: str | None = None,
        reasoning_exclude: bool | None = None,
        input_cost_per_million: float | None = None,
        cached_input_cost_per_million: float | None = None,
        output_cost_per_million: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        import openai

        kwargs: dict[str, Any] = {"api_key": api_key}
        base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
        if base_url:
            kwargs["base_url"] = base_url
        if "openrouter.ai" in base_url:
            kwargs["default_headers"] = {
                "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", "https://frontier-science.ai"),
                "X-Title": os.getenv("OPENROUTER_APP_NAME", "Frontier Science"),
            }
        self.client = openai.AsyncOpenAI(**kwargs)
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
        self.provider_name = provider_name or os.getenv(
            "APOLLOBOT_MODEL_BILLING_PROVIDER", "openai"
        )
        self.provider_tag = (
            os.getenv("OPENROUTER_PROVIDER_TAG", "")
            if provider_tag is None
            else provider_tag
        ).strip()
        self.data_collection = (
            os.getenv("OPENROUTER_DATA_COLLECTION", "deny")
            if data_collection is None
            else data_collection
        ).strip()
        self.reasoning_effort = (
            os.getenv("OPENROUTER_REASONING_EFFORT", "")
            if reasoning_effort is None
            else reasoning_effort
        ).strip().lower()
        if self.reasoning_effort not in {"", "low", "medium", "high"}:
            raise RuntimeError("OPENROUTER_REASONING_EFFORT must be low, medium, high, or empty")
        self.reasoning_exclude = (
            os.getenv("OPENROUTER_REASONING_EXCLUDE", "1").strip().lower()
            in {"1", "true", "yes"}
            if reasoning_exclude is None
            else reasoning_exclude
        )
        self.input_cost_per_million = (
            environment_price("APOLLOBOT_MODEL_INPUT_COST_PER_M", self.INPUT_COST_PER_M)
            if input_cost_per_million is None
            else checked_price(input_cost_per_million, "input_cost_per_million")
        )
        self.cached_input_cost_per_million = (
            environment_price(
                "APOLLOBOT_MODEL_CACHED_INPUT_COST_PER_M", self.CACHED_INPUT_COST_PER_M
            )
            if cached_input_cost_per_million is None
            else checked_price(
                cached_input_cost_per_million, "cached_input_cost_per_million"
            )
        )
        self.output_cost_per_million = (
            environment_price("APOLLOBOT_MODEL_OUTPUT_COST_PER_M", self.OUTPUT_COST_PER_M)
            if output_cost_per_million is None
            else checked_price(output_cost_per_million, "output_cost_per_million")
        )
        self.max_tokens = (
            environment_token_limit("APOLLOBOT_MODEL_MAX_OUTPUT_TOKENS", 8192)
            if max_tokens is None
            else checked_token_limit(max_tokens, "max_tokens")
        )

    async def complete(self, messages: list[dict[str, str]], system: str = "") -> LLMResponse:
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": all_messages,
        }
        if "Return only one valid JSON object" in system:
            request["response_format"] = {"type": "json_object"}
        if self.provider_tag:
            request["extra_body"] = {
                "provider": {
                    "only": [self.provider_tag],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                    "data_collection": self.data_collection,
                    "max_price": {
                        "prompt": self.input_cost_per_million,
                        "completion": self.output_cost_per_million,
                    },
                }
            }
            if self.reasoning_effort:
                request["extra_body"]["reasoning"] = {
                    "effort": self.reasoning_effort,
                    "exclude": self.reasoning_exclude,
                }
        response = await self.client.chat.completions.create(**request)

        if not response.choices:
            raise RuntimeError(f"LLM returned no choices (model={self.model})")
        raw_text = response.choices[0].message.content or ""
        text = self._clean_text(raw_text)
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        details = response.usage.prompt_tokens_details if response.usage else None
        cached_input_tokens = int(getattr(details, "cached_tokens", 0) or 0)
        reported_cost = getattr(response.usage, "cost", None) if response.usage else None
        cost = float(reported_cost) if isinstance(reported_cost, (int, float)) else token_cost_usd(
            input_tokens,
            output_tokens,
            cached_input_tokens,
            input_cost_per_million=self.input_cost_per_million,
            output_cost_per_million=self.output_cost_per_million,
            cached_input_cost_per_million=self.cached_input_cost_per_million,
        )

        return LLMResponse(
            text=text,
            provider=self.provider_name,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            cached_input_tokens=cached_input_tokens,
        )


class MiniMaxProvider(OpenAIProvider):
    """MiniMax implementation using OpenAI-compatible API."""

    # MiniMax M2.5 pricing
    INPUT_COST_PER_M = 0.30
    OUTPUT_COST_PER_M = 1.20

    def __init__(self, api_key: str, model: str = "MiniMax-M2.5") -> None:
        super().__init__(
            api_key=api_key,
            model=model,
            base_url="https://api.minimax.io/v1",
        )

    async def complete(self, messages: list[dict[str, str]], system: str = "") -> LLMResponse:
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=16384,
            messages=all_messages,
        )

        if not response.choices:
            raise RuntimeError(f"LLM returned no choices (model={self.model})")
        raw_text = response.choices[0].message.content or ""
        text = self._clean_text(raw_text)
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        cost = (input_tokens / 1_000_000) * self.INPUT_COST_PER_M + (
            output_tokens / 1_000_000
        ) * self.OUTPUT_COST_PER_M

        return LLMResponse(
            text=text,
            provider="minimax",
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )


class AcceptanceProvider(LLMProvider):
    """Deterministic responses for isolated end-to-end acceptance runs only."""

    def __init__(self) -> None:
        if os.getenv("APOLLOBOT_ENV", "development").lower() != "acceptance":
            raise ValueError("The acceptance provider is disabled outside acceptance runs")

    async def complete(self, messages: list[dict[str, str]], system: str = "") -> LLMResponse:
        prompt = "\n".join(message.get("content", "") for message in messages)
        text = self._response(prompt, system)
        return LLMResponse(
            text=text,
            provider="acceptance",
            model="deterministic-scientific-fixture-v1",
            input_tokens=max(1, len(prompt) // 4),
            output_tokens=max(1, len(text) // 4),
            cost_usd=0.0,
        )

    @staticmethod
    def _response(prompt: str, system: str) -> str:
        if "Scientific question:" in prompt:
            question = prompt.split("Scientific question:", 1)[1].strip()
            return json.dumps(
                {
                    "title": question.rstrip("?"),
                    "domain": "climate",
                    "apollo_domain": "economics",
                    "mode": "hypothesis",
                    "answerability": "investigable",
                    "novelty": "adjacent-work-likely",
                    "rationale": (
                        "The question supports a falsifiable computational comparison using "
                        "public evidence and a preregistered robustness check."
                    ),
                    "hypotheses": [
                        "The primary relationship is detectable in the declared sample.",
                        "The direction remains stable under the preregistered sensitivity test.",
                    ],
                    "proposedSteps": [
                        {
                            "type": "literature",
                            "label": "Map prior work",
                            "detail": "Locate the closest claims and unresolved disagreements.",
                        },
                        {
                            "type": "hypothesis",
                            "label": "Preregister hypotheses",
                            "detail": "Declare support, failure, and stopping conditions.",
                        },
                        {
                            "type": "experiment",
                            "label": "Design a bounded test",
                            "detail": "Capture controls, seed, and expected outputs.",
                        },
                        {
                            "type": "compute",
                            "label": "Execute and stress-test",
                            "detail": "Run the approved code in the configured sandbox.",
                        },
                        {
                            "type": "review",
                            "label": "Review the evidence",
                            "detail": "Connect every conclusion to captured artifacts.",
                        },
                    ],
                    "estimate": {
                        "durationMinutes": 8,
                        "computeUsd": 1.0,
                        "literatureTargets": 6,
                    },
                    "source": "apollobot",
                }
            )
        if "skeptical peer reviewer" in system:
            return json.dumps({"issues": [], "severity": "low", "suggestions": []})
        if "senior research scientist planning" in system:
            return json.dumps(
                {
                    "summary": "Run one bounded deterministic analysis and preserve its lineage.",
                    "approach": (
                        "Use a preregistered synthetic fixture to verify the research execution "
                        "contract, not to make a real-world scientific claim."
                    ),
                    "hypotheses": [
                        {
                            "hypothesis": "The controlled fixture completes successfully.",
                            "test": "Execute the captured script and inspect its JSON output.",
                            "null_hypothesis": "The controlled fixture does not complete.",
                        }
                    ],
                    "literature_queries": [],
                    "data_requirements": [],
                    "analysis_steps": [
                        {
                            "name": "controlled_acceptance_analysis",
                            "description": "Emit a preregistered result for workflow validation.",
                            "method": "deterministic_fixture",
                            "inputs": [],
                            "parameters": {"fixture": True},
                            "expected_output": "A JSON result and processed artifact.",
                            "statistical_tests": ["workflow assertion"],
                        }
                    ],
                    "statistical_framework": "Deterministic workflow assertions only.",
                    "expected_outputs": ["JSON result", "manuscript", "replication materials"],
                    "risks": [
                        "Acceptance output is synthetic and must never be interpreted as science."
                    ],
                    "estimated_compute_cost": 0.0,
                    "estimated_time_hours": 0.05,
                }
            )
        if "computational scientist writing analysis code" in system:
            seed_match = re.search(
                r"(?:preregistered random seed|random seed|seed)\D{0,24}(\d+)",
                prompt,
                flags=re.IGNORECASE,
            )
            random_seed = int(seed_match.group(1)) if seed_match else 1729
            return (
                "from pathlib import Path\n"
                "import json\n"
                "Path('data/processed').mkdir(parents=True, exist_ok=True)\n"
                "result = {'fixture': True, 'workflow_complete': True, "
                f"'effect_size': 0.0, 'random_seed': {random_seed}}}\n"
                "Path('data/processed/acceptance-result.json').write_text(json.dumps(result))\n"
                "print(json.dumps(result))\n"
            )
        if "biostatistician" in system:
            return (
                "import json\n"
                "print(json.dumps({'hypothesis': 'workflow assertion', "
                "'status': 'supported', 'fixture': True, 'effect_size': 0.0}))\n"
            )
        if "statistical auditor" in system:
            return json.dumps(
                {
                    "checks": [
                        {
                            "name": "fixture_disclosure",
                            "status": "pass",
                            "note": "The manuscript identifies the acceptance-only evidence.",
                        }
                    ],
                    "overall": "pass_with_notes",
                    "fabrication_detected": False,
                }
            )
        if "technology transfer specialist" in system:
            return json.dumps(
                {
                    "commercial_relevance": 0,
                    "implementation_feasibility": 10,
                    "novelty": 0,
                }
            )
        if "revising a scientific manuscript" in system:
            return (
                "# Acceptance-only research record\n\n"
                "## Abstract\nThis synthetic run validates the publication workflow only.\n\n"
                "## Introduction\nNo real-world scientific claim is evaluated.\n\n"
                "## Methods\nA deterministic script ran in the configured sandbox.\n\n"
                "## Results\nThe captured workflow assertion completed successfully.\n\n"
                "## Discussion\nThe output cannot support a scientific conclusion.\n\n"
                "## Conclusion\nThe end-to-end research contract remained intact.\n"
            )
        if "writing a scientific paper" in system:
            section = "section"
            match = re.search(r"Write the ([A-Z]+) section", prompt)
            if match:
                section = match.group(1).lower()
            return (
                f"This {section} describes an acceptance-only synthetic execution. "
                "The captured artifact reports that the workflow assertion completed; it does "
                "not constitute evidence for any real-world scientific claim."
            )
        if "harsh but fair peer reviewer" in system:
            return (
                "The workflow is reproducible and the synthetic boundary is clear. "
                "Do not interpret the fixture output as a scientific finding."
            )
        if "expert peer reviewer evaluating a computational research paper" in system:
            return json.dumps(
                {
                    "overall_verdict": "accept",
                    "confidence": 0.99,
                    "issues": [],
                    "strengths": [
                        "The record clearly labels its evidence as an acceptance-only fixture.",
                        "Execution artifacts and provenance remain attached to the record.",
                    ],
                    "summary": (
                        "The synthetic record is suitable for validating the publication "
                        "workflow. It makes no real-world scientific claim."
                    ),
                }
            )
        if "senior journal editor scoring a manuscript" in system:
            return json.dumps(
                {
                    "recommendation": "accept",
                    "confidence": 0.99,
                    "scores": [
                        {
                            "dimension": "statistical_rigor",
                            "score": 8,
                            "justification": "No statistical claim extends beyond the fixture.",
                        },
                        {
                            "dimension": "methodological_soundness",
                            "score": 9,
                            "justification": "The bounded workflow and controls are explicit.",
                        },
                        {
                            "dimension": "reproducibility",
                            "score": 10,
                            "justification": (
                                "The deterministic run preserves artifacts and lineage."
                            ),
                        },
                        {
                            "dimension": "novelty",
                            "score": 5,
                            "justification": "Novelty is intentionally outside acceptance scope.",
                        },
                        {
                            "dimension": "clarity",
                            "score": 9,
                            "justification": "The synthetic evidence boundary is unambiguous.",
                        },
                    ],
                    "key_issues": [],
                    "strengths": [
                        "Explicit acceptance-only scope",
                        "Auditable execution lineage",
                    ],
                    "revision_requests": [],
                    "summary": (
                        "The record is publication-ready as a deterministic workflow "
                        "validation artifact, not as a scientific discovery."
                    ),
                }
            )
        if "conducting a literature review" in system:
            return (
                "No external scientific conclusion is synthesized in acceptance mode. "
                "The literature stage is retained only to verify orchestration and provenance."
            )
        return json.dumps({"status": "acceptance_fixture", "fixture": True})


def create_llm(provider: str, api_key: str) -> LLMProvider:
    """Factory function to create an LLM provider."""
    if provider == "anthropic":
        return AnthropicProvider(api_key)
    elif provider == "openai":
        return OpenAIProvider(api_key)
    elif provider == "minimax":
        return MiniMaxProvider(api_key)
    elif provider == "acceptance":
        return AcceptanceProvider()
    else:
        raise ValueError(f"Unknown provider: {provider}")


__all__ = [
    "LLMResponse",
    "LLMProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "MiniMaxProvider",
    "AcceptanceProvider",
    "create_llm",
]
