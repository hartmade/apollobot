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


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @staticmethod
    def _clean_text(text: str) -> str:
        """Strip <think>...</think> reasoning blocks from LLM output."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @abstractmethod
    async def complete(
        self, messages: list[dict[str, str]], system: str = ""
    ) -> LLMResponse:
        """Generate a completion from the LLM."""
        ...

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any]:
        """Extract and parse JSON from LLM output, handling common issues."""
        text = raw.strip()
        # Strip <think>...</think> blocks (e.g. from reasoning models)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # Handle markdown code blocks
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        # Fix trailing commas before } or ] (common LLM output issue)
        text = re.sub(r",\s*([}\]])", r"\1", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: find the outermost { ... } block via brace matching
            start = text.find("{")
            if start == -1:
                raise
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
                        # Fix trailing commas again on extracted block
                        block = re.sub(r",\s*([}\]])", r"\1", block)
                        return json.loads(block)
            raise

    async def complete_json(
        self, messages: list[dict[str, str]], system: str = ""
    ) -> dict[str, Any]:
        """Generate a completion and parse as JSON."""
        response = await self.complete(messages, system)
        return self._extract_json(response.text)


class AnthropicProvider(LLMProvider):
    """Claude implementation of LLMProvider."""

    # Cost per million tokens (approximate, Claude Sonnet 3.5)
    INPUT_COST_PER_M = 3.0
    OUTPUT_COST_PER_M = 15.0

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514") -> None:
        import anthropic

        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model

    async def complete(
        self, messages: list[dict[str, str]], system: str = ""
    ) -> LLMResponse:
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

        cost = (
            (input_tokens / 1_000_000) * self.INPUT_COST_PER_M
            + (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_M
        )

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
    OUTPUT_COST_PER_M = 15.0

    def __init__(self, api_key: str, model: str = "") -> None:
        import os

        import openai

        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o")

    async def complete(
        self, messages: list[dict[str, str]], system: str = ""
    ) -> LLMResponse:
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        response = await self.client.chat.completions.create(
            model=self.model,
            max_tokens=4096,
            messages=all_messages,
        )

        raw_text = response.choices[0].message.content or ""
        text = self._clean_text(raw_text)
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        cost = (
            (input_tokens / 1_000_000) * self.INPUT_COST_PER_M
            + (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_M
        )

        return LLMResponse(
            text=text,
            provider="openai",
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )


def create_llm(provider: str, api_key: str) -> LLMProvider:
    """Factory function to create an LLM provider."""
    if provider == "anthropic":
        return AnthropicProvider(api_key)
    elif provider == "openai":
        return OpenAIProvider(api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")


__all__ = [
    "LLMResponse",
    "LLMProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "create_llm",
]
