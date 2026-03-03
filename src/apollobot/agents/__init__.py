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
    def _fix_json(text: str) -> str:
        """Apply common JSON fixes for non-standard LLM output."""
        # Strip /* block comments */
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
        # Strip // line comments only at start of lines (safe — won't hit URLs)
        text = re.sub(r"^\s*//[^\n]*\n?", "", text, flags=re.MULTILINE)
        # Strip // comments after JSON structural tokens (not inside strings)
        text = re.sub(r'(?<=[,\]\}\d])\s*//[^\n]*', "", text)
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
        last_error = None
        for attempt in range(1 + retries):
            if attempt == 0:
                response = await self.complete(messages, system)
            else:
                # Retry with error feedback appended
                retry_messages = messages + [
                    {"role": "assistant", "content": response.text},
                    {"role": "user", "content": (
                        f"Your response could not be parsed as JSON: {last_error}\n"
                        "Please return ONLY valid JSON with no comments, no trailing "
                        "commas, and double-quoted keys. No markdown fences."
                    )},
                ]
                response = await self.complete(retry_messages, system)
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

    def __init__(self, api_key: str, model: str = "", base_url: str = "") -> None:
        import os

        import openai

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = openai.AsyncOpenAI(**kwargs)
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
            max_tokens=8192,
            messages=all_messages,
        )

        if not response.choices:
            raise RuntimeError(f"LLM returned no choices (model={self.model})")
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

    async def complete(
        self, messages: list[dict[str, str]], system: str = ""
    ) -> LLMResponse:
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

        cost = (
            (input_tokens / 1_000_000) * self.INPUT_COST_PER_M
            + (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_M
        )

        return LLMResponse(
            text=text,
            provider="minimax",
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
    elif provider == "minimax":
        return MiniMaxProvider(api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")


__all__ = [
    "LLMResponse",
    "LLMProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "MiniMaxProvider",
    "create_llm",
]
