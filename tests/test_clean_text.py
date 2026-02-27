"""Tests for LLMProvider._clean_text() think-tag stripping."""

import pytest

from apollobot.agents import LLMProvider


class TestCleanText:
    """Tests for _clean_text static method."""

    def test_no_think_tags(self):
        """Text without think tags passes through unchanged."""
        text = "Hello, world!"
        assert LLMProvider._clean_text(text) == "Hello, world!"

    def test_single_think_block(self):
        """Single think block is removed."""
        text = "<think>reasoning here</think>The answer is 42."
        assert LLMProvider._clean_text(text) == "The answer is 42."

    def test_multiline_think_block(self):
        """Multi-line think block is removed."""
        text = "<think>\nLet me think...\nStep 1: ...\nStep 2: ...\n</think>\nResult: success"
        assert LLMProvider._clean_text(text) == "Result: success"

    def test_multiple_think_blocks(self):
        """Multiple think blocks are all removed."""
        text = "<think>first</think>Hello <think>second</think>world"
        assert LLMProvider._clean_text(text) == "Hello world"

    def test_think_block_with_json(self):
        """Think block before JSON output is removed."""
        text = '<think>I need to return JSON</think>{"key": "value"}'
        assert LLMProvider._clean_text(text) == '{"key": "value"}'

    def test_think_block_with_code(self):
        """Think block before code fence is removed."""
        text = "<think>Let me write Python</think>\n```python\nprint('hello')\n```"
        assert LLMProvider._clean_text(text) == "```python\nprint('hello')\n```"

    def test_empty_string(self):
        """Empty string returns empty string."""
        assert LLMProvider._clean_text("") == ""
