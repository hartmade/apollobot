"""Structured error codes for MCP server tool responses."""

from __future__ import annotations

from typing import Any


# Error code constants
SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
PHASE_NOT_AVAILABLE = "PHASE_NOT_AVAILABLE"
BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
CONFIG_MISSING = "CONFIG_MISSING"
SERVER_NOT_FOUND = "SERVER_NOT_FOUND"
INVALID_INPUT = "INVALID_INPUT"
INTERNAL_ERROR = "INTERNAL_ERROR"


def error_response(code: str, message: str) -> dict[str, Any]:
    """Build a structured error dict for tool responses."""
    return {
        "error": True,
        "error_code": code,
        "error_message": message,
    }
