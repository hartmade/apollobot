"""
Shared utilities for MCP fallback adapters.

Rate limiting, retry logic, and helpers shared across all domain-specific
fallback handler modules.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Awaitable, Callable

import httpx

logger = logging.getLogger(__name__)

# Type for fallback handler functions
FallbackHandler = Callable[
    [str, dict[str, Any], httpx.AsyncClient], Awaitable[dict[str, Any]]
]

# Per-server rate limiting — prevents 429s by throttling proactively
_MIN_INTERVALS: dict[str, float] = {
    "semantic-scholar": 1.5,  # S2 public API: ~1 req/sec
    "uniprot": 1.0,
    "ensembl": 0.5,
    "chembl": 1.0,
    "materials-project": 1.0,
    "bls": 2.0,
    "sec-edgar": 0.15,
}
_last_request_time: dict[str, float] = {}
_throttle_lock = asyncio.Lock()


async def throttle(server_name: str) -> None:
    """Wait if needed to respect per-server rate limits."""
    min_interval = _MIN_INTERVALS.get(server_name)
    if min_interval is None:
        return
    async with _throttle_lock:
        last = _last_request_time.get(server_name, 0.0)
        elapsed = time.monotonic() - last
        if elapsed < min_interval:
            wait = min_interval - elapsed
            logger.debug("Throttling %s for %.2fs", server_name, wait)
            await asyncio.sleep(wait)
        _last_request_time[server_name] = time.monotonic()


async def get_with_retry(
    http: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> httpx.Response:
    """HTTP GET with exponential backoff retry on 429 rate-limit responses."""
    for attempt in range(max_retries + 1):
        resp = await http.get(url, params=params, headers=headers)
        if resp.status_code != 429 or attempt == max_retries:
            return resp
        delay = base_delay * (2 ** attempt)
        logger.info("Rate limited (429), retrying in %.1fs (attempt %d/%d)", delay, attempt + 1, max_retries)
        await asyncio.sleep(delay)
    return resp  # unreachable but satisfies type checker


async def post_with_retry(
    http: httpx.AsyncClient,
    url: str,
    *,
    json: Any | None = None,
    data: Any | None = None,
    headers: dict[str, str] | None = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> httpx.Response:
    """HTTP POST with exponential backoff retry on 429 rate-limit responses."""
    for attempt in range(max_retries + 1):
        resp = await http.post(url, json=json, content=data, headers=headers)
        if resp.status_code != 429 or attempt == max_retries:
            return resp
        delay = base_delay * (2 ** attempt)
        logger.info("Rate limited (429), retrying in %.1fs (attempt %d/%d)", delay, attempt + 1, max_retries)
        await asyncio.sleep(delay)
    return resp  # unreachable but satisfies type checker


def require_api_key(env_var: str, server_name: str) -> str:
    """Return the API key from the environment, or raise ValueError."""
    key = os.environ.get(env_var, "")
    if not key:
        raise ValueError(
            f"API key required for {server_name}: set {env_var} environment variable"
        )
    return key


def safe_int(val: Any, default: int = 0) -> int:
    """Convert a value to int, returning *default* on failure."""
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def extract_query(params: dict[str, Any], *extra_keys: str) -> str:
    """Extract a text query from LLM-generated params.

    The LLM often uses domain-specific parameter names instead of the
    canonical ``"query"`` key.  This helper checks a standard set of
    fallback keys plus any *extra_keys* supplied by the caller.

    If the first matching value is a list, the items are joined with
    ``" OR "``.
    """
    keys = ("query", "term", "search", "q", "text", *extra_keys)
    for k in keys:
        val = params.get(k)
        if val is not None:
            if isinstance(val, list):
                return " OR ".join(str(v) for v in val)
            return str(val)
    # Last resort: stringify the first non-control value in params
    for k, v in params.items():
        if k in ("limit", "retmax", "max_results", "format", "size", "retmode"):
            continue
        if isinstance(v, str) and v:
            return v
        if isinstance(v, list) and v:
            return " OR ".join(str(x) for x in v)
    return ""
