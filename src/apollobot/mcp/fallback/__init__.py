"""
Direct API fallback adapters for MCP servers.

When the MCP proxy is unreachable, these adapters translate
MCP search queries into native API calls for each domain.

Usage (unchanged from the single-file version)::

    from apollobot.mcp.fallback import fallback_query
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ._base import FallbackHandler, throttle
from ._literature import HANDLERS as _lit_handlers
from ._cs_ml import HANDLERS as _cs_handlers
from ._bioinformatics import HANDLERS as _bio_handlers
from ._comp_chem import HANDLERS as _chem_handlers
from ._physics import HANDLERS as _phys_handlers
from ._economics import HANDLERS as _econ_handlers

logger = logging.getLogger(__name__)

# Merge all domain handler registries into one dispatcher map
_FALLBACK_HANDLERS: dict[str, FallbackHandler] = {}
for _handlers in (
    _lit_handlers,
    _cs_handlers,
    _bio_handlers,
    _chem_handlers,
    _phys_handlers,
    _econ_handlers,
):
    _FALLBACK_HANDLERS.update(_handlers)


async def fallback_query(
    server_name: str,
    api_base: str,
    params: dict[str, Any],
    http_client: httpx.AsyncClient,
) -> dict[str, Any]:
    """
    Dispatch a fallback query to the appropriate adapter.

    Raises ValueError if no handler is registered for the server name.
    """
    handler = _FALLBACK_HANDLERS.get(server_name)
    if handler is None:
        raise ValueError(
            f"No fallback handler for server '{server_name}'. "
            f"Available: {sorted(_FALLBACK_HANDLERS.keys())}"
        )
    logger.info("Falling back to direct API for server=%s", server_name)
    await throttle(server_name)
    return await handler(api_base, params, http_client)


__all__ = ["fallback_query"]
