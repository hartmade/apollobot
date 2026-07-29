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
from ._geology import HANDLERS as _geo_handlers
from ._materials import HANDLERS as _mat_handlers
from ._psychology import HANDLERS as _psych_handlers
from ._mathematics import HANDLERS as _math_handlers
from ._social_science import HANDLERS as _socsci_handlers
from ._astronomy import HANDLERS as _astro_handlers
from ._climate import HANDLERS as _climate_handlers
from ._neuroscience import HANDLERS as _neuro_handlers
from ._epidemiology import HANDLERS as _epi_handlers
from ._ecology import HANDLERS as _eco_handlers

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
    _geo_handlers,
    _mat_handlers,
    _psych_handlers,
    _math_handlers,
    _socsci_handlers,
    _astro_handlers,
    _climate_handlers,
    _neuro_handlers,
    _epi_handlers,
    _eco_handlers,
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
