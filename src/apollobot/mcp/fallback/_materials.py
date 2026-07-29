"""
Fallback adapters for materials science servers:
AFLOW, NOMAD, ICSD Web.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ._base import FallbackHandler, extract_query, get_with_retry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AFLOW — Computational Materials Database
# ---------------------------------------------------------------------------


async def _aflow_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search AFLOW for computational materials data by formula or keyword."""
    query = extract_query(params, "formula", "compound")
    limit = min(params.get("limit", 20), 100)

    # AFLOW search endpoint: search by species/formula
    resp = await get_with_retry(
        http,
        f"{api_base}/search/catalog",
        params={"species": query, "format": "json", "paging": limit},
    )
    if resp.status_code != 200:
        logger.info("AFLOW fallback: status %d for query=%r", resp.status_code, query)
        return {"materials": []}

    data = resp.json()
    items = data if isinstance(data, list) else data.get("results", data.get("data", []))

    materials = []
    for item in items[:limit]:
        materials.append(
            {
                "auid": item.get("auid", item.get("aurl", "")),
                "compound": item.get("compound", item.get("species", "")),
                "prototype": item.get("prototype", ""),
                "spacegroup": item.get("spacegroup_relax", item.get("sg", "")),
                "energy_atom": item.get("energy_atom", item.get("enthalpy_atom", None)),
                "source": "aflow",
            }
        )

    logger.info("AFLOW fallback: %d materials for query=%r", len(materials), query)
    return {"materials": materials}


# ---------------------------------------------------------------------------
# NOMAD — Novel Materials Discovery Laboratory
# ---------------------------------------------------------------------------


async def _nomad_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search NOMAD materials science archive."""
    query = extract_query(params, "formula", "material")
    limit = min(params.get("limit", 20), 100)

    resp = await get_with_retry(
        http,
        f"{api_base}/entries",
        params={
            "q": query,
            "page_size": limit,
        },
    )
    if resp.status_code != 200:
        logger.info("NOMAD fallback: status %d", resp.status_code)
        return {"entries": []}

    data = resp.json()
    items = data.get("data", [])

    entries = []
    for item in items[:limit]:
        attrs = item.get("attributes", item)
        entries.append(
            {
                "entry_id": item.get("id", attrs.get("entry_id", "")),
                "formula": attrs.get("results", {})
                .get("material", {})
                .get("chemical_formula_descriptive", ""),
                "upload_name": attrs.get("upload_name", ""),
                "program_name": attrs.get("results", {})
                .get("method", {})
                .get("simulation", {})
                .get("program_name", ""),
                "source": "nomad",
            }
        )

    logger.info("NOMAD fallback: %d entries for query=%r", len(entries), query)
    return {"entries": entries}


# ---------------------------------------------------------------------------
# ICSD Web — Inorganic Crystal Structure Database (web search)
# ---------------------------------------------------------------------------


async def _icsd_web_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search ICSD inorganic crystal structures via web API."""
    query = extract_query(params, "formula", "compound")
    limit = min(params.get("limit", 20), 100)

    resp = await get_with_retry(
        http,
        f"{api_base}/search",
        params={"query": query, "limit": limit},
    )
    if resp.status_code != 200:
        logger.info("ICSD web fallback: status %d", resp.status_code)
        return {"structures": []}

    data = resp.json()
    items = data.get("results", data.get("hits", data.get("data", [])))

    structures = []
    for item in items[:limit]:
        structures.append(
            {
                "collection_code": item.get("CollectionCode", item.get("id", "")),
                "formula": item.get("StructuredFormula", item.get("formula", "")),
                "mineral_name": item.get("MineralName", item.get("mineral", "")),
                "spacegroup": item.get("SpaceGroup", item.get("spacegroup", "")),
                "source": "icsd-web",
            }
        )

    logger.info("ICSD web fallback: %d structures for query=%r", len(structures), query)
    return {"structures": structures}


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, FallbackHandler] = {
    "aflow": _aflow_search,
    "nomad": _nomad_search,
    "icsd-web": _icsd_web_search,
}
