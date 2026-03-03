"""
Fallback adapters for physics servers:
Materials Project, NIST, CERN Open Data.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from ._base import FallbackHandler, extract_query, get_with_retry, require_api_key, safe_int

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Materials Project — REST API (requires MP_API_KEY)
# ---------------------------------------------------------------------------

async def _materials_project_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search Materials Project for computed material properties."""
    api_key = require_api_key("MP_API_KEY", "materials-project")
    query = extract_query(params)
    limit = min(params.get("limit", 20), 100)

    # The query could be a formula (e.g., "Fe2O3") or keywords
    headers = {"X-API-KEY": api_key}

    # Try summary endpoint with formula filter
    resp = await get_with_retry(
        http,
        f"{api_base}/v2/materials/summary/",
        params={
            "formula": query,
            "_limit": limit,
            "_fields": "material_id,formula_pretty,energy_above_hull,band_gap,density",
        },
        headers=headers,
    )
    if resp.status_code != 200:
        logger.info("Materials Project fallback: status %d for query=%r", resp.status_code, query)
        return {"materials": []}

    data = resp.json()
    items = data.get("data", [])

    materials = []
    for item in items[:limit]:
        materials.append({
            "material_id": item.get("material_id", ""),
            "formula": item.get("formula_pretty", ""),
            "energy_above_hull": item.get("energy_above_hull"),
            "band_gap": item.get("band_gap"),
            "density": item.get("density"),
            "source": "materials-project",
        })

    logger.info("Materials Project fallback: %d materials for query=%r", len(materials), query)
    return {"materials": materials}


# ---------------------------------------------------------------------------
# NIST — Fundamental Physical Constants (text parsing)
# ---------------------------------------------------------------------------

# Pre-parsed subset of NIST fundamental constants (no JSON API available).
# The CGI endpoint returns HTML/text; we parse the structured text output.

async def _nist_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search NIST fundamental physical constants."""
    query = extract_query(params).lower()

    # Use the ASCII output of the constants CGI
    resp = await http.get(
        f"{api_base}/Value",
        params={"search_for": query},
    )
    if resp.status_code != 200:
        return {
            "constants": [],
            "note": "NIST CGI returned non-200; only fundamental constants search is supported.",
        }

    text = resp.text
    constants = []

    # NIST ASCII output has lines like:
    #   Quantity                       Value          Uncertainty   Unit
    # We parse line-by-line looking for data rows
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if not line or line.startswith("*") or line.startswith("-"):
            continue

        # Try to match a constant line with value and uncertainty
        # Format varies; look for lines with numeric values
        # Simple heuristic: lines containing digits and possibly "e" or "E"
        parts = re.split(r"\s{2,}", line)
        if len(parts) >= 2:
            name_part = parts[0].strip()
            value_part = parts[1].strip() if len(parts) > 1 else ""
            uncertainty_part = parts[2].strip() if len(parts) > 2 else ""
            unit_part = parts[3].strip() if len(parts) > 3 else ""

            # Basic validation: value should contain a digit
            if name_part and value_part and any(c.isdigit() for c in value_part):
                constants.append({
                    "name": name_part,
                    "value": value_part,
                    "uncertainty": uncertainty_part,
                    "unit": unit_part,
                    "source": "nist",
                })

    if not constants:
        return {
            "constants": [],
            "note": (
                "No constants matched. NIST fallback currently supports fundamental "
                "physical constants only. Chemistry Webbook queries are not yet supported."
            ),
        }

    logger.info("NIST fallback: %d constants for query=%r", len(constants), query)
    return {"constants": constants}


# ---------------------------------------------------------------------------
# CERN Open Data — REST API
# ---------------------------------------------------------------------------

async def _cern_opendata_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search CERN Open Data Portal for particle physics datasets."""
    query = extract_query(params)
    limit = min(params.get("limit", 20), 100)

    resp = await get_with_retry(
        http,
        f"{api_base}/records/",
        params={
            "q": query,
            "size": limit,
            "type": "Dataset",
        },
    )
    if resp.status_code != 200:
        logger.info("CERN Open Data fallback: status %d", resp.status_code)
        return {"datasets": []}

    data = resp.json()
    hits = data.get("hits", {}).get("hits", [])

    datasets = []
    for hit in hits[:limit]:
        metadata = hit.get("metadata", {})
        # Extract collision info from metadata
        collision_info = metadata.get("collision_information", {})

        datasets.append({
            "recid": safe_int(hit.get("id", metadata.get("recid", 0))),
            "title": metadata.get("title", ""),
            "experiment": metadata.get("experiment", ""),
            "collision_type": collision_info.get("type", ""),
            "collision_energy": collision_info.get("energy", ""),
            "source": "cern-opendata",
        })

    logger.info("CERN Open Data fallback: %d datasets for query=%r", len(datasets), query)
    return {"datasets": datasets}


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, FallbackHandler] = {
    "materials-project": _materials_project_search,
    "nist": _nist_search,
    "cern-opendata": _cern_opendata_search,
}
