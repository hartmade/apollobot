"""
Fallback adapters for ecology servers:
GBIF, IUCN Red List, OBIS.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ._base import (
    FallbackHandler,
    extract_query,
    get_with_retry,
    require_api_key,
    safe_int,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GBIF — Global Biodiversity Information Facility
# ---------------------------------------------------------------------------


async def _gbif_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search GBIF for species occurrence records."""
    query = extract_query(params, "species", "taxon", "scientificName")
    limit = min(params.get("limit", 20), 300)

    # First resolve the species name to a taxon key
    name_resp = await get_with_retry(
        http,
        f"{api_base}/species/match",
        params={"name": query, "verbose": "true"},
    )
    if name_resp.status_code != 200:
        logger.info(
            "GBIF fallback: name match status %d for query=%r", name_resp.status_code, query
        )
        return {"occurrences": []}

    match = name_resp.json()
    taxon_key = match.get("usageKey")

    if taxon_key:
        # Fetch occurrences for the matched taxon
        occ_resp = await get_with_retry(
            http,
            f"{api_base}/occurrence/search",
            params={"taxonKey": taxon_key, "limit": limit},
        )
        occ_resp.raise_for_status()
        occ_data = occ_resp.json()
        results = occ_data.get("results", [])
    else:
        # Fall back to free-text occurrence search
        occ_resp = await get_with_retry(
            http,
            f"{api_base}/occurrence/search",
            params={"q": query, "limit": limit},
        )
        occ_resp.raise_for_status()
        occ_data = occ_resp.json()
        results = occ_data.get("results", [])

    occurrences = []
    for rec in results[:limit]:
        occurrences.append(
            {
                "occurrence_id": rec.get("key", ""),
                "scientific_name": rec.get("scientificName", ""),
                "country": rec.get("country", ""),
                "latitude": rec.get("decimalLatitude"),
                "longitude": rec.get("decimalLongitude"),
                "year": safe_int(rec.get("year")),
                "basis_of_record": rec.get("basisOfRecord", ""),
                "source": "gbif",
            }
        )

    logger.info("GBIF fallback: %d occurrences for query=%r", len(occurrences), query)
    return {"occurrences": occurrences}


# ---------------------------------------------------------------------------
# IUCN Red List — species conservation status (requires IUCN_API_KEY)
# ---------------------------------------------------------------------------


async def _iucn_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search the IUCN Red List for species conservation assessments."""
    api_key = require_api_key("IUCN_API_KEY", "iucn")
    query = extract_query(params, "species", "taxon", "name")

    resp = await get_with_retry(
        http,
        f"{api_base}/species/{query}",
        params={"token": api_key},
    )
    if resp.status_code != 200:
        # Try the narrative endpoint for common names
        resp = await get_with_retry(
            http,
            f"{api_base}/species/common_names/{query}",
            params={"token": api_key},
        )

    if resp.status_code != 200:
        logger.info("IUCN fallback: status %d for query=%r", resp.status_code, query)
        return {"species": []}

    data = resp.json()
    results = data.get("result", [])

    species = []
    for rec in results:
        species.append(
            {
                "taxon_id": safe_int(rec.get("taxonid")),
                "scientific_name": rec.get("scientific_name", rec.get("taxonname", "")),
                "category": rec.get("category", ""),
                "population_trend": rec.get("population_trend", ""),
                "main_common_name": rec.get("main_common_name", rec.get("taxonname", "")),
                "source": "iucn",
            }
        )

    logger.info("IUCN fallback: %d species for query=%r", len(species), query)
    return {"species": species}


# ---------------------------------------------------------------------------
# OBIS — Ocean Biodiversity Information System
# ---------------------------------------------------------------------------


async def _obis_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search OBIS for marine species occurrence records."""
    query = extract_query(params, "species", "taxon", "scientificname")
    limit = min(params.get("limit", 20), 200)

    resp = await get_with_retry(
        http,
        f"{api_base}/occurrence",
        params={"scientificname": query, "size": limit},
    )
    if resp.status_code != 200:
        logger.info("OBIS fallback: status %d for query=%r", resp.status_code, query)
        return {"occurrences": []}

    data = resp.json()
    results = data.get("results", [])

    occurrences = []
    for rec in results[:limit]:
        occurrences.append(
            {
                "occurrence_id": rec.get("id", rec.get("occurrenceID", "")),
                "scientific_name": rec.get("scientificName", rec.get("species", "")),
                "latitude": rec.get("decimalLatitude"),
                "longitude": rec.get("decimalLongitude"),
                "depth": rec.get("depth"),
                "dataset_name": rec.get("dataset_id", rec.get("datasetName", "")),
                "year": safe_int(rec.get("date_year", rec.get("year"))),
                "source": "obis",
            }
        )

    logger.info("OBIS fallback: %d occurrences for query=%r", len(occurrences), query)
    return {"occurrences": occurrences}


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, FallbackHandler] = {
    "gbif": _gbif_search,
    "iucn": _iucn_search,
    "obis": _obis_search,
}
