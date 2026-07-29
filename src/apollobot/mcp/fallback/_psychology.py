"""
Fallback adapters for psychology and open science servers:
Open Science Framework (OSF), PsychArchives, CORE.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ._base import FallbackHandler, extract_query, get_with_retry, require_api_key

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OSF — Open Science Framework
# ---------------------------------------------------------------------------


async def _osf_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search Open Science Framework for projects and preprints."""
    query = extract_query(params)
    limit = min(params.get("limit", 20), 100)

    resp = await get_with_retry(
        http,
        f"{api_base}/search/",
        params={"q": query, "page[size]": limit},
    )
    if resp.status_code != 200:
        logger.info("OSF fallback: status %d", resp.status_code)
        return {"results": []}

    data = resp.json()
    items = data.get("search_results", data.get("data", []))

    results = []
    for item in items[:limit]:
        attrs = item.get("attributes", item)
        results.append(
            {
                "id": item.get("id", ""),
                "title": attrs.get("title", ""),
                "description": attrs.get("description", "")[:300],
                "category": attrs.get("category", attrs.get("type", "")),
                "date_created": attrs.get("date_created", attrs.get("created", "")),
                "source": "osf",
            }
        )

    logger.info("OSF fallback: %d results for query=%r", len(results), query)
    return {"results": results}


# ---------------------------------------------------------------------------
# PsychArchives — PsychOpen / ZPID Open Access Repository
# ---------------------------------------------------------------------------


async def _psychopen_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search PsychArchives for open access psychology publications."""
    query = extract_query(params)
    limit = min(params.get("limit", 20), 100)

    resp = await get_with_retry(
        http,
        f"{api_base}/search",
        params={"query": query, "size": limit},
    )
    if resp.status_code != 200:
        logger.info("PsychArchives fallback: status %d", resp.status_code)
        return {"records": []}

    data = resp.json()
    hits = data.get("hits", {}).get("hits", data.get("records", data.get("results", [])))

    records = []
    for hit in hits[:limit]:
        src = hit.get("_source", hit.get("metadata", hit))
        records.append(
            {
                "id": hit.get("_id", hit.get("id", "")),
                "title": src.get("title", ""),
                "authors": src.get("authors", src.get("creators", "")),
                "year": src.get("year", src.get("publication_date", "")),
                "doi": src.get("doi", ""),
                "source": "psychopen",
            }
        )

    logger.info("PsychArchives fallback: %d records for query=%r", len(records), query)
    return {"records": records}


# ---------------------------------------------------------------------------
# CORE — Open Access Research Papers Aggregator (requires CORE_API_KEY)
# ---------------------------------------------------------------------------


async def _core_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search CORE aggregator for open access research papers."""
    api_key = require_api_key("CORE_API_KEY", "core")
    query = extract_query(params)
    limit = min(params.get("limit", 20), 100)

    resp = await get_with_retry(
        http,
        f"{api_base}/search/works",
        params={"q": query, "limit": limit},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    if resp.status_code != 200:
        logger.info("CORE fallback: status %d", resp.status_code)
        return {"papers": []}

    data = resp.json()
    items = data.get("results", data.get("data", []))

    papers = []
    for item in items[:limit]:
        papers.append(
            {
                "id": item.get("id", ""),
                "title": item.get("title", ""),
                "authors": [a.get("name", "") for a in item.get("authors", [])]
                if isinstance(item.get("authors"), list)
                else item.get("authors", ""),
                "year": item.get("yearPublished", item.get("year", "")),
                "doi": item.get("doi", ""),
                "download_url": item.get("downloadUrl", ""),
                "source": "core",
            }
        )

    logger.info("CORE fallback: %d papers for query=%r", len(papers), query)
    return {"papers": papers}


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, FallbackHandler] = {
    "osf": _osf_search,
    "psychopen": _psychopen_search,
    "core": _core_search,
}
