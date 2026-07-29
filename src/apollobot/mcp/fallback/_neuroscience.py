"""
Fallback adapters for neuroscience servers:
Allen Brain Atlas, OpenNeuro, NeuroVault.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ._base import (
    FallbackHandler,
    extract_query,
    get_with_retry,
    safe_int,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Allen Brain Atlas — Allen Institute API
# ---------------------------------------------------------------------------


async def _allen_brain_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search the Allen Brain Atlas for gene expression and structure data."""
    query = extract_query(params, "gene", "structure", "acronym")
    limit = min(params.get("limit", 20), 100)

    # Search genes in the mouse brain atlas by default
    resp = await get_with_retry(
        http,
        f"{api_base}/data/Gene/query.json",
        params={
            "criteria": f"products[abbreviation$eq'Mouse'][name$li'*{query}*']",
            "num_rows": limit,
            "include": "organism",
        },
    )
    if resp.status_code != 200:
        logger.info("Allen Brain fallback: status %d for query=%r", resp.status_code, query)
        return {"genes": []}

    data = resp.json()
    rows = data.get("msg", [])

    genes = []
    for row in rows[:limit]:
        genes.append(
            {
                "gene_id": safe_int(row.get("id")),
                "acronym": row.get("acronym", ""),
                "name": row.get("name", ""),
                "chromosome": row.get("chromosome_id", ""),
                "organism": row.get("organism", {}).get("name", "")
                if isinstance(row.get("organism"), dict)
                else "",
                "source": "allen-brain",
            }
        )

    logger.info("Allen Brain fallback: %d genes for query=%r", len(genes), query)
    return {"genes": genes}


# ---------------------------------------------------------------------------
# OpenNeuro — datasets API
# ---------------------------------------------------------------------------


async def _openneuro_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search OpenNeuro for neuroimaging datasets."""
    query = extract_query(params, "modality", "task")
    limit = min(params.get("limit", 20), 50)

    # OpenNeuro uses a GraphQL API
    graphql_query = {
        "query": """
            query ($query: String, $first: Int) {
                datasets(filterBy: {query: $query}, first: $first, orderBy: {created: descending}) {
                    edges {
                        node {
                            id
                            name
                            publishDate
                            analytics { downloads }
                        }
                    }
                }
            }
        """,
        "variables": {"query": query, "first": limit},
    }

    resp = await get_with_retry(
        http,
        f"{api_base}/graphql",
        params={"query": graphql_query["query"], "variables": str(graphql_query["variables"])},
    )
    # Try POST if GET fails (GraphQL endpoints often prefer POST)
    if resp.status_code != 200:
        from ._base import post_with_retry

        resp = await post_with_retry(
            http,
            f"{api_base}/graphql",
            json=graphql_query,
            headers={"Content-Type": "application/json"},
        )

    if resp.status_code != 200:
        logger.info("OpenNeuro fallback: status %d for query=%r", resp.status_code, query)
        return {"datasets": []}

    data = resp.json()
    edges = data.get("data", {}).get("datasets", {}).get("edges", [])

    datasets = []
    for edge in edges[:limit]:
        node = edge.get("node", {})
        analytics = node.get("analytics", {}) or {}
        datasets.append(
            {
                "dataset_id": node.get("id", ""),
                "name": node.get("name", ""),
                "publish_date": node.get("publishDate", ""),
                "downloads": safe_int(analytics.get("downloads")),
                "source": "openneuro",
            }
        )

    logger.info("OpenNeuro fallback: %d datasets for query=%r", len(datasets), query)
    return {"datasets": datasets}


# ---------------------------------------------------------------------------
# NeuroVault — brain imaging results
# ---------------------------------------------------------------------------


async def _neurovault_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search NeuroVault for brain imaging collections and statistical maps."""
    query = extract_query(params, "task", "contrast", "map_type")
    limit = min(params.get("limit", 20), 100)

    resp = await get_with_retry(
        http,
        f"{api_base}/collections/",
        params={"name": query, "limit": limit, "format": "json"},
    )
    if resp.status_code != 200:
        logger.info("NeuroVault fallback: status %d for query=%r", resp.status_code, query)
        return {"collections": []}

    data = resp.json()
    items = data.get("results", data if isinstance(data, list) else [])

    collections = []
    for item in items[:limit]:
        collections.append(
            {
                "collection_id": safe_int(item.get("id")),
                "name": item.get("name", ""),
                "doi": item.get("DOI", item.get("doi", "")),
                "owner_name": item.get("owner_name", item.get("owner", "")),
                "number_of_images": safe_int(item.get("number_of_images")),
                "source": "neurovault",
            }
        )

    logger.info("NeuroVault fallback: %d collections for query=%r", len(collections), query)
    return {"collections": collections}


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, FallbackHandler] = {
    "allen-brain": _allen_brain_search,
    "openneuro": _openneuro_search,
    "neurovault": _neurovault_search,
}
