"""
Fallback adapters for CS/ML servers: HuggingFace, Papers With Code, OpenML.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ._base import FallbackHandler, extract_query

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HuggingFace adapter
# ---------------------------------------------------------------------------

async def _huggingface_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search HuggingFace models/datasets and normalize as papers."""
    query = extract_query(params)
    limit = min(params.get("limit", 20), 100)

    # Search models endpoint
    resp = await http.get(
        f"{api_base}/models",
        params={"search": query, "limit": limit, "sort": "downloads", "direction": "-1"},
    )
    resp.raise_for_status()
    items = resp.json()

    papers = []
    for item in items[:limit]:
        model_id = item.get("modelId", item.get("id", ""))
        papers.append({
            "title": model_id,
            "abstract": item.get("pipeline_tag", ""),
            "year": 0,
            "doi": "",
            "source": "huggingface",
            "url": f"https://huggingface.co/{model_id}",
        })

    logger.info("HuggingFace fallback: %d results for query=%r", len(papers), query)
    return {"papers": papers}


# ---------------------------------------------------------------------------
# Papers With Code adapter
# ---------------------------------------------------------------------------

async def _papers_with_code_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search papers via HuggingFace daily papers API (PWC API now redirects there)."""
    query = extract_query(params)
    limit = min(params.get("limit", 20), 50)

    resp = await http.get(
        "https://huggingface.co/api/daily_papers",
        params={"search": query},
    )
    if resp.status_code != 200:
        logger.info("Papers With Code fallback: no results (status %d)", resp.status_code)
        return {"papers": []}

    items = resp.json()

    papers = []
    for item in items[:limit]:
        paper = item.get("paper", {})
        arxiv_id = paper.get("id", "")
        title = item.get("title", "")
        abstract = item.get("summary", "")

        papers.append({
            "title": title,
            "abstract": abstract,
            "year": 0,
            "doi": "",
            "arxiv_id": arxiv_id,
            "source": "papers_with_code",
            "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
        })

    logger.info("Papers With Code fallback: %d papers for query=%r", len(papers), query)
    return {"papers": papers}


# ---------------------------------------------------------------------------
# OpenML adapter
# ---------------------------------------------------------------------------

async def _openml_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search OpenML datasets and normalize results."""
    query = extract_query(params)
    limit = min(params.get("limit", 20), 100)

    resp = await http.get(
        f"{api_base}/json/data/list",
        params={"limit": limit},
        headers={"Accept": "application/json"},
    )
    # OpenML may return non-200 for empty results
    if resp.status_code != 200:
        logger.info("OpenML fallback: no results (status %d)", resp.status_code)
        return {"papers": []}

    data = resp.json()
    datasets = data.get("data", {}).get("dataset", [])

    papers = []
    for ds in datasets[:limit]:
        name = ds.get("name", "")
        if query.lower() not in name.lower():
            continue
        papers.append({
            "title": name,
            "abstract": ds.get("format", ""),
            "year": 0,
            "doi": "",
            "source": "openml",
            "url": f"https://www.openml.org/d/{ds.get('did', '')}",
        })

    logger.info("OpenML fallback: %d results for query=%r", len(papers), query)
    return {"papers": papers}


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, FallbackHandler] = {
    "huggingface": _huggingface_search,
    "papers-with-code": _papers_with_code_search,
    "openml": _openml_search,
}
