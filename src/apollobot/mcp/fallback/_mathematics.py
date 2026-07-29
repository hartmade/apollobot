"""
Fallback adapters for mathematics servers:
OEIS, zbMATH Open, Crossref (math-tagged works).
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ._base import FallbackHandler, apollobot_user_agent, extract_query, get_with_retry, safe_int

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OEIS — On-Line Encyclopedia of Integer Sequences
# ---------------------------------------------------------------------------


async def _oeis_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search OEIS for integer sequences by keyword or sequence terms."""
    query = extract_query(params, "sequence", "terms")
    limit = min(params.get("limit", 10), 50)

    resp = await get_with_retry(
        http,
        f"{api_base}/search",
        params={"q": query, "fmt": "json"},
    )
    if resp.status_code != 200:
        logger.info("OEIS fallback: status %d", resp.status_code)
        return {"sequences": []}

    data = resp.json()
    items = data.get("results", [])
    if items is None:
        items = []

    sequences = []
    for item in items[:limit]:
        sequences.append(
            {
                "id": f"A{item.get('number', '')}",
                "name": item.get("name", ""),
                "data": item.get("data", "")[:200],
                "formula": (item.get("formula", [""])[0] if item.get("formula") else ""),
                "source": "oeis",
            }
        )

    logger.info("OEIS fallback: %d sequences for query=%r", len(sequences), query)
    return {"sequences": sequences}


# ---------------------------------------------------------------------------
# zbMATH Open — Mathematical publications database
# ---------------------------------------------------------------------------


async def _zbmath_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search zbMATH Open for mathematical publications."""
    query = extract_query(params)
    limit = min(params.get("limit", 20), 100)

    resp = await get_with_retry(
        http,
        f"{api_base}/document/_search",
        params={"search_string": query, "page_size": limit},
    )
    if resp.status_code != 200:
        logger.info("zbMATH fallback: status %d", resp.status_code)
        return {"articles": []}

    data = resp.json()
    items = data.get("result", data.get("results", data.get("hits", [])))

    articles = []
    for item in items[:limit]:
        articles.append(
            {
                "zbl_id": item.get("id", item.get("zbl_id", "")),
                "title": item.get("title", ""),
                "authors": item.get("authors", []),
                "year": safe_int(item.get("year")),
                "msc_codes": item.get("msc", item.get("classification", [])),
                "doi": item.get("doi", ""),
                "source": "zbmath",
            }
        )

    logger.info("zbMATH fallback: %d articles for query=%r", len(articles), query)
    return {"articles": articles}


# ---------------------------------------------------------------------------
# Crossref (math-tagged) — Crossref works API filtered by math subjects
# ---------------------------------------------------------------------------


async def _crossref_math_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search Crossref for mathematics-tagged scholarly works."""
    query = extract_query(params)
    limit = min(params.get("limit", 20), 100)

    resp = await get_with_retry(
        http,
        api_base,
        params={
            "query": query,
            "rows": limit,
            "filter": "has-abstract:true",
            "select": "DOI,title,author,published-print,subject,abstract",
        },
        headers={"User-Agent": apollobot_user_agent()},
    )
    if resp.status_code != 200:
        logger.info("Crossref math fallback: status %d", resp.status_code)
        return {"works": []}

    data = resp.json()
    items = data.get("message", {}).get("items", [])

    works = []
    for item in items[:limit]:
        # Extract first published year
        date_parts = item.get("published-print", {}).get("date-parts", [[]])
        year = safe_int(date_parts[0][0]) if date_parts and date_parts[0] else 0

        # Flatten author list
        authors = []
        for auth in item.get("author", []):
            name = f"{auth.get('given', '')} {auth.get('family', '')}".strip()
            if name:
                authors.append(name)

        title_list = item.get("title", [])
        title = title_list[0] if title_list else ""

        works.append(
            {
                "doi": item.get("DOI", ""),
                "title": title,
                "authors": authors,
                "year": year,
                "subjects": item.get("subject", []),
                "source": "crossref-math",
            }
        )

    logger.info("Crossref math fallback: %d works for query=%r", len(works), query)
    return {"works": works}


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, FallbackHandler] = {
    "oeis": _oeis_search,
    "zbmath": _zbmath_search,
    "crossref-math": _crossref_math_search,
}
