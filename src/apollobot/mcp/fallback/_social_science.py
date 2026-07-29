"""
Fallback adapters for social science servers:
Harvard Dataverse, ICPSR, US Census Bureau.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ._base import FallbackHandler, extract_query, get_with_retry, require_api_key, safe_int

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Harvard Dataverse — Search API
# ---------------------------------------------------------------------------


async def _dataverse_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search Harvard Dataverse for social science datasets."""
    query = extract_query(params)
    limit = min(params.get("limit", 20), 100)
    dtype = params.get("type", "dataset")

    resp = await get_with_retry(
        http,
        f"{api_base}/search",
        params={"q": query, "per_page": limit, "type": dtype},
    )
    if resp.status_code != 200:
        logger.info("Dataverse fallback: status %d", resp.status_code)
        return {"datasets": []}

    data = resp.json()
    items = data.get("data", {}).get("items", [])

    datasets = []
    for item in items[:limit]:
        datasets.append(
            {
                "name": item.get("name", ""),
                "description": item.get("description", "")[:300],
                "published_at": item.get("published_at", ""),
                "doi": item.get("global_id", item.get("identifier_of_dataverse", "")),
                "citation": item.get("citation", ""),
                "source": "dataverse",
            }
        )

    logger.info("Dataverse fallback: %d datasets for query=%r", len(datasets), query)
    return {"datasets": datasets}


# ---------------------------------------------------------------------------
# ICPSR — Inter-university Consortium for Political and Social Research
# ---------------------------------------------------------------------------


async def _icpsr_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search ICPSR social science data archive."""
    query = extract_query(params)
    limit = min(params.get("limit", 20), 100)

    resp = await get_with_retry(
        http,
        f"{api_base}/search",
        params={"q": query, "rows": limit, "format": "json"},
    )
    if resp.status_code != 200:
        logger.info("ICPSR fallback: status %d", resp.status_code)
        return {"studies": []}

    data = resp.json()
    items = data.get("response", {}).get("docs", data.get("results", data.get("data", [])))

    studies = []
    for item in items[:limit]:
        studies.append(
            {
                "study_id": item.get("STUDY_NUMBER", item.get("id", "")),
                "title": item.get("TITLE", item.get("title", "")),
                "pi": item.get("PI_NAME", item.get("principal_investigator", "")),
                "year": safe_int(item.get("YEAR", item.get("year"))),
                "description": item.get("ABSTRACT", item.get("description", ""))[:300],
                "source": "icpsr",
            }
        )

    logger.info("ICPSR fallback: %d studies for query=%r", len(studies), query)
    return {"studies": studies}


# ---------------------------------------------------------------------------
# US Census Bureau — Data API (requires CENSUS_API_KEY)
# ---------------------------------------------------------------------------


async def _census_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Query US Census Bureau data API."""
    api_key = require_api_key("CENSUS_API_KEY", "census")
    query = extract_query(params)
    dataset = params.get("dataset", "acs/acs5")
    year = params.get("year", "2022")
    variables = params.get("variables", params.get("get", "NAME"))
    geo = params.get("geo", params.get("for", "state:*"))

    # If only a query is given with no specific variables, search available datasets
    if query and not params.get("variables") and not params.get("get"):
        resp = await get_with_retry(
            http,
            f"{api_base}.json",
        )
        if resp.status_code != 200:
            logger.info("Census fallback: dataset list status %d", resp.status_code)
            return {"datasets": []}

        data = resp.json()
        items = data.get("dataset", [])
        query_lower = query.lower()
        datasets = []
        for item in items:
            title = item.get("title", "")
            desc = item.get("description", "")
            if query_lower in title.lower() or query_lower in desc.lower():
                datasets.append(
                    {
                        "title": title,
                        "description": desc[:200],
                        "identifier": item.get("identifier", ""),
                        "source": "census",
                    }
                )
        logger.info("Census fallback: %d matching datasets for query=%r", len(datasets), query)
        return {"datasets": datasets}

    # Fetch specific data
    if isinstance(variables, list):
        variables = ",".join(variables)

    resp = await get_with_retry(
        http,
        f"{api_base}/{year}/{dataset}",
        params={"get": variables, "for": geo, "key": api_key},
    )
    if resp.status_code != 200:
        logger.info("Census fallback: data query status %d", resp.status_code)
        return {"data": []}

    rows = resp.json()
    if not rows or len(rows) < 2:
        return {"data": [], "source": "census"}

    headers = rows[0]
    records = []
    for row in rows[1:]:
        record = dict(zip(headers, row))
        record["source"] = "census"
        records.append(record)

    logger.info("Census fallback: %d records for dataset=%s year=%s", len(records), dataset, year)
    return {"data": records}


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, FallbackHandler] = {
    "dataverse": _dataverse_search,
    "icpsr": _icpsr_search,
    "census": _census_search,
}
