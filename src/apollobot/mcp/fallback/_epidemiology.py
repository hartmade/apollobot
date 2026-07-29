"""
Fallback adapters for epidemiology servers:
Our World in Data, WHO Global Health Observatory, CDC WONDER.
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
# Our World in Data — catalog and GitHub-hosted CSV data
# ---------------------------------------------------------------------------


async def _owid_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search Our World in Data for indicators and datasets."""
    query = extract_query(params, "indicator", "variable", "topic")
    limit = min(params.get("limit", 20), 100)

    resp = await get_with_retry(
        http,
        f"{api_base}/search",
        params={"q": query, "limit": limit},
    )
    if resp.status_code != 200:
        # Fall back to the variables endpoint
        resp = await get_with_retry(
            http,
            f"{api_base}/variables",
            params={"search": query, "limit": limit},
        )

    if resp.status_code != 200:
        logger.info("OWID fallback: status %d for query=%r", resp.status_code, query)
        return {"indicators": []}

    data = resp.json()
    items = data.get("results", data.get("variables", []))
    if isinstance(items, dict):
        items = list(items.values())

    indicators = []
    for item in items[:limit]:
        indicators.append(
            {
                "variable_id": safe_int(item.get("id")),
                "name": item.get("name", item.get("title", "")),
                "dataset": item.get("dataset", item.get("datasetName", "")),
                "description": (item.get("description", "") or "")[:200],
                "source": "owid",
            }
        )

    logger.info("OWID fallback: %d indicators for query=%r", len(indicators), query)
    return {"indicators": indicators}


# ---------------------------------------------------------------------------
# WHO Global Health Observatory (GHO) — OData API
# ---------------------------------------------------------------------------


async def _gho_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search WHO GHO for health indicators and data values."""
    query = extract_query(params, "indicator", "code")
    indicator_code = params.get("indicator_code", "")
    limit = min(params.get("limit", 25), 200)

    # If an indicator code is given, fetch values for it
    if indicator_code:
        resp = await get_with_retry(
            http,
            f"{api_base}/{indicator_code}",
            params={"$top": limit, "$format": "json"},
        )
        if resp.status_code != 200:
            return {"values": [], "source": "gho"}

        data = resp.json()
        records = data.get("value", [])
        values = []
        for rec in records[:limit]:
            values.append(
                {
                    "country": rec.get("SpatialDim", ""),
                    "year": safe_int(rec.get("TimeDim")),
                    "value": rec.get("NumericValue", rec.get("Value", "")),
                    "indicator": indicator_code,
                    "source": "gho",
                }
            )
        logger.info("GHO fallback: %d values for indicator=%r", len(values), indicator_code)
        return {"values": values}

    # Search for indicators matching query
    resp = await get_with_retry(
        http,
        f"{api_base}/Indicator",
        params={"$format": "json"},
    )
    if resp.status_code != 200:
        logger.info("GHO fallback: status %d", resp.status_code)
        return {"indicators": []}

    data = resp.json()
    items = data.get("value", [])

    indicators = []
    query_lower = query.lower()
    for item in items:
        name = item.get("IndicatorName", "")
        if query_lower and query_lower not in name.lower():
            continue
        indicators.append(
            {
                "indicator_code": item.get("IndicatorCode", ""),
                "name": name,
                "language": item.get("Language", ""),
                "source": "gho",
            }
        )
        if len(indicators) >= limit:
            break

    logger.info("GHO fallback: %d indicators for query=%r", len(indicators), query)
    return {"indicators": indicators}


# ---------------------------------------------------------------------------
# CDC WONDER — data request API
# ---------------------------------------------------------------------------


async def _cdc_wonder_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Query CDC WONDER for public health statistics.

    CDC WONDER's API is XML/form-based and complex.  This fallback
    provides a simplified search of available databases and returns
    guidance on database IDs for more targeted queries.
    """
    query = extract_query(params, "database", "cause", "topic")

    # CDC WONDER does not have a simple REST search; provide a curated
    # listing of commonly used databases with their IDs.
    databases = [
        {
            "database_id": "D76",
            "name": "Underlying Cause of Death, 1999-2020",
            "category": "mortality",
        },
        {
            "database_id": "D77",
            "name": "Multiple Cause of Death, 1999-2020",
            "category": "mortality",
        },
        {
            "database_id": "D176",
            "name": "Underlying Cause of Death, 2018-last month",
            "category": "mortality",
        },
        {"database_id": "D149", "name": "Natality, 2016-2022", "category": "natality"},
        {"database_id": "D66", "name": "Cancer Statistics, 1999-2020", "category": "cancer"},
        {"database_id": "D8", "name": "AIDS Public Use Data", "category": "infectious-disease"},
        {"database_id": "D159", "name": "COVID-19 Provisional Deaths", "category": "covid"},
        {"database_id": "D16", "name": "Tuberculosis Data", "category": "infectious-disease"},
    ]

    if query:
        query_lower = query.lower()
        databases = [
            db
            for db in databases
            if query_lower in db["name"].lower() or query_lower in db["category"]
        ]

    for db in databases:
        db["source"] = "cdc-wonder"

    logger.info("CDC WONDER fallback: %d databases for query=%r", len(databases), query)
    return {"databases": databases}


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, FallbackHandler] = {
    "owid": _owid_search,
    "gho": _gho_search,
    "cdc-wonder": _cdc_wonder_search,
}
