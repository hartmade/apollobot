"""
Fallback adapters for climate servers:
Copernicus CDS, NOAA GISS, NOAA NCEI.
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
# Copernicus Climate Data Store
# ---------------------------------------------------------------------------


async def _copernicus_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search the Copernicus Climate Data Store for available datasets."""
    query = extract_query(params, "dataset", "variable")
    limit = min(params.get("limit", 20), 100)

    resp = await get_with_retry(
        http,
        f"{api_base}/resources",
        params={"limit": limit},
    )
    if resp.status_code != 200:
        logger.info("Copernicus fallback: status %d", resp.status_code)
        return {"datasets": []}

    data = resp.json()
    items = data if isinstance(data, list) else data.get("resources", data.get("results", []))

    datasets = []
    query_lower = query.lower()
    for item in items:
        title = item.get("title", item.get("name", ""))
        description = item.get("abstract", item.get("description", ""))
        if (
            query_lower
            and query_lower not in title.lower()
            and query_lower not in description.lower()
        ):
            continue
        datasets.append(
            {
                "dataset_id": item.get("id", item.get("name", "")),
                "title": title,
                "description": description[:200],
                "source": "copernicus",
            }
        )

    logger.info("Copernicus fallback: %d datasets for query=%r", len(datasets), query)
    return {"datasets": datasets}


# ---------------------------------------------------------------------------
# NOAA GISS — NASA Goddard Institute surface temperature data
# ---------------------------------------------------------------------------


async def _noaa_giss_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Fetch GISTEMP surface temperature anomaly data from NASA GISS."""
    query = extract_query(params, "dataset", "variable")
    # GISS provides CSV files for global temperature anomalies
    resp = await get_with_retry(
        http,
        f"{api_base}/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
    )
    if resp.status_code != 200:
        logger.info("NOAA GISS fallback: status %d", resp.status_code)
        return {"records": []}

    lines = resp.text.strip().split("\n")
    records = []
    # Skip header lines (first two lines are headers)
    for line in lines[2:]:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        year = parts[0].strip()
        if not year.isdigit():
            continue
        annual = parts[13].strip() if len(parts) > 13 else ""
        records.append(
            {
                "year": safe_int(year),
                "jan": parts[1].strip() if len(parts) > 1 else "",
                "annual_mean": annual,
                "source": "noaa-giss",
            }
        )

    logger.info("NOAA GISS fallback: %d temperature records", len(records))
    return {"records": records}


# ---------------------------------------------------------------------------
# NOAA NCEI — National Centers for Environmental Information
# (requires NOAA_API_KEY)
# ---------------------------------------------------------------------------


async def _noaa_ncei_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search NOAA NCEI for climate datasets and station data."""
    api_key = require_api_key("NOAA_API_KEY", "noaa-ncei")
    query = extract_query(params, "dataset", "datatype", "location")
    limit = min(params.get("limit", 25), 100)
    dataset_id = params.get("dataset_id", "")

    headers = {"token": api_key}

    # If a specific dataset is requested, fetch data types for it
    if dataset_id:
        resp = await get_with_retry(
            http,
            f"{api_base}/datatypes",
            params={"datasetid": dataset_id, "limit": limit},
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        datatypes = []
        for dt in data.get("results", []):
            datatypes.append(
                {
                    "datatype_id": dt.get("id", ""),
                    "name": dt.get("name", ""),
                    "min_date": dt.get("mindate", ""),
                    "max_date": dt.get("maxdate", ""),
                    "source": "noaa-ncei",
                }
            )
        logger.info("NOAA NCEI fallback: %d datatypes for dataset=%r", len(datatypes), dataset_id)
        return {"datatypes": datatypes}

    # Search available datasets
    resp = await get_with_retry(
        http,
        f"{api_base}/datasets",
        params={"limit": limit},
        headers=headers,
    )
    resp.raise_for_status()
    data = resp.json()

    datasets = []
    query_lower = query.lower()
    for ds in data.get("results", []):
        name = ds.get("name", "")
        if query_lower and query_lower not in name.lower():
            continue
        datasets.append(
            {
                "dataset_id": ds.get("id", ""),
                "name": name,
                "min_date": ds.get("mindate", ""),
                "max_date": ds.get("maxdate", ""),
                "source": "noaa-ncei",
            }
        )

    logger.info("NOAA NCEI fallback: %d datasets for query=%r", len(datasets), query)
    return {"datasets": datasets}


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, FallbackHandler] = {
    "copernicus": _copernicus_search,
    "noaa-giss": _noaa_giss_search,
    "noaa-ncei": _noaa_ncei_search,
}
