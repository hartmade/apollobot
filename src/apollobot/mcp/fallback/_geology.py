"""
Fallback adapters for geology servers:
USGS Earthquake Catalog, IRIS Seismology, PANGAEA Earth Science Data.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ._base import FallbackHandler, extract_query, get_with_retry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# USGS Earthquake Catalog — FDSN Event Web Service
# ---------------------------------------------------------------------------


async def _usgs_earthquake_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search USGS earthquake catalog for recent seismic events."""
    query = extract_query(params)
    limit = min(params.get("limit", 20), 200)
    min_magnitude = params.get("min_magnitude", params.get("minmagnitude", 4.0))
    start_time = params.get("start_time", params.get("starttime", ""))
    end_time = params.get("end_time", params.get("endtime", ""))

    search_params: dict[str, Any] = {
        "format": "geojson",
        "limit": limit,
        "minmagnitude": min_magnitude,
        "orderby": "time",
    }
    if start_time:
        search_params["starttime"] = start_time
    if end_time:
        search_params["endtime"] = end_time
    if query:
        # Use the query as a geographic region keyword if it looks like a place
        search_params["producttype"] = "moment-tensor"

    resp = await get_with_retry(http, f"{api_base}/query", params=search_params)
    if resp.status_code != 200:
        logger.info("USGS earthquake fallback: status %d", resp.status_code)
        return {"earthquakes": []}

    data = resp.json()
    features = data.get("features", [])

    earthquakes = []
    for feat in features[:limit]:
        props = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [])
        earthquakes.append(
            {
                "title": props.get("title", ""),
                "magnitude": props.get("mag"),
                "place": props.get("place", ""),
                "time": props.get("time"),
                "depth": coords[2] if len(coords) > 2 else None,
                "longitude": coords[0] if len(coords) > 0 else None,
                "latitude": coords[1] if len(coords) > 1 else None,
                "url": props.get("url", ""),
                "source": "usgs-earthquake",
            }
        )

    logger.info("USGS earthquake fallback: %d events for query=%r", len(earthquakes), query)
    return {"earthquakes": earthquakes}


# ---------------------------------------------------------------------------
# IRIS — FDSN Station/Event Web Service
# ---------------------------------------------------------------------------


async def _iris_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search IRIS seismology station and event data."""
    query = extract_query(params)
    network = params.get("network", "*")
    station = params.get("station", "*")
    level = params.get("level", "station")
    limit = min(params.get("limit", 20), 100)

    # Station search via FDSN station service
    search_params: dict[str, Any] = {
        "format": "text",
        "network": network,
        "station": station,
        "level": level,
    }
    if query:
        # Use query as station name filter if provided
        search_params["station"] = f"*{query}*" if station == "*" else station

    resp = await get_with_retry(http, f"{api_base}/station/1/query", params=search_params)
    if resp.status_code != 200:
        logger.info("IRIS fallback: status %d", resp.status_code)
        return {"stations": []}

    lines = resp.text.strip().split("\n")
    stations = []
    for line in lines[1 : limit + 1]:  # skip header line
        parts = line.split("|")
        if len(parts) < 6:
            continue
        stations.append(
            {
                "network": parts[0].strip(),
                "station": parts[1].strip(),
                "latitude": parts[2].strip(),
                "longitude": parts[3].strip(),
                "elevation": parts[4].strip(),
                "site_name": parts[5].strip() if len(parts) > 5 else "",
                "source": "iris",
            }
        )

    logger.info("IRIS fallback: %d stations for query=%r", len(stations), query)
    return {"stations": stations}


# ---------------------------------------------------------------------------
# PANGAEA — Earth Science Data Portal (Elasticsearch)
# ---------------------------------------------------------------------------


async def _pangaea_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search PANGAEA earth science datasets."""
    query = extract_query(params)
    limit = min(params.get("limit", 20), 100)

    resp = await get_with_retry(
        http,
        f"{api_base}/search",
        params={"q": query, "count": limit},
    )
    if resp.status_code != 200:
        logger.info("PANGAEA fallback: status %d", resp.status_code)
        return {"datasets": []}

    data = resp.json()
    hits = data.get("hits", {}).get("hits", data.get("results", []))

    datasets = []
    for hit in hits[:limit]:
        src = hit.get("_source", hit)
        datasets.append(
            {
                "doi": src.get("URI", src.get("doi", "")),
                "title": src.get("citation", src.get("title", "")),
                "year": src.get("year", ""),
                "authors": src.get("authors", ""),
                "source": "pangaea",
            }
        )

    logger.info("PANGAEA fallback: %d datasets for query=%r", len(datasets), query)
    return {"datasets": datasets}


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, FallbackHandler] = {
    "usgs-earthquake": _usgs_earthquake_search,
    "iris": _iris_search,
    "pangaea": _pangaea_search,
}
