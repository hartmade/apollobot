"""
Fallback adapters for economics servers:
FRED, World Bank, BLS, SEC EDGAR.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from ._base import (
    FallbackHandler,
    extract_query,
    get_with_retry,
    post_with_retry,
    require_api_key,
    safe_int,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FRED (Federal Reserve Economic Data) — requires FRED_API_KEY
# ---------------------------------------------------------------------------

async def _fred_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search FRED economic data series or fetch observations."""
    api_key = require_api_key("FRED_API_KEY", "fred")
    query = extract_query(params)
    series_id = params.get("series_id", "")
    limit = min(params.get("limit", 20), 100)

    # If a series_id is given, fetch observations for that series
    if series_id:
        resp = await get_with_retry(
            http,
            f"{api_base}/series/observations",
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": limit,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        observations = []
        for obs in data.get("observations", []):
            observations.append({
                "date": obs.get("date", ""),
                "value": obs.get("value", ""),
            })
        return {
            "series_id": series_id,
            "observations": observations,
            "source": "fred",
        }

    # Otherwise, search for matching series
    resp = await get_with_retry(
        http,
        f"{api_base}/series/search",
        params={
            "search_text": query,
            "api_key": api_key,
            "file_type": "json",
            "limit": limit,
        },
    )
    resp.raise_for_status()
    data = resp.json()

    series = []
    for s in data.get("seriess", []):
        series.append({
            "series_id": s.get("id", ""),
            "title": s.get("title", ""),
            "frequency": s.get("frequency", ""),
            "units": s.get("units", ""),
            "source": "fred",
        })

    logger.info("FRED fallback: %d series for query=%r", len(series), query)
    return {"series": series}


# ---------------------------------------------------------------------------
# World Bank — Open Data API
# ---------------------------------------------------------------------------

async def _world_bank_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search World Bank indicators and data."""
    query = extract_query(params)
    country = params.get("country", "all")
    indicator = params.get("indicator", "")
    limit = min(params.get("limit", 50), 500)

    # If an indicator is specified, fetch data for that indicator
    if indicator:
        resp = await get_with_retry(
            http,
            f"{api_base}/country/{country}/indicator/{indicator}",
            params={"format": "json", "per_page": limit},
        )
        resp.raise_for_status()
        data = resp.json()

        # World Bank returns [metadata, data_array]
        records = data[1] if isinstance(data, list) and len(data) > 1 else []
        if records is None:
            records = []

        indicators = []
        for rec in records:
            indicators.append({
                "country": rec.get("country", {}).get("value", ""),
                "indicator": rec.get("indicator", {}).get("value", ""),
                "year": safe_int(rec.get("date")),
                "value": rec.get("value"),
                "source": "world-bank",
            })

        logger.info("World Bank fallback: %d records for indicator=%r", len(indicators), indicator)
        return {"indicators": indicators}

    # Search for indicators matching query
    resp = await get_with_retry(
        http,
        f"{api_base}/indicator",
        params={"format": "json", "per_page": limit},
    )
    resp.raise_for_status()
    data = resp.json()

    # [metadata, indicator_list]
    items = data[1] if isinstance(data, list) and len(data) > 1 else []
    if items is None:
        items = []

    indicators = []
    query_lower = query.lower()
    for item in items:
        name = item.get("name", "")
        if query_lower and query_lower not in name.lower():
            continue
        indicators.append({
            "indicator_id": item.get("id", ""),
            "name": name,
            "source": "world-bank",
        })

    logger.info("World Bank fallback: %d indicators for query=%r", len(indicators), query)
    return {"indicators": indicators}


# ---------------------------------------------------------------------------
# BLS (Bureau of Labor Statistics) — uses POST
# ---------------------------------------------------------------------------

async def _bls_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Fetch BLS time series data via the public API (POST)."""
    series_ids = params.get("series_ids", [])
    query = extract_query(params)

    # BLS API requires series IDs; if only a query is given, provide guidance
    if not series_ids:
        if query:
            return {
                "series": [],
                "note": (
                    f"BLS API requires series IDs (e.g., 'CES0000000001' for total nonfarm). "
                    f"Search https://www.bls.gov/data/ to find series IDs for: {query}"
                ),
                "source": "bls",
            }
        return {"series": [], "source": "bls"}

    start_year = params.get("start_year", "2020")
    end_year = params.get("end_year", "2025")

    payload: dict[str, Any] = {
        "seriesid": series_ids if isinstance(series_ids, list) else [series_ids],
        "startyear": str(start_year),
        "endyear": str(end_year),
    }

    # Optional: registered API key for higher rate limits
    bls_key = os.environ.get("BLS_API_KEY", "")
    if bls_key:
        payload["registrationkey"] = bls_key

    resp = await post_with_retry(
        http,
        f"{api_base}/timeseries/data/",
        json=payload,
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()

    series = []
    for result in data.get("Results", {}).get("series", []):
        observations = []
        for obs in result.get("data", []):
            observations.append({
                "year": obs.get("year", ""),
                "period": obs.get("period", ""),
                "value": obs.get("value", ""),
            })
        series.append({
            "series_id": result.get("seriesID", ""),
            "observations": observations,
            "source": "bls",
        })

    logger.info("BLS fallback: %d series returned", len(series))
    return {"series": series}


# ---------------------------------------------------------------------------
# SEC EDGAR — full-text search API
# ---------------------------------------------------------------------------

async def _sec_edgar_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search SEC EDGAR filings via the full-text search API."""
    query = extract_query(params)
    limit = min(params.get("limit", 20), 100)
    form_type = params.get("form_type", "")

    search_params: dict[str, Any] = {
        "q": query,
        "dateRange": "custom",
        "startdt": params.get("start_date", "2020-01-01"),
        "enddt": params.get("end_date", "2026-12-31"),
    }
    if form_type:
        search_params["forms"] = form_type

    # SEC EDGAR requires a descriptive User-Agent
    resp = await get_with_retry(
        http,
        f"{api_base}/search-index",
        params=search_params,
        headers={
            "User-Agent": "ApolloBot/0.1 (research@frontierscience.ai)",
            "Accept": "application/json",
        },
    )
    if resp.status_code != 200:
        logger.info("SEC EDGAR fallback: status %d", resp.status_code)
        return {"filings": []}

    data = resp.json()
    hits = data.get("hits", {}).get("hits", [])

    filings = []
    for hit in hits[:limit]:
        src = hit.get("_source", {})
        filings.append({
            "accession_number": src.get("file_num", src.get("accession_no", "")),
            "company_name": src.get("display_names", [""])[0] if src.get("display_names") else src.get("entity_name", ""),
            "form_type": src.get("form_type", src.get("file_type", "")),
            "filing_date": src.get("file_date", src.get("period_of_report", "")),
            "source": "sec-edgar",
        })

    logger.info("SEC EDGAR fallback: %d filings for query=%r", len(filings), query)
    return {"filings": filings}


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, FallbackHandler] = {
    "fred": _fred_search,
    "world-bank": _world_bank_search,
    "bls": _bls_search,
    "sec-edgar": _sec_edgar_search,
}
