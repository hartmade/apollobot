"""
Fallback adapters for astronomy servers:
SIMBAD, MAST, VizieR, NASA ADS.
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
# SIMBAD — CDS astronomical database (script interface)
# ---------------------------------------------------------------------------


async def _simbad_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search SIMBAD for astronomical objects by identifier or coordinates."""
    query = extract_query(params, "object", "identifier", "name")
    if not query:
        return {"objects": []}

    # Use the TAP endpoint for structured queries
    adql = (
        f"SELECT TOP 20 main_id, ra, dec, otype_txt, sp_type "
        f"FROM basic JOIN ident ON oidref = oid "
        f"WHERE id = '{query}' OR main_id LIKE '%{query}%'"
    )
    resp = await get_with_retry(
        http,
        "https://simbad.cds.unistra.fr/simbad/sim-tap/sync",
        params={
            "request": "doQuery",
            "lang": "adql",
            "format": "json",
            "query": adql,
        },
    )
    if resp.status_code != 200:
        logger.info("SIMBAD fallback: status %d for query=%r", resp.status_code, query)
        return {"objects": []}

    data = resp.json()
    rows = data.get("data", [])
    columns = [c.get("name", "") for c in data.get("metadata", [])]

    objects = []
    for row in rows:
        rec = dict(zip(columns, row)) if columns else {}
        objects.append(
            {
                "main_id": rec.get("main_id", ""),
                "ra": rec.get("ra"),
                "dec": rec.get("dec"),
                "object_type": rec.get("otype_txt", ""),
                "spectral_type": rec.get("sp_type", ""),
                "source": "simbad",
            }
        )

    logger.info("SIMBAD fallback: %d objects for query=%r", len(objects), query)
    return {"objects": objects}


# ---------------------------------------------------------------------------
# MAST — Mikulski Archive for Space Telescopes
# ---------------------------------------------------------------------------


async def _mast_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search MAST archive for astronomical observations."""
    query = extract_query(params, "target", "object")
    limit = min(params.get("limit", 20), 100)

    resp = await get_with_retry(
        http,
        f"{api_base}/invoke",
        params={
            "request": '{"service":"Mast.Name.Lookup","params":{"input":"'
            + query
            + '","format":"json"}}',
        },
    )
    if resp.status_code != 200:
        logger.info("MAST fallback: name lookup status %d", resp.status_code)
        return {"observations": []}

    data = resp.json()
    resolved = data.get("resolvedCoordinate", [])
    if not resolved:
        logger.info("MAST fallback: could not resolve target=%r", query)
        return {"observations": []}

    ra = resolved[0].get("ra", 0)
    dec = resolved[0].get("decl", 0)

    # Cone search around resolved coordinates
    cone_resp = await get_with_retry(
        http,
        f"{api_base}/invoke",
        params={
            "request": (
                '{"service":"Mast.Caom.Cone",'
                f'"params":{{"ra":{ra},"dec":{dec},"radius":0.01}},'
                f'"pagesize":{limit},"page":1}}'
            ),
        },
    )
    if cone_resp.status_code != 200:
        return {"observations": []}

    cone_data = cone_resp.json()
    rows = cone_data.get("data", [])

    observations = []
    for row in rows[:limit]:
        observations.append(
            {
                "obs_id": row.get("obsid", ""),
                "target_name": row.get("target_name", query),
                "instrument": row.get("instrument_name", ""),
                "project": row.get("project", ""),
                "wavelength_region": row.get("wavelength_region", ""),
                "source": "mast",
            }
        )

    logger.info("MAST fallback: %d observations for query=%r", len(observations), query)
    return {"observations": observations}


# ---------------------------------------------------------------------------
# VizieR — catalog service
# ---------------------------------------------------------------------------


async def _vizier_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search VizieR for astronomical catalogs."""
    query = extract_query(params, "catalog", "object")
    limit = min(params.get("limit", 20), 100)

    resp = await get_with_retry(
        http,
        f"{api_base}/votable",
        params={
            "-words": query,
            "-meta.max": limit,
            "-out.max": limit,
        },
    )
    if resp.status_code != 200:
        logger.info("VizieR fallback: status %d for query=%r", resp.status_code, query)
        return {"catalogs": []}

    # VizieR returns VOTable XML; parse a simplified version
    import xml.etree.ElementTree as ET

    catalogs = []
    try:
        root = ET.fromstring(resp.text)
        ns = {"vot": "http://www.ivoa.net/xml/VOTable/v1.3"}
        for resource in root.findall(".//vot:RESOURCE", ns):
            name = resource.get("name", "")
            desc_el = resource.find("vot:DESCRIPTION", ns)
            description = ""
            if desc_el is not None and desc_el.text:
                description = desc_el.text.strip()
            catalogs.append(
                {
                    "catalog_id": name,
                    "description": description,
                    "source": "vizier",
                }
            )
    except ET.ParseError:
        logger.info("VizieR fallback: could not parse VOTable response")
        return {"catalogs": []}

    logger.info("VizieR fallback: %d catalogs for query=%r", len(catalogs), query)
    return {"catalogs": catalogs}


# ---------------------------------------------------------------------------
# NASA ADS — Astrophysics Data System (requires ADS_API_KEY)
# ---------------------------------------------------------------------------


async def _nasa_ads_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search NASA ADS for astrophysics papers."""
    api_key = require_api_key("ADS_API_KEY", "nasa-ads")
    query = extract_query(params)
    limit = min(params.get("limit", 20), 100)

    resp = await get_with_retry(
        http,
        f"{api_base}/search/query",
        params={
            "q": query,
            "rows": limit,
            "fl": "title,author,year,doi,bibcode,abstract",
        },
        headers={"Authorization": f"Bearer {api_key}"},
    )
    resp.raise_for_status()
    data = resp.json()

    docs = data.get("response", {}).get("docs", [])
    papers = []
    for doc in docs:
        title_list = doc.get("title", [])
        papers.append(
            {
                "bibcode": doc.get("bibcode", ""),
                "title": title_list[0] if title_list else "",
                "authors": doc.get("author", [])[:5],
                "year": safe_int(doc.get("year")),
                "doi": (doc.get("doi", [""]) or [""])[0],
                "abstract": doc.get("abstract", ""),
                "source": "nasa-ads",
            }
        )

    logger.info("NASA ADS fallback: %d papers for query=%r", len(papers), query)
    return {"papers": papers}


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

HANDLERS: dict[str, FallbackHandler] = {
    "simbad": _simbad_search,
    "mast": _mast_search,
    "vizier": _vizier_search,
    "nasa-ads": _nasa_ads_search,
}
