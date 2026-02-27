"""
Direct API fallback adapters for MCP servers.

When the MCP proxy is unreachable, these adapters translate
MCP search queries into native API calls for arXiv, Semantic Scholar,
and PubMed.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any, Callable, Awaitable

import httpx

logger = logging.getLogger(__name__)

# Type for fallback handler functions
FallbackHandler = Callable[
    [str, dict[str, Any], httpx.AsyncClient], Awaitable[dict[str, Any]]
]


# ---------------------------------------------------------------------------
# arXiv adapter
# ---------------------------------------------------------------------------

async def _arxiv_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search arXiv via its Atom API and normalize results."""
    query = params.get("query", "")
    limit = min(params.get("limit", 20), 100)

    resp = await http.get(
        f"{api_base}/query",
        params={"search_query": f"all:{query}", "max_results": limit},
    )
    resp.raise_for_status()

    papers = []
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)

    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        abstract_el = entry.find("atom:summary", ns)
        published_el = entry.find("atom:published", ns)
        id_el = entry.find("atom:id", ns)

        title = title_el.text.strip() if title_el is not None and title_el.text else ""
        abstract = abstract_el.text.strip() if abstract_el is not None and abstract_el.text else ""
        published = published_el.text.strip() if published_el is not None and published_el.text else ""
        arxiv_url = id_el.text.strip() if id_el is not None and id_el.text else ""

        year = int(published[:4]) if len(published) >= 4 else 0

        # Extract arXiv ID from URL (e.g. http://arxiv.org/abs/2301.12345v1)
        arxiv_id = ""
        if "/abs/" in arxiv_url:
            arxiv_id = arxiv_url.split("/abs/")[-1]

        # Look for DOI link
        doi = ""
        for link in entry.findall("atom:link", ns):
            if link.get("title") == "doi":
                doi = link.get("href", "")

        papers.append({
            "title": title,
            "abstract": abstract,
            "year": year,
            "doi": doi,
            "arxiv_id": arxiv_id,
            "source": "arxiv",
        })

    logger.info("arXiv fallback: %d papers for query=%r", len(papers), query)
    return {"papers": papers}


# ---------------------------------------------------------------------------
# Semantic Scholar adapter
# ---------------------------------------------------------------------------

async def _semantic_scholar_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search Semantic Scholar via its public API and normalize results."""
    query = params.get("query", "")
    limit = min(params.get("limit", 20), 100)

    resp = await http.get(
        f"{api_base}/paper/search",
        params={
            "query": query,
            "limit": limit,
            "fields": "title,year,abstract,externalIds",
        },
    )
    resp.raise_for_status()
    data = resp.json()

    papers = []
    for item in data.get("data", []):
        ext_ids = item.get("externalIds") or {}
        papers.append({
            "title": item.get("title", ""),
            "abstract": item.get("abstract", ""),
            "year": item.get("year", 0) or 0,
            "doi": ext_ids.get("DOI", ""),
            "arxiv_id": ext_ids.get("ArXiv", ""),
            "source": "semantic_scholar",
        })

    logger.info("Semantic Scholar fallback: %d papers for query=%r", len(papers), query)
    return {"papers": papers}


# ---------------------------------------------------------------------------
# PubMed adapter (two-step: esearch → efetch)
# ---------------------------------------------------------------------------

async def _pubmed_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search PubMed via E-utilities and normalize results."""
    query = params.get("query", "")
    limit = min(params.get("limit", 20), 100)

    # Step 1: esearch to get PMIDs
    search_resp = await http.get(
        f"{api_base}/esearch.fcgi",
        params={
            "db": "pubmed",
            "term": query,
            "retmax": limit,
            "retmode": "json",
        },
    )
    search_resp.raise_for_status()
    search_data = search_resp.json()
    id_list = search_data.get("esearchresult", {}).get("idlist", [])

    if not id_list:
        logger.info("PubMed fallback: 0 papers for query=%r", query)
        return {"papers": []}

    # Step 2: efetch to get article details
    fetch_resp = await http.get(
        f"{api_base}/efetch.fcgi",
        params={
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml",
        },
    )
    fetch_resp.raise_for_status()

    papers = []
    root = ET.fromstring(fetch_resp.text)

    for article in root.findall(".//PubmedArticle"):
        medline = article.find(".//MedlineCitation")
        if medline is None:
            continue

        art = medline.find("Article")
        if art is None:
            continue

        title_el = art.find("ArticleTitle")
        title = title_el.text.strip() if title_el is not None and title_el.text else ""

        abstract_el = art.find(".//AbstractText")
        abstract = abstract_el.text.strip() if abstract_el is not None and abstract_el.text else ""

        # Year from PubDate or MedlineDate
        year = 0
        pub_date = art.find(".//PubDate")
        if pub_date is not None:
            year_el = pub_date.find("Year")
            if year_el is not None and year_el.text:
                try:
                    year = int(year_el.text)
                except ValueError:
                    pass

        # DOI from ELocationID or ArticleIdList
        doi = ""
        eloc = art.find(".//ELocationID[@EIdType='doi']")
        if eloc is not None and eloc.text:
            doi = eloc.text
        else:
            pub_data = article.find(".//PubmedData")
            if pub_data is not None:
                for aid in pub_data.findall(".//ArticleId[@IdType='doi']"):
                    if aid.text:
                        doi = aid.text
                        break

        pmid_el = medline.find("PMID")
        pmid = pmid_el.text if pmid_el is not None and pmid_el.text else ""

        papers.append({
            "title": title,
            "abstract": abstract,
            "year": year,
            "doi": doi,
            "pmid": pmid,
            "source": "pubmed",
        })

    logger.info("PubMed fallback: %d papers for query=%r", len(papers), query)
    return {"papers": papers}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_FALLBACK_HANDLERS: dict[str, FallbackHandler] = {
    "arxiv": _arxiv_search,
    "semantic-scholar": _semantic_scholar_search,
    "pubmed": _pubmed_search,
}


async def fallback_query(
    server_name: str,
    api_base: str,
    params: dict[str, Any],
    http_client: httpx.AsyncClient,
) -> dict[str, Any]:
    """
    Dispatch a fallback query to the appropriate adapter.

    Raises ValueError if no handler is registered for the server name.
    """
    handler = _FALLBACK_HANDLERS.get(server_name)
    if handler is None:
        raise ValueError(
            f"No fallback handler for server '{server_name}'. "
            f"Available: {list(_FALLBACK_HANDLERS.keys())}"
        )
    logger.info("Falling back to direct API for server=%s", server_name)
    return await handler(api_base, params, http_client)
