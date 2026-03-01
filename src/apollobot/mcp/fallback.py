"""
Direct API fallback adapters for MCP servers.

When the MCP proxy is unreachable, these adapters translate
MCP search queries into native API calls for arXiv, Semantic Scholar,
and PubMed.
"""

from __future__ import annotations

import asyncio
import logging
import time
import xml.etree.ElementTree as ET
from typing import Any, Callable, Awaitable

import httpx

logger = logging.getLogger(__name__)

# Per-server rate limiting — prevents 429s by throttling proactively
_MIN_INTERVALS: dict[str, float] = {
    "semantic-scholar": 1.5,  # S2 public API: ~1 req/sec
}
_last_request_time: dict[str, float] = {}
_throttle_lock = asyncio.Lock()


async def _throttle(server_name: str) -> None:
    """Wait if needed to respect per-server rate limits."""
    min_interval = _MIN_INTERVALS.get(server_name)
    if min_interval is None:
        return
    async with _throttle_lock:
        last = _last_request_time.get(server_name, 0.0)
        elapsed = time.monotonic() - last
        if elapsed < min_interval:
            wait = min_interval - elapsed
            logger.debug("Throttling %s for %.2fs", server_name, wait)
            await asyncio.sleep(wait)
        _last_request_time[server_name] = time.monotonic()


async def _get_with_retry(
    http: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> httpx.Response:
    """HTTP GET with exponential backoff retry on 429 rate-limit responses."""
    for attempt in range(max_retries + 1):
        resp = await http.get(url, params=params, headers=headers)
        if resp.status_code != 429 or attempt == max_retries:
            return resp
        delay = base_delay * (2 ** attempt)
        logger.info("Rate limited (429), retrying in %.1fs (attempt %d/%d)", delay, attempt + 1, max_retries)
        await asyncio.sleep(delay)
    return resp  # unreachable but satisfies type checker

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

    resp = await _get_with_retry(
        http,
        f"{api_base}/paper/search",
        params={
            "query": query,
            "limit": limit,
            "fields": "title,year,abstract,externalIds",
        },
        max_retries=5,
        base_delay=2.0,
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
# HuggingFace adapter
# ---------------------------------------------------------------------------

async def _huggingface_search(
    api_base: str,
    params: dict[str, Any],
    http: httpx.AsyncClient,
) -> dict[str, Any]:
    """Search HuggingFace models/datasets and normalize as papers."""
    query = params.get("query", "")
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
    query = params.get("query", "")
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
    query = params.get("query", "")
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
# Dispatcher
# ---------------------------------------------------------------------------

_FALLBACK_HANDLERS: dict[str, FallbackHandler] = {
    "arxiv": _arxiv_search,
    "semantic-scholar": _semantic_scholar_search,
    "pubmed": _pubmed_search,
    "huggingface": _huggingface_search,
    "papers-with-code": _papers_with_code_search,
    "openml": _openml_search,
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
    await _throttle(server_name)
    return await handler(api_base, params, http_client)
