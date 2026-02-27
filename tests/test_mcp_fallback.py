"""Tests for MCP direct API fallback adapters."""

import pytest
import httpx
from unittest.mock import AsyncMock, patch

from apollobot.mcp.fallback import (
    _arxiv_search,
    _pubmed_search,
    _semantic_scholar_search,
    fallback_query,
)

# ---------------------------------------------------------------------------
# Sample payloads
# ---------------------------------------------------------------------------

ARXIV_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <title>Test Paper on ML</title>
    <summary>This paper explores machine learning.</summary>
    <published>2023-01-15T00:00:00Z</published>
    <link title="doi" href="10.1234/test" />
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2302.00002v1</id>
    <title>Another Paper</title>
    <summary>Second abstract.</summary>
    <published>2023-02-20T00:00:00Z</published>
  </entry>
</feed>
"""

ARXIV_EMPTY_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>
"""

S2_JSON = {
    "data": [
        {
            "title": "Semantic Scholar Paper",
            "year": 2022,
            "abstract": "An abstract from S2.",
            "externalIds": {"DOI": "10.5678/s2", "ArXiv": "2201.00001"},
        },
        {
            "title": "Paper Without IDs",
            "year": None,
            "abstract": None,
            "externalIds": None,
        },
    ]
}

PUBMED_SEARCH_JSON = {
    "esearchresult": {
        "idlist": ["12345678", "87654321"],
    }
}

PUBMED_FETCH_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345678</PMID>
      <Article>
        <ArticleTitle>PubMed Test Article</ArticleTitle>
        <Abstract><AbstractText>A biomedical abstract.</AbstractText></Abstract>
        <Journal><JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue></Journal>
        <ELocationID EIdType="doi">10.9999/pm</ELocationID>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="doi">10.9999/pm</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>87654321</PMID>
      <Article>
        <ArticleTitle>Minimal Article</ArticleTitle>
        <Journal><JournalIssue><PubDate></PubDate></JournalIssue></Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""

PUBMED_EMPTY_SEARCH_JSON = {
    "esearchresult": {"idlist": []},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_http_get(responses: dict[str, httpx.Response]):
    """Create a mock httpx client whose .get() returns based on URL prefix."""
    async def _get(url, params=None, **kwargs):
        for key, resp in responses.items():
            if key in url:
                return resp
        raise httpx.ConnectError(f"No mock for {url}")

    client = AsyncMock(spec=httpx.AsyncClient)
    client.get = AsyncMock(side_effect=_get)
    return client


def _response(text: str = "", json_data: dict | None = None, status: int = 200):
    """Build a fake httpx.Response."""
    resp = httpx.Response(
        status_code=status,
        request=httpx.Request("GET", "http://test"),
    )
    if json_data is not None:
        import json as _json
        resp._content = _json.dumps(json_data).encode()
    else:
        resp._content = text.encode()
    return resp


# ---------------------------------------------------------------------------
# arXiv tests
# ---------------------------------------------------------------------------

class TestArxivFallback:
    @pytest.mark.asyncio
    async def test_basic_search(self):
        http = _mock_http_get({"query": _response(text=ARXIV_XML)})
        result = await _arxiv_search("http://export.arxiv.org/api", {"query": "machine learning"}, http)
        papers = result["papers"]
        assert len(papers) == 2
        assert papers[0]["title"] == "Test Paper on ML"
        assert papers[0]["year"] == 2023
        assert papers[0]["arxiv_id"] == "2301.00001v1"
        assert papers[0]["doi"] == "10.1234/test"
        assert papers[0]["source"] == "arxiv"

    @pytest.mark.asyncio
    async def test_empty_results(self):
        http = _mock_http_get({"query": _response(text=ARXIV_EMPTY_XML)})
        result = await _arxiv_search("http://export.arxiv.org/api", {"query": "zzzzz"}, http)
        assert result["papers"] == []

    @pytest.mark.asyncio
    async def test_missing_fields(self):
        """Paper without DOI link still parses."""
        http = _mock_http_get({"query": _response(text=ARXIV_XML)})
        result = await _arxiv_search("http://export.arxiv.org/api", {"query": "test"}, http)
        assert result["papers"][1]["doi"] == ""
        assert result["papers"][1]["title"] == "Another Paper"


# ---------------------------------------------------------------------------
# Semantic Scholar tests
# ---------------------------------------------------------------------------

class TestSemanticScholarFallback:
    @pytest.mark.asyncio
    async def test_basic_search(self):
        http = _mock_http_get({"paper/search": _response(json_data=S2_JSON)})
        result = await _semantic_scholar_search(
            "https://api.semanticscholar.org/graph/v1", {"query": "nlp"}, http
        )
        papers = result["papers"]
        assert len(papers) == 2
        assert papers[0]["title"] == "Semantic Scholar Paper"
        assert papers[0]["doi"] == "10.5678/s2"
        assert papers[0]["arxiv_id"] == "2201.00001"
        assert papers[0]["source"] == "semantic_scholar"

    @pytest.mark.asyncio
    async def test_missing_external_ids(self):
        """Paper with None externalIds doesn't crash."""
        http = _mock_http_get({"paper/search": _response(json_data=S2_JSON)})
        result = await _semantic_scholar_search(
            "https://api.semanticscholar.org/graph/v1", {"query": "test"}, http
        )
        assert result["papers"][1]["doi"] == ""
        assert result["papers"][1]["year"] == 0

    @pytest.mark.asyncio
    async def test_empty_results(self):
        http = _mock_http_get({"paper/search": _response(json_data={"data": []})})
        result = await _semantic_scholar_search(
            "https://api.semanticscholar.org/graph/v1", {"query": "zzzzz"}, http
        )
        assert result["papers"] == []


# ---------------------------------------------------------------------------
# PubMed tests
# ---------------------------------------------------------------------------

class TestPubMedFallback:
    @pytest.mark.asyncio
    async def test_basic_search(self):
        http = _mock_http_get({
            "esearch": _response(json_data=PUBMED_SEARCH_JSON),
            "efetch": _response(text=PUBMED_FETCH_XML),
        })
        result = await _pubmed_search(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils", {"query": "cancer"}, http
        )
        papers = result["papers"]
        assert len(papers) == 2
        assert papers[0]["title"] == "PubMed Test Article"
        assert papers[0]["year"] == 2021
        assert papers[0]["doi"] == "10.9999/pm"
        assert papers[0]["pmid"] == "12345678"
        assert papers[0]["source"] == "pubmed"

    @pytest.mark.asyncio
    async def test_empty_search(self):
        http = _mock_http_get({
            "esearch": _response(json_data=PUBMED_EMPTY_SEARCH_JSON),
        })
        result = await _pubmed_search(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils", {"query": "zzzzz"}, http
        )
        assert result["papers"] == []

    @pytest.mark.asyncio
    async def test_missing_fields(self):
        """Article with minimal fields still parses."""
        http = _mock_http_get({
            "esearch": _response(json_data=PUBMED_SEARCH_JSON),
            "efetch": _response(text=PUBMED_FETCH_XML),
        })
        result = await _pubmed_search(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils", {"query": "test"}, http
        )
        minimal = result["papers"][1]
        assert minimal["title"] == "Minimal Article"
        assert minimal["abstract"] == ""
        assert minimal["year"] == 0


# ---------------------------------------------------------------------------
# Dispatcher tests
# ---------------------------------------------------------------------------

class TestFallbackDispatcher:
    @pytest.mark.asyncio
    async def test_routes_to_arxiv(self):
        http = _mock_http_get({"query": _response(text=ARXIV_XML)})
        result = await fallback_query("arxiv", "http://export.arxiv.org/api", {"query": "test"}, http)
        assert len(result["papers"]) == 2

    @pytest.mark.asyncio
    async def test_routes_to_semantic_scholar(self):
        http = _mock_http_get({"paper/search": _response(json_data=S2_JSON)})
        result = await fallback_query(
            "semantic-scholar", "https://api.semanticscholar.org/graph/v1",
            {"query": "test"}, http,
        )
        assert len(result["papers"]) == 2

    @pytest.mark.asyncio
    async def test_unknown_server_raises(self):
        http = AsyncMock()
        with pytest.raises(ValueError, match="No fallback handler"):
            await fallback_query("unknown-server", "http://example.com", {}, http)
