"""Contracts for direct-first built-in scientific data connectors."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from apollobot.mcp import MCPClient, MCPServerInfo
from apollobot.mcp.fallback import _FALLBACK_HANDLERS
from apollobot.mcp.servers.builtin import (
    ALL_BUILTIN_SERVERS,
    LITERATURE_SERVERS,
    resolve_builtin_proxy_url,
)


def test_builtin_connectors_default_to_direct_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("APOLLOBOT_MCP_PROXY_URL", raising=False)

    assert all(server.url == "" for server in ALL_BUILTIN_SERVERS)
    assert {server.name for server in ALL_BUILTIN_SERVERS} == set(_FALLBACK_HANDLERS)


def test_builtin_proxy_url_is_explicit_and_path_preserving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APOLLOBOT_MCP_PROXY_URL", "https://mcp.example.org/adapters/")

    assert LITERATURE_SERVERS[0].url == "https://mcp.example.org/adapters/pubmed"


@pytest.mark.parametrize(
    "proxy_url",
    [
        "mcp.example.org",
        "ftp://mcp.example.org",
        "https://user:secret@mcp.example.org",
        "https://mcp.example.org?tenant=one",
        "https://mcp.example.org#fragment",
    ],
)
def test_builtin_proxy_url_rejects_unsafe_values(proxy_url: str) -> None:
    with pytest.raises(ValueError, match="APOLLOBOT_MCP_PROXY_URL"):
        resolve_builtin_proxy_url(LITERATURE_SERVERS[0], proxy_url)


@pytest.mark.asyncio
async def test_direct_connector_bypasses_proxy_request() -> None:
    client = MCPClient()
    http = AsyncMock(spec=httpx.AsyncClient)
    client._http = http
    client.register(
        MCPServerInfo(
            name="pubmed",
            url="",
            description="PubMed",
            api_base="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        )
    )
    expected = {"papers": [{"title": "Direct result"}]}

    with patch(
        "apollobot.mcp.fallback.fallback_query",
        new=AsyncMock(return_value=expected),
    ) as fallback:
        result = await client.query("pubmed", "search", {"query": "calibration"})

    assert result == expected
    http.post.assert_not_awaited()
    fallback.assert_awaited_once()
    await client.close()


@pytest.mark.asyncio
async def test_failed_proxy_query_uses_direct_adapter() -> None:
    client = MCPClient()
    http = AsyncMock(spec=httpx.AsyncClient)
    http.post.side_effect = httpx.ConnectError("proxy unavailable")
    client._http = http
    client.register(
        MCPServerInfo(
            name="arxiv",
            url="https://mcp.example.org/arxiv",
            description="arXiv",
            api_base="https://export.arxiv.org/api",
        )
    )

    with patch(
        "apollobot.mcp.fallback.fallback_query",
        new=AsyncMock(return_value={"papers": []}),
    ) as fallback:
        result = await client.query("arxiv", "search", {"query": "test"})

    assert result == {"papers": []}
    fallback.assert_awaited_once()
    await client.close()
