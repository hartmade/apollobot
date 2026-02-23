"""
MCP Client — connects the research agent to external data sources,
compute resources, and tools via the Model Context Protocol.

Each MCP server exposes a standard interface:
  - discover() → what capabilities/data are available
  - query()    → request specific data or computation
  - status()   → check on running jobs
  - results()  → retrieve completed outputs
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx


@dataclass
class MCPCapability:
    """A single capability advertised by an MCP server."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)
    category: str = ""  # data, compute, analysis, writing


@dataclass
class MCPServerInfo:
    """Registration info for an MCP server."""

    name: str
    url: str
    description: str = ""
    domain: str = ""  # bioinformatics, physics, etc.
    auth_type: str = "none"  # none, bearer, api_key
    auth_token: str = ""
    capabilities: list[MCPCapability] = field(default_factory=list)
    healthy: bool = False


class MCPClient:
    """
    Client for communicating with MCP servers.

    Handles discovery, querying, and result retrieval across
    multiple registered servers.
    """

    def __init__(self, timeout: float = 30.0) -> None:
        self._servers: dict[str, MCPServerInfo] = {}
        self._http = httpx.AsyncClient(timeout=timeout)

    # ------------------------------------------------------------------
    # Server registration
    # ------------------------------------------------------------------

    def register(self, server: MCPServerInfo) -> None:
        """Register an MCP server."""
        self._servers[server.name] = server

    def register_from_config(self, config: dict[str, Any]) -> None:
        """Register a server from a config dict (e.g. from servers.yaml)."""
        self.register(MCPServerInfo(
            name=config["name"],
            url=config["url"],
            description=config.get("description", ""),
            domain=config.get("domain", ""),
            auth_type=config.get("auth", "none"),
            auth_token=config.get("token", ""),
        ))

    def get_servers(self, domain: str | None = None) -> list[MCPServerInfo]:
        """List registered servers, optionally filtered by domain."""
        servers = list(self._servers.values())
        if domain:
            servers = [s for s in servers if s.domain == domain or s.domain == ""]
        return servers

    # ------------------------------------------------------------------
    # Core protocol methods
    # ------------------------------------------------------------------

    async def discover(self, server_name: str) -> list[MCPCapability]:
        """
        Ask a server what capabilities it offers.
        Returns a list of MCPCapability objects.
        """
        server = self._get_server(server_name)
        resp = await self._request(server, "discover", {})

        capabilities = []
        for cap in resp.get("capabilities", []):
            capabilities.append(MCPCapability(
                name=cap["name"],
                description=cap.get("description", ""),
                parameters=cap.get("parameters", {}),
                category=cap.get("category", ""),
            ))
        server.capabilities = capabilities
        server.healthy = True
        return capabilities

    async def query(
        self,
        server_name: str,
        capability: str,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a query against a server capability.
        Returns the query result.
        """
        server = self._get_server(server_name)
        return await self._request(server, "query", {
            "capability": capability,
            "parameters": parameters or {},
        })

    async def status(self, server_name: str, job_id: str) -> dict[str, Any]:
        """Check status of a running job."""
        server = self._get_server(server_name)
        return await self._request(server, "status", {"job_id": job_id})

    async def results(self, server_name: str, job_id: str) -> dict[str, Any]:
        """Retrieve results from a completed job."""
        server = self._get_server(server_name)
        return await self._request(server, "results", {"job_id": job_id})

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    async def discover_all(self, domain: str | None = None) -> dict[str, list[MCPCapability]]:
        """Discover capabilities from all registered servers."""
        servers = self.get_servers(domain)
        tasks = {s.name: self.discover(s.name) for s in servers}
        results = {}
        for name, task in tasks.items():
            try:
                results[name] = await task
            except Exception as e:
                results[name] = []
                self._servers[name].healthy = False
        return results

    async def search_capabilities(
        self, query: str, domain: str | None = None
    ) -> list[tuple[str, MCPCapability]]:
        """
        Search across all servers for capabilities matching a query.
        Returns (server_name, capability) pairs.
        """
        all_caps = await self.discover_all(domain)
        matches = []
        query_lower = query.lower()
        for server_name, caps in all_caps.items():
            for cap in caps:
                if (
                    query_lower in cap.name.lower()
                    or query_lower in cap.description.lower()
                ):
                    matches.append((server_name, cap))
        return matches

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health_check(self, server_name: str) -> bool:
        """Check if a server is reachable and responding."""
        try:
            server = self._get_server(server_name)
            resp = await self._http.get(
                f"{server.url}/health",
                headers=self._auth_headers(server),
            )
            server.healthy = resp.status_code == 200
            return server.healthy
        except Exception:
            self._servers[server_name].healthy = False
            return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_server(self, name: str) -> MCPServerInfo:
        if name not in self._servers:
            raise ValueError(f"MCP server '{name}' not registered")
        return self._servers[name]

    def _auth_headers(self, server: MCPServerInfo) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if server.auth_type == "bearer" and server.auth_token:
            headers["Authorization"] = f"Bearer {server.auth_token}"
        elif server.auth_type == "api_key" and server.auth_token:
            headers["X-API-Key"] = server.auth_token
        return headers

    async def _request(
        self, server: MCPServerInfo, method: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Send a request to an MCP server."""
        resp = await self._http.post(
            f"{server.url}/{method}",
            json=payload,
            headers=self._auth_headers(server),
        )
        resp.raise_for_status()
        return resp.json()

    async def close(self) -> None:
        await self._http.aclose()


# ---------------------------------------------------------------------------
# Convenience: global server registry
# ---------------------------------------------------------------------------

_registry: dict[str, dict[str, Any]] = {}


def register_server(
    name: str,
    url: str,
    description: str = "",
    domain: str = "",
    auth: str = "none",
    token: str = "",
) -> None:
    """Register an MCP server in the global registry (for use by plugins/domain packs)."""
    _registry[name] = {
        "name": name,
        "url": url,
        "description": description,
        "domain": domain,
        "auth": auth,
        "token": token,
    }


def get_registry() -> dict[str, dict[str, Any]]:
    return _registry.copy()
