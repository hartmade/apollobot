"""
Health check HTTP server for container deployment.

Exposes /health endpoint for Docker HEALTHCHECK, Kubernetes probes,
and backend auto-restart logic. Inspired by OpenCat's entrypoint health server.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone

from aiohttp import web

logger = logging.getLogger(__name__)


class HealthServer:
    """Lightweight HTTP server exposing /health for liveness checks."""

    def __init__(self, port: int = 8080) -> None:
        self.port = port
        self._app = web.Application()
        self._app.router.add_get("/health", self._health_handler)
        self._app.router.add_post("/guardrails", self._guardrails_handler)
        self._runner: web.AppRunner | None = None

        # Callback for guardrails updates (set by the runner)
        self.on_guardrails_update: callable | None = None

        # State updated by the runner
        self.running = False
        self.tick_count = 0
        self.last_tick_time: str = ""
        self.start_time: float = 0.0
        self.watchdog_state: str = "closed"
        self.domain: str = ""
        self.active_sessions: int = 0
        self.total_papers: int = 0
        self.daily_cost: float = 0.0

    async def start(self) -> None:
        """Start the health server."""
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", self.port)
        try:
            await site.start()
            logger.info("Health server listening on :%d", self.port)
        except OSError as e:
            logger.warning("Could not start health server on :%d: %s", self.port, e)

    async def stop(self) -> None:
        """Stop the health server."""
        if self._runner:
            await self._runner.cleanup()
            self._runner = None

    def update(self, **kwargs: object) -> None:
        """Update health state from the runner."""
        for key, val in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, val)

    async def _health_handler(self, request: web.Request) -> web.Response:
        healthy = self.running and self.watchdog_state != "open"
        uptime = time.monotonic() - self.start_time if self.start_time else 0

        body = {
            "status": "healthy" if healthy else "degraded",
            "running": self.running,
            "tick_count": self.tick_count,
            "last_tick": self.last_tick_time,
            "uptime_seconds": int(uptime),
            "watchdog": self.watchdog_state,
            "domain": self.domain,
            "active_sessions": self.active_sessions,
            "total_papers": self.total_papers,
            "daily_cost_usd": round(self.daily_cost, 2),
        }

        status = 200 if healthy else 503
        return web.Response(
            text=json.dumps(body),
            content_type="application/json",
            status=status,
        )

    async def _guardrails_handler(self, request: web.Request) -> web.Response:
        """Handle runtime guardrails updates via POST /guardrails."""
        if not self.on_guardrails_update:
            return web.Response(
                text=json.dumps({"error": "Guardrails updates not supported"}),
                content_type="application/json",
                status=501,
            )

        try:
            updates = await request.json()
        except Exception:
            return web.Response(
                text=json.dumps({"error": "Invalid JSON"}),
                content_type="application/json",
                status=400,
            )

        try:
            result = self.on_guardrails_update(updates)
            return web.Response(
                text=json.dumps({"updated": result}),
                content_type="application/json",
                status=200,
            )
        except Exception as e:
            return web.Response(
                text=json.dumps({"error": str(e)}),
                content_type="application/json",
                status=500,
            )
