"""
Remote log transport — sends structured runtime logs to a remote endpoint.

Inspired by OpenCat's sendLog() pattern: fire-and-forget HTTP POST with
structured metadata. Failures are silently swallowed so logging never
disrupts the runtime.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


class RemoteLogTransport:
    """
    Sends structured log entries to a remote HTTP endpoint.

    Usage:
        transport = RemoteLogTransport(url="https://api.example.com/logs", api_key="...")
        await transport.send("info", "Tick completed", {"tick": 5, "cost": 0.12})
    """

    def __init__(
        self,
        url: str,
        api_key: str = "",
        agent_id: str = "",
        batch_size: int = 10,
        flush_interval: float = 30.0,
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.agent_id = agent_id
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self._buffer: list[dict[str, Any]] = []
        self._session: aiohttp.ClientSession | None = None
        self._flush_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the background flush loop."""
        if not self.url:
            return
        self._session = aiohttp.ClientSession()
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())

    async def stop(self) -> None:
        """Flush remaining entries and close."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        # Final flush
        await self._flush()
        if self._session:
            await self._session.close()
            self._session = None

    async def send(
        self,
        level: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Queue a log entry for batch sending. Non-blocking, never raises."""
        if not self.url:
            return
        entry = {
            "level": level,
            "message": message,
            "agent_id": self.agent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        self._buffer.append(entry)

        # Flush immediately if buffer is full
        if len(self._buffer) >= self.batch_size:
            asyncio.create_task(self._flush())

    async def _flush_loop(self) -> None:
        """Periodic flush of buffered entries."""
        while self._running:
            await asyncio.sleep(self.flush_interval)
            await self._flush()

    async def _flush(self) -> None:
        """Send buffered entries to remote endpoint. Fire-and-forget."""
        if not self._buffer or not self._session:
            return

        batch = self._buffer[:]
        self._buffer.clear()

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            async with self._session.post(
                self.url,
                json={"entries": batch},
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status >= 400:
                    logger.debug("Remote log POST returned %d", resp.status)
        except Exception:
            # Fire-and-forget: never let logging failures affect the runtime
            logger.debug("Remote log flush failed (buffered %d entries)", len(batch))
