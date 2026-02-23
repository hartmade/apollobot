"""
Generic webhook channel — POST JSON to any URL.

Supports HMAC signature verification and configurable headers.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from typing import Any

import httpx

from apollobot.notifications.channel import NotificationChannel
from apollobot.notifications.events import NotificationEvent

logger = logging.getLogger(__name__)


class WebhookChannel(NotificationChannel):
    """Generic webhook notification channel."""

    name: str = "webhook"
    supports_responses: bool = False

    def __init__(
        self,
        url: str,
        *,
        secret: str = "",
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.url = url
        self.secret = secret
        self.headers = headers or {}
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        self._client = httpx.AsyncClient(timeout=self.timeout)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def send(self, event: NotificationEvent) -> None:
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        try:
            payload = event.model_dump(mode="json")
            body = json.dumps(payload)

            send_headers: dict[str, str] = {
                "Content-Type": "application/json",
                **self.headers,
            }

            if self.secret:
                signature = hmac.new(
                    self.secret.encode(),
                    body.encode(),
                    hashlib.sha256,
                ).hexdigest()
                send_headers["X-Apollo-Signature"] = f"sha256={signature}"

            resp = await client.post(self.url, content=body, headers=send_headers)
            resp.raise_for_status()
        except httpx.HTTPError:
            logger.exception("Webhook delivery failed to %s", self.url)
        finally:
            if not self._client:
                await client.aclose()
