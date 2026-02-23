"""
Slack channel — webhook-based outbound notifications.

Sends formatted Block Kit messages to a Slack incoming webhook URL.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from apollobot.notifications.channel import NotificationChannel
from apollobot.notifications.events import (
    EventSeverity,
    EventType,
    NotificationEvent,
)

logger = logging.getLogger(__name__)

_SEVERITY_EMOJI = {
    EventSeverity.INFO: ":information_source:",
    EventSeverity.WARNING: ":warning:",
    EventSeverity.ERROR: ":x:",
    EventSeverity.CRITICAL: ":rotating_light:",
}


class SlackChannel(NotificationChannel):
    """Slack incoming webhook notification channel."""

    name: str = "slack"
    supports_responses: bool = False

    def __init__(self, webhook_url: str) -> None:
        self.webhook_url = webhook_url
        self._client: httpx.AsyncClient | None = None

    async def connect(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def send(self, event: NotificationEvent) -> None:
        blocks = self._build_blocks(event)
        payload = {"blocks": blocks}

        client = self._client or httpx.AsyncClient(timeout=30.0)
        try:
            resp = await client.post(self.webhook_url, json=payload)
            resp.raise_for_status()
        except httpx.HTTPError:
            logger.exception("Failed to send Slack notification")
        finally:
            if not self._client:
                await client.aclose()

    def _build_blocks(self, event: NotificationEvent) -> list[dict[str, Any]]:
        emoji = _SEVERITY_EMOJI.get(event.severity, ":information_source:")

        blocks: list[dict[str, Any]] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{event.title}",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"{emoji} {event.summary}",
                },
            },
        ]

        fields = []
        if event.phase:
            fields.append({"type": "mrkdwn", "text": f"*Phase:* {event.phase}"})
        fields.append({"type": "mrkdwn", "text": f"*Session:* `{event.session_id}`"})

        if fields:
            blocks.append({"type": "section", "fields": fields})

        blocks.append({"type": "divider"})

        return blocks
