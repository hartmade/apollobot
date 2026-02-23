"""
Google Chat channel — webhook-based outbound notifications.

Sends card-formatted messages to a Google Chat webhook URL.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from apollobot.notifications.channel import NotificationChannel
from apollobot.notifications.events import (
    EventSeverity,
    NotificationEvent,
)

logger = logging.getLogger(__name__)


class GoogleChatChannel(NotificationChannel):
    """Google Chat webhook notification channel."""

    name: str = "google_chat"
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
        card = self._build_card(event)
        client = self._client or httpx.AsyncClient(timeout=30.0)
        try:
            resp = await client.post(self.webhook_url, json=card)
            resp.raise_for_status()
        except httpx.HTTPError:
            logger.exception("Failed to send Google Chat notification")
        finally:
            if not self._client:
                await client.aclose()

    def _build_card(self, event: NotificationEvent) -> dict[str, Any]:
        severity_icon = {
            EventSeverity.INFO: "BOOKMARK",
            EventSeverity.WARNING: "DESCRIPTION",
            EventSeverity.ERROR: "BUG_REPORT",
            EventSeverity.CRITICAL: "URGENT",
        }

        widgets: list[dict[str, Any]] = [
            {
                "decoratedText": {
                    "text": event.summary,
                    "startIcon": {
                        "knownIcon": severity_icon.get(event.severity, "BOOKMARK"),
                    },
                },
            },
        ]

        if event.phase:
            widgets.append({
                "decoratedText": {
                    "topLabel": "Phase",
                    "text": event.phase,
                },
            })

        widgets.append({
            "decoratedText": {
                "topLabel": "Session",
                "text": event.session_id,
            },
        })

        return {
            "cardsV2": [
                {
                    "cardId": f"apollo-{event.event_type.value}",
                    "card": {
                        "header": {
                            "title": event.title,
                            "subtitle": f"ApolloBot | {event.event_type.value}",
                        },
                        "sections": [{"widgets": widgets}],
                    },
                }
            ]
        }
