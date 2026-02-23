"""
Discord channel — bidirectional notifications.

Uses webhook for outbound notifications. For approval responses,
uses bot token + gateway to watch for reactions on the approval message.
"""

from __future__ import annotations

import asyncio
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

_SEVERITY_COLOR = {
    EventSeverity.INFO: 0x3498DB,      # blue
    EventSeverity.WARNING: 0xF39C12,   # orange
    EventSeverity.ERROR: 0xE74C3C,     # red
    EventSeverity.CRITICAL: 0x8E44AD,  # purple
}

APPROVE_EMOJI = "\u2705"  # check mark
DENY_EMOJI = "\u274c"     # cross mark


class DiscordChannel(NotificationChannel):
    """Discord notification channel with optional bidirectional approval."""

    name: str = "discord"
    supports_responses: bool = True

    def __init__(
        self,
        webhook_url: str,
        *,
        bot_token: str = "",
        channel_id: str = "",
    ) -> None:
        self.webhook_url = webhook_url
        self.bot_token = bot_token
        self.channel_id = channel_id
        self._client: httpx.AsyncClient | None = None
        # Only support responses if bot_token and channel_id are provided
        if not (bot_token and channel_id):
            self.supports_responses = False

    async def connect(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def send(self, event: NotificationEvent) -> None:
        embed = self._build_embed(event)
        payload: dict[str, Any] = {"embeds": [embed]}

        if event.requires_response and self.supports_responses:
            # Send via bot API to get a message we can add reactions to
            await self._send_via_bot(event, embed)
        else:
            await self._send_via_webhook(payload)

    async def wait_for_response(
        self, event: NotificationEvent, timeout: float = 3600
    ) -> bool:
        """Poll for reactions on the approval message."""
        if not self.supports_responses:
            return True

        client = self._client or httpx.AsyncClient(timeout=30.0)
        headers = {"Authorization": f"Bot {self.bot_token}"}
        deadline = asyncio.get_event_loop().time() + timeout

        # Find the most recent bot message in the channel
        try:
            resp = await client.get(
                f"https://discord.com/api/v10/channels/{self.channel_id}/messages",
                headers=headers,
                params={"limit": 5},
            )
            messages = resp.json()
            message_id = messages[0]["id"] if messages else None
        except Exception:
            logger.exception("Failed to get Discord messages")
            return True

        if not message_id:
            return True

        try:
            while asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(5)
                try:
                    # Check reactions
                    for emoji in [APPROVE_EMOJI, DENY_EMOJI]:
                        resp = await client.get(
                            f"https://discord.com/api/v10/channels/{self.channel_id}"
                            f"/messages/{message_id}/reactions/{emoji}",
                            headers=headers,
                        )
                        if resp.status_code == 200:
                            users = resp.json()
                            # Filter out bot's own reactions
                            human_reacted = any(
                                not u.get("bot", False) for u in users
                            )
                            if human_reacted:
                                return emoji == APPROVE_EMOJI
                except httpx.HTTPError:
                    logger.debug("Discord poll error", exc_info=True)
        finally:
            if not self._client:
                await client.aclose()

        return True  # timeout → auto-approve

    async def _send_via_webhook(self, payload: dict[str, Any]) -> None:
        client = self._client or httpx.AsyncClient(timeout=30.0)
        try:
            resp = await client.post(self.webhook_url, json=payload)
            resp.raise_for_status()
        except httpx.HTTPError:
            logger.exception("Failed to send Discord webhook")
        finally:
            if not self._client:
                await client.aclose()

    async def _send_via_bot(
        self, event: NotificationEvent, embed: dict[str, Any]
    ) -> None:
        client = self._client or httpx.AsyncClient(timeout=30.0)
        headers = {"Authorization": f"Bot {self.bot_token}"}
        try:
            resp = await client.post(
                f"https://discord.com/api/v10/channels/{self.channel_id}/messages",
                headers=headers,
                json={"embeds": [embed]},
            )
            resp.raise_for_status()
            msg = resp.json()
            msg_id = msg["id"]

            # Add reaction buttons
            for emoji in [APPROVE_EMOJI, DENY_EMOJI]:
                await client.put(
                    f"https://discord.com/api/v10/channels/{self.channel_id}"
                    f"/messages/{msg_id}/reactions/{emoji}/@me",
                    headers=headers,
                )
        except httpx.HTTPError:
            logger.exception("Failed to send Discord bot message")
        finally:
            if not self._client:
                await client.aclose()

    def _build_embed(self, event: NotificationEvent) -> dict[str, Any]:
        color = _SEVERITY_COLOR.get(event.severity, 0x3498DB)
        embed: dict[str, Any] = {
            "title": event.title,
            "description": event.summary,
            "color": color,
            "footer": {"text": f"Session: {event.session_id}"},
            "timestamp": event.timestamp,
        }
        if event.phase:
            embed["fields"] = [{"name": "Phase", "value": event.phase, "inline": True}]
        return embed
