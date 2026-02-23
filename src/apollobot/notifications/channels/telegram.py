"""
Telegram channel — bidirectional notifications via Bot API.

Sends formatted messages with session info. For approvals, sends
inline keyboard buttons [Approve] [Deny] and polls for callbacks.
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

_BASE_URL = "https://api.telegram.org/bot{token}"

_SEVERITY_EMOJI = {
    EventSeverity.INFO: "\u2139\ufe0f",
    EventSeverity.WARNING: "\u26a0\ufe0f",
    EventSeverity.ERROR: "\u274c",
    EventSeverity.CRITICAL: "\U0001f6a8",
}


class TelegramChannel(NotificationChannel):
    """Bidirectional Telegram notification channel using Bot API."""

    name: str = "telegram"
    supports_responses: bool = True

    def __init__(self, token: str, chat_id: str) -> None:
        self.token = token
        self.chat_id = chat_id
        self._client: httpx.AsyncClient | None = None
        self._base_url = _BASE_URL.format(token=token)
        self._last_update_id: int = 0

    async def connect(self) -> None:
        self._client = httpx.AsyncClient(timeout=30.0)
        # Flush pending updates
        try:
            resp = await self._client.get(
                f"{self._base_url}/getUpdates",
                params={"offset": -1, "limit": 1},
            )
            data = resp.json()
            if data.get("ok") and data.get("result"):
                self._last_update_id = data["result"][-1]["update_id"] + 1
        except Exception:
            logger.debug("Failed to flush Telegram updates", exc_info=True)

    async def disconnect(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def send(self, event: NotificationEvent) -> None:
        text = self._format_message(event)
        payload: dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
        }

        if event.requires_response:
            payload["reply_markup"] = {
                "inline_keyboard": [
                    [
                        {"text": "\u2705 Approve", "callback_data": f"approve:{event.session_id}:{event.phase}"},
                        {"text": "\u274c Deny", "callback_data": f"deny:{event.session_id}:{event.phase}"},
                    ]
                ]
            }

        client = self._client or httpx.AsyncClient(timeout=30.0)
        try:
            resp = await client.post(
                f"{self._base_url}/sendMessage",
                json=payload,
            )
            resp.raise_for_status()
        except httpx.HTTPError:
            logger.exception("Failed to send Telegram message")
        finally:
            if not self._client:
                await client.aclose()

    async def wait_for_response(
        self, event: NotificationEvent, timeout: float = 3600
    ) -> bool:
        """Poll for inline keyboard callback response."""
        client = self._client or httpx.AsyncClient(timeout=30.0)
        deadline = asyncio.get_event_loop().time() + timeout
        expected_prefix = f"approve:{event.session_id}:{event.phase}"
        deny_prefix = f"deny:{event.session_id}:{event.phase}"

        try:
            while asyncio.get_event_loop().time() < deadline:
                try:
                    resp = await client.get(
                        f"{self._base_url}/getUpdates",
                        params={
                            "offset": self._last_update_id,
                            "timeout": 30,
                            "allowed_updates": '["callback_query"]',
                        },
                    )
                    data = resp.json()
                    if data.get("ok"):
                        for update in data.get("result", []):
                            self._last_update_id = update["update_id"] + 1
                            callback = update.get("callback_query", {})
                            cb_data = callback.get("data", "")

                            if cb_data == expected_prefix:
                                await self._answer_callback(
                                    client, callback["id"], "Approved"
                                )
                                return True
                            elif cb_data == deny_prefix:
                                await self._answer_callback(
                                    client, callback["id"], "Denied"
                                )
                                return False
                except httpx.HTTPError:
                    logger.debug("Telegram poll error", exc_info=True)
                    await asyncio.sleep(5)
        finally:
            if not self._client:
                await client.aclose()

        return True  # timeout → auto-approve

    async def _answer_callback(
        self, client: httpx.AsyncClient, callback_id: str, text: str
    ) -> None:
        try:
            await client.post(
                f"{self._base_url}/answerCallbackQuery",
                json={"callback_query_id": callback_id, "text": text},
            )
        except httpx.HTTPError:
            pass

    def _format_message(self, event: NotificationEvent) -> str:
        emoji = _SEVERITY_EMOJI.get(event.severity, "\u2139\ufe0f")
        lines = [
            f"{emoji} <b>{event.title}</b>",
            "",
            event.summary,
        ]
        if event.phase:
            lines.append(f"\n<i>Phase:</i> {event.phase}")
        lines.append(f"<i>Session:</i> <code>{event.session_id}</code>")
        return "\n".join(lines)
