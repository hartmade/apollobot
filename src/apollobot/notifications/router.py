"""
NotificationRouter — dispatches events to registered channels.

Handles fan-out (send to all matching channels) and approval routing
(first-responder wins across bidirectional channels).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from apollobot.notifications.channel import NotificationChannel
from apollobot.notifications.events import EventType, NotificationEvent

logger = logging.getLogger(__name__)


class NotificationRouter:
    """Dispatches events to registered channels with filtering."""

    def __init__(self) -> None:
        self.channels: list[NotificationChannel] = []
        self.filters: dict[str, list[str]] = {}  # channel_name → subscribed event types

    def register(
        self,
        channel: NotificationChannel,
        events: list[str] | None = None,
    ) -> None:
        """Register a channel with optional event type filter."""
        self.channels.append(channel)
        self.filters[channel.name] = events or ["*"]

    def _matches(self, channel: NotificationChannel, event: NotificationEvent) -> bool:
        """Check if a channel is subscribed to this event type."""
        subscribed = self.filters.get(channel.name, ["*"])
        return "*" in subscribed or event.event_type.value in subscribed

    async def dispatch(self, event: NotificationEvent) -> None:
        """Fan-out event to all matching channels concurrently."""
        tasks = []
        for ch in self.channels:
            if self._matches(ch, event):
                tasks.append(self._safe_send(ch, event))
        if tasks:
            await asyncio.gather(*tasks)

    async def request_approval(self, event: NotificationEvent) -> bool:
        """
        Send approval request to bidirectional channels.

        Returns as soon as any one channel responds (first-responder wins).
        Falls back to True (auto-approve) if no bidirectional channels.
        Also dispatches to non-bidirectional channels as a notification.
        """
        bidirectional: list[NotificationChannel] = []
        notify_only: list[NotificationChannel] = []

        for ch in self.channels:
            if not self._matches(ch, event):
                continue
            if ch.supports_responses:
                bidirectional.append(ch)
            else:
                notify_only.append(ch)

        # Notify non-bidirectional channels
        for ch in notify_only:
            asyncio.create_task(self._safe_send(ch, event))

        if not bidirectional:
            return True  # auto-approve if no bidirectional channels

        # Send to all bidirectional channels, wait for first response
        for ch in bidirectional:
            await self._safe_send(ch, event)

        response_tasks = [
            asyncio.create_task(ch.wait_for_response(event))
            for ch in bidirectional
        ]

        done, pending = await asyncio.wait(
            response_tasks, return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel remaining
        for task in pending:
            task.cancel()

        # Get result from first completed
        for task in done:
            try:
                return task.result()
            except Exception:
                logger.exception("Error getting approval response")

        return True  # fallback: auto-approve

    async def connect_all(self) -> None:
        """Connect all registered channels."""
        for ch in self.channels:
            try:
                await ch.connect()
            except Exception:
                logger.exception("Failed to connect channel %s", ch.name)

    async def disconnect_all(self) -> None:
        """Disconnect all registered channels."""
        for ch in self.channels:
            try:
                await ch.disconnect()
            except Exception:
                logger.exception("Failed to disconnect channel %s", ch.name)

    async def _safe_send(
        self, channel: NotificationChannel, event: NotificationEvent
    ) -> None:
        """Send with error handling so one channel failure doesn't break others."""
        try:
            await channel.send(event)
        except Exception:
            logger.exception("Failed to send to channel %s", channel.name)
