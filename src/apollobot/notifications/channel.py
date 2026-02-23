"""
NotificationChannel — abstract base class for all notification channels.

Each channel implementation (Telegram, Discord, Slack, etc.) inherits
from this ABC and implements `send()` and optionally `wait_for_response()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from apollobot.notifications.events import NotificationEvent


class NotificationChannel(ABC):
    """Base class for notification channels."""

    name: str = "unnamed"
    supports_responses: bool = False

    @abstractmethod
    async def send(self, event: NotificationEvent) -> None:
        """Send a notification event to this channel."""
        ...

    async def wait_for_response(
        self, event: NotificationEvent, timeout: float = 3600
    ) -> bool:
        """Wait for a user response (approve/deny). Default: auto-approve."""
        return True

    async def connect(self) -> None:
        """Establish connection (e.g. bot login). No-op by default."""

    async def disconnect(self) -> None:
        """Tear down connection. No-op by default."""
