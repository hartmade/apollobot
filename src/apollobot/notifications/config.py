"""
Configuration models for the notification system.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ChannelConfig(BaseModel):
    """Configuration for a single notification channel."""

    type: str  # telegram, discord, slack, google_chat, email, webhook, console
    enabled: bool = True
    events: list[str] = ["*"]  # event type filter, "*" = all

    # Channel-specific fields stored as extras
    model_config = ConfigDict(extra="allow")


class NotificationsConfig(BaseModel):
    """Top-level notifications configuration."""

    enabled: bool = False
    heartbeat_interval: int = 300  # seconds, 0 to disable
    channels: list[ChannelConfig] = []
