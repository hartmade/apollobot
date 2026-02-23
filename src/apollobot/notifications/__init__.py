"""
Notification system for ApolloBot.

Provides multi-channel notifications for research session events
including progress updates, findings, checkpoint approvals, and errors.
"""

from apollobot.notifications.channel import NotificationChannel
from apollobot.notifications.config import ChannelConfig, NotificationsConfig
from apollobot.notifications.events import EventSeverity, EventType, NotificationEvent
from apollobot.notifications.router import NotificationRouter

__all__ = [
    "EventSeverity",
    "EventType",
    "NotificationChannel",
    "NotificationEvent",
    "NotificationRouter",
    "ChannelConfig",
    "NotificationsConfig",
]
