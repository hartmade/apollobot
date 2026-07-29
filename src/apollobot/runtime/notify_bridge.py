"""
Bridge between runtime events and the notification system.

Subscribes to RunnerEventEmitter and translates runtime events into
NotificationEvents dispatched through the existing NotificationRouter.
This connects the autonomous runtime to Telegram, Discord, Slack,
email, and webhook channels.
"""

from __future__ import annotations

import logging
from typing import Any

from apollobot.notifications.events import EventSeverity, EventType, NotificationEvent
from apollobot.notifications.router import NotificationRouter
from apollobot.runtime.events import RunnerEvent, RunnerEventEmitter, RunnerEventType

logger = logging.getLogger(__name__)

# Map runner events to notification events
_EVENT_MAP: dict[RunnerEventType, tuple[EventType, EventSeverity]] = {
    RunnerEventType.RUNTIME_STARTED: (EventType.SESSION_STARTED, EventSeverity.INFO),
    RunnerEventType.RUNTIME_STOPPED: (EventType.SESSION_COMPLETED, EventSeverity.INFO),
    RunnerEventType.SESSION_STARTED: (EventType.SESSION_STARTED, EventSeverity.INFO),
    RunnerEventType.SESSION_COMPLETED: (EventType.SESSION_COMPLETED, EventSeverity.INFO),
    RunnerEventType.SESSION_FAILED: (EventType.SESSION_FAILED, EventSeverity.WARNING),
    RunnerEventType.BUDGET_WARNING: (EventType.BUDGET_WARNING, EventSeverity.WARNING),
    RunnerEventType.WATCHDOG_OPENED: (EventType.PHASE_FAILED, EventSeverity.ERROR),
    RunnerEventType.ACTION_BLOCKED: (EventType.BUDGET_WARNING, EventSeverity.WARNING),
}


class NotifyBridge:
    """
    Subscribes to runtime events and dispatches notifications.

    Usage:
        bridge = NotifyBridge(runner.events, notification_router)
        await bridge.start()
        # ... runtime runs ...
        await bridge.stop()
    """

    def __init__(
        self,
        emitter: RunnerEventEmitter,
        router: NotificationRouter,
        runtime_id: str = "runtime",
    ) -> None:
        self.emitter = emitter
        self.router = router
        self.runtime_id = runtime_id

    async def start(self) -> None:
        """Connect notification channels and subscribe to events."""
        await self.router.connect_all()
        self.emitter.subscribe(None, self._on_event)  # subscribe to all events

    async def stop(self) -> None:
        """Unsubscribe and disconnect channels."""
        self.emitter.unsubscribe(self._on_event)
        await self.router.disconnect_all()

    async def _on_event(self, event: RunnerEvent) -> None:
        """Translate a runner event into a notification dispatch."""
        mapping = _EVENT_MAP.get(event.event_type)
        if not mapping:
            return  # Skip events that don't map to notifications

        notif_type, severity = mapping
        title = self._build_title(event)
        summary = self._build_summary(event)

        notif = NotificationEvent(
            event_type=notif_type,
            severity=severity,
            session_id=event.data.get("session_id", self.runtime_id),
            title=title,
            summary=summary,
            details=event.data,
        )

        try:
            await self.router.dispatch(notif)
        except Exception:
            logger.debug("Notification dispatch failed for %s", event.event_type.value)

    def _build_title(self, event: RunnerEvent) -> str:
        """Build a human-readable title for the notification."""
        titles = {
            RunnerEventType.RUNTIME_STARTED: "Runtime started",
            RunnerEventType.RUNTIME_STOPPED: "Runtime stopped",
            RunnerEventType.SESSION_STARTED: f"Research started: {event.data.get('objective', '?')[:50]}",
            RunnerEventType.SESSION_COMPLETED: f"Paper completed: {event.data.get('session_id', '?')}",
            RunnerEventType.SESSION_FAILED: f"Session failed: {event.data.get('session_id', '?')}",
            RunnerEventType.BUDGET_WARNING: "Budget warning",
            RunnerEventType.WATCHDOG_OPENED: "Watchdog circuit breaker opened",
            RunnerEventType.ACTION_BLOCKED: f"Action blocked: {event.data.get('action', '?')}",
        }
        return titles.get(event.event_type, event.event_type.value)

    def _build_summary(self, event: RunnerEvent) -> str:
        """Build a summary message for the notification."""
        if event.event_type == RunnerEventType.RUNTIME_STARTED:
            return f"Domain: {event.data.get('domain', '?')}"
        elif event.event_type == RunnerEventType.RUNTIME_STOPPED:
            return f"Reason: {event.data.get('reason', '?')}"
        elif event.event_type == RunnerEventType.SESSION_COMPLETED:
            cost = event.data.get("cost", 0)
            return f"Cost: ${cost:.2f}" if cost else "Session completed"
        elif event.event_type == RunnerEventType.SESSION_FAILED:
            return event.data.get("error", "Unknown error")[:200]
        elif event.event_type == RunnerEventType.BUDGET_WARNING:
            daily = event.data.get("daily_cost", 0)
            budget = event.data.get("budget", 0)
            return f"${daily:.2f} of ${budget:.2f} daily budget used"
        elif event.event_type == RunnerEventType.ACTION_BLOCKED:
            return event.data.get("reason", "")[:200]
        return ""
