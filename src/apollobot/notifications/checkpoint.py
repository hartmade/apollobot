"""
ChannelCheckpointHandler — bridges the executor's CheckpointHandler
interface to the notification router.

This allows the existing executor to send notifications and request
approvals through external channels without any changes to its API.
"""

from __future__ import annotations

from apollobot.agents.executor import CheckpointHandler
from apollobot.notifications.events import (
    EventSeverity,
    EventType,
    NotificationEvent,
)
from apollobot.notifications.router import NotificationRouter


class ChannelCheckpointHandler(CheckpointHandler):
    """Bridges executor's CheckpointHandler to NotificationRouter."""

    def __init__(self, router: NotificationRouter, session_id: str = "") -> None:
        self.router = router
        self.session_id = session_id

    async def request_approval(self, phase: str, summary: str) -> bool:
        event = NotificationEvent(
            event_type=EventType.CHECKPOINT_APPROVAL,
            severity=EventSeverity.WARNING,
            session_id=self.session_id,
            phase=phase,
            title=f"Approval needed: {phase}",
            summary=summary,
            requires_response=True,
        )
        return await self.router.request_approval(event)

    async def notify(self, phase: str, summary: str) -> None:
        event = NotificationEvent(
            event_type=EventType.PHASE_COMPLETED,
            session_id=self.session_id,
            phase=phase,
            title=f"Phase complete: {phase}",
            summary=summary,
        )
        await self.router.dispatch(event)
