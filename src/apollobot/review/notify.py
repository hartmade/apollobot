"""
ReviewNotifier — notification dispatcher for the submission review pipeline.

Uses the existing notification router to send formatted notifications
for submission lifecycle events.
"""

from __future__ import annotations

import logging
from typing import Any

from apollobot.notifications.events import EventSeverity, EventType, NotificationEvent
from apollobot.notifications.router import NotificationRouter

logger = logging.getLogger(__name__)


class ReviewNotifier:
    """
    Sends formatted notifications for submission review lifecycle events.

    Events:
    - submission_received: new submission arrived
    - ai_review_complete: AI review finished, ready for committee
    - revision_requested: submitter needs to revise
    - decision_made: final accept/reject decision
    """

    def __init__(self, router: NotificationRouter) -> None:
        self.router = router

    async def submission_received(
        self,
        *,
        title: str,
        session_id: str = "",
        track: str = "",
        provenance_badge: str = "bronze",
    ) -> None:
        """Notify that a new submission has been received."""
        badge_label = f" [{provenance_badge.title()} provenance]" if provenance_badge else ""
        track_label = f" | Track: {track}" if track else ""

        event = NotificationEvent(
            event_type=EventType.SESSION_STARTED,
            severity=EventSeverity.INFO,
            session_id=session_id or "manual",
            title=f"New submission: {title}",
            summary=f"Submission received{track_label}{badge_label}. AI review starting.",
            details={
                "submission_title": title,
                "session_id": session_id,
                "track": track,
                "provenance_badge": provenance_badge,
            },
        )
        await self.router.dispatch(event)

    async def ai_review_complete(
        self,
        *,
        title: str,
        session_id: str = "",
        recommendation: str = "",
        confidence: float = 0.0,
        provenance_badge: str = "bronze",
        summary: str = "",
    ) -> None:
        """Notify editorial committee that AI review is complete."""
        event = NotificationEvent(
            event_type=EventType.SESSION_COMPLETED,
            severity=EventSeverity.INFO,
            session_id=session_id or "manual",
            title=f"AI review complete: {title}",
            summary=(
                f"Recommendation: {recommendation.replace('_', ' ').title()} "
                f"(confidence: {confidence:.0%}). "
                f"Provenance: {provenance_badge.title()}. "
                f"{summary}"
            ),
            details={
                "submission_title": title,
                "recommendation": recommendation,
                "confidence": confidence,
                "provenance_badge": provenance_badge,
            },
        )
        await self.router.dispatch(event)

    async def revision_requested(
        self,
        *,
        title: str,
        session_id: str = "",
        revision_requests: list[str] | None = None,
    ) -> None:
        """Notify submitter that revisions are needed."""
        requests_text = ""
        if revision_requests:
            requests_text = " Requirements: " + "; ".join(revision_requests[:3])
            if len(revision_requests) > 3:
                requests_text += f" (+{len(revision_requests) - 3} more)"

        event = NotificationEvent(
            event_type=EventType.PHASE_COMPLETED,
            severity=EventSeverity.WARNING,
            session_id=session_id or "manual",
            title=f"Revision requested: {title}",
            summary=f"Revisions needed for your submission.{requests_text}",
            details={
                "submission_title": title,
                "revision_requests": revision_requests or [],
            },
        )
        await self.router.dispatch(event)

    async def decision_made(
        self,
        *,
        title: str,
        session_id: str = "",
        decision: str = "",
        summary: str = "",
    ) -> None:
        """Notify submitter of final accept/reject decision."""
        severity = EventSeverity.INFO if decision == "accept" else EventSeverity.WARNING

        event = NotificationEvent(
            event_type=EventType.SESSION_COMPLETED,
            severity=severity,
            session_id=session_id or "manual",
            title=f"Decision: {decision.replace('_', ' ').title()} — {title}",
            summary=summary or f"Your submission has been {decision.replace('_', ' ')}.",
            details={
                "submission_title": title,
                "decision": decision,
            },
        )
        await self.router.dispatch(event)
