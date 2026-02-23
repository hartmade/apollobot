"""
Notification events — the data flowing through the notification system.

Defines event types, severity levels, and the NotificationEvent model
that channels consume.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    SESSION_STARTED = "session_started"
    PHASE_STARTED = "phase_started"
    PHASE_COMPLETED = "phase_completed"
    PHASE_FAILED = "phase_failed"
    CHECKPOINT_APPROVAL = "checkpoint_approval"
    FINDING = "finding"
    BUDGET_WARNING = "budget_warning"
    SESSION_COMPLETED = "session_completed"
    SESSION_FAILED = "session_failed"
    HEARTBEAT = "heartbeat"


class EventSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationEvent(BaseModel):
    """A single notification event dispatched to channels."""

    event_type: EventType
    severity: EventSeverity = EventSeverity.INFO
    session_id: str
    phase: str = ""
    title: str
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)
    requires_response: bool = False
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
