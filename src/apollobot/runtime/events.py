"""
Event emitter for the ResearchRunner.

Inspired by OpenCat's EventEmitter pattern — lets external code subscribe
to lifecycle events without coupling to the runner internals.

Subscribers are fire-and-forget: a failing callback is logged but never
crashes the runner.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class RunnerEventType(str, Enum):
    TICK_START = "tick_start"
    TICK_COMPLETE = "tick_complete"
    TICK_FAILED = "tick_failed"
    ACTION_EXECUTED = "action_executed"
    ACTION_BLOCKED = "action_blocked"
    SESSION_STARTED = "session_started"
    SESSION_COMPLETED = "session_completed"
    SESSION_FAILED = "session_failed"
    WATCHDOG_OPENED = "watchdog_opened"
    WATCHDOG_CLOSED = "watchdog_closed"
    BUDGET_WARNING = "budget_warning"
    RUNTIME_STARTED = "runtime_started"
    RUNTIME_STOPPED = "runtime_stopped"


@dataclass
class RunnerEvent:
    """Payload delivered to every matching subscriber."""

    event_type: RunnerEventType
    tick: int
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    data: dict[str, Any] = field(default_factory=dict)


class RunnerEventEmitter:
    """
    Simple async event bus.

    * ``subscribe(event_type, cb)`` — pass *None* as event_type to receive
      every event.
    * ``unsubscribe(cb)`` — removes the callback from all lists.
    * ``emit(event)`` — invokes every matching callback.  Exceptions are
      caught and logged so one bad subscriber can never break the runner.
    """

    def __init__(self) -> None:
        # None key = wildcard subscribers
        self._listeners: dict[RunnerEventType | None, list[Callable]] = {}

    def subscribe(
        self,
        event_type: RunnerEventType | None,
        callback: Callable,
    ) -> None:
        self._listeners.setdefault(event_type, []).append(callback)

    def unsubscribe(self, callback: Callable) -> None:
        for lst in self._listeners.values():
            try:
                lst.remove(callback)
            except ValueError:
                pass

    async def emit(self, event: RunnerEvent) -> None:
        callbacks = list(self._listeners.get(event.event_type, []))
        callbacks += list(self._listeners.get(None, []))
        for cb in callbacks:
            try:
                result = cb(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception(
                    "Event subscriber %r failed on %s",
                    cb,
                    event.event_type.value,
                )
