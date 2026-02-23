"""
HeartbeatMonitor — periodic status pings during long-running sessions.

Runs as an asyncio background task alongside the research executor,
sending periodic heartbeat events so users know the agent is alive.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from apollobot.notifications.events import EventType, NotificationEvent
from apollobot.notifications.router import NotificationRouter

logger = logging.getLogger(__name__)


class HeartbeatMonitor:
    """Sends periodic heartbeat notifications during a research session."""

    def __init__(
        self,
        router: NotificationRouter,
        session_id: str,
        interval: float = 300,
    ) -> None:
        self.router = router
        self.session_id = session_id
        self.interval = interval
        self._task: asyncio.Task[None] | None = None
        self._current_phase: str = ""
        self._started_at: str = datetime.now(timezone.utc).isoformat()
        self._datasets_acquired: int = 0
        self._cost_so_far: float = 0.0

    def update_status(
        self,
        *,
        phase: str = "",
        datasets: int = 0,
        cost: float = 0.0,
    ) -> None:
        """Update the status fields included in heartbeat messages."""
        if phase:
            self._current_phase = phase
        self._datasets_acquired = datasets
        self._cost_so_far = cost

    async def start(self) -> None:
        """Start the heartbeat loop as a background task."""
        if self.interval <= 0:
            return
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Cancel the background task."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.interval)
                await self._send_heartbeat()
        except asyncio.CancelledError:
            return

    async def _send_heartbeat(self) -> None:
        now = datetime.now(timezone.utc)
        elapsed = (now - datetime.fromisoformat(self._started_at)).total_seconds()
        elapsed_min = int(elapsed / 60)

        summary_parts = [f"Running for {elapsed_min}m"]
        if self._current_phase:
            summary_parts.append(f"phase: {self._current_phase}")
        if self._cost_so_far > 0:
            summary_parts.append(f"cost: ${self._cost_so_far:.2f}")
        if self._datasets_acquired > 0:
            summary_parts.append(f"datasets: {self._datasets_acquired}")

        event = NotificationEvent(
            event_type=EventType.HEARTBEAT,
            session_id=self.session_id,
            phase=self._current_phase,
            title="Heartbeat",
            summary=" | ".join(summary_parts),
            details={
                "elapsed_seconds": int(elapsed),
                "current_phase": self._current_phase,
                "cost_usd": self._cost_so_far,
                "datasets_acquired": self._datasets_acquired,
            },
        )
        try:
            await self.router.dispatch(event)
        except Exception:
            logger.exception("Failed to send heartbeat")
