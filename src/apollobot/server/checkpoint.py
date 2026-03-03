"""MCP checkpoint handler — non-interactive checkpoint via asyncio events."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from apollobot.agents.executor import CheckpointHandler

if TYPE_CHECKING:
    from apollobot.server.registry import ActiveSession

logger = logging.getLogger(__name__)


class MCPCheckpointHandler(CheckpointHandler):
    """
    Checkpoint handler for MCP server sessions.

    Instead of prompting a terminal user, stores pending checkpoint state
    on the ActiveSession and waits for the AI to call approve_checkpoint.
    """

    def __init__(self, get_active_session: Any = None) -> None:
        self._get_active_session = get_active_session
        self._progress_callbacks: list[Any] = []

    def set_session_getter(self, getter: Any) -> None:
        self._get_active_session = getter

    def add_progress_callback(self, callback: Any) -> None:
        self._progress_callbacks.append(callback)

    async def request_approval(self, phase: str, summary: str) -> bool:
        """Store pending checkpoint and wait for approve_checkpoint tool call."""
        if not self._get_active_session:
            # No session getter configured — auto-approve
            return True

        active: ActiveSession = self._get_active_session()
        if not active:
            return True

        # Reset the event for a new checkpoint
        active.checkpoint_event = asyncio.Event()
        active.checkpoint_approved = False
        active.pending_checkpoint = {
            "phase": phase,
            "summary": summary,
        }

        logger.info("Checkpoint pending: phase=%s summary=%s", phase, summary)

        # Wait for the AI to approve via approve_checkpoint tool
        await active.checkpoint_event.wait()

        approved = active.checkpoint_approved
        active.pending_checkpoint = None
        return approved

    async def notify(self, phase: str, summary: str) -> None:
        """Log progress notification."""
        logger.info("Phase progress: phase=%s summary=%s", phase, summary)
        for callback in self._progress_callbacks:
            try:
                await callback(phase, summary)
            except Exception:
                pass
