"""Session registry — manages live sessions across MCP tool calls."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from apollobot.agents.executor import CheckpointHandler
from apollobot.agents.orchestrator import Orchestrator
from apollobot.agents.planner import ResearchPlan, ResearchPlanner
from apollobot.core import ApolloConfig, APOLLO_SESSIONS_DIR, load_config
from apollobot.core.mission import Mission
from apollobot.core.provenance import ProvenanceEngine
from apollobot.core.session import Session


@dataclass
class ActiveSession:
    """Holds all state for an in-progress session."""

    session_id: str
    mission: Mission
    session: Session
    orchestrator: Orchestrator
    provenance: ProvenanceEngine
    plan: Optional[ResearchPlan] = None

    # Checkpoint approval state
    pending_checkpoint: Optional[dict[str, Any]] = None
    checkpoint_event: asyncio.Event = field(default_factory=asyncio.Event)
    checkpoint_approved: bool = False


class SessionRegistry:
    """Thread-safe registry of active MCP sessions."""

    def __init__(self, config: ApolloConfig | None = None) -> None:
        self.config = config or load_config()
        self._sessions: dict[str, ActiveSession] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        mission: Mission,
        checkpoint_handler: CheckpointHandler | None = None,
    ) -> ActiveSession:
        """Create a new active session from a mission."""
        orchestrator = Orchestrator(config=self.config, interactive=False)
        if checkpoint_handler:
            orchestrator.checkpoint = checkpoint_handler

        session = Session(mission=mission)
        session.mission.metadata["output_dir"] = self.config.output_dir
        session.init_directories()

        provenance = ProvenanceEngine(session.session_dir)
        provenance.log_event("session_started", {
            "mission_id": mission.id,
            "objective": mission.objective,
            "mode": mission.mode.value,
            "domain": mission.domain,
            "source": "mcp_server",
        })

        active = ActiveSession(
            session_id=mission.id,
            mission=mission,
            session=session,
            orchestrator=orchestrator,
            provenance=provenance,
        )

        async with self._lock:
            self._sessions[mission.id] = active

        return active

    async def get(self, session_id: str) -> ActiveSession:
        """Get an active session by ID. Raises KeyError if not found."""
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Session not found: {session_id}")
            return self._sessions[session_id]

    async def list_active(self) -> list[dict[str, Any]]:
        """List all active sessions as summary dicts."""
        async with self._lock:
            return [
                {
                    "session_id": s.session_id,
                    "objective": s.mission.objective,
                    "mode": s.mission.mode.value,
                    "domain": s.mission.domain,
                    "current_phase": s.session.current_phase.value,
                    "cost": s.session.cost.total_cost,
                    "active": True,
                }
                for s in self._sessions.values()
            ]

    async def load_from_disk(self, session_id: str) -> ActiveSession:
        """Load a historical session from disk into active memory."""
        sessions_dir = Path(self.config.output_dir) if self.config.output_dir else APOLLO_SESSIONS_DIR
        session_dir = sessions_dir / session_id

        session = Session.load_state(session_dir)
        mission = session.mission

        orchestrator = Orchestrator(config=self.config, interactive=False)
        provenance = ProvenanceEngine(session.session_dir)

        active = ActiveSession(
            session_id=mission.id,
            mission=mission,
            session=session,
            orchestrator=orchestrator,
            provenance=provenance,
        )

        async with self._lock:
            self._sessions[mission.id] = active

        return active

    async def remove(self, session_id: str) -> None:
        """Remove a session from active memory."""
        async with self._lock:
            self._sessions.pop(session_id, None)

    def list_historical(self) -> list[dict[str, Any]]:
        """Scan disk for historical sessions."""
        sessions_dir = Path(self.config.output_dir) if self.config.output_dir else APOLLO_SESSIONS_DIR
        results = []
        if not sessions_dir.exists():
            return results

        for d in sorted(sessions_dir.iterdir(), reverse=True):
            state_file = d / "session_state.json"
            if d.is_dir() and state_file.exists():
                try:
                    session = Session.load_state(d)
                    results.append({
                        "session_id": session.mission.id,
                        "objective": session.mission.objective,
                        "mode": session.mission.mode.value,
                        "domain": session.mission.domain,
                        "current_phase": session.current_phase.value,
                        "cost": session.cost.total_cost,
                        "active": False,
                    })
                except Exception:
                    results.append({
                        "session_id": d.name,
                        "error": "corrupted",
                        "active": False,
                    })

        return results
