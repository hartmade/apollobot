"""
Runtime provenance logger — bridges the continuous runtime with ApolloBot's
provenance engine.

Records every brain decision, guardrails enforcement, action result, and
watchdog state change to an immutable audit trail. This gives the runtime
the same reproducibility guarantees as individual research sessions.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default provenance directory for the runtime
_DEFAULT_DIR = Path.home() / ".apollobot" / "runtime_provenance"


class RuntimeProvenanceLogger:
    """
    Lightweight provenance logger for the continuous runtime.

    Unlike ProvenanceEngine (which is per-session), this logs runtime-level
    events: brain decisions, guardrails enforcement, watchdog transitions, and
    session lifecycle events.

    Writes append-only JSONL files for fast streaming writes without
    re-serializing the full log on every tick.
    """

    def __init__(self, provenance_dir: str = "") -> None:
        self.provenance_dir = Path(provenance_dir) if provenance_dir else _DEFAULT_DIR
        self.provenance_dir.mkdir(parents=True, exist_ok=True)

        self._decisions_path = self.provenance_dir / "brain_decisions.jsonl"
        self._enforcements_path = self.provenance_dir / "collar_enforcements.jsonl"
        self._actions_path = self.provenance_dir / "action_results.jsonl"
        self._lifecycle_path = self.provenance_dir / "lifecycle.jsonl"

    def _append(self, path: Path, entry: dict[str, Any]) -> None:
        """Append a single JSON line to a log file."""
        entry["timestamp"] = datetime.now(timezone.utc).isoformat()
        try:
            with open(path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            logger.debug("Failed to write provenance entry to %s", path)

    def log_decision(
        self,
        tick: int,
        reasoning: str,
        actions: list[str],
        next_check_in: int,
        memory_updates: dict[str, str] | None = None,
    ) -> None:
        """Record a brain reasoning decision."""
        self._append(
            self._decisions_path,
            {
                "tick": tick,
                "reasoning": reasoning,
                "actions": actions,
                "next_check_in": next_check_in,
                "memory_updates": list((memory_updates or {}).keys()),
            },
        )

    def log_enforcement(
        self,
        tick: int,
        action_type: str,
        allowed: bool,
        reason: str = "",
        objective: str = "",
        domain: str = "",
    ) -> None:
        """Record a guardrails enforcement check."""
        self._append(
            self._enforcements_path,
            {
                "tick": tick,
                "action_type": action_type,
                "allowed": allowed,
                "reason": reason,
                "objective": objective[:200],
                "domain": domain,
            },
        )

    def log_action_result(
        self,
        tick: int,
        action_type: str,
        result: str,
        details: str = "",
        session_id: str = "",
        cost_usd: float = 0.0,
    ) -> None:
        """Record the result of an executed action."""
        self._append(
            self._actions_path,
            {
                "tick": tick,
                "action_type": action_type,
                "result": result,
                "details": details[:500],
                "session_id": session_id,
                "cost_usd": cost_usd,
            },
        )

    def log_lifecycle(self, event: str, data: dict[str, Any] | None = None) -> None:
        """Record a runtime lifecycle event (start, stop, watchdog, etc.)."""
        self._append(
            self._lifecycle_path,
            {
                "event": event,
                **(data or {}),
            },
        )

    def get_recent_entries(
        self, log_type: str = "lifecycle", limit: int = 50
    ) -> list[dict[str, Any]]:
        """Read recent entries from a provenance log. For debugging/reporting."""
        path_map = {
            "decisions": self._decisions_path,
            "enforcements": self._enforcements_path,
            "actions": self._actions_path,
            "lifecycle": self._lifecycle_path,
        }
        path = path_map.get(log_type)
        if not path or not path.exists():
            return []

        lines = path.read_text().strip().split("\n")
        entries = []
        for line in lines[-limit:]:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries
