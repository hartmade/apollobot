"""
SQLite persistence for the continuous runtime.

Stores brain memory, action/decision history, session registry,
and spend tracking for guardrails enforcement.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from apollobot.core import APOLLO_HOME
from apollobot.runtime.types import ActionRecord, DecisionRecord, SessionSummary


class RunnerStorage:
    """SQLite-backed persistence for the continuous runtime."""

    def __init__(self, db_path: str = "") -> None:
        if not db_path:
            db_path = str(APOLLO_HOME / "runtime.db")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS brain_memory (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS action_history (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                tick       INTEGER NOT NULL,
                action_type TEXT NOT NULL,
                objective  TEXT,
                result     TEXT,
                details    TEXT,
                timestamp  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS decision_history (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                tick          INTEGER NOT NULL,
                reasoning     TEXT NOT NULL,
                actions       TEXT,
                next_check_in INTEGER,
                timestamp     TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_registry (
                session_id   TEXT PRIMARY KEY,
                objective    TEXT NOT NULL,
                domain       TEXT NOT NULL,
                mode         TEXT NOT NULL,
                phase        TEXT NOT NULL,
                cost_usd     REAL DEFAULT 0.0,
                started_at   TEXT NOT NULL,
                completed_at TEXT,
                key_findings TEXT,
                translation_score REAL DEFAULT 0.0
            );

            CREATE TABLE IF NOT EXISTS spend_history (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                amount    REAL NOT NULL,
                category  TEXT NOT NULL,
                timestamp TEXT NOT NULL
            );
        """)
        self.db.commit()

    # ------------------------------------------------------------------
    # Brain memory
    # ------------------------------------------------------------------

    def load_memory(self) -> dict[str, str]:
        rows = self.db.execute("SELECT key, value FROM brain_memory").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def save_memory(self, memory: dict[str, str]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        for key, value in memory.items():
            self.db.execute(
                "INSERT OR REPLACE INTO brain_memory (key, value, updated_at) VALUES (?, ?, ?)",
                (key, value, now),
            )
        self.db.commit()

    # ------------------------------------------------------------------
    # Action history
    # ------------------------------------------------------------------

    def record_action(self, record: ActionRecord) -> None:
        self.db.execute(
            "INSERT INTO action_history (tick, action_type, objective, result, details, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                record.tick,
                record.action_type,
                record.objective,
                record.result,
                record.details,
                record.timestamp,
            ),
        )
        self.db.commit()

    def recent_actions(self, limit: int = 10) -> list[ActionRecord]:
        rows = self.db.execute(
            "SELECT * FROM action_history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            ActionRecord(
                tick=r["tick"],
                action_type=r["action_type"],
                objective=r["objective"] or "",
                result=r["result"] or "",
                details=r["details"] or "",
                timestamp=r["timestamp"],
            )
            for r in reversed(rows)
        ]

    def get_actions_since(
        self,
        since: str,
        action_type: str = "",
        limit: int = 50,
    ) -> list[ActionRecord]:
        """Return actions recorded after *since* (ISO-8601 timestamp).

        Optionally filter by *action_type* (e.g. "scan_literature").
        Results are returned in chronological order.
        """
        clauses = ["timestamp >= ?"]
        params: list[object] = [since]
        if action_type:
            clauses.append("action_type = ?")
            params.append(action_type)
        where = " AND ".join(clauses)
        params.append(limit)
        rows = self.db.execute(
            f"SELECT * FROM action_history WHERE {where} "  # noqa: S608
            "ORDER BY id DESC LIMIT ?",
            params,
        ).fetchall()
        return [
            ActionRecord(
                tick=r["tick"],
                action_type=r["action_type"],
                objective=r["objective"] or "",
                result=r["result"] or "",
                details=r["details"] or "",
                timestamp=r["timestamp"],
            )
            for r in reversed(rows)
        ]

    def get_decisions_since(
        self,
        since: str,
        limit: int = 50,
    ) -> list[DecisionRecord]:
        """Return decisions recorded after *since* (ISO-8601 timestamp)."""
        rows = self.db.execute(
            "SELECT * FROM decision_history WHERE timestamp >= ? ORDER BY id DESC LIMIT ?",
            (since, limit),
        ).fetchall()
        return [
            DecisionRecord(
                tick=r["tick"],
                reasoning=r["reasoning"],
                actions=json.loads(r["actions"]) if r["actions"] else [],
                next_check_in=r["next_check_in"] or 300,
                timestamp=r["timestamp"],
            )
            for r in reversed(rows)
        ]

    # ------------------------------------------------------------------
    # Decision history
    # ------------------------------------------------------------------

    def record_decision(self, record: DecisionRecord) -> None:
        self.db.execute(
            "INSERT INTO decision_history (tick, reasoning, actions, next_check_in, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                record.tick,
                record.reasoning,
                json.dumps(record.actions),
                record.next_check_in,
                record.timestamp,
            ),
        )
        self.db.commit()

    def recent_decisions(self, limit: int = 5) -> list[DecisionRecord]:
        rows = self.db.execute(
            "SELECT * FROM decision_history ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            DecisionRecord(
                tick=r["tick"],
                reasoning=r["reasoning"],
                actions=json.loads(r["actions"]) if r["actions"] else [],
                next_check_in=r["next_check_in"] or 300,
                timestamp=r["timestamp"],
            )
            for r in reversed(rows)
        ]

    # ------------------------------------------------------------------
    # Session registry
    # ------------------------------------------------------------------

    def register_session(self, summary: SessionSummary) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO session_registry "
            "(session_id, objective, domain, mode, phase, cost_usd, "
            " started_at, completed_at, key_findings, translation_score) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                summary.session_id,
                summary.objective,
                summary.domain,
                summary.mode,
                summary.phase,
                summary.cost_usd,
                summary.started_at,
                summary.completed_at,
                json.dumps(summary.key_findings),
                summary.translation_score,
            ),
        )
        self.db.commit()

    def update_session(self, session_id: str, **kwargs: object) -> None:
        if not kwargs:
            return
        sets = ", ".join(f"{k} = ?" for k in kwargs)
        vals = list(kwargs.values())
        vals.append(session_id)
        self.db.execute(
            f"UPDATE session_registry SET {sets} WHERE session_id = ?",
            vals,  # noqa: S608
        )
        self.db.commit()

    def get_active_sessions(self) -> list[SessionSummary]:
        return self._query_sessions(
            "SELECT * FROM session_registry WHERE phase NOT IN ('complete', 'failed', 'cancelled') "
            "ORDER BY started_at DESC"
        )

    def get_completed_sessions(self, limit: int = 20) -> list[SessionSummary]:
        return self._query_sessions(
            "SELECT * FROM session_registry WHERE phase = 'complete' "
            "ORDER BY completed_at DESC LIMIT ?",
            (limit,),
        )

    def get_failed_sessions(self, limit: int = 10) -> list[SessionSummary]:
        return self._query_sessions(
            "SELECT * FROM session_registry WHERE phase IN ('failed', 'cancelled') "
            "ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )

    def _query_sessions(self, query: str, params: tuple[object, ...] = ()) -> list[SessionSummary]:
        rows = self.db.execute(query, params).fetchall()
        return [
            SessionSummary(
                session_id=r["session_id"],
                objective=r["objective"],
                domain=r["domain"],
                mode=r["mode"],
                phase=r["phase"],
                cost_usd=r["cost_usd"] or 0.0,
                started_at=r["started_at"] or "",
                completed_at=r["completed_at"] or "",
                key_findings=json.loads(r["key_findings"]) if r["key_findings"] else [],
                translation_score=r["translation_score"] or 0.0,
            )
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Spend tracking (for guardrails enforcement)
    # ------------------------------------------------------------------

    def record_spend(self, amount: float, category: str = "session") -> None:
        self.db.execute(
            "INSERT INTO spend_history (amount, category, timestamp) VALUES (?, ?, ?)",
            (amount, category, datetime.now(timezone.utc).isoformat()),
        )
        self.db.commit()

    def daily_spend(self) -> float:
        """Total spend in the rolling 24h window."""
        cutoff = datetime.now(timezone.utc).isoformat()[:10]  # today
        row = self.db.execute(
            "SELECT COALESCE(SUM(amount), 0) as total FROM spend_history WHERE timestamp >= ?",
            (cutoff + "T00:00:00",),
        ).fetchone()
        return float(row["total"]) if row else 0.0

    def sessions_started_today(self) -> int:
        cutoff = datetime.now(timezone.utc).isoformat()[:10]
        row = self.db.execute(
            "SELECT COUNT(*) as cnt FROM session_registry WHERE started_at >= ?",
            (cutoff + "T00:00:00",),
        ).fetchone()
        return int(row["cnt"]) if row else 0

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        self.db.close()
