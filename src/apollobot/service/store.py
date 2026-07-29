"""Small durable SQLite store for service state and ordered events.

The web platform remains the public system of record. This local store gives an
ApolloBot worker crash-safe ownership of job state and a resumable event stream
without introducing a required external queue for local or single-node use.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from apollobot.service.checkpoint import DurableCheckpoint, restore_checkpoint
from apollobot.service.models import ResearchNode, ServiceEvent, utc_now


class ServiceStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        checkpoint_url = os.getenv("APOLLOBOT_CHECKPOINT_URL", "")
        restored = restore_checkpoint(self.path, checkpoint_url)
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._migrate()
        self._checkpoint = DurableCheckpoint(
            self._db,
            self._lock,
            checkpoint_url,
            restore_ready=restored,
        )
        self._checkpoint.mark_dirty()

    def _changed(self) -> None:
        self._checkpoint.mark_dirty()

    def durability_ready(self) -> bool:
        ready = self._checkpoint.ready()
        if not ready:
            # A transient Worker/DO rollout race must not strand an otherwise
            # healthy container in a permanently degraded state.
            self._checkpoint.mark_dirty()
        return ready

    def durability_status(self) -> dict[str, object]:
        return self._checkpoint.status()

    def _migrate(self) -> None:
        statements = [
            """CREATE TABLE IF NOT EXISTS investigations (
                id TEXT PRIMARY KEY,
                user_id TEXT,
                title TEXT NOT NULL,
                objective TEXT NOT NULL,
                domain TEXT NOT NULL,
                mode TEXT NOT NULL,
                status TEXT NOT NULL,
                current_node TEXT NOT NULL,
                budget_usd REAL NOT NULL DEFAULT 0,
                cost_usd REAL NOT NULL DEFAULT 0,
                engine TEXT NOT NULL DEFAULT 'apollobot',
                model_id TEXT NOT NULL DEFAULT 'openai/gpt-oss-120b',
                model_provider_tag TEXT NOT NULL DEFAULT 'groq',
                check_json TEXT NOT NULL,
                mission_json TEXT,
                plan_json TEXT,
                result_json TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS nodes (
                investigation_id TEXT NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
                node_key TEXT NOT NULL,
                label TEXT NOT NULL,
                node_type TEXT NOT NULL,
                sequence INTEGER NOT NULL,
                status TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL,
                PRIMARY KEY (investigation_id, node_key)
            )""",
            """CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                investigation_id TEXT NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
                node_id TEXT,
                sequence INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                status TEXT NOT NULL,
                public_summary TEXT NOT NULL,
                artifact_refs TEXT NOT NULL DEFAULT '[]',
                cost_delta_usd REAL NOT NULL DEFAULT 0,
                occurred_at TEXT NOT NULL,
                data_json TEXT NOT NULL DEFAULT '{}',
                published_at TEXT,
                delivery_attempts INTEGER NOT NULL DEFAULT 0,
                UNIQUE(investigation_id, sequence)
            )""",
            """CREATE TABLE IF NOT EXISTS artifacts (
                id TEXT PRIMARY KEY,
                investigation_id TEXT NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
                artifact_type TEXT NOT NULL,
                label TEXT NOT NULL,
                path TEXT NOT NULL,
                media_type TEXT,
                size_bytes INTEGER,
                checksum_sha256 TEXT,
                storage_path TEXT,
                uploaded_at TEXT,
                upload_attempts INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                UNIQUE(investigation_id, path)
            )""",
            """CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                investigation_id TEXT NOT NULL UNIQUE
                    REFERENCES investigations(id) ON DELETE CASCADE,
                node_key TEXT NOT NULL DEFAULT 'design_experiment',
                title TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                method_json TEXT NOT NULL DEFAULT '{}',
                controls_json TEXT NOT NULL DEFAULT '[]',
                success_criteria_json TEXT NOT NULL DEFAULT '[]',
                failure_criteria_json TEXT NOT NULL DEFAULT '[]',
                preregistered_at TEXT,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS experiment_runs (
                id TEXT PRIMARY KEY,
                experiment_id TEXT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
                investigation_id TEXT NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
                attempt INTEGER NOT NULL,
                status TEXT NOT NULL,
                environment_digest TEXT,
                code_artifact_id TEXT REFERENCES artifacts(id) ON DELETE SET NULL,
                result_artifact_id TEXT REFERENCES artifacts(id) ON DELETE SET NULL,
                random_seed INTEGER,
                exit_code INTEGER,
                assertions_json TEXT NOT NULL DEFAULT '[]',
                metrics_json TEXT NOT NULL DEFAULT '{}',
                started_at TEXT,
                completed_at TEXT,
                created_at TEXT NOT NULL,
                UNIQUE(experiment_id, attempt)
            )""",
            """CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                investigation_id TEXT NOT NULL REFERENCES investigations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK(role IN ('researcher', 'apollobot')),
                phase TEXT NOT NULL CHECK(phase IN ('direction', 'experiment_plan')),
                body TEXT NOT NULL,
                revision INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL
            )""",
            "CREATE INDEX IF NOT EXISTS idx_events_resume ON events(investigation_id, sequence)",
            (
                "CREATE INDEX IF NOT EXISTS idx_messages_investigation "
                "ON messages(investigation_id, created_at)"
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_experiment_runs_investigation "
                "ON experiment_runs(investigation_id, attempt)"
            ),
            (
                "CREATE INDEX IF NOT EXISTS idx_investigations_status "
                "ON investigations(status, updated_at)"
            ),
        ]
        with self._lock, self._db:
            for statement in statements:
                self._db.execute(statement)
            columns = {row["name"] for row in self._db.execute("PRAGMA table_info(investigations)")}
            if "plan_json" not in columns:
                self._db.execute("ALTER TABLE investigations ADD COLUMN plan_json TEXT")
            if "result_json" not in columns:
                self._db.execute("ALTER TABLE investigations ADD COLUMN result_json TEXT")
            if "model_id" not in columns:
                self._db.execute(
                    "ALTER TABLE investigations ADD COLUMN model_id TEXT NOT NULL "
                    "DEFAULT 'openai/gpt-oss-120b'"
                )
            if "model_provider_tag" not in columns:
                self._db.execute(
                    "ALTER TABLE investigations ADD COLUMN model_provider_tag TEXT NOT NULL "
                    "DEFAULT 'groq'"
                )
            event_columns = {row["name"] for row in self._db.execute("PRAGMA table_info(events)")}
            if "published_at" not in event_columns:
                self._db.execute("ALTER TABLE events ADD COLUMN published_at TEXT")
            if "delivery_attempts" not in event_columns:
                self._db.execute(
                    "ALTER TABLE events ADD COLUMN delivery_attempts INTEGER NOT NULL DEFAULT 0"
                )
            artifact_columns = {
                row["name"] for row in self._db.execute("PRAGMA table_info(artifacts)")
            }
            if "storage_path" not in artifact_columns:
                self._db.execute("ALTER TABLE artifacts ADD COLUMN storage_path TEXT")
            if "uploaded_at" not in artifact_columns:
                self._db.execute("ALTER TABLE artifacts ADD COLUMN uploaded_at TEXT")
            if "upload_attempts" not in artifact_columns:
                self._db.execute(
                    "ALTER TABLE artifacts ADD COLUMN upload_attempts INTEGER NOT NULL DEFAULT 0"
                )

    def create_investigation(self, payload: dict[str, Any], nodes: list[ResearchNode]) -> None:
        now = utc_now()
        with self._lock, self._db:
            self._db.execute(
                """INSERT INTO investigations
                (id, user_id, title, objective, domain, mode, status, current_node,
                 budget_usd, cost_usd, engine, model_id, model_provider_tag,
                 check_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 'apollobot', ?, ?, ?, ?, ?)""",
                (
                    payload["id"],
                    payload.get("user_id"),
                    payload["title"],
                    payload["objective"],
                    payload["domain"],
                    payload["mode"],
                    payload["status"],
                    payload["current_node"],
                    payload["budget_usd"],
                    payload["model_id"],
                    payload["model_provider_tag"],
                    json.dumps(payload["check"]),
                    now,
                    now,
                ),
            )
            self._db.executemany(
                """INSERT INTO nodes
                (investigation_id, node_key, label, node_type, sequence, status,
                 summary, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [
                    (
                        payload["id"],
                        node.key,
                        node.label,
                        node.node_type,
                        node.sequence,
                        node.status,
                        node.summary,
                        now,
                    )
                    for node in nodes
                ],
            )
        self._changed()

    def ping(self) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1").fetchone()[0] == 1

    def operational_metrics(self) -> dict[str, Any]:
        with self._lock:
            statuses = {
                row["status"]: row["count"]
                for row in self._db.execute(
                    "SELECT status, COUNT(*) AS count FROM investigations GROUP BY status"
                )
            }
            unpublished_events = self._db.execute(
                "SELECT COUNT(*) FROM events WHERE published_at IS NULL"
            ).fetchone()[0]
            pending_artifacts = self._db.execute(
                "SELECT COUNT(*) FROM artifacts WHERE uploaded_at IS NULL AND upload_attempts < 20"
            ).fetchone()[0]
            failed_artifacts = self._db.execute(
                "SELECT COUNT(*) FROM artifacts WHERE uploaded_at IS NULL AND upload_attempts >= 20"
            ).fetchone()[0]
        return {
            "investigations": statuses,
            "unpublished_events": unpublished_events,
            "pending_artifacts": pending_artifacts,
            "failed_artifacts": failed_artifacts,
        }

    def get_investigation(self, investigation_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM investigations WHERE id = ?", (investigation_id,)
            ).fetchone()
            return self._investigation(row) if row else None

    def list_investigations(self, statuses: set[str]) -> list[dict[str, Any]]:
        if not statuses:
            return []
        placeholders = ",".join("?" for _ in statuses)
        with self._lock:
            rows = self._db.execute(
                f"SELECT * FROM investigations WHERE status IN ({placeholders}) "  # noqa: S608 - placeholders only
                "ORDER BY updated_at",
                tuple(sorted(statuses)),
            )
            return [self._investigation(row) for row in rows]

    def snapshot(self, investigation_id: str, after: int = 0) -> dict[str, Any] | None:
        investigation = self.get_investigation(investigation_id)
        if not investigation:
            return None
        with self._lock:
            nodes = [
                dict(row)
                for row in self._db.execute(
                    "SELECT node_key AS key, label, node_type, sequence, status, summary "
                    "FROM nodes WHERE investigation_id = ? ORDER BY sequence",
                    (investigation_id,),
                )
            ]
            events = [
                self._event(row)
                for row in self._db.execute(
                    "SELECT * FROM events WHERE investigation_id = ? AND sequence > ? "
                    "ORDER BY sequence",
                    (investigation_id, after),
                )
            ]
            artifacts = [
                dict(row)
                for row in self._db.execute(
                    "SELECT id, artifact_type, label, path, media_type, size_bytes, "
                    "checksum_sha256, storage_path, uploaded_at, created_at FROM artifacts "
                    "WHERE investigation_id = ? ORDER BY created_at",
                    (investigation_id,),
                )
            ]
            experiments = []
            for row in self._db.execute(
                "SELECT * FROM experiments WHERE investigation_id = ? ORDER BY created_at",
                (investigation_id,),
            ):
                experiment = self._experiment(row)
                experiment["runs"] = [
                    self._experiment_run(run)
                    for run in self._db.execute(
                        "SELECT * FROM experiment_runs WHERE experiment_id = ? ORDER BY attempt",
                        (experiment["id"],),
                    )
                ]
                experiments.append(experiment)
            messages = [
                dict(row)
                for row in self._db.execute(
                    "SELECT id, role, phase, body, revision, created_at FROM messages "
                    "WHERE investigation_id = ? ORDER BY created_at, rowid",
                    (investigation_id,),
                )
            ]
        return {
            "investigation": investigation,
            "nodes": nodes,
            "events": events,
            "artifacts": artifacts,
            "experiments": experiments,
            "messages": messages,
            "check": investigation.pop("check"),
        }

    def append_message(
        self,
        investigation_id: str,
        role: str,
        phase: str,
        body: str,
        revision: int,
    ) -> dict[str, Any]:
        if role not in {"researcher", "apollobot"}:
            raise ValueError("Unsupported investigation message role")
        if phase not in {"direction", "experiment_plan"}:
            raise ValueError("Unsupported investigation message phase")
        message = {
            "id": str(uuid4()),
            "investigation_id": investigation_id,
            "role": role,
            "phase": phase,
            "body": body,
            "revision": max(0, int(revision)),
            "created_at": utc_now(),
        }
        with self._lock, self._db:
            self._db.execute(
                """INSERT INTO messages
                (id, investigation_id, role, phase, body, revision, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    message["id"],
                    investigation_id,
                    role,
                    phase,
                    body,
                    message["revision"],
                    message["created_at"],
                ),
            )
        self._changed()
        return message

    def list_messages(self, investigation_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [
                dict(row)
                for row in self._db.execute(
                    "SELECT id, role, phase, body, revision, created_at FROM messages "
                    "WHERE investigation_id = ? ORDER BY created_at, rowid",
                    (investigation_id,),
                )
            ]

    def update_investigation(self, investigation_id: str, **fields: object) -> None:
        allowed = {
            "status",
            "current_node",
            "cost_usd",
            "mission_json",
            "plan_json",
            "result_json",
            "error",
            "completed_at",
        }
        values = {key: value for key, value in fields.items() if key in allowed}
        if not values:
            return
        values["updated_at"] = utc_now()
        columns = ", ".join(f"{key} = ?" for key in values)
        with self._lock, self._db:
            self._db.execute(
                f"UPDATE investigations SET {columns} WHERE id = ?",  # noqa: S608 - columns are allowlisted
                (*values.values(), investigation_id),
            )
        self._changed()

    def update_node(
        self, investigation_id: str, node_key: str, status: str, summary: str | None = None
    ) -> None:
        with self._lock, self._db:
            if summary is None:
                self._db.execute(
                    "UPDATE nodes SET status = ?, updated_at = ? "
                    "WHERE investigation_id = ? AND node_key = ?",
                    (status, utc_now(), investigation_id, node_key),
                )
            else:
                self._db.execute(
                    "UPDATE nodes SET status = ?, summary = ?, updated_at = ? "
                    "WHERE investigation_id = ? AND node_key = ?",
                    (status, summary, utc_now(), investigation_id, node_key),
                )
        self._changed()

    def append_event(self, event: ServiceEvent) -> ServiceEvent:
        with self._lock, self._db:
            next_sequence = self._db.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM events WHERE investigation_id = ?",
                (event.investigation_id,),
            ).fetchone()[0]
            event.sequence = int(next_sequence)
            self._db.execute(
                """INSERT INTO events
                (id, investigation_id, node_id, sequence, event_type, status,
                 public_summary, artifact_refs, cost_delta_usd, occurred_at, data_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.id,
                    event.investigation_id,
                    event.node_id,
                    event.sequence,
                    event.event_type,
                    event.status,
                    event.public_summary,
                    json.dumps(event.artifact_refs),
                    event.cost_delta_usd,
                    event.occurred_at,
                    json.dumps(event.data),
                ),
            )
        self._changed()
        return event

    def add_artifact(self, investigation_id: str, artifact: dict[str, Any]) -> dict[str, Any]:
        with self._lock, self._db:
            self._db.execute(
                """INSERT OR IGNORE INTO artifacts
                (id, investigation_id, artifact_type, label, path, media_type,
                 size_bytes, checksum_sha256, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    artifact["id"],
                    investigation_id,
                    artifact["artifact_type"],
                    artifact["label"],
                    artifact["path"],
                    artifact.get("media_type"),
                    artifact.get("size_bytes"),
                    artifact.get("checksum_sha256"),
                    utc_now(),
                ),
            )
            row = self._db.execute(
                "SELECT id, investigation_id, artifact_type, label, path, media_type, "
                "size_bytes, checksum_sha256, storage_path, uploaded_at, created_at "
                "FROM artifacts WHERE investigation_id = ? AND path = ?",
                (investigation_id, artifact["path"]),
            ).fetchone()
            if not row:
                raise RuntimeError("Artifact insert did not persist")
            result = dict(row)
        self._changed()
        return result

    def upsert_experiment(self, investigation_id: str, experiment: dict[str, Any]) -> None:
        now = utc_now()
        created_at = str(experiment.get("created_at") or now)
        with self._lock, self._db:
            self._db.execute(
                """INSERT INTO experiments
                (id, investigation_id, node_key, title, hypothesis, method_json,
                 controls_json, success_criteria_json, failure_criteria_json,
                 preregistered_at, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                  node_key = excluded.node_key,
                  title = excluded.title,
                  hypothesis = excluded.hypothesis,
                  method_json = excluded.method_json,
                  controls_json = excluded.controls_json,
                  success_criteria_json = excluded.success_criteria_json,
                  failure_criteria_json = excluded.failure_criteria_json,
                  preregistered_at = excluded.preregistered_at,
                  status = excluded.status,
                  updated_at = excluded.updated_at""",
                (
                    experiment["id"],
                    investigation_id,
                    experiment.get("node_key", "design_experiment"),
                    experiment["title"],
                    experiment["hypothesis"],
                    json.dumps(experiment.get("method", {})),
                    json.dumps(experiment.get("controls", [])),
                    json.dumps(experiment.get("success_criteria", [])),
                    json.dumps(experiment.get("failure_criteria", [])),
                    experiment.get("preregistered_at"),
                    experiment.get("status", "draft"),
                    created_at,
                    now,
                ),
            )
        self._changed()

    def update_experiment(self, investigation_id: str, **fields: object) -> None:
        allowed = {"status", "preregistered_at"}
        values = {key: value for key, value in fields.items() if key in allowed}
        if not values:
            return
        values["updated_at"] = utc_now()
        columns = ", ".join(f"{key} = ?" for key in values)
        with self._lock, self._db:
            self._db.execute(
                f"UPDATE experiments SET {columns} WHERE investigation_id = ?",  # noqa: S608 - columns are allowlisted
                (*values.values(), investigation_id),
            )
        self._changed()

    def create_experiment_run(
        self,
        investigation_id: str,
        run_id: str,
        *,
        environment_digest: str | None = None,
        random_seed: int | None = None,
    ) -> dict[str, Any]:
        now = utc_now()
        with self._lock, self._db:
            experiment = self._db.execute(
                "SELECT id FROM experiments WHERE investigation_id = ?",
                (investigation_id,),
            ).fetchone()
            if not experiment:
                raise ValueError("The investigation has no preregistered experiment")
            attempt = int(
                self._db.execute(
                    "SELECT COALESCE(MAX(attempt), 0) + 1 FROM experiment_runs "
                    "WHERE experiment_id = ?",
                    (experiment["id"],),
                ).fetchone()[0]
            )
            self._db.execute(
                """INSERT INTO experiment_runs
                (id, experiment_id, investigation_id, attempt, status,
                 environment_digest, random_seed, created_at)
                VALUES (?, ?, ?, ?, 'queued', ?, ?, ?)""",
                (
                    run_id,
                    experiment["id"],
                    investigation_id,
                    attempt,
                    environment_digest,
                    random_seed,
                    now,
                ),
            )
            row = self._db.execute(
                "SELECT * FROM experiment_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if not row:
                raise RuntimeError("Experiment run insert did not persist")
        self._changed()
        return self._experiment_run(row)

    def current_experiment_run(self, investigation_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM experiment_runs WHERE investigation_id = ? "
                "ORDER BY attempt DESC LIMIT 1",
                (investigation_id,),
            ).fetchone()
        return self._experiment_run(row) if row else None

    def update_experiment_run(self, run_id: str, **fields: object) -> None:
        json_fields = {"assertions": "assertions_json", "metrics": "metrics_json"}
        allowed = {
            "status",
            "environment_digest",
            "code_artifact_id",
            "result_artifact_id",
            "random_seed",
            "exit_code",
            "started_at",
            "completed_at",
        }
        values: dict[str, object] = {key: value for key, value in fields.items() if key in allowed}
        for key, column in json_fields.items():
            if key in fields:
                values[column] = json.dumps(fields[key])
        if not values:
            return
        columns = ", ".join(f"{key} = ?" for key in values)
        with self._lock, self._db:
            self._db.execute(
                f"UPDATE experiment_runs SET {columns} WHERE id = ?",  # noqa: S608 - columns are allowlisted
                (*values.values(), run_id),
            )
        self._changed()

    def get_artifact(self, investigation_id: str, artifact_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT id, investigation_id, artifact_type, label, path, media_type, "
                "size_bytes, checksum_sha256, storage_path, uploaded_at, created_at "
                "FROM artifacts "
                "WHERE investigation_id = ? AND id = ?",
                (investigation_id, artifact_id),
            ).fetchone()
            return dict(row) if row else None

    def pending_artifacts(self, limit: int = 25) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, investigation_id, artifact_type, label, path, media_type, "
                "size_bytes, checksum_sha256, upload_attempts FROM artifacts "
                "WHERE uploaded_at IS NULL AND upload_attempts < 20 "
                "ORDER BY created_at LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in rows]

    def mark_artifact_uploaded(self, artifact_id: str, storage_path: str) -> None:
        with self._lock, self._db:
            self._db.execute(
                "UPDATE artifacts SET storage_path = ?, uploaded_at = ?, "
                "upload_attempts = upload_attempts + 1 WHERE id = ?",
                (storage_path, utc_now(), artifact_id),
            )
        self._changed()

    def mark_artifact_attempt(self, artifact_id: str) -> None:
        with self._lock, self._db:
            self._db.execute(
                "UPDATE artifacts SET upload_attempts = upload_attempts + 1 WHERE id = ?",
                (artifact_id,),
            )
        self._changed()

    def pending_events(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM events WHERE published_at IS NULL "
                "ORDER BY occurred_at, sequence LIMIT ?",
                (limit,),
            )
            return [self._event(row) for row in rows]

    def mark_event_published(self, event_id: str) -> None:
        with self._lock, self._db:
            self._db.execute(
                "UPDATE events SET published_at = ?, delivery_attempts = delivery_attempts + 1 "
                "WHERE id = ?",
                (utc_now(), event_id),
            )
        self._changed()

    def mark_event_attempt(self, event_id: str) -> None:
        with self._lock, self._db:
            self._db.execute(
                "UPDATE events SET delivery_attempts = delivery_attempts + 1 WHERE id = ?",
                (event_id,),
            )
        self._changed()

    def close(self) -> None:
        self._checkpoint.close()
        with self._lock:
            self._db.close()

    @staticmethod
    def _investigation(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["check"] = json.loads(value.pop("check_json"))
        if value.get("mission_json"):
            value["mission"] = json.loads(value["mission_json"])
        if value.get("plan_json"):
            value["plan"] = json.loads(value["plan_json"])
        if value.get("result_json"):
            value["result"] = json.loads(value["result_json"])
        value.pop("mission_json", None)
        value.pop("plan_json", None)
        value.pop("result_json", None)
        return value

    @staticmethod
    def _event(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["artifact_refs"] = json.loads(value.pop("artifact_refs"))
        value["data"] = json.loads(value.pop("data_json"))
        return value

    @staticmethod
    def _experiment(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        for field in ("method", "controls", "success_criteria", "failure_criteria"):
            value[field] = json.loads(value.pop(f"{field}_json"))
        return value

    @staticmethod
    def _experiment_run(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["assertions"] = json.loads(value.pop("assertions_json"))
        value["metrics"] = json.loads(value.pop("metrics_json"))
        return value
