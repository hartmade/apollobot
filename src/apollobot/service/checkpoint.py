"""Durable SQLite checkpoints for ephemeral container filesystems."""

from __future__ import annotations

import gzip
import logging
import sqlite3
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


def restore_checkpoint(path: Path, endpoint: str) -> bool:
    """Restore the latest compressed database snapshot before SQLite opens it."""
    if not endpoint:
        return True
    try:
        with urllib.request.urlopen(endpoint, timeout=30) as response:  # noqa: S310
            payload = response.read(32 * 1024 * 1024 + 1)
        if len(payload) > 32 * 1024 * 1024:
            raise RuntimeError("Durable checkpoint exceeds the 32 MiB safety limit")
        database = gzip.decompress(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f"{path.suffix}.restore")
        temporary.write_bytes(database)
        temporary.replace(path)
        return True
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return True
        logger.error("Checkpoint restore failed with HTTP %d", error.code)
    except Exception as error:  # pragma: no cover - defensive startup logging
        logger.error("Checkpoint restore failed: %s", error)
    return False


class DurableCheckpoint:
    """Debounced, crash-resistant snapshots of a live SQLite connection."""

    def __init__(
        self,
        database: sqlite3.Connection,
        database_lock: threading.RLock,
        endpoint: str,
        *,
        restore_ready: bool,
    ) -> None:
        self._database = database
        self._database_lock = database_lock
        self._endpoint = endpoint
        self._restore_ready = restore_ready
        self._last_error: str | None = None
        self._condition = threading.Condition()
        self._dirty = False
        self._flushing = False
        self._stopping = False
        self._thread: threading.Thread | None = None
        if endpoint:
            self._thread = threading.Thread(
                target=self._run,
                name="apollobot-checkpoint",
                daemon=True,
            )
            self._thread.start()

    def mark_dirty(self) -> None:
        if not self._endpoint:
            return
        with self._condition:
            self._dirty = True
            self._condition.notify_all()

    def ready(self) -> bool:
        return self._restore_ready and self._last_error is None

    def status(self) -> dict[str, object]:
        return {
            "ready": self.ready(),
            "restore_ready": self._restore_ready,
            "last_error": self._last_error,
        }

    def close(self, timeout: float = 20.0) -> None:
        if not self._thread:
            return
        with self._condition:
            self._dirty = True
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            logger.error("Durable checkpoint did not finish before shutdown")

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._dirty and not self._stopping:
                    self._condition.wait()
                if self._dirty and not self._stopping:
                    self._condition.wait(timeout=0.5)
                should_flush = self._dirty
                self._dirty = False
                self._flushing = should_flush
            if should_flush:
                self._flush()
                with self._condition:
                    self._flushing = False
                    self._condition.notify_all()
            with self._condition:
                if self._stopping and not self._dirty:
                    return

    def _flush(self) -> None:
        try:
            with tempfile.TemporaryDirectory(prefix="apollo-checkpoint-") as directory:
                snapshot = Path(directory) / "service.db"
                target = sqlite3.connect(snapshot)
                try:
                    with self._database_lock:
                        self._database.backup(target)
                finally:
                    target.close()
                payload = gzip.compress(snapshot.read_bytes(), compresslevel=6)
            request = urllib.request.Request(  # noqa: S310 - endpoint is deployment-controlled
                self._endpoint,
                data=payload,
                method="PUT",
                headers={"content-type": "application/gzip"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                if response.status not in {200, 201, 204}:
                    raise RuntimeError(f"Checkpoint returned HTTP {response.status}")
            self._restore_ready = True
            self._last_error = None
        except Exception as error:  # pragma: no cover - integration behavior
            suffix = f":{error.code}" if isinstance(error, urllib.error.HTTPError) else ""
            self._last_error = f"{type(error).__name__}{suffix}"
            logger.error("Checkpoint upload failed: %s", error)
