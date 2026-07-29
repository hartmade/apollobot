"""
PID file management for ApolloBot runtime.

Prevents double-launching and enables process detection by writing the
current process ID to a lock file with advisory file locking.
"""

from __future__ import annotations

import fcntl
import logging
import os
from pathlib import Path

from apollobot.core import APOLLO_HOME

logger = logging.getLogger(__name__)

_DEFAULT_PID_PATH = str(APOLLO_HOME / "runtime.pid")


class PidFile:
    """Manage a PID file to prevent concurrent runtime instances."""

    def __init__(self, path: str = "") -> None:
        self.path = Path(path) if path else Path(_DEFAULT_PID_PATH)
        self._fd: int | None = None

    def acquire(self) -> bool:
        """Write current PID to the file.

        If the file already exists and the recorded process is still alive,
        return False (another instance is running).  If the file is stale
        (process dead), overwrite it.

        Returns True on success, False if another runtime is active.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)

        running, existing_pid = self.is_running()
        if running:
            logger.warning("PID file %s held by running process %s", self.path, existing_pid)
            return False

        # Open (or create) the file and lock it
        fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # Another process holds the lock
            os.close(fd)
            return False

        os.write(fd, str(os.getpid()).encode())
        os.fsync(fd)
        self._fd = fd
        return True

    def release(self) -> None:
        """Remove the PID file and release the lock."""
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass

    def is_running(self) -> tuple[bool, int | None]:
        """Check whether a runtime is currently active.

        Returns (running, pid).  ``running`` is True only when the PID file
        exists, contains a valid PID, and that process is alive.
        """
        if not self.path.exists():
            return False, None

        try:
            text = self.path.read_text().strip()
            if not text:
                return False, None
            pid = int(text)
        except (ValueError, OSError):
            return False, None

        if not _process_alive(pid):
            return False, pid

        return True, pid


def _process_alive(pid: int) -> bool:
    """Return True if a process with the given PID exists."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we lack permission to signal it
        return True
