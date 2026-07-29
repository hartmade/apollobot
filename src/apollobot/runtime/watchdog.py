"""
Watchdog — circuit breaker for the continuous runtime.

Prevents the runtime from hammering a failing brain/LLM by tracking
consecutive failures and entering a cooldown period.

States:
    CLOSED    — Normal operation, all ticks proceed
    OPEN      — Too many failures, ticks are skipped
    HALF_OPEN — Cooldown expired, single probe tick allowed
"""

from __future__ import annotations

import logging
import time
from enum import Enum

from apollobot.runtime.config import WatchdogConfig

logger = logging.getLogger(__name__)


class WatchdogState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class Watchdog:
    """Circuit breaker for brain failures."""

    def __init__(self, config: WatchdogConfig) -> None:
        self.config = config
        self.state = WatchdogState.CLOSED
        self.consecutive_failures = 0
        self.last_failure_time: float = 0.0

    def should_attempt(self) -> bool:
        """Check if the current tick should proceed."""
        if self.state == WatchdogState.CLOSED:
            return True

        if self.state == WatchdogState.OPEN:
            elapsed = time.monotonic() - self.last_failure_time
            if elapsed >= self.config.cooldown_seconds:
                logger.info("Watchdog cooldown expired, entering half-open state")
                self.state = WatchdogState.HALF_OPEN
                return True
            return False

        # HALF_OPEN — allow the probe
        return True

    def record_success(self) -> None:
        """Record a successful tick."""
        if self.state != WatchdogState.CLOSED:
            logger.info("Watchdog: probe succeeded, returning to closed state")
        self.state = WatchdogState.CLOSED
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        """Record a failed tick."""
        self.consecutive_failures += 1
        self.last_failure_time = time.monotonic()

        if self.state == WatchdogState.HALF_OPEN:
            logger.warning("Watchdog: probe failed, returning to open state")
            self.state = WatchdogState.OPEN
        elif self.consecutive_failures >= self.config.failure_threshold:
            logger.warning(
                "Watchdog: %d consecutive failures (threshold=%d), entering open state",
                self.consecutive_failures,
                self.config.failure_threshold,
            )
            self.state = WatchdogState.OPEN
