"""
ResearchGuardrails — safety constraints for the continuous runtime.

Enforced before every action the brain proposes. Checks:
- Daily compute budget (rolling 24h)
- Concurrent session limit
- Daily session count limit
- Domain approval
- Emergency stop
"""

from __future__ import annotations

import logging

from apollobot.runtime.config import GuardrailsConfig
from apollobot.runtime.storage import RunnerStorage
from apollobot.runtime.types import BrainAction, EnforcementResult, RunnerState

logger = logging.getLogger(__name__)


class ResearchGuardrails:
    """Pre-flight safety checks before any research action."""

    def __init__(self, config: GuardrailsConfig, storage: RunnerStorage) -> None:
        self.config = config
        self.storage = storage

    def check(self, action: BrainAction, state: RunnerState) -> EnforcementResult:
        """Run all guardrail checks. Returns first failure or allowed."""
        checks = [
            self._check_emergency_stop,
            self._check_daily_budget,
            self._check_concurrent_sessions,
            self._check_daily_session_count,
            self._check_domain_approved,
        ]
        for check_fn in checks:
            result = check_fn(action, state)
            if not result.allowed:
                logger.warning("Guardrails blocked action %s: %s", action.type, result.reason)
                return result
        return EnforcementResult(allowed=True)

    def _check_emergency_stop(self, action: BrainAction, state: RunnerState) -> EnforcementResult:
        if self.config.emergency_stop:
            return EnforcementResult(allowed=False, reason="Emergency stop is active")
        return EnforcementResult(allowed=True)

    def _check_daily_budget(self, action: BrainAction, state: RunnerState) -> EnforcementResult:
        if action.type.value == "idle":
            return EnforcementResult(allowed=True)
        daily = self.storage.daily_spend()
        remaining = self.config.daily_compute_budget_usd - daily
        if remaining <= 0:
            return EnforcementResult(
                allowed=False,
                reason=f"Daily budget exhausted: ${daily:.2f} / ${self.config.daily_compute_budget_usd:.2f}",
            )
        return EnforcementResult(allowed=True)

    def _check_concurrent_sessions(
        self, action: BrainAction, state: RunnerState
    ) -> EnforcementResult:
        if action.type.value != "start_research":
            return EnforcementResult(allowed=True)
        active = len(state.active_sessions)
        if active >= self.config.max_concurrent_sessions:
            return EnforcementResult(
                allowed=False,
                reason=f"Concurrent session limit reached: {active}/{self.config.max_concurrent_sessions}",
            )
        return EnforcementResult(allowed=True)

    def _check_daily_session_count(
        self, action: BrainAction, state: RunnerState
    ) -> EnforcementResult:
        if action.type.value != "start_research":
            return EnforcementResult(allowed=True)
        today_count = self.storage.sessions_started_today()
        if today_count >= self.config.max_sessions_per_day:
            return EnforcementResult(
                allowed=False,
                reason=f"Daily session limit reached: {today_count}/{self.config.max_sessions_per_day}",
            )
        return EnforcementResult(allowed=True)

    def _check_domain_approved(self, action: BrainAction, state: RunnerState) -> EnforcementResult:
        if action.type.value == "idle":
            return EnforcementResult(allowed=True)
        domain = action.domain or state.domain
        if domain and domain not in self.config.approved_domains:
            return EnforcementResult(
                allowed=False,
                reason=f"Domain '{domain}' not in approved list: {self.config.approved_domains}",
            )
        return EnforcementResult(allowed=True)
