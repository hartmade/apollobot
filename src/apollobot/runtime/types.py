"""
Shared types for the continuous runtime.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action types the brain can emit
# ---------------------------------------------------------------------------


class ActionType(str, Enum):
    START_RESEARCH = "start_research"
    SCAN_LITERATURE = "scan_literature"
    REVIEW_SESSION = "review_session"
    AUTO_SUBMIT = "auto_submit"
    AUTO_REVIEW = "auto_review"
    IDLE = "idle"


class BrainAction(BaseModel):
    """A single action the brain wants to take."""

    type: ActionType
    objective: str = ""
    mode: str = "hypothesis"
    domain: str = ""
    session_id: str = ""  # for review_session
    reasoning: str = ""


class BrainDecision(BaseModel):
    """Full decision from one brain reasoning cycle."""

    actions: list[BrainAction] = Field(default_factory=list)
    reasoning: str = ""
    next_check_in: int = 300  # seconds until next tick
    memory: dict[str, str] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Runner state — snapshot passed to the brain each tick
# ---------------------------------------------------------------------------


class SessionSummary(BaseModel):
    """Compact summary of a research session for brain context."""

    session_id: str
    objective: str
    domain: str
    mode: str
    phase: str
    cost_usd: float = 0.0
    started_at: str = ""
    completed_at: str = ""
    key_findings: list[str] = Field(default_factory=list)
    translation_score: float = 0.0


class RunnerState(BaseModel):
    """Full state snapshot assembled for the brain each tick."""

    tick_number: int
    uptime_seconds: float
    domain: str
    active_sessions: list[SessionSummary] = Field(default_factory=list)
    completed_sessions: list[SessionSummary] = Field(default_factory=list)
    failed_sessions: list[SessionSummary] = Field(default_factory=list)
    total_papers: int = 0
    total_cost_usd: float = 0.0
    daily_cost_usd: float = 0.0
    daily_sessions_started: int = 0
    guardrails_remaining_budget: float = 0.0
    guardrails_max_concurrent: int = 3
    watchdog_state: str = "closed"
    memory: dict[str, str] = Field(default_factory=dict)
    trajectory_summary: str = ""  # cross-session learning insights


# ---------------------------------------------------------------------------
# History records
# ---------------------------------------------------------------------------


class ActionRecord(BaseModel):
    """Record of an action taken, stored in history."""

    tick: int
    action_type: str
    objective: str = ""
    result: str = ""  # "completed", "failed", "blocked"
    details: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DecisionRecord(BaseModel):
    """Record of a brain reasoning cycle."""

    tick: int
    reasoning: str
    actions: list[str] = Field(default_factory=list)
    next_check_in: int = 300
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# Guardrails types
# ---------------------------------------------------------------------------


class EnforcementResult(BaseModel):
    """Result of a guardrails pre-flight check."""

    allowed: bool
    reason: str = ""
