"""
Configuration for the continuous runtime.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class GuardrailsConfig(BaseModel):
    """Safety guardrail configuration."""

    daily_compute_budget_usd: float = 100.0
    max_concurrent_sessions: int = 3
    max_sessions_per_day: int = 10
    api_calls_per_hour: int = 1000
    approved_domains: list[str] = Field(
        default_factory=lambda: [
            "bioinformatics",
            "physics",
            "cs_ml",
            "comp_chem",
            "economics",
            "astronomy",
            "climate",
            "neuroscience",
            "epidemiology",
            "ecology",
            "geology",
            "materials",
            "psychology",
            "mathematics",
            "social_science",
        ]
    )
    max_session_cost_usd: float = 50.0
    emergency_stop: bool = False


class WatchdogConfig(BaseModel):
    """Circuit breaker configuration."""

    failure_threshold: int = 3
    cooldown_seconds: float = 300.0


class RuntimeConfig(BaseModel):
    """Full configuration for the continuous runtime."""

    # Domain focus
    domain: str = "bioinformatics"

    # Tick scheduling
    default_interval: int = 300  # seconds between ticks
    min_interval: int = 60
    max_interval: int = 3600
    error_interval: int = 120  # interval after an error

    # Safety
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    watchdog: WatchdogConfig = Field(default_factory=WatchdogConfig)

    # Brain
    memory_window: int = 10  # action history entries to show brain
    reasoning_window: int = 5  # decision history entries to show brain
    user_instructions: str = ""

    # Execution mode
    dry_run: bool = False  # no real API calls

    # Health server
    health_port: int = 8080

    # Remote logging (empty = disabled)
    remote_log_url: str = ""

    # Storage
    db_path: str = ""  # empty = ~/.apollobot/runtime.db
