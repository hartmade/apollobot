"""
Session — runtime state of a research execution.

A Session wraps a Mission and tracks progress through research phases,
accumulates findings, manages the output directory, and handles
checkpoint approvals.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from apollobot.core import APOLLO_SESSIONS_DIR
from apollobot.core.mission import Mission

# Avoid circular import — use TYPE_CHECKING for type hints only
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from apollobot.core.translation import TranslationReport


class Phase(str, Enum):
    # Discover mode phases (original research)
    PLANNING = "planning"
    LITERATURE_REVIEW = "literature_review"
    DATA_ACQUISITION = "data_acquisition"
    ANALYSIS = "analysis"
    STATISTICAL_TESTING = "statistical_testing"
    MANUSCRIPT_DRAFTING = "manuscript_drafting"
    SELF_REVIEW = "self_review"

    # Translate mode phases
    TRANSLATE_ASSESS = "translate_assess"
    TRANSLATE_PRIOR_ART = "translate_prior_art"
    TRANSLATE_SPECIFY = "translate_specify"
    TRANSLATE_VALIDATE = "translate_validate"
    TRANSLATE_REPORT = "translate_report"

    # Implement mode phases
    IMPLEMENT_SCAFFOLD = "implement_scaffold"
    IMPLEMENT_BUILD = "implement_build"
    IMPLEMENT_TEST = "implement_test"
    IMPLEMENT_DOCUMENT = "implement_document"
    IMPLEMENT_PACKAGE = "implement_package"
    IMPLEMENT_VALIDATE = "implement_validate"

    # Commercialize mode phases
    COMMERCIALIZE_MARKET = "commercialize_market"
    COMMERCIALIZE_IP = "commercialize_ip"
    COMMERCIALIZE_GTM = "commercialize_gtm"

    # Terminal states
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PhaseResult(BaseModel):
    phase: Phase
    started_at: str = ""
    completed_at: str = ""
    summary: str = ""
    artifacts: list[str] = Field(default_factory=list)  # file paths
    findings: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class CostTracker(BaseModel):
    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    estimated_cost_usd: float = 0.0
    compute_cost_usd: float = 0.0

    @property
    def total_cost(self) -> float:
        return self.estimated_cost_usd + self.compute_cost_usd

    def record_llm_call(
        self, input_tokens: int, output_tokens: int, cost: float
    ) -> None:
        self.llm_calls += 1
        self.llm_input_tokens += input_tokens
        self.llm_output_tokens += output_tokens
        self.estimated_cost_usd += cost


class Session(BaseModel):
    """
    Mutable runtime state for a research session.
    """

    mission: Mission
    current_phase: Phase = Phase.PLANNING
    phase_results: dict[str, PhaseResult] = Field(default_factory=dict)
    cost: CostTracker = Field(default_factory=CostTracker)
    started_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: str = ""
    provenance_log: list[dict[str, Any]] = Field(default_factory=list)

    # Accumulated knowledge the agent builds during the session
    literature_corpus: list[dict[str, Any]] = Field(default_factory=list)
    datasets: list[dict[str, Any]] = Field(default_factory=list)
    hypotheses_status: dict[str, str] = Field(default_factory=dict)  # hypothesis → supported|rejected|inconclusive
    key_findings: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    # Translation potential scores (set during Discover self-review)
    translation_scores: dict[str, float] = Field(default_factory=dict)

    # Translation report (populated by Translate mode)
    translation_report: Optional[Any] = None  # TranslationReport (Any to avoid circular import)

    # ------------------------------------------------------------------
    # Directory management
    # ------------------------------------------------------------------

    @property
    def session_dir(self) -> Path:
        base = Path(self.mission.metadata.get("output_dir", str(APOLLO_SESSIONS_DIR)))
        return base / self.mission.id

    def init_directories(self) -> None:
        """Create the session output directory structure."""
        dirs = [
            self.session_dir,
            self.session_dir / "figures",
            self.session_dir / "data" / "raw",
            self.session_dir / "data" / "processed",
            self.session_dir / "analysis" / "scripts",
            self.session_dir / "analysis" / "notebooks",
            self.session_dir / "provenance",
            self.session_dir / "review",
            self.session_dir / "replication_kit",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        # Save mission file
        self.mission.to_yaml(self.session_dir / "mission.yaml")

    # ------------------------------------------------------------------
    # Phase transitions
    # ------------------------------------------------------------------

    def begin_phase(self, phase: Phase) -> None:
        self.current_phase = phase
        self.phase_results[phase.value] = PhaseResult(
            phase=phase,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self.log_event("phase_started", {"phase": phase.value})

    def complete_phase(self, phase: Phase, summary: str = "", findings: list[dict[str, Any]] | None = None) -> None:
        result = self.phase_results.get(phase.value)
        if result:
            result.completed_at = datetime.now(timezone.utc).isoformat()
            result.summary = summary
            if findings:
                result.findings = findings
        self.log_event("phase_completed", {"phase": phase.value, "summary": summary})

    def fail_phase(self, phase: Phase, error: str) -> None:
        result = self.phase_results.get(phase.value)
        if result:
            result.errors.append(error)
        self.current_phase = Phase.FAILED
        self.log_event("phase_failed", {"phase": phase.value, "error": error})

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------

    def log_event(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "phase": self.current_phase.value,
            **(data or {}),
        }
        self.provenance_log.append(entry)

    def save_provenance(self) -> None:
        """Write provenance log to disk."""
        out = self.session_dir / "provenance" / "execution_log.json"
        out.write_text(json.dumps(self.provenance_log, indent=2))

    # ------------------------------------------------------------------
    # Cost guard
    # ------------------------------------------------------------------

    def check_budget(self) -> bool:
        """Return True if still within budget."""
        return self.cost.total_cost < self.mission.constraints.compute_budget

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_state(self) -> None:
        """Persist full session state to disk."""
        out = self.session_dir / "session_state.json"
        out.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load_state(cls, session_dir: str | Path) -> "Session":
        """Resume a session from disk."""
        state_file = Path(session_dir) / "session_state.json"
        return cls.model_validate_json(state_file.read_text())
