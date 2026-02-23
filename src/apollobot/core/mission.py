"""
Mission — the structured representation of a research objective.

A mission can be created from a one-liner CLI string or from a detailed
YAML mission file.  It is the single source of truth for what the agent
is trying to accomplish in a research session.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import yaml
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ResearchMode(str, Enum):
    # Discover mode (original research)
    HYPOTHESIS = "hypothesis"
    EXPLORATORY = "exploratory"
    META_ANALYSIS = "meta-analysis"
    REPLICATION = "replication"
    SIMULATION = "simulation"

    # Pipeline modes
    DISCOVER = "discover"  # Alias for the above modes (routes to mode-specific)
    TRANSLATE = "translate"
    IMPLEMENT = "implement"
    COMMERCIALIZE = "commercialize"
    PIPELINE = "pipeline"  # Full Discover → Translate → Implement → Commercialize


class CheckpointAction(str, Enum):
    NOTIFY = "notify"
    REQUIRE_APPROVAL = "require_approval"
    AUTO_CONTINUE = "auto_continue"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class Constraints(BaseModel):
    compute_budget: float = 50.0  # USD
    time_limit: str = "48h"
    data_sources: str = "public_only"  # public_only | all
    ethics: str = "observational_only"
    max_llm_calls: int = 5000
    max_datasets: int = 20


class Checkpoint(BaseModel):
    after: str  # phase name
    action: CheckpointAction = CheckpointAction.NOTIFY


class NotificationOverrides(BaseModel):
    """Per-mission notification overrides."""

    enabled: Optional[bool] = None  # None = use global config
    extra_channels: list[dict[str, Any]] = Field(default_factory=list)
    heartbeat_interval: Optional[int] = None


class OutputSpec(BaseModel):
    format: str = "paper_draft"
    target_journal: str = "frontier"
    include_provenance: bool = True
    generate_notebooks: bool = False
    latex_template: str = "default"


# ---------------------------------------------------------------------------
# Mission
# ---------------------------------------------------------------------------


class Mission(BaseModel):
    """
    The complete specification of a research objective.
    """

    id: str = Field(default_factory=lambda: f"session-{uuid4().hex[:8]}")
    title: str = ""
    objective: str
    hypotheses: list[str] = Field(default_factory=list)
    mode: ResearchMode = ResearchMode.HYPOTHESIS
    domain: str = "bioinformatics"
    resource_pack: str = ""  # defaults to domain if empty
    constraints: Constraints = Field(default_factory=Constraints)
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    output: OutputSpec = Field(default_factory=OutputSpec)
    notifications: NotificationOverrides = Field(default_factory=NotificationOverrides)
    paper_id: str = ""  # for replication mode — arxiv id, DOI, etc.
    dataset_id: str = ""  # for exploratory mode — specific dataset

    # Pipeline inputs — for cross-mode references
    source_session: str = ""  # session ID to translate/implement from
    source_paper: str = ""  # external paper DOI to translate

    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.resource_pack:
            self.resource_pack = self.domain
        if not self.title:
            # Auto-generate a short title from the objective
            self.title = self.objective[:80].strip()

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Mission":
        """Load a mission from a YAML file."""
        raw = yaml.safe_load(Path(path).read_text())
        return cls(**raw)

    @classmethod
    def from_objective(
        cls,
        objective: str,
        *,
        mode: str = "hypothesis",
        domain: str = "bioinformatics",
        paper_id: str = "",
        dataset_id: str = "",
    ) -> "Mission":
        """Create a mission from a one-line objective string."""
        return cls(
            objective=objective,
            mode=ResearchMode(mode),
            domain=domain,
            paper_id=paper_id,
            dataset_id=dataset_id,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def time_limit_seconds(self) -> int:
        """Parse the time_limit string into seconds."""
        tl = self.constraints.time_limit.strip().lower()
        if tl.endswith("h"):
            return int(float(tl[:-1]) * 3600)
        if tl.endswith("m"):
            return int(float(tl[:-1]) * 60)
        if tl.endswith("d"):
            return int(float(tl[:-1]) * 86400)
        return int(tl)

    def to_yaml(self, path: str | Path) -> None:
        """Serialize mission to YAML."""
        Path(path).write_text(yaml.dump(self.model_dump(), default_flow_style=False))
