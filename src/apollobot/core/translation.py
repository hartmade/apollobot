"""
Translation models — Pydantic models for the Translate mode output.

The Translate mode converts research findings into actionable
implementation specifications with IP landscape analysis and
feasibility assessments.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TranslationStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class FeasibilityRating(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# IP Landscape
# ---------------------------------------------------------------------------


class PatentEntry(BaseModel):
    """A single patent or patent application in the IP landscape."""

    patent_id: str = ""
    title: str = ""
    assignee: str = ""
    filing_date: str = ""
    status: str = ""  # granted, pending, expired
    relevance_score: float = 0.0
    summary: str = ""


class IPLandscape(BaseModel):
    """Intellectual property landscape analysis."""

    existing_patents: list[PatentEntry] = Field(default_factory=list)
    freedom_to_operate: str = ""  # clear, restricted, blocked
    patentability_assessment: str = ""
    key_claims_at_risk: list[str] = Field(default_factory=list)
    recommended_ip_strategy: str = ""
    prior_art_summary: str = ""


# ---------------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------------


class TechnicalRequirement(BaseModel):
    """A single technical requirement for implementation."""

    name: str
    description: str = ""
    difficulty: str = "medium"  # easy, medium, hard
    estimated_effort: str = ""  # e.g. "2 weeks"
    dependencies: list[str] = Field(default_factory=list)


class FeasibilityAssessment(BaseModel):
    """Assessment of implementation feasibility."""

    overall_rating: FeasibilityRating = FeasibilityRating.UNKNOWN
    technical_feasibility: float = 0.0  # 0-10
    resource_requirements: str = ""
    timeline_estimate: str = ""
    key_risks: list[str] = Field(default_factory=list)
    mitigation_strategies: list[str] = Field(default_factory=list)
    technical_requirements: list[TechnicalRequirement] = Field(default_factory=list)
    infrastructure_needs: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Implementation Spec
# ---------------------------------------------------------------------------


class ImplementationSpec(BaseModel):
    """Specification for implementing research findings."""

    title: str = ""
    description: str = ""
    target_platform: str = ""  # e.g. "Python library", "Web API", "Clinical tool"
    architecture_overview: str = ""
    components: list[dict[str, Any]] = Field(default_factory=list)
    data_requirements: list[str] = Field(default_factory=list)
    testing_strategy: str = ""
    deployment_strategy: str = ""
    estimated_cost: float = 0.0
    estimated_timeline: str = ""


# ---------------------------------------------------------------------------
# Market Analysis (used by both Translate and Commercialize)
# ---------------------------------------------------------------------------


class MarketSegment(BaseModel):
    """A target market segment."""

    name: str
    size_estimate: str = ""
    growth_rate: str = ""
    key_players: list[str] = Field(default_factory=list)
    entry_barriers: list[str] = Field(default_factory=list)


class MarketAnalysis(BaseModel):
    """Market analysis for commercialization."""

    total_addressable_market: str = ""
    serviceable_market: str = ""
    segments: list[MarketSegment] = Field(default_factory=list)
    competitive_landscape: str = ""
    differentiation: list[str] = Field(default_factory=list)
    pricing_strategy: str = ""
    go_to_market_summary: str = ""


# ---------------------------------------------------------------------------
# Translation Report (top-level output)
# ---------------------------------------------------------------------------


class TranslationScores(BaseModel):
    """Scores assigned during translation potential assessment."""

    commercial_relevance: float = 0.0  # 0-10
    implementation_feasibility: float = 0.0  # 0-10
    novelty: float = 0.0  # 0-10

    @property
    def average(self) -> float:
        return (self.commercial_relevance + self.implementation_feasibility + self.novelty) / 3

    @property
    def is_translation_candidate(self) -> bool:
        return self.average >= 7.0


class TranslationReport(BaseModel):
    """
    The complete output of a Translate mode session.

    Contains IP landscape, feasibility assessment, implementation spec,
    and market analysis.
    """

    id: str = ""
    source_session_id: str = ""
    source_paper_doi: str = ""
    status: TranslationStatus = TranslationStatus.PENDING
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: str = ""

    # Scores from Discover self-review
    translation_scores: TranslationScores = Field(default_factory=TranslationScores)

    # Phase outputs
    assessment_summary: str = ""
    ip_landscape: IPLandscape = Field(default_factory=IPLandscape)
    implementation_spec: ImplementationSpec = Field(default_factory=ImplementationSpec)
    feasibility: FeasibilityAssessment = Field(default_factory=FeasibilityAssessment)
    market_analysis: MarketAnalysis = Field(default_factory=MarketAnalysis)

    # Provenance
    provenance_chain: list[str] = Field(default_factory=list)  # session IDs
    metadata: dict[str, Any] = Field(default_factory=dict)
