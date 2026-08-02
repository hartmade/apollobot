"""Typed service contracts shared by the HTTP layer and job manager."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import PurePosixPath
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class ProposedStep(BaseModel):
    type: str
    label: str
    detail: str


class ContextAttachment(BaseModel):
    """Private, untrusted context supplied by the Frontier Science platform."""

    id: str
    kind: Literal["paper", "dataset", "image", "code", "notebook", "link", "doi"]
    label: str = Field(min_length=1, max_length=255)
    media_type: str | None = Field(default=None, max_length=160)
    size_bytes: int | None = Field(default=None, ge=1, le=25 * 1024 * 1024)
    checksum_sha256: str | None = None
    source_url: str | None = None
    source_doi: str | None = None
    private_url: str | None = None
    private_url_expires_in: int | None = Field(default=None, ge=1, le=3600)
    local_path: str | None = Field(default=None, max_length=512)
    cache_status: Literal["verified", "unavailable", "not-required"] | None = None
    dataset_profile: dict[str, Any] | None = None
    execution_allowed: bool = False
    analysis_allowed: bool = False
    untrusted: bool = True
    instructions_are_data: bool = True

    model_config = {"extra": "ignore"}

    @field_validator("id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        from uuid import UUID

        try:
            return str(UUID(value))
        except ValueError as error:
            raise ValueError("Context attachment id must be a UUID") from error

    @field_validator("checksum_sha256")
    @classmethod
    def valid_checksum(cls, value: str | None) -> str | None:
        import re

        normalized = value.lower() if isinstance(value, str) else None
        if normalized is not None and not re.fullmatch(r"[0-9a-f]{64}", normalized):
            raise ValueError("Context checksum must be SHA-256")
        return normalized

    @field_validator("source_url", "private_url")
    @classmethod
    def valid_https_url(cls, value: str | None) -> str | None:
        from urllib.parse import urlsplit

        if value is None:
            return None
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Context URLs must be public HTTPS URLs without embedded credentials")
        return value

    @field_validator("local_path")
    @classmethod
    def valid_local_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("Context cache path must stay inside its investigation")
        return path.as_posix()

    def planner_manifest(self) -> dict[str, Any]:
        """Return value-free metadata; never disclose a signed download URL to an LLM."""
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "checksum_sha256": self.checksum_sha256,
            "source_url": self.source_url,
            "source_doi": self.source_doi,
            "dataset_profile": self.dataset_profile if self.kind == "dataset" else None,
            "analysis_allowed": bool(self.kind == "dataset" and self.analysis_allowed),
            "untrusted": True,
            "instructions_are_data": True,
        }


def parse_context_attachments(value: object) -> list[ContextAttachment]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 8:
        raise ValueError("Attach no more than eight context items")
    attachments = [ContextAttachment.model_validate(item) for item in value]
    if len({attachment.id for attachment in attachments}) != len(attachments):
        raise ValueError("Context attachment ids must be unique")
    return attachments


class RunEstimate(BaseModel):
    duration_minutes: int = Field(alias="durationMinutes", ge=1, le=10_080)
    compute_usd: float = Field(alias="computeUsd", ge=0, le=100_000)
    literature_targets: int = Field(alias="literatureTargets", ge=0, le=10_000)

    model_config = {"populate_by_name": True}


class QuestionCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    title: str
    domain: str
    apollo_domain: str = "bioinformatics"
    mode: str
    answerability: Literal["investigable", "needs-refinement", "unsafe"]
    novelty: Literal["open", "adjacent-work-likely", "well-studied"]
    rationale: str
    hypotheses: list[str] = Field(default_factory=list)
    proposed_steps: list[ProposedStep] = Field(alias="proposedSteps", default_factory=list)
    estimate: RunEstimate
    source: Literal["apollobot", "local-framer"] = "apollobot"

    model_config = {"populate_by_name": True}


class ResearchNode(BaseModel):
    key: str
    label: str
    node_type: str
    sequence: int
    status: Literal[
        "pending",
        "ready",
        "awaiting_approval",
        "running",
        "complete",
        "failed",
        "skipped",
        "cancelled",
    ] = "pending"
    summary: str = ""


class ServiceEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    investigation_id: str
    node_id: str | None = None
    sequence: int
    event_type: str
    status: str
    public_summary: str
    artifact_refs: list[str] = Field(default_factory=list)
    cost_delta_usd: float = 0.0
    occurred_at: str = Field(default_factory=utc_now)
    data: dict[str, Any] = Field(default_factory=dict)


NODE_BLUEPRINT: tuple[ResearchNode, ...] = (
    ResearchNode(
        key="frame_question",
        label="Frame the question",
        node_type="frame_question",
        sequence=1,
        status="complete",
        summary="Question translated into an inspectable research objective.",
    ),
    ResearchNode(
        key="search_literature",
        label="Map prior work",
        node_type="search_literature",
        sequence=2,
        status="ready",
        summary="Search the closest claims, datasets, and disagreements.",
    ),
    ResearchNode(
        key="define_hypotheses",
        label="Define hypotheses",
        node_type="define_hypotheses",
        sequence=3,
        summary="Specify support, failure, and falsification conditions.",
    ),
    ResearchNode(
        key="design_experiment",
        label="Design experiment",
        node_type="design_experiment",
        sequence=4,
        summary="Precommit data, controls, metrics, exclusions, and robustness checks.",
    ),
    ResearchNode(
        key="approve_plan",
        label="Approve plan",
        node_type="approve_plan",
        sequence=5,
        status="awaiting_approval",
        summary="Human checkpoint before tools or compute run.",
    ),
    ResearchNode(
        key="select_data",
        label="Select and acquire data",
        node_type="select_data",
        sequence=6,
        summary="Verify access, licensing, leakage risk, power, and fit.",
    ),
    ResearchNode(
        key="execute_analysis",
        label="Run experiment",
        node_type="execute_analysis",
        sequence=7,
        summary="Execute analysis in a captured environment.",
    ),
    ResearchNode(
        key="run_robustness_checks",
        label="Run robustness checks",
        node_type="run_robustness_checks",
        sequence=8,
        summary="Test alternate specifications and sensitivity boundaries.",
    ),
    ResearchNode(
        key="adversarial_review",
        label="Stress-test findings",
        node_type="adversarial_review",
        sequence=9,
        summary="Seek disconfirming evidence and methodological failures.",
    ),
    ResearchNode(
        key="draft_record",
        label="Draft research record",
        node_type="draft_record",
        sequence=10,
        summary="Link claims, evidence, files, limitations, and provenance.",
    ),
    ResearchNode(
        key="prepare_replication_kit",
        label="Prepare replication kit",
        node_type="prepare_replication_kit",
        sequence=11,
        summary="Capture the environment and one-command reproduction materials.",
    ),
)


PHASE_TO_NODE = {
    "literature_review": "search_literature",
    "data_acquisition": "select_data",
    "analysis": "execute_analysis",
    "statistical_testing": "run_robustness_checks",
    "manuscript_drafting": "draft_record",
    "self_review": "adversarial_review",
    "manuscript_revision": "draft_record",
}
