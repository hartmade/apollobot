"""
SubmissionReviewer — AI-powered manuscript review with structured scoring.

Builds on ReviewEngine to provide a committee-ready review report
with per-dimension scores, provenance-aware badging, and
actionable revision requests.
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from apollobot.agents import LLMProvider
from apollobot.review import ReviewEngine, ReviewReport

logger = logging.getLogger(__name__)


class ProvenanceBadge(str, Enum):
    GOLD = "gold"  # Full ApolloBot session with provenance chain
    SILVER = "silver"  # Partial provenance (session but incomplete chain)
    BRONZE = "bronze"  # Manual submission, no provenance


class Recommendation(str, Enum):
    ACCEPT = "accept"
    MINOR_REVISION = "minor_revision"
    MAJOR_REVISION = "major_revision"
    REJECT = "reject"


class DimensionScore(BaseModel):
    dimension: str
    score: float  # 1-10
    justification: str = ""


class SubmissionReviewReport(BaseModel):
    """Full committee-ready review report."""

    session_id: str = ""
    recommendation: str = ""  # accept / minor_revision / major_revision / reject
    confidence: float = 0.0  # 0-1
    provenance_badge: str = "bronze"
    scores: list[DimensionScore] = Field(default_factory=list)
    key_issues: list[dict[str, str]] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    revision_requests: list[str] = Field(default_factory=list)
    summary: str = ""
    base_review: ReviewReport | None = None


class SubmissionReviewer:
    """
    Reviews manuscripts for journal submission with structured scoring.

    Produces a committee-ready report with:
    - Per-dimension scores (1-10)
    - Overall recommendation
    - Provenance badge (Gold/Silver/Bronze)
    - Key issues and revision requests
    """

    DIMENSIONS = [
        "statistical_rigor",
        "methodological_soundness",
        "reproducibility",
        "novelty",
        "clarity",
    ]

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm
        self.engine = ReviewEngine(llm)

    async def review(
        self,
        manuscript_text: str,
        *,
        provenance_path: Path | None = None,
        session_id: str = "",
    ) -> SubmissionReviewReport:
        """
        Run a full submission review.

        1. Run base ReviewEngine review
        2. Score each dimension via LLM
        3. Determine provenance badge
        4. Generate committee-ready report
        """
        # Base review
        base_report = await self.engine.review_manuscript(
            manuscript_text, provenance_path=provenance_path, session_id=session_id
        )

        # Determine provenance badge
        badge = self._assess_provenance(provenance_path, session_id)

        # Structured scoring
        scoring_resp = await self.llm.complete(
            messages=[{"role": "user", "content": (
                f"Score this manuscript on each dimension (1-10) with justification.\n\n"
                f"MANUSCRIPT (excerpt):\n{manuscript_text[:6000]}\n\n"
                f"BASE REVIEW SUMMARY:\n{base_report.summary}\n\n"
                "Score these dimensions:\n"
                "1. statistical_rigor — p-values, effect sizes, corrections, appropriate tests\n"
                "2. methodological_soundness — controls, confounders, experimental design\n"
                "3. reproducibility — provenance, data availability, code availability\n"
                "4. novelty — contribution beyond existing literature\n"
                "5. clarity — writing quality, logical flow, figures\n\n"
                f"Provenance badge: {badge.value} "
                f"({'full ApolloBot provenance chain' if badge == ProvenanceBadge.GOLD else 'limited/no provenance'})\n\n"
                "Return ONLY JSON:\n"
                "{\n"
                '  "recommendation": "accept|minor_revision|major_revision|reject",\n'
                '  "confidence": 0.0-1.0,\n'
                '  "scores": [\n'
                '    {"dimension": "statistical_rigor", "score": 1-10, "justification": "..."},\n'
                '    {"dimension": "methodological_soundness", "score": 1-10, "justification": "..."},\n'
                '    {"dimension": "reproducibility", "score": 1-10, "justification": "..."},\n'
                '    {"dimension": "novelty", "score": 1-10, "justification": "..."},\n'
                '    {"dimension": "clarity", "score": 1-10, "justification": "..."}\n'
                "  ],\n"
                '  "key_issues": [{"severity": "critical|major|minor", "description": "..."}],\n'
                '  "strengths": ["..."],\n'
                '  "revision_requests": ["specific request 1", "..."],\n'
                '  "summary": "2-3 sentence overall assessment"\n'
                "}"
            )}],
            system=(
                "You are a senior journal editor scoring a manuscript for peer review. "
                "Be rigorous but fair. Gold-badge submissions with full provenance "
                "should receive appropriate credit for reproducibility. "
                "Score conservatively — a 7 means genuinely good, not average."
            ),
        )

        try:
            review_data = LLMProvider._extract_json(scoring_resp.text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse scoring response, using defaults")
            review_data = {
                "recommendation": base_report.overall_verdict or "major_revision",
                "confidence": base_report.confidence,
                "scores": [
                    {"dimension": d, "score": 5, "justification": "Scoring parse failed"}
                    for d in self.DIMENSIONS
                ],
                "key_issues": [
                    {"severity": i.severity, "description": i.description}
                    for i in base_report.issues
                ],
                "strengths": base_report.strengths,
                "revision_requests": [],
                "summary": base_report.summary,
            }

        scores = [
            DimensionScore(**s) for s in review_data.get("scores", [])
        ]

        return SubmissionReviewReport(
            session_id=session_id,
            recommendation=review_data.get("recommendation", "major_revision"),
            confidence=float(review_data.get("confidence", 0.5)),
            provenance_badge=badge.value,
            scores=scores,
            key_issues=review_data.get("key_issues", []),
            strengths=review_data.get("strengths", []),
            revision_requests=review_data.get("revision_requests", []),
            summary=review_data.get("summary", ""),
            base_review=base_report,
        )

    @staticmethod
    def _assess_provenance(
        provenance_path: Path | None, session_id: str
    ) -> ProvenanceBadge:
        """Determine provenance badge based on available provenance data."""
        if not session_id or not provenance_path:
            return ProvenanceBadge.BRONZE

        if not provenance_path.exists():
            return ProvenanceBadge.BRONZE

        # Check for full provenance chain
        exec_log = provenance_path / "execution_log.json"
        model_calls = provenance_path / "model_calls.json"
        data_lineage = provenance_path / "data_lineage.json"

        has_exec = exec_log.exists()
        has_models = model_calls.exists()
        has_lineage = data_lineage.exists()

        if has_exec and has_models and has_lineage:
            return ProvenanceBadge.GOLD
        elif has_exec or has_models:
            return ProvenanceBadge.SILVER
        return ProvenanceBadge.BRONZE

    def format_report(self, report: SubmissionReviewReport) -> str:
        """Format a review report as markdown for display or saving."""
        lines = [
            f"# Submission Review Report",
            "",
            f"**Recommendation**: {report.recommendation.replace('_', ' ').title()}",
            f"**Confidence**: {report.confidence:.0%}",
            f"**Provenance**: {report.provenance_badge.title()} badge",
            "",
            "## Scores",
            "",
        ]

        for s in report.scores:
            bar = "#" * int(s.score) + "." * (10 - int(s.score))
            lines.append(
                f"- **{s.dimension.replace('_', ' ').title()}**: "
                f"{s.score:.0f}/10 [{bar}]"
            )
            if s.justification:
                lines.append(f"  {s.justification}")

        if report.strengths:
            lines.extend(["", "## Strengths", ""])
            for s in report.strengths:
                lines.append(f"- {s}")

        if report.key_issues:
            lines.extend(["", "## Key Issues", ""])
            for i in report.key_issues:
                sev = i.get("severity", "minor").upper()
                lines.append(f"- [{sev}] {i.get('description', '')}")

        if report.revision_requests:
            lines.extend(["", "## Revision Requests", ""])
            for idx, r in enumerate(report.revision_requests, 1):
                lines.append(f"{idx}. {r}")

        if report.summary:
            lines.extend(["", "## Summary", "", report.summary])

        return "\n".join(lines)
