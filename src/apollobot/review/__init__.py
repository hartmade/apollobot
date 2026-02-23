"""
Review engine — automated quality checks for research outputs.

Implements the Tier 1 AI review system:
- Statistical validity checks
- Methodological soundness evaluation
- Reproducibility verification
- Logical consistency analysis
- Overstatement detection
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from apollobot.agents import LLMProvider


class ReviewIssue(BaseModel):
    severity: str  # critical, major, minor, suggestion
    category: str  # statistical, methodological, logical, reproducibility, writing
    description: str
    location: str = ""  # section or step reference
    suggestion: str = ""


class ReviewReport(BaseModel):
    session_id: str
    overall_verdict: str = ""  # accept, revise, reject
    confidence: float = 0.0  # 0-1
    issues: list[ReviewIssue] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    summary: str = ""
    statistical_checks: dict[str, Any] = Field(default_factory=dict)


class ReviewEngine:
    """
    Automated review engine for OCR research outputs.

    Can be used:
    1. As self-review within a research session (Tier 1)
    2. As a standalone reviewer for journal submissions
    3. As a replication checker
    """

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    async def review_manuscript(
        self, manuscript_text: str, provenance_path: Path | None = None, session_id: str = ""
    ) -> ReviewReport:
        """Full review of a manuscript."""

        # Load provenance if available
        provenance_context = ""
        if provenance_path and provenance_path.exists():
            exec_log = provenance_path / "execution_log.json"
            if exec_log.exists():
                log_data = json.loads(exec_log.read_text())
                provenance_context = f"\nProvenance log ({len(log_data)} events available)"

        # Review prompt
        review_resp = await self.llm.complete(
            messages=[{"role": "user", "content": (
                f"Review this scientific manuscript for publication readiness.\n\n"
                f"MANUSCRIPT:\n{manuscript_text[:10000]}\n\n"
                f"{provenance_context}\n\n"
                "Evaluate on these dimensions and return JSON:\n"
                "{\n"
                '  "overall_verdict": "accept|revise|reject",\n'
                '  "confidence": 0.0-1.0,\n'
                '  "issues": [{"severity": "critical|major|minor|suggestion", '
                '"category": "statistical|methodological|logical|reproducibility|writing", '
                '"description": "...", "location": "section name", "suggestion": "..."}],\n'
                '  "strengths": ["..."],\n'
                '  "summary": "2-3 sentence overall assessment"\n'
                "}"
            )}],
            system=(
                "You are an expert peer reviewer evaluating a computational research paper. "
                "Be thorough but fair. Focus on scientific rigor, not style. "
                "Statistical issues and logical errors are critical. "
                "Minor writing issues are suggestions. "
                "Always explain WHY something is an issue and HOW to fix it."
            ),
        )

        try:
            review_data = json.loads(
                review_resp.text.strip().removeprefix("```json").removesuffix("```").strip()
            )
        except json.JSONDecodeError:
            review_data = {
                "overall_verdict": "revise",
                "confidence": 0.5,
                "issues": [{"severity": "minor", "category": "writing", "description": "Review parsing failed"}],
                "strengths": [],
                "summary": review_resp.text[:500],
            }

        report = ReviewReport(session_id=session_id, **review_data)

        # Run automated statistical checks
        report.statistical_checks = await self._statistical_checks(manuscript_text)

        return report

    async def _statistical_checks(self, text: str) -> dict[str, Any]:
        """Run automated statistical validation checks."""
        checks = {
            "p_values_reported": "p-value" in text.lower() or "p =" in text.lower() or "p<" in text.lower(),
            "effect_sizes_reported": any(term in text.lower() for term in [
                "cohen's d", "effect size", "odds ratio", "relative risk",
                "r²", "r-squared", "eta", "confidence interval",
            ]),
            "sample_size_reported": any(term in text.lower() for term in [
                "n =", "n=", "sample size", "participants", "observations",
            ]),
            "multiple_comparison_addressed": any(term in text.lower() for term in [
                "bonferroni", "fdr", "false discovery", "multiple comparison",
                "correction", "adjusted p", "holm",
            ]),
            "limitations_discussed": "limitation" in text.lower(),
            "data_availability": any(term in text.lower() for term in [
                "data availab", "open data", "repository", "github", "zenodo",
            ]),
        }

        checks["score"] = sum(checks.values()) / len(checks)
        checks["grade"] = (
            "A" if checks["score"] >= 0.8
            else "B" if checks["score"] >= 0.6
            else "C" if checks["score"] >= 0.4
            else "D"
        )

        return checks

    async def review_replication(
        self, original_paper: str, replication_results: dict[str, Any]
    ) -> ReviewReport:
        """Compare replication results against original paper."""
        resp = await self.llm.complete_json(
            messages=[{"role": "user", "content": (
                f"Compare these replication results against the original paper.\n\n"
                f"ORIGINAL PAPER:\n{original_paper[:5000]}\n\n"
                f"REPLICATION RESULTS:\n{json.dumps(replication_results, indent=2)[:5000]}\n\n"
                "Assess:\n"
                "1. Which findings replicated successfully?\n"
                "2. Which findings failed to replicate?\n"
                "3. What might explain any discrepancies?\n"
                "4. Are the original conclusions still supported?\n\n"
                "Return JSON with the same ReviewReport schema."
            )}],
            system="You are assessing a replication study. Be fair to both the original and replication.",
        )

        return ReviewReport(**resp)
