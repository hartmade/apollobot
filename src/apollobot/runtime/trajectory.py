"""
Research trajectory analysis — cross-session learning for the brain.

Analyzes completed sessions to extract patterns about what works:
- Which domains are most productive?
- Which question-framing patterns yield high-quality papers?
- What cost/quality trade-offs exist?
- What research gaps remain in the domain?

This feeds insights back into the brain's memory, enabling the
runtime to improve over time.
"""

from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from apollobot.runtime.storage import RunnerStorage
from apollobot.runtime.types import SessionSummary

logger = logging.getLogger(__name__)


@dataclass
class DomainInsight:
    """Aggregated insight for a single research domain."""

    domain: str
    total_sessions: int = 0
    completed: int = 0
    failed: int = 0
    avg_quality: float = 0.0
    avg_cost: float = 0.0
    best_objectives: list[str] = field(default_factory=list)
    common_failure_modes: list[str] = field(default_factory=list)


@dataclass
class TrajectoryAnalysis:
    """Full trajectory analysis result fed to the brain."""

    # Per-domain breakdown
    domain_insights: list[DomainInsight] = field(default_factory=list)

    # Cross-domain patterns
    best_performing_domains: list[str] = field(default_factory=list)
    underexplored_domains: list[str] = field(default_factory=list)
    avg_quality_trend: str = ""  # "improving", "stable", "declining"
    cost_efficiency_trend: str = ""  # "improving", "stable", "declining"

    # Actionable recommendations
    recommendations: list[str] = field(default_factory=list)

    # Stats
    total_papers: int = 0
    total_cost: float = 0.0
    overall_quality: float = 0.0


class ResearchTrajectory:
    """
    Analyzes research history to generate actionable insights.

    Call `analyze()` to produce a TrajectoryAnalysis, then format it
    as a prompt section for the brain via `format_for_brain()`.
    """

    # Domains the runtime knows about (from guardrails config defaults)
    ALL_DOMAINS = [
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

    def __init__(self, storage: RunnerStorage) -> None:
        self.storage = storage

    def analyze(self, approved_domains: list[str] | None = None) -> TrajectoryAnalysis:
        """Analyze all completed and failed sessions."""
        completed = self.storage.get_completed_sessions(limit=10000)
        failed = self.storage.get_failed_sessions(limit=10000)

        if not completed and not failed:
            return TrajectoryAnalysis(
                recommendations=["No research history yet. Start with a broad exploratory scan."]
            )

        all_sessions = completed + failed
        domains_used = approved_domains or self.ALL_DOMAINS

        # Per-domain analysis
        domain_groups: dict[str, list[SessionSummary]] = defaultdict(list)
        for s in all_sessions:
            domain_groups[s.domain].append(s)

        domain_insights = []
        for domain in sorted(domain_groups.keys()):
            sessions = domain_groups[domain]
            completed_in_domain = [s for s in sessions if s.phase == "complete"]
            failed_in_domain = [s for s in sessions if s.phase == "failed"]

            scores = [s.translation_score for s in completed_in_domain if s.translation_score > 0]
            costs = [s.cost_usd for s in completed_in_domain if s.cost_usd > 0]

            # Best objectives (by translation score)
            scored_objectives = sorted(
                [
                    (s.translation_score, s.objective)
                    for s in completed_in_domain
                    if s.translation_score > 0
                ],
                reverse=True,
            )

            insight = DomainInsight(
                domain=domain,
                total_sessions=len(sessions),
                completed=len(completed_in_domain),
                failed=len(failed_in_domain),
                avg_quality=round(sum(scores) / len(scores), 2) if scores else 0.0,
                avg_cost=round(sum(costs) / len(costs), 2) if costs else 0.0,
                best_objectives=[obj[:100] for _, obj in scored_objectives[:3]],
            )
            domain_insights.append(insight)

        # Cross-domain rankings
        scored_domains = [
            (d.avg_quality, d.domain)
            for d in domain_insights
            if d.completed >= 2 and d.avg_quality > 0
        ]
        scored_domains.sort(reverse=True)
        best_performing = [domain for _, domain in scored_domains[:3]]

        explored_domains = set(domain_groups.keys())
        underexplored = [d for d in domains_used if d not in explored_domains]

        # Quality trend (compare first half vs second half of completed sessions)
        quality_trend = "stable"
        if len(completed) >= 6:
            mid = len(completed) // 2
            first_half_scores = [
                s.translation_score for s in completed[:mid] if s.translation_score > 0
            ]
            second_half_scores = [
                s.translation_score for s in completed[mid:] if s.translation_score > 0
            ]
            if first_half_scores and second_half_scores:
                first_avg = sum(first_half_scores) / len(first_half_scores)
                second_avg = sum(second_half_scores) / len(second_half_scores)
                if second_avg > first_avg + 0.5:
                    quality_trend = "improving"
                elif second_avg < first_avg - 0.5:
                    quality_trend = "declining"

        # Cost efficiency trend
        cost_trend = "stable"
        if len(completed) >= 6:
            mid = len(completed) // 2
            first_costs = [s.cost_usd for s in completed[:mid] if s.cost_usd > 0]
            second_costs = [s.cost_usd for s in completed[mid:] if s.cost_usd > 0]
            if first_costs and second_costs:
                first_avg_cost = sum(first_costs) / len(first_costs)
                second_avg_cost = sum(second_costs) / len(second_costs)
                if second_avg_cost < first_avg_cost * 0.85:
                    cost_trend = "improving"
                elif second_avg_cost > first_avg_cost * 1.15:
                    cost_trend = "declining"

        # Generate recommendations
        recommendations = self._generate_recommendations(
            domain_insights,
            best_performing,
            underexplored,
            quality_trend,
            cost_trend,
            completed,
            failed,
        )

        total_cost = sum(s.cost_usd for s in completed)
        all_scores = [s.translation_score for s in completed if s.translation_score > 0]

        return TrajectoryAnalysis(
            domain_insights=domain_insights,
            best_performing_domains=best_performing,
            underexplored_domains=underexplored[:5],
            avg_quality_trend=quality_trend,
            cost_efficiency_trend=cost_trend,
            recommendations=recommendations,
            total_papers=len(completed),
            total_cost=round(total_cost, 2),
            overall_quality=round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0,
        )

    def _generate_recommendations(
        self,
        insights: list[DomainInsight],
        best_domains: list[str],
        underexplored: list[str],
        quality_trend: str,
        cost_trend: str,
        completed: list[SessionSummary],
        failed: list[SessionSummary],
    ) -> list[str]:
        """Generate actionable research recommendations."""
        recs: list[str] = []

        # Domain recommendations
        if best_domains:
            recs.append(
                f"Your strongest domains are {', '.join(best_domains)}. "
                f"Consider deeper dives here for higher-quality output."
            )

        if underexplored:
            recs.append(
                f"Unexplored domains: {', '.join(underexplored[:3])}. "
                f"A scan in these areas could reveal new opportunities."
            )

        # Failure rate
        total = len(completed) + len(failed)
        if total > 5:
            fail_rate = len(failed) / total
            if fail_rate > 0.3:
                recs.append(
                    f"Failure rate is {fail_rate:.0%}. Consider more conservative "
                    f"objectives or literature scans before full research sessions."
                )

        # Quality trend
        if quality_trend == "declining":
            recs.append(
                "Quality trend is declining. Consider reviewing recent objectives "
                "and focusing on well-scoped, specific questions."
            )
        elif quality_trend == "improving":
            recs.append("Quality is improving. Current research strategy is working well.")

        # Cost efficiency
        if cost_trend == "declining":
            recs.append(
                "Cost per paper is increasing. Consider literature scans before "
                "committing to full research sessions."
            )

        # High-quality pattern analysis
        high_quality = [s for s in completed if s.translation_score >= 8.0]
        if len(high_quality) >= 3:
            modes = Counter(s.mode for s in high_quality)
            best_mode = modes.most_common(1)[0][0] if modes else None
            if best_mode:
                recs.append(
                    f"High-quality papers most often use '{best_mode}' mode. "
                    f"Favor this mode for ambitious objectives."
                )

        if not recs:
            recs.append("Keep researching. More data needed for meaningful recommendations.")

        return recs

    def format_for_brain(self, analysis: TrajectoryAnalysis) -> str:
        """Format trajectory analysis as a prompt section for the brain."""
        if analysis.total_papers == 0:
            return "No research history yet."

        lines = [
            f"Trajectory: {analysis.total_papers} papers, "
            f"${analysis.total_cost:.2f} total, "
            f"quality {analysis.overall_quality}/10",
            f"Quality trend: {analysis.avg_quality_trend}",
            f"Cost trend: {analysis.cost_efficiency_trend}",
        ]

        if analysis.best_performing_domains:
            lines.append(f"Best domains: {', '.join(analysis.best_performing_domains)}")

        if analysis.underexplored_domains:
            lines.append(f"Unexplored: {', '.join(analysis.underexplored_domains)}")

        if analysis.domain_insights:
            lines.append("Domain breakdown:")
            for d in sorted(analysis.domain_insights, key=lambda x: x.avg_quality, reverse=True):
                lines.append(
                    f"  {d.domain}: {d.completed} papers, "
                    f"quality={d.avg_quality}, cost=${d.avg_cost}"
                )

        lines.append("Recommendations:")
        for r in analysis.recommendations:
            lines.append(f"  - {r}")

        return "\n".join(lines)
