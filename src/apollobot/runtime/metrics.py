"""
Research quality metrics and reputation scoring.

Tracks research output quality over time, inspired by OpenCat's
reputation scoring (uptime, earnings consistency, provenance, compliance).

ApolloBot equivalent dimensions:
- Productivity: papers per day, sessions completed vs failed
- Quality: average translation score, self-review pass rate
- Efficiency: cost per paper, budget utilization
- Consistency: variance in output quality over time
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from apollobot.runtime.storage import RunnerStorage


@dataclass
class ResearchMetrics:
    """Computed metrics snapshot for the runtime."""

    # Productivity
    total_sessions: int = 0
    completed_sessions: int = 0
    failed_sessions: int = 0
    completion_rate: float = 0.0
    papers_per_day: float = 0.0

    # Quality
    avg_translation_score: float = 0.0
    high_quality_papers: int = 0  # translation score >= 7

    # Efficiency
    total_cost_usd: float = 0.0
    avg_cost_per_paper: float = 0.0

    # Reputation (0-100)
    reputation_score: float = 0.0

    # Computed at
    computed_at: str = ""


def compute_metrics(storage: RunnerStorage) -> ResearchMetrics:
    """Compute current research metrics from storage."""
    completed = storage.get_completed_sessions(limit=1000)
    failed = storage.get_failed_sessions(limit=1000)

    total = len(completed) + len(failed)
    if total == 0:
        return ResearchMetrics(computed_at=datetime.now(timezone.utc).isoformat())

    completion_rate = len(completed) / total if total > 0 else 0.0

    # Cost
    total_cost = sum(s.cost_usd for s in completed)
    avg_cost = total_cost / len(completed) if completed else 0.0

    # Quality
    scores = [s.translation_score for s in completed if s.translation_score > 0]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    high_quality = sum(1 for s in completed if s.translation_score >= 7.0)

    # Papers per day
    if completed:
        dates = [s.completed_at for s in completed if s.completed_at]
        if len(dates) >= 2:
            try:
                first = datetime.fromisoformat(min(dates))
                last = datetime.fromisoformat(max(dates))
                days = max((last - first).total_seconds() / 86400, 1.0)
                papers_per_day = len(completed) / days
            except (ValueError, TypeError):
                papers_per_day = 0.0
        else:
            papers_per_day = float(len(completed))
    else:
        papers_per_day = 0.0

    # Reputation: weighted composite (0-100)
    # 40% completion rate, 30% quality, 20% efficiency, 10% volume
    rep_completion = completion_rate * 40
    rep_quality = min(avg_score / 10, 1.0) * 30
    rep_efficiency = max(0, 1.0 - avg_cost / 50.0) * 20 if avg_cost > 0 else 10.0
    rep_volume = min(len(completed) / 50, 1.0) * 10
    reputation = rep_completion + rep_quality + rep_efficiency + rep_volume

    return ResearchMetrics(
        total_sessions=total,
        completed_sessions=len(completed),
        failed_sessions=len(failed),
        completion_rate=round(completion_rate, 3),
        papers_per_day=round(papers_per_day, 2),
        avg_translation_score=round(avg_score, 2),
        high_quality_papers=high_quality,
        total_cost_usd=round(total_cost, 2),
        avg_cost_per_paper=round(avg_cost, 2),
        reputation_score=round(min(reputation, 100.0), 1),
        computed_at=datetime.now(timezone.utc).isoformat(),
    )
