"""
Tests for the AI Submission Reviewer pipeline.

Tests SubmissionReviewer, ReviewNotifier, and SubmissionPipeline
using mock LLM responses.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apollobot.agents import LLMResponse
from apollobot.notifications.events import EventType, NotificationEvent
from apollobot.notifications.router import NotificationRouter
from apollobot.review import ReviewReport
from apollobot.review.notify import ReviewNotifier
from apollobot.review.submission import (
    DimensionScore,
    ProvenanceBadge,
    SubmissionReviewer,
    SubmissionReviewReport,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


MOCK_REVIEW_JSON = json.dumps({
    "overall_verdict": "revise",
    "confidence": 0.7,
    "issues": [
        {
            "severity": "major",
            "category": "statistical",
            "description": "No multiple comparison correction applied",
            "location": "Results",
            "suggestion": "Apply FDR correction",
        }
    ],
    "strengths": ["Novel dataset", "Clear writing"],
    "summary": "Solid work with some statistical issues.",
})

MOCK_SCORING_JSON = json.dumps({
    "recommendation": "minor_revision",
    "confidence": 0.75,
    "scores": [
        {"dimension": "statistical_rigor", "score": 6, "justification": "Missing FDR correction"},
        {"dimension": "methodological_soundness", "score": 8, "justification": "Well-designed study"},
        {"dimension": "reproducibility", "score": 9, "justification": "Full provenance chain"},
        {"dimension": "novelty", "score": 7, "justification": "New approach to known problem"},
        {"dimension": "clarity", "score": 8, "justification": "Clear and well-structured"},
    ],
    "key_issues": [{"severity": "major", "description": "No FDR correction"}],
    "strengths": ["Novel dataset", "Full provenance"],
    "revision_requests": ["Apply Benjamini-Hochberg FDR correction", "Add effect size table"],
    "summary": "Strong submission with minor statistical gaps.",
})


def _make_mock_llm(*responses: str) -> MagicMock:
    """Create a mock LLM that returns the given responses in order."""
    llm = MagicMock()
    side_effects = [
        LLMResponse(
            text=r,
            provider="mock",
            model="mock-v1",
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.01,
        )
        for r in responses
    ]
    llm.complete = AsyncMock(side_effect=side_effects)
    llm.complete_json = AsyncMock(side_effect=[json.loads(r) for r in responses])
    return llm


# ---------------------------------------------------------------------------
# SubmissionReviewer tests
# ---------------------------------------------------------------------------


class TestSubmissionReviewer:
    @pytest.fixture
    def reviewer(self):
        llm = _make_mock_llm(MOCK_REVIEW_JSON, MOCK_SCORING_JSON)
        return SubmissionReviewer(llm)

    @pytest.mark.asyncio
    async def test_review_returns_report(self, reviewer):
        report = await reviewer.review("# Test Manuscript\n\nSome content here.")
        assert isinstance(report, SubmissionReviewReport)
        assert report.recommendation == "minor_revision"
        assert report.confidence == 0.75

    @pytest.mark.asyncio
    async def test_review_has_scores(self, reviewer):
        report = await reviewer.review("# Test Manuscript\n\nSome content here.")
        assert len(report.scores) == 5
        dimensions = {s.dimension for s in report.scores}
        assert "statistical_rigor" in dimensions
        assert "reproducibility" in dimensions

    @pytest.mark.asyncio
    async def test_review_has_issues_and_strengths(self, reviewer):
        report = await reviewer.review("# Test Manuscript")
        assert len(report.key_issues) > 0
        assert len(report.strengths) > 0

    @pytest.mark.asyncio
    async def test_review_has_revision_requests(self, reviewer):
        report = await reviewer.review("# Test Manuscript")
        assert len(report.revision_requests) == 2
        assert "FDR" in report.revision_requests[0]

    @pytest.mark.asyncio
    async def test_review_includes_base_review(self, reviewer):
        report = await reviewer.review("# Test Manuscript")
        assert report.base_review is not None
        assert isinstance(report.base_review, ReviewReport)


class TestProvenanceBadge:
    def test_no_session_is_bronze(self):
        badge = SubmissionReviewer._assess_provenance(None, "")
        assert badge == ProvenanceBadge.BRONZE

    def test_no_path_is_bronze(self):
        badge = SubmissionReviewer._assess_provenance(None, "session-123")
        assert badge == ProvenanceBadge.BRONZE

    def test_nonexistent_path_is_bronze(self):
        badge = SubmissionReviewer._assess_provenance(
            Path("/nonexistent/path"), "session-123"
        )
        assert badge == ProvenanceBadge.BRONZE

    def test_full_provenance_is_gold(self, tmp_path):
        prov = tmp_path / "provenance"
        prov.mkdir()
        (prov / "execution_log.json").write_text("[]")
        (prov / "model_calls.json").write_text("[]")
        (prov / "data_lineage.json").write_text("[]")
        badge = SubmissionReviewer._assess_provenance(prov, "session-123")
        assert badge == ProvenanceBadge.GOLD

    def test_partial_provenance_is_silver(self, tmp_path):
        prov = tmp_path / "provenance"
        prov.mkdir()
        (prov / "execution_log.json").write_text("[]")
        badge = SubmissionReviewer._assess_provenance(prov, "session-123")
        assert badge == ProvenanceBadge.SILVER

    def test_empty_provenance_dir_is_bronze(self, tmp_path):
        prov = tmp_path / "provenance"
        prov.mkdir()
        badge = SubmissionReviewer._assess_provenance(prov, "session-123")
        assert badge == ProvenanceBadge.BRONZE


class TestFormatReport:
    def test_format_produces_markdown(self):
        llm = _make_mock_llm()
        reviewer = SubmissionReviewer(llm)
        report = SubmissionReviewReport(
            recommendation="minor_revision",
            confidence=0.8,
            provenance_badge="gold",
            scores=[
                DimensionScore(dimension="novelty", score=8, justification="Good"),
            ],
            strengths=["Clear writing"],
            key_issues=[{"severity": "minor", "description": "Typo in abstract"}],
            revision_requests=["Fix typo"],
            summary="Good paper.",
        )
        md = reviewer.format_report(report)
        assert "# Submission Review Report" in md
        assert "Minor Revision" in md
        assert "Gold badge" in md
        assert "Novelty" in md
        assert "8/10" in md
        assert "Fix typo" in md


# ---------------------------------------------------------------------------
# ReviewNotifier tests
# ---------------------------------------------------------------------------


class TestReviewNotifier:
    @pytest.fixture
    def notifier(self):
        router = NotificationRouter()
        channel = MagicMock()
        channel.name = "test_channel"
        channel.supports_responses = False
        channel.send = AsyncMock()
        router.register(channel)
        return ReviewNotifier(router), channel

    @pytest.mark.asyncio
    async def test_submission_received(self, notifier):
        notifier_obj, channel = notifier
        await notifier_obj.submission_received(
            title="Test Paper", session_id="s-123", track="bioinformatics"
        )
        channel.send.assert_called_once()
        event = channel.send.call_args[0][0]
        assert "Test Paper" in event.title
        assert "bioinformatics" in event.summary

    @pytest.mark.asyncio
    async def test_ai_review_complete(self, notifier):
        notifier_obj, channel = notifier
        await notifier_obj.ai_review_complete(
            title="Test Paper",
            session_id="s-123",
            recommendation="minor_revision",
            confidence=0.8,
            provenance_badge="gold",
            summary="Good paper.",
        )
        channel.send.assert_called_once()
        event = channel.send.call_args[0][0]
        assert "Minor Revision" in event.summary
        assert "Gold" in event.summary

    @pytest.mark.asyncio
    async def test_revision_requested(self, notifier):
        notifier_obj, channel = notifier
        await notifier_obj.revision_requested(
            title="Test Paper",
            revision_requests=["Fix stats", "Add table", "Clarify methods"],
        )
        channel.send.assert_called_once()
        event = channel.send.call_args[0][0]
        assert "Fix stats" in event.summary

    @pytest.mark.asyncio
    async def test_decision_made(self, notifier):
        notifier_obj, channel = notifier
        await notifier_obj.decision_made(
            title="Test Paper",
            decision="accept",
            summary="Accepted for publication.",
        )
        channel.send.assert_called_once()
        event = channel.send.call_args[0][0]
        assert "Accept" in event.title


# ---------------------------------------------------------------------------
# SubmissionPipeline tests (unit, with mock LLM)
# ---------------------------------------------------------------------------


class TestSubmissionPipeline:
    @pytest.fixture
    def pipeline(self):
        llm = _make_mock_llm(MOCK_REVIEW_JSON, MOCK_SCORING_JSON)
        return llm

    @pytest.mark.asyncio
    async def test_process_with_manuscript_text(self, pipeline):
        from apollobot.review.pipeline import SubmissionPipeline

        pipe = SubmissionPipeline(pipeline)
        result = await pipe.process({
            "manuscript_text": "# Test Paper\n\nContent here.",
            "title": "Test Paper",
        })
        assert result["status"] == "reviewed"
        assert result["recommendation"] == "minor_revision"
        assert "report_markdown" in result

    @pytest.mark.asyncio
    async def test_process_with_manuscript_file(self, pipeline, tmp_path):
        from apollobot.review.pipeline import SubmissionPipeline

        ms = tmp_path / "paper.md"
        ms.write_text("# Test Paper\n\nContent.")
        pipe = SubmissionPipeline(pipeline)
        result = await pipe.process({
            "manuscript_path": str(ms),
            "title": "File Paper",
        })
        assert result["status"] == "reviewed"

    @pytest.mark.asyncio
    async def test_process_no_manuscript_errors(self, pipeline):
        from apollobot.review.pipeline import SubmissionPipeline

        pipe = SubmissionPipeline(pipeline)
        result = await pipe.process({"title": "Empty"})
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_process_missing_file_errors(self, pipeline):
        from apollobot.review.pipeline import SubmissionPipeline

        pipe = SubmissionPipeline(pipeline)
        result = await pipe.process({
            "manuscript_path": "/nonexistent/paper.md",
        })
        assert result["status"] == "error"

    @pytest.mark.asyncio
    async def test_process_with_notifier(self, pipeline):
        from apollobot.review.pipeline import SubmissionPipeline

        router = NotificationRouter()
        channel = MagicMock()
        channel.name = "test"
        channel.supports_responses = False
        channel.send = AsyncMock()
        router.register(channel)
        notifier = ReviewNotifier(router)

        pipe = SubmissionPipeline(pipeline, notifier=notifier)
        result = await pipe.process({
            "manuscript_text": "# Paper\n\nContent.",
            "title": "Notified Paper",
        })
        assert result["status"] == "reviewed"
        # Should have sent at least 2 notifications (received + review complete)
        assert channel.send.call_count >= 2

    @pytest.mark.asyncio
    async def test_process_with_journal_client(self, pipeline):
        from apollobot.review.journal_client import JournalClient
        from apollobot.review.pipeline import SubmissionPipeline

        journal = MagicMock(spec=JournalClient)
        journal.post_ai_review = AsyncMock(return_value={"review_id": "r-1"})
        journal.post_notification = AsyncMock(return_value={"success": True})

        pipe = SubmissionPipeline(pipeline, journal_client=journal)
        result = await pipe.process({
            "manuscript_text": "# Paper\n\nContent.",
            "title": "Journal Paper",
            "paper_id": "paper-abc",
        })
        assert result["status"] == "reviewed"
        journal.post_ai_review.assert_called_once()
        journal.post_notification.assert_called_once()
        # Verify paper_id was passed to post_ai_review
        call_args = journal.post_ai_review.call_args
        assert call_args[0][0] == "paper-abc"

    @pytest.mark.asyncio
    async def test_process_journal_failure_does_not_kill_review(self, pipeline):
        from apollobot.review.journal_client import JournalClient
        from apollobot.review.pipeline import SubmissionPipeline

        journal = MagicMock(spec=JournalClient)
        journal.post_ai_review = AsyncMock(side_effect=Exception("Connection refused"))
        journal.post_notification = AsyncMock(side_effect=Exception("Connection refused"))

        pipe = SubmissionPipeline(pipeline, journal_client=journal)
        result = await pipe.process({
            "manuscript_text": "# Paper\n\nContent.",
            "title": "Failing Journal",
            "paper_id": "paper-xyz",
        })
        # Review still succeeds even though journal calls failed
        assert result["status"] == "reviewed"
        assert result["recommendation"] == "minor_revision"

    @pytest.mark.asyncio
    async def test_process_without_paper_id_skips_journal(self, pipeline):
        from apollobot.review.journal_client import JournalClient
        from apollobot.review.pipeline import SubmissionPipeline

        journal = MagicMock(spec=JournalClient)
        journal.post_ai_review = AsyncMock()
        journal.post_notification = AsyncMock()

        pipe = SubmissionPipeline(pipeline, journal_client=journal)
        result = await pipe.process({
            "manuscript_text": "# Paper\n\nContent.",
            "title": "No Paper ID",
        })
        assert result["status"] == "reviewed"
        # Journal should NOT be called when no paper_id
        journal.post_ai_review.assert_not_called()
        journal.post_notification.assert_not_called()
