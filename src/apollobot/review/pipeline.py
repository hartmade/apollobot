"""
SubmissionPipeline — end-to-end submission processing.

Orchestrates the flow from submission receipt through AI review
to committee notification.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from apollobot.agents import LLMProvider
from apollobot.review.journal_client import JournalClient
from apollobot.review.notify import ReviewNotifier
from apollobot.review.submission import SubmissionReviewer, SubmissionReviewReport

logger = logging.getLogger(__name__)


class SubmissionPipeline:
    """Processes submissions from receipt through AI review to committee notification."""

    def __init__(
        self,
        llm: LLMProvider,
        notifier: ReviewNotifier | None = None,
        journal_client: JournalClient | None = None,
    ) -> None:
        self.reviewer = SubmissionReviewer(llm)
        self.notifier = notifier
        self.journal_client = journal_client

    async def process(self, submission: dict[str, Any]) -> dict[str, Any]:
        """
        Full pipeline:
        1. Validate submission (required fields, format)
        2. Load manuscript text
        3. Load provenance if session_id provided
        4. Run AI review via SubmissionReviewer
        5. Determine provenance badge (Gold/Silver/Bronze)
        6. Notify submitter of receipt
        7. Notify editorial committee with AI review
        8. Return structured result
        """
        # 1. Validate
        manuscript_text = submission.get("manuscript_text", "")
        manuscript_path = submission.get("manuscript_path", "")
        session_id = submission.get("session_id", "")
        title = submission.get("title", "Untitled Submission")
        track = submission.get("track", "")

        if not manuscript_text and not manuscript_path:
            return {
                "status": "error",
                "error": "No manuscript provided. Supply manuscript_text or manuscript_path.",
            }

        # 2. Load manuscript text
        if not manuscript_text and manuscript_path:
            path = Path(manuscript_path)
            if not path.exists():
                return {
                    "status": "error",
                    "error": f"Manuscript file not found: {manuscript_path}",
                }
            manuscript_text = path.read_text()

        # 3. Load provenance if session_id provided
        provenance_path = None
        if session_id:
            from apollobot.core import APOLLO_SESSIONS_DIR

            session_dir = Path(APOLLO_SESSIONS_DIR) / session_id
            prov_dir = session_dir / "provenance"
            if prov_dir.exists():
                provenance_path = prov_dir

            # Try loading manuscript from session if not provided directly
            if not manuscript_text:
                for name in ("manuscript.md", "manuscript.tex"):
                    candidate = session_dir / name
                    if candidate.exists():
                        manuscript_text = candidate.read_text()
                        break

            if not manuscript_text:
                return {
                    "status": "error",
                    "error": f"No manuscript found in session {session_id}",
                }

        # 6. Notify submitter of receipt
        if self.notifier:
            try:
                await self.notifier.submission_received(
                    title=title,
                    session_id=session_id,
                    track=track,
                )
            except Exception:
                logger.exception("Failed to send submission received notification")

        # 4+5. Run AI review (badge is determined inside reviewer)
        report = await self.reviewer.review(
            manuscript_text,
            provenance_path=provenance_path,
            session_id=session_id,
        )

        # 7. Notify editorial committee with AI review
        if self.notifier:
            try:
                await self.notifier.ai_review_complete(
                    title=title,
                    session_id=session_id,
                    recommendation=report.recommendation,
                    confidence=report.confidence,
                    provenance_badge=report.provenance_badge,
                    summary=report.summary,
                )
            except Exception:
                logger.exception("Failed to send review complete notification")

        # 8. Post to journal API if configured
        paper_id = submission.get("paper_id", "")
        if self.journal_client and paper_id:
            try:
                result_data = {
                    "recommendation": report.recommendation,
                    "confidence": report.confidence,
                    "scores": [s.model_dump() for s in report.scores],
                    "key_issues": report.key_issues,
                    "strengths": report.strengths,
                    "summary": report.summary,
                    "provenance_badge": report.provenance_badge,
                }
                await self.journal_client.post_ai_review(paper_id, result_data)
                logger.info("Posted AI review to journal for paper %s", paper_id)
            except Exception:
                logger.exception("Failed to post AI review to journal for paper %s", paper_id)

            try:
                await self.journal_client.post_notification(
                    paper_id,
                    event="ai_review_complete",
                    recipients=["submitter", "editors"],
                    data={"recommendation": report.recommendation},
                )
                logger.info("Posted notification to journal for paper %s", paper_id)
            except Exception:
                logger.exception("Failed to post notification to journal for paper %s", paper_id)

        # 9. Return structured result
        return {
            "status": "reviewed",
            "title": title,
            "session_id": session_id,
            "recommendation": report.recommendation,
            "confidence": report.confidence,
            "provenance_badge": report.provenance_badge,
            "scores": [s.model_dump() for s in report.scores],
            "key_issues": report.key_issues,
            "strengths": report.strengths,
            "revision_requests": report.revision_requests,
            "summary": report.summary,
            "report_markdown": self.reviewer.format_report(report),
        }
