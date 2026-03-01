"""
Tests for JournalClient — HMAC signing and API posting.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apollobot.review.journal_client import DIMENSION_MAP, JournalClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_SECRET = "test-secret-key"
SAMPLE_BASE_URL = "https://journal.example.com"


@pytest.fixture
def client():
    return JournalClient(
        base_url=SAMPLE_BASE_URL,
        hmac_secret=SAMPLE_SECRET,
    )


SAMPLE_SCORES_LIST = [
    {"dimension": "statistical_rigor", "score": 6, "justification": "Missing FDR correction"},
    {"dimension": "methodological_soundness", "score": 8, "justification": "Well-designed"},
    {"dimension": "reproducibility", "score": 9, "justification": "Full provenance"},
    {"dimension": "novelty", "score": 7, "justification": "New approach"},
    {"dimension": "clarity", "score": 8, "justification": "Clear writing"},
]

SAMPLE_REVIEW_DATA = {
    "recommendation": "minor_revision",
    "confidence": 0.75,
    "scores": SAMPLE_SCORES_LIST,
    "key_issues": [{"severity": "major", "description": "No FDR correction"}],
    "strengths": ["Novel dataset", "Full provenance"],
    "summary": "Strong submission with minor statistical gaps.",
    "provenance_badge": "gold",
}


# ---------------------------------------------------------------------------
# HMAC signature tests
# ---------------------------------------------------------------------------


class TestHMACSignature:
    def test_sign_produces_correct_hmac(self, client):
        body = '{"test": true}'
        expected = hmac_mod.new(
            SAMPLE_SECRET.encode(),
            body.encode(),
            hashlib.sha256,
        ).hexdigest()
        assert client._sign(body) == expected

    def test_headers_include_signature(self, client):
        body = '{"hello": "world"}'
        headers = client._headers(body)
        assert "X-Apollo-Signature" in headers
        assert headers["X-Apollo-Signature"].startswith("sha256=")
        assert headers["Content-Type"] == "application/json"

    def test_headers_without_secret(self):
        client = JournalClient(base_url=SAMPLE_BASE_URL, hmac_secret="")
        headers = client._headers('{"test": 1}')
        assert "X-Apollo-Signature" not in headers
        assert headers["Content-Type"] == "application/json"


# ---------------------------------------------------------------------------
# Score dimension mapping tests
# ---------------------------------------------------------------------------


class TestScoreMapping:
    def test_map_scores_converts_dimensions(self):
        mapped = JournalClient.map_scores(SAMPLE_SCORES_LIST)
        assert mapped == {
            "statistical": 6,
            "methodology": 8,
            "reproducibility": 9,
            "novelty": 7,
            "clarity": 8,
        }

    def test_map_scores_preserves_unknown_dimensions(self):
        scores = [{"dimension": "unknown_dim", "score": 5}]
        mapped = JournalClient.map_scores(scores)
        assert mapped == {"unknown_dim": 5}

    def test_map_scores_empty_list(self):
        assert JournalClient.map_scores([]) == {}

    def test_dimension_map_covers_all_five(self):
        assert len(DIMENSION_MAP) == 5
        assert set(DIMENSION_MAP.values()) == {
            "statistical", "methodology", "reproducibility", "novelty", "clarity"
        }


# ---------------------------------------------------------------------------
# post_ai_review tests
# ---------------------------------------------------------------------------


class TestPostAIReview:
    @pytest.mark.asyncio
    async def test_post_ai_review_sends_correct_payload(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {"review_id": "abc-123"}
        mock_response.raise_for_status = MagicMock()

        with patch("apollobot.review.journal_client.httpx.AsyncClient") as MockClient:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_http

            result = await client.post_ai_review("paper-001", SAMPLE_REVIEW_DATA)

            assert result == {"review_id": "abc-123"}
            mock_http.post.assert_called_once()

            call_args = mock_http.post.call_args
            url = call_args[0][0]
            assert url == f"{SAMPLE_BASE_URL}/api/papers/paper-001/ai-review"

            # Verify the body has mapped scores
            sent_body = json.loads(call_args[1]["content"])
            assert sent_body["scores"] == {
                "statistical": 6,
                "methodology": 8,
                "reproducibility": 9,
                "novelty": 7,
                "clarity": 8,
            }
            assert sent_body["recommendation"] == "minor_revision"
            assert sent_body["confidence"] == 0.75

            # Verify HMAC header
            headers = call_args[1]["headers"]
            assert "X-Apollo-Signature" in headers

    @pytest.mark.asyncio
    async def test_post_ai_review_with_dict_scores(self, client):
        """If scores are already a dict (not list), they pass through."""
        review_data = {
            **SAMPLE_REVIEW_DATA,
            "scores": {"statistical": 6, "methodology": 8, "reproducibility": 9, "novelty": 7, "clarity": 8},
        }

        mock_response = MagicMock()
        mock_response.json.return_value = {"review_id": "def-456"}
        mock_response.raise_for_status = MagicMock()

        with patch("apollobot.review.journal_client.httpx.AsyncClient") as MockClient:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_http

            result = await client.post_ai_review("paper-002", review_data)
            assert result == {"review_id": "def-456"}


# ---------------------------------------------------------------------------
# post_notification tests
# ---------------------------------------------------------------------------


class TestPostNotification:
    @pytest.mark.asyncio
    async def test_post_notification_sends_event(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status = MagicMock()

        with patch("apollobot.review.journal_client.httpx.AsyncClient") as MockClient:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_http

            result = await client.post_notification(
                "paper-001",
                event="ai_review_complete",
                recipients=["submitter", "editors"],
                data={"recommendation": "minor_revision"},
            )

            assert result == {"success": True}

            call_args = mock_http.post.call_args
            url = call_args[0][0]
            assert url == f"{SAMPLE_BASE_URL}/api/papers/paper-001/notify"

            sent_body = json.loads(call_args[1]["content"])
            assert sent_body["event"] == "ai_review_complete"
            assert sent_body["recipients"] == ["submitter", "editors"]
            assert sent_body["data"]["recommendation"] == "minor_revision"

    @pytest.mark.asyncio
    async def test_post_notification_without_data(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {"success": True}
        mock_response.raise_for_status = MagicMock()

        with patch("apollobot.review.journal_client.httpx.AsyncClient") as MockClient:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_http

            await client.post_notification(
                "paper-001",
                event="submission_received",
                recipients=["editors"],
            )

            sent_body = json.loads(mock_http.post.call_args[1]["content"])
            assert "data" not in sent_body


# ---------------------------------------------------------------------------
# submit_paper tests
# ---------------------------------------------------------------------------


class TestSubmitPaper:
    @pytest.mark.asyncio
    async def test_submit_paper_sends_correct_payload(self, client):
        mock_response = MagicMock()
        mock_response.json.return_value = {"paper": {"id": "paper-new", "title": "Test"}}
        mock_response.raise_for_status = MagicMock()

        with patch("apollobot.review.journal_client.httpx.AsyncClient") as MockClient:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_http

            result = await client.submit_paper(
                title="Test Paper",
                abstract="An abstract.",
                track="bioinformatics",
                session_id="session-abc",
                submitter_email="user@example.com",
            )

            assert result["paper"]["id"] == "paper-new"

            call_args = mock_http.post.call_args
            url = call_args[0][0]
            assert url == f"{SAMPLE_BASE_URL}/api/papers/cli-submit"

            sent_body = json.loads(call_args[1]["content"])
            assert sent_body["title"] == "Test Paper"
            assert sent_body["abstract"] == "An abstract."
            assert sent_body["track"] == "bioinformatics"
            assert sent_body["sessionId"] == "session-abc"
            assert sent_body["submitterEmail"] == "user@example.com"

            # HMAC header present
            headers = call_args[1]["headers"]
            assert "X-Apollo-Signature" in headers

    @pytest.mark.asyncio
    async def test_submit_paper_minimal(self, client):
        """Submit with only required fields."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"paper": {"id": "paper-min"}}
        mock_response.raise_for_status = MagicMock()

        with patch("apollobot.review.journal_client.httpx.AsyncClient") as MockClient:
            mock_http = AsyncMock()
            mock_http.post = AsyncMock(return_value=mock_response)
            mock_http.__aenter__ = AsyncMock(return_value=mock_http)
            mock_http.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_http

            result = await client.submit_paper(
                title="Minimal",
                abstract="Short.",
                track="physics",
            )

            assert result["paper"]["id"] == "paper-min"
            sent_body = json.loads(mock_http.post.call_args[1]["content"])
            assert "sessionId" not in sent_body
            assert "submitterEmail" not in sent_body
