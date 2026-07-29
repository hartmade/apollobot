"""Tests for the intentionally retired direct journal compatibility surface."""

from __future__ import annotations

import hashlib
import hmac as hmac_mod

import pytest

from apollobot.review.journal_client import (
    DIMENSION_MAP,
    JournalClient,
    LegacyJournalIntegrationRetiredError,
)

SAMPLE_SECRET = "test-secret-key"  # noqa: S105
SAMPLE_SCORES = [
    {"dimension": "statistical_rigor", "score": 6},
    {"dimension": "methodological_soundness", "score": 8},
    {"dimension": "reproducibility", "score": 9},
    {"dimension": "novelty", "score": 7},
    {"dimension": "clarity", "score": 8},
]


@pytest.fixture
def client() -> JournalClient:
    return JournalClient("https://journal.example.com", SAMPLE_SECRET)


def test_signing_helpers_remain_stable_for_downstream_imports(client: JournalClient) -> None:
    body = '{"test": true}'
    expected = hmac_mod.new(SAMPLE_SECRET.encode(), body.encode(), hashlib.sha256).hexdigest()
    assert client._sign(body) == expected
    assert client._headers(body)["X-Apollo-Signature"] == f"sha256={expected}"


def test_score_mapping_remains_stable_for_local_review_exports() -> None:
    assert JournalClient.map_scores(SAMPLE_SCORES) == {
        "statistical": 6,
        "methodology": 8,
        "reproducibility": 9,
        "novelty": 7,
        "clarity": 8,
    }
    assert len(DIMENSION_MAP) == 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    [
        lambda client: client.post_ai_review("paper", {}),
        lambda client: client.submit_paper("title", "abstract", "track"),
        lambda client: client.upload_manuscript("paper", "manuscript.md"),
        lambda client: client.post_notification("paper", "reviewed", ["editors"]),
    ],
)
async def test_all_network_mutations_fail_closed(client: JournalClient, operation) -> None:
    with pytest.raises(LegacyJournalIntegrationRetiredError, match="retired"):
        await operation(client)
