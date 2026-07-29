"""Compatibility shim for the retired shared-secret journal integration.

Publication now happens through authenticated Frontier Science living records.
Keeping this inert class for one release gives downstream imports a clear error
instead of silently sending data to obsolete endpoints.
"""

from __future__ import annotations

import hashlib
import hmac
from typing import Any, Never

# Maps ApolloBot dimension names to journal API dimension names.
DIMENSION_MAP: dict[str, str] = {
    "statistical_rigor": "statistical",
    "methodological_soundness": "methodology",
    "reproducibility": "reproducibility",
    "novelty": "novelty",
    "clarity": "clarity",
}


class LegacyJournalIntegrationRetiredError(RuntimeError):
    """Raised when code calls the retired direct-journal integration."""


class JournalClient:
    """Inert compatibility surface for the retired direct-journal client."""

    def __init__(
        self,
        base_url: str,
        hmac_secret: str,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.hmac_secret = hmac_secret
        self.timeout = timeout

    def _sign(self, body: str) -> str:
        """Compute HMAC-SHA256 signature for a JSON body."""
        return hmac.new(
            self.hmac_secret.encode(),
            body.encode(),
            hashlib.sha256,
        ).hexdigest()

    def _headers(self, body: str) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.hmac_secret:
            headers["X-Apollo-Signature"] = f"sha256={self._sign(body)}"
        return headers

    @staticmethod
    def _retired() -> Never:
        raise LegacyJournalIntegrationRetiredError(
            "Direct shared-secret journal posting was retired in ApolloBot v0.2. "
            "Create reviews and publications through an authenticated Frontier Science "
            "living record."
        )

    @staticmethod
    def map_scores(scores: list[dict[str, Any]]) -> dict[str, int]:
        """Convert list of DimensionScore dicts to the flat {name: score} the journal expects."""
        mapped: dict[str, int] = {}
        for s in scores:
            dim = s.get("dimension", "")
            key = DIMENSION_MAP.get(dim, dim)
            mapped[key] = s.get("score", 0)
        return mapped

    async def post_ai_review(
        self,
        paper_id: str,
        review_data: dict[str, Any],
    ) -> dict[str, Any]:
        """Reject a call to the retired direct review endpoint."""
        _ = (paper_id, review_data)
        self._retired()

    async def submit_paper(
        self,
        title: str,
        abstract: str,
        track: str,
        session_id: str = "",
        submitter_email: str = "",
        authors: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Reject a call to the retired direct submission endpoint."""
        _ = (title, abstract, track, session_id, submitter_email, authors)
        self._retired()

    async def upload_manuscript(
        self,
        paper_id: str,
        file_path: str,
    ) -> dict[str, Any]:
        """Reject a call to the retired direct manuscript endpoint."""
        _ = (paper_id, file_path)
        self._retired()

    async def post_notification(
        self,
        paper_id: str,
        event: str,
        recipients: list[str],
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reject a call to the retired direct notification endpoint."""
        _ = (paper_id, event, recipients, data)
        self._retired()
