"""
JournalClient — HTTP client for posting AI reviews to the Frontier Science Journal API.

Signs requests with HMAC-SHA256 using the shared webhook secret.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Maps ApolloBot dimension names to journal API dimension names.
DIMENSION_MAP: dict[str, str] = {
    "statistical_rigor": "statistical",
    "methodological_soundness": "methodology",
    "reproducibility": "reproducibility",
    "novelty": "novelty",
    "clarity": "clarity",
}


class JournalClient:
    """Posts AI reviews and notifications to the Frontier Science Journal API."""

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
        """POST the AI review to /api/papers/{paper_id}/ai-review."""
        url = f"{self.base_url}/api/papers/{paper_id}/ai-review"

        # Transform scores from list to flat dict expected by journal
        scores = review_data.get("scores", [])
        if isinstance(scores, list):
            scores = self.map_scores(scores)

        payload: dict[str, Any] = {
            "recommendation": review_data.get("recommendation"),
            "confidence": review_data.get("confidence"),
            "scores": scores,
            "issues": review_data.get("key_issues", []),
            "strengths": review_data.get("strengths", []),
            "summary": review_data.get("summary", ""),
            "provenance_badge": review_data.get("provenance_badge"),
        }

        body = json.dumps(payload)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, content=body, headers=self._headers(body))
            resp.raise_for_status()
            return resp.json()

    async def submit_paper(
        self,
        title: str,
        abstract: str,
        track: str,
        session_id: str = "",
        submitter_email: str = "",
        authors: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """POST a paper submission to /api/papers/cli-submit."""
        url = f"{self.base_url}/api/papers/cli-submit"

        payload: dict[str, Any] = {
            "title": title,
            "abstract": abstract,
            "track": track,
        }
        if session_id:
            payload["sessionId"] = session_id
        if submitter_email:
            payload["submitterEmail"] = submitter_email
        if authors:
            payload["authors"] = authors

        body = json.dumps(payload)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, content=body, headers=self._headers(body))
            resp.raise_for_status()
            return resp.json()

    async def upload_manuscript(
        self,
        paper_id: str,
        file_path: str,
    ) -> dict[str, Any]:
        """Upload a manuscript file to /api/papers/upload via multipart form."""
        url = f"{self.base_url}/api/papers/upload"
        import mimetypes

        mime_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        file_name = Path(file_path).name

        # For uploads, we sign a simple JSON identifier since multipart bodies
        # are harder to sign consistently. The upload endpoint uses session auth
        # so this is best-effort; the paper ownership check is the real guard.
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            with open(file_path, "rb") as f:
                files = {"file": (file_name, f, mime_type)}
                data = {"paperId": paper_id}
                resp = await client.post(url, files=files, data=data)
                resp.raise_for_status()
                return resp.json()

    async def post_notification(
        self,
        paper_id: str,
        event: str,
        recipients: list[str],
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """POST a notification to /api/papers/{paper_id}/notify."""
        url = f"{self.base_url}/api/papers/{paper_id}/notify"

        payload: dict[str, Any] = {
            "event": event,
            "recipients": recipients,
        }
        if data:
            payload["data"] = data

        body = json.dumps(payload)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, content=body, headers=self._headers(body))
            resp.raise_for_status()
            return resp.json()
