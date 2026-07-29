"""Leased automated integrity review worker for Frontier research records."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import socket
import time
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx

from apollobot import __version__
from apollobot.agents import create_llm
from apollobot.core import ApolloConfig
from apollobot.review.submission import SubmissionReviewer

logger = logging.getLogger(__name__)


class AutomatedReviewWorker:
    def __init__(
        self,
        config: ApolloConfig,
        platform_url: str,
        secret: str,
        output_dir: str | Path,
        reviewer: SubmissionReviewer | None = None,
    ) -> None:
        key = config.api.get_key()
        if reviewer is None and not key:
            raise ValueError("An LLM API key is required for automated record review")
        self.reviewer = reviewer or SubmissionReviewer(create_llm(config.api.default_provider, key))
        self.provider = config.api.default_provider
        self.model = {
            "anthropic": "claude-sonnet-4-20250514",
            "openai": os.getenv("OPENAI_MODEL", "gpt-4o"),
            "minimax": "MiniMax-M2.5",
            "acceptance": "deterministic-scientific-fixture-v1",
        }.get(self.provider, "unknown")
        self.endpoint = f"{platform_url.rstrip('/')}/api/apollobot/reviews"
        self.secret = secret.encode()
        self.output_dir = Path(output_dir).resolve()
        self.worker_id = os.getenv(
            "APOLLOBOT_REVIEW_WORKER_ID",
            f"{socket.gethostname()}-{uuid4().hex[:10]}",
        )
        self.task: asyncio.Task[None] | None = None
        self.stopping = False

    async def start(self) -> None:
        if self.task and not self.task.done():
            return
        self.stopping = False
        self.task = asyncio.create_task(self._loop(), name="frontier-review-worker")

    async def stop(self) -> None:
        self.stopping = True
        if self.task:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task

    async def _loop(self) -> None:
        async with httpx.AsyncClient(timeout=20) as client:
            while not self.stopping:
                processed = await self._run_once(client)
                await asyncio.sleep(1 if processed else 8)

    async def _run_once(self, client: httpx.AsyncClient) -> bool:
        try:
            claimed = await self._signed_post(
                client,
                {"stage": "claim", "worker_id": self.worker_id},
            )
        except httpx.HTTPError as error:
            logger.warning("Frontier review queue unavailable (%s)", type(error).__name__)
            return False
        if claimed.status_code == 204:
            return False
        if claimed.status_code >= 300:
            logger.warning("Frontier review claim rejected: %s", claimed.status_code)
            return False
        job = claimed.json().get("job", {})
        review_id = job.get("reviewId", "")
        if not review_id:
            return False

        try:
            investigation_id = str(job.get("investigationId") or "")
            provenance = self.output_dir / investigation_id / "provenance"
            report = await self.reviewer.review(
                str(job.get("manuscriptText") or ""),
                provenance_path=provenance if provenance.is_dir() else None,
                session_id=investigation_id,
            )
            payload = report.model_dump(mode="json", exclude={"base_review"})
            payload["report_markdown"] = self.reviewer.format_report(report)
            now = datetime.now(UTC)
            capability_stamp = {
                "kind": "automated",
                "cohort": f"{now.year}-Q{((now.month - 1) // 3) + 1}",
                "provider": self.provider,
                "model": self.model,
                "apollobot_version": __version__,
                "review_protocol": "frontier-integrity-review/v1",
                "reviewed_at": now.isoformat(),
            }
            completed = await self._signed_post(
                client,
                {
                    "stage": "complete",
                    "worker_id": self.worker_id,
                    "review_id": review_id,
                    "report": payload,
                    "capability_stamp": capability_stamp,
                },
            )
            if completed.status_code >= 300:
                logger.warning(
                    "Frontier automated review result rejected: %s", completed.status_code
                )
                return False
            return True
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.warning("Automated record review failed (%s)", type(error).__name__)
            with suppress(httpx.HTTPError):
                await self._signed_post(
                    client,
                    {
                        "stage": "fail",
                        "worker_id": self.worker_id,
                        "review_id": review_id,
                        "error": f"{type(error).__name__}: {str(error)[:700]}",
                    },
                )
            return False

    async def _signed_post(
        self,
        client: httpx.AsyncClient,
        payload: dict[str, object],
    ) -> httpx.Response:
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        timestamp = str(int(time.time()))
        nonce = str(uuid4())
        signature = hmac.new(
            self.secret,
            f"{timestamp}.{nonce}.".encode() + body,
            hashlib.sha256,
        ).hexdigest()
        return await client.post(
            self.endpoint,
            content=body,
            headers={
                "content-type": "application/json",
                "x-apollo-signature": f"sha256={signature}",
                "x-apollo-timestamp": timestamp,
                "x-apollo-nonce": nonce,
            },
        )
