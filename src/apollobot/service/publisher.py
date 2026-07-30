"""Durable outbox delivery from ApolloBot to the Frontier platform."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path
from uuid import uuid4

import httpx

from apollobot.service.store import ServiceStore

logger = logging.getLogger(__name__)


class EventPublisher:
    def __init__(
        self,
        store: ServiceStore,
        platform_url: str,
        secret: str,
        output_dir: str | Path,
    ) -> None:
        self.store = store
        self.endpoint = f"{platform_url.rstrip('/')}/api/apollobot/events"
        self.artifact_endpoint = f"{platform_url.rstrip('/')}/api/apollobot/artifacts"
        self.secret = secret.encode()
        self.output_dir = Path(output_dir).resolve()
        self.task: asyncio.Task[None] | None = None
        self.stopping = False
        self.last_cycle_at: float | None = None
        self.last_error: str | None = None
        self.last_event_status: int | None = None
        self.last_artifact_status: int | None = None

    async def start(self) -> None:
        if self.task and not self.task.done():
            return
        self.stopping = False
        self.task = asyncio.create_task(self._loop(), name="frontier-event-publisher")

    async def stop(self) -> None:
        self.stopping = True
        if self.task:
            self.task.cancel()
            with suppress(asyncio.CancelledError):
                await self.task

    async def _loop(self) -> None:
        async with httpx.AsyncClient(timeout=10) as client:
            while not self.stopping:
                uploaded, delivered = await self._cycle(client)
                await asyncio.sleep(0.5 if uploaded or delivered else 3)

    async def flush_once(self) -> dict[str, object]:
        """Run one observable outbox cycle for recovery and operational checks."""
        async with httpx.AsyncClient(timeout=10) as client:
            await self._cycle(client)
        return self.diagnostics()

    async def _cycle(self, client: httpx.AsyncClient) -> tuple[bool, bool]:
        self.last_cycle_at = time.time()
        try:
            uploaded = await self._flush_artifacts(client)
            delivered = await self._flush(client)
            if uploaded or delivered:
                self.last_error = None
            return uploaded, delivered
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {str(error)[:300]}"
            logger.exception("Frontier publisher cycle failed")
            return False, False

    def diagnostics(self) -> dict[str, object]:
        return {
            "running": bool(self.task and not self.task.done()),
            "last_cycle_at": self.last_cycle_at,
            "last_error": self.last_error,
            "last_event_status": self.last_event_status,
            "last_artifact_status": self.last_artifact_status,
        }

    async def _flush_artifacts(self, client: httpx.AsyncClient) -> bool:
        uploaded = False
        for artifact in self.store.pending_artifacts(limit=10):
            path = self._artifact_path(artifact)
            if not path:
                self.store.mark_artifact_attempt(artifact["id"])
                continue
            request_payload = {
                "stage": "presign",
                "investigation_id": artifact["investigation_id"],
                "artifact": {
                    "id": artifact["id"],
                    "artifact_type": artifact["artifact_type"],
                    "label": artifact["label"],
                    "media_type": artifact["media_type"],
                    "size_bytes": artifact["size_bytes"],
                    "checksum_sha256": artifact["checksum_sha256"],
                },
            }
            try:
                binding = f"investigation:{artifact['investigation_id']}"
                presign = await self._signed_post(
                    client, self.artifact_endpoint, request_payload, binding
                )
                self.last_artifact_status = presign.status_code
                if presign.status_code >= 300:
                    self.store.mark_artifact_attempt(artifact["id"])
                    break
                upload = presign.json()
                response = await client.put(
                    upload["signedUrl"],
                    content=stream_file(path),
                    headers={
                        "content-type": artifact["media_type"] or "application/octet-stream",
                        "cache-control": "max-age=31536000, immutable",
                        "x-upsert": "true",
                    },
                )
                if response.status_code >= 300:
                    self.store.mark_artifact_attempt(artifact["id"])
                    break
                confirmation = await self._signed_post(
                    client,
                    self.artifact_endpoint,
                    {
                        "stage": "confirm",
                        "investigation_id": artifact["investigation_id"],
                        "artifact_id": artifact["id"],
                        "storage_path": upload["storagePath"],
                        "checksum_sha256": artifact["checksum_sha256"],
                        "size_bytes": artifact["size_bytes"],
                    },
                    binding,
                )
                self.last_artifact_status = confirmation.status_code
                if confirmation.status_code >= 300:
                    self.store.mark_artifact_attempt(artifact["id"])
                    break
                self.store.mark_artifact_uploaded(artifact["id"], upload["storagePath"])
                uploaded = True
            except (httpx.HTTPError, KeyError, ValueError) as error:
                self.store.mark_artifact_attempt(artifact["id"])
                logger.warning("Frontier artifact upload unavailable (%s)", type(error).__name__)
                break
        return uploaded

    async def _flush(self, client: httpx.AsyncClient) -> bool:
        delivered = False
        for event in self.store.pending_events(limit=50):
            snapshot = self.store.snapshot(event["investigation_id"], after=event["sequence"])
            if not snapshot:
                self.store.mark_event_attempt(event["id"])
                continue
            payload = {
                "event": event,
                "investigation": snapshot["investigation"],
                "nodes": snapshot["nodes"],
                "artifacts": snapshot["artifacts"],
                "experiments": snapshot["experiments"],
            }
            body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
            try:
                response = await client.post(
                    self.endpoint,
                    content=body,
                    headers=self._signed_headers(
                        body, f"investigation:{event['investigation_id']}"
                    ),
                )
                self.last_event_status = response.status_code
                if 200 <= response.status_code < 300:
                    self.store.mark_event_published(event["id"])
                    delivered = True
                else:
                    self.store.mark_event_attempt(event["id"])
                    logger.warning("Frontier event delivery rejected: %s", response.status_code)
                    break
            except httpx.HTTPError as error:
                self.store.mark_event_attempt(event["id"])
                logger.warning("Frontier event delivery unavailable (%s)", type(error).__name__)
                break
        return delivered

    async def _signed_post(
        self,
        client: httpx.AsyncClient,
        endpoint: str,
        payload: dict[str, object],
        binding: str,
    ) -> httpx.Response:
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        return await client.post(
            endpoint,
            content=body,
            headers=self._signed_headers(body, binding),
        )

    def _signed_headers(self, body: bytes, binding: str) -> dict[str, str]:
        timestamp = str(int(time.time()))
        nonce = str(uuid4())
        resource_key = hmac.new(
            self.secret,
            f"frontier-apollo-binding-v1:{binding}".encode(),
            hashlib.sha256,
        ).digest()
        signed = f"{timestamp}.{nonce}.{binding}.".encode() + body
        signature = hmac.new(resource_key, signed, hashlib.sha256).hexdigest()
        return {
            "content-type": "application/json",
            "x-apollo-signature": f"sha256={signature}",
            "x-apollo-timestamp": timestamp,
            "x-apollo-nonce": nonce,
            "x-apollo-binding": binding,
        }

    def _artifact_path(self, artifact: dict[str, object]) -> Path | None:
        session_root = (self.output_dir / str(artifact["investigation_id"])).resolve()
        path = (session_root / str(artifact["path"])).resolve()
        if session_root not in path.parents or not path.is_file():
            return None
        return path


async def stream_file(path: Path, chunk_size: int = 1024 * 1024) -> AsyncIterator[bytes]:
    with path.open("rb") as handle:
        while chunk := await asyncio.to_thread(handle.read, chunk_size):
            yield chunk
