"""Authenticated aiohttp API for Frontier Science and ApolloBot workers."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import re
import secrets
import shutil
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from pathlib import Path
from time import monotonic
from urllib.parse import urlsplit

from aiohttp import web

from apollobot import __version__
from apollobot.core import APOLLO_HOME, load_config
from apollobot.service.framer import QuestionFramer
from apollobot.service.manager import InvestigationManager
from apollobot.service.models import QuestionCheck, parse_context_attachments
from apollobot.service.publisher import EventPublisher
from apollobot.service.reviewer import AutomatedReviewWorker
from apollobot.service.store import ServiceStore

logger = logging.getLogger(__name__)


class SlidingWindowLimiter:
    def __init__(self, limit: int = 20, window_seconds: int = 60) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = monotonic()
        queue = self.hits[key]
        while queue and queue[0] < now - self.window_seconds:
            queue.popleft()
        if len(queue) >= self.limit:
            return False
        queue.append(now)
        return True


STORE_KEY = web.AppKey("store", ServiceStore)
MANAGER_KEY = web.AppKey("manager", InvestigationManager)
FRAMER_KEY = web.AppKey("framer", QuestionFramer)
LIMITER_KEY = web.AppKey("limiter", SlidingWindowLimiter)
PUBLISHER_KEY: web.AppKey[EventPublisher | None] = web.AppKey("publisher")
REVIEW_WORKER_KEY: web.AppKey[AutomatedReviewWorker | None] = web.AppKey("review_worker")
SANDBOX_DIAGNOSTIC: dict[str, object] = {"status": None, "error": None}


def create_app(
    *,
    store_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    service_token: str | None = None,
) -> web.Application:
    config = load_config()
    store = ServiceStore(store_path or APOLLO_HOME / "service.db")
    manager = InvestigationManager(store, config=config, output_dir=output_dir)
    framer = QuestionFramer(config)
    limiter = SlidingWindowLimiter()
    token = service_token if service_token is not None else os.getenv("APOLLOBOT_SERVICE_TOKEN", "")
    allowed_origin = os.getenv("APOLLOBOT_ALLOWED_ORIGIN", "")
    platform_url = os.getenv("FRONTIER_PLATFORM_URL", "")
    webhook_secret = os.getenv("APOLLOBOT_WEBHOOK_SECRET", "")
    build_sha = os.getenv("APOLLOBOT_BUILD_SHA", "unknown")
    sandbox_image = os.getenv("APOLLOBOT_SANDBOX_IMAGE", "frontier-science/apollobot-sandbox:py312")
    production = os.getenv("APOLLOBOT_ENV", "development").lower() == "production"
    if production:
        problems = []
        if len(token) < 32:
            problems.append("APOLLOBOT_SERVICE_TOKEN must contain at least 32 characters")
        if contains_placeholder(token):
            problems.append("APOLLOBOT_SERVICE_TOKEN still contains a placeholder value")
        if len(webhook_secret) < 32:
            problems.append("APOLLOBOT_WEBHOOK_SECRET must contain at least 32 characters")
        if contains_placeholder(webhook_secret):
            problems.append("APOLLOBOT_WEBHOOK_SECRET still contains a placeholder value")
        if token and hmac.compare_digest(token, webhook_secret):
            problems.append("Apollo bearer and webhook secrets must be independent")
        if config.api.default_provider not in {"anthropic", "openai", "minimax"}:
            problems.append("APOLLOBOT_MODEL_PROVIDER is not supported")
        if not config.api.get_key():
            problems.append("The selected model provider API key is missing")
        if not secure_https_url(platform_url, origin_only=True):
            problems.append("FRONTIER_PLATFORM_URL must be an HTTPS URL")
        cloudflare_internal = os.getenv("APOLLOBOT_CLOUDFLARE_INTERNAL", "0") == "1"
        openai_base_url = os.getenv("OPENAI_BASE_URL", "")
        internal_model_url = cloudflare_internal and openai_base_url == "http://model.internal/v1"
        if openai_base_url and not internal_model_url and not secure_https_url(openai_base_url):
            problems.append("OPENAI_BASE_URL must be an HTTPS URL without credentials")
        mcp_proxy_url = os.getenv("APOLLOBOT_MCP_PROXY_URL", "")
        if mcp_proxy_url and not secure_https_url(mcp_proxy_url):
            problems.append("APOLLOBOT_MCP_PROXY_URL must be an HTTPS URL without credentials")
        sandbox_mode = os.getenv("APOLLOBOT_SANDBOX_MODE", "container")
        if sandbox_mode not in {"container", "cloudflare"}:
            problems.append("APOLLOBOT_SANDBOX_MODE must be container or cloudflare")
        if sandbox_mode == "cloudflare":
            if not cloudflare_internal:
                problems.append("Cloudflare sandboxing requires internal routing")
            if os.getenv("APOLLOBOT_SANDBOX_URL") != "http://sandbox.internal":
                problems.append("APOLLOBOT_SANDBOX_URL must use the internal sandbox route")
            if os.getenv("APOLLOBOT_CHECKPOINT_URL") != "http://checkpoint.internal/state":
                problems.append("APOLLOBOT_CHECKPOINT_URL must use durable internal storage")
        if os.getenv("APOLLOBOT_ALLOW_LOCAL_EXECUTION", "0") != "0":
            problems.append("APOLLOBOT_ALLOW_LOCAL_EXECUTION must be disabled")
        if not re.fullmatch(r"[a-fA-F0-9]{7,64}", build_sha):
            problems.append("APOLLOBOT_BUILD_SHA must contain 7 to 64 hexadecimal characters")
        if sandbox_mode == "container" and (
            contains_placeholder(sandbox_image)
            or (build_sha not in sandbox_image and "@sha256:" not in sandbox_image)
        ):
            problems.append(
                "APOLLOBOT_SANDBOX_IMAGE must be versioned with "
                "APOLLOBOT_BUILD_SHA or an image digest"
            )
        if problems:
            store.close()
            raise RuntimeError("Unsafe production configuration: " + "; ".join(problems))
    publisher = (
        EventPublisher(store, platform_url, webhook_secret, manager.output_dir)
        if platform_url and webhook_secret
        else None
    )
    review_worker = None
    if platform_url and webhook_secret and config.api.get_key():
        review_worker = AutomatedReviewWorker(
            config,
            platform_url,
            webhook_secret,
            manager.output_dir,
        )

    @web.middleware
    async def security(
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        if request.method == "OPTIONS":
            return web.Response(status=204, headers=cors_headers(request, allowed_origin))
        request_id = request.headers.get("x-request-id", "")
        if (
            not request_id.isascii()
            or not request_id.replace("-", "").isalnum()
            or len(request_id) > 80
        ):
            request_id = secrets.token_hex(12)
        started = monotonic()
        status = 500
        try:
            if request.path not in {"/health", "/ready"} and token:
                supplied = request.headers.get("authorization", "")
                if not hmac.compare_digest(supplied, f"Bearer {token}"):
                    raise web.HTTPUnauthorized(
                        text=json.dumps({"error": "Unauthorized"}),
                        content_type="application/json",
                    )
            response = await handler(request)
            status = response.status
            for key, value in cors_headers(request, allowed_origin).items():
                response.headers[key] = value
            response.headers["x-content-type-options"] = "nosniff"
            response.headers["cache-control"] = "no-store"
            response.headers["x-request-id"] = request_id
            return response
        except web.HTTPException as error:
            status = error.status
            error.headers["x-request-id"] = request_id
            raise
        finally:
            logger.info(
                json.dumps(
                    {
                        "event": "http_request",
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.path,
                        "status": status,
                        "duration_ms": round((monotonic() - started) * 1000, 2),
                    },
                    separators=(",", ":"),
                )
            )

    app = web.Application(middlewares=[security], client_max_size=256 * 1024)
    app[STORE_KEY] = store
    app[MANAGER_KEY] = manager
    app[FRAMER_KEY] = framer
    app[LIMITER_KEY] = limiter
    app[PUBLISHER_KEY] = publisher
    app[REVIEW_WORKER_KEY] = review_worker
    app.router.add_get("/health", health)
    app.router.add_get("/ready", readiness)
    app.router.add_get("/v1/metrics", service_metrics)
    app.router.add_post("/v1/internal/publisher/flush", publisher_flush)
    app.router.add_post("/v1/questions/check", question_check)
    app.router.add_post("/v1/investigations", create_investigation)
    app.router.add_get("/v1/investigations/{investigation_id}", get_investigation)
    app.router.add_post("/v1/investigations/{investigation_id}/actions", investigation_action)
    app.router.add_get("/v1/investigations/{investigation_id}/events", stream_events)
    app.router.add_get(
        "/v1/investigations/{investigation_id}/artifacts/{artifact_id}",
        get_artifact,
    )
    app.router.add_route("OPTIONS", "/{tail:.*}", options_response)

    async def startup(_app: web.Application) -> None:
        recovered = await manager.recover_interrupted()
        if recovered:
            logger.warning("Recovered %d interrupted investigation(s)", recovered)
        if publisher:
            await publisher.start()
        if review_worker:
            await review_worker.start()

    async def cleanup(_app: web.Application) -> None:
        if publisher:
            await publisher.stop()
        if review_worker:
            await review_worker.stop()
        for task in list(manager.tasks.values()):
            if not task.done():
                task.cancel()
        if manager.tasks:
            await asyncio.gather(*manager.tasks.values(), return_exceptions=True)
        store.close()

    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)
    return app


def contains_placeholder(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in ("placeholder", "replace-with", "example"))


def secure_https_url(value: str, *, origin_only: bool = False) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        return False
    return not origin_only or parsed.path in {"", "/"}


async def options_response(_request: web.Request) -> web.Response:
    return web.Response(status=204)


async def health(request: web.Request) -> web.Response:
    manager = request.app[MANAGER_KEY]
    running = sum(not task.done() for task in manager.tasks.values())
    return web.json_response(
        {
            "status": "ok",
            "service": "apollobot",
            "version": __version__,
            "release": os.getenv("APOLLOBOT_BUILD_SHA", "unknown"),
            "running_jobs": running,
        }
    )


async def readiness(request: web.Request) -> web.Response:
    store = request.app[STORE_KEY]
    manager = request.app[MANAGER_KEY]
    production = os.getenv("APOLLOBOT_ENV", "development").lower() == "production"
    sandbox_mode = os.getenv("APOLLOBOT_SANDBOX_MODE", "local")
    if sandbox_mode == "container":
        sandbox_ready = await container_runtime_ready()
    elif sandbox_mode == "cloudflare":
        sandbox_ready = await cloudflare_sandbox_ready()
    else:
        sandbox_ready = not production
    checks = {
        "store": store.ping(),
        "durability": store.durability_ready(),
        "output": manager.output_dir.exists() and os.access(manager.output_dir, os.W_OK),
        "model": bool(manager.config.api.get_key()),
        "publisher": request.app[PUBLISHER_KEY] is not None,
        "sandbox": sandbox_ready,
    }
    required = ["store", "output"] if not production else list(checks)
    ready = all(checks[name] for name in required)
    return web.json_response(
        {"status": "ready" if ready else "not_ready", "checks": checks},
        status=200 if ready else 503,
    )


async def service_metrics(request: web.Request) -> web.Response:
    manager = request.app[MANAGER_KEY]
    store = request.app[STORE_KEY]
    metrics = store.operational_metrics()
    metrics["running_jobs"] = sum(not task.done() for task in manager.tasks.values())
    metrics["max_concurrent_jobs"] = manager.max_concurrent_jobs
    metrics["max_concurrent_plans"] = manager.max_concurrent_plans
    metrics["durability"] = store.durability_status()
    metrics["sandbox"] = dict(SANDBOX_DIAGNOSTIC)
    metrics["framer"] = {"provider_error": request.app[FRAMER_KEY].last_provider_error}
    publisher = request.app[PUBLISHER_KEY]
    metrics["publisher"] = publisher.diagnostics() if publisher else {"running": False}
    return web.json_response(metrics)


async def publisher_flush(request: web.Request) -> web.Response:
    publisher = request.app[PUBLISHER_KEY]
    if not publisher:
        return web.json_response({"error": "Publisher is not configured"}, status=503)
    return web.json_response(await publisher.flush_once())


async def cloudflare_sandbox_ready() -> bool:
    import httpx

    endpoint = os.getenv("APOLLOBOT_SANDBOX_URL", "").rstrip("/")
    if not endpoint:
        return False
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.get(f"{endpoint}/ready")
        SANDBOX_DIAGNOSTIC.update(status=response.status_code, error=None)
        return response.status_code == 200
    except httpx.HTTPError as error:
        SANDBOX_DIAGNOSTIC.update(status=None, error=type(error).__name__)
        return False


async def question_check(request: web.Request) -> web.Response:
    limiter = request.app[LIMITER_KEY]
    peer = request.headers.get("x-forwarded-for", request.remote or "unknown").split(",")[0].strip()
    if not limiter.allow(peer):
        raise web.HTTPTooManyRequests(
            text=json.dumps({"error": "Question-check rate limit exceeded"}),
            content_type="application/json",
        )
    payload = await request.json()
    question = payload.get("question", "") if isinstance(payload, dict) else ""
    try:
        context_attachments = parse_context_attachments(
            payload.get("context_attachments") if isinstance(payload, dict) else None
        )
        check = await request.app[FRAMER_KEY].frame(question, context_attachments)
    except ValueError as error:
        raise web.HTTPBadRequest(
            text=json.dumps({"error": str(error)}), content_type="application/json"
        ) from error
    return web.json_response({"check": check.model_dump(by_alias=True)})


async def create_investigation(request: web.Request) -> web.Response:
    payload = await request.json()
    try:
        if not isinstance(payload, dict):
            raise ValueError("Investigation payload must be an object")
        check = QuestionCheck.model_validate(payload.get("check", {}))
        investigation = request.app[MANAGER_KEY].create(
            check,
            user_id=payload.get("user_id"),
            investigation_id=payload.get("id"),
            model_id=payload.get("model_id"),
            provider_tag=payload.get("provider_tag"),
            context_attachments=payload.get("context_attachments"),
        )
    except ValueError as error:
        raise web.HTTPUnprocessableEntity(
            text=json.dumps({"error": str(error)}), content_type="application/json"
        ) from error
    return web.json_response(investigation, status=201)


async def get_investigation(request: web.Request) -> web.Response:
    after = max(0, int(request.query.get("after", "0")))
    snapshot = request.app[MANAGER_KEY].snapshot(request.match_info["investigation_id"], after)
    if not snapshot:
        raise web.HTTPNotFound(
            text=json.dumps({"error": "Investigation not found"}), content_type="application/json"
        )
    return web.json_response(snapshot)


async def investigation_action(request: web.Request) -> web.Response:
    payload = await request.json()
    action = payload.get("action", "") if isinstance(payload, dict) else ""
    try:
        result = await request.app[MANAGER_KEY].action(
            request.match_info["investigation_id"], action, payload
        )
    except KeyError as error:
        raise web.HTTPNotFound(
            text=json.dumps({"error": "Investigation not found"}), content_type="application/json"
        ) from error
    except ValueError as error:
        raise web.HTTPConflict(
            text=json.dumps({"error": str(error)}), content_type="application/json"
        ) from error
    return web.json_response(result)


async def get_artifact(request: web.Request) -> web.StreamResponse:
    result = request.app[MANAGER_KEY].artifact_path(
        request.match_info["investigation_id"], request.match_info["artifact_id"]
    )
    if not result:
        raise web.HTTPNotFound(
            text=json.dumps({"error": "Artifact not found"}),
            content_type="application/json",
        )
    artifact, path = result
    response = web.FileResponse(path, headers={"content-type": artifact["media_type"]})
    response.headers["content-disposition"] = (
        f'inline; filename="{safe_filename(artifact["label"])}"'
    )
    response.headers["x-content-sha256"] = artifact["checksum_sha256"] or ""
    return response


async def stream_events(request: web.Request) -> web.StreamResponse:
    investigation_id = request.match_info["investigation_id"]
    after = max(0, int(request.query.get("after", request.headers.get("last-event-id", "0"))))
    if not request.app[MANAGER_KEY].snapshot(investigation_id, after):
        raise web.HTTPNotFound(
            text=json.dumps({"error": "Investigation not found"}), content_type="application/json"
        )

    response = web.StreamResponse(
        status=200,
        headers={
            "content-type": "text/event-stream",
            "cache-control": "no-cache, no-transform",
            "connection": "keep-alive",
            "x-accel-buffering": "no",
        },
    )
    await response.prepare(request)
    idle_ticks = 0
    try:
        while idle_ticks < 300:
            snapshot = request.app[MANAGER_KEY].snapshot(investigation_id, after)
            if not snapshot:
                break
            events = snapshot["events"]
            for event in events:
                after = event["sequence"]
                body = json.dumps(event, separators=(",", ":"))
                await response.write(
                    f"id: {after}\nevent: {event['event_type']}\ndata: {body}\n\n".encode()
                )
                idle_ticks = 0
            if (
                snapshot["investigation"]["status"] in {"complete", "failed", "cancelled"}
                and not events
            ):
                break
            if not events:
                idle_ticks += 1
                if idle_ticks % 15 == 0:
                    await response.write(b": keepalive\n\n")
            await asyncio.sleep(1)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return response


def cors_headers(request: web.Request, allowed_origin: str) -> dict[str, str]:
    origin = request.headers.get("origin", "")
    if not allowed_origin or not origin or origin != allowed_origin:
        return {}
    return {
        "access-control-allow-origin": origin,
        "access-control-allow-methods": "GET,POST,OPTIONS",
        "access-control-allow-headers": "authorization,content-type,last-event-id",
        "vary": "origin",
    }


def safe_filename(value: str) -> str:
    return "".join(character for character in value if character.isalnum() or character in "._-")


async def container_runtime_ready() -> bool:
    docker = shutil.which("docker")
    if not docker:
        return False
    image = os.getenv("APOLLOBOT_SANDBOX_IMAGE", "frontier-science/apollobot-sandbox:py312")
    environment = {
        "PATH": os.getenv("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "HOME": os.getenv("HOME") or str(Path.cwd()),
        "DOCKER_HOST": os.getenv("DOCKER_HOST", "unix:///var/run/docker.sock"),
    }
    try:
        async with asyncio.timeout(4):
            info = await asyncio.create_subprocess_exec(
                docker,
                "info",
                "--format",
                "{{.ServerVersion}}",
                env=environment,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            if await info.wait() != 0:
                return False
            inspect = await asyncio.create_subprocess_exec(
                docker,
                "image",
                "inspect",
                "--format",
                '{{ index .Config.Labels "org.opencontainers.image.revision" }}',
                image,
                env=environment,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            output, _ = await inspect.communicate()
            if inspect.returncode != 0:
                return False
            if os.getenv("APOLLOBOT_ENV", "development").lower() == "production":
                expected = os.getenv("APOLLOBOT_BUILD_SHA", "")
                actual = output.decode("utf-8", errors="replace").strip()
                if not expected or not hmac.compare_digest(actual, expected):
                    return False
    except (TimeoutError, OSError):
        return False
    return True


def run_api() -> None:
    host = os.getenv("APOLLOBOT_API_HOST", "127.0.0.1")
    port = int(os.getenv("APOLLOBOT_API_PORT", "8765"))
    app = create_app(
        store_path=os.getenv("APOLLOBOT_SERVICE_DB") or None,
        output_dir=os.getenv("APOLLOBOT_OUTPUT_DIR") or None,
    )
    web.run_app(app, host=host, port=port, print=lambda message: print(message))
