FROM ghcr.io/astral-sh/uv:0.8.22 AS uv

FROM python:3.12-slim-bookworm

ARG APOLLOBOT_UID=10001
ARG APOLLOBOT_GID=10001
ARG APOLLOBOT_RELEASE_SHA=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=0 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}" \
    HOME=/home/apollobot \
    APOLLOBOT_BUILD_SHA=${APOLLOBOT_RELEASE_SHA}

LABEL org.opencontainers.image.revision=${APOLLOBOT_RELEASE_SHA}

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates docker.io \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid "${APOLLOBOT_GID}" apollobot \
    && useradd --uid "${APOLLOBOT_UID}" --gid "${APOLLOBOT_GID}" \
        --create-home --shell /usr/sbin/nologin apollobot \
    && mkdir -p /var/lib/apollobot/output \
    && chown -R apollobot:apollobot /var/lib/apollobot

COPY --from=uv /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

USER apollobot:apollobot
EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/ready', timeout=4).read()"]

ENTRYPOINT ["apollobot-api"]
