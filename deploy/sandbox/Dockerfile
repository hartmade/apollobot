FROM python:3.12-slim-bookworm

ARG APOLLOBOT_RELEASE_SHA=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

LABEL org.opencontainers.image.revision=${APOLLOBOT_RELEASE_SHA}

COPY deploy/sandbox/requirements.lock /tmp/requirements.lock
RUN python -m pip install --no-cache-dir --require-hashes -r /tmp/requirements.lock \
    && rm /tmp/requirements.lock

WORKDIR /workspace
USER 65534:65534

ENTRYPOINT []
