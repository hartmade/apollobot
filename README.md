# ApolloBot

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

ApolloBot is Frontier Science's inspectable computational-research engine. It
turns a research question into a reviewable experiment plan, pauses for human
approval, executes the approved work, and captures the resulting evidence,
artifacts, and provenance.

Version 0.2 is alpha software. Its output is a research draft and evidence
record, not proof that a claim is true or ready for publication.

## What is implemented

ApolloBot has two supported surfaces:

- A local CLI for contributor-controlled research sessions.
- An authenticated worker API used by the Frontier Science web platform.

The managed worker provides:

- deterministic safety screening before model-based question framing;
- a durable SQLite/WAL investigation and event store;
- an explicit human checkpoint between experiment design and execution;
- bounded planning and execution concurrency;
- resumable server-sent event streams and indexed, checksummed artifacts;
- crash recovery that pauses interrupted compute instead of silently rerunning it;
- short-lived, network-disabled Docker analysis containers in production;
- dataset access manifests that distinguish public, synthetic,
  access-controlled, and code-only inputs and exclude restricted raw data from
  publication unless redistribution is allowed;
- conservative discovery triage and citation-safe related-literature results
  that explicitly do not establish a breakthrough;
- signed event and artifact delivery to the Frontier Science platform; and
- an automated review worker for completed living records.

ApolloBot does **not** guarantee novelty, causal validity, connector
availability, successful execution, or acceptance for publication. It does not
perform wet-lab work. Direct CLI submission to Frontier Science was retired in
v0.2 so identity, funding, provenance, review, and DOI state remain attached to
an authenticated living record.

## Install from source

Prerequisites:

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- an Anthropic, OpenAI/OpenAI-compatible, or MiniMax API key
- Docker for the production worker sandbox

```sh
git clone https://github.com/frontier-science-ai/apollobot.git
cd apollobot
uv sync --frozen --extra dev
uv run apollo --version
```

ApolloBot is not currently advertised as a published PyPI package; use the
locked source installation above for a reproducible development environment.

## Local CLI

Configure a local profile and start a session:

```sh
uv run apollo init
uv run apollo discover \
  "Does a measurable relationship exist between urban tree-canopy coverage and summer surface temperature in comparable census tracts?"
```

The `research` command is an alias for `discover`. Local sessions are stored in
`~/apollobot-research` unless `output_dir` is changed in
`~/.apollobot/config.yaml`.

Useful inspection commands include:

```sh
uv run apollo status
uv run apollo list
uv run apollo provenance <session-id>
uv run apollo review --session <session-id>
uv run apollo export --output research-export.tar.gz
```

Local CLI sessions run with the permissions of the invoking user. Treat model-
generated code and third-party data as untrusted and inspect the proposed work
before allowing it to run.

## Managed platform lifecycle

The production web flow is intentionally stateful:

1. `POST /v1/questions/check` frames and screens the question.
2. `POST /v1/investigations` creates a durable, unexecuted investigation.
3. The `prepare` action develops hypotheses and an executable experiment plan.
4. ApolloBot waits in `awaiting_approval` until a person approves the plan.
5. The `approve` action queues execution; `pause`, `resume`, and `cancel` are
   explicit operator controls.
6. `GET /v1/investigations/{id}/events` streams progress with resumable event
   sequence numbers.
7. Completed files are indexed with media type, byte size, and SHA-256 checksum.
8. The completed result includes discovery triage and related literature, while
   retaining null, inconclusive, and failed outcomes honestly.
9. The web platform turns the investigation into a living research record for
   reproduction, challenge, branching, derivation, discussion, review, and
   publication.

All `/v1/*` routes require the service bearer token. Browsers must call the
Frontier Science gateway, never ApolloBot directly.

## Scientific data connectors

The 54 built-in connectors use audited direct adapters for public scientific
APIs by default. This is the supported deployment mode and does not depend on a
Frontier Science-owned MCP hostname. If an adapter service is deployed later,
set `APOLLOBOT_MCP_PROXY_URL` to its HTTPS base URL; failed query requests still
fall back to the direct adapters. Connectors that require provider credentials
identify the required environment variable in the domain-pack source.

## Production deployment

The deployment package builds separate service and sandbox images. Start with
[`deploy/README.md`](deploy/README.md) and [`deploy/.env.example`](deploy/.env.example).
The production service refuses to start when required secrets, HTTPS callback
configuration, a supported model provider, or the container sandbox policy are
missing.

Keep the API on a private network behind an HTTPS reverse proxy. Access to a
Docker socket is a high-trust capability; use a dedicated worker host or a
rootless Docker daemon. Back up the SQLite database and output directory as one
unit.

## Research records and artifacts

The exact files depend on the selected mode, data sources, and whether each
phase succeeds. A session may contain:

```text
<session>/
├── mission.yaml
├── session_state.json
├── manuscript.md or manuscript.tex
├── data/
├── analysis/
├── figures/
├── provenance/
├── review/
└── replication_kit/
```

An indexed artifact is evidence that a file was produced; it is not an
endorsement of the file's scientific validity. Review the provenance log,
source licensing, assumptions, exclusions, statistical tests, and limitations
before relying on a result. Managed runs also create `data/access-manifest.json`;
its access and redistribution fields govern which raw inputs may enter the
publishable artifact set.

## Development

```sh
uv run ruff format --check src tests
uv run ruff check src tests
uv run pytest -q
```

The CI workflow runs formatting, static checks, and the full test suite across
the supported Python versions. The current release candidate passes 558 tests.
Deployment changes should also exercise the container build and `/ready` check
on a Docker-capable host.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Particularly useful contributions are
new tested data adapters, statistical validation, provenance improvements,
sandbox hardening, and realistic evaluation cases. A registry entry describes
an adapter; it is not a promise that a remote endpoint is deployed or reachable.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
