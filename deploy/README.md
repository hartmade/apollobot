# ApolloBot service deployment

This package runs the authenticated ApolloBot API and executes generated
analysis in short-lived, network-disabled sibling containers. Use a dedicated
worker host or a rootless Docker daemon: access to a Docker socket is a
high-trust capability even though each analysis container is read-only,
capability-free, resource-limited, and disconnected from the network.

## First deployment

1. Copy `.env.example` to `.env` and set every required secret and URL. Keep
   `.env` off the repository. The webhook secret must exactly match the web
   platform's `APOLLOBOT_WEBHOOK_SECRET`.
2. Set `APOLLOBOT_DATA_DIR` to an absolute host path. Create it and its
   `output` child, then make both writable by `APOLLOBOT_UID:APOLLOBOT_GID`.
3. Set `DOCKER_GID` to the group owning `APOLLOBOT_DOCKER_SOCKET`. For rootless
   Docker, also set the socket path and normally use the rootless user's UID.
   Docker Desktop can translate the socket group across the Linux VM boundary;
   in that case, inspect the mounted socket from a disposable container and use
   the numeric group visible inside the container. A wrong group leaves
   `/ready` truthfully reporting `sandbox: false` instead of weakening socket
   permissions.
4. Build both immutable images:

   Set `APOLLOBOT_RELEASE_SHA` to the exact Git commit being released and use
   that value in both image tags. The Dockerfiles stamp the revision into both
   images; production readiness rejects a sandbox whose revision label does not
   match the service image.

   ```sh
   docker compose --profile images build apollobot sandbox-image
   ```

   The sandbox installs only the hash-locked scientific stack in
   `sandbox/requirements.lock`. After deliberately changing
   `sandbox/requirements.in`, regenerate and review it with:

   ```sh
   uv pip compile deploy/sandbox/requirements.in \
     --generate-hashes --universal \
     --output-file deploy/sandbox/requirements.lock
   ```

5. Start only the API service behind an HTTPS reverse proxy:

   ```sh
   docker compose up -d apollobot
   ```

6. Verify the public liveness/readiness endpoints and authenticated metrics:

   ```sh
   curl --fail http://127.0.0.1:8765/health
   curl --fail http://127.0.0.1:8765/ready
   curl --fail -H "Authorization: Bearer $APOLLOBOT_SERVICE_TOKEN" \
     http://127.0.0.1:8765/v1/metrics
   ```

   `/health` must report the expected `release`, and `/ready` must report every
   check as true. The platform production smoke gate compares the reported
   release to the intended commit instead of trusting a mutable image tag.

Production startup refuses a short service token or webhook secret, a non-HTTPS
platform or model-gateway URL, reused or placeholder secrets, an unsupported or
unconfigured model provider, local execution, or a non-container sandbox.
Readiness additionally checks the database, output volume, model, event
publisher, and Docker worker.

## Data, recovery, and upgrades

`service.db` is a SQLite WAL database; research outputs live under `output/`.
Back up both together. For a simple consistent backup, briefly stop the service
before taking a filesystem snapshot. Test restoration on a separate worker.

For upgrades, build versioned image tags, run the full test suite, replace the
compose image tags, and recreate the service. Retain the previous service and
sandbox tags for rollback. Database changes must be forward-compatible because
work can remain queued across a deploy.

Do not expose port 8765 directly to the public internet. The Frontier Science
web server is the trusted gateway and uses the bearer token; browsers should
never receive that token or call ApolloBot directly.

Built-in scientific data sources use direct public-API adapters when
`APOLLOBOT_MCP_PROXY_URL` is empty. Only set it after deploying an HTTPS MCP
adapter service; production startup rejects an insecure proxy URL.
