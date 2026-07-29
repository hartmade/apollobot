import { Container, ContainerProxy, getContainer } from "@cloudflare/containers";
import { DurableObject, env } from "cloudflare:workers";

type Bindings = {
  AI: Ai;
  APOLLO_SERVICE: DurableObjectNamespace<ApolloService>;
  APOLLO_SANDBOX: DurableObjectNamespace<ApolloSandbox>;
  APOLLO_CHECKPOINT: DurableObjectNamespace<ApolloCheckpoint>;
  APOLLOBOT_SERVICE_TOKEN: string;
  APOLLOBOT_WEBHOOK_SECRET: string;
  APOLLOBOT_ENV: string;
  APOLLOBOT_API_HOST: string;
  APOLLOBOT_API_PORT: string;
  APOLLOBOT_SERVICE_DB: string;
  APOLLOBOT_OUTPUT_DIR: string;
  APOLLOBOT_SANDBOX_MODE: string;
  APOLLOBOT_SANDBOX_URL: string;
  APOLLOBOT_CHECKPOINT_URL: string;
  APOLLOBOT_CLOUDFLARE_INTERNAL: string;
  APOLLOBOT_ALLOW_LOCAL_EXECUTION: string;
  APOLLOBOT_SANDBOX_NETWORK: string;
  APOLLOBOT_SANDBOX_OUTPUT_BYTES: string;
  APOLLOBOT_BUILD_SHA: string;
  APOLLOBOT_MODEL_PROVIDER: string;
  OPENROUTER_API_KEY: string;
  OPENAI_BASE_URL: string;
  OPENAI_MODEL: string;
  OPENROUTER_PROVIDER_TAG: string;
  OPENROUTER_DATA_COLLECTION: string;
  OPENROUTER_SITE_URL: string;
  OPENROUTER_APP_NAME: string;
  OPENROUTER_REASONING_EFFORT: string;
  OPENROUTER_REASONING_EXCLUDE: string;
  APOLLOBOT_MODEL_BILLING_PROVIDER: string;
  APOLLOBOT_MODEL_INPUT_COST_PER_M: string;
  APOLLOBOT_MODEL_CACHED_INPUT_COST_PER_M: string;
  APOLLOBOT_MODEL_OUTPUT_COST_PER_M: string;
  APOLLOBOT_MODEL_MAX_OUTPUT_TOKENS: string;
  APOLLOBOT_WORKERS_AI_MODEL: string;
  APOLLOBOT_FRAMER_TIMEOUT: string;
  APOLLOBOT_PLANNER_TIMEOUT: string;
  APOLLOBOT_MAX_LITERATURE_QUERIES: string;
  APOLLOBOT_MAX_DATA_REQUIREMENTS: string;
  APOLLOBOT_MAX_ANALYSIS_STEPS: string;
  FRONTIER_PLATFORM_URL: string;
  APOLLOBOT_ALLOWED_ORIGIN: string;
  APOLLOBOT_MAX_CONCURRENT_JOBS: string;
  APOLLOBOT_MAX_CONCURRENT_PLANS: string;
};

const bindings = env as unknown as Bindings;

export class ApolloService extends Container {
  defaultPort = 8765;
  pingEndpoint = "localhost/health";
  sleepAfter = "1h";
  enableInternet = true;
  envVars = {
    APOLLOBOT_ENV: bindings.APOLLOBOT_ENV,
    APOLLOBOT_API_HOST: bindings.APOLLOBOT_API_HOST,
    APOLLOBOT_API_PORT: bindings.APOLLOBOT_API_PORT,
    APOLLOBOT_SERVICE_DB: bindings.APOLLOBOT_SERVICE_DB,
    APOLLOBOT_OUTPUT_DIR: bindings.APOLLOBOT_OUTPUT_DIR,
    APOLLOBOT_SANDBOX_MODE: bindings.APOLLOBOT_SANDBOX_MODE,
    APOLLOBOT_SANDBOX_URL: bindings.APOLLOBOT_SANDBOX_URL,
    APOLLOBOT_CHECKPOINT_URL: bindings.APOLLOBOT_CHECKPOINT_URL,
    APOLLOBOT_CLOUDFLARE_INTERNAL: bindings.APOLLOBOT_CLOUDFLARE_INTERNAL,
    APOLLOBOT_ALLOW_LOCAL_EXECUTION: bindings.APOLLOBOT_ALLOW_LOCAL_EXECUTION,
    APOLLOBOT_SANDBOX_NETWORK: bindings.APOLLOBOT_SANDBOX_NETWORK,
    APOLLOBOT_SANDBOX_OUTPUT_BYTES: bindings.APOLLOBOT_SANDBOX_OUTPUT_BYTES,
    APOLLOBOT_BUILD_SHA: bindings.APOLLOBOT_BUILD_SHA,
    APOLLOBOT_MODEL_PROVIDER: bindings.APOLLOBOT_MODEL_PROVIDER,
    OPENAI_API_KEY: bindings.OPENROUTER_API_KEY,
    OPENAI_BASE_URL: bindings.OPENAI_BASE_URL,
    OPENAI_MODEL: bindings.OPENAI_MODEL,
    OPENROUTER_PROVIDER_TAG: bindings.OPENROUTER_PROVIDER_TAG,
    OPENROUTER_DATA_COLLECTION: bindings.OPENROUTER_DATA_COLLECTION,
    OPENROUTER_SITE_URL: bindings.OPENROUTER_SITE_URL,
    OPENROUTER_APP_NAME: bindings.OPENROUTER_APP_NAME,
    OPENROUTER_REASONING_EFFORT: bindings.OPENROUTER_REASONING_EFFORT,
    OPENROUTER_REASONING_EXCLUDE: bindings.OPENROUTER_REASONING_EXCLUDE,
    APOLLOBOT_MODEL_BILLING_PROVIDER: bindings.APOLLOBOT_MODEL_BILLING_PROVIDER,
    APOLLOBOT_MODEL_INPUT_COST_PER_M: bindings.APOLLOBOT_MODEL_INPUT_COST_PER_M,
    APOLLOBOT_MODEL_CACHED_INPUT_COST_PER_M:
      bindings.APOLLOBOT_MODEL_CACHED_INPUT_COST_PER_M,
    APOLLOBOT_MODEL_OUTPUT_COST_PER_M: bindings.APOLLOBOT_MODEL_OUTPUT_COST_PER_M,
    APOLLOBOT_MODEL_MAX_OUTPUT_TOKENS: bindings.APOLLOBOT_MODEL_MAX_OUTPUT_TOKENS,
    APOLLOBOT_FRAMER_TIMEOUT: bindings.APOLLOBOT_FRAMER_TIMEOUT,
    APOLLOBOT_PLANNER_TIMEOUT: bindings.APOLLOBOT_PLANNER_TIMEOUT,
    APOLLOBOT_MAX_LITERATURE_QUERIES: bindings.APOLLOBOT_MAX_LITERATURE_QUERIES,
    APOLLOBOT_MAX_DATA_REQUIREMENTS: bindings.APOLLOBOT_MAX_DATA_REQUIREMENTS,
    APOLLOBOT_MAX_ANALYSIS_STEPS: bindings.APOLLOBOT_MAX_ANALYSIS_STEPS,
    FRONTIER_PLATFORM_URL: bindings.FRONTIER_PLATFORM_URL,
    APOLLOBOT_ALLOWED_ORIGIN: bindings.APOLLOBOT_ALLOWED_ORIGIN,
    APOLLOBOT_MAX_CONCURRENT_JOBS: bindings.APOLLOBOT_MAX_CONCURRENT_JOBS,
    APOLLOBOT_MAX_CONCURRENT_PLANS: bindings.APOLLOBOT_MAX_CONCURRENT_PLANS,
    APOLLOBOT_SERVICE_TOKEN: bindings.APOLLOBOT_SERVICE_TOKEN,
    APOLLOBOT_WEBHOOK_SECRET: bindings.APOLLOBOT_WEBHOOK_SECRET,
  };

}

// Assignment is intentional: Container exposes an inherited static setter
// that registers handlers with ContainerProxy. A static class field would
// shadow that setter and leave the virtual hostnames unresolved.
ApolloService.outboundByHost = {
  "model.internal": workersAiRequest,
  "checkpoint.internal": checkpointRequest,
  "sandbox.internal": sandboxRequest,
};

export class ApolloSandbox extends Container {
  defaultPort = 8090;
  pingEndpoint = "localhost/ready";
  sleepAfter = "30s";
  enableInternet = false;
  envVars = {};
}

type CheckpointManifest = {
  generation: string;
  chunks: number;
  size: number;
  sha256: string;
  updatedAt: string;
};

export class ApolloCheckpoint extends DurableObject<Bindings> {
  async fetch(request: Request): Promise<Response> {
    if (request.method === "PUT") return this.put(request);
    if (request.method === "GET") return this.get();
    if (request.method === "HEAD") return this.head();
    return new Response("Method not allowed", { status: 405 });
  }

  private async put(request: Request): Promise<Response> {
    const body = await request.arrayBuffer();
    if (body.byteLength === 0 || body.byteLength > 32 * 1024 * 1024) {
      return new Response("Invalid checkpoint size", { status: 413 });
    }
    const generation = crypto.randomUUID();
    const chunkSize = 96 * 1024;
    const chunks = Math.ceil(body.byteLength / chunkSize);
    for (let offset = 0; offset < chunks; offset += 100) {
      const values: Record<string, ArrayBuffer> = {};
      for (let index = offset; index < Math.min(chunks, offset + 100); index += 1) {
        values[`chunk:${generation}:${index}`] = body.slice(
          index * chunkSize,
          Math.min(body.byteLength, (index + 1) * chunkSize),
        );
      }
      await this.ctx.storage.put(values);
    }
    const digest = await crypto.subtle.digest("SHA-256", body);
    const manifest: CheckpointManifest = {
      generation,
      chunks,
      size: body.byteLength,
      sha256: [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join(""),
      updatedAt: new Date().toISOString(),
    };
    const previous = await this.ctx.storage.get<CheckpointManifest>("manifest");
    await this.ctx.storage.put("manifest", manifest);
    if (previous && previous.generation !== generation) {
      const old = await this.ctx.storage.list({ prefix: `chunk:${previous.generation}:` });
      const keys = [...old.keys()];
      for (let offset = 0; offset < keys.length; offset += 100) {
        await this.ctx.storage.delete(keys.slice(offset, offset + 100));
      }
    }
    return Response.json({ stored: true, ...manifest });
  }

  private async get(): Promise<Response> {
    const manifest = await this.ctx.storage.get<CheckpointManifest>("manifest");
    if (!manifest) return new Response("Not found", { status: 404 });
    const output = new Uint8Array(manifest.size);
    let position = 0;
    for (let offset = 0; offset < manifest.chunks; offset += 100) {
      const keys = Array.from(
        { length: Math.min(100, manifest.chunks - offset) },
        (_, index) => `chunk:${manifest.generation}:${offset + index}`,
      );
      const values = await this.ctx.storage.get<ArrayBuffer>(keys);
      for (const key of keys) {
        const value = values.get(key);
        if (!value) return new Response("Checkpoint is incomplete", { status: 500 });
        output.set(new Uint8Array(value), position);
        position += value.byteLength;
      }
    }
    return new Response(output, {
      headers: {
        "content-type": "application/gzip",
        "content-length": String(manifest.size),
        "x-content-sha256": manifest.sha256,
      },
    });
  }

  private async head(): Promise<Response> {
    const manifest = await this.ctx.storage.get<CheckpointManifest>("manifest");
    return new Response(null, {
      status: manifest ? 200 : 404,
      headers: manifest
        ? {
            "content-length": String(manifest.size),
            "x-content-sha256": manifest.sha256,
            "last-modified": manifest.updatedAt,
          }
        : {},
    });
  }
}

async function workersAiRequest(request: Request, workerEnv: Bindings): Promise<Response> {
  const url = new URL(request.url);
  if (request.method !== "POST" || url.pathname !== "/v1/chat/completions") {
    return Response.json({ error: { message: "Unsupported model route" } }, { status: 404 });
  }
  const input = (await request.json()) as Record<string, unknown>;
  if (input.stream === true) {
    return Response.json({ error: { message: "Streaming is not enabled" } }, { status: 400 });
  }
  const model = String(input.model || workerEnv.APOLLOBOT_WORKERS_AI_MODEL);
  delete input.model;
  delete input.stream;
  if (model.includes("/kimi-k2.6") && input.chat_template_kwargs === undefined) {
    input.chat_template_kwargs = { thinking: false };
  }
  const result = (await workerEnv.AI.run(model as keyof AiModels, input)) as Record<string, unknown>;
  if (Array.isArray(result.choices)) return Response.json(result);
  const content = typeof result.response === "string" ? result.response : "";
  const usage = (result.usage as Record<string, number> | undefined) ?? {};
  return Response.json({
    id: `chatcmpl-${crypto.randomUUID()}`,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content },
        finish_reason: "stop",
      },
    ],
    usage: {
      prompt_tokens: usage.prompt_tokens ?? usage.input_tokens ?? 0,
      completion_tokens: usage.completion_tokens ?? usage.output_tokens ?? 0,
      total_tokens: usage.total_tokens ?? 0,
    },
  });
}

async function checkpointRequest(request: Request, workerEnv: Bindings): Promise<Response> {
  const stub = workerEnv.APOLLO_CHECKPOINT.getByName("primary");
  const response = await stub.fetch(new Request("https://checkpoint.local/state", request));
  console.log("Apollo checkpoint route", request.method, response.status);
  return response;
}

async function sandboxRequest(request: Request, workerEnv: Bindings): Promise<Response> {
  const url = new URL(request.url);
  const match = url.pathname.match(/^\/run\/([a-f0-9]{32})$/);
  const instance = match ? match[1] : url.pathname === "/ready" ? "readiness" : "";
  if (!instance) return new Response("Not found", { status: 404 });
  const sandbox = getContainer(workerEnv.APOLLO_SANDBOX, instance);
  const response = await sandbox.fetch(new Request(`http://sandbox.local${url.pathname}`, request));
  console.log("Apollo sandbox route", request.method, response.status, instance);
  return response;
}

export { ContainerProxy };

export default {
  async fetch(request: Request, workerEnv: Bindings): Promise<Response> {
    const url = new URL(request.url);
    if (url.pathname === "/v1/internal/restart-service") {
      if (request.method !== "POST") {
        return new Response("Method not allowed", { status: 405 });
      }
      if (request.headers.get("authorization") !== `Bearer ${workerEnv.APOLLOBOT_SERVICE_TOKEN}`) {
        return Response.json({ error: "Unauthorized" }, { status: 401 });
      }
      await getContainer(workerEnv.APOLLO_SERVICE, "primary").destroy();
      return Response.json({ restarted: true });
    }
    if (url.pathname === "/v1/internal/diagnostics") {
      if (request.headers.get("authorization") !== `Bearer ${workerEnv.APOLLOBOT_SERVICE_TOKEN}`) {
        return Response.json({ error: "Unauthorized" }, { status: 401 });
      }
      const checkpoint = await workerEnv.APOLLO_CHECKPOINT.getByName("primary").fetch(
        "https://checkpoint.local/state",
        { method: "HEAD" },
      );
      const sandbox = await sandboxRequest(
        new Request("http://sandbox.internal/ready"),
        workerEnv,
      );
      const model = await workersAiRequest(
        new Request("http://model.internal/v1/chat/completions", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model: workerEnv.APOLLOBOT_WORKERS_AI_MODEL,
            messages: [{ role: "user", content: "Reply with the single word ready." }],
            max_tokens: 16,
          }),
        }),
        workerEnv,
      );
      const modelPayload = (await model.clone().json().catch(() => ({}))) as Record<
        string,
        unknown
      >;
      const choices = Array.isArray(modelPayload.choices) ? modelPayload.choices : [];
      const first = (choices[0] ?? {}) as Record<string, unknown>;
      const message = (first.message ?? {}) as Record<string, unknown>;
      const checkpointProbe = workerEnv.APOLLO_CHECKPOINT.getByName("diagnostic");
      let checkpointProbeResult: Record<string, unknown>;
      try {
        const put = await checkpointProbe.fetch("https://checkpoint.local/state", {
          method: "PUT",
          headers: { "content-type": "application/gzip" },
          body: new Uint8Array([31, 139, 8, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        });
        const get = await checkpointProbe.fetch("https://checkpoint.local/state");
        checkpointProbeResult = { put: put.status, get: get.status };
      } catch (error) {
        checkpointProbeResult = {
          error: error instanceof Error ? `${error.name}: ${error.message}` : "Unknown error",
        };
      }
      return Response.json({
        checkpoint: { status: checkpoint.status, probe: checkpointProbeResult },
        sandbox: { status: sandbox.status },
        model: {
          status: model.status,
          choices: choices.length,
          content: typeof message.content === "string" && message.content.length > 0,
          reasoning: typeof message.reasoning === "string" && message.reasoning.length > 0,
          messageFields: Object.keys(message).sort(),
          error: modelPayload.error ?? null,
        },
      });
    }
    if (!url.pathname.startsWith("/v1/") && !["/health", "/ready"].includes(url.pathname)) {
      return new Response("Not found", { status: 404 });
    }
    const service = getContainer(workerEnv.APOLLO_SERVICE, "primary");
    const response = await service.fetch(request);
    const secured = new Response(response.body, response);
    secured.headers.set("strict-transport-security", "max-age=31536000; includeSubDomains");
    secured.headers.set("referrer-policy", "no-referrer");
    return secured;
  },
} satisfies ExportedHandler<Bindings>;
