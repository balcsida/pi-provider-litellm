import { type AuthContext, createModels, InMemoryCredentialStore, InMemoryModelsStore } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
// The routing decision belongs to Pi's internal provider composer. Importing the
// installed implementation makes this a canary for changes to that seam instead of
// testing a local reimplementation.
import { composeModelProvider } from "../node_modules/@earendil-works/pi-coding-agent/dist/core/provider-composer.js";
import { createLiteLLMProvider } from "../src/provider.js";
import type { DiscoveredModel } from "../src/types.js";

const CREDENTIAL_ROOT = "https://proxy.example.com";
const FOREIGN_ROOT = "https://foreign.example.com";
const CANARY_CREDENTIAL = ["canary", "credential"].join("-");

type WireRequest = { url: string; authorization: string | null; tenant: string | null };

function model(id: string, api: string, baseUrl: string) {
  return {
    id,
    name: id,
    provider: "litellm",
    api,
    baseUrl,
    reasoning: false,
    input: ["text"] as ("text" | "image")[],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 4096,
    maxTokens: 1024,
  };
}

function discovered(id: string): DiscoveredModel {
  return {
    id,
    name: id,
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 4096,
    maxTokens: 1024,
    api: "openai-completions",
  };
}

function harness(options: { configuredModels: ReturnType<typeof model>[]; discoveredApis?: string[] }) {
  const wire: WireRequest[] = [];
  const payloads: unknown[] = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const headers = new Headers(init?.headers);
    wire.push({
      url: String(input),
      authorization: headers.get("authorization"),
      tenant: headers.get("x-tenant"),
    });
    if (typeof init?.body === "string") payloads.push(JSON.parse(init.body));
    return new Response('data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n', {
      headers: { "content-type": "text/event-stream" },
    });
  });

  const listed = (options.discoveredApis ?? ["openai-completions"]).map((api, index) =>
    model(`listed-${index}`, api, `${CREDENTIAL_ROOT}/v1`),
  );
  const base = createLiteLLMProvider({
    id: "litellm",
    name: "LiteLLM",
    baseUrl: `${CREDENTIAL_ROOT}/v1`,
    auth: {
      apiKey: {
        name: "API key",
        resolve: async () => ({
          auth: { apiKey: CANARY_CREDENTIAL, headers: { "x-tenant": "canary-tenant" } },
        }),
      },
    },
    models: listed as never,
    resolveCredentialRoot: () => CREDENTIAL_ROOT,
    discover: async () => ({ models: [discovered("chat")], source: "model_info" }),
  });

  const config = {
    getProvider: () => ({ models: options.configuredModels.map(({ id, api, baseUrl }) => ({ id, api, baseUrl })) }),
    getProviderIds: () => ["litellm"],
  };
  const composed = composeModelProvider("litellm", base, config as never, undefined);
  const authContext: AuthContext = { env: async () => undefined, fileExists: async () => false };
  const models = createModels({
    credentials: new InMemoryCredentialStore(),
    modelsStore: new InMemoryModelsStore(),
    authContext,
  });
  models.setProvider(composed);
  return { wire, payloads, models };
}

describe("dispatch routing through Pi's provider composer", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("runs the LiteLLM host guard for a protocol in the discovered catalog", async () => {
    const entry = model("configured-chat", "openai-completions", `${FOREIGN_ROOT}/v1`);
    const { wire, models } = harness({ configuredModels: [entry] });

    const result = await models.complete(entry, { messages: [] });

    expect(result.stopReason).toBe("error");
    expect(wire).toEqual([]);
  });

  it("runs the LiteLLM host guard once Responses is in the discovered catalog", async () => {
    const entry = model("configured-responses", "openai-responses", `${FOREIGN_ROOT}/v1`);
    const { wire, models } = harness({
      configuredModels: [entry],
      discoveredApis: ["openai-completions", "openai-responses"],
    });

    const result = await models.complete(entry, { messages: [] });

    expect(result.stopReason).toBe("error");
    expect(wire).toEqual([]);
  });

  it("documents Pi's generic fallback when the configured API is absent from the catalog", async () => {
    const entry = model("configured-responses", "openai-responses", `${FOREIGN_ROOT}/v1`);
    const { wire, models } = harness({ configuredModels: [entry], discoveredApis: ["openai-completions"] });

    await models.complete(entry, { messages: [] });

    // Pi 0.84 infers provider protocol support from the current model list. Because
    // Responses is absent, this bypasses the provider-owned host guard and uses the
    // global Responses implementation with the LiteLLM credential. README documents
    // the limitation; this canary should change when Pi exposes protocol capabilities.
    expect(wire).toEqual([
      {
        url: `${FOREIGN_ROOT}/v1/responses`,
        authorization: `Bearer ${CANARY_CREDENTIAL}`,
        tenant: "canary-tenant",
      },
    ]);
  });

  it("uses generic dispatch for an unsupported configured API absent from the catalog", async () => {
    const entry = model("configured-google", "google-generative-ai", FOREIGN_ROOT);
    const { wire, models } = harness({ configuredModels: [entry], discoveredApis: ["openai-completions"] });

    const result = await models.complete(entry, { messages: [] });

    // The mock covers OpenAI-compatible APIs only, so this generic provider fails
    // before issuing a request. The important seam is that LiteLLM's guard is bypassed.
    expect(result.stopReason).toBe("error");
    expect(wire).toEqual([]);
  });

  it("applies cacheControlFormat only to Chat Completions payloads", async () => {
    // Deliberately escape the Responses compat type to exercise the same completions-only flag on both transports.
    const compat = { cacheControlFormat: "anthropic" } as never;
    const completions = {
      ...model("anthropic/claude-sonnet-4-6-completions", "openai-completions", `${CREDENTIAL_ROOT}/v1`),
      compat,
    };
    const responses = {
      ...model("anthropic/claude-sonnet-4-6-responses", "openai-responses", `${CREDENTIAL_ROOT}/v1`),
      compat,
    };
    const { payloads, models } = harness({
      configuredModels: [completions, responses],
      discoveredApis: ["openai-completions", "openai-responses"],
    });
    const context = {
      systemPrompt: "system",
      messages: [{ role: "user" as const, content: [{ type: "text" as const, text: "hello" }], timestamp: 1 }],
    };

    await models.complete(completions, context, { sessionId: "cache-session", cacheRetention: "short" });
    await models.complete(responses, context, { sessionId: "cache-session", cacheRetention: "short" });

    expect(payloads).toHaveLength(2);
    expect(payloads[0]).toMatchObject({
      model: "anthropic/claude-sonnet-4-6-completions",
      messages: expect.any(Array),
    });
    expect(JSON.stringify(payloads[0])).toContain("cache_control");
    expect(payloads[1]).toMatchObject({
      model: "anthropic/claude-sonnet-4-6-responses",
      prompt_cache_key: "cache-session",
      input: expect.any(Array),
    });
    expect(JSON.stringify(payloads[1])).not.toContain("cache_control");
  });
});
