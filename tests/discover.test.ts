import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { getSupportedThinkingLevels } from "@earendil-works/pi-ai";
import { getModel } from "@earendil-works/pi-ai/compat";
import { afterAll, afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-discover-"));
vi.mock("@earendil-works/pi-coding-agent", () => ({
  getAgentDir: () => agentDir,
}));

import {
  buildCompat,
  discoverModels,
  emitsThinkTags,
  enrichCachedModel,
  modelProtocol,
  moonshotPolicy,
  normalizeBaseUrl,
  resolveModelInfoCatalog,
} from "../src/discover.js";

// Deterministic endpoint mock: only the listed suffixes are served and every
// other URL fails loudly, so a stray fetch can never reach the network or be
// silently satisfied by another endpoint's payload.
function mockEndpoints(routes: Record<string, () => Response>): void {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = input instanceof URL ? input.toString() : String(input);
    for (const [suffix, respond] of Object.entries(routes)) {
      if (url.endsWith(suffix)) return respond();
    }
    throw new Error(`unexpected URL: ${url}`);
  });
}

// No test in this file may reach the network. Every test starts with a fetch that
// refuses, so forgetting to stub an endpoint fails loudly instead of dialling out
// or silently falling through to a real implementation.
beforeEach(() => {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    throw new Error(`unstubbed fetch: ${input instanceof URL ? input.toString() : String(input)}`);
  });
});

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

afterAll(async () => {
  await rm(agentDir, { recursive: true, force: true });
});

afterEach(() => {
  vi.restoreAllMocks();
});

const NO_REASONING_LEVELS = {
  off: null,
  minimal: null,
  low: null,
  medium: null,
  high: null,
  xhigh: null,
  max: null,
};

function cachedReasoningModel(api: "openai-completions" | "openai-responses", overrides: Record<string, unknown> = {}) {
  return {
    id: "cached-reasoning",
    name: "Cached reasoning",
    provider: "litellm",
    api,
    baseUrl: "https://litellm.example.com/v1",
    reasoning: true,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 128_000,
    maxTokens: 16_384,
    ...overrides,
  } as never;
}

describe("normalizeBaseUrl", () => {
  it("rejects insecure non-loopback endpoints", () => {
    expect(() => normalizeBaseUrl("http://litellm.example.com")).toThrow(/HTTPS/);
  });

  it("allows an explicitly configured insecure endpoint", () => {
    expect(normalizeBaseUrl("http://host.docker.internal/v1", true)).toBe("http://host.docker.internal");
  });

  it("does not allow other insecure protocols", () => {
    expect(() => normalizeBaseUrl("ftp://host.docker.internal", true)).toThrow(/HTTPS/);
  });

  it("strips trailing slashes", () => {
    expect(normalizeBaseUrl("https://x.example.com/")).toBe("https://x.example.com");
    expect(normalizeBaseUrl("https://x.example.com///")).toBe("https://x.example.com");
  });

  it("strips a single trailing /v1 suffix", () => {
    expect(normalizeBaseUrl("https://x.example.com/v1")).toBe("https://x.example.com");
    expect(normalizeBaseUrl("https://x.example.com/v1/")).toBe("https://x.example.com");
  });

  it("is case-insensitive on /v1", () => {
    expect(normalizeBaseUrl("https://x.example.com/V1")).toBe("https://x.example.com");
  });

  it("does not strip /v2 or /v1xxx", () => {
    expect(normalizeBaseUrl("https://x.example.com/v2")).toBe("https://x.example.com/v2");
    expect(normalizeBaseUrl("https://x.example.com/v1beta")).toBe("https://x.example.com/v1beta");
  });

  it("preserves a base path that is not /v1", () => {
    expect(normalizeBaseUrl("https://x.example.com/proxy")).toBe("https://x.example.com/proxy");
  });
});

describe("modelProtocol", () => {
  it("keeps an Azure deployment with a non-string API version on Chat", () => {
    expect(
      modelProtocol("opaque-route", {
        litellm_params: { model: "azure/gpt-5", api_version: 20240101 as unknown as string },
      }),
    ).toMatchObject({ api: "openai-completions" });
  });

  it("does not infer a deployment protocol from its public route name", () => {
    expect(modelProtocol("openai/gpt-5", { model_name: "openai/gpt-5" })).toEqual({
      api: "openai-completions",
      compat: { supportsStore: false },
    });
  });

  it("pairs each upstream-selected mode with protocol-specific compatibility", () => {
    expect(modelProtocol("openai/gpt-4o")).toEqual({
      api: "openai-responses",
      compat: undefined,
    });
    expect(modelProtocol("openai/gpt-4o", "responses")).toEqual({
      api: "openai-responses",
      compat: undefined,
    });
    for (const id of ["anthropic/claude-sonnet-4-6", "fable-5", "sonnet-4.6"]) {
      expect(modelProtocol(id, "chat")).toEqual({
        api: "openai-completions",
        compat: { supportsStore: false, cacheControlFormat: "anthropic" },
      });
      expect(modelProtocol(id, "responses")).toEqual({
        api: "openai-responses",
        compat: undefined,
      });
    }
    expect(modelProtocol("moonshotai/kimi-k2", "responses")).toEqual({
      api: "openai-responses",
      compat: { supportsDeveloperRole: false },
    });
  });

  it.each(["azure", "azure_ai"])("requires explicit Responses evidence for %s deployments", (provider) => {
    for (const api_version of [undefined, "2024-12-01-preview", "2025-03-01", "2025-04-01-preview", "v1"]) {
      for (const evidence of [
        { litellm_params: { model: `${provider}/gpt-5`, api_version } },
        { litellm_params: { model: "gpt-5", custom_llm_provider: provider, api_version } },
        {
          litellm_params: { model: "gpt-5", api_version },
          model_info: { mode: "chat", litellm_provider: provider },
        },
      ]) {
        expect(modelProtocol("opaque-route", evidence)).toMatchObject({ api: "openai-completions" });
      }
    }
  });

  it.each([
    { info: { mode: "responses" }, api: "openai-responses" },
    { info: { mode: "chat", supported_endpoints: ["/v1/responses"] }, api: "openai-responses" },
    { info: { mode: "responses", supported_endpoints: ["/v1/chat/completions"] }, api: "openai-completions" },
    { info: { mode: "responses", supported_endpoints: [] }, api: "openai-completions" },
  ])("honors explicit Azure transport evidence: $info", ({ info, api }) => {
    expect(
      modelProtocol("opaque-route", {
        litellm_params: { model: "azure/gpt-5", api_version: "2025-04-01-preview" },
        model_info: info,
      }),
    ).toMatchObject({ api });
  });

  it("uses Chat Completions when the model prefix conflicts with custom_llm_provider", () => {
    expect(
      modelProtocol("gpt-prod", {
        model_name: "gpt-prod",
        litellm_params: { model: "openai/gpt-5", custom_llm_provider: "fireworks_ai" },
        model_info: { mode: "chat" },
      }),
    ).toMatchObject({ api: "openai-completions" });
  });
});

describe("buildCompat", () => {
  it("returns supportsStore: false for non-anthropic models", () => {
    expect(buildCompat("openai/gpt-4o")).toEqual({ supportsStore: false });
    expect(buildCompat("gemini/gemini-2.0-flash")).toEqual({ supportsStore: false });
    expect(buildCompat("gpt-5.5")).toEqual({ supportsStore: false });
  });

  it("adds Moonshot-compatible tool calling flags for Kimi models", () => {
    expect(buildCompat("kimi-k2.6")).toEqual({
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsStrictMode: false,
      maxTokensField: "max_tokens",
    });
    expect(buildCompat("moonshotai/kimi-k2")).toEqual({
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsStrictMode: false,
      maxTokensField: "max_tokens",
    });
  });

  it("adds cacheControlFormat for anthropic-prefixed models", () => {
    expect(buildCompat("anthropic/claude-3-5-sonnet")).toEqual({
      supportsStore: false,
      cacheControlFormat: "anthropic",
    });
  });

  it("adds cacheControlFormat for bare Claude aliases", () => {
    for (const id of ["claude-3-5-sonnet", "opus-4.7", "sonnet-4.6", "haiku-4.5"]) {
      expect(buildCompat(id)).toEqual({
        supportsStore: false,
        cacheControlFormat: "anthropic",
      });
    }
  });

  it("adds cacheControlFormat for routed Anthropic aliases", () => {
    expect(buildCompat("google/claude-sonnet-4-6")).toEqual({
      supportsStore: false,
      cacheControlFormat: "anthropic",
    });
  });

  it("does not match non-Anthropic tokens that start with Anthropic family names", () => {
    expect(buildCompat("openai/sonnetic-gpt")).toEqual({ supportsStore: false });
    expect(buildCompat("vendor/opusflow")).toEqual({ supportsStore: false });
  });

  it("matches case-insensitively", () => {
    expect(buildCompat("Opus-4.7")).toEqual({
      supportsStore: false,
      cacheControlFormat: "anthropic",
    });
    expect(buildCompat("CLAUDE-3-5-SONNET")).toEqual({
      supportsStore: false,
      cacheControlFormat: "anthropic",
    });
  });
});

describe("Kimi reasoning compatibility", () => {
  it("still parses think tags for unknown routes that look like Kimi", () => {
    expect(emitsThinkTags("kimi-k3")).toBe(true);
    expect(emitsThinkTags("openai/gpt-4o")).toBe(false);
  });
});

describe("moonshotPolicy", () => {
  it("keeps request suppression disabled for route-name-only fallback evidence", () => {
    expect(moonshotPolicy("kimi-k2.6")).toEqual({
      normalizeStrictToolMessages: false,
      normalizeThinkTags: true,
      suppressReasoningVisibility: false,
    });
  });

  it("preserves always-thinking output and visibility behavior", () => {
    expect(moonshotPolicy("kimi-k2-thinking")).toEqual({
      normalizeStrictToolMessages: false,
      normalizeThinkTags: false,
      suppressReasoningVisibility: false,
    });
  });
});

describe("enrichCachedModel fallback context window", () => {
  afterEach(() => vi.unstubAllEnvs());

  const cachedFallback = (contextWindow: number, id = "private-route") =>
    cachedReasoningModel("openai-completions", {
      id,
      name: `${id} (no metadata)`,
      reasoning: false,
      contextWindow,
    });

  it("re-applies the configured default to a model cached under the old one", () => {
    vi.stubEnv("LITELLM_DEFAULT_CONTEXT_WINDOW", "922000");

    expect(enrichCachedModel(cachedFallback(128_000))).toMatchObject({ contextWindow: 922_000 });
    // A window already stored under the setting is left where it is.
    expect(enrichCachedModel(cachedFallback(922_000))).toMatchObject({ contextWindow: 922_000 });
  });

  it("still enriches a cached fallback from the catalog after the setting changes", () => {
    vi.stubEnv("LITELLM_DEFAULT_CONTEXT_WINDOW", "922000");

    const enriched = enrichCachedModel(cachedFallback(128_000, "claude-haiku-4-5"));

    // Catalog evidence outranks the fallback; the setting only fills a gap.
    expect(enriched.name).not.toContain("(no metadata)");
    expect(enriched.contextWindow).not.toBe(922_000);
  });

  it("leaves a model that carries real metadata alone", () => {
    vi.stubEnv("LITELLM_DEFAULT_CONTEXT_WINDOW", "922000");

    // A window matching neither default is partial enrichment, not an assumption.
    expect(enrichCachedModel(cachedFallback(128_001))).toMatchObject({ contextWindow: 128_001 });
    const measured = cachedReasoningModel("openai-completions", { reasoning: false, contextWindow: 64_000 });
    expect(enrichCachedModel(measured)).toMatchObject({ contextWindow: 64_000 });
  });
});

describe("enrichCachedModel reasoning policy", () => {
  it("removes a stale thinking level map from a cached non-reasoning model", () => {
    const enriched = enrichCachedModel(
      cachedReasoningModel("openai-completions", {
        reasoning: false,
        thinkingLevelMap: { low: "low", high: "high" },
      }),
    );

    expect(enriched.reasoning).toBe(false);
    expect(enriched).not.toHaveProperty("thinkingLevelMap");
  });

  it("denies Pi default Chat levels when a legacy cache has no level or carrier evidence", () => {
    expect(enrichCachedModel(cachedReasoningModel("openai-completions"))).toMatchObject({
      reasoning: true,
      thinkingLevelMap: NO_REASONING_LEVELS,
    });
  });

  it("denies Pi default Chat levels after catalog enrichment has no level or carrier evidence", () => {
    const enriched = enrichCachedModel(
      cachedReasoningModel("openai-completions", {
        id: "claude-haiku-4-5",
        name: "claude-haiku-4-5 (no metadata)",
        reasoning: false,
      }),
    );

    expect(enriched).toMatchObject({
      name: "Claude Haiku 4.5 (latest)",
      reasoning: true,
      thinkingLevelMap: NO_REASONING_LEVELS,
    });
  });

  it("updates cached fallback transport from the resolved Pi catalog entry", () => {
    const enriched = enrichCachedModel(
      cachedReasoningModel("openai-completions", {
        id: "openai/gpt-4o",
        name: "openai/gpt-4o (no metadata)",
        reasoning: false,
        compat: { supportsStore: false },
      }),
    );

    expect(enriched).toMatchObject({
      name: "GPT-4o",
      api: "openai-responses",
      compat: undefined,
    });
  });

  it("denies cached Chat levels when no compatibility metadata proves a carrier", () => {
    expect(
      enrichCachedModel(
        cachedReasoningModel("openai-completions", {
          thinkingLevelMap: { low: "low", high: "high" },
        }),
      ),
    ).toMatchObject({
      reasoning: true,
      thinkingLevelMap: NO_REASONING_LEVELS,
      compat: { supportsReasoningEffort: false },
    });
  });

  it("closes cached Chat levels when compatibility metadata proves a carrier", () => {
    expect(
      enrichCachedModel(
        cachedReasoningModel("openai-completions", {
          thinkingLevelMap: { low: "low", high: "high" },
          compat: { supportsReasoningEffort: true },
        }),
      ).thinkingLevelMap,
    ).toEqual({ low: "low", high: "high" });
  });

  it("keeps Responses cache behavior unchanged with and without explicit control evidence", () => {
    expect(enrichCachedModel(cachedReasoningModel("openai-responses"))).toMatchObject({
      reasoning: true,
      thinkingLevelMap: NO_REASONING_LEVELS,
    });

    expect(
      enrichCachedModel(
        cachedReasoningModel("openai-responses", {
          thinkingLevelMap: { low: "low", high: "high" },
          litellmResponsesReasoningControl: true,
        }),
      ).thinkingLevelMap,
    ).toEqual({
      off: "none",
      minimal: "minimal",
      low: "low",
      medium: "medium",
      high: "high",
      xhigh: null,
      max: null,
    });
  });
});

describe("context window fallback diagnostic", () => {
  // Reported routes are remembered per process, so each case starts from a fresh module.
  let discover: typeof discoverModels;
  beforeEach(async () => {
    vi.resetModules();
    ({ discoverModels: discover } = await import("../src/discover.js"));
    vi.stubEnv("LITELLM_VERBOSE_DISCOVERY", "1");
    vi.stubEnv("LITELLM_DEFAULT_CONTEXT_WINDOW", undefined);
  });
  afterEach(() => vi.unstubAllEnvs());

  it.each([undefined, null])("reports a defaulted window for limit %s with an escaped route", async (limit) => {
    const route = 'private"\n\u001b[31m';
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, { data: [{ model_name: route, model_info: { mode: "chat", max_input_tokens: limit } }] }),
    });
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);

    const result = await discover("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]).toMatchObject({ contextWindow: 128000, maxTokens: 16384 });
    expect(stderr).toHaveBeenCalledTimes(1);
    const message = String(stderr.mock.calls[0]?.[0]);
    expect(message).toContain('"private\\"\\n\\u001b[31m"');
    expect(message).toContain("1 route(s) default contextWindow to 128000");
    expect(message).toContain("model_info.max_input_tokens");
    expect(message.trimEnd()).not.toContain("\n");
    expect(message).not.toContain("\u001b");
  });

  it.each([undefined, "0", "true"])("stays quiet with LITELLM_VERBOSE_DISCOVERY=%s", async (verbose) => {
    vi.stubEnv("LITELLM_VERBOSE_DISCOVERY", verbose);
    mockEndpoints({
      "/model/info": () => jsonResponse(200, { data: [{ model_name: "private-route", model_info: { mode: "chat" } }] }),
    });
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);

    const result = await discover("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.contextWindow).toBe(128000);
    expect(stderr).not.toHaveBeenCalled();
  });

  it.each([
    { source: "explicit", limit: 128000, backend: undefined },
    { source: "catalog", limit: undefined, backend: "openai/gpt-4o" },
  ])("does not mistake a $source window equal to the default for a fallback", async ({ limit, backend }) => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "private-route",
              litellm_params: { model: backend },
              model_info: { mode: "chat", max_input_tokens: limit },
            },
          ],
        }),
    });
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);

    const result = await discover("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.contextWindow).toBe(128000);
    expect(stderr).not.toHaveBeenCalled();
  });

  it.each([
    { knownLimit: 64000, expected: 64000, warns: false },
    { knownLimit: 1048576, expected: 922000, warns: true },
  ])("reports only a winning configured fallback beside limit $knownLimit", async ({ knownLimit, expected, warns }) => {
    vi.stubEnv("LITELLM_DEFAULT_CONTEXT_WINDOW", "922000");
    const rows = [
      { model_name: "private-route", model_info: { mode: "chat", max_input_tokens: knownLimit } },
      { model_name: "private-route", model_info: { mode: "chat" } },
    ];
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    for (const data of [rows, [...rows].reverse()]) {
      stderr.mockClear();
      vi.resetModules();
      ({ discoverModels: discover } = await import("../src/discover.js"));
      mockEndpoints({ "/model/info": () => jsonResponse(200, { data }) });

      const result = await discover("https://litellm.example.com", "sk-test", { modelsDev: false });

      expect(result.models[0]?.contextWindow).toBe(expected);
      if (warns) {
        expect(stderr.mock.calls).toEqual([[expect.stringContaining("default contextWindow to 922000")]]);
      } else {
        expect(stderr).not.toHaveBeenCalled();
      }
    }
  });

  it("reports each published wildcard child once, excluding exact and tighter measured routes", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            { model_name: "private/*", model_info: { mode: "chat" } },
            { model_name: "private/small*", model_info: { mode: "chat", max_input_tokens: 64000 } },
            { model_name: "private/exact", model_info: { mode: "chat", max_input_tokens: 128000 } },
          ],
        }),
      "/v1/models": () =>
        jsonResponse(200, {
          data: ["private/child", "private/child", "private/exact", "private/small-child"].map((id) => ({ id })),
        }),
    });
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);

    const result = await discover("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models.map(({ id, contextWindow }) => [id, contextWindow]).sort()).toEqual([
      ["private/child", 128000],
      ["private/exact", 128000],
      ["private/small-child", 64000],
    ]);
    expect(stderr.mock.calls).toEqual([
      [
        expect.stringMatching(
          /^LiteLLM discovery: 1 route\(s\) default contextWindow to 128000; .*: "private\/child"\n$/,
        ),
      ],
    ]);
  });

  it.each(["chat", "embedding"])("does not report unpublished %s routes", async (mode) => {
    mockEndpoints({
      "/model/info": () => jsonResponse(200, { data: [{ model_name: "private/*", model_info: { mode } }] }),
      "/v1/models": () => jsonResponse(200, { data: [] }),
    });
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);

    const result = await discover("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models).toEqual([]);
    expect(stderr.mock.calls.join("\n")).not.toContain("default contextWindow");
  });

  it("reports many defaulted routes on one bounded line, once per process", async () => {
    const data = Array.from({ length: 50 }, (_, index) => ({
      model_name: `route-${index}`,
      model_info: { mode: "chat" },
    }));
    mockEndpoints({ "/model/info": () => jsonResponse(200, { data }) });
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);

    await discover("https://litellm.example.com", "sk-test", { modelsDev: false });
    await discover("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(stderr.mock.calls).toEqual([
      [expect.stringMatching(/^LiteLLM discovery: 50 route\(s\) default contextWindow to 128000; .* \(\+47 more\)\n$/)],
    ]);
  });
});

describe("discoverModels via /model/info", () => {
  it.each([
    { mode: "chat", api: "openai-completions", params: { model: "chatgpt/gpt-5.6-sol" } },
    {
      mode: "responses",
      api: "openai-responses",
      params: { model: "gpt-5.6-sol", custom_llm_provider: "chatgpt" },
    },
  ])("enriches a ChatGPT subscription alias while retaining $mode routing", async ({ mode, api, params }) => {
    const entry = {
      model_name: "high",
      litellm_params: params,
      model_info: {
        mode,
        supported_endpoints: [api === "openai-responses" ? "/v1/responses" : "/v1/chat/completions"],
      },
    };
    // The synchronous fallback must work even when public-catalog loading exceeds its budget.
    expect(resolveModelInfoCatalog(entry)).toMatchObject({
      provider: "chatgpt",
      catalogModelId: "gpt-5.6-sol",
      contextWindow: 272_000,
      maxTokens: 128_000,
    });
    mockEndpoints({ "/model/info": () => jsonResponse(200, { data: [entry] }) });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]).toMatchObject({
      id: "high",
      api,
      contextWindow: 272_000,
      maxTokens: 128_000,
      reasoning: true,
    });
  });

  it("enriches the backend identity from models.dev before synchronous group reduction", async () => {
    const dir = await mkdtemp(join(tmpdir(), "litellm-model-info-catalog-"));
    const cachePath = join(dir, "models-dev.json");
    await writeFile(
      cachePath,
      JSON.stringify({
        fetchedAt: 1,
        catalog: {
          "fireworks-ai": {
            models: {
              "accounts/fireworks/models/kimi-k3": {
                modalities: { input: ["text", "image"] },
                limit: { context: 131_072, output: 16_384 },
                cost: { input: 0.4, output: 2, cache_read: 0.2, cache_write: 0.8 },
              },
            },
          },
        },
      }),
    );
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "kimi-k3",
              litellm_params: { model: "azure_ai/FW-Kimi-K3", custom_llm_provider: "azure" },
              model_info: {
                id: "deployment-a",
                mode: "chat",
                litellm_provider: "azure",
                base_model: "fireworks_ai/accounts/fireworks/models/kimi-k3",
                max_input_tokens: 262_144,
                input_cost_per_token: 0.000001,
              },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {
      modelsDev: false,
      modelsDevCachePath: cachePath,
    });

    expect(result.models[0]).toMatchObject({
      id: "kimi-k3",
      input: ["text", "image"],
      contextWindow: 131_072,
      maxTokens: 16_384,
      cost: { input: 1, output: 2, cacheRead: 0.2, cacheWrite: 0.8 },
    });
  });

  it("fills partial models.dev cost from Pi catalog pricing", async () => {
    const dir = await mkdtemp(join(tmpdir(), "litellm-model-info-cost-"));
    const cachePath = join(dir, "models-dev.json");
    await writeFile(
      cachePath,
      JSON.stringify({
        fetchedAt: 1,
        catalog: {
          openai: {
            models: {
              "gpt-5.5": { cost: { input: 7 } },
            },
          },
        },
      }),
    );
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "gpt-route",
              litellm_params: { model: "openai/gpt-5.5" },
              model_info: { mode: "chat" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {
      modelsDev: false,
      modelsDevCachePath: cachePath,
    });

    expect(result.models[0]).toMatchObject({
      name: "gpt-route",
      cost: { input: 7, output: 30, cacheRead: 0.5, cacheWrite: 0 },
    });
  });

  it("rejects a negative models.dev input price and marks the reduced model incomplete", async () => {
    const dir = await mkdtemp(join(tmpdir(), "litellm-model-info-negative-cost-"));
    const cachePath = join(dir, "models-dev.json");
    await writeFile(
      cachePath,
      JSON.stringify({
        fetchedAt: 1,
        catalog: {
          private: {
            models: {
              "priced-model": {
                modalities: { input: ["text"] },
                limit: { context: 64_000, output: 8_000 },
                cost: { input: -1, output: 2, cache_read: 0, cache_write: 0 },
              },
            },
          },
        },
      }),
    );
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "private-route",
              litellm_params: { model: "private/priced-model" },
              model_info: { mode: "chat", supports_reasoning: false, supports_vision: false },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {
      modelsDev: false,
      modelsDevCachePath: cachePath,
    });

    expect(result.models[0]).toMatchObject({
      name: "private-route (incomplete metadata)",
      cost: { input: 0, output: 2, cacheRead: 0, cacheWrite: 0 },
    });
  });

  it("keeps partial models.dev pricing incomplete without a Pi catalog fallback", async () => {
    const dir = await mkdtemp(join(tmpdir(), "litellm-model-info-partial-cost-"));
    const cachePath = join(dir, "models-dev.json");
    await writeFile(
      cachePath,
      JSON.stringify({
        fetchedAt: 1,
        catalog: {
          private: {
            models: {
              "priced-model": {
                modalities: { input: ["text"] },
                limit: { context: 64_000, output: 8_000 },
                cost: { input: 7 },
              },
            },
          },
        },
      }),
    );
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "private-route",
              litellm_params: { model: "private/priced-model" },
              model_info: { mode: "chat", supports_reasoning: false },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {
      modelsDev: false,
      modelsDevCachePath: cachePath,
    });

    expect(result.models[0]).toMatchObject({
      name: "private-route (incomplete metadata)",
      cost: { input: 7, output: 0, cacheRead: 0, cacheWrite: 0 },
    });
  });

  it.each([undefined, false])("combines explicit reasoning=%s with models.dev effort evidence", async (reasoning) => {
    const dir = await mkdtemp(join(tmpdir(), "litellm-model-info-reasoning-"));
    const cachePath = join(dir, "models-dev.json");
    await writeFile(
      cachePath,
      JSON.stringify({
        fetchedAt: 1,
        catalog: {
          private: {
            models: {
              reasoner: { reasoning_options: { type: "effort", values: ["low", "high"] } },
            },
          },
        },
      }),
    );
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "private-route",
              litellm_params: { model: "private/reasoner" },
              model_info: { mode: "chat", supports_reasoning: reasoning },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {
      modelsDev: false,
      modelsDevCachePath: cachePath,
    });

    expect(result.models[0]).toMatchObject({
      reasoning: reasoning !== false,
      name: "private-route (incomplete metadata)",
    });
    if (reasoning === false) expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
    else expect(result.models[0]?.thinkingLevelMap).toEqual(NO_REASONING_LEVELS);
  });

  it("lets initial public-catalog callers abort independently", async () => {
    const dir = await mkdtemp(join(tmpdir(), "litellm-model-info-abort-"));
    const cachePath = join(dir, "models-dev.json");
    const first = new AbortController();
    const second = new AbortController();
    let modelsDevRequests = 0;
    let resolveModelsDev!: (response: Response) => void;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            { model_name: "gpt-route", litellm_params: { model: "openai/gpt-5.5" }, model_info: { mode: "chat" } },
          ],
        });
      }
      if (url === "https://models.dev/api.json") {
        modelsDevRequests++;
        return new Promise<Response>((resolve) => {
          resolveModelsDev = resolve;
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const discover = (signal: AbortSignal) =>
      discoverModels("https://litellm.example.com", "sk-test", { modelsDevCachePath: cachePath, signal });
    const firstResult = discover(first.signal);
    const secondResult = discover(second.signal);

    await vi.waitFor(() => expect(modelsDevRequests).toBe(1));
    first.abort(new Error("first aborted"));
    await expect(firstResult).rejects.toThrow("first aborted");
    resolveModelsDev(
      jsonResponse(200, {
        openai: { models: { "gpt-5.5": { limit: { context: 1_050_000 }, cost: { input: 5, output: 30 } } } },
      }),
    );

    await expect(secondResult).resolves.toMatchObject({ models: [{ id: "gpt-route", contextWindow: 1_050_000 }] });
    expect(modelsDevRequests).toBe(1);
  });

  it("returns the activation seed before its caller signal aborts when models.dev hangs", async () => {
    const dir = await mkdtemp(join(tmpdir(), "litellm-model-info-seed-timeout-"));
    const cachePath = join(dir, "models-dev.json");
    const controller = new AbortController();
    const abortTimer = setTimeout(() => controller.abort(new Error("seed aborted")), 3_000);
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "private-route",
              litellm_params: { model: "private/catalog-model" },
              model_info: { mode: "chat" },
            },
          ],
        });
      }
      if (url === "https://models.dev/api.json") return new Promise<Response>(() => {});
      throw new Error(`unexpected URL: ${url}`);
    });

    try {
      await expect(
        discoverModels("https://litellm.example.com", "sk-test", {
          modelsDevCachePath: cachePath,
          signal: controller.signal,
          timeoutMs: 5_000,
        }),
      ).resolves.toMatchObject({
        models: [{ id: "private-route", name: "private-route (incomplete metadata)", contextWindow: 128_000 }],
      });
      expect(controller.signal.aborted).toBe(false);
    } finally {
      clearTimeout(abortTimer);
    }
  });

  it("lets each caller stop waiting for best-effort enrichment at its own budget", async () => {
    const dir = await mkdtemp(join(tmpdir(), "litellm-model-info-timeout-"));
    const cachePath = join(dir, "models-dev.json");
    let modelsDevRequests = 0;
    let resolveModelsDev!: (response: Response) => void;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "private-route",
              litellm_params: { model: "private/catalog-model" },
              model_info: { mode: "chat" },
            },
          ],
        });
      }
      if (url === "https://models.dev/api.json") {
        modelsDevRequests++;
        return new Promise<Response>((resolve) => {
          resolveModelsDev = resolve;
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const short = discoverModels("https://short.example.com", "sk-test", {
      modelsDevCachePath: cachePath,
      timeoutMs: 30,
    });
    const long = discoverModels("https://long.example.com", "sk-test", {
      modelsDevCachePath: cachePath,
      timeoutMs: 30_000,
    });

    await vi.waitFor(() => expect(modelsDevRequests).toBe(1));
    await expect(short).resolves.toMatchObject({
      models: [{ id: "private-route", name: "private-route (incomplete metadata)", contextWindow: 128_000 }],
    });
    resolveModelsDev(
      jsonResponse(200, {
        private: { models: { "catalog-model": { limit: { context: 1_050_000 }, cost: { input: 5, output: 30 } } } },
      }),
    );
    await expect(long).resolves.toMatchObject({ models: [{ id: "private-route", contextWindow: 1_050_000 }] });
    expect(modelsDevRequests).toBe(1);
  });

  it("keeps models with a null mode", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [{ model_name: "local/model", model_info: { mode: null } }],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models.map((model) => model.id)).toEqual(["local/model"]);
  });

  it("withholds backend family when the model prefix conflicts with custom_llm_provider", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "gpt-prod",
            litellm_params: { model: "openai/gpt-5", custom_llm_provider: "fireworks_ai" },
            model_info: { mode: "chat" },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({ id: "gpt-prod", api: "openai-completions" });
    expect(result.models[0]).not.toHaveProperty("litellmBackendFamily");
  });

  it.each([false, true])(
    "keeps an Azure group on Chat when a sibling lacks Responses evidence (%s)",
    async (explicitSibling) => {
      mockEndpoints({
        "/model/info": () =>
          jsonResponse(200, {
            data: [false, explicitSibling].map((responses, index) => ({
              model_name: "azure-chat-route",
              litellm_params: { model: "azure/gpt-5", api_version: "2025-04-01-preview" },
              model_info: {
                id: `deployment-${index}`,
                mode: responses ? "responses" : "chat",
                supports_reasoning: true,
                supported_openai_params: ["reasoning_effort"],
              },
            })),
          }),
      });

      const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

      expect(result.models[0]).toMatchObject({
        api: "openai-completions",
        reasoning: true,
        compat: { supportsReasoningEffort: true },
      });
      expect(getSupportedThinkingLevels(result.models[0] as never)).toContain("medium");
    },
  );

  it.each(["azure", "azure_ai"])("preserves %s-specific context limits through discovery", async (provider) => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "large-context-route",
              litellm_params: { model: `${provider}/gpt-5.6-sol` },
              model_info: {
                mode: "chat",
                litellm_provider: provider,
                max_input_tokens: 1_050_000,
                max_output_tokens: 128_000,
              },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]).toMatchObject({ contextWindow: 1_050_000, maxTokens: 128_000 });
  });

  it("reduces mixed Azure deployment versions to Chat Completions", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "gpt-production",
            litellm_params: { model: "azure/gpt-5", api_version: "2025-04-01-preview" },
            model_info: { mode: "chat" },
          },
          {
            model_name: "gpt-production",
            litellm_params: { model: "azure/gpt-5", api_version: "2024-12-01-preview" },
            model_info: { mode: "chat" },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      id: "gpt-production",
      api: "openai-completions",
      litellmBackendFamily: "openai",
      litellmDiscoveryVersion: 3,
    });
  });

  it("parses a /model/info success response with cost mapping", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "anthropic/claude-3-5-sonnet",
              litellm_params: { model: "anthropic/claude-3-5-sonnet" },
              model_info: {
                mode: "chat",
                max_input_tokens: 200000,
                max_output_tokens: 8192,
                supports_vision: true,
                supports_reasoning: false,
                input_cost_per_token: 0.000003,
                output_cost_per_token: 0.000015,
                cache_read_input_token_cost: 0.0000003,
                cache_creation_input_token_cost: 0.00000375,
              },
            },
            {
              model_name: "openai/gpt-4o",
              litellm_params: { model: "openai/gpt-4o" },
              model_info: {
                mode: "chat",
                max_input_tokens: 128000,
                max_output_tokens: 16384,
              },
            },
            {
              model_name: "openai/text-embedding-3-large",
              model_info: { mode: "embedding" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.source).toBe("model_info");
    // embedding model filtered out by mode !== "chat"
    expect(result.models).toHaveLength(2);

    const anthropic = result.models.find((m) => m.id === "anthropic/claude-3-5-sonnet");
    expect(anthropic).toMatchObject({
      id: "anthropic/claude-3-5-sonnet",
      name: "anthropic/claude-3-5-sonnet",
      contextWindow: 200000,
      maxTokens: 8192,
      input: ["text", "image"],
      compat: { supportsStore: false, cacheControlFormat: "anthropic" },
    });
    // cost is per-token in LiteLLM, per-million-tokens in pi-ai
    expect(anthropic?.cost).toEqual({ input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 });

    const openai = result.models.find((m) => m.id === "openai/gpt-4o");
    expect(openai).toMatchObject({
      id: "openai/gpt-4o",
      name: "openai/gpt-4o",
      input: ["text", "image"],
      api: "openai-responses",
      compat: undefined,
    });
  });

  it("maps LiteLLM reasoning effort capabilities for a singleton", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "custom/reasoner",
            litellm_params: { allowed_openai_params: ["reasoning_effort"] },
            model_info: {
              mode: "chat",
              supports_reasoning: true,
              supports_none_reasoning_effort: true,
              supports_minimal_reasoning_effort: false,
              supports_low_reasoning_effort: false,
              supports_xhigh_reasoning_effort: false,
              supports_max_reasoning_effort: true,
            },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.thinkingLevelMap).toEqual({
      off: "none",
      minimal: null,
      low: null,
      xhigh: null,
      max: "max",
    });
  });

  it("maps router medium and high effort flags for a singleton", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "custom/reasoner",
            litellm_params: { allowed_openai_params: ["reasoning_effort"] },
            model_info: {
              mode: "chat",
              supports_reasoning: true,
              supports_none_reasoning_effort: true,
              supports_low_reasoning_effort: true,
              supports_medium_reasoning_effort: true,
              supports_high_reasoning_effort: true,
              supports_xhigh_reasoning_effort: true,
              supports_max_reasoning_effort: true,
            },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.thinkingLevelMap).toMatchObject({
      off: "none",
      low: "low",
      medium: "medium",
      high: "high",
      xhigh: "xhigh",
      max: "max",
    });
  });

  it("denies medium and high when the router reports them unsupported", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "custom/reasoner",
            litellm_params: { allowed_openai_params: ["reasoning_effort"] },
            model_info: {
              mode: "chat",
              supports_reasoning: true,
              supports_low_reasoning_effort: true,
              supports_medium_reasoning_effort: false,
              supports_high_reasoning_effort: false,
            },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.thinkingLevelMap).toMatchObject({ low: "low", medium: null, high: null });
  });

  it("keeps max selectable on a Responses route that opts into it", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "high",
            litellm_params: { model: "chatgpt/gpt-5.6-sol" },
            model_info: {
              mode: "responses",
              litellm_provider: "chatgpt",
              supported_openai_params: ["reasoning_effort"],
              supports_reasoning: true,
              supports_none_reasoning_effort: true,
              supports_low_reasoning_effort: true,
              supports_medium_reasoning_effort: true,
              supports_high_reasoning_effort: true,
              supports_xhigh_reasoning_effort: true,
              supports_max_reasoning_effort: true,
            },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.api).toBe("openai-responses");
    expect(result.models[0]?.thinkingLevelMap).toMatchObject({
      off: "none",
      low: "low",
      medium: "medium",
      high: "high",
      xhigh: "xhigh",
      max: "max",
    });
  });

  // Issue #182: the Codex catalog map states only xhigh, max, and minimal, so the
  // standard levels it omits must stay at Pi's defaults rather than be denied.
  it("keeps standard levels a Codex catalog map leaves implicit", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "gpt-5.6-sol",
            litellm_params: { model: "chatgpt/gpt-5.6-sol" },
            model_info: {
              mode: "responses",
              litellm_provider: "chatgpt",
              supported_openai_params: ["reasoning_effort"],
              supports_reasoning: null,
              supports_minimal_reasoning_effort: false,
              supports_xhigh_reasoning_effort: true,
              supports_max_reasoning_effort: true,
            },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });
    const model = { ...result.models[0]!, provider: "litellm", baseUrl: "https://proxy.example.com" };

    expect(getSupportedThinkingLevels(model)).toEqual(["off", "low", "medium", "high", "xhigh", "max"]);
  });

  // Issue #167: LiteLLM reads a declared level list whole, ahead of null flags.
  it("reads a declared reasoning_effort_levels list as the complete level set", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "zai-org/GLM-5.3",
            litellm_params: { model: "hosted_vllm/zai-org/GLM-5.3", allowed_openai_params: ["reasoning_effort"] },
            model_info: {
              mode: "chat",
              supports_reasoning: true,
              reasoning_effort_levels: ["none", "low", "high", "max"],
              supports_none_reasoning_effort: null,
              supports_minimal_reasoning_effort: null,
              supports_low_reasoning_effort: null,
              supports_xhigh_reasoning_effort: null,
              supports_max_reasoning_effort: null,
            },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });
    const model = { ...result.models[0]!, provider: "litellm", baseUrl: "https://proxy.example.com" };

    expect(getSupportedThinkingLevels(model)).toEqual(["off", "low", "high", "max"]);
    expect(model.thinkingLevelMap).toMatchObject({ off: "none", max: "max" });
  });

  it("merges singleton router effort flags into supported Responses levels", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "openai/gpt-5.6-luna",
            litellm_params: { model: "openai/gpt-5.6-luna", allowed_openai_params: ["reasoning_effort"] },
            model_info: {
              mode: "chat",
              supports_reasoning: true,
              supports_xhigh_reasoning_effort: false,
              supports_max_reasoning_effort: true,
            },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.thinkingLevelMap).toMatchObject({ off: "none", xhigh: null, max: "max" });
  });

  it("uses family-only identity to look up models.dev under its public provider", async () => {
    const dir = await mkdtemp(join(tmpdir(), "litellm-model-info-family-catalog-"));
    const cachePath = join(dir, "models-dev.json");
    await writeFile(
      cachePath,
      JSON.stringify({
        fetchedAt: 1,
        catalog: {
          anthropic: {
            models: {
              "opus-4.8": { limit: { context: 750_000, output: 96_000 } },
            },
          },
        },
      }),
    );
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "opus-4.8",
              litellm_params: { model: "opus-4.8" },
              model_info: { mode: "chat" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {
      modelsDev: false,
      modelsDevCachePath: cachePath,
    });

    expect(result.source).toBe("model_info");
    expect(result.models[0]).toMatchObject({
      id: "opus-4.8",
      name: "opus-4.8",
      reasoning: true,
      input: ["text", "image"],
      contextWindow: 750_000,
      maxTokens: 96_000,
    });
    expect(result.models[0]?.cost.input).toBeGreaterThan(0);
  });

  it("withholds catalog metadata when /model/info supplies only the public route name", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [{ model_name: "opus-4.8", model_info: { mode: "chat" } }],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.source).toBe("model_info");
    expect(result.models[0]).toMatchObject({
      id: "opus-4.8",
      name: "opus-4.8 (incomplete metadata)",
      reasoning: false,
      input: ["text"],
      contextWindow: 128_000,
      maxTokens: 16_384,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    });
  });

  it("withholds catalog metadata when backend evidence is only whitespace", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "openai/gpt-5",
              litellm_params: { model: "   " },
              model_info: { mode: "chat" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.source).toBe("model_info");
    expect(result.models[0]).toMatchObject({
      id: "openai/gpt-5",
      name: "openai/gpt-5 (incomplete metadata)",
      reasoning: false,
      input: ["text"],
      contextWindow: 128_000,
      maxTokens: 16_384,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    });
  });

  it("preserves catalog pricing tiers for /model/info models", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "openai/gpt-5.5",
            litellm_params: { model: "openai/gpt-5.5" },
            model_info: { mode: "chat" },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.cost?.tiers).toEqual([
      { inputTokensAbove: 272000, input: 10, output: 45, cacheRead: 1, cacheWrite: 0 },
    ]);
  });

  it("preserves catalog xhigh and max on Responses", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "openai/gpt-5.6-luna",
            litellm_params: { model: "openai/gpt-5.6-luna", allowed_openai_params: ["reasoning_effort"] },
            model_info: { mode: "chat", supports_xhigh_reasoning_effort: true, supports_max_reasoning_effort: true },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.thinkingLevelMap).toMatchObject({ off: "none", xhigh: "xhigh", max: "max" });
  });

  it("reduces duplicate model ids conservatively instead of merging richer fields", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            { model_name: "custom-model", model_info: { mode: "chat" } },
            {
              model_name: "custom-model",
              model_info: {
                mode: "chat",
                max_input_tokens: 200000,
                max_output_tokens: 8192,
                input_cost_per_token: 0.000003,
                output_cost_per_token: 0.000015,
              },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.source).toBe("model_info");
    expect(result.models).toHaveLength(1);
    expect(result.models[0]).toMatchObject({
      id: "custom-model",
      name: "custom-model (incomplete metadata)",
      contextWindow: 128000,
      maxTokens: 8192,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    });
  });

  it("falls back to Chat and group guarantees for mixed deployments regardless of row order", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const route = `shared-route-${process.pid}-${Date.now()}-${Math.random()}`;
    const deployments = [
      {
        model_name: route,
        litellm_params: { model: "openai/gpt-4o" },
        model_info: {
          id: "deployment-a",
          mode: "responses",
          supports_reasoning: true,
          supports_vision: true,
          max_input_tokens: 128000,
          max_output_tokens: 16384,
          input_cost_per_token: 0.000005,
          output_cost_per_token: 0.000015,
          cache_read_input_token_cost: 0.0000025,
          cache_creation_input_token_cost: 0,
        },
      },
      {
        model_name: route,
        litellm_params: { model: "internal/unknown" },
        model_info: {
          id: "deployment-b",
          mode: "chat",
          supports_reasoning: false,
          supports_vision: false,
          max_input_tokens: 64000,
          max_output_tokens: 8192,
        },
      },
    ];
    for (const rows of [deployments, [...deployments].reverse()]) {
      mockEndpoints({ "/model/info": () => jsonResponse(200, { data: rows }) });
      const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });
      expect(result.models).toEqual([
        expect.objectContaining({
          id: route,
          name: `${route} (incomplete metadata)`,
          reasoning: false,
          input: ["text"],
          contextWindow: 64000,
          maxTokens: 8192,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        }),
      ]);
      expect(result.models[0]).toHaveProperty("api", "openai-completions");
      expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
    }
    expect(stderr).toHaveBeenCalledTimes(1);
  });

  it.each([
    ["moonshot", "moonshot/kimi-k2.6"],
    ["gemini", "gemini/gemini-3.1-pro-preview"],
    ["xai", "xai/grok-4.5"],
  ])("maps the %s adapter conservatively to its catalog", async (adapter, backend) => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: `${adapter}-route`,
            litellm_params: { model: backend },
            model_info: { mode: "chat", litellm_provider: adapter },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.name).toBe(`${adapter}-route`);
    expect(result.models[0]?.name).not.toContain("no metadata");
    expect(result.models[0]?.contextWindow).toBeGreaterThan(128_000);
    expect(result.models[0]?.cost.input).toBeGreaterThan(0);
    expect(result.models[0]?.id).toBe(`${adapter}-route`);
    if (adapter === "gemini") {
      expect(result.models[0]?.litellmPolicy?.normalizeGeminiReasoningEffort).toBe(true);
    }
  });

  it("trims backend candidates before catalog resolution", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "spacey",
            model_info: { mode: "chat", base_model: " openai/gpt-4o " },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]).toMatchObject({ name: "spacey", input: ["text", "image"], contextWindow: 128_000 });
    expect(result.models[0]?.cost.input).toBeGreaterThan(0);
  });

  it("does not use a qualified public route as evidence for a multi-deployment group", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const route = `openai/gpt-5.5-${process.pid}-${Date.now()}-${Math.random()}`;
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: route,
            litellm_params: { model: "openai/gpt-5.5" },
            model_info: { id: "known", mode: "chat" },
          },
          {
            model_name: route,
            litellm_params: { model: "internal/mystery" },
            model_info: { id: "unknown", mode: "chat" },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]).toMatchObject({
      name: `${route} (incomplete metadata)`,
      reasoning: false,
      input: ["text"],
      contextWindow: 128_000,
      maxTokens: 16_384,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    });
    expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
    expect(stderr).toHaveBeenCalledTimes(1);
    expect(String(stderr.mock.calls[0]?.[0])).toContain(route);
  });

  it.each([
    ["opaque backend model", { litellm_params: { model: "internal/mystery" }, model_info: {} }],
    ["opaque base model", { model_info: { base_model: "internal/mystery" } }],
  ])("withholds catalog metadata for a singleton with %s evidence", async (_case, evidence) => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "openai/gpt-5.5",
              ...evidence,
              model_info: { ...evidence.model_info, id: "only", mode: "chat" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]).toMatchObject({
      id: "openai/gpt-5.5",
      name: "openai/gpt-5.5 (incomplete metadata)",
      reasoning: false,
      input: ["text"],
      contextWindow: 128_000,
      maxTokens: 16_384,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    });
    expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
  });

  it("treats repeated identified deployment rows as one effective singleton", async () => {
    const deployment = {
      model_name: "openai/gpt-5.5",
      litellm_params: { model: "openai/gpt-5.5" },
      model_info: { id: "deployment-a", mode: "chat" },
    };
    const fetchMock = vi.spyOn(globalThis, "fetch");
    for (const data of [[deployment], [deployment, deployment]]) {
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { data }));
      const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });
      expect(result.models[0]).toMatchObject({
        id: "openai/gpt-5.5",
        name: "openai/gpt-5.5",
        reasoning: true,
        contextWindow: 272_000,
        maxTokens: 128_000,
      });
    }
  });

  it("keeps unresolved Azure Foundry backend evidence incomplete", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "foundry-route",
              litellm_params: { model: " azure_ai/DeepSeek-V4 " },
              model_info: { mode: "chat", litellm_provider: "azure_ai" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]).toMatchObject({
      id: "foundry-route",
      name: "foundry-route (incomplete metadata)",
      reasoning: false,
      contextWindow: 128_000,
    });
  });

  it("uses the backend candidate that supplies catalog authority", () => {
    expect(
      resolveModelInfoCatalog({
        model_name: "mixed-evidence",
        litellm_params: { model: "internal/claude-magic" },
        model_info: { mode: "chat", base_model: "openai/gpt-4o" },
      }),
    ).toMatchObject({ provider: "openai" });

    expect(
      resolveModelInfoCatalog({
        model_name: "aliased",
        litellm_params: { model: "anthropic/opus-4-7" },
        model_info: { mode: "chat" },
      }),
    ).toMatchObject({ provider: "anthropic", catalogModelId: "claude-opus-4-7" });
  });

  it("uses base_model as authority over configured model and transport adapter", () => {
    expect(
      resolveModelInfoCatalog({
        model_name: "kimi-k3",
        litellm_params: { model: "azure_ai/FW-Kimi-K3", custom_llm_provider: "azure" },
        model_info: {
          mode: "chat",
          litellm_provider: "azure",
          base_model: "fireworks_ai/accounts/fireworks/models/kimi-k3",
        },
      }),
    ).toMatchObject({ provider: "fireworks_ai", catalogModelId: "accounts/fireworks/models/kimi-k3" });
  });

  it("uses custom_llm_provider as catalog authority for an unprefixed backend model", () => {
    expect(
      resolveModelInfoCatalog({
        model_name: "kimi-route",
        litellm_params: { model: "kimi-k3", custom_llm_provider: "moonshot" },
        model_info: { mode: "chat" },
      }),
    ).toMatchObject({ provider: "moonshot", catalogModelId: "kimi-k3" });
  });

  it("withholds catalog authority when custom_llm_provider conflicts with the model prefix", async () => {
    const entry = {
      model_name: "openai/gpt-4o",
      litellm_params: { model: "openai/gpt-4o", custom_llm_provider: "anthropic" },
      model_info: { mode: "chat" },
    };
    expect(resolveModelInfoCatalog(entry)).toBeUndefined();
    mockEndpoints({ "/model/info": () => jsonResponse(200, { data: [entry] }) });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]).toMatchObject({
      id: "openai/gpt-4o",
      name: "openai/gpt-4o (incomplete metadata)",
      reasoning: false,
      input: ["text"],
      contextWindow: 128_000,
      maxTokens: 16_384,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    });
  });

  it("keeps the catalog provider stable across asymmetric public-catalog hits", () => {
    const hit = {
      source: "models.dev" as const,
      provider: "fireworks-ai",
      modelId: "accounts/fireworks/models/kimi-k3",
      limits: { context: 131_072 },
    };
    const publicCatalog = {
      lookup: (_provider: string | undefined, id: string) => (id.endsWith("kimi-k3") ? hit : undefined),
    };
    const entry = (model: string) => ({
      model_name: "kimi-route",
      litellm_params: { model },
      model_info: { mode: "chat" },
    });

    expect(
      resolveModelInfoCatalog(entry("fireworks_ai/accounts/fireworks/models/kimi-k3"), publicCatalog),
    ).toMatchObject({ provider: "fireworks_ai" });
    expect(resolveModelInfoCatalog(entry("fireworks_ai/accounts/fireworks/models/other"), publicCatalog)).toMatchObject(
      {
        provider: "fireworks_ai",
      },
    );
  });

  it("canonicalizes a models.dev-only model across qualified and bare spellings", () => {
    const hit = {
      source: "models.dev" as const,
      provider: "anthropic",
      modelId: "claude-future",
      limits: { context: 200_000 },
    };
    const publicCatalog = {
      lookup: (_provider: string | undefined, id: string) => (id.endsWith("claude-future") ? hit : undefined),
    };
    const entry = (model: string, custom?: string) => ({
      model_name: "future-route",
      litellm_params: { model, ...(custom ? { custom_llm_provider: custom } : {}) },
      model_info: { mode: "chat" },
    });

    const qualified = resolveModelInfoCatalog(entry("anthropic/claude-future"), publicCatalog);
    const bare = resolveModelInfoCatalog(entry("claude-future", "anthropic"), publicCatalog);
    const suffixed = resolveModelInfoCatalog(entry("anthropic/us.claude-future"), publicCatalog);

    expect(qualified).toMatchObject({ provider: "anthropic", catalogModelId: "claude-future", contextWindow: 200_000 });
    expect(bare?.catalogModelId).toBe("claude-future");
    expect(suffixed?.catalogModelId).toBe("claude-future");
  });

  it("uses base_model before a conflicting configured model", () => {
    expect(
      resolveModelInfoCatalog({
        model_name: "conflicting-evidence",
        litellm_params: { model: "openai/gpt-4o" },
        model_info: { mode: "chat", litellm_provider: "openai", base_model: "anthropic/opus-4-7" },
      }),
    ).toMatchObject({ provider: "anthropic", catalogModelId: "claude-opus-4-7" });
  });

  it("grants catalog authority when prefixed and family-only deployments identify one model", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const route = `anthropic-alias-group-${process.pid}-${Date.now()}-${Math.random()}`;
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: route,
              litellm_params: { model: "anthropic/claude-opus-4-7" },
              model_info: { id: "canonical", mode: "chat" },
            },
            {
              model_name: route,
              litellm_params: { model: "opus-4.7" },
              model_info: { id: "family-only", mode: "chat" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]).toMatchObject({
      id: route,
      name: route,
      reasoning: true,
      input: ["text", "image"],
      contextWindow: 1_000_000,
      maxTokens: 128_000,
      cost: { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
    });
    expect(stderr).not.toHaveBeenCalled();
  });

  it("withholds catalog authority for genuinely different Anthropic models", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const route = `anthropic-model-conflict-${process.pid}-${Date.now()}-${Math.random()}`;
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: route,
              litellm_params: { model: "anthropic/claude-opus-4-7" },
              model_info: { id: "opus", mode: "chat" },
            },
            {
              model_name: route,
              litellm_params: { model: "anthropic/claude-sonnet-4-6" },
              model_info: { id: "sonnet", mode: "chat" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]).toMatchObject({
      id: route,
      name: `${route} (incomplete metadata)`,
      reasoning: false,
      input: ["text"],
      contextWindow: 128_000,
      maxTokens: 16_384,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    });
    expect(stderr).toHaveBeenCalledTimes(1);
    expect(String(stderr.mock.calls[0]?.[0])).toContain("missing or conflicting deployment provider evidence");
    expect(String(stderr.mock.calls[0]?.[0])).toContain(route);
  });

  it("withholds different concrete catalog identities from one provider across a route group", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const route = `same-provider-group-conflict-${process.pid}-${Date.now()}-${Math.random()}`;
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: route,
              litellm_params: { model: "openai/gpt-4o" },
              model_info: { id: "gpt-4o", mode: "chat", litellm_provider: "openai" },
            },
            {
              model_name: route,
              litellm_params: { model: "openai/gpt-4.1" },
              model_info: { id: "gpt-4.1", mode: "chat", litellm_provider: "openai" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]).toMatchObject({
      id: route,
      name: `${route} (incomplete metadata)`,
      reasoning: false,
      input: ["text"],
      contextWindow: 128_000,
      maxTokens: 16_384,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    });
    expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
    const diagnostics = stderr.mock.calls.map(([message]) => String(message));
    expect(diagnostics).toHaveLength(1);
    expect(diagnostics[0]).toContain("missing or conflicting deployment provider evidence");
    expect(diagnostics[0]).toContain(route);
  });

  it("preserves Bedrock catalog authority", () => {
    expect(
      resolveModelInfoCatalog({
        model_name: "bedrock-claude-route",
        litellm_params: { model: "bedrock/anthropic.claude-sonnet-4-6" },
        model_info: { mode: "chat", litellm_provider: "bedrock" },
      }),
    ).toMatchObject({ provider: "bedrock" });
  });

  it("withholds a cross-host Claude route spanning Vertex and Bedrock", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const route = `cross-host-claude-${process.pid}-${Date.now()}-${Math.random()}`;
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: route,
              litellm_params: { model: "vertex_ai/claude-sonnet-4@20250514" },
              model_info: {
                id: "vertex",
                mode: "chat",
                litellm_provider: "vertex_ai",
                supports_reasoning: true,
                supports_vision: true,
                max_input_tokens: 96_000,
                max_output_tokens: 8_000,
              },
            },
            {
              model_name: route,
              litellm_params: { model: "bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0" },
              model_info: {
                id: "bedrock",
                mode: "chat",
                litellm_provider: "bedrock",
                supports_reasoning: false,
                supports_vision: false,
                max_input_tokens: 64_000,
                max_output_tokens: 4_000,
              },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models).toEqual([
      expect.objectContaining({
        id: route,
        name: `${route} (incomplete metadata)`,
        reasoning: false,
        input: ["text"],
        contextWindow: 64_000,
        maxTokens: 4_000,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      }),
    ]);
    expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
    expect(stderr).toHaveBeenCalledTimes(1);
    expect(String(stderr.mock.calls[0]?.[0])).toContain("missing or conflicting deployment provider evidence");
    expect(String(stderr.mock.calls[0]?.[0])).toContain(route);
  });

  it("does not enrich an unqualified OpenAI public route without backend identity", async () => {
    mockEndpoints({
      "/model/info": () => jsonResponse(200, { data: [{ model_name: "gpt-4o", model_info: { mode: "chat" } }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]).toMatchObject({
      id: "gpt-4o",
      name: "gpt-4o (incomplete metadata)",
      reasoning: false,
      input: ["text"],
      contextWindow: 128_000,
      maxTokens: 16_384,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    });
  });

  it("keeps proven display prices while marking any unresolved cost field incomplete", async () => {
    // Display cost reduces per field. Proven input/output survive, only unresolved
    // fields fall to zero, and the model is still marked incomplete so unknown
    // cache pricing is never presented as complete or free. A model must not lose
    // the marker just because input and output happen to be known.
    const priced = (id: string, input: number) => ({
      model_name: "priced-without-cache-rates",
      litellm_params: { model: "internal/unknown" },
      model_info: {
        id,
        mode: "chat",
        max_input_tokens: 32_000,
        max_output_tokens: 4_000,
        input_cost_per_token: input,
        output_cost_per_token: 0.000015,
      },
    });

    for (const [data, expectedInput] of [
      [[priced("only", 0.000003)], 3],
      [[priced("a", 0.000003), priced("b", 0.000004)], 4],
    ] as const) {
      mockEndpoints({ "/model/info": () => jsonResponse(200, { data }) });

      const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

      expect(result.models).toHaveLength(1);
      expect(result.models[0]).toMatchObject({
        id: "priced-without-cache-rates",
        name: "priced-without-cache-rates (incomplete metadata)",
        cost: { input: expectedInput, output: 15, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 32_000,
        maxTokens: 4_000,
      });
    }
  });

  it("marks an ambiguous group incomplete despite complete router pricing and reports withheld authority", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const route = `priced-ambiguous-${process.pid}-${Date.now()}-${Math.random()}`;
    const priced = (id: string, model: string) => ({
      model_name: route,
      litellm_params: { model },
      model_info: {
        id,
        mode: "chat",
        input_cost_per_token: 0.000003,
        output_cost_per_token: 0.000015,
        cache_read_input_token_cost: 0.0000003,
        cache_creation_input_token_cost: 0.00000375,
      },
    });
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [priced("openai", "openai/gpt-4o"), priced("anthropic", "anthropic/claude-sonnet-4-6")],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]).toMatchObject({
      id: route,
      name: `${route} (incomplete metadata)`,
      cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
      contextWindow: 128_000,
      maxTokens: 16_384,
    });
    const diagnostics = stderr.mock.calls.map(([message]) => String(message));
    expect(diagnostics).toHaveLength(1);
    expect(diagnostics[0]).toContain(route);
    expect(diagnostics[0]).toContain("catalog limits, pricing, and reasoning metadata are withheld");
  });

  it("uses Responses compatibility metadata for Responses-mode routes", async () => {
    // Compat is derived from the model id independently of transport, so a
    // Responses-mode Moonshot route retains the complete Kimi repair object.
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [{ model_name: "moonshot/kimi-k2.6", model_info: { id: "one", mode: "responses" } }],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]).toMatchObject({ api: "openai-responses" });
    expect(result.models[0]?.compat).toEqual({
      supportsDeveloperRole: false,
      supportsStrictMode: false,
    });
  });

  it("uses native prompt caching for a Responses-mode alias", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [{ model_name: "anthropic/claude-sonnet-4-6", model_info: { id: "one", mode: "responses" } }],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]).toMatchObject({ api: "openai-responses" });
    expect(result.models[0]?.compat).toBeUndefined();
  });

  it.each([
    ["a numeric mode", { model_name: "bad-mode", model_info: { id: "a", mode: 7 } }],
    ["a numeric deployment id", { model_name: "bad-id", model_info: { id: 7, mode: "chat" } }],
    // Every untrusted string read from deployment metadata must tolerate YAML
    // values such as `model_name: 4.1` parsing as numbers.
    ["a numeric route name", { model_name: 4.1, model_info: { id: "a", mode: "chat" } }],
    ["a numeric adapter", { model_name: "bad-adapter", model_info: { id: "a", mode: "chat", litellm_provider: 7 } }],
    [
      "a numeric backend model",
      { model_name: "bad-backend", litellm_params: { model: 7 }, model_info: { id: "a", mode: "chat" } },
    ],
    ["a numeric base model", { model_name: "bad-base", model_info: { id: "a", mode: "chat", base_model: 7 } }],
  ])("withholds a row with %s instead of failing the whole discovery", async (_case, bad) => {
    // One operator typo in proxy config must not cost every other model.
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            bad,
            {
              model_name: "healthy-route",
              litellm_params: { model: "openai/gpt-4o" },
              model_info: { id: "b", mode: "chat" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models.map((model) => model.id)).toContain("healthy-route");
    expect(result.models.find((model) => model.id === "healthy-route")?.cost.input).toBeGreaterThan(0);
  });

  it("withholds a fallback or health entry whose id is not a string", async () => {
    // The same invariant on the two paths that build a model straight from an id.
    mockEndpoints({
      "/model/info": () => jsonResponse(403, {}),
      "/v1/models": () => jsonResponse(200, { data: [{ id: 7 }, { id: "gpt-4o", owned_by: 7 }] }),
    });
    const fallback = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });
    expect(fallback.models.map((model) => model.id)).toEqual(["gpt-4o"]);

    mockEndpoints({
      "/model/info": () => jsonResponse(403, {}),
      "/v1/models": () => jsonResponse(404, {}),
      "/health": () => jsonResponse(200, { healthy_endpoints: [{ model: 7 }, { model: "anthropic/claude-opus-4-7" }] }),
    });
    const health = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });
    expect(health.models.map((model) => model.id)).toEqual(["anthropic/claude-opus-4-7"]);
  });

  it("falls back to the health route name without granting deployment thinking levels", async () => {
    // The detail row's `model_name` is unusable, but `/health` named the route, so the
    // model must survive under that name without treating the route as level authority.
    mockEndpoints({
      "/model/info?litellm_model_id=uuid-1": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: 7,
              model_info: {
                mode: "chat",
                supports_reasoning: true,
                supports_low_reasoning_effort: true,
              },
            },
          ],
        }),
      "/model/info": () => jsonResponse(403, {}),
      "/v1/models": () => jsonResponse(404, {}),
      "/health": () => jsonResponse(200, { healthy_endpoints: [{ model: "named-route", model_id: "uuid-1" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.source).toBe("health");
    expect(result.models.map((model) => model.id)).toEqual(["named-route"]);
    expect(result.models[0]?.thinkingLevelMap).toEqual(NO_REASONING_LEVELS);
  });

  it("withholds a health deployment whose route name is not a string", async () => {
    // This path bypasses the grouping loop: `/health` supplies the route name
    // directly to the reducer, so it needs its own guard.
    mockEndpoints({
      "/model/info?litellm_model_id=uuid-1": () => jsonResponse(200, { data: [{ model_info: { mode: "chat" } }] }),
      "/model/info": () => jsonResponse(403, {}),
      "/v1/models": () => jsonResponse(404, {}),
      "/health": () =>
        jsonResponse(200, {
          healthy_endpoints: [{ model: 7, model_id: "uuid-1" }, { model: "anthropic/claude-opus-4-7" }],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models.map((model) => model.id)).toEqual(["anthropic/claude-opus-4-7"]);
  });

  it("survives a deeply nested deployment row", async () => {
    // Canonicalization is depth-bounded, so a pathological payload cannot exhaust
    // the stack and take every model down with it. Build the JSON text iteratively
    // so the fixture itself does not depend on JSON.stringify's recursion limit.
    const depth = 20_000;
    const nestedJson = `${'{"nested":'.repeat(depth)}"leaf"${"}".repeat(depth)}`;
    const payload =
      '{"data":[{"model_name":"deep-route","litellm_params":{"model":"openai/gpt-4o"},' +
      `"model_info":{"id":"a","mode":"chat","extra":${nestedJson}}}]}`;
    mockEndpoints({
      "/model/info": () => new Response(payload, { status: 200, headers: { "content-type": "application/json" } }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.id).toBe("deep-route");
  });

  it("never emits the fallback-only sentinel for a reduced deployment group", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    // The ` (no metadata)` suffix authorizes catalog re-derivation from the model
    // id during offline cache reads, so no `/model/info` group may carry it.
    const groups = [
      [{ model_name: "singleton-route", model_info: { id: "one", mode: "chat" } }],
      [
        { model_name: "plural-route", model_info: { id: "a", mode: "chat" } },
        { model_name: "plural-route", model_info: { id: "b", mode: "chat" }, litellm_params: { model: "x/y" } },
      ],
      [
        { model_name: "conflicting-route", model_info: { id: "same", mode: "chat" } },
        { model_name: "conflicting-route", model_info: { id: "same", mode: "chat", max_input_tokens: 8_000 } },
      ],
      [
        { model_name: "embedding-sibling-route", model_info: { id: "chat", mode: "chat" } },
        { model_name: "embedding-sibling-route", model_info: { id: "embed", mode: "embedding" } },
      ],
    ];

    for (const data of groups) {
      stderr.mockClear();
      mockEndpoints({ "/model/info": () => jsonResponse(200, { data }) });
      const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

      if (data[0]?.model_name === "embedding-sibling-route") {
        expect(result.models).toEqual([]);
        expect(stderr).toHaveBeenCalledTimes(1);
      } else {
        expect(result.models).toHaveLength(1);
        expect(result.models[0]?.name).not.toContain(" (no metadata)");
        expect(result.models[0]?.name).toBe(`${result.models[0]?.id} (incomplete metadata)`);
      }
    }
  });

  it("withholds a group whose duplicate deployment id carries conflicting backends", async () => {
    // One deployment id with two disagreeing backends is not a single deployment,
    // so its catalog authority is ambiguous and cannot enrich the group.
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            { model_name: "openai/gpt-5.5", model_info: { id: "same", mode: "chat" } },
            {
              model_name: "openai/gpt-5.5",
              model_info: { id: "same", mode: "chat", max_input_tokens: 64_000 },
              litellm_params: { model: "internal/mystery" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]).toMatchObject({
      id: "openai/gpt-5.5",
      name: "openai/gpt-5.5 (incomplete metadata)",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      maxTokens: 16_384,
    });
    expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
  });

  it("reports conflicting deployment provider identity once with bounded detail", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const nonce = `${process.pid}-${Date.now()}-${Math.random()}`;
    const routes = Array.from({ length: 4 }, (_, index) => `diagnostic-${nonce}-${index + 1}`);
    const conflicting = (route: string) => [
      { model_name: route, model_info: { id: `${route}-a`, mode: "chat" }, litellm_params: { model: "openai/gpt-4o" } },
      {
        model_name: route,
        model_info: { id: `${route}-b`, mode: "chat" },
        litellm_params: { model: "anthropic/claude-sonnet-4-6" },
      },
    ];
    mockEndpoints({
      "/model/info": () => jsonResponse(200, { data: routes.flatMap(conflicting) }),
    });

    await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    const diagnostics = stderr.mock.calls.map(([message]) => String(message));
    expect(diagnostics).toHaveLength(1);
    expect(diagnostics[0]).toContain("4 route group(s) have missing or conflicting deployment provider evidence");
    // Bounded: a count, at most three route ids, and no deployment ids or params.
    expect(diagnostics[0]).toContain(`${routes[0]}, ${routes[1]}, ${routes[2]} (+1 more)`);
    expect(diagnostics[0]).not.toContain(routes[3]);
    expect(diagnostics[0]).not.toContain(`${routes[0]}-a`);
  });

  it("reports each ambiguous route once per process, not once per discovery", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const nonce = `${process.pid}-${Date.now()}-${Math.random()}`;
    const routes = [`once-${nonce}-a`, `once-${nonce}-b`, `once-${nonce}-c`];
    const conflicting = (route: string) => [
      { model_name: route, model_info: { id: `${route}-a`, mode: "chat" }, litellm_params: { model: "openai/gpt-4o" } },
      {
        model_name: route,
        model_info: { id: `${route}-b`, mode: "chat" },
        litellm_params: { model: "anthropic/claude-sonnet-4-6" },
      },
    ];
    const discover = async (routes: string[]) => {
      mockEndpoints({ "/model/info": () => jsonResponse(200, { data: routes.flatMap(conflicting) }) });
      await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });
    };

    await discover(routes.slice(0, 2));
    expect(stderr.mock.calls).toHaveLength(1);
    expect(String(stderr.mock.calls[0]?.[0])).toContain("2 route group(s)");

    // A background refresh of the same misconfiguration must not repeat itself.
    await discover(routes.slice(0, 2));
    expect(stderr.mock.calls).toHaveLength(1);

    // A newly ambiguous route is still worth reporting, and only that one.
    await discover(routes);
    expect(stderr.mock.calls).toHaveLength(2);
    const second = String(stderr.mock.calls[1]?.[0]);
    expect(second).toContain("1 route group(s)");
    expect(second).toContain(routes[2]);
    expect(second).not.toContain(routes[0]);
  });

  it("reports a route whose deployments supply partial provider evidence", async () => {
    // Withholding also happens when one deployment resolves a provider and another
    // supplies none, so the wording must not claim a conflict is the only cause.
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const route = `partial-evidence-${process.pid}-${Date.now()}-${Math.random()}`;
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: route,
              model_info: { id: "a", mode: "chat" },
              litellm_params: { model: "openai/gpt-4o" },
            },
            { model_name: route, model_info: { id: "b", mode: "chat" } },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.name).toBe(`${route} (incomplete metadata)`);
    const diagnostics = stderr.mock.calls.map(([message]) => String(message));
    expect(diagnostics).toHaveLength(1);
    expect(diagnostics[0]).toContain("missing or conflicting");
    expect(diagnostics[0]).toContain(route);
  });

  it("withholds mixed-mode routes and reports one bounded diagnostic", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const nonce = `${process.pid}-${Date.now()}-${Math.random()}`;
    const routes = Array.from({ length: 4 }, (_, index) => `mixed-mode-${nonce}-${index + 1}`);
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: routes.flatMap((route) => [
            {
              model_name: route,
              litellm_params: { model: "openai/gpt-5.5" },
              model_info: { id: `${route}-chat`, mode: "chat" },
            },
            { model_name: route, model_info: { id: `${route}-embed`, mode: "embedding" } },
          ]),
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models).toEqual([]);
    expect(stderr).toHaveBeenCalledTimes(1);
    const diagnostic = String(stderr.mock.calls[0]?.[0]);
    expect(diagnostic).toContain("4 route group(s) mix chat-style and explicitly incompatible deployment modes");
    expect(diagnostic).toContain(`${routes[0]}, ${routes[1]}, ${routes[2]} (+1 more)`);
    expect(diagnostic).not.toContain(routes[3]);
  });

  it("reports groups whose resolved fallback identities disagree", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            { model_name: "agreed", model_info: { id: "a", mode: "chat" }, litellm_params: { model: "openai/gpt-4o" } },
            { model_name: "agreed", model_info: { id: "b", mode: "chat" }, litellm_params: { model: "openai/gpt-4o" } },
            { model_name: "unknown", model_info: { id: "c", mode: "chat" }, litellm_params: { model: "internal/x" } },
            { model_name: "unknown", model_info: { id: "d", mode: "chat" }, litellm_params: { model: "internal/y" } },
          ],
        }),
    });

    await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(stderr).toHaveBeenCalledTimes(1);
    expect(String(stderr.mock.calls[0]?.[0])).toContain("unknown");
  });

  it("keeps Kimi compatibility on non-Moonshot routes", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "kimi-k3",
            litellm_params: { model: "azure_ai/FW-Kimi-K3" },
            model_info: { mode: "chat" },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.compat).toMatchObject({
      ...buildCompat("kimi-k3"),
      requiresReasoningContentOnAssistantMessages: true,
    });
  });

  it.each([
    [
      "Azure-hosted Kimi",
      {
        model_name: "kimi-k3",
        litellm_params: { model: "azure/FW-Kimi-K3" },
        model_info: {
          mode: "chat",
          base_model: "fireworks/accounts/fireworks/models/kimi-k3",
          litellm_provider: "azure",
        },
      },
      false,
    ],
    [
      "Bedrock-hosted Kimi",
      {
        model_name: "moonshotai-kimi-k2-5",
        litellm_params: { model: "bedrock/moonshotai.kimi-k2.5" },
        model_info: {
          mode: "chat",
          base_model: "moonshotai.kimi-k2.5",
          litellm_provider: "bedrock_converse",
        },
      },
      false,
    ],
    [
      "opaque Moonshot alias",
      {
        model_name: "k3-prod",
        litellm_params: { model: "moonshot/kimi-k2.5" },
        model_info: { mode: "chat" },
      },
      true,
    ],
  ] as const)("derives reasoning visibility from Moonshot routing evidence for %s", async (_name, entry, suppress) => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { data: [entry] }));

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.litellmPolicy?.suppressReasoningVisibility).toBe(suppress);
    expect(result.models[0]?.litellmPolicy?.normalizeThinkTags).toBe(true);
  });

  it.each([
    ["Moonshot deployments", ["moonshot/kimi-k3", "moonshot/kimi-k3"], true, true],
    ["mixed deployments", ["moonshot/kimi-k3", "azure_ai/FW-Kimi-K3"], false, true],
    ["reversed mixed deployments", ["azure_ai/FW-Kimi-K3", "moonshot/kimi-k3"], false, true],
    ["incomplete deployment metadata", ["moonshot/kimi-k3", undefined], false, false],
  ] as const)("requires Moonshot transport for suppression across %s", async (_name, routes, suppress, normalize) => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: routes.map((model) => ({
          model_name: "kimi-prod",
          ...(model ? { litellm_params: { model } } : {}),
          model_info: { mode: "chat" },
        })),
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.litellmPolicy?.suppressReasoningVisibility).toBe(suppress);
    expect(result.models[0]?.litellmPolicy?.normalizeThinkTags).toBe(normalize);
  });

  it.each([
    [
      "withholds a non-Moonshot incompatible sibling",
      [
        { model: "moonshot/kimi-k3", mode: "chat" },
        { model: "openai/text-embedding-3-small", mode: "embedding" },
      ],
    ],
    [
      "withholds a Moonshot incompatible sibling",
      [
        { model: "azure_ai/FW-Kimi-K3", mode: "chat" },
        { model: "moonshot/kimi-k3", mode: "embedding" },
      ],
    ],
  ] as const)("%s when reducing suppression", async (_name, deployments) => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const route = `kimi-incompatible-${process.pid}-${Date.now()}-${Math.random()}`;
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: deployments.map(({ model, mode }, index) => ({
          model_name: route,
          litellm_params: { model },
          model_info: { id: `deployment-${index}`, mode },
        })),
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models).toEqual([]);
    expect(stderr).toHaveBeenCalledTimes(1);
  });

  it.each([
    [
      "Azure-hosted Kimi",
      {
        model_name: "kimi-k3",
        litellm_params: { model: "azure/FW-Kimi-K3" },
        model_info: {
          mode: "chat",
          base_model: "fireworks/accounts/fireworks/models/kimi-k3",
          litellm_provider: "azure",
        },
      },
      false,
    ],
    [
      "Bedrock-hosted Kimi",
      {
        model_name: "moonshotai-kimi-k2-5",
        litellm_params: { model: "bedrock/moonshotai.kimi-k2.5" },
        model_info: {
          mode: "chat",
          base_model: "moonshotai.kimi-k2.5",
          litellm_provider: "bedrock_converse",
        },
      },
      false,
    ],
    [
      "Moonshot-hosted opaque alias",
      {
        model_name: "k3-prod",
        litellm_params: { model: "moonshot/kimi-k2.5" },
        model_info: { mode: "chat" },
      },
      true,
    ],
  ] as const)(
    "keeps think-tag normalization independent of hosting transport for %s",
    async (_name, entry, suppress) => {
      vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { data: [entry] }));

      const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

      expect(result.models[0]?.litellmPolicy?.suppressReasoningVisibility).toBe(suppress);
      expect(result.models[0]?.litellmPolicy?.normalizeThinkTags).toBe(true);
    },
  );

  it("does not suppress an alias routed to a forced-thinking Moonshot model", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "k3-prod",
            litellm_params: { model: "moonshot/kimi-k2-thinking" },
            model_info: { mode: "chat" },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.litellmPolicy?.suppressReasoningVisibility === true).toBe(false);
  });

  it.each([
    ["provider-only metadata", { custom_llm_provider: "moonshot" }, true],
    [
      "Moonshot provider and Azure-hosted Kimi backend",
      { custom_llm_provider: "moonshot", model: "azure_ai/FW-Kimi-K3" },
      false,
    ],
  ] as const)("checks every declared Moonshot routing signal with %s", async (_name, litellm_params, suppress) => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [{ model_name: "kimi-prod", litellm_params, model_info: { mode: "chat" } }],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.litellmPolicy?.suppressReasoningVisibility).toBe(suppress);
    expect(result.models[0]?.litellmPolicy?.normalizeThinkTags).toBe(true);
  });

  it("keeps route evidence isolated between discoveries", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      const model = url.startsWith("https://moonshot.example.com") ? "moonshot/kimi-k3" : "azure/gpt-4o";
      return jsonResponse(200, {
        data: [
          {
            model_name: "kimi-prod",
            litellm_params: { model },
            model_info: { mode: "chat" },
          },
        ],
      });
    });

    const moonshot = await discoverModels("https://moonshot.example.com", "sk-test", { modelsDev: false });
    const azure = await discoverModels("https://azure.example.com", "sk-test", { modelsDev: false });

    expect(moonshot.models[0]?.litellmPolicy?.suppressReasoningVisibility).toBe(true);
    expect(azure.models[0]?.litellmPolicy?.suppressReasoningVisibility === true).toBe(false);
  });

  it("does not reuse route evidence after metadata fallback", async () => {
    let fallback = false;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return fallback
          ? jsonResponse(403, {})
          : jsonResponse(200, {
              data: [
                {
                  model_name: "kimi-prod",
                  litellm_params: { model: "moonshot/kimi-k3" },
                  model_info: { mode: "chat" },
                },
              ],
            });
      }
      if (url.endsWith("/v1/models")) return jsonResponse(200, { data: [{ id: "kimi-prod" }] });
      throw new Error(`unexpected URL: ${url}`);
    });

    await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });
    fallback = true;
    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.litellmPolicy?.suppressReasoningVisibility === true).toBe(false);
  });
  it("uses models.dev reasoning options for an Azure GPT-5 deployment", async () => {
    vi.resetModules();
    const { discoverModels: isolatedDiscoverModels } = await import("../src/discover.js");
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "azure-gpt-5",
              litellm_params: { model: "azure/gpt-5", allowed_openai_params: ["reasoning_effort"] },
              model_info: { id: "one", mode: "chat", litellm_provider: "azure", supports_reasoning: true },
            },
          ],
        }),
      "models.dev/api.json": () =>
        jsonResponse(200, {
          azure: {
            models: {
              "gpt-5": { reasoning_options: { type: "effort", values: ["minimal", "low", "medium", "high"] } },
              "gpt-5-adapter": { reasoning_options: { type: "effort", values: ["low", "high"] } },
            },
          },
        }),
    });

    const result = await isolatedDiscoverModels("https://litellm.example.com", "sk-test", {
      modelsDevCachePath: join(await mkdtemp(join(agentDir, "public-efforts-")), "models-dev.json"),
    });

    expect(result.models[0]?.thinkingLevelMap).toEqual({
      off: null,
      minimal: "minimal",
      low: "low",
      medium: "medium",
      high: "high",
      xhigh: null,
      max: null,
    });
  });

  it("uses the adapter as the public-catalog lookup provider when backend identity has none", async () => {
    vi.resetModules();
    const { discoverModels: isolatedDiscoverModels } = await import("../src/discover.js");
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "azure-gpt-5",
              litellm_params: { model: "gpt-5-adapter", allowed_openai_params: ["reasoning_effort"] },
              model_info: { id: "one", mode: "chat", litellm_provider: "azure", supports_reasoning: true },
            },
          ],
        }),
      "models.dev/api.json": () =>
        jsonResponse(200, {
          azure: {
            models: { "gpt-5-adapter": { reasoning_options: { type: "effort", values: ["low", "high"] } } },
          },
        }),
    });

    const result = await isolatedDiscoverModels("https://litellm.example.com", "sk-test", {
      modelsDevCachePath: join(await mkdtemp(join(agentDir, "public-efforts-")), "models-dev.json"),
    });

    expect(result.models[0]?.thinkingLevelMap).toEqual({
      off: null,
      minimal: null,
      low: "low",
      medium: null,
      high: "high",
      xhigh: null,
      max: null,
    });
  });

  // models.dev serves ChatGPT routes from OpenAI's key, but a field it omits must
  // still come from the subscription's Codex catalog, not OpenAI API pricing.
  it("keeps Codex catalog pricing for a partial models.dev ChatGPT record", async () => {
    vi.resetModules();
    const { discoverModels: isolatedDiscoverModels } = await import("../src/discover.js");
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "gpt-5.6-sol",
              litellm_params: { model: "chatgpt/gpt-5.6-sol" },
              model_info: {
                mode: "responses",
                litellm_provider: "chatgpt",
                supported_openai_params: ["reasoning_effort"],
              },
            },
          ],
        }),
      "models.dev/api.json": () =>
        jsonResponse(200, {
          openai: {
            models: { "gpt-5.6-sol": { reasoning_options: [{ type: "effort", values: ["none", "low", "high"] }] } },
          },
        }),
    });

    const result = await isolatedDiscoverModels("https://litellm.example.com", "sk-test", {
      modelsDevCachePath: join(await mkdtemp(join(agentDir, "public-efforts-")), "models-dev.json"),
    });

    expect(result.models[0]?.cost).toMatchObject({ input: 5, output: 30, cacheRead: 0.5, cacheWrite: 6.25 });
  });

  it.each(["azure", "azure_ai"])(
    "uses custom %s authority for public reasoning efforts over a generic adapter",
    async (customProvider) => {
      vi.resetModules();
      const { discoverModels: isolatedDiscoverModels } = await import("../src/discover.js");
      mockEndpoints({
        "/model/info": () =>
          jsonResponse(200, {
            data: [
              {
                model_name: "custom-azure-route",
                litellm_params: {
                  model: "gpt-5-adapter",
                  custom_llm_provider: customProvider,
                  allowed_openai_params: ["reasoning_effort"],
                },
                model_info: { mode: "chat", litellm_provider: "openai", supports_reasoning: true },
              },
            ],
          }),
        "models.dev/api.json": () =>
          jsonResponse(200, {
            azure: { models: { "gpt-5-adapter": { reasoning_options: { type: "effort", values: ["low", "high"] } } } },
            openai: {
              models: { "gpt-5-adapter": { reasoning_options: { type: "effort", values: ["minimal", "high"] } } },
            },
          }),
      });

      const result = await isolatedDiscoverModels("https://litellm.example.com", "sk-test", {
        modelsDevCachePath: join(await mkdtemp(join(agentDir, "public-efforts-")), "models-dev.json"),
      });

      expect(result.models[0]).toMatchObject({
        api: "openai-completions",
        thinkingLevelMap: { off: null, minimal: null, low: "low", medium: null, high: "high", xhigh: null, max: null },
      });
    },
  );

  it("ignores malformed accepted-parameter arrays without dropping healthy models", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "malformed",
              litellm_params: { model: "internal/malformed", allowed_openai_params: 7 },
              model_info: { id: "bad", mode: "chat", supported_openai_params: {} },
            },
            {
              model_name: "healthy-route",
              litellm_params: { model: "openai/gpt-4o" },
              model_info: { id: "good", mode: "chat" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models.map((model) => model.id)).toContain("healthy-route");
  });

  it("uses custom_llm_provider authority for catalog metadata and the Chat carrier", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "private/kimi-route",
            litellm_params: {
              model: "kimi-k2.6",
              custom_llm_provider: "moonshot",
              allowed_openai_params: ["thinking"],
            },
            model_info: { mode: "chat", supports_reasoning: true },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      id: "private/kimi-route",
      name: "private/kimi-route",
      input: ["text", "image"],
      cost: { input: 0.95, output: 4, cacheRead: 0.16, cacheWrite: 0 },
      contextWindow: 262_144,
      maxTokens: 262_144,
      reasoning: true,
      thinkingLevelMap: {
        off: "off",
        minimal: null,
        low: null,
        medium: null,
        high: "high",
        xhigh: null,
        max: null,
      },
      compat: { thinkingFormat: "deepseek", supportsReasoningEffort: false },
    });
  });

  it("prefers custom_llm_provider over a generic adapter for catalog lookup", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "kimi-route",
            litellm_params: { model: "kimi-k2.5", custom_llm_provider: "moonshot" },
            model_info: { litellm_provider: "openai", mode: "chat" },
          },
          {
            model_name: "moonshot-control",
            litellm_params: { model: "kimi-k2.5" },
            model_info: { litellm_provider: "moonshot", mode: "chat" },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    for (const id of ["kimi-route", "moonshot-control"]) {
      expect(result.models.find((model) => model.id === id)).toMatchObject({
        id,
        name: id,
        contextWindow: 262_144,
        maxTokens: 262_144,
        cost: { input: 0.6, output: 3, cacheRead: 0.1, cacheWrite: 0 },
      });
    }
  });

  it("derives Gemini normalization from adapter evidence for an opaque route", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "internal/prod-route",
              litellm_params: { model: "internal/opaque" },
              model_info: { id: "a", mode: "chat", litellm_provider: "gemini" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.litellmPolicy?.normalizeGeminiReasoningEffort).toBe(true);
  });

  it("withholds Gemini normalization when deployment family evidence is mixed", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "gemini-looking-route",
              litellm_params: { model: "gemini/gemini-3.1-pro-preview" },
              model_info: { id: "a", mode: "chat", litellm_provider: "gemini" },
            },
            {
              model_name: "gemini-looking-route",
              litellm_params: { model: "openai/gpt-4o" },
              model_info: { id: "b", mode: "chat", litellm_provider: "openai" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.litellmPolicy).toBeUndefined();
  });

  it("does not use a qualified singleton public route as catalog authority", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [{ model_name: "openai/gpt-5.5", model_info: { id: "one", mode: "chat" } }],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      id: "openai/gpt-5.5",
      name: "openai/gpt-5.5 (incomplete metadata)",
      reasoning: false,
      input: ["text"],
      contextWindow: 128_000,
      maxTokens: 16_384,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    });
    expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
  });

  it.each([
    ["opaque backend model", { litellm_params: { model: "internal/mystery" }, model_info: {} }],
    ["opaque base model", { model_info: { base_model: "internal/mystery" } }],
    ["unresolved adapter", { model_info: { litellm_provider: "custom_proxy" } }],
  ])("withholds route-text catalog metadata for a singleton with %s evidence", async (_case, evidence) => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "openai/gpt-5.5",
              ...evidence,
              model_info: { ...evidence.model_info, id: "only", mode: "chat" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      id: "openai/gpt-5.5",
      name: "openai/gpt-5.5 (incomplete metadata)",
      reasoning: false,
      input: ["text"],
      contextWindow: 128_000,
      maxTokens: 16_384,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    });
    expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
  });

  it.each(["openai", "custom_openai", "openai_like", "text-completion-openai", "azure", "azure_ai"])(
    "keeps Kimi generation controls on the %s transport",
    async (adapter) => {
      mockEndpoints({
        "/model/info": () =>
          jsonResponse(200, {
            data: [
              {
                model_name: "kimi-route",
                litellm_params: { model: "openai/kimi-k2.5", allowed_openai_params: ["thinking"] },
                model_info: { mode: "chat", litellm_provider: adapter, supports_reasoning: true },
              },
            ],
          }),
      });
      const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });
      expect(result.models[0]).toMatchObject({
        compat: { thinkingFormat: "deepseek", supportsReasoningEffort: false },
        litellmPolicy: {
          normalizeStrictToolMessages: true,
          normalizeThinkTags: true,
          suppressReasoningVisibility: false,
        },
      });
    },
  );

  it.each(["kimi-proxy/gpt-4-turbo", "kimi-k2.5-proxy/gpt-4-turbo"])(
    "ignores family and generation words in unknown backend provider %s",
    async (model) => {
      mockEndpoints({
        "/model/info": () =>
          jsonResponse(200, {
            data: [
              {
                model_name: "opaque",
                litellm_params: { model },
                model_info: {
                  mode: "chat",
                  supported_endpoints: ["/v1/chat/completions"],
                  supports_reasoning: true,
                  supported_openai_params: ["thinking"],
                },
              },
            ],
          }),
      });

      const { models } = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

      expect(models).toHaveLength(1);
      expect(models[0]).toMatchObject({ api: "openai-completions", litellmBackendFamily: "openai" });
      expect(models[0]?.compat).toEqual({ supportsStore: false });
      expect(models[0]?.thinkingLevelMap).toEqual(NO_REASONING_LEVELS);
      expect(models[0]).not.toHaveProperty("litellmPolicy");
    },
  );

  it("publishes Kimi compatibility without Moonshot request parameters through an OpenAI transport", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "kimi-through-openai",
              litellm_params: { model: "openai/kimi-k2.5", allowed_openai_params: ["thinking"] },
              model_info: { id: "kimi-openai", mode: "chat", litellm_provider: "openai", supports_reasoning: true },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      id: "kimi-through-openai",
      reasoning: true,
      compat: {
        supportsStore: false,
        supportsDeveloperRole: false,
        supportsReasoningEffort: false,
        supportsStrictMode: false,
        maxTokensField: "max_tokens",
      },
      litellmPolicy: {
        normalizeStrictToolMessages: true,
        normalizeThinkTags: true,
        suppressReasoningVisibility: false,
      },
    });
  });

  it("applies Kimi policy from a Moonshot custom provider with an opaque model", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "opaque-kimi-route",
              litellm_params: { model: "opaque-model", custom_llm_provider: "moonshot" },
              model_info: { id: "kimi-custom-provider", mode: "chat" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.litellmPolicy).toEqual({
      normalizeStrictToolMessages: true,
      normalizeThinkTags: true,
      suppressReasoningVisibility: true,
    });
  });

  it.each(["openai", "custom_openai", "openai_like", "text-completion-openai", "azure", "azure_ai"])(
    "keeps unknown catalog metadata unresolved through the %s adapter",
    (adapter) => {
      expect(
        resolveModelInfoCatalog({
          model_name: "opaque-route",
          litellm_params: { model: "internal/model" },
          model_info: { mode: "chat", litellm_provider: adapter },
        }),
      ).toMatchObject({ provider: "internal", catalogModelId: "internal/model" });
    },
  );

  it("keeps provider identity from the backend candidate that resolves", () => {
    expect(
      resolveModelInfoCatalog({
        model_name: "mixed-evidence",
        litellm_params: { model: "internal/claude-magic" },
        model_info: { mode: "chat", base_model: "openai/gpt-4o" },
      }),
    ).toMatchObject({ provider: "openai" });

    expect(
      resolveModelInfoCatalog({
        model_name: "aliased",
        litellm_params: { model: "anthropic/opus-4-7" },
        model_info: { mode: "chat" },
      }),
    ).toMatchObject({ provider: "anthropic" });
  });

  it.each([
    {
      name: "routing model conflicts with base model",
      entry: {
        model_name: "conflicting-models",
        litellm_params: { model: "openai/gpt-4o" },
        model_info: { mode: "chat", base_model: "anthropic/claude-sonnet-4-6" },
      },
    },
    {
      name: "adapter conflicts with the qualified model",
      entry: {
        model_name: "conflicting-adapter",
        litellm_params: { model: "openai/gpt-4o" },
        model_info: { mode: "chat", litellm_provider: "anthropic" },
      },
    },
  ])("retains catalog authority independently of family disagreement for $name", ({ entry }) => {
    expect(resolveModelInfoCatalog(entry)).toMatchObject({
      provider: entry.model_info.base_model ? "anthropic" : "openai",
    });
  });

  it("retains the qualified catalog identity despite a different adapter family", () => {
    expect(
      resolveModelInfoCatalog({
        model_name: "specific-adapter-conflict",
        litellm_params: { model: "openai/kimi-k2.5" },
        model_info: { mode: "chat", litellm_provider: "anthropic" },
      }),
    ).toMatchObject({ provider: "openai", catalogModelId: "openai/kimi-k2.5" });
  });

  it.each([
    ["routing then base", "openai/gpt-4o", "openai/o3"],
    ["base then routing", "openai/o3", "openai/gpt-4o"],
  ])("uses base_model catalog authority when configured models differ in %s order", (_case, routing, base) => {
    const result = resolveModelInfoCatalog({
      model_name: "conflicting-openai-models",
      litellm_params: { model: routing },
      model_info: { mode: "chat", litellm_provider: "openai", base_model: base },
    });

    expect(result).toMatchObject({ provider: "openai", catalogModelId: base.slice("openai/".length) });
  });

  it.each([
    "moonshot/kimi-k2.7",
    "moonshot/kimi-k2.7-instruct",
    "moonshot/kimi-k2.7-codec",
    "moonshot/kimi-k2.7-highspeedy",
  ])("does not classify K2.7 without an exact Code suffix: %s", async (model) => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "unknown-k2.7-variant",
              litellm_params: { model, allowed_openai_params: ["thinking"] },
              model_info: { mode: "chat", supports_reasoning: true },
            },
          ],
        }),
    });
    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });
    expect(result.models[0]?.thinkingLevelMap).toEqual(NO_REASONING_LEVELS);
    expect(result.models[0]?.compat).not.toHaveProperty("thinkingFormat");
  });

  it.each(["moonshot/kimi-k2.7-code", "moonshot/kimi-k2.7_highspeed", "moonshot/kimi-k2.7.code"])(
    "classifies the supported K2.7 Code suffix: %s",
    async (model) => {
      mockEndpoints({
        "/model/info": () =>
          jsonResponse(200, {
            data: [
              {
                model_name: "known-k2.7-variant",
                litellm_params: { model, allowed_openai_params: ["thinking"] },
                model_info: { mode: "chat", supports_reasoning: true },
              },
            ],
          }),
      });
      const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });
      expect(result.models[0]).toMatchObject({
        thinkingLevelMap: { off: null, minimal: null, low: null, medium: null, high: "high", xhigh: null, max: null },
        compat: { thinkingFormat: "deepseek", supportsReasoningEffort: false },
      });
    },
  );

  it("publishes base_model catalog metadata when configured models differ within one deployment", async () => {
    const rows = [
      {
        model_name: "one-conflicted-openai-deployment",
        litellm_params: { model: "openai/gpt-4o" },
        model_info: { id: "one", mode: "chat", litellm_provider: "openai", base_model: "openai/o3" },
      },
      {
        model_name: "one-conflicted-openai-deployment",
        litellm_params: { model: "openai/o3" },
        model_info: { id: "one", mode: "chat", litellm_provider: "openai", base_model: "openai/gpt-4o" },
      },
    ];
    for (const data of rows) {
      mockEndpoints({ "/model/info": () => jsonResponse(200, { data: [data] }) });
      const result = await discoverModels("https://litellm.example.com", "sk-test", {});

      const catalog = getModel("openai", data.model_info.base_model.slice("openai/".length) as "o3" | "gpt-4o");
      expect(result.models[0]).toMatchObject({
        name: "one-conflicted-openai-deployment",
        reasoning: catalog.reasoning,
        input: catalog.input,
        contextWindow: catalog.contextWindow,
        maxTokens: catalog.maxTokens,
        cost: catalog.cost,
      });
      if (catalog.reasoning) expect(result.models[0]?.thinkingLevelMap).toEqual(NO_REASONING_LEVELS);
      else expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
    }
  });

  it("uses base_model metadata while withholding conflicting provider family policy", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "one-conflicted-deployment",
              litellm_params: { model: "openai/gpt-4o" },
              model_info: { id: "one", mode: "chat", base_model: "anthropic/claude-sonnet-4-6" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    const catalog = getModel("anthropic", "claude-sonnet-4-6");
    expect(result.models[0]).toMatchObject({
      name: "one-conflicted-deployment",
      reasoning: catalog.reasoning,
      input: catalog.input,
      contextWindow: catalog.contextWindow,
      maxTokens: catalog.maxTokens,
      cost: catalog.cost,
      thinkingLevelMap: NO_REASONING_LEVELS,
    });
    expect(result.models[0]).not.toHaveProperty("litellmPolicy");
  });

  it("does not restore route-name Kimi policy after an intra-row authority conflict", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "kimi-k2.6-vanity",
              litellm_params: { model: "openai/gpt-4o" },
              model_info: { id: "one", mode: "chat", base_model: "anthropic/claude-sonnet-4-6" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      id: "kimi-k2.6-vanity",
      reasoning: true,
      thinkingLevelMap: NO_REASONING_LEVELS,
    });
    expect(result.models[0]?.compat).not.toHaveProperty("requiresReasoningContentOnAssistantMessages");
    expect(result.models[0]).not.toHaveProperty("litellmPolicy");
    expect(stderr.mock.calls.flat().join(" ")).not.toContain("strict tool-message repair is withheld");
  });

  it("retains public effort evidence while withholding conflicting family policy", async () => {
    vi.resetModules();
    const { discoverModels: isolatedDiscoverModels } = await import("../src/discover.js");
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "conflicting-family-hit",
              litellm_params: { model: "openai/private-gpt", allowed_openai_params: ["reasoning_effort"] },
              model_info: {
                id: "one",
                mode: "chat",
                litellm_provider: "openai",
                base_model: "openai/kimi-k2.5",
                supports_reasoning: true,
              },
            },
          ],
        }),
      "models.dev/api.json": () =>
        jsonResponse(200, {
          openai: {
            models: {
              "kimi-k2.5": { reasoning_options: { type: "effort", values: ["low", "medium", "high"] } },
            },
          },
        }),
    });

    const result = await isolatedDiscoverModels("https://litellm.example.com", "sk-test", {
      modelsDevCachePath: join(await mkdtemp(join(agentDir, "conflicting-efforts-")), "models-dev.json"),
    });

    expect(result.models[0]).toMatchObject({
      id: "conflicting-family-hit",
      reasoning: true,
      compat: { supportsStore: false },
    });
    expect(result.models[0]?.compat).not.toHaveProperty("cacheControlFormat");
    expect(result.models[0]?.thinkingLevelMap).toEqual({
      off: null,
      minimal: null,
      low: "low",
      medium: "medium",
      high: "high",
      xhigh: null,
      max: null,
    });
    expect(result.models[0]).not.toHaveProperty("litellmPolicy");
  });

  it("applies Anthropic cache compatibility to a Fable-backed alias", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "team-writer",
              litellm_params: { model: "bedrock/fable-5" },
              model_info: { id: "one", mode: "chat" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      id: "team-writer",
      compat: { supportsStore: false, cacheControlFormat: "anthropic" },
    });
  });

  it("applies Anthropic cache compatibility to an evidence-free Fable route", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [{ model_name: "team-fable-5", model_info: { id: "one", mode: "chat" } }],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      id: "team-fable-5",
      compat: { supportsStore: false, cacheControlFormat: "anthropic" },
    });
  });

  it("classifies a Codex deployment identity as OpenAI", () => {
    expect(
      resolveModelInfoCatalog({
        model_name: "codex-route",
        litellm_params: { model: "azure/codex-mini" },
        model_info: { mode: "chat" },
      }),
    ).toMatchObject({ provider: "openai", catalogModelId: "codex-mini" });
  });

  it("does not enrich an unqualified route from an unrelated provider catalog", async () => {
    mockEndpoints({
      "/model/info": () => jsonResponse(200, { data: [{ model_name: "gpt-4o", model_info: { mode: "chat" } }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      id: "gpt-4o",
      name: "gpt-4o (incomplete metadata)",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128000,
      maxTokens: 16384,
    });
  });

  it("diagnoses conflicting provider evidence within one deployment", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "conflicting-authority",
              litellm_params: { model: "openai/gpt-4o", custom_llm_provider: "anthropic" },
              model_info: { id: "one", mode: "chat" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      name: "conflicting-authority (incomplete metadata)",
      reasoning: false,
    });
    expect(stderr.mock.calls.flat().join(" ")).toContain("conflicting-authority");
  });

  it("applies complete Moonshot compat to a vanity route from unanimous deployment evidence", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "internal/prod-chat",
              litellm_params: { model: "MoOnShOt/KiMi-K2.6" },
              model_info: { id: "one", mode: "chat" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.compat).toEqual({
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsStrictMode: false,
      maxTokensField: "max_tokens",
    });
    expect(result.models[0]?.litellmPolicy?.normalizeStrictToolMessages).toBe(true);
  });

  it("applies Kimi compat to a Fireworks account-scoped path routed by custom_llm_provider", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "fireworks/kimi-k2p6",
              litellm_params: { model: "accounts/fireworks/models/kimi-k2p6", custom_llm_provider: "fireworks_ai" },
              model_info: { id: "one", mode: "chat" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.compat).toMatchObject({ maxTokensField: "max_tokens", supportsStrictMode: false });
    expect(stderr.mock.calls.flat().join(" ")).not.toContain("conflicting deployment family evidence");
  });

  it("keeps strict repair but withholds visibility suppression for mixed Kimi thinking modes", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "mixed-kimi-mode-route",
              litellm_params: { model: "moonshot/kimi-k2.6" },
              model_info: { id: "normal", mode: "chat" },
            },
            {
              model_name: "mixed-kimi-mode-route",
              litellm_params: { model: "moonshot/kimi-k2-thinking" },
              model_info: { id: "thinking", mode: "chat" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.litellmPolicy).toEqual({
      normalizeStrictToolMessages: true,
      normalizeThinkTags: false,
      suppressReasoningVisibility: false,
    });
  });

  it("withholds strict tool repair for partial Moonshot evidence and reports a vanity route", async () => {
    const writes: string[] = [];
    vi.spyOn(process.stderr, "write").mockImplementation((chunk) => {
      writes.push(String(chunk));
      return true;
    });
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "internal/mixed-tool-route",
              litellm_params: { model: "moonshot/kimi-k2.6" },
              model_info: { id: "a", mode: "chat" },
            },
            {
              model_name: "internal/mixed-tool-route",
              litellm_params: { model: "internal/opaque" },
              model_info: { id: "b", mode: "chat" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.litellmPolicy).toMatchObject({
      normalizeStrictToolMessages: false,
      normalizeThinkTags: false,
    });
    expect(writes.filter((line) => line.includes("strict tool-message repair is withheld"))).toHaveLength(1);
    expect(writes.join("\n")).toContain("internal/mixed-tool-route");
  });

  // Issue #184: LiteLLM omits supported_openai_params for a deployment its model map
  // does not describe, so an explicit supports_reasoning is the operator's opt-in.
  it("keeps Pi's standard levels for an off-map deployment that opts into reasoning", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "opaque-chat-reasoner",
              litellm_params: { model: "internal/reasoner" },
              model_info: { id: "one", mode: "chat", supports_reasoning: true },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});
    const model = { ...result.models[0]!, provider: "litellm", baseUrl: "https://proxy.example.com" };

    expect(model.compat).toMatchObject({ supportsReasoningEffort: true });
    expect(getSupportedThinkingLevels(model)).toEqual(["off", "minimal", "low", "medium", "high"]);
  });

  it("denies Chat reasoning levels without an accepted carrier", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "opaque-chat-reasoner",
              litellm_params: { model: "internal/reasoner" },
              model_info: {
                id: "one",
                mode: "chat",
                supported_openai_params: ["temperature"],
                supports_reasoning: true,
              },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({ api: "openai-completions", reasoning: true });
    expect(result.models[0]?.thinkingLevelMap).toEqual({
      off: null,
      minimal: null,
      low: null,
      medium: null,
      high: null,
      xhigh: null,
      max: null,
    });
  });

  it("denies Responses reasoning levels when the only accepted control is Chat thinking", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "thinking-only-responses",
              litellm_params: { model: "moonshot/kimi-k2.6", allowed_openai_params: ["thinking"] },
              model_info: { id: "one", mode: "responses", supports_reasoning: true },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({ api: "openai-responses", reasoning: true });
    expect(result.models[0]?.thinkingLevelMap).toEqual({
      off: null,
      minimal: null,
      low: null,
      medium: null,
      high: null,
      xhigh: null,
      max: null,
    });
    expect(result.models[0]).not.toHaveProperty("litellmResponsesReasoningControl");
  });

  it("denies Responses reasoning levels unless every deployment accepts reasoning_effort", async () => {
    const deployments = [
      {
        model_name: "mixed-control-responses",
        litellm_params: { model: "moonshot/kimi-k2.6", allowed_openai_params: ["reasoning_effort", "thinking"] },
        model_info: { id: "effort", mode: "responses", supports_reasoning: true },
      },
      {
        model_name: "mixed-control-responses",
        litellm_params: { model: "moonshot/kimi-k2.6", allowed_openai_params: ["thinking"] },
        model_info: { id: "thinking", mode: "responses", supports_reasoning: true },
      },
    ];
    const fetchMock = vi.spyOn(globalThis, "fetch");
    for (const rows of [deployments, [...deployments].reverse()]) {
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { data: rows }));
      const result = await discoverModels("https://litellm.example.com", "sk-test", {});

      expect(result.models[0]).toMatchObject({ api: "openai-responses", reasoning: true });
      expect(result.models[0]?.thinkingLevelMap).toEqual({
        off: null,
        minimal: null,
        low: null,
        medium: null,
        high: null,
        xhigh: null,
        max: null,
      });
      expect(result.models[0]).not.toHaveProperty("litellmResponsesReasoningControl");
    }
  });

  it("preserves accepted Responses reasoning control through cached enrichment", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "effort-responses",
              litellm_params: { model: "moonshot/kimi-k3", allowed_openai_params: ["reasoning_effort"] },
              model_info: { id: "one", mode: "responses", supports_reasoning: true },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});
    const model = result.models[0];

    expect(model).toMatchObject({ api: "openai-responses", litellmResponsesReasoningControl: true });
    expect(model?.thinkingLevelMap).toMatchObject({ low: "low", high: "high", max: null });
    expect(model && enrichCachedModel(model as never).thinkingLevelMap).toEqual(model?.thinkingLevelMap);
  });

  it("keeps a Responses-mode deployment on Responses when discovered through /health", async () => {
    mockEndpoints({
      "/model/info?litellm_model_id=uuid-1": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "responses-route",
              litellm_params: { model: "openai/gpt-5.5" },
              model_info: { id: "uuid-1", mode: "responses" },
            },
          ],
        }),
      "/model/info": () => jsonResponse(403, {}),
      "/v1/models": () => jsonResponse(404, {}),
      "/health": () => jsonResponse(200, { healthy_endpoints: [{ model: "openai/gpt-5.5", model_id: "uuid-1" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.source).toBe("health");
    expect(result.models[0]).toMatchObject({ id: "responses-route", api: "openai-responses" });
  });

  it("falls back to the health route name when a deployment row's own name is unreadable", async () => {
    // The detail row's `model_name` is unusable, but `/health` named the route, so the
    // model must survive under that name rather than being discarded.
    mockEndpoints({
      "/model/info?litellm_model_id=uuid-1": () =>
        jsonResponse(200, { data: [{ model_name: 7, model_info: { mode: "chat" } }] }),
      "/model/info": () => jsonResponse(403, {}),
      "/v1/models": () => jsonResponse(404, {}),
      "/health": () => jsonResponse(200, { healthy_endpoints: [{ model: "named-route", model_id: "uuid-1" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.source).toBe("health");
    expect(result.models.map((model) => model.id)).toEqual(["named-route"]);
  });

  it("preserves route-only Kimi think-tag policy without enabling request controls", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [{ model_name: "kimi-k2.6", model_info: { id: "only", mode: "chat" } }],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      litellmPolicy: {
        normalizeStrictToolMessages: false,
        normalizeThinkTags: true,
        suppressReasoningVisibility: false,
      },
    });
    expect(result.models[0]?.reasoning).toBe(false);
    expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
    expect(result.models[0]?.compat).toMatchObject({ supportsReasoningEffort: false });
  });

  it("does not use a public route name as evidence for conflicting duplicate deployment ids", async () => {
    // One deployment id with two disagreeing backends is not a single deployment,
    // so the catalog-resolvable route name must not enrich the group.
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            { model_name: "openai/gpt-5.5", model_info: { id: "same", mode: "chat" } },
            {
              model_name: "openai/gpt-5.5",
              model_info: { id: "same", mode: "chat", max_input_tokens: 64_000 },
              litellm_params: { model: "internal/mystery" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      id: "openai/gpt-5.5",
      name: "openai/gpt-5.5 (incomplete metadata)",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      maxTokens: 16_384,
    });
    expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
  });

  it.each([
    ["chat first", "chat", "embedding"],
    ["embedding first", "embedding", "chat"],
    ["Responses first", "responses", "embedding"],
    ["embedding before Responses", "embedding", "responses"],
  ])(
    "withholds and diagnoses a mixed chat-style and incompatible route with $0",
    async (caseName, firstMode, secondMode) => {
      const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
      const route = `mixed-route-${caseName.replaceAll(" ", "-")}`;
      mockEndpoints({
        "/model/info": () =>
          jsonResponse(200, {
            data: [
              { model_name: route, model_info: { id: "first", mode: firstMode } },
              { model_name: route, model_info: { id: "second", mode: secondMode } },
            ],
          }),
      });

      const result = await discoverModels("https://litellm.example.com", "sk-test", {});

      expect(result.models).toEqual([]);
      expect(stderr).toHaveBeenCalledTimes(1);
      expect(String(stderr.mock.calls[0]?.[0])).toContain(
        "1 route group(s) mix chat-style and explicitly incompatible deployment modes",
      );
      expect(String(stderr.mock.calls[0]?.[0])).toContain(route);
    },
  );

  it("bounds incompatible-mode diagnostics and reports each route once", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const mixed = (route: string) => [
      { model_name: route, model_info: { id: `${route}-chat`, mode: "chat" } },
      { model_name: route, model_info: { id: `${route}-embedding`, mode: "embedding" } },
    ];
    const routes = ["bounded-mode-a", "bounded-mode-b", "bounded-mode-c", "bounded-mode-d"];
    const discover = async () => {
      mockEndpoints({ "/model/info": () => jsonResponse(200, { data: routes.flatMap(mixed) }) });
      await discoverModels("https://litellm.example.com", "sk-test", {});
    };

    await discover();
    expect(stderr).toHaveBeenCalledTimes(1);
    expect(String(stderr.mock.calls[0]?.[0])).toContain("bounded-mode-a, bounded-mode-b, bounded-mode-c (+1 more)");
    expect(String(stderr.mock.calls[0]?.[0])).not.toContain("bounded-mode-d");

    await discover();
    expect(stderr).toHaveBeenCalledTimes(1);
  });

  it("stays silent when provider identity is unanimous or wholly unknown", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            { model_name: "agreed", model_info: { id: "a", mode: "chat" }, litellm_params: { model: "openai/gpt-4o" } },
            { model_name: "agreed", model_info: { id: "b", mode: "chat" }, litellm_params: { model: "openai/gpt-4o" } },
            { model_name: "unknown", model_info: { id: "c", mode: "chat" }, litellm_params: { model: "internal/x" } },
            { model_name: "unknown", model_info: { id: "d", mode: "chat" }, litellm_params: { model: "internal/y" } },
          ],
        }),
    });

    await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(stderr).not.toHaveBeenCalled();
  });
});

describe("discoverModels via /health", () => {
  it("enriches deployment details without granting catalog metadata to health-only routes", async () => {
    const dir = await mkdtemp(join(tmpdir(), "litellm-health-detail-catalog-"));
    const cachePath = join(dir, "models-dev.json");
    await writeFile(
      cachePath,
      JSON.stringify({
        fetchedAt: 1,
        catalog: {
          private: {
            models: {
              "priced-model": {
                modalities: { input: ["text", "image"] },
                limit: { context: 64_000, output: 8_000 },
                cost: { input: 7, output: 9, cache_read: 1, cache_write: 2 },
              },
            },
          },
        },
      }),
    );
    mockEndpoints({
      "/model/info": () => jsonResponse(404, {}),
      "/v1/models": () => jsonResponse(404, {}),
      "/health": () =>
        jsonResponse(200, {
          healthy_endpoints: [
            { model: "detailed-route", model_id: "detailed-deployment" },
            { model: "private/priced-model" },
          ],
        }),
      "/model/info?litellm_model_id=detailed-deployment": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "detailed-route",
              litellm_params: { model: "private/priced-model" },
              model_info: { mode: "chat", supports_reasoning: false, supports_vision: true },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {
      modelsDev: false,
      modelsDevCachePath: cachePath,
    });

    expect(result.models.find((model) => model.id === "detailed-route")).toMatchObject({
      name: "detailed-route",
      input: ["text", "image"],
      contextWindow: 64_000,
      maxTokens: 8_000,
      cost: { input: 7, output: 9, cacheRead: 1, cacheWrite: 2 },
    });
    expect(result.models.find((model) => model.id === "private/priced-model")).toMatchObject({
      name: "private/priced-model (no metadata)",
      input: ["text"],
      contextWindow: 128_000,
      maxTokens: 16_384,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    });
  });

  it("withholds mixed chat and embedding detail rows with one diagnostic", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const route = `mixed-health-route-${process.pid}-${Date.now()}-${Math.random()}`;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info") || url.endsWith("/v1/models")) return jsonResponse(404, {});
      if (url.endsWith("/health")) {
        return jsonResponse(200, {
          healthy_endpoints: [
            { model: route, model_id: "chat-detail" },
            { model: route, model_id: "embedding-detail" },
            { model: "safe-health-route" },
          ],
        });
      }
      if (url.endsWith("litellm_model_id=chat-detail")) {
        return jsonResponse(200, { data: [{ model_name: route, model_info: { id: "chat", mode: "chat" } }] });
      }
      if (url.endsWith("litellm_model_id=embedding-detail")) {
        return jsonResponse(200, {
          data: [{ model_name: route, model_info: { id: "embedding", mode: "embedding" } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models.map((model) => model.id)).toEqual(["safe-health-route"]);
    expect(stderr).toHaveBeenCalledTimes(1);
    expect(String(stderr.mock.calls[0]?.[0])).toContain(
      `1 route group(s) mix chat-style and explicitly incompatible deployment modes; the routes are withheld because not every deployment can accept chat requests: ${route}`,
    );
  });

  it("withholds an embedding-only health route without a diagnostic", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const route = `embedding-health-route-${process.pid}-${Date.now()}-${Math.random()}`;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info") || url.endsWith("/v1/models")) return jsonResponse(404, {});
      if (url.endsWith("/health")) {
        return jsonResponse(200, {
          healthy_endpoints: [{ model: route, model_id: "embedding-detail" }, { model: "safe-health-route" }],
        });
      }
      if (url.endsWith("litellm_model_id=embedding-detail")) {
        return jsonResponse(200, {
          data: [{ model_name: route, model_info: { id: "embedding", mode: "embedding" } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models.map((model) => model.id)).toEqual(["safe-health-route"]);
    expect(stderr).not.toHaveBeenCalled();
  });

  it.each([
    ["Responses first", ["responses", "chat"]],
    ["Chat first", ["chat", "responses"]],
  ] as const)("reduces conflicting deployment details conservatively with %s", async (_name, modes) => {
    const route = `shared-health-route-${process.pid}-${Date.now()}-${Math.random()}`;
    const details = {
      responses: {
        model_name: route,
        model_info: {
          id: "responses",
          mode: "responses",
          supports_reasoning: true,
          supports_vision: true,
          max_input_tokens: 200_000,
          max_output_tokens: 64_000,
          input_cost_per_token: 0.000001,
          output_cost_per_token: 0.00001,
          cache_read_input_token_cost: 0.0000001,
          cache_creation_input_token_cost: 0.000002,
        },
      },
      chat: {
        model_name: route,
        model_info: {
          id: "chat",
          mode: "chat",
          supports_reasoning: false,
          supports_vision: false,
          max_input_tokens: 8_000,
          max_output_tokens: 1_000,
          input_cost_per_token: 0.000002,
          output_cost_per_token: 0.00002,
          cache_read_input_token_cost: 0.0000002,
          cache_creation_input_token_cost: 0.000003,
        },
      },
    };
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info") || url.endsWith("/v1/models")) return jsonResponse(404, {});
      if (url.endsWith("/health")) {
        return jsonResponse(200, {
          healthy_endpoints: modes.map((mode) => ({ model: route, model_id: mode })),
        });
      }
      const mode = modes.find((candidate) => url.endsWith(`litellm_model_id=${candidate}`));
      if (mode) return jsonResponse(200, { data: [details[mode]] });
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models).toHaveLength(1);
    expect(result.models[0]).toMatchObject({
      id: route,
      name: route,
      api: "openai-completions",
      reasoning: false,
      input: ["text"],
      contextWindow: 8_000,
      maxTokens: 1_000,
      cost: { input: 2, output: 20, cacheWrite: 3 },
    });
    expect(result.models[0]?.cost.cacheRead).toBeCloseTo(0.2);
  });

  it.each([
    ["Moonshot first", ["moonshot/kimi-k3", "azure/gpt-4o"]],
    ["Azure first", ["azure/gpt-4o", "moonshot/kimi-k3"]],
  ] as const)("does not suppress duplicate routes with different backend families when %s", async (_name, routes) => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info") || url.endsWith("/v1/models")) return jsonResponse(403, {});
      if (url.endsWith("/health")) {
        return jsonResponse(200, {
          healthy_endpoints: routes.map((_, index) => ({ model: "kimi-prod", model_id: `route-${index}` })),
        });
      }
      const route = routes[Number(url.match(/route-(\d+)/)?.[1])];
      return jsonResponse(200, { data: [{ litellm_params: { model: route }, model_info: { mode: "chat" } }] });
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.source).toBe("health");
    expect(result.models).toHaveLength(1);
    expect(result.models[0]?.litellmPolicy?.suppressReasoningVisibility === true).toBe(false);
  });

  it("suppresses duplicate proven non-forced Moonshot health routes", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info") || url.endsWith("/v1/models")) return jsonResponse(403, {});
      if (url.endsWith("/health")) {
        return jsonResponse(200, {
          healthy_endpoints: [
            { model: "kimi-prod", model_id: "route-1" },
            { model: "kimi-prod", model_id: "route-2" },
          ],
        });
      }
      return jsonResponse(200, {
        data: [{ litellm_params: { model: "moonshot/kimi-k3" }, model_info: { mode: "chat" } }],
      });
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.litellmPolicy?.suppressReasoningVisibility).toBe(true);
  });
});

describe("discoverModels wildcard expansion via /v1/models", () => {
  it("restores Chat reasoning carriers and cache markers after wildcard protocol selection", async () => {
    const row = {
      model_name: "team/*",
      litellm_params: { model: "openai/*" },
      model_info: {
        supports_reasoning: true,
        supported_openai_params: ["reasoning_effort"],
        supports_low_reasoning_effort: true,
      },
    };
    expect(modelProtocol(row.model_name, row).api).toBe("openai-responses");
    mockEndpoints({
      "/model/info": () => jsonResponse(200, { data: [row] }),
      "/v1/models": () => jsonResponse(200, { data: [{ id: "team/claude-sonnet-4-5" }] }),
    });

    const { models } = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(models).toHaveLength(1);
    expect(models[0]).toMatchObject({
      id: "team/claude-sonnet-4-5",
      api: "openai-completions",
      litellmBackendFamily: "claude",
      thinkingLevelMap: { low: "low" },
      compat: { supportsReasoningEffort: true, cacheControlFormat: "anthropic" },
    });
    expect(models[0]).not.toHaveProperty("litellmResponsesReasoningControl");
  });

  it("expands a wildcard /model/info entry and drops the literal wildcard id", async () => {
    const urls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      urls.push(url);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            { model_name: "lemonade/*", model_info: { mode: "chat" } },
            { model_name: "lemonade/Qwen3.6-35B-A3B-MTP-GGUF-UD-IQ4_NL", model_info: { mode: "chat" } },
          ],
        });
      }
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, {
          data: [
            { id: "lemonade/*", object: "model", owned_by: "openai" },
            { id: "lemonade/Bonsai-1.7B-gguf", object: "model", owned_by: "openai" },
            { id: "lemonade/Laguna-S-2.1-GGUF-UD-IQ4_NL", object: "model", owned_by: "openai" },
            { id: "lemonade/Qwen3.6-35B-A3B-MTP-GGUF-UD-IQ4_NL", object: "model", owned_by: "openai" },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.source).toBe("model_info");
    expect(result.models.map((m) => m.id).sort()).toEqual([
      "lemonade/Bonsai-1.7B-gguf",
      "lemonade/Laguna-S-2.1-GGUF-UD-IQ4_NL",
      "lemonade/Qwen3.6-35B-A3B-MTP-GGUF-UD-IQ4_NL",
    ]);
    expect(result.models).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: "lemonade/Bonsai-1.7B-gguf", api: "openai-completions" }),
        expect.objectContaining({ id: "lemonade/Laguna-S-2.1-GGUF-UD-IQ4_NL", api: "openai-completions" }),
      ]),
    );
    // the raw wildcard id must NOT surface as a selectable model
    expect(result.models.some((m) => m.id.includes("*"))).toBe(false);
    // the concrete /model/info entry is not duplicated by /v1/models
    expect(result.models.filter((m) => m.id === "lemonade/Qwen3.6-35B-A3B-MTP-GGUF-UD-IQ4_NL")).toHaveLength(1);
    // /v1/models was actually queried (the expansion path ran)
    expect(urls.some((u) => u.endsWith("/v1/models"))).toBe(true);
  });

  it("keeps a wildcard child on Chat when its wildcard row pins an older Azure API version", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "azure/*",
              litellm_params: { model: "azure/*", api_version: "2024-10-21" },
              model_info: { mode: "chat", litellm_provider: "azure" },
            },
          ],
        });
      }
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, { data: [{ id: "azure/gpt-5.5", object: "model", owned_by: "openai" }] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    // The Pi catalog knows gpt-5.5 as a Responses model, but the wildcard row that serves
    // the child is an Azure deployment on an API version that requires Chat Completions.
    expect(result.models).toHaveLength(1);
    expect(result.models[0]).toMatchObject({
      id: "azure/gpt-5.5",
      api: "openai-completions",
      litellmBackendFamily: "openai",
    });
  });

  it("uses only the wildcard route LiteLLM would select for a child", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "*",
              litellm_params: { model: "anthropic/*" },
              model_info: { supported_endpoints: ["/v1/chat/completions"] },
            },
            {
              model_name: "openai/*",
              litellm_params: { model: "openai/*" },
              model_info: { supported_endpoints: ["/v1/responses"] },
            },
          ],
        });
      }
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, { data: [{ id: "openai/gpt-5", object: "model", owned_by: "openai" }] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models).toHaveLength(1);
    expect(result.models[0]).toMatchObject({
      id: "openai/gpt-5",
      api: "openai-responses",
      litellmBackendFamily: "openai",
    });
  });

  it("resolves equally specific wildcard routes independently of row order", async () => {
    const chatFirst = [
      { model_name: "team-*", model_info: { mode: "chat" } },
      { model_name: "*-team", model_info: { mode: "embedding" } },
    ];
    const results = [];

    for (const rows of [chatFirst, [...chatFirst].reverse()]) {
      vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
        const url = input instanceof URL ? input.toString() : String(input);
        if (url.endsWith("/model/info")) return jsonResponse(200, { data: rows });
        if (url.endsWith("/v1/models")) {
          return jsonResponse(200, { data: [{ id: "team-x-team", object: "model" }] });
        }
        throw new Error(`unexpected URL: ${url}`);
      });

      results.push((await discoverModels("https://litellm.example.com", "sk-test", {})).models);
      vi.restoreAllMocks();
    }

    expect(results).toEqual([[], []]);
  });

  it("lets rows sharing the selected wildcard route vote together", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "team/*",
              litellm_params: { model: "openai/*" },
              model_info: { supported_endpoints: ["/v1/responses"] },
            },
            {
              model_name: "team/*",
              litellm_params: { model: "openai/*" },
              model_info: { supported_endpoints: ["/v1/chat/completions"] },
            },
          ],
        });
      }
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, { data: [{ id: "team/production", object: "model", owned_by: "openai" }] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models).toHaveLength(1);
    expect(result.models[0]).toMatchObject({
      id: "team/production",
      api: "openai-completions",
      litellmBackendFamily: "openai",
    });
  });

  it("does not let a rejected embedding wildcard authorize a child", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            { model_name: "chat/*", model_info: { mode: "chat" } },
            { model_name: "embed/*", model_info: { mode: "embedding" } },
          ],
        });
      }
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, {
          data: [
            { id: "chat/a", object: "model" },
            { id: "embed/x", object: "model" },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models.map((model) => model.id)).toEqual(["chat/a"]);
  });

  it("does not fall through to a published catch-all when the selected wildcard route is rejected", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "*",
              litellm_params: { model: "openai/*" },
              model_info: { mode: "chat" },
            },
            { model_name: "embed/*", model_info: { mode: "embedding" } },
          ],
        });
      }
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, {
          data: [{ id: "embed/text-embedding-3-large", object: "model" }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models).toEqual([]);
  });

  it("recomputes concrete-id compatibility while preserving conservative wildcard metadata", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const nonce = `${process.pid}-${Date.now()}-${Math.random()}`;
    const wildcard = `openai/${nonce}/*`;
    const claudeId = `openai/${nonce}/claude-sonnet-4-6`;
    const kimiId = `openai/${nonce}/kimi-k2.6`;
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: wildcard,
              litellm_params: { model: "openai/gpt-4o" },
              model_info: {
                id: "openai-route",
                mode: "responses",
                supports_vision: true,
                max_input_tokens: 32_000,
                max_output_tokens: 4_000,
                input_cost_per_token: 0.000004,
                output_cost_per_token: 0.00002,
                cache_read_input_token_cost: 0.0000004,
                cache_creation_input_token_cost: 0.000005,
              },
            },
            {
              model_name: wildcard,
              litellm_params: { model: "anthropic/claude-sonnet-4-6" },
              model_info: {
                id: "anthropic-route",
                mode: "responses",
                supports_reasoning: false,
                supports_vision: false,
                max_input_tokens: 16_000,
                max_output_tokens: 2_000,
                input_cost_per_token: 0.000003,
                output_cost_per_token: 0.000015,
                cache_read_input_token_cost: 0.0000003,
                cache_creation_input_token_cost: 0.00000375,
              },
            },
          ],
        }),
      "/v1/models": () =>
        jsonResponse(200, {
          data: [
            { id: claudeId, owned_by: "openai" },
            { id: kimiId, owned_by: "openai" },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models).toEqual([
      expect.objectContaining({
        id: claudeId,
        name: `${getModel("anthropic", "claude-sonnet-4-6").name} (incomplete metadata)`,
        api: "openai-responses",
        reasoning: false,
        input: ["text"],
        contextWindow: 16_000,
        maxTokens: 2_000,
        cost: expect.objectContaining({ input: 4, output: 20, cacheWrite: 5 }),
        compat: undefined,
      }),
      expect.objectContaining({
        id: kimiId,
        name: `${kimiId} (incomplete metadata)`,
        reasoning: false,
        input: ["text"],
        contextWindow: 16_000,
        maxTokens: 2_000,
        compat: undefined,
      }),
    ]);
    expect(stderr).toHaveBeenCalled();
  });

  it("recomputes Kimi compatibility from a catch-all expansion id", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "*",
              litellm_params: { model: "private/unknown" },
              model_info: {
                mode: "responses",
                supports_reasoning: false,
                max_input_tokens: 8_000,
                max_output_tokens: 1_000,
                input_cost_per_token: 0.000001,
              },
            },
          ],
        }),
      "/v1/models": () => jsonResponse(200, { data: [{ id: "kimi-k2.6" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models).toEqual([
      expect.objectContaining({
        id: "kimi-k2.6",
        name: "kimi-k2.6 (incomplete metadata)",
        api: "openai-responses",
        reasoning: false,
        contextWindow: 8_000,
        maxTokens: 1_000,
        cost: { input: 1, output: 0, cacheRead: 0, cacheWrite: 0 },
        compat: {
          supportsDeveloperRole: false,
          supportsStrictMode: false,
        },
      }),
    ]);
  });

  it("preserves unresolved conservative wildcard metadata on expanded catalog-matching ids", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "openai/*",
              litellm_params: { model: "private/unknown" },
              model_info: {
                mode: "chat",
                supports_reasoning: false,
                max_input_tokens: 8_000,
                input_cost_per_token: 0.000001,
              },
            },
          ],
        }),
      "/v1/models": () => jsonResponse(200, { data: [{ id: "openai/gpt-4o", owned_by: "openai" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models).toEqual([
      expect.objectContaining({
        id: "openai/gpt-4o",
        name: "GPT-4o (incomplete metadata)",
        reasoning: false,
        input: ["text"],
        contextWindow: 8_000,
        maxTokens: 16_384,
        cost: { input: 1, output: 0, cacheRead: 0, cacheWrite: 0 },
      }),
    ]);
  });

  it.each([
    ["Moonshot", "moonshot/*", true],
    ["non-Moonshot", "openai/*", false],
  ] as const)("aggregates %s suppression evidence onto wildcard expansions", async (_case, backend, suppress) => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "team/*",
              litellm_params: { model: backend },
              model_info: { id: "wildcard", mode: "chat" },
            },
          ],
        }),
      "/v1/models": () => jsonResponse(200, { data: [{ id: "team/model-a" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.litellmPolicy?.suppressReasoningVisibility === true).toBe(suppress);
  });

  it("does not suppress a forced-thinking id expanded from a Moonshot wildcard", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "*",
              litellm_params: { model: "moonshot/*" },
              model_info: { id: "wildcard", mode: "chat" },
            },
          ],
        }),
      "/v1/models": () => jsonResponse(200, { data: [{ id: "kimi-k2-thinking" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.id).toBe("kimi-k2-thinking");
    expect(result.models[0]?.litellmPolicy?.suppressReasoningVisibility === true).toBe(false);
  });

  it("requires unanimous suppression evidence across matching wildcard groups", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "team/*",
              litellm_params: { model: "moonshot/*" },
              model_info: { id: "team-wildcard", mode: "chat" },
            },
            {
              model_name: "*",
              litellm_params: { model: "openai/*" },
              model_info: { id: "catch-all-wildcard", mode: "chat" },
            },
          ],
        }),
      "/v1/models": () => jsonResponse(200, { data: [{ id: "team/model-a" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.id).toBe("team/model-a");
    expect(result.models[0]?.litellmPolicy?.suppressReasoningVisibility === true).toBe(false);
  });

  it("preserves a wildcard Responses route mode on expanded models", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            { model_name: "responses/*", model_info: { mode: "responses" } },
            { model_name: "chat/*", model_info: { mode: "chat" } },
          ],
        }),
      "/v1/models": () =>
        jsonResponse(200, {
          data: [
            { id: "responses/model-a", owned_by: "openai" },
            { id: "chat/model-b", owned_by: "openai" },
            { id: "other/model-c", owned_by: "openai" },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: "responses/model-a", api: "openai-responses" }),
        expect.objectContaining({ id: "chat/model-b", api: "openai-completions" }),
      ]),
    );
    expect(result.models.map((model) => model.id)).not.toContain("other/model-c");
  });

  it("uses thinking levels from the selected most-specific wildcard", async () => {
    const wildcard = (modelName: string, id: string, supports: { low?: boolean; max?: boolean }) => ({
      model_name: modelName,
      litellm_params: { model: `internal/${id}`, allowed_openai_params: ["reasoning_effort"] },
      model_info: {
        id,
        mode: "chat",
        supports_reasoning: true,
        supports_low_reasoning_effort: supports.low,
        supports_max_reasoning_effort: supports.max,
        input_cost_per_token: 0,
        output_cost_per_token: 0,
        cache_read_input_token_cost: 0,
        cache_creation_input_token_cost: 0,
        max_input_tokens: 8_000,
        max_output_tokens: 1_000,
        supports_vision: false,
      },
    });

    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [wildcard("*", "broad", { low: true }), wildcard("team/*", "narrow", { low: false, max: true })],
        }),
      "/v1/models": () => jsonResponse(200, { data: [{ id: "team/model" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.thinkingLevelMap).toEqual({ low: null, xhigh: null, max: "max" });
  });

  it("preserves tiered pricing from a single wildcard match", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "team/*",
              litellm_params: { model: "openai/gpt-5.5" },
              model_info: { id: "wildcard", mode: "chat" },
            },
          ],
        }),
      "/v1/models": () => jsonResponse(200, { data: [{ id: "team/model" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.cost.tiers).toEqual([
      { inputTokensAbove: 272_000, input: 10, output: 45, cacheRead: 1, cacheWrite: 0 },
    ]);
  });

  it("combines compatible tiered pricing across overlapping wildcard matches conservatively", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "*",
              litellm_params: { model: "openai/gpt-5.5" },
              model_info: { id: "broad", mode: "chat" },
            },
            {
              model_name: "team/*",
              litellm_params: { model: "openai/gpt-5.5-pro" },
              model_info: { id: "narrow", mode: "chat" },
            },
          ],
        }),
      "/v1/models": () => jsonResponse(200, { data: [{ id: "team/model" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.cost.tiers).toEqual([
      { inputTokensAbove: 272_000, input: 60, output: 270, cacheRead: 1, cacheWrite: 0 },
    ]);
  });

  it("constructs a safe tier envelope when overlapping wildcard thresholds differ", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "*",
              litellm_params: { model: "openai/gpt-5.5" },
              model_info: { id: "broad", mode: "chat" },
            },
            {
              model_name: "team/*",
              litellm_params: { model: "github-copilot/grok-4.5" },
              model_info: { id: "narrow", mode: "chat" },
            },
          ],
        }),
      "/v1/models": () => jsonResponse(200, { data: [{ id: "team/model" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.cost.tiers).toEqual([
      { inputTokensAbove: 200_000, input: 5, output: 30, cacheRead: 1, cacheWrite: 0 },
      { inputTokensAbove: 272_000, input: 10, output: 45, cacheRead: 1, cacheWrite: 0 },
    ]);
  });

  it("retains known higher tiers when an overlapping wildcard has incomplete metadata", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "*",
              litellm_params: { model: "openai/gpt-5.5" },
              model_info: { id: "complete", mode: "chat" },
            },
            {
              model_name: "team/*",
              litellm_params: { model: "internal/unknown" },
              model_info: {
                id: "incomplete",
                mode: "chat",
                input_cost_per_token: 0.000002,
                output_cost_per_token: 0.00001,
                cache_read_input_token_cost: 0.0000002,
                cache_creation_input_token_cost: 0.0000025,
              },
            },
          ],
        }),
      "/v1/models": () => jsonResponse(200, { data: [{ id: "team/model" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.name).toBe("team/model (incomplete metadata)");
    expect(result.models[0]?.cost).toEqual({
      input: 5,
      output: 30,
      cacheRead: 0.5,
      cacheWrite: 2.5,
      tiers: [{ inputTokensAbove: 272_000, input: 10, output: 45, cacheRead: 1, cacheWrite: 2.5 }],
    });
  });

  it("combines overlapping wildcard metadata conservatively regardless of route order", async () => {
    const broad = {
      model_name: "*",
      litellm_params: { model: "internal/broad" },
      model_info: {
        id: "broad",
        mode: "responses",
        supports_reasoning: true,
        supports_vision: true,
        supports_low_reasoning_effort: true,
        max_input_tokens: 80_000,
        max_output_tokens: 8_000,
        input_cost_per_token: 0.000002,
        output_cost_per_token: 0.00001,
        cache_read_input_token_cost: 0.0000002,
        cache_creation_input_token_cost: 0.0000025,
      },
    };
    const narrow = {
      model_name: "team/*",
      litellm_params: { model: "internal/narrow" },
      model_info: {
        id: "narrow",
        mode: "chat",
        supports_reasoning: false,
        supports_vision: false,
        supports_low_reasoning_effort: false,
        max_input_tokens: 40_000,
        max_output_tokens: 4_000,
        input_cost_per_token: 0.000003,
        output_cost_per_token: 0.000015,
        cache_read_input_token_cost: 0.0000003,
        cache_creation_input_token_cost: 0.00000375,
      },
    };

    for (const data of [
      [broad, narrow],
      [narrow, broad],
    ]) {
      mockEndpoints({
        "/model/info": () => jsonResponse(200, { data }),
        "/v1/models": () => jsonResponse(200, { data: [{ id: "team/claude-sonnet-4-6" }] }),
      });

      const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

      expect(result.models).toEqual([
        {
          id: "team/claude-sonnet-4-6",
          name: "Claude Sonnet 4.6",
          api: "openai-completions",
          litellmDiscoveryVersion: 3,
          reasoning: false,
          input: ["text"],
          contextWindow: 40_000,
          maxTokens: 4_000,
          cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
          compat: { supportsStore: false, cacheControlFormat: "anthropic" },
        },
      ]);
    }
  });

  it.each(["team/*", "*"])(
    "keeps a dropped concrete group from being re-admitted through %s expansion",
    async (wildcardRoute) => {
      mockEndpoints({
        "/model/info": () =>
          jsonResponse(200, {
            data: [
              { model_name: wildcardRoute, model_info: { id: "wildcard", mode: "chat" } },
              { model_name: "team/embed", model_info: { id: "embedding", mode: "embedding" } },
            ],
          }),
        "/v1/models": () => jsonResponse(200, { data: [{ id: "team/chat" }, { id: "team/embed" }] }),
      });

      const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

      expect(result.models.map((model) => model.id)).toEqual(["team/chat"]);
    },
  );

  it("keeps a deliberately dropped wildcard group from being re-admitted", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            { model_name: "team/*", model_info: { id: "wildcard", mode: "chat" } },
            { model_name: "blocked/*", model_info: { id: "blocked", mode: "embedding" } },
          ],
        }),
      "/v1/models": () => jsonResponse(200, { data: [{ id: "team/chat" }, { id: "blocked/chat" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models.map((model) => model.id)).toEqual(["team/chat"]);
  });

  it("never exposes a literal wildcard when /v1/models expansion fails", async () => {
    mockEndpoints({
      "/model/info": () => jsonResponse(200, { data: [{ model_name: "team/*", model_info: { mode: "chat" } }] }),
      "/v1/models": () => jsonResponse(500, { error: "unavailable" }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models).toEqual([]);
  });

  it("drops literal wildcard ids when expansion fails", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "*", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/v1/models")) return new Response(null, { status: 500 });
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result).toEqual({ source: "model_info", models: [] });
  });

  it("does not publish list ids that match no wildcard route", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [{ model_name: "team/*", litellm_params: { model: "openai/*" }, model_info: { mode: "chat" } }],
        });
      }
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, {
          data: [
            { id: "team/gpt-production", object: "model", owned_by: "openai" },
            { id: "unrelated/model", object: "model", owned_by: "openai" },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models.map((model) => model.id)).toEqual(["team/gpt-production"]);
  });

  it("gives a wildcard child Responses when its wildcard row is a current OpenAI deployment", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [{ model_name: "team/*", litellm_params: { model: "openai/*" }, model_info: { mode: "chat" } }],
        });
      }
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, { data: [{ id: "team/gpt-production", object: "model", owned_by: "openai" }] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({ id: "team/gpt-production", api: "openai-responses" });
  });

  it("does not query /v1/models when /model/info has no wildcards", async () => {
    const urls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      urls.push(url);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [{ model_name: "openai/gpt-4o", model_info: { mode: "chat" } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.source).toBe("model_info");
    expect(result.models.map((m) => m.id)).toEqual(["openai/gpt-4o"]);
    expect(urls.some((u) => u.endsWith("/v1/models"))).toBe(false);
  });
  it("carries a wildcard parent's tiered pricing into its child", async () => {
    const entry = {
      model_name: "team/*",
      litellm_params: { model: "openai/gpt-5.4" },
      model_info: { id: "wildcard", mode: "chat" },
    };
    const tiers = resolveModelInfoCatalog(entry)?.cost?.tiers;
    expect(tiers).toBeDefined();
    mockEndpoints({
      "/model/info": () => jsonResponse(200, { data: [entry] }),
      "/v1/models": () => jsonResponse(200, { data: [{ id: "team/assistant" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.cost.tiers).toEqual(tiers);
  });

  it("does not reintroduce a withheld model-info route through wildcard expansion", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            { model_name: "team/*", model_info: { id: "wildcard", mode: "chat" } },
            { model_name: "blocked/model", model_info: { id: "chat", mode: "chat" } },
            { model_name: "blocked/model", model_info: { id: "embedding", mode: "embedding" } },
          ],
        }),
      "/v1/models": () =>
        jsonResponse(200, {
          data: [
            { id: "team/expanded", owned_by: "openai" },
            { id: "blocked/model", owned_by: "openai" },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models.map((model) => model.id)).toEqual(["team/expanded"]);
  });

  it("keeps wildcard tool-repair withholding on the expanded model", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "team/*",
              litellm_params: { model: "moonshot/kimi-k2.6" },
              model_info: { id: "kimi", mode: "chat" },
            },
            { model_name: "team/*", model_info: { id: "unidentified", mode: "chat" } },
          ],
        }),
      "/v1/models": () => jsonResponse(200, { data: [{ id: "team/assistant" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.litellmPolicy).toEqual({
      normalizeStrictToolMessages: false,
      normalizeThinkTags: false,
      suppressReasoningVisibility: false,
    });
    expect(stderr.mock.calls.flat().join(" ")).toContain("team/assistant");
  });

  it("warns when a route-only wildcard expands to a Moonshot id", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, { data: [{ model_name: "*", model_info: { id: "wildcard", mode: "chat" } }] }),
      "/v1/models": () => jsonResponse(200, { data: [{ id: "kimi-wildcard-review" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.litellmPolicy?.normalizeStrictToolMessages).toBe(false);
    expect(stderr.mock.calls.flat().join(" ")).toContain("kimi-wildcard-review");
  });

  it("does not warn for a wildcard candidate already published by an exact route", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            { model_name: "*", model_info: { id: "wildcard", mode: "chat" } },
            {
              model_name: "kimi-k2.6-exact-authority",
              litellm_params: { model: "moonshot/kimi-k2.6", allowed_openai_params: ["thinking"] },
              model_info: { id: "exact", mode: "chat", supports_reasoning: true },
            },
          ],
        }),
      "/v1/models": () => jsonResponse(200, { data: [{ id: "kimi-k2.6-exact-authority" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models).toHaveLength(1);
    expect(result.models[0]).toMatchObject({
      id: "kimi-k2.6-exact-authority",
      litellmPolicy: { normalizeStrictToolMessages: true },
    });
    expect(stderr.mock.calls.flat().join(" ")).not.toContain("kimi-k2.6-exact-authority");
  });

  it.each([
    {
      label: "complete",
      pricing: {
        input_cost_per_token: 0.000001,
        output_cost_per_token: 0.000002,
        cache_read_input_token_cost: 0.0000001,
        cache_creation_input_token_cost: 0.0000002,
      },
      name: "GPT-5.5",
      cost: { input: 1, output: 2, cacheRead: 0.09999999999999999, cacheWrite: 0.19999999999999998 },
    },
    {
      label: "incomplete",
      pricing: {},
      name: "GPT-5.5 (incomplete metadata)",
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    },
  ])("takes only catalog presentation metadata for a $label wildcard child", async ({ pricing, name, cost }) => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "openai/*",
              litellm_params: { model: "internal/proxy" },
              model_info: {
                id: "wildcard",
                mode: "chat",
                supported_openai_params: [],
                supports_reasoning: true,
                supports_vision: false,
                max_input_tokens: 200_000,
                max_output_tokens: 100_000,
                ...pricing,
              },
            },
          ],
        }),
      "/v1/models": () =>
        jsonResponse(200, {
          data: [
            { id: "openai/gpt-5.5", owned_by: "openai" },
            { id: "unmatched/catalog-model", owned_by: "openai" },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models).toHaveLength(1);
    expect(result.models[0]).toMatchObject({
      id: "openai/gpt-5.5",
      name,
      input: ["text"],
      cost,
      contextWindow: 200_000,
      maxTokens: 100_000,
      reasoning: true,
      thinkingLevelMap: NO_REASONING_LEVELS,
    });
    expect(result.models[0]?.compat).not.toMatchObject({ supportsReasoningEffort: true });
  });

  it("keeps parent deployment limits when a wildcard child has smaller catalog limits", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "openai/*",
              model_info: {
                id: "wildcard",
                mode: "chat",
                max_input_tokens: 200_000,
                max_output_tokens: 100_000,
              },
            },
          ],
        }),
      "/v1/models": () => jsonResponse(200, { data: [{ id: "openai/gpt-4o", owned_by: "openai" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      id: "openai/gpt-4o",
      contextWindow: 200_000,
      maxTokens: 100_000,
    });
  });
});

describe("discoverModels response-mode models", () => {
  it("retains upstream automatic API choices and never selects Messages", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          { model_name: "anthropic/claude-sonnet-4-6", model_info: { mode: "chat" } },
          { model_name: "openai/gpt-5.3-codex-openai", model_info: { mode: "responses" } },
          { model_name: "unknown-mode", model_info: { mode: "messages" } },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models.map(({ id, api }) => ({ id, api }))).toEqual([
      { id: "anthropic/claude-sonnet-4-6", api: "openai-completions" },
      { id: "openai/gpt-5.3-codex-openai", api: "openai-responses" },
    ]);
    expect(result.models.map((model) => model.api)).not.toContain("anthropic-messages");
  });

  it("keeps /model/info response-mode models with Responses-specific compatibility", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "openai/gpt-5.3-codex-openai",
              model_info: {
                mode: "responses",
                max_input_tokens: 272000,
                max_output_tokens: 128000,
              },
            },
            { model_name: "anthropic/claude-sonnet-4-6", model_info: { mode: "responses" } },
            { model_name: "sonnet-4.6", model_info: { mode: "responses" } },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.source).toBe("model_info");
    expect(result.models).toHaveLength(3);
    expect(result.models[0]).toMatchObject({
      id: "openai/gpt-5.3-codex-openai",
      api: "openai-responses",
      contextWindow: 272000,
      maxTokens: 128000,
    });
    for (const id of ["anthropic/claude-sonnet-4-6", "sonnet-4.6"]) {
      expect(result.models.find((model) => model.id === id)).toMatchObject({
        api: "openai-responses",
        compat: undefined,
      });
    }
  });

  it("keeps a /health route on Chat when any of its deployments needs Chat, regardless of order", async () => {
    const detail = (id: string, apiVersion?: string) => ({
      model_name: "shared-route",
      litellm_params: { model: "azure/gpt-5.5", ...(apiVersion ? { api_version: apiVersion } : {}) },
      model_info: { id, litellm_provider: "azure" },
    });
    const run = async (order: string[]) => {
      vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
        const url = input instanceof URL ? input.toString() : String(input);
        if (url.endsWith("/model/info")) return jsonResponse(404, {});
        if (url.endsWith("/v1/models")) return jsonResponse(404, {});
        if (url.endsWith("/health")) {
          return jsonResponse(200, {
            healthy_endpoints: order.map((id) => ({ model: "azure/gpt-5.5", model_id: id })),
          });
        }
        if (url.endsWith("/model/info?litellm_model_id=new")) return jsonResponse(200, { data: [detail("new")] });
        if (url.endsWith("/model/info?litellm_model_id=old")) {
          return jsonResponse(200, { data: [detail("old", "2024-10-21")] });
        }
        throw new Error(`unexpected URL: ${url}`);
      });
      const result = await discoverModels("https://litellm.example.com", "sk-test", {});
      expect(result.source).toBe("health");
      return result.models.find((model) => model.id === "shared-route");
    };

    expect(await run(["new", "old"])).toMatchObject({ api: "openai-completions" });
    expect(await run(["old", "new"])).toMatchObject({ api: "openai-completions" });
  });

  it("drops the backend family when /health deployments of one route disagree", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) return jsonResponse(404, {});
      if (url.endsWith("/v1/models")) return jsonResponse(404, {});
      if (url.endsWith("/health")) {
        return jsonResponse(200, {
          healthy_endpoints: [
            { model: "openai/gpt-5.5", model_id: "gpt" },
            { model: "anthropic/claude-sonnet-4-6", model_id: "claude" },
          ],
        });
      }
      if (url.endsWith("/model/info?litellm_model_id=gpt")) {
        return jsonResponse(200, {
          data: [{ model_name: "mixed-route", litellm_params: { model: "openai/gpt-5.5" }, model_info: { id: "gpt" } }],
        });
      }
      if (url.endsWith("/model/info?litellm_model_id=claude")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "mixed-route",
              litellm_params: { model: "anthropic/claude-sonnet-4-6" },
              model_info: { id: "claude" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    const mixed = result.models.find((model) => model.id === "mixed-route");
    expect(mixed).toMatchObject({ api: "openai-completions" });
    expect(mixed).not.toHaveProperty("litellmBackendFamily");
  });

  it("keeps /health response-mode model_info fallbacks with a Responses API override", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) return jsonResponse(404, {});
      if (url.endsWith("/v1/models")) return jsonResponse(404, {});
      if (url.endsWith("/health")) {
        return jsonResponse(200, {
          healthy_endpoints: [{ model: "openai/gpt-5.3-codex-openai", model_id: "uuid-1" }],
        });
      }
      if (url.endsWith("/model/info?litellm_model_id=uuid-1")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "openai/gpt-5.3-codex-openai",
              model_info: { mode: "response" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.source).toBe("health");
    expect(result.models).toHaveLength(1);
    expect(result.models[0]).toMatchObject({
      id: "openai/gpt-5.3-codex-openai",
      api: "openai-responses",
    });
  });

  it("does not derive thinking controls from a health-only route name", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return jsonResponse(404, {});
      if (url.endsWith("/v1/models")) return jsonResponse(404, {});
      if (url.endsWith("/health")) {
        return jsonResponse(200, { healthy_endpoints: [{ model: "openai/gpt-5.5", model_id: "uuid-1" }] });
      }
      if (url.endsWith("/model/info?litellm_model_id=uuid-1")) {
        return jsonResponse(200, { data: [{ model_info: { mode: "chat" } }] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.source).toBe("health");
    expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
  });

  it("uses Pi catalog protocol and presentation for a health endpoint without deployment detail", async () => {
    mockEndpoints({
      "/model/info": () => jsonResponse(404, {}),
      "/v1/models": () => jsonResponse(404, {}),
      "/health": () => jsonResponse(200, { healthy_endpoints: [{ model: "openai/gpt-5.5" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });
    const catalog = getModel("openai", "gpt-5.5");

    expect(result.source).toBe("health");
    expect(result.models[0]).toMatchObject({
      id: "openai/gpt-5.5",
      name: catalog.name,
      api: "openai-responses",
      reasoning: catalog.reasoning,
      thinkingLevelMap: NO_REASONING_LEVELS,
      input: catalog.input,
      cost: catalog.cost,
      contextWindow: catalog.contextWindow,
      maxTokens: catalog.maxTokens,
    });
    expect(result.models[0]).not.toHaveProperty("litellmBackendFamily");
  });

  it("keeps an unknown health-only id on Chat", async () => {
    mockEndpoints({
      "/model/info": () => jsonResponse(404, {}),
      "/v1/models": () => jsonResponse(404, {}),
      "/health": () => jsonResponse(200, { healthy_endpoints: [{ model: "unknown-health-route" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]).toMatchObject({
      id: "unknown-health-route",
      name: "unknown-health-route (no metadata)",
      api: "openai-completions",
    });
  });
  it("diagnoses a /health route that mixes chat and incompatible deployment modes", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    mockEndpoints({
      "/model/info?litellm_model_id=chat-id": () =>
        jsonResponse(200, { data: [{ model_name: "health-mixed-mode", model_info: { mode: "chat" } }] }),
      "/model/info?litellm_model_id=embed-id": () =>
        jsonResponse(200, { data: [{ model_name: "health-mixed-mode", model_info: { mode: "embedding" } }] }),
      "/model/info": () => jsonResponse(404, {}),
      "/v1/models": () => jsonResponse(404, {}),
      "/health": () =>
        jsonResponse(200, {
          healthy_endpoints: [
            { model: "health-mixed-mode", model_id: "chat-id" },
            { model: "health-mixed-mode", model_id: "embed-id" },
          ],
        }),
    });

    await expect(discoverModels("https://litellm.example.com", "sk-test", {})).rejects.toThrow(
      "/v1/models returned 404",
    );

    expect(stderr).toHaveBeenCalledTimes(1);
    expect(String(stderr.mock.calls[0]?.[0])).toContain("health-mixed-mode");
    expect(String(stderr.mock.calls[0]?.[0])).toContain("explicitly incompatible deployment modes");
  });

  it("uses the correlated detail model_name instead of the health backend model", async () => {
    mockEndpoints({
      "/model/info?litellm_model_id=uuid-redirect": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "different-route",
              litellm_params: { model: "moonshot/kimi-k3", allowed_openai_params: ["reasoning_effort"] },
              model_info: { id: "uuid-redirect", mode: "chat", supports_reasoning: true },
            },
          ],
        }),
      "/model/info": () => jsonResponse(404, {}),
      "/v1/models": () => jsonResponse(404, {}),
      "/health": () =>
        jsonResponse(200, { healthy_endpoints: [{ model: "authoritative-route", model_id: "uuid-redirect" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({ id: "different-route", reasoning: true });
    expect(result.models[0]?.thinkingLevelMap).toEqual({
      off: null,
      minimal: null,
      low: "low",
      medium: null,
      high: "high",
      xhigh: null,
      max: null,
    });
  });

  it.each([
    ["openai/gpt-5.5", "openai-responses"],
    ["unknown/health-route", "openai-completions"],
  ])("uses catalog protocol %s for health entries without deployment detail", async (route, api) => {
    mockEndpoints({
      "/model/info": () => jsonResponse(404, {}),
      "/v1/models": () => jsonResponse(404, {}),
      "/health": () => jsonResponse(200, { healthy_endpoints: [{ model: route }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.source).toBe("health");
    expect(result.models[0]).toMatchObject({ id: route, api });
    if (result.models[0]?.reasoning) expect(result.models[0]?.thinkingLevelMap).toEqual(NO_REASONING_LEVELS);
    else expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
  });
});

describe("catalog provider candidates", () => {
  it.each([
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-fable-5",
    "claude-opus-4-5-20251101",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "opus-4-7",
    "sonnet-4-6",
    "haiku-4-5",
    "opus-4.7",
    "fable-5",
    "opus-5",
  ])("resolves the bare Anthropic backend id %s from the shared lookup rule", async (id) => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [{ model_name: `route-${id}`, litellm_params: { model: id }, model_info: { mode: "chat" } }],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.name).not.toContain("metadata");
    expect(result.models[0]?.cost.input).toBeGreaterThan(0);
  });

  it.each(["claudia-x", "opusclip-2", "haiku"])("does not treat %s as an Anthropic alias", async (id) => {
    mockEndpoints({
      "/model/info": () => jsonResponse(200, { data: [{ model_name: id, model_info: { mode: "chat" } }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.name).toBe(`${id} (incomplete metadata)`);
    expect(result.models[0]?.cost).toEqual({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });
  });
  it.each([
    ["decorated mixed-case Kimi", "ToGeThEr/MoOnShOtAi/KiMi-K2.6@prod", "together_ai", 262_144],
    ["mixed-case Claude", "AnThRoPiC/ClAuDe-SoNnEt-4-6", undefined, 1_000_000],
    ["decorated Claude", "BeDrOcK/US.AnThRoPiC.ClAuDe-SoNnEt-4-6-V1:0", "bedrock", 1_000_000],
  ])("retains catalog metadata for a %s provider-qualified backend", async (_case, backend, adapter, context) => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "private/catalog-route",
              litellm_params: { model: backend },
              model_info: { mode: "chat", ...(adapter ? { litellm_provider: adapter } : {}) },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      id: "private/catalog-route",
      name: "private/catalog-route",
      contextWindow: context,
      maxTokens: expect.any(Number),
    });
    expect(result.models[0]?.cost.input).toBeGreaterThan(0);
  });

  it("does not strip arbitrary suffixes while normalizing provider-qualified ids", () => {
    const resolved = resolveModelInfoCatalog({
      model_name: "private/catalog-route",
      litellm_params: { model: "anthropic/Claude-Sonnet-4-6-preview" },
      model_info: { mode: "chat" },
    });

    expect(resolved).toEqual({ provider: "anthropic", catalogModelId: "anthropic/Claude-Sonnet-4-6-preview" });
    expect(resolved).not.toHaveProperty("cost");
  });
});

describe("discoverModels fallback to /v1/models", () => {
  it("keeps unqualified fallback ids bounded instead of scanning every provider catalog", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, { data: [{ id: "gpt-4o" }, { id: "kimi-k2.6" }, { id: "grok-4.5" }] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: "gpt-4o", name: "gpt-4o (no metadata)" }),
        expect.objectContaining({ id: "kimi-k2.6", name: "kimi-k2.6 (no metadata)" }),
        expect.objectContaining({ id: "grok-4.5", name: "grok-4.5 (no metadata)" }),
      ]),
    );
  });

  for (const status of [401, 403, 404]) {
    it(`falls back when /model/info returns ${status}`, async () => {
      vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
        const url = input instanceof URL ? input.toString() : String(input);
        if (url.endsWith("/model/info")) return new Response(null, { status });
        if (url.endsWith("/v1/models")) {
          return jsonResponse(200, {
            data: [{ id: "openai/gpt-4o" }, { id: "anthropic/claude-3-5-sonnet" }],
          });
        }
        throw new Error(`unexpected URL: ${url}`);
      });
      const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });
      expect(result.source).toBe("models_list");
      expect(result.models.map((m) => m.id).sort()).toEqual(["anthropic/claude-3-5-sonnet", "openai/gpt-4o"]);
      const anthropic = result.models.find((m) => m.id === "anthropic/claude-3-5-sonnet")!;
      expect(anthropic.name).toBe("anthropic/claude-3-5-sonnet (no metadata)");
      expect(anthropic.compat).toEqual({ supportsStore: false, cacheControlFormat: "anthropic" });
    });
  }

  it("uses Pi catalog metadata for the /v1/models fallback", async () => {
    const urls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      urls.push(url);
      if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, {
          data: [{ id: "gpt-5.5", object: "model", owned_by: "openai" }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]).toMatchObject({
      id: "gpt-5.5",
      name: "GPT-5.5",
      contextWindow: 272000,
      api: "openai-responses",
    });
  });

  it("keeps an opaque /v1/models fallback id on Chat", async () => {
    mockEndpoints({
      "/model/info": () => jsonResponse(403, {}),
      "/v1/models": () => jsonResponse(200, { data: [{ id: "opaque-fallback-route" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]).toMatchObject({
      id: "opaque-fallback-route",
      name: "opaque-fallback-route (no metadata)",
      api: "openai-completions",
    });
  });

  it("keeps an opaque /v1/models fallback id on Chat Completions without a backend family", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) return new Response(null, { status: 401 });
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, { data: [{ id: "gpt-production", object: "model", owned_by: "openai" }] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    // The id spells like an OpenAI model, but only a deployment row can show the adapter and
    // Azure API version that decide Responses eligibility.
    expect(result.models[0]).toMatchObject({
      id: "gpt-production",
      name: "gpt-production (no metadata)",
      api: "openai-completions",
    });
    expect(result.models[0]).not.toHaveProperty("litellmBackendFamily");
  });

  it("enriches a bare Fable 5 fallback model from the Pi catalog without inferring a carrier", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, {
          data: [{ id: "fable-5", object: "model", owned_by: "openai" }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result).toMatchObject({
      source: "models_list",
      models: [{ id: "fable-5", name: "Claude Fable 5", reasoning: true }],
    });
    expect(result.models[0]?.thinkingLevelMap).toEqual(NO_REASONING_LEVELS);
    expect(result.models[0]?.compat).not.toMatchObject({ supportsReasoningEffort: true });
  });

  it("enriches a bare Opus 5 fallback model from the Pi catalog without inferring a carrier", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, {
          data: [{ id: "opus-5", object: "model", owned_by: "openai" }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result).toMatchObject({
      source: "models_list",
      models: [{ id: "opus-5", name: "Claude Opus 5", reasoning: true }],
    });
    expect(result.models[0]?.thinkingLevelMap).toEqual(NO_REASONING_LEVELS);
    expect(result.models[0]?.compat).not.toMatchObject({ supportsReasoningEffort: true });
  });

  it("throws when /model/info returns a non-401/403/404 error", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(null, { status: 500 }));
    await expect(discoverModels("https://litellm.example.com", "sk-test", {})).rejects.toThrow(/500/);
  });
  it("uses Pi catalog protocol for fallback entries while denying reasoning carriers", async () => {
    mockEndpoints({
      "/model/info": () => jsonResponse(404, {}),
      "/v1/models": () => jsonResponse(200, { data: [{ id: "openai/gpt-5.5", owned_by: "openai" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      api: "openai-responses",
      reasoning: true,
      thinkingLevelMap: NO_REASONING_LEVELS,
    });
    expect(result.models[0]?.compat).not.toMatchObject({ supportsReasoningEffort: true });
  });

  it("denies Pi default Chat levels when /v1/models catalog metadata has no level or carrier evidence", async () => {
    mockEndpoints({
      "/model/info": () => new Response(null, { status: 403 }),
      "/v1/models": () =>
        jsonResponse(200, {
          data: [{ id: "claude-haiku-4-5", object: "model", owned_by: "anthropic" }],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result).toMatchObject({
      source: "models_list",
      models: [
        {
          id: "claude-haiku-4-5",
          reasoning: true,
          thinkingLevelMap: NO_REASONING_LEVELS,
        },
      ],
    });
  });
});

describe("discoverModels fallback to /health", () => {
  it("bounds health detail concurrency while preserving endpoint order and completion progress", async () => {
    const endpoints = Array.from({ length: 11 }, (_, index) => ({
      model: `model-${index + 1}`,
      model_id: `uuid-${index + 1}`,
    }));
    const pending = new Map<string, () => void>();
    const progress = vi.fn();
    let active = 0;
    let maxActive = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return Promise.resolve(new Response(null, { status: 404 }));
      if (url.endsWith("/v1/models")) return Promise.resolve(new Response(null, { status: 404 }));
      if (url.endsWith("/health")) return Promise.resolve(jsonResponse(200, { healthy_endpoints: endpoints }));
      const deploymentId = new URL(url).searchParams.get("litellm_model_id");
      if (!deploymentId) throw new Error(`unexpected URL: ${url}`);
      const modelNumber = deploymentId.slice("uuid-".length);
      active++;
      maxActive = Math.max(maxActive, active);
      return new Promise<Response>((resolve) => {
        pending.set(deploymentId, () => {
          active--;
          resolve(jsonResponse(200, { data: [{ model_name: `model-${modelNumber}`, model_info: { mode: "chat" } }] }));
        });
      });
    });
    const release = (deploymentId: string): void => {
      const resolve = pending.get(deploymentId);
      if (!resolve) throw new Error(`${deploymentId} is not pending`);
      pending.delete(deploymentId);
      resolve();
    };

    const discovery = discoverModels("https://litellm.example.com", "sk-test", { onProgress: progress });
    await vi.waitFor(() => expect([...pending.keys()]).toEqual(endpoints.slice(0, 8).map(({ model_id }) => model_id)));
    expect(maxActive).toBe(8);

    release("uuid-8");
    await vi.waitFor(() => expect(pending.has("uuid-9")).toBe(true));
    release("uuid-7");
    await vi.waitFor(() => expect(pending.has("uuid-10")).toBe(true));
    release("uuid-6");
    await vi.waitFor(() => expect(pending.has("uuid-11")).toBe(true));
    expect(progress).not.toHaveBeenCalledWith("Fetched 10/11 models...");
    expect(maxActive).toBe(8);

    for (const deploymentId of ["uuid-11", "uuid-10", "uuid-9", "uuid-5", "uuid-4", "uuid-3", "uuid-2", "uuid-1"]) {
      release(deploymentId);
    }
    const result = await discovery;

    expect(result.models.map((model) => model.id)).toEqual(endpoints.map(({ model }) => model));
    expect(progress).toHaveBeenCalledWith("Fetched 10/11 models...");
    expect(progress).toHaveBeenCalledWith("Fetched 11/11 models...");
  });

  it("uses /health and per-endpoint /model/info when OpenAI model listing is unavailable", async () => {
    const urls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      urls.push(url);
      if (url.endsWith("/model/info")) return new Response(null, { status: 404 });
      if (url.endsWith("/v1/models")) return new Response(null, { status: 404 });
      if (url.endsWith("/health")) {
        return jsonResponse(200, {
          healthy_endpoints: [
            { model: "vertex/claude-sonnet", model_id: "uuid-1" },
            { model: "openai/gpt-4o-mini", model_id: "uuid-2" },
          ],
        });
      }
      if (url.endsWith("/model/info?litellm_model_id=uuid-1")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "vertex/claude-sonnet",
              model_info: {
                mode: "chat",
                max_input_tokens: 200000,
                supports_vision: true,
                input_cost_per_token: 0.000003,
                output_cost_per_token: 0.000015,
              },
            },
          ],
        });
      }
      if (url.endsWith("/model/info?litellm_model_id=uuid-2")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "openai/gpt-4o-mini",
              model_info: {
                mode: "chat",
                max_input_tokens: 128000,
                max_output_tokens: 16384,
              },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(urls).toEqual([
      "https://litellm.example.com/model/info",
      "https://litellm.example.com/v1/models",
      "https://litellm.example.com/health",
      "https://litellm.example.com/model/info?litellm_model_id=uuid-1",
      "https://litellm.example.com/model/info?litellm_model_id=uuid-2",
    ]);
    expect(result.source).toBe("health");
    expect(result.models.map((model) => model.id)).toEqual(["vertex/claude-sonnet", "openai/gpt-4o-mini"]);
    expect(result.models[0]).toMatchObject({
      input: ["text", "image"],
      contextWindow: 200000,
      compat: { supportsStore: false, cacheControlFormat: "anthropic" },
    });
  });

  it("uses healthy endpoint model names when /health entries do not include model ids", async () => {
    const urls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      urls.push(url);
      if (url.endsWith("/model/info")) return new Response(null, { status: 404 });
      if (url.endsWith("/v1/models")) return new Response(null, { status: 404 });
      if (url.endsWith("/health")) {
        return jsonResponse(200, {
          healthy_endpoints: [
            { model: "azure/gpt-35-turbo", api_base: "https://azure.example.com" },
            { model: "anthropic/claude-3-5-sonnet", api_base: "https://anthropic.example.com" },
            { model: "openai/gpt-5.5", api_base: "https://openai.example.com" },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(urls).toEqual([
      "https://litellm.example.com/model/info",
      "https://litellm.example.com/v1/models",
      "https://litellm.example.com/health",
    ]);
    expect(result.source).toBe("health");
    expect(result.models.map((model) => model.id)).toEqual([
      "azure/gpt-35-turbo",
      "anthropic/claude-3-5-sonnet",
      "openai/gpt-5.5",
    ]);
    expect(result.models[1]).toMatchObject({
      // Neither route resolves in the Pi catalog, so both are evidence-free and
      // must say so rather than presenting default limits and zero cost as fact.
      name: "anthropic/claude-3-5-sonnet (no metadata)",
      contextWindow: 128000,
      maxTokens: 16384,
      compat: { supportsStore: false, cacheControlFormat: "anthropic" },
    });
    expect(result.models[0]?.name).toBe("azure/gpt-35-turbo (no metadata)");
  });

  it("uses bounded Pi metadata for health-only routes", async () => {
    // Bare health entries use the same bounded Pi lookup as `/v1/models`.
    mockEndpoints({
      "/model/info": () => jsonResponse(404, {}),
      "/v1/models": () => jsonResponse(404, {}),
      "/health": () =>
        jsonResponse(200, {
          healthy_endpoints: [{ model: "totally-unknown-route" }, { model: "anthropic/claude-opus-4-7" }],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.source).toBe("health");
    const [unresolved, resolved] = result.models;
    expect(unresolved).toMatchObject({
      id: "totally-unknown-route",
      name: "totally-unknown-route (no metadata)",
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128_000,
      maxTokens: 16_384,
    });
    expect(unresolved?.name).toContain(" (no metadata)");
    expect(resolved).toMatchObject({
      name: getModel("anthropic", "claude-opus-4-7").name,
      api: "openai-completions",
      cost: getModel("anthropic", "claude-opus-4-7").cost,
    });
  });

  it("keeps mixed detailed and bare health rows conservative", async () => {
    mockEndpoints({
      "/model/info": () => jsonResponse(404, {}),
      "/v1/models": () => jsonResponse(404, {}),
      "/health": () =>
        jsonResponse(200, {
          healthy_endpoints: [{ model: "openai/gpt-5.5", model_id: "detail" }, { model: "openai/gpt-5.5" }],
        }),
      "/model/info?litellm_model_id=detail": () =>
        jsonResponse(200, {
          data: [{ model_name: "openai/gpt-5.5", litellm_params: { model: "private/unknown" } }],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]).toMatchObject({
      name: "openai/gpt-5.5 (incomplete metadata)",
      reasoning: false,
      contextWindow: 128_000,
      maxTokens: 16_384,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    });
  });

  it("marks an unresolved health route reached through per-endpoint /model/info", async () => {
    mockEndpoints({
      "/model/info?litellm_model_id=uuid-1": () => jsonResponse(200, { data: [{ model_info: { mode: "chat" } }] }),
      "/model/info": () => jsonResponse(404, {}),
      "/v1/models": () => jsonResponse(404, {}),
      "/health": () =>
        jsonResponse(200, { healthy_endpoints: [{ model: "vertex/claude-sonnet", model_id: "uuid-1" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.source).toBe("health");
    expect(result.models[0]?.name).toBe("vertex/claude-sonnet (incomplete metadata)");
    expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
  });
  it.each([
    ["uuid-roomy", "uuid-cramped"],
    ["uuid-cramped", "uuid-roomy"],
  ])("groups healthy deployments by route before publication in order %j", async (...endpointIds) => {
    mockEndpoints({
      "/model/info?litellm_model_id=uuid-roomy": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "shared-health-route",
              litellm_params: { model: "openai/gpt-4o" },
              model_info: {
                id: "roomy",
                mode: "responses",
                supports_reasoning: true,
                supports_vision: true,
                max_input_tokens: 128_000,
                max_output_tokens: 16_384,
                input_cost_per_token: 0.000005,
                output_cost_per_token: 0.000015,
                cache_read_input_token_cost: 0.0000025,
                cache_creation_input_token_cost: 0,
              },
            },
          ],
        }),
      "/model/info?litellm_model_id=uuid-cramped": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "shared-health-route",
              litellm_params: { model: "internal/unknown" },
              model_info: {
                id: "cramped",
                mode: "chat",
                supports_reasoning: false,
                supports_vision: false,
                max_input_tokens: 64_000,
                max_output_tokens: 8_192,
              },
            },
          ],
        }),
      "/model/info": () => jsonResponse(404, {}),
      "/v1/models": () => jsonResponse(404, {}),
      "/health": () =>
        jsonResponse(200, {
          healthy_endpoints: endpointIds.map((model_id) => ({ model: "shared-health-route", model_id })),
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.source).toBe("health");
    expect(result.models).toHaveLength(1);
    expect(result.models[0]).toMatchObject({
      id: "shared-health-route",
      name: "shared-health-route (incomplete metadata)",
      api: "openai-completions",
      reasoning: false,
      input: ["text"],
      contextWindow: 64_000,
      maxTokens: 8_192,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    });
    expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
  });

  it("groups health backends that correlate to one detail model_name", async () => {
    mockEndpoints({
      "/model/info?litellm_model_id=bedrock-a": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "claude-opus-5",
              litellm_params: { model: "bedrock/us.anthropic.claude-opus-5-a" },
              model_info: { id: "bedrock-a", mode: "chat" },
            },
          ],
        }),
      "/model/info?litellm_model_id=bedrock-b": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "claude-opus-5",
              litellm_params: { model: "bedrock/eu.anthropic.claude-opus-5-b" },
              model_info: { id: "bedrock-b", mode: "chat" },
            },
          ],
        }),
      "/model/info": () => jsonResponse(404, {}),
      "/v1/models": () => jsonResponse(404, {}),
      "/health": () =>
        jsonResponse(200, {
          healthy_endpoints: [
            { model: "bedrock/us.anthropic.claude-opus-5", model_id: "bedrock-a" },
            { model: "bedrock/eu.anthropic.claude-opus-5", model_id: "bedrock-b" },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.source).toBe("health");
    expect(result.models).toHaveLength(1);
    expect(result.models[0]?.id).toBe("claude-opus-5");
  });
});

describe("discoverModels timeout", () => {
  it("aborts the fetch after timeoutMs", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((_input, init) => {
      return new Promise((_resolve, reject) => {
        const signal = (init as { signal?: AbortSignal } | undefined)?.signal;
        signal?.addEventListener("abort", () => reject(signal.reason ?? new Error("aborted")));
      });
    });
    const start = Date.now();
    await expect(discoverModels("https://litellm.example.com", "sk-test", { timeoutMs: 30 })).rejects.toBeDefined();
    expect(Date.now() - start).toBeLessThan(500);
  });
});

describe("native Messages discovery", () => {
  it.each([
    { flags: [null], low: true, max: true },
    { flags: [true, undefined], low: true, max: false },
    { flags: [true, null], low: true, max: false },
    { flags: [null, undefined], low: true, max: true },
    { flags: [false, undefined], low: false, max: false },
    { flags: [true, true], low: true, max: true },
  ])("treats native reasoning flags $flags as boolean evidence", async ({ flags, low, max }) => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: flags.map((flag, index) => ({
            model_name: "native-flag-evidence",
            litellm_params: { model: "anthropic/claude-sonnet-4-6" },
            model_info: {
              id: String(index),
              mode: "chat",
              litellm_provider: "anthropic",
              supports_low_reasoning_effort: flag,
              supports_max_reasoning_effort: flag,
            },
          })),
        }),
    });
    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });
    const model = { ...result.models[0]!, provider: "litellm", baseUrl: "https://proxy.example.com" };
    expect(model.api).toBe("anthropic-messages");
    const levels = getSupportedThinkingLevels(model);
    expect(levels.includes("low")).toBe(low);
    expect(levels.includes("max")).toBe(max);
  });

  it.each(["claude-sonnet-4-6", "claude-opus-4-5"])(
    "honors denied default native reasoning levels for %s",
    async (backend) => {
      mockEndpoints({
        "/model/info": () =>
          jsonResponse(200, {
            data: [
              {
                model_name: "native-levels",
                litellm_params: { model: `anthropic/${backend}` },
                model_info: {
                  mode: "chat",
                  litellm_provider: "anthropic",
                  supports_none_reasoning_effort: false,
                  supports_low_reasoning_effort: false,
                },
              },
            ],
          }),
      });
      const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });
      const model = { ...result.models[0]!, provider: "litellm", baseUrl: "https://proxy.example.com" };
      expect(model.api).toBe("anthropic-messages");
      expect(getSupportedThinkingLevels(model)).not.toContain("off");
      expect(getSupportedThinkingLevels(model)).not.toContain("low");
      expect(getSupportedThinkingLevels(model)).toContain("medium");
    },
  );

  it("retains native reasoning denials across different Claude generations", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: ["claude-sonnet-4-6", "claude-fable-5"].map((backend) => ({
            model_name: "mixed-native-levels",
            litellm_params: { model: `anthropic/${backend}` },
            model_info: { id: backend, mode: "chat", litellm_provider: "anthropic", supports_reasoning: true },
          })),
        }),
    });
    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });
    const model = { ...result.models[0]!, provider: "litellm", baseUrl: "https://proxy.example.com" };
    expect(model.api).toBe("anthropic-messages");
    expect(getSupportedThinkingLevels(model)).not.toContain("off");
    expect(getSupportedThinkingLevels(model)).toContain("high");
  });

  it("uses Anthropic catalog metadata and Messages for the standard Vertex Claude adapter", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "vertex-claude",
              litellm_params: { model: "vertex_ai/claude-opus-4-5" },
              model_info: { mode: "chat", litellm_provider: "vertex_ai-anthropic_models" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models).toEqual([
      expect.objectContaining({
        id: "vertex-claude",
        name: "vertex-claude",
        reasoning: true,
        contextWindow: 200_000,
        maxTokens: 64_000,
        cost: { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
        api: "anthropic-messages",
      }),
    ]);
  });

  it("uses Anthropic catalog metadata and Messages for the vertex_ai adapter alias", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "vertex-claude",
              litellm_params: { model: "vertex_ai/claude-opus-4-5" },
              model_info: { mode: "chat", litellm_provider: "vertex_ai" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]).toMatchObject({
      id: "vertex-claude",
      name: "vertex-claude",
      reasoning: true,
      contextWindow: 200_000,
      maxTokens: 64_000,
      cost: { input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 },
      api: "anthropic-messages",
    });
  });

  it("withholds native Messages when the Vertex Anthropic adapter conflicts with the routed provider", () => {
    const resolved = resolveModelInfoCatalog({
      model_name: "contradictory-adapter",
      litellm_params: { model: "openai/gpt-4o" },
      model_info: { mode: "chat", litellm_provider: "vertex_ai-anthropic_models" },
    });

    expect(resolved).not.toHaveProperty("messagesCompat");
  });

  it("withholds native Messages when a recognized adapter conflicts with the routed provider", () => {
    const resolved = resolveModelInfoCatalog({
      model_name: "contradictory-adapter",
      litellm_params: { model: "anthropic/claude-sonnet-4-6" },
      model_info: { mode: "chat", litellm_provider: "openai" },
    });

    expect(resolved).not.toHaveProperty("messagesCompat");
  });

  it("withholds native Messages policy for conflicting Claude generations", () => {
    const resolved = resolveModelInfoCatalog({
      model_name: "contradictory-claude-generations",
      litellm_params: { model: "anthropic/claude-opus-4-7" },
      model_info: {
        mode: "chat",
        litellm_provider: "anthropic",
        base_model: "anthropic/claude-opus-4-5",
      },
    });

    expect(resolved).not.toHaveProperty("messagesCompat");
  });

  it("withholds native Messages when any declared Claude backend lacks a compatible policy", () => {
    const resolved = resolveModelInfoCatalog({
      model_name: "partially-resolved-claude-generations",
      litellm_params: { model: "anthropic/claude-opus-4-7" },
      model_info: {
        mode: "chat",
        litellm_provider: "anthropic",
        base_model: "anthropic/claude-future-9-9",
      },
    });

    expect(resolved).toMatchObject({ provider: "anthropic", catalogModelId: "anthropic/claude-future-9-9" });
    expect(resolved).not.toHaveProperty("cost");
    expect(resolved).not.toHaveProperty("reasoning");
    expect(resolved).not.toHaveProperty("contextWindow");
    expect(resolved).not.toHaveProperty("maxTokens");
    expect(resolved).not.toHaveProperty("messagesCompat");
  });

  it("downgrades health-derived Messages when an unreadable detail name uses the route fallback", async () => {
    mockEndpoints({
      "/model/info": () => jsonResponse(404, {}),
      "/v1/models": () => jsonResponse(404, {}),
      "/health": () => jsonResponse(200, { healthy_endpoints: [{ model: "team-claude", model_id: "messages-uuid" }] }),
      "/model/info?litellm_model_id=messages-uuid": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: 7,
              model_info: { mode: "chat", litellm_provider: "anthropic" },
              litellm_params: { model: "anthropic/claude-opus-5" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models).toEqual([
      expect.objectContaining({
        id: "team-claude",
        api: "openai-completions",
        compat: {
          supportsStore: false,
          supportsReasoningEffort: false,
          cacheControlFormat: "anthropic",
        },
      }),
    ]);
  });

  it("rechecks Chat reasoning levels when health discovery downgrades Messages", async () => {
    mockEndpoints({
      "/model/info": () => jsonResponse(404, {}),
      "/v1/models": () => jsonResponse(404, {}),
      "/health": () => jsonResponse(200, { healthy_endpoints: [{ model: "claude-route", model_id: "claude-id" }] }),
      "/model/info?litellm_model_id=claude-id": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "claude-route",
              litellm_params: { model: "anthropic/claude-opus-5" },
              model_info: {
                mode: "chat",
                litellm_provider: "anthropic",
                supports_reasoning: true,
                supports_low_reasoning_effort: true,
                supported_openai_params: ["reasoning_effort"],
              },
            },
          ],
        }),
    });
    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });
    const model = { ...result.models[0]!, provider: "litellm", baseUrl: "https://proxy.example.com/v1" };
    expect(model).toMatchObject({ api: "openai-completions", compat: { supportsReasoningEffort: true } });
    const levels = getSupportedThinkingLevels(model);
    expect(levels).toContain("low");
    expect(levels).not.toContain("xhigh");
    expect(levels).not.toContain("max");
  });

  it("never selects native Messages from /health, even with complete matching detail", async () => {
    mockEndpoints({
      "/model/info?litellm_model_id=uuid-claude": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "claude-route",
              model_info: {
                id: "uuid-claude",
                mode: "chat",
                litellm_provider: "anthropic",
                supported_openai_params: [],
                supports_reasoning: true,
                supports_high_reasoning_effort: true,
              },
              litellm_params: { model: "anthropic/claude-sonnet-4-6" },
            },
          ],
        }),
      "/model/info": () => jsonResponse(404, {}),
      "/v1/models": () => jsonResponse(404, {}),
      "/health": () => jsonResponse(200, { healthy_endpoints: [{ model: "claude-route", model_id: "uuid-claude" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.source).toBe("health");
    expect(result.models[0]).toMatchObject({
      id: "claude-route",
      api: "openai-completions",
      compat: {
        supportsStore: false,
        cacheControlFormat: "anthropic",
      },
      thinkingLevelMap: {
        off: null,
        minimal: null,
        low: null,
        medium: null,
        high: null,
        xhigh: null,
        max: null,
      },
    });
    expect(
      getSupportedThinkingLevels({
        ...result.models[0]!,
        provider: "litellm",
        baseUrl: "https://proxy.example.com/v1",
      }),
    ).toEqual([]);
  });

  it("uses Claude base_model evidence when every declared adapter target resolves the same policy", () => {
    expect(
      resolveModelInfoCatalog({
        model_name: "qualified-route",
        litellm_params: { model: "bedrock/us.anthropic.claude-sonnet-4-6-v1:0" },
        model_info: {
          mode: "chat",
          litellm_provider: "bedrock",
          base_model: "bedrock/us.anthropic.claude-sonnet-4-6-v1:0",
        },
      }),
    ).toMatchObject({ messagesCompat: expect.anything() });
  });
});

describe("Moonshot transport suppression", () => {
  it.each([
    [
      "Azure-hosted Kimi",
      {
        model_name: "kimi-k3",
        litellm_params: { model: "azure/FW-Kimi-K3" },
        model_info: {
          mode: "chat",
          base_model: "fireworks/accounts/fireworks/models/kimi-k3",
          litellm_provider: "azure",
        },
      },
      false,
    ],
    [
      "Bedrock-hosted Kimi",
      {
        model_name: "moonshotai-kimi-k2-5",
        litellm_params: { model: "bedrock/moonshotai.kimi-k2.5" },
        model_info: {
          mode: "chat",
          base_model: "moonshotai.kimi-k2.5",
          litellm_provider: "bedrock_converse",
        },
      },
      false,
    ],
    [
      "opaque Moonshot route",
      {
        model_name: "k3-prod",
        litellm_params: { model: "moonshot/kimi-k2.5" },
        model_info: { mode: "chat" },
      },
      true,
    ],
    [
      "conflicting Moonshot provider and Azure backend",
      {
        model_name: "kimi-prod",
        litellm_params: { custom_llm_provider: "moonshot", model: "azure_ai/FW-Kimi-K3" },
        model_info: { mode: "chat" },
      },
      false,
    ],
  ] as const)("keeps request-side reasoning suppression transport-scoped for %s", async (_name, entry, suppress) => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { data: [entry] }));

    const result = await discoverModels("https://litellm.example.com", "sk-test", { modelsDev: false });

    expect(result.models[0]?.litellmPolicy?.suppressReasoningVisibility === true).toBe(suppress);
  });
});
