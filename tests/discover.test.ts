import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  buildCompat,
  discoverModels,
  enrichCachedModel,
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

afterEach(() => {
  vi.restoreAllMocks();
});

describe("normalizeBaseUrl", () => {
  it("rejects insecure non-loopback endpoints", () => {
    expect(() => normalizeBaseUrl("http://litellm.example.com")).toThrow(/HTTPS/);
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

  it("keeps cached Chat levels when compatibility metadata proves a carrier", () => {
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

describe("discoverModels via /model/info", () => {
  it("keeps models with a null mode", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [{ model_name: "local/model", model_info: { mode: null } }],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models.map((model) => model.id)).toEqual(["local/model"]);
  });

  it("parses a /model/info success response with cost mapping", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "anthropic/claude-3-5-sonnet",
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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

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
      input: ["text", "image"],
      compat: { supportsStore: false },
    });
  });

  it.each([
    ["uses custom provider evidence for an opaque Moonshot backend", "moonshot", "internal/opaque", true],
    ["rejects conflicting custom provider and model evidence", "openai", "moonshot/kimi-k3", false],
  ] as const)("%s", async (_name, customProvider, backend, normalized) => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "private-route",
            litellm_params: { model: backend, custom_llm_provider: customProvider },
            model_info: { mode: "chat", supports_reasoning: true },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.litellmPolicy?.normalizeStrictToolMessages === true).toBe(normalized);
  });

  it("maps LiteLLM reasoning effort capabilities for a singleton", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "custom/reasoner",
            model_info: {
              mode: "chat",
              supports_reasoning: true,
              supports_none_reasoning_effort: true,
              supports_minimal_reasoning_effort: false,
              supports_low_reasoning_effort: false,
              supports_medium_reasoning_effort: false,
              supports_high_reasoning_effort: true,
              supports_xhigh_reasoning_effort: false,
              supports_max_reasoning_effort: true,
            },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.thinkingLevelMap).toEqual({
      off: "none",
      minimal: null,
      low: null,
      medium: null,
      high: "high",
      xhigh: null,
      max: "max",
    });
  });

  it("denies unreported singleton router reasoning efforts without catalog evidence", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "private/reasoner",
            litellm_params: { model: "internal/reasoner" },
            model_info: { mode: "chat", supports_reasoning: true, supports_high_reasoning_effort: true },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.thinkingLevelMap).toEqual({
      off: null,
      minimal: null,
      low: null,
      medium: null,
      high: "high",
      xhigh: null,
      max: null,
    });
  });

  it("merges singleton router reasoning effort flags over catalog metadata", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "openai/gpt-5.6-luna",
            model_info: { mode: "chat", supports_reasoning: true, supports_xhigh_reasoning_effort: false },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.thinkingLevelMap).toMatchObject({ off: "none", xhigh: null, max: "max" });
  });

  it("uses catalog costs when /model/info omits costs for Anthropic aliases", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [{ model_name: "opus-4.8", model_info: { mode: "chat" } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.source).toBe("model_info");
    expect(result.models[0]?.cost).toEqual({ input: 5, output: 25, cacheRead: 0.5, cacheWrite: 6.25 });
  });

  it("preserves catalog pricing tiers for /model/info models", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [{ model_name: "openai/gpt-5.5", model_info: { mode: "chat" } }],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.cost?.tiers).toEqual([
      { inputTokensAbove: 272000, input: 10, output: 45, cacheRead: 1, cacheWrite: 0 },
    ]);
  });

  it("preserves catalog max thinking metadata for /model/info models", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [{ model_name: "openai/gpt-5.6-luna", model_info: { mode: "chat" } }],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

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
    const deployments = [
      {
        model_name: "shared-route",
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
        model_name: "shared-route",
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
      const result = await discoverModels("https://litellm.example.com", "sk-test", {});
      expect(result.models).toEqual([
        expect.objectContaining({
          id: "shared-route",
          name: "shared-route (incomplete metadata)",
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

  it.each([
    ["Kimi", "moonshot/kimi-k2.6", "openai/gpt-4o"],
    ["DeepSeek", "openai/gpt-4o", "deepseek-v4"],
  ])("withholds %s policy when one deployment has contradictory family evidence", async (_case, routing, base) => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "conflicting-family-route",
              litellm_params: { model: routing, allowed_openai_params: ["thinking", "reasoning_effort"] },
              model_info: { id: "a", mode: "chat", base_model: base, supports_reasoning: true },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.litellmPolicy).toBeUndefined();
    expect(result.models[0]?.compat).toEqual({ supportsStore: false });
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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({ name: "spacey", input: ["text", "image"], contextWindow: 128_000 });
    expect(result.models[0]?.cost.input).toBeGreaterThan(0);
  });

  it("does not use a qualified public route as evidence for a multi-deployment group", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "openai/gpt-5.5",
            litellm_params: { model: "openai/gpt-5.5" },
            model_info: { id: "known", mode: "chat" },
          },
          {
            model_name: "openai/gpt-5.5",
            litellm_params: { model: "internal/mystery" },
            model_info: { id: "unknown", mode: "chat" },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
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

  it("treats repeated identified deployment rows as one effective singleton", async () => {
    const deployment = {
      model_name: "openai/gpt-5.5",
      model_info: { id: "deployment-a", mode: "chat" },
    };
    const fetchMock = vi.spyOn(globalThis, "fetch");
    for (const data of [[deployment], [deployment, deployment]]) {
      fetchMock.mockResolvedValueOnce(jsonResponse(200, { data }));
      const result = await discoverModels("https://litellm.example.com", "sk-test", {});
      expect(result.models[0]).toMatchObject({
        id: "openai/gpt-5.5",
        name: "openai/gpt-5.5",
        reasoning: true,
        contextWindow: 272_000,
        maxTokens: 128_000,
      });
    }
  });

  it.each(["openai", "custom_openai", "openai_like", "text-completion-openai", "azure", "azure_ai"])(
    "treats the %s adapter as transport when the model identifies Kimi",
    (adapter) => {
      expect(
        resolveModelInfoCatalog({
          model_name: "kimi-route",
          litellm_params: { model: "openai/kimi-k2.5" },
          model_info: { mode: "chat", litellm_provider: adapter },
        }),
      ).toEqual({ semanticFamily: "kimi", semanticModel: "kimi-k2.5-k2.6" });
    },
  );

  it("publishes Kimi compatibility when an OpenAI transport adapter routes openai/kimi", async () => {
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
        suppressReasoningVisibility: true,
      },
    });
  });

  it.each(["openai", "custom_openai", "openai_like", "text-completion-openai", "azure", "azure_ai"])(
    "uses the %s adapter family only when model and base identities provide no family",
    (adapter) => {
      expect(
        resolveModelInfoCatalog({
          model_name: "opaque-route",
          litellm_params: { model: "internal/model" },
          model_info: { mode: "chat", litellm_provider: adapter },
        }),
      ).toEqual({ semanticFamily: "openai" });
    },
  );

  it("derives DeepSeek family and accepted controls from Azure Foundry backend evidence", async () => {
    expect(
      resolveModelInfoCatalog({
        model_name: "public-route-without-family-text",
        litellm_params: { model: " azure_ai/DeepSeek-V4 ", allowed_openai_params: ["reasoning_effort"] },
        model_info: { mode: "chat", litellm_provider: "azure_ai" },
      }),
    ).toEqual({ semanticFamily: "deepseek", semanticModel: "deepseek-v4" });

    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "foundry-route",
              litellm_params: { model: " azure_ai/DeepSeek-V4 ", allowed_openai_params: ["reasoning_effort"] },
              model_info: { mode: "chat", litellm_provider: "azure_ai" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      id: "foundry-route",
      name: "foundry-route (incomplete metadata)",
      reasoning: true,
      contextWindow: 128_000,
      thinkingLevelMap: { off: null, high: "high", max: "max" },
      compat: { thinkingFormat: "openai", supportsReasoningEffort: true },
    });
  });

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
  ])("withholds catalog resolution when $name", ({ entry }) => {
    expect(resolveModelInfoCatalog(entry)).toBeUndefined();
  });

  it("preserves conflicts between a vendor-bearing identity and a specific adapter", () => {
    expect(
      resolveModelInfoCatalog({
        model_name: "specific-adapter-conflict",
        litellm_params: { model: "openai/kimi-k2.5" },
        model_info: { mode: "chat", litellm_provider: "anthropic" },
      }),
    ).toEqual({ semanticFamily: "conflicting" });
  });

  it.each([
    ["routing then base", "openai/gpt-4o", "openai/o3"],
    ["base then routing", "openai/o3", "openai/gpt-4o"],
  ])("withholds catalog metadata for same-provider model conflicts in %s order", (_case, routing, base) => {
    const result = resolveModelInfoCatalog({
      model_name: "conflicting-openai-models",
      litellm_params: { model: routing },
      model_info: { mode: "chat", litellm_provider: "openai", base_model: base },
    });

    expect(result).toEqual({ semanticFamily: "openai" });
  });

  it.each([
    "moonshot/kimi-k2.7",
    "moonshot/kimi-k2.7-instruct",
    "moonshot/kimi-k2.7-codec",
    "moonshot/kimi-k2.7-highspeedy",
  ])("does not classify K2.7 without an exact Code suffix: %s", (model) => {
    expect(
      resolveModelInfoCatalog({
        model_name: "unknown-k2.7-variant",
        litellm_params: { model, allowed_openai_params: ["thinking"] },
        model_info: { mode: "chat", supports_reasoning: true },
      }),
    ).toEqual({ semanticFamily: "kimi" });
  });

  it.each(["moonshot/kimi-k2.7-code", "moonshot/kimi-k2.7_highspeed", "moonshot/kimi-k2.7.code"])(
    "classifies the supported K2.7 Code suffix: %s",
    (model) => {
      expect(
        resolveModelInfoCatalog({
          model_name: "known-k2.7-variant",
          litellm_params: { model, allowed_openai_params: ["thinking"] },
          model_info: { mode: "chat", supports_reasoning: true },
        }),
      ).toMatchObject({ semanticFamily: "kimi", semanticModel: "kimi-k2.7-code" });
    },
  );

  it("withholds same-provider conflicting catalog metadata from the published model", async () => {
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

      expect(result.models[0]).toMatchObject({
        name: "one-conflicted-openai-deployment (incomplete metadata)",
        reasoning: false,
        input: ["text"],
        contextWindow: 128_000,
        maxTokens: 16_384,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      });
      expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
    }
  });

  it("withholds catalog metadata for conflicting provider evidence within one deployment", async () => {
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

    expect(result.models[0]).toMatchObject({
      name: "one-conflicted-deployment (incomplete metadata)",
      reasoning: false,
      input: ["text"],
      contextWindow: 128_000,
      maxTokens: 16_384,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    });
    expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
  });

  it("preserves Bedrock catalog authority", () => {
    expect(
      resolveModelInfoCatalog({
        model_name: "bedrock-claude-route",
        litellm_params: { model: "bedrock/anthropic.claude-sonnet-4-6" },
        model_info: { mode: "chat", litellm_provider: "bedrock" },
      }),
    ).toMatchObject({ provider: "amazon-bedrock" });
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

      const result = await discoverModels("https://litellm.example.com", "sk-test", {});

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

  it("keeps an ambiguous group with complete router pricing unsuffixed and reports withheld authority", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const priced = (id: string, model: string) => ({
      model_name: "priced-ambiguous-authority",
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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      id: "priced-ambiguous-authority",
      name: "priced-ambiguous-authority",
      cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
      contextWindow: 128_000,
      maxTokens: 16_384,
    });
    const diagnostics = stderr.mock.calls.map(([message]) => String(message));
    expect(diagnostics).toHaveLength(1);
    expect(diagnostics[0]).toContain("priced-ambiguous-authority");
    expect(diagnostics[0]).toContain("catalog limits, pricing, and reasoning metadata are withheld");
  });

  it("denies Chat reasoning levels without an accepted carrier", async () => {
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

  it("keeps baseline compatibility metadata for Responses-mode routes", async () => {
    // Compat is derived from the model id alone on this branch. If API-aware compat
    // were reintroduced at the call site, a Responses-mode Moonshot route would
    // silently lose these Kimi repairs, so the whole object is pinned exactly.
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [{ model_name: "moonshot/kimi-k2.6", model_info: { id: "one", mode: "responses" } }],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({ api: "openai-responses" });
    expect(result.models[0]?.compat).toEqual({
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsStrictMode: false,
      maxTokensField: "max_tokens",
    });
  });

  it("keeps the Anthropic prompt-cache marker on a Responses-mode alias", async () => {
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [{ model_name: "anthropic/claude-sonnet-4-6", model_info: { id: "one", mode: "responses" } }],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({ api: "openai-responses" });
    expect(result.models[0]?.compat).toEqual({ supportsStore: false, cacheControlFormat: "anthropic" });
  });

  it.each([
    ["a numeric mode", { model_name: "bad-mode", model_info: { id: "a", mode: 7 } }],
    ["a numeric deployment id", { model_name: "bad-id", model_info: { id: 7, mode: "chat" } }],
    // Every untrusted string this branch newly reads. A YAML value such as
    // `model_name: 4.1` parses as a number, which is an ordinary operator typo.
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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models.map((model) => model.id)).toContain("healthy-route");
    expect(result.models.find((model) => model.id === "healthy-route")?.cost.input).toBeGreaterThan(0);
  });

  it("withholds a fallback or health entry whose id is not a string", async () => {
    // The same invariant on the two paths that build a model straight from an id.
    mockEndpoints({
      "/model/info": () => jsonResponse(403, {}),
      "/v1/models": () => jsonResponse(200, { data: [{ id: 7 }, { id: "gpt-4o", owned_by: 7 }] }),
    });
    const fallback = await discoverModels("https://litellm.example.com", "sk-test", {});
    expect(fallback.models.map((model) => model.id)).toEqual(["gpt-4o"]);

    mockEndpoints({
      "/model/info": () => jsonResponse(403, {}),
      "/v1/models": () => jsonResponse(404, {}),
      "/health": () => jsonResponse(200, { healthy_endpoints: [{ model: 7 }, { model: "anthropic/claude-opus-4-7" }] }),
    });
    const health = await discoverModels("https://litellm.example.com", "sk-test", {});
    expect(health.models.map((model) => model.id)).toEqual(["anthropic/claude-opus-4-7"]);
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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models.map((model) => model.id)).toEqual(["anthropic/claude-opus-4-7"]);
  });

  it("survives a deeply nested deployment row", async () => {
    // Canonicalization is depth-bounded, so a pathological payload cannot exhaust
    // the stack and take every model down with it.
    let nested: unknown = "leaf";
    for (let depth = 0; depth < 20_000; depth++) nested = { nested };
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "deep-route",
              litellm_params: { model: "openai/gpt-4o" },
              model_info: { id: "a", mode: "chat", extra: nested },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.id).toBe("deep-route");
  });

  it("never emits the fallback-only sentinel for a reduced deployment group", async () => {
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
    ];

    for (const data of groups) {
      mockEndpoints({ "/model/info": () => jsonResponse(200, { data }) });
      const result = await discoverModels("https://litellm.example.com", "sk-test", {});

      expect(result.models).toHaveLength(1);
      expect(result.models[0]?.name).not.toContain(" (no metadata)");
      expect(result.models[0]?.name).toBe(`${result.models[0]?.id} (incomplete metadata)`);
    }
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

  it("reports genuinely conflicting family evidence once with bounded detail", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const conflicting = (route: string) => ({
      model_name: route,
      model_info: { id: route, mode: "chat", base_model: "anthropic/claude-sonnet-4-6" },
      litellm_params: { model: "openai/gpt-4o" },
    });
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: ["family-a", "family-b", "family-c", "family-d"].map(conflicting),
        }),
    });

    await discoverModels("https://litellm.example.com", "sk-test", {});

    const diagnostics = stderr.mock.calls.map(([message]) => String(message));
    expect(diagnostics).toHaveLength(1);
    expect(diagnostics[0]).toContain("4 route group(s) have conflicting deployment family evidence");
    expect(diagnostics[0]).toContain("family-a, family-b, family-c (+1 more)");
    expect(diagnostics[0]).not.toContain("family-d");
    expect(diagnostics[0]).not.toContain("claude-sonnet");
  });

  it("reports conflicting deployment provider identity once with bounded detail", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const conflicting = (route: string) => [
      { model_name: route, model_info: { id: `${route}-a`, mode: "chat" }, litellm_params: { model: "openai/gpt-4o" } },
      {
        model_name: route,
        model_info: { id: `${route}-b`, mode: "chat" },
        litellm_params: { model: "anthropic/claude-sonnet-4-6" },
      },
    ];
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: ["route-a", "route-b", "route-c", "route-d"].flatMap(conflicting),
        }),
    });

    await discoverModels("https://litellm.example.com", "sk-test", {});

    const diagnostics = stderr.mock.calls.map(([message]) => String(message));
    expect(diagnostics).toHaveLength(1);
    expect(diagnostics[0]).toContain("4 route group(s) have missing or conflicting deployment provider evidence");
    // Bounded: a count, at most three route ids, and no deployment ids or params.
    expect(diagnostics[0]).toContain("route-a, route-b, route-c (+1 more)");
    expect(diagnostics[0]).not.toContain("route-d");
    expect(diagnostics[0]).not.toContain("route-a-a");
  });

  it("reports each ambiguous route once per process, not once per discovery", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
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
      await discoverModels("https://litellm.example.com", "sk-test", {});
    };

    await discover(["once-a", "once-b"]);
    expect(stderr.mock.calls).toHaveLength(1);
    expect(String(stderr.mock.calls[0]?.[0])).toContain("2 route group(s)");

    // A background refresh of the same misconfiguration must not repeat itself.
    await discover(["once-a", "once-b"]);
    expect(stderr.mock.calls).toHaveLength(1);

    // A newly ambiguous route is still worth reporting, and only that one.
    await discover(["once-a", "once-b", "once-c"]);
    expect(stderr.mock.calls).toHaveLength(2);
    const second = String(stderr.mock.calls[1]?.[0]);
    expect(second).toContain("1 route group(s)");
    expect(second).toContain("once-c");
    expect(second).not.toContain("once-a");
  });

  it("reports a route whose deployments supply partial provider evidence", async () => {
    // Withholding also happens when one deployment resolves a provider and another
    // supplies none, so the wording must not claim a conflict is the only cause.
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "partial-evidence",
              model_info: { id: "a", mode: "chat" },
              litellm_params: { model: "openai/gpt-4o" },
            },
            { model_name: "partial-evidence", model_info: { id: "b", mode: "chat" } },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.name).toBe("partial-evidence (incomplete metadata)");
    const diagnostics = stderr.mock.calls.map(([message]) => String(message));
    expect(diagnostics).toHaveLength(1);
    expect(diagnostics[0]).toContain("missing or conflicting");
    expect(diagnostics[0]).toContain("partial-evidence");
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

describe("discoverModels wildcard expansion via /v1/models", () => {
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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.source).toBe("model_info");
    expect(result.models.map((m) => m.id).sort()).toEqual([
      "lemonade/Bonsai-1.7B-gguf",
      "lemonade/Laguna-S-2.1-GGUF-UD-IQ4_NL",
      "lemonade/Qwen3.6-35B-A3B-MTP-GGUF-UD-IQ4_NL",
    ]);
    // the raw wildcard id must NOT surface as a selectable model
    expect(result.models.some((m) => m.id.includes("*"))).toBe(false);
    // the concrete /model/info entry is not duplicated by /v1/models
    expect(result.models.filter((m) => m.id === "lemonade/Qwen3.6-35B-A3B-MTP-GGUF-UD-IQ4_NL")).toHaveLength(1);
    // /v1/models was actually queried (the expansion path ran)
    expect(urls.some((u) => u.endsWith("/v1/models"))).toBe(true);
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

  it("never exposes a literal wildcard when /v1/models expansion fails", async () => {
    mockEndpoints({
      "/model/info": () => jsonResponse(200, { data: [{ model_name: "team/*", model_info: { mode: "chat" } }] }),
      "/v1/models": () => jsonResponse(500, { error: "unavailable" }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models).toEqual([]);
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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.source).toBe("model_info");
    expect(result.models.map((m) => m.id)).toEqual(["openai/gpt-4o"]);
    expect(urls.some((u) => u.endsWith("/v1/models"))).toBe(false);
  });
});

describe("discoverModels response-mode models", () => {
  it("keeps /model/info response-mode models with a Responses API override", async () => {
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
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.source).toBe("model_info");
    expect(result.models).toHaveLength(1);
    expect(result.models[0]).toMatchObject({
      id: "openai/gpt-5.3-codex-openai",
      api: "openai-responses",
      contextWindow: 272000,
      maxTokens: 128000,
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

  it("keeps /health response-mode model_info fallbacks on Chat", async () => {
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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.source).toBe("health");
    expect(result.models).toHaveLength(1);
    expect(result.models[0]).toMatchObject({
      id: "openai/gpt-5.3-codex-openai",
      api: "openai-completions",
      compat: { supportsStore: false },
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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.source).toBe("health");
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

  it("does not derive thinking controls from a health endpoint without deployment detail", async () => {
    // No `model_id`, so the route name is the only input. It still enriches
    // limits and pricing, but it must not produce a reasoning selector.
    mockEndpoints({
      "/model/info": () => jsonResponse(404, {}),
      "/v1/models": () => jsonResponse(404, {}),
      "/health": () => jsonResponse(200, { healthy_endpoints: [{ model: "openai/gpt-5.5" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.source).toBe("health");
    expect(result.models[0]).toMatchObject({ id: "openai/gpt-5.5", reasoning: true });
    expect(result.models[0]?.thinkingLevelMap).toEqual({
      off: null,
      minimal: null,
      low: null,
      medium: null,
      high: null,
      xhigh: null,
      max: null,
    });
    expect(result.models[0]?.compat).not.toMatchObject({ supportsReasoningEffort: true });
  });
});

describe("catalog provider candidates", () => {
  it.each([
    ["decorated mixed-case Kimi", "ToGeThEr/MoOnShOtAi/KiMi-K2.6@prod", "together_ai", 262_144],
    ["mixed-case Claude", "AnThRoPiC/ClAuDe-SoNnEt-4-6", undefined, 1_000_000],
    ["decorated Claude", "BeDrOcK/US.AnThRoPiC.ClAuDe-SoNnEt-4-6-V1:0", "bedrock", 1_000_000],
  ])("retains catalog metadata for a $name provider-qualified backend", async (_case, backend, adapter, context) => {
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

    expect(resolved).toEqual({ semanticFamily: "claude" });
    expect(resolved).not.toHaveProperty("provider");
    expect(resolved).not.toHaveProperty("cost");
  });

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
  ])("resolves the bare Anthropic catalog id %s from the shared lookup rule", async (id) => {
    mockEndpoints({
      "/model/info": () => jsonResponse(200, { data: [{ model_name: id, model_info: { mode: "chat" } }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.name).not.toContain("metadata");
    expect(result.models[0]?.cost.input).toBeGreaterThan(0);
  });

  it.each(["claudia-x", "opusclip-2", "gpt-4o", "haiku"])("does not treat %s as an Anthropic alias", async (id) => {
    mockEndpoints({
      "/model/info": () => jsonResponse(200, { data: [{ model_name: id, model_info: { mode: "chat" } }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.name).toBe(`${id} (incomplete metadata)`);
    expect(result.models[0]?.cost).toEqual({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });
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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

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
      const result = await discoverModels("https://litellm.example.com", "sk-test", {});
      expect(result.source).toBe("models_list");
      expect(result.models.map((m) => m.id).sort()).toEqual(["anthropic/claude-3-5-sonnet", "openai/gpt-4o"]);
      const anthropic = result.models.find((m) => m.id === "anthropic/claude-3-5-sonnet")!;
      expect(anthropic.name).toBe("anthropic/claude-3-5-sonnet (no metadata)");
      expect(anthropic.compat).toEqual({ supportsStore: false, cacheControlFormat: "anthropic" });
    });
  }

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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]).toMatchObject({
      id: "gpt-5.5",
      name: "GPT-5.5",
      contextWindow: 272000,
    });
  });

  it("enriches a bare Fable 5 fallback model from the Pi catalog", async () => {
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
      models: [
        {
          id: "fable-5",
          name: "Claude Fable 5",
          reasoning: true,
          thinkingLevelMap: { xhigh: "xhigh", max: "max" },
        },
      ],
    });
  });

  it("enriches a bare Opus 5 fallback model from the Pi catalog", async () => {
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
      models: [
        {
          id: "opus-5",
          name: "Claude Opus 5",
          reasoning: true,
          thinkingLevelMap: { xhigh: "xhigh", max: "max" },
        },
      ],
    });
  });

  it("throws when /model/info returns a non-401/403/404 error", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(null, { status: 500 }));
    await expect(discoverModels("https://litellm.example.com", "sk-test", {})).rejects.toThrow(/500/);
  });
});

describe("discoverModels fallback to /health", () => {
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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

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
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(urls).toEqual([
      "https://litellm.example.com/model/info",
      "https://litellm.example.com/v1/models",
      "https://litellm.example.com/health",
    ]);
    expect(result.source).toBe("health");
    expect(result.models.map((model) => model.id)).toEqual(["azure/gpt-35-turbo", "anthropic/claude-3-5-sonnet"]);
    expect(result.models[1]).toMatchObject({
      // Neither route resolves in the Pi catalog, so both are evidence-free and
      // must say so rather than presenting default limits and zero cost as fact.
      name: "anthropic/claude-3-5-sonnet (incomplete metadata)",
      contextWindow: 128000,
      maxTokens: 16384,
      compat: { supportsStore: false, cacheControlFormat: "anthropic" },
    });
    expect(result.models[0]?.name).toBe("azure/gpt-35-turbo (incomplete metadata)");
  });

  it("marks evidence-free health routes and leaves catalog-resolved ones plain", async () => {
    // `/health` route text is never authorized for later cache re-enrichment, so an
    // unresolved route carries the permanent marker rather than the `/v1/models`
    // sentinel. A route the catalog resolves has real metadata and stays plain.
    mockEndpoints({
      "/model/info": () => jsonResponse(404, {}),
      "/v1/models": () => jsonResponse(404, {}),
      "/health": () =>
        jsonResponse(200, {
          healthy_endpoints: [{ model: "totally-unknown-route" }, { model: "anthropic/claude-opus-4-7" }],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.source).toBe("health");
    const [unresolved, resolved] = result.models;
    expect(unresolved).toMatchObject({
      id: "totally-unknown-route",
      name: "totally-unknown-route (incomplete metadata)",
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128_000,
      maxTokens: 16_384,
    });
    expect(unresolved?.name).not.toContain(" (no metadata)");
    expect(resolved?.name).toBe("Claude Opus 4.7");
    expect(resolved?.cost.input).toBeGreaterThan(0);
  });

  it("marks an unresolved health route reached through per-endpoint /model/info", async () => {
    mockEndpoints({
      "/model/info?litellm_model_id=uuid-1": () => jsonResponse(200, { data: [{ model_info: { mode: "chat" } }] }),
      "/model/info": () => jsonResponse(404, {}),
      "/v1/models": () => jsonResponse(404, {}),
      "/health": () =>
        jsonResponse(200, { healthy_endpoints: [{ model: "vertex/claude-sonnet", model_id: "uuid-1" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.source).toBe("health");
    expect(result.models[0]?.name).toBe("vertex/claude-sonnet (incomplete metadata)");
    expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
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
