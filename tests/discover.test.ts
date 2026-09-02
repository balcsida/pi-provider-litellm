import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  buildCompat,
  discoverModels,
  emitsThinkTags,
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
      name: "openai/gpt-4o (incomplete metadata)",
      input: ["text"],
      compat: { supportsStore: false },
    });
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

  it("merges singleton router reasoning effort flags over catalog metadata", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "openai/gpt-5.6-luna",
            litellm_params: { model: "openai/gpt-5.6-luna" },
            model_info: { mode: "chat", supports_reasoning: true, supports_xhigh_reasoning_effort: false },
          },
        ],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.thinkingLevelMap).toMatchObject({ off: "none", xhigh: null, max: "max" });
  });

  it("does not use an unqualified public route alias as backend catalog evidence", async () => {
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
    expect(result.models[0]).toMatchObject({
      id: "opus-4.8",
      name: "opus-4.8 (incomplete metadata)",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128_000,
      maxTokens: 16_384,
    });
    expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.cost?.tiers).toEqual([
      { inputTokensAbove: 272000, input: 10, output: 45, cacheRead: 1, cacheWrite: 0 },
    ]);
  });

  it("preserves catalog max thinking metadata for /model/info models", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [
          {
            model_name: "openai/gpt-5.6-luna",
            litellm_params: { model: "openai/gpt-5.6-luna" },
            model_info: { mode: "chat" },
          },
        ],
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
      const result = await discoverModels("https://litellm.example.com", "sk-test", {});
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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

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
    ["no backend identity", { model_info: {} }],
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
      litellm_params: { model: "openai/gpt-5.5" },
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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

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

  it("includes custom_llm_provider in catalog authority conflicts", () => {
    expect(
      resolveModelInfoCatalog({
        model_name: "custom-provider-conflict",
        litellm_params: { model: "openai/gpt-4o", custom_llm_provider: "anthropic" },
        model_info: { mode: "chat" },
      }),
    ).toBeUndefined();

    expect(
      resolveModelInfoCatalog({
        model_name: "custom-provider-agreement",
        litellm_params: { model: "openai/gpt-4o", custom_llm_provider: "openai" },
        model_info: { mode: "chat" },
      }),
    ).toMatchObject({ provider: "openai", catalogModelId: "gpt-4o" });
  });

  it("includes resolved and unresolved known provider prefixes in conflict checks", () => {
    expect(
      resolveModelInfoCatalog({
        model_name: "adapter-first-prefix-conflict",
        litellm_params: { model: "openai/gpt-4o" },
        model_info: { mode: "chat", litellm_provider: "azure" },
      }),
    ).toBeUndefined();

    expect(
      resolveModelInfoCatalog({
        model_name: "conflicting-unresolved-prefix",
        litellm_params: { model: "openai/not-in-catalog" },
        model_info: { mode: "chat", base_model: "anthropic/opus-4-7" },
      }),
    ).toBeUndefined();

    expect(
      resolveModelInfoCatalog({
        model_name: "consistent-unresolved-prefix",
        litellm_params: { model: "anthropic/not-in-catalog" },
        model_info: { mode: "chat", base_model: "anthropic/opus-4-7" },
      }),
    ).toMatchObject({ provider: "anthropic" });
  });

  it.each([
    ["openai/gpt-4o", "openai/gpt-4.1"],
    ["openai/gpt-4.1", "openai/gpt-4o"],
  ])("withholds two different concrete %s and %s catalog identities from one row", (backend, baseModel) => {
    expect(
      resolveModelInfoCatalog({
        model_name: "same-provider-model-conflict",
        litellm_params: { model: backend },
        model_info: { mode: "chat", litellm_provider: "openai", base_model: baseModel },
      }),
    ).toBeUndefined();
  });

  it.each(["azure/my-deployment", "my-deployment"])(
    "resolves the Azure deployment name %s through its concrete base model",
    (backend) => {
      expect(
        resolveModelInfoCatalog({
          model_name: "azure-deployment-route",
          litellm_params: { model: backend },
          model_info: { mode: "chat", litellm_provider: "azure", base_model: "gpt-4o" },
        }),
      ).toMatchObject({ provider: "azure-openai-responses" });
    },
  );

  it.each([
    ["anthropic", "openai/gpt-4o", "bedrock/anthropic.claude-sonnet-4-6"],
    ["anthropic", "bedrock/anthropic.claude-sonnet-4-6", "openai/gpt-4o"],
    ["openai", "anthropic/opus-4-7", "bedrock/anthropic.claude-sonnet-4-6"],
    ["openai", "bedrock/anthropic.claude-sonnet-4-6", "anthropic/opus-4-7"],
    ["bedrock", "anthropic/opus-4-7", "openai/gpt-4o"],
    ["bedrock", "openai/gpt-4o", "anthropic/opus-4-7"],
  ])(
    "withholds catalog authority for the %s adapter with conflicting model %s and base model %s",
    (adapter, backend, baseModel) => {
      expect(
        resolveModelInfoCatalog({
          model_name: "conflicting-evidence",
          litellm_params: { model: backend },
          model_info: { mode: "chat", litellm_provider: adapter, base_model: baseModel },
        }),
      ).toBeUndefined();
    },
  );

  it("withholds and diagnoses different concrete catalog identities from one provider row", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const route = `same-provider-model-conflict-${process.pid}-${Date.now()}-${Math.random()}`;
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: route,
              litellm_params: { model: "openai/gpt-4o" },
              model_info: { mode: "chat", litellm_provider: "openai", base_model: "openai/gpt-4.1" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

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

  it("withholds catalog metadata and diagnoses a singleton with contradictory provider signals", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const route = `conflicting-evidence-${process.pid}-${Date.now()}-${Math.random()}`;
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: route,
              litellm_params: { model: "openai/not-in-catalog" },
              model_info: { mode: "chat", base_model: "anthropic/opus-4-7" },
            },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

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
    ).toMatchObject({ provider: "amazon-bedrock" });
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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

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
      [
        { model_name: "embedding-sibling-route", model_info: { id: "chat", mode: "chat" } },
        { model_name: "embedding-sibling-route", model_info: { id: "embed", mode: "embedding" } },
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

    await discoverModels("https://litellm.example.com", "sk-test", {});

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
      await discoverModels("https://litellm.example.com", "sk-test", {});
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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.name).toBe(`${route} (incomplete metadata)`);
    const diagnostics = stderr.mock.calls.map(([message]) => String(message));
    expect(diagnostics).toHaveLength(1);
    expect(diagnostics[0]).toContain("missing or conflicting");
    expect(diagnostics[0]).toContain(route);
  });

  it("keeps catalog authority for a lone identified chat deployment beside a non-chat sibling", async () => {
    // An embedding sibling votes on transport but is not a deployment, so the
    // remaining chat row can retain independently supplied backend authority.
    mockEndpoints({
      "/model/info": () =>
        jsonResponse(200, {
          data: [
            {
              model_name: "openai/gpt-5.5",
              litellm_params: { model: "openai/gpt-5.5" },
              model_info: { id: "chat", mode: "chat" },
            },
            { model_name: "openai/gpt-5.5", model_info: { id: "embed", mode: "embedding" } },
          ],
        }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models).toHaveLength(1);
    expect(result.models[0]).toMatchObject({
      id: "openai/gpt-5.5",
      name: "openai/gpt-5.5",
      reasoning: true,
      contextWindow: 272_000,
      api: "openai-completions",
    });
    expect(result.models[0]?.cost.input).toBeGreaterThan(0);
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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.compat).toEqual(buildCompat("kimi-k3"));
  });

  it.each([
    ["Moonshot deployments", ["moonshot/kimi-k3", "moonshot/kimi-k3"], true],
    ["mixed deployments", ["moonshot/kimi-k3", "azure_ai/FW-Kimi-K3"], false],
    ["reversed mixed deployments", ["azure_ai/FW-Kimi-K3", "moonshot/kimi-k3"], false],
    ["incomplete deployment metadata", ["moonshot/kimi-k3", undefined], false],
  ] as const)("aggregates %s conservatively", async (_name, routes, suppress) => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: routes.map((model) => ({
          model_name: "kimi-prod",
          ...(model ? { litellm_params: { model } } : {}),
          model_info: { mode: "chat" },
        })),
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.suppressReasoningContent === true).toBe(suppress);
  });

  it.each([
    [
      "ignores an unsupported non-Moonshot sibling",
      [
        { model: "moonshot/kimi-k3", mode: "chat" },
        { model: "openai/text-embedding-3-small", mode: "embedding" },
      ],
      true,
    ],
    [
      "does not use an unsupported Moonshot sibling as evidence",
      [
        { model: "azure_ai/FW-Kimi-K3", mode: "chat" },
        { model: "moonshot/kimi-k3", mode: "embedding" },
      ],
      false,
    ],
  ] as const)("%s when reducing suppression", async (_name, deployments, suppress) => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: deployments.map(({ model, mode }, index) => ({
          model_name: "kimi-prod",
          litellm_params: { model },
          model_info: { id: `deployment-${index}`, mode },
        })),
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.suppressReasoningContent === true).toBe(suppress);
  });

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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.suppressReasoningContent).toBeUndefined();
  });

  it.each([
    ["provider-only metadata", { custom_llm_provider: "moonshot" }],
    [
      "conflicting Moonshot provider and Azure backend",
      { custom_llm_provider: "moonshot", model: "azure_ai/FW-Kimi-K3" },
    ],
  ])("does not suppress with %s", async (_name, litellm_params) => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [{ model_name: "kimi-prod", litellm_params, model_info: { mode: "chat" } }],
      }),
    );

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.suppressReasoningContent).toBeUndefined();
  });

  it("keeps route evidence isolated between discoveries", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      const model = url.startsWith("https://moonshot.example.com") ? "moonshot/kimi-k3" : "azure_ai/FW-Kimi-K3";
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

    const moonshot = await discoverModels("https://moonshot.example.com", "sk-test", {});
    const azure = await discoverModels("https://azure.example.com", "sk-test", {});

    expect(moonshot.models[0]?.suppressReasoningContent).toBe(true);
    expect(azure.models[0]?.suppressReasoningContent).toBeUndefined();
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

    await discoverModels("https://litellm.example.com", "sk-test", {});
    fallback = true;
    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.suppressReasoningContent).toBeUndefined();
  });
});

describe("discoverModels via /health", () => {
  it.each([
    ["Moonshot first", ["moonshot/kimi-k3", "azure_ai/FW-Kimi-K3"]],
    ["Azure first", ["azure_ai/FW-Kimi-K3", "moonshot/kimi-k3"]],
  ] as const)("does not suppress duplicate mixed routes when %s", async (_name, routes) => {
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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.source).toBe("health");
    expect(result.models).toHaveLength(1);
    expect(result.models[0]?.suppressReasoningContent).toBeUndefined();
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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.suppressReasoningContent).toBe(true);
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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models).toEqual([
      expect.objectContaining({
        id: claudeId,
        name: `${claudeId} (incomplete metadata)`,
        api: "openai-responses",
        reasoning: false,
        input: ["text"],
        contextWindow: 16_000,
        maxTokens: 2_000,
        cost: expect.objectContaining({ input: 4, output: 20, cacheWrite: 5 }),
        compat: { supportsStore: false, cacheControlFormat: "anthropic" },
      }),
      expect.objectContaining({
        id: kimiId,
        name: `${kimiId} (incomplete metadata)`,
        reasoning: false,
        input: ["text"],
        contextWindow: 16_000,
        maxTokens: 2_000,
        compat: { supportsStore: false },
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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

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
          supportsStore: false,
          supportsDeveloperRole: false,
          supportsReasoningEffort: false,
          supportsStrictMode: false,
          maxTokensField: "max_tokens",
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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models).toEqual([
      expect.objectContaining({
        id: "openai/gpt-4o",
        name: "openai/gpt-4o (incomplete metadata)",
        reasoning: false,
        input: ["text"],
        contextWindow: 8_000,
        maxTokens: 16_384,
        cost: { input: 1, output: 0, cacheRead: 0, cacheWrite: 0 },
      }),
    ]);
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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: "responses/model-a", api: "openai-responses" }),
        expect.objectContaining({ id: "chat/model-b", api: "openai-completions" }),
      ]),
    );
    expect(result.models.map((model) => model.id)).not.toContain("other/model-c");
  });

  it("intersects thinking levels across overlapping wildcard matches", async () => {
    const wildcard = (modelName: string, id: string, supports: { low?: boolean; high?: boolean; max?: boolean }) => ({
      model_name: modelName,
      litellm_params: { model: `internal/${id}` },
      model_info: {
        id,
        mode: "chat",
        supports_reasoning: true,
        supports_low_reasoning_effort: supports.low,
        supports_high_reasoning_effort: supports.high,
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
          data: [
            wildcard("*", "broad", { low: true, high: true }),
            wildcard("team/*", "narrow", { low: false, high: true, max: true }),
          ],
        }),
      "/v1/models": () => jsonResponse(200, { data: [{ id: "team/model" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.thinkingLevelMap).toEqual({ low: null, high: "high", max: null });
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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

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
              litellm_params: { model: "github-copilot/gemini-3.1-pro-preview" },
              model_info: { id: "narrow", mode: "chat" },
            },
          ],
        }),
      "/v1/models": () => jsonResponse(200, { data: [{ id: "team/model" }] }),
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models[0]?.cost.tiers).toEqual([
      { inputTokensAbove: 200_000, input: 5, output: 30, cacheRead: 0.5, cacheWrite: 0 },
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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

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
        supports_high_reasoning_effort: true,
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
        supports_high_reasoning_effort: true,
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

      const result = await discoverModels("https://litellm.example.com", "sk-test", {});

      expect(result.models).toEqual([
        {
          id: "team/claude-sonnet-4-6",
          name: "team/claude-sonnet-4-6",
          api: "openai-completions",
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

      const result = await discoverModels("https://litellm.example.com", "sk-test", {});

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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models.map((model) => model.id)).toEqual(["team/chat"]);
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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

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

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.source).toBe("health");
    expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
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
    expect(result.models[0]).not.toHaveProperty("thinkingLevelMap");
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
  it("reports completed health detail requests rather than endpoint indexes", async () => {
    const endpoints = Array.from({ length: 11 }, (_, index) => ({
      model: `model-${index + 1}`,
      model_id: `uuid-${index + 1}`,
    }));
    const pending: Array<(response: Response) => void> = [];
    const progress = vi.fn();
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return Promise.resolve(new Response(null, { status: 404 }));
      if (url.endsWith("/v1/models")) return Promise.resolve(new Response(null, { status: 404 }));
      if (url.endsWith("/health")) return Promise.resolve(jsonResponse(200, { healthy_endpoints: endpoints }));
      if (url.endsWith("uuid-11")) {
        return Promise.resolve(jsonResponse(200, { data: [{ model_name: "model-11", model_info: { mode: "chat" } }] }));
      }
      return new Promise<Response>((resolve) => pending.push(resolve));
    });

    const discovery = discoverModels("https://litellm.example.com", "sk-test", { onProgress: progress });
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(14));
    const reportedBeforeCompletion = progress.mock.calls.some(([message]) => message === "Fetched 11/11 models...");
    for (const [index, resolve] of pending.entries()) {
      resolve(jsonResponse(200, { data: [{ model_name: `model-${index + 1}`, model_info: { mode: "chat" } }] }));
    }
    await discovery;

    expect(reportedBeforeCompletion).toBe(false);
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
