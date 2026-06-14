import { afterEach, describe, expect, it, vi } from "vitest";
import { buildCompat, discoverModels, normalizeBaseUrl, shouldSuppressReasoningContent } from "../src/discover.js";

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

describe("shouldSuppressReasoningContent", () => {
  it("suppresses separate reasoning streams for Kimi/Moonshot aliases", () => {
    expect(shouldSuppressReasoningContent("kimi-k2.6")).toBe(true);
    expect(shouldSuppressReasoningContent("moonshotai/kimi-k2")).toBe(true);
  });

  it("does not suppress explicit forced-thinking models", () => {
    expect(shouldSuppressReasoningContent("kimi-k2-thinking")).toBe(false);
  });

  it("does not suppress unrelated models", () => {
    expect(shouldSuppressReasoningContent("openai/gpt-4o")).toBe(false);
  });
});

describe("discoverModels via /model/info", () => {
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
      input: ["text"],
      compat: { supportsStore: false },
    });
  });

  it("keeps responses-mode models (e.g. ChatGPT subscription) and drops other non-chat modes", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            { model_name: "chatgpt-5.5", model_info: { mode: "responses", supports_reasoning: true } },
            { model_name: "openai/text-embedding-3-large", model_info: { mode: "embedding" } },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.source).toBe("model_info");
    expect(result.models.map((m) => m.id)).toEqual(["chatgpt-5.5"]);
    expect(result.models[0]).toMatchObject({ id: "chatgpt-5.5", reasoning: true });
  });

  it("borrows thinkingLevelMap from the catalog for known model ids (keeps xhigh etc.)", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "deepseek-v4-pro",
              model_info: { mode: "chat", supports_reasoning: true, litellm_provider: "deepseek" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.source).toBe("model_info");
    const model = result.models.find((m) => m.id === "deepseek-v4-pro");
    // catalog entry for deepseek-v4-pro maps xhigh -> "max"
    expect(model?.thinkingLevelMap?.xhigh).toBe("max");
  });

  it("resolves thinkingLevelMap via the underlying key when model_name is an alias", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = input instanceof URL ? input.toString() : String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            // public alias "ds-pro"; real catalog key in litellm_params.model
            {
              model_name: "ds-pro",
              litellm_params: { model: "deepseek/deepseek-v4-pro" },
              model_info: { mode: "chat", supports_reasoning: true, litellm_provider: "deepseek" },
            },
            // same, but the key is carried in model_info.key instead
            {
              model_name: "ds-pro-2",
              model_info: {
                mode: "chat",
                supports_reasoning: true,
                litellm_provider: "deepseek",
                key: "deepseek/deepseek-v4-pro",
              },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(result.models.find((m) => m.id === "ds-pro")?.thinkingLevelMap?.xhigh).toBe("max");
    expect(result.models.find((m) => m.id === "ds-pro-2")?.thinkingLevelMap?.xhigh).toBe("max");
  });
});

describe("discoverModels fallback to /v1/models", () => {
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

  it("uses models.dev metadata when LiteLLM returns provider ownership", async () => {
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
      if (url === "https://models.dev/api.json") {
        return jsonResponse(200, {
          openai: {
            models: {
              "gpt-5.5": {
                id: "gpt-5.5",
                name: "GPT-5.5",
                reasoning: true,
                modalities: { input: ["text", "image", "pdf"], output: ["text"] },
                limit: { context: 1_050_000, input: 922_000, output: 128_000 },
                cost: { input: 5, output: 30, cache_read: 0.5 },
              },
            },
          },
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});

    expect(urls).toContain("https://models.dev/api.json");
    expect(result.source).toBe("models_list");
    expect(result.models).toHaveLength(1);
    expect(result.models[0]).toMatchObject({
      id: "gpt-5.5",
      name: "GPT-5.5",
      reasoning: true,
      thinkingLevelMap: { off: "none", xhigh: "xhigh" },
      input: ["text", "image"],
      contextWindow: 1050000,
      maxTokens: 128000,
      compat: { supportsStore: false },
    });
    expect(result.models[0]?.cost).toEqual({ input: 5, output: 30, cacheRead: 0.5, cacheWrite: 0 });
  });

  it("throws when /model/info returns a non-401/403/404 error", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(null, { status: 500 }));
    await expect(discoverModels("https://litellm.example.com", "sk-test", {})).rejects.toThrow(/500/);
  });
});

describe("discoverModels fallback to /health", () => {
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
      name: "anthropic/claude-3-5-sonnet",
      contextWindow: 128000,
      maxTokens: 16384,
      compat: { supportsStore: false, cacheControlFormat: "anthropic" },
    });
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
