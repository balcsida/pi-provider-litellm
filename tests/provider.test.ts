import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  type Api,
  type Credential,
  type Model,
  type ModelsPublication,
  normalizeContext,
  type ProviderAuth,
  type RefreshModelsContext,
} from "@earendil-works/pi-ai";
import { afterAll, afterEach, beforeEach, describe, expect, it, type MockInstance, vi } from "vitest";
import { discoverModels } from "../src/discover.js";
import { createLiteLLMProvider, toNativeModels } from "../src/provider.js";
import type { DiscoveryResult, LiteLLMApi, LiteLLMModel } from "../src/types.js";

const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-provider-"));
const apiSpies = vi.hoisted(() => ({ anthropic: vi.fn(), completions: vi.fn(), responses: vi.fn() }));
vi.mock("@earendil-works/pi-ai/compat", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@earendil-works/pi-ai/compat")>()),
  anthropicMessagesApi: () => ({ stream: apiSpies.anthropic, streamSimple: apiSpies.anthropic }),
  openAICompletionsApi: () => ({ stream: apiSpies.completions, streamSimple: apiSpies.completions }),
  openAIResponsesApi: () => ({ stream: apiSpies.responses, streamSimple: apiSpies.responses }),
}));
vi.mock("@earendil-works/pi-coding-agent", () => ({
  getAgentDir: () => agentDir,
}));

const credential: Credential = { type: "api_key", key: "secret" };
const auth: ProviderAuth = {
  apiKey: { name: "API key", resolve: async () => ({ auth: { apiKey: "secret" } }) },
};

afterAll(async () => {
  await rm(agentDir, { recursive: true, force: true });
});

afterEach(() => {
  vi.restoreAllMocks();
});

const discovered = (id: string): DiscoveryResult => ({
  source: "model_info",
  models: [
    {
      id,
      name: id,
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128_000,
      maxTokens: 4096,
      api: "openai-completions",
    },
  ],
});

function native(id: string): Model<LiteLLMApi> {
  return toNativeModels("litellm", "https://proxy.example/v1", discovered(id).models)[0];
}

function foreignApiModel(id: string, api = "google-generative-ai"): Model<"openai-completions"> {
  return { ...native(id), api } as unknown as Model<"openai-completions">;
}

type TestRefreshContext = RefreshModelsContext & { publications: ModelsPublication[] };

function context(initial: readonly Model<Api>[] | undefined, allowNetwork: boolean, checkedAt = 1): TestRefreshContext {
  const publications: ModelsPublication[] = [];
  return {
    stored: initial ? { models: initial, checkedAt } : undefined,
    allowNetwork,
    credential,
    signal: new AbortController().signal,
    publish: vi.fn(async (publication) => {
      publications.push(publication);
      publication.update?.();
      return true;
    }),
    publications,
  };
}

function controller(overrides: Partial<Parameters<typeof createLiteLLMProvider>[0]> = {}) {
  return createLiteLLMProvider({
    id: "litellm",
    name: "LiteLLM",
    baseUrl: "https://proxy.example/v1",
    auth,
    discover: vi.fn(async () => discovered("fresh")),
    resolveCredentialRoot: () => "https://proxy.example",
    ...overrides,
  });
}

describe("toNativeModels", () => {
  it("converts discovery models into complete native models", () => {
    expect(toNativeModels("litellm", "https://proxy.example/v1", discovered("model-a").models)).toEqual([
      expect.objectContaining({
        id: "model-a",
        provider: "litellm",
        api: "openai-completions",
        baseUrl: "https://proxy.example/v1",
      }),
    ]);
  });

  it("projects each supported protocol from one normalized proxy root", () => {
    const baseModel = discovered("model").models[0];
    const [messages, completions, responses] = toNativeModels("litellm", "https://proxy.example/v1/", [
      { ...baseModel, id: "messages", api: "anthropic-messages", compat: {} },
      { ...baseModel, id: "completions", api: "openai-completions" },
      { ...baseModel, id: "responses", api: "openai-responses", compat: undefined },
    ]);

    expect([messages, completions, responses].map(({ api, baseUrl }) => ({ api, baseUrl }))).toEqual([
      { api: "anthropic-messages", baseUrl: "https://proxy.example" },
      { api: "openai-completions", baseUrl: "https://proxy.example/v1" },
      { api: "openai-responses", baseUrl: "https://proxy.example/v1" },
    ]);
  });
});

describe("createLiteLLMProvider", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });
  it("restores current-version stored models offline without discovery", async () => {
    const discover = vi.fn(async () => discovered("fresh"));
    const value = controller({ discover });
    const stored = { ...native("stored"), litellmDiscoveryVersion: 3 as const };

    await value.refreshModels?.(context([stored], false));

    expect(value.getModels()).toEqual([stored]);
    expect(discover).not.toHaveBeenCalled();
  });

  it("removes an inherited strict-tool grant from a v2 Messages cache and refreshes it online", async () => {
    const stale = {
      ...native("claude-fable-5-1"),
      api: "anthropic-messages" as const,
      baseUrl: "https://proxy.example",
      compat: { forceAdaptiveThinking: true, supportsStrictTools: true },
      litellmDiscoveryVersion: 2 as const,
    };
    const fresh: DiscoveryResult = {
      source: "model_info",
      models: [
        {
          ...discovered("claude-fable-5-1").models[0],
          api: "anthropic-messages",
          compat: { forceAdaptiveThinking: true },
          litellmDiscoveryVersion: 3,
        },
      ],
    };
    const discover = vi.fn(async () => fresh);
    const value = controller({ discover });

    await value.refreshModels?.(context([stale], false));
    expect(value.getModels()[0]?.compat).toEqual({ forceAdaptiveThinking: true });
    expect(discover).not.toHaveBeenCalled();

    await value.refreshModels?.(context([stale], true));
    expect(discover).toHaveBeenCalledOnce();
    expect(value.getModels()[0]).toMatchObject({
      litellmDiscoveryVersion: 3,
      compat: { forceAdaptiveThinking: true },
    });
  });

  it("keeps a startup-discovered strict-tool grant over its v2 cache entry offline", async () => {
    const [seed] = toNativeModels("litellm", "https://proxy.example/v1", [
      {
        ...discovered("claude-sonnet-5").models[0],
        api: "anthropic-messages",
        compat: { supportsStrictTools: true },
        litellmDiscoveryVersion: 3,
      },
    ]);
    const stale = {
      ...seed,
      compat: { supportsStrictTools: true },
      litellmDiscoveryVersion: 2 as const,
    };
    const discover = vi.fn(async () => discovered("fresh"));
    const value = controller({ discover, models: [seed] });

    await value.refreshModels?.(context([stale], false));

    expect(value.getModels()).toEqual([seed]);
    expect(discover).not.toHaveBeenCalled();
  });

  it("restores mixed legacy and current-version entries per entry", async () => {
    const discover = vi.fn(async () => discovered("fresh"));
    const legacy = { ...native("legacy"), name: "legacy (no metadata)" };
    const current = {
      ...native("opus-5"),
      name: "opus-5 (no metadata)",
      reasoning: false,
      maxTokens: 16_384,
      litellmDiscoveryVersion: 3 as const,
    };
    const value = controller({ discover });

    await value.refreshModels?.(context([legacy, current], false));

    expect(value.getModels()).toEqual([
      legacy,
      expect.objectContaining({ id: "opus-5", name: "Claude Opus 5", reasoning: true }),
    ]);
    expect(discover).not.toHaveBeenCalled();
  });

  it("replaces legacy stored models after a successful forced refresh", async () => {
    const discover = vi.fn(async () => discovered("fresh"));
    const refreshContext = context([native("stored")], true);
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const value = controller({ discover });

    await value.refreshModels?.(refreshContext);

    expect(value.getModels()).toEqual([native("fresh")]);
    expect(discover).toHaveBeenCalledOnce();
    expect(refreshContext.publications[0]?.update).toBeDefined();
    expect(stderr).not.toHaveBeenCalled();
  });

  it("keeps legacy stored models and reports one version diagnostic when refresh fails", async () => {
    const discover = vi.fn(async () => {
      throw new Error("offline");
    });
    const stored = native("stored");
    const value = controller({ discover });
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);

    await value.refreshModels?.(context([stored], false));
    expect(stderr).not.toHaveBeenCalled();

    await expect(value.refreshModels?.(context([stored], true))).rejects.toThrow("offline");
    const otherStored = native("other-stored");
    await expect(value.refreshModels?.(context([stored, otherStored], true))).rejects.toThrow("offline");

    expect(value.getModels()).toEqual([stored, otherStored]);
    expect(discover).toHaveBeenCalledTimes(2);
    expect(stderr).toHaveBeenCalledTimes(1);
    expect(stderr).toHaveBeenCalledWith(expect.stringMatching(/version does not match 3.*network refresh failed/));
  });

  it("re-enriches stale cached catalog aliases offline without discovery", async () => {
    const discover = vi.fn(async () => discovered("fresh"));
    const value = controller({ discover });

    await value.refreshModels?.(
      context(
        [
          {
            ...native("opus-5"),
            name: "opus-5 (no metadata)",
            reasoning: false,
            maxTokens: 16_384,
            litellmDiscoveryVersion: 3,
          } as LiteLLMModel,
        ],
        false,
      ),
    );

    expect(value.getModels()).toEqual([
      expect.objectContaining({
        id: "opus-5",
        name: "Claude Opus 5",
        reasoning: true,
        thinkingLevelMap: { off: null, minimal: null, low: null, medium: null, high: null, xhigh: null, max: null },
        provider: "litellm",
        api: "openai-completions",
        compat: { supportsStore: false, cacheControlFormat: "anthropic", supportsReasoningEffort: false },
        baseUrl: "https://proxy.example/v1",
      }),
    ]);
    expect(discover).not.toHaveBeenCalled();
  });

  it("uses the catalog transport when enriching an evidence-free cached model", async () => {
    const discover = vi.fn(async () => discovered("fresh"));
    const value = controller({ discover });

    await value.refreshModels?.(
      context(
        [
          {
            ...native("openai/gpt-5.5"),
            name: "openai/gpt-5.5 (no metadata)",
            reasoning: false,
            maxTokens: 16_384,
            litellmDiscoveryVersion: 3,
            compat: undefined,
          } as LiteLLMModel,
        ],
        false,
      ),
    );

    expect(value.getModels()).toEqual([
      expect.objectContaining({
        id: "openai/gpt-5.5",
        name: "GPT-5.5",
        api: "openai-responses",
        compat: undefined,
      }),
    ]);
    expect(discover).not.toHaveBeenCalled();
  });

  it("updates a cached catalog model to the Responses transport offline", async () => {
    const discover = vi.fn(async () => discovered("fresh"));
    const value = controller({ discover });

    await value.refreshModels?.(
      context(
        [
          {
            ...native("openai/gpt-5.5"),
            name: "openai/gpt-5.5 (no metadata)",
            reasoning: false,
            maxTokens: 16_384,
            litellmDiscoveryVersion: 3,
          } as LiteLLMModel,
        ],
        false,
      ),
    );

    expect(value.getModels()).toEqual([
      expect.objectContaining({
        id: "openai/gpt-5.5",
        name: "GPT-5.5",
        api: "openai-responses",
        compat: undefined,
      }),
    ]);
    expect(discover).not.toHaveBeenCalled();
  });

  it("does not re-enrich partially enriched cached aliases offline", async () => {
    const legacyFallback: LiteLLMModel = {
      ...native("opus-5"),
      name: "opus-5 (no metadata)",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128_000,
      maxTokens: 16_384,
      litellmDiscoveryVersion: 3,
    };
    const partialCached: Model<Api>[] = [
      { ...legacyFallback, reasoning: true },
      { ...legacyFallback, input: ["text", "image"] },
      { ...legacyFallback, cost: { input: 1, output: 0, cacheRead: 0, cacheWrite: 0 } },
      { ...legacyFallback, contextWindow: 128_001 },
      { ...legacyFallback, maxTokens: 16_385 },
    ];
    for (const cached of partialCached) {
      const value = controller();

      await value.refreshModels?.(context([cached], false));

      expect(value.getModels()).toEqual([
        cached.reasoning
          ? {
              ...cached,
              thinkingLevelMap: {
                off: null,
                minimal: null,
                low: null,
                medium: null,
                high: null,
                xhigh: null,
                max: null,
              },
              compat: undefined,
            }
          : cached,
      ]);
    }
  });

  it("keeps unknown stale cached models unchanged offline", async () => {
    const discover = vi.fn(async () => discovered("fresh"));
    const cached = { ...native("unknown-model"), name: "unknown-model (no metadata)" };
    const value = controller({ discover });

    await value.refreshModels?.(context([cached], false));

    expect(value.getModels()).toEqual([cached]);
    expect(discover).not.toHaveBeenCalled();
  });

  it("publishes and persists successful discovery", async () => {
    const refreshContext = context([native("old")], true);
    const value = controller({ discover: vi.fn(async () => discovered("fresh")) });

    await value.refreshModels?.(refreshContext);

    expect(value.getModels()).toEqual([native("fresh")]);
    expect(refreshContext.publications.at(-1)?.persist).toEqual({
      models: [native("fresh")],
      checkedAt: expect.any(Number),
    });
  });

  it("publishes discovered models with the credential URL", async () => {
    const value = controller({
      discover: vi.fn(async () => ({
        ...discovered("fresh"),
        baseUrl: "https://credential.example/v1",
      })),
    });

    await value.refreshModels?.(context(undefined, true));

    expect(value.getModels()[0]?.baseUrl).toBe("https://credential.example/v1");
  });

  it("returns an ordinary model catalog with native Array semantics", () => {
    const listed = native("listed");
    const value = controller({ models: [listed] });
    const models = value.getModels();

    expect(models).toEqual([listed]);
    expect(Object.hasOwn(models, "some")).toBe(false);
    expect(models.some((model) => model.api === "openai-responses")).toBe(false);
  });

  it("reprojects matching cached hosts and rejects stale or placeholder hosts", () => {
    const baseModel = discovered("model").models[0];
    const models = toNativeModels("litellm", "https://proxy.example", [
      baseModel,
      { ...baseModel, id: "responses", api: "openai-responses", compat: undefined },
    ]);
    const value = controller();

    expect(value.filterModels?.(models, credential).map(({ api, baseUrl }) => ({ api, baseUrl }))).toEqual([
      { api: "openai-completions", baseUrl: "https://proxy.example/v1" },
      { api: "openai-responses", baseUrl: "https://proxy.example/v1" },
    ]);

    const stale = controller({ resolveCredentialRoot: () => "https://other.example" });
    expect(stale.filterModels?.(models, credential)).toEqual([]);

    const placeholder = controller({ resolveCredentialRoot: () => "https://litellm.example.com:8443" });
    expect(placeholder.filterModels?.(models, credential)).toEqual([]);
  });

  it("rejects a cached model from a different path-scoped credential root", () => {
    const [model] = toNativeModels("litellm", "https://gateway.example/team-a", discovered("model").models);
    const value = controller({ resolveCredentialRoot: () => "https://gateway.example/team-b" });

    expect(value.filterModels?.([model], credential)).toEqual([]);
    expect(() => value.stream(model, normalizeContext({ messages: [] }))).toThrow(
      /stale LiteLLM model root.*network refresh/i,
    );
    expect(apiSpies.completions).not.toHaveBeenCalled();
  });

  it("keeps valid models while filtering unsupported protocols and malformed or stale URLs", () => {
    const value = controller();
    const malformed = { ...native("malformed"), baseUrl: "not a URL" };
    const stale = { ...native("stale"), baseUrl: "https://stale.example/v1" };

    expect(value.filterModels?.([foreignApiModel("messages"), native("valid"), malformed, stale], credential)).toEqual([
      native("valid"),
    ]);
  });

  it("reports hidden cached models when no LiteLLM base URL is configured", () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const value = controller({ resolveCredentialRoot: () => undefined });

    expect(value.filterModels?.([native("hidden"), native("also-hidden")], credential)).toEqual([]);
    expect(stderr).toHaveBeenCalledWith(
      "LiteLLM (litellm): 2 model(s) hidden because no LiteLLM base URL is configured; " +
        "set LITELLM_BASE_URL or run /login litellm\n",
    );
  });

  it("stays silent when an unconfigured provider has no models to hide", () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const value = controller({ resolveCredentialRoot: () => undefined });

    expect(value.filterModels?.([], credential)).toEqual([]);
    expect(stderr).not.toHaveBeenCalled();
  });

  it("blocks placeholder cached hosts on non-default ports before protocol dispatch", () => {
    const value = controller();
    const model = { ...native("placeholder"), baseUrl: "https://litellm.example.com:8443/v1" };

    expect(() => value.stream(model, normalizeContext({ messages: [] }))).toThrow(
      /placeholder LiteLLM model host.*network refresh/i,
    );
    expect(apiSpies.completions).not.toHaveBeenCalled();
  });

  it("blocks stale hosts before tool-cap validation and protocol dispatch", () => {
    const value = controller({ resolveCredentialRoot: () => "https://other.example" });
    const staleModel = { ...native("stale"), litellmBackendFamily: "openai" as const };
    const oversizedTools = Array.from({ length: 129 }, (_, index) => ({
      name: `tool-${index}`,
      description: "test tool",
      parameters: { type: "object" as const, properties: {} },
    }));

    expect(() => value.stream(staleModel, normalizeContext({ messages: [], tools: oversizedTools }))).toThrow(
      /stale LiteLLM model root.*network refresh/i,
    );
    expect(() => value.streamSimple(native("stale"), normalizeContext({ messages: [] }))).toThrow(
      /stale LiteLLM model root.*network refresh/i,
    );
    expect(apiSpies.completions).not.toHaveBeenCalled();
  });

  it("preserves explicitly allowed insecure HTTP through host validation and dispatch", () => {
    apiSpies.completions.mockReturnValueOnce({});
    const [model] = toNativeModels("litellm", "http://host.docker.internal", discovered("local").models, true);
    const value = controller({
      allowInsecureHttp: true,
      resolveCredentialRoot: () => "http://host.docker.internal",
    });

    value.stream(model, normalizeContext({ messages: [] }));

    expect(model.baseUrl).toBe("http://host.docker.internal/v1");
    expect(apiSpies.completions).toHaveBeenCalledWith(
      expect.objectContaining({ baseUrl: "http://host.docker.internal/v1" }),
      normalizeContext({ messages: [] }),
      undefined,
    );
  });

  it.each(["stream", "streamSimple"] as const)(
    "passes the AuthResult env root and API key to resolveCredentialRoot for %s",
    (method) => {
      const resolveCredentialRoot = vi.fn(() => "https://proxy.example");
      const value = controller({ resolveCredentialRoot });

      value[method](native(method), normalizeContext({ messages: [] }), {
        apiKey: "resolved-key",
        env: { LITELLM_BASE_URL: "https://auth-result.example" },
      });

      expect(resolveCredentialRoot).toHaveBeenCalledWith(undefined, "https://auth-result.example", "resolved-key");
    },
  );

  it("blocks oversized OpenAI-family Chat tool catalogs before protocol dispatch", () => {
    const model = { ...native("opaque-gpt-route"), litellmBackendFamily: "openai" as const };
    const tools = Array.from({ length: 129 }, (_, index) => ({
      name: `tool-${index}`,
      description: "test tool",
      parameters: { type: "object" as const, properties: {} },
    }));
    const value = controller();

    expect(() => value.stream(model, normalizeContext({ messages: [], tools }))).toThrow(
      "LiteLLM model opaque-gpt-route uses Chat Completions with 129 tools, exceeding the 128-tool cap; route this model via Responses or reduce enabled extensions",
    );
    expect(apiSpies.completions).not.toHaveBeenCalled();
  });

  it("does not apply the Chat tool cap to non-OpenAI backend families", () => {
    apiSpies.completions.mockReturnValueOnce({});
    const model = { ...native("kimi-k3"), litellmBackendFamily: "kimi" as const };
    const tools = Array.from({ length: 129 }, (_, index) => ({
      name: `tool-${index}`,
      description: "test tool",
      parameters: { type: "object" as const, properties: {} },
    }));
    const value = controller();

    expect(() => value.stream(model, normalizeContext({ messages: [], tools }))).not.toThrow();
    expect(apiSpies.completions).toHaveBeenCalledOnce();
  });

  it("blocks requests when active credentials have no model host", () => {
    const value = controller({ resolveCredentialRoot: () => undefined });

    expect(() => value.stream(native("missing-root"), normalizeContext({ messages: [] }))).toThrow(
      /Active credentials do not identify a LiteLLM model host.*network refresh/i,
    );
    expect(apiSpies.completions).not.toHaveBeenCalled();
  });

  it("blocks unsupported protocols with actionable configuration guidance", () => {
    const model = foreignApiModel("messages");
    const value = controller();

    expect(() => value.stream(model, normalizeContext({ messages: [] }))).toThrow(
      /declares unsupported protocol "google-generative-ai".*set "api" to one of anthropic-messages, openai-completions, openai-responses/i,
    );
    expect(() => value.streamSimple(model, normalizeContext({ messages: [] }))).toThrow(
      /declares unsupported protocol "google-generative-ai".*models\.json/i,
    );
    expect(apiSpies.completions).not.toHaveBeenCalled();
    expect(apiSpies.responses).not.toHaveBeenCalled();
  });

  it("blocks malformed active credential roots with guided refresh advice", () => {
    const value = controller({ resolveCredentialRoot: () => "not a URL" });

    expect(() => value.stream(native("invalid-root"), normalizeContext({ messages: [] }))).toThrow(
      /Active credentials have an invalid LiteLLM model URL.*network refresh/i,
    );
    expect(apiSpies.completions).not.toHaveBeenCalled();
  });

  it("blocks cached models with invalid URLs", () => {
    const model = { ...native("invalid-url"), baseUrl: "not a URL" };
    const value = controller();

    expect(() => value.stream(model, normalizeContext({ messages: [] }))).toThrow(
      /invalid LiteLLM model URL.*network refresh/i,
    );
    expect(apiSpies.completions).not.toHaveBeenCalled();
  });

  it("retains previous models when discovery rejects", async () => {
    const old = native("old");
    const refreshContext = context([old], true);
    const discover = vi.fn(async () => {
      throw new Error("rejected");
    });
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const value = controller({ discover });

    await expect(value.refreshModels?.(refreshContext)).rejects.toThrow("rejected");
    expect(stderr).toHaveBeenCalledOnce();

    expect(value.getModels()).toEqual([old]);
    expect(refreshContext.publications.every((publication) => publication.persist === undefined)).toBe(true);
  });

  it("retains previous models when discovery is aborted", async () => {
    const old = native("old");
    const refreshContext = context([old], true);
    const abort = new AbortController();
    const discover = vi.fn(async () => {
      abort.abort();
      return discovered("fresh");
    });
    const value = controller({ discover });

    await value.refreshModels?.({ ...refreshContext, signal: abort.signal });

    expect(value.getModels()).toEqual([old]);
    expect(refreshContext.publications.every((publication) => publication.persist === undefined)).toBe(true);
  });

  it("accepts equivalent cached roots after URL canonicalization", () => {
    const baseModel = discovered("model").models[0];
    const models = toNativeModels("litellm", "HTTPS://PROXY.EXAMPLE:443/team-a", [
      baseModel,
      { ...baseModel, id: "messages", api: "anthropic-messages", compat: {} },
    ]);
    const value = controller({ resolveCredentialRoot: () => "https://proxy.example/team-a" });

    expect(value.filterModels?.(models, credential).map(({ api, baseUrl }) => ({ api, baseUrl }))).toEqual([
      { api: "openai-completions", baseUrl: "https://proxy.example/team-a/v1" },
      { api: "anthropic-messages", baseUrl: "https://proxy.example/team-a" },
    ]);
    expect(() => value.stream(models[0], normalizeContext({ messages: [] }))).not.toThrow();
    expect(apiSpies.completions).toHaveBeenCalledWith(
      expect.objectContaining({ baseUrl: "https://proxy.example/team-a/v1" }),
      normalizeContext({ messages: [] }),
      undefined,
    );
  });

  it("keeps proxy-root paths case-sensitive", () => {
    const models = toNativeModels("litellm", "https://gateway.example/Team-A", discovered("model").models);
    const value = controller({ resolveCredentialRoot: () => "https://gateway.example/team-a" });

    expect(value.filterModels?.(models, credential)).toEqual([]);
    expect(() => value.stream(models[0], normalizeContext({ messages: [] }))).toThrow(
      /stale LiteLLM model root.*network refresh/i,
    );
    expect(apiSpies.completions).not.toHaveBeenCalled();
  });

  it("rejects cached models from another same-origin proxy prefix", () => {
    const baseModel = discovered("model").models[0];
    const models = toNativeModels("litellm", "https://gateway.example/team-a", [
      baseModel,
      { ...baseModel, id: "messages", api: "anthropic-messages", compat: {} },
    ]);
    const value = controller({ resolveCredentialRoot: () => "https://gateway.example/team-b" });

    expect(value.filterModels?.(models, credential)).toEqual([]);
    expect(() => value.stream(models[0], normalizeContext({ messages: [] }))).toThrow(
      /stale LiteLLM model root https:\/\/gateway\.example\/team-a.*https:\/\/gateway\.example\/team-b.*network refresh/i,
    );
    expect(() => value.stream(models[1], normalizeContext({ messages: [] }))).toThrow(
      /stale LiteLLM model root https:\/\/gateway\.example\/team-a.*https:\/\/gateway\.example\/team-b.*network refresh/i,
    );
    expect(apiSpies.completions).not.toHaveBeenCalled();
    expect(apiSpies.anthropic).not.toHaveBeenCalled();
  });

  it("preserves explicitly allowed insecure HTTP for Messages and OpenAI dispatch", () => {
    apiSpies.anthropic.mockReturnValueOnce({});
    apiSpies.completions.mockReturnValueOnce({});
    const baseModel = discovered("local").models[0];
    const models = toNativeModels(
      "litellm",
      "http://host.docker.internal",
      [
        { ...baseModel, id: "messages", api: "anthropic-messages", compat: {} },
        { ...baseModel, id: "completions", api: "openai-completions" },
      ],
      true,
    );
    const value = controller({
      allowInsecureHttp: true,
      resolveCredentialRoot: () => "http://host.docker.internal",
    });

    value.stream(models[0], normalizeContext({ messages: [] }));
    value.stream(models[1], normalizeContext({ messages: [] }));

    expect(models.map(({ api, baseUrl }) => ({ api, baseUrl }))).toEqual([
      { api: "anthropic-messages", baseUrl: "http://host.docker.internal" },
      { api: "openai-completions", baseUrl: "http://host.docker.internal/v1" },
    ]);
    expect(apiSpies.anthropic).toHaveBeenCalledWith(
      expect.objectContaining({ baseUrl: "http://host.docker.internal" }),
      normalizeContext({ messages: [] }),
      undefined,
    );
    expect(apiSpies.completions).toHaveBeenCalledWith(
      expect.objectContaining({ baseUrl: "http://host.docker.internal/v1" }),
      normalizeContext({ messages: [] }),
      undefined,
    );
  });

  it("rejects insecure HTTP during projection and request guarding by default", () => {
    expect(() => toNativeModels("litellm", "http://host.docker.internal", discovered("local").models)).toThrow(/HTTPS/);

    const model = {
      ...native("local"),
      baseUrl: "http://host.docker.internal/v1",
    };
    const value = controller({ resolveCredentialRoot: () => "http://host.docker.internal" });

    expect(value.filterModels?.([model], credential)).toEqual([]);
    expect(() => value.stream(model, normalizeContext({ messages: [] }))).toThrow(
      /invalid LiteLLM model URL.*network refresh/i,
    );
    expect(apiSpies.completions).not.toHaveBeenCalled();
  });

  it("routes synthetic Messages models through the Anthropic API", () => {
    apiSpies.anthropic.mockReturnValueOnce({});
    const messagesModel = toNativeModels("litellm", "https://proxy.example/v1", [
      { ...discovered("messages").models[0], api: "anthropic-messages", compat: {} },
    ])[0];
    const value = controller();

    value.stream(messagesModel, normalizeContext({ messages: [] }));

    expect(messagesModel.baseUrl).toBe("https://proxy.example");
    expect(apiSpies.anthropic).toHaveBeenCalledOnce();
    expect(apiSpies.completions).not.toHaveBeenCalled();
    expect(apiSpies.responses).not.toHaveBeenCalled();
  });

  it("routes Chat Completions models through the Completions API", () => {
    apiSpies.completions.mockReturnValueOnce({});
    const value = controller();

    value.stream(native("chat"), normalizeContext({ messages: [] }));

    expect(apiSpies.completions).toHaveBeenCalledOnce();
    expect(apiSpies.responses).not.toHaveBeenCalled();
  });

  it("routes Responses models through the Responses API", () => {
    apiSpies.responses.mockReturnValueOnce({});
    const responseModel = toNativeModels("litellm", "https://proxy.example/v1", [
      { ...discovered("responses").models[0], api: "openai-responses", compat: undefined },
    ])[0];
    const value = controller();

    value.stream(responseModel, normalizeContext({ messages: [] }));

    expect(apiSpies.responses).toHaveBeenCalledOnce();
    expect(apiSpies.completions).not.toHaveBeenCalled();
  });
});

// Reduced groups deliberately use a permanent marker so offline cache reads
// cannot re-authorize metadata that discovery withheld.
describe("discovery cache version transition", () => {
  const legacyModel = () => ({
    ...native("legacy"),
    reasoning: true,
    thinkingLevelMap: { low: null },
    litellmDiscoveryVersion: undefined,
  });
  const legacyMoonshotModel = () => ({
    ...legacyModel(),
    id: "moonshot/kimi-k2.6",
    compat: {
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsStrictMode: false,
      maxTokensField: "max_tokens" as const,
    },
  });
  const restorableV2Model = () => ({
    ...native("moonshot/kimi-k2.6"),
    litellmDiscoveryVersion: 2 as const,
    compat: {
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsStrictMode: false,
      maxTokensField: "max_tokens" as const,
    },
    litellmPolicy: undefined,
  });

  it("refreshes and replaces a legacy store when the network phase runs", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const discover = vi.fn(async () => discovered("fresh"));
    const provider = controller({ discover });
    const legacy = legacyModel();

    await provider.refreshModels?.(context([legacy], false));
    const networkRefresh = context([legacy], true, Date.now());
    await provider.refreshModels?.(networkRefresh);

    expect(discover).toHaveBeenCalledOnce();
    expect(provider.getModels().map((model) => model.id)).toEqual(["fresh"]);
    expect(networkRefresh.publications.find((publication) => publication.persist)?.persist?.models[0]?.id).toBe(
      "fresh",
    );
    expect(stderr).not.toHaveBeenCalled();
  });

  it("restores response policy without rewriting legacy reasoning levels offline", async () => {
    const discover = vi.fn();
    const provider = controller({ discover });
    const legacy = legacyMoonshotModel();

    await provider.refreshModels?.(context([legacy], false));

    expect(discover).not.toHaveBeenCalled();
    expect(provider.getModels()[0]).toMatchObject({
      thinkingLevelMap: { low: null },
      litellmPolicy: {
        normalizeStrictToolMessages: false,
        normalizeThinkTags: true,
        suppressReasoningVisibility: false,
      },
    });
  });

  it("handles mixed stores per entry without warning during offline restore", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const discover = vi.fn();
    const provider = controller({ discover });
    const legacy = legacyModel();
    const v2 = restorableV2Model();

    await provider.refreshModels?.(context([legacy, v2], false));

    expect(discover).not.toHaveBeenCalled();
    expect(provider.getModels()[0]).toEqual(legacy);
    expect((provider.getModels()[1] as ReturnType<typeof restorableV2Model> | undefined)?.litellmPolicy).toEqual({
      normalizeStrictToolMessages: false,
      normalizeThinkTags: true,
      suppressReasoningVisibility: false,
    });
    expect(stderr).not.toHaveBeenCalled();
  });

  it("warns once when repeated refresh attempts leave a mixed legacy store in place", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const discover = vi.fn(async () => {
      throw new Error("refresh failed");
    });
    const provider = controller({ discover });
    const legacy = legacyModel();
    const stored = [legacy, restorableV2Model()];

    await provider.refreshModels?.(context(stored, false));
    await expect(provider.refreshModels?.(context(stored, true))).rejects.toThrow("refresh failed");
    await provider.refreshModels?.(context(stored, false));
    await expect(provider.refreshModels?.(context(stored, true))).rejects.toThrow("refresh failed");

    expect(provider.getModels()[0]).toEqual(legacy);
    expect((provider.getModels()[1] as ReturnType<typeof restorableV2Model> | undefined)?.litellmPolicy).toBeDefined();
    expect(stderr).toHaveBeenCalledTimes(1);
    expect(stderr).toHaveBeenCalledWith(expect.stringContaining("required network refresh failed"));
  });
});

describe("discovery and offline cache parity", () => {
  let fetchSpy: MockInstance<typeof fetch> | undefined;

  afterEach(() => {
    fetchSpy?.mockRestore();
    fetchSpy = undefined;
  });

  it("preserves reduced discovery metadata through the provider cache", async () => {
    const cases = [
      [
        [
          {
            model_name: "openai/gpt-5.5",
            model_info: { id: "only", mode: "chat" },
            litellm_params: { model: "openai/gpt-5.5" },
          },
        ],
        "openai/gpt-5.5",
      ],
      [
        [
          {
            model_name: "openai/gpt-5.5",
            model_info: { id: "only", mode: "chat" },
            litellm_params: { model: "openai/gpt-5.5-internal-preview" },
          },
        ],
        "openai/gpt-5.5 (incomplete metadata)",
      ],
      [
        [
          {
            model_name: "openai/gpt-5.5",
            model_info: { id: "only", mode: "chat" },
            litellm_params: { model: "internal/mystery" },
          },
        ],
        "openai/gpt-5.5 (incomplete metadata)",
      ],
      [
        [
          { model_name: "openai/gpt-5.5", model_info: { id: "a", mode: "chat" } },
          {
            model_name: "openai/gpt-5.5",
            model_info: { id: "b", mode: "chat" },
            litellm_params: { model: "internal/mystery" },
          },
        ],
        "openai/gpt-5.5 (incomplete metadata)",
      ],
      [
        [
          { model_name: "openai/gpt-5.5", model_info: { id: "same", mode: "chat" } },
          { model_name: "openai/gpt-5.5", model_info: { id: "same", mode: "chat", max_input_tokens: 64_000 } },
        ],
        "openai/gpt-5.5 (incomplete metadata)",
      ],
    ] as const;

    for (const [data, expectedName] of cases) {
      fetchSpy?.mockRestore();
      fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
        new Response(JSON.stringify({ data }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      );
      const onlineResult = await discoverModels("https://proxy.example/v1", "sk-test", { modelsDev: false });
      const onlineModels = toNativeModels("litellm", "https://proxy.example/v1", onlineResult.models);

      expect(onlineModels).toHaveLength(1);
      expect(onlineModels[0]?.name).toBe(expectedName);

      const provider = controller();
      await provider.refreshModels?.(context(onlineModels, false));

      expect(provider.getModels()).toEqual(onlineModels);
    }
  });
});
