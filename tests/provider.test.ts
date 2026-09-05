import type {
  Api,
  Credential,
  Model,
  ModelsPublication,
  ProviderAuth,
  RefreshModelsContext,
} from "@earendil-works/pi-ai";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createLiteLLMProvider, toNativeModels } from "../src/provider.js";
import type { DiscoveryResult, LiteLLMModel } from "../src/types.js";

const apiSpies = vi.hoisted(() => ({ completions: vi.fn(), responses: vi.fn() }));
vi.mock("@earendil-works/pi-ai/compat", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@earendil-works/pi-ai/compat")>()),
  openAICompletionsApi: () => ({ stream: apiSpies.completions, streamSimple: apiSpies.completions }),
  openAIResponsesApi: () => ({ stream: apiSpies.responses, streamSimple: apiSpies.responses }),
}));

const credential: Credential = { type: "api_key", key: "secret" };
const auth: ProviderAuth = {
  apiKey: { name: "API key", resolve: async () => ({ auth: { apiKey: "secret" } }) },
};

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

function native(id: string): Model<"openai-completions" | "openai-responses"> {
  return toNativeModels("litellm", "https://proxy.example/v1", discovered(id).models)[0];
}

function foreignApiModel(id: string, api = "anthropic-messages"): Model<"openai-completions"> {
  return { ...native(id), api } as unknown as Model<"openai-completions">;
}

type TestRefreshContext = RefreshModelsContext & { publications: ModelsPublication[] };

function context(initial: readonly Model<Api>[] | undefined, allowNetwork: boolean): TestRefreshContext {
  const publications: ModelsPublication[] = [];
  return {
    stored: initial ? { models: initial, checkedAt: 1 } : undefined,
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
    const [completions, responses] = toNativeModels("litellm", "https://proxy.example/v1/", [
      { ...baseModel, id: "completions", api: "openai-completions" },
      { ...baseModel, id: "responses", api: "openai-responses", compat: undefined },
    ]);

    expect([completions, responses].map(({ api, baseUrl }) => ({ api, baseUrl }))).toEqual([
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
    const stored = { ...native("stored"), litellmDiscoveryVersion: 2 as const };

    await value.refreshModels?.(context([stored], false));

    expect(value.getModels()).toEqual([stored]);
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
      litellmDiscoveryVersion: 2 as const,
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
    expect(stderr).toHaveBeenCalledWith(expect.stringMatching(/version does not match 2.*network refresh failed/));
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
            litellmDiscoveryVersion: 2,
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
        thinkingLevelMap: { xhigh: "xhigh", max: "max" },
        provider: "litellm",
        api: "openai-completions",
        baseUrl: "https://proxy.example/v1",
      }),
    ]);
    expect(discover).not.toHaveBeenCalled();
  });

  it("keeps partially enriched stale cached aliases unchanged offline", async () => {
    const legacyFallback: LiteLLMModel = {
      ...native("opus-5"),
      name: "opus-5 (no metadata)",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128_000,
      maxTokens: 16_384,
      litellmDiscoveryVersion: 2,
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

      expect(value.getModels()).toEqual([cached]);
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

    expect(() => value.stream(model, { messages: [] })).toThrow(/placeholder LiteLLM model host.*network refresh/i);
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

    expect(() => value.stream(staleModel, { messages: [], tools: oversizedTools })).toThrow(
      /stale LiteLLM model host.*network refresh/i,
    );
    expect(() => value.streamSimple(native("stale"), { messages: [] })).toThrow(
      /stale LiteLLM model host.*network refresh/i,
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

    value.stream(model, { messages: [] });

    expect(model.baseUrl).toBe("http://host.docker.internal/v1");
    expect(apiSpies.completions).toHaveBeenCalledWith(
      expect.objectContaining({ baseUrl: "http://host.docker.internal/v1" }),
      { messages: [] },
      undefined,
    );
  });

  it.each(["stream", "streamSimple"] as const)(
    "passes the AuthResult env root and API key to resolveCredentialRoot for %s",
    (method) => {
      const resolveCredentialRoot = vi.fn(() => "https://proxy.example");
      const value = controller({ resolveCredentialRoot });

      value[method](
        native(method),
        { messages: [] },
        {
          apiKey: "resolved-key",
          env: { LITELLM_BASE_URL: "https://auth-result.example" },
        },
      );

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

    expect(() => value.stream(model, { messages: [], tools })).toThrow(
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

    expect(() => value.stream(model, { messages: [], tools })).not.toThrow();
    expect(apiSpies.completions).toHaveBeenCalledOnce();
  });

  it("blocks requests when active credentials have no model host", () => {
    const value = controller({ resolveCredentialRoot: () => undefined });

    expect(() => value.stream(native("missing-root"), { messages: [] })).toThrow(
      /Active credentials do not identify a LiteLLM model host.*network refresh/i,
    );
    expect(apiSpies.completions).not.toHaveBeenCalled();
  });

  it("blocks unsupported protocols with actionable configuration guidance", () => {
    const model = foreignApiModel("messages");
    const value = controller();

    expect(() => value.stream(model, { messages: [] })).toThrow(
      /declares unsupported protocol "anthropic-messages".*set "api" to one of openai-completions, openai-responses/i,
    );
    expect(() => value.streamSimple(model, { messages: [] })).toThrow(
      /declares unsupported protocol "anthropic-messages".*models\.json/i,
    );
    expect(apiSpies.completions).not.toHaveBeenCalled();
    expect(apiSpies.responses).not.toHaveBeenCalled();
  });

  it("blocks malformed active credential roots with guided refresh advice", () => {
    const value = controller({ resolveCredentialRoot: () => "not a URL" });

    expect(() => value.stream(native("invalid-root"), { messages: [] })).toThrow(
      /Active credentials have an invalid LiteLLM model URL.*network refresh/i,
    );
    expect(apiSpies.completions).not.toHaveBeenCalled();
  });

  it("blocks cached models with invalid URLs", () => {
    const model = { ...native("invalid-url"), baseUrl: "not a URL" };
    const value = controller();

    expect(() => value.stream(model, { messages: [] })).toThrow(/invalid LiteLLM model URL.*network refresh/i);
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

  it("routes Chat Completions models through the Completions API", () => {
    apiSpies.completions.mockReturnValueOnce({});
    const value = controller();

    value.stream(native("chat"), { messages: [] });

    expect(apiSpies.completions).toHaveBeenCalledOnce();
    expect(apiSpies.responses).not.toHaveBeenCalled();
  });

  it("routes Responses models through the Responses API", () => {
    apiSpies.responses.mockReturnValueOnce({});
    const responseModel = toNativeModels("litellm", "https://proxy.example/v1", [
      { ...discovered("responses").models[0], api: "openai-responses", compat: undefined },
    ])[0];
    const value = controller();

    value.stream(responseModel, { messages: [] });

    expect(apiSpies.responses).toHaveBeenCalledOnce();
    expect(apiSpies.completions).not.toHaveBeenCalled();
  });
});
