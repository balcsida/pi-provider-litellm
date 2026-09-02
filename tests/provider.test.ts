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
import type { DiscoveryResult } from "../src/types.js";

const apiSpies = vi.hoisted(() => ({ anthropic: vi.fn(), completions: vi.fn(), responses: vi.fn() }));
vi.mock("@earendil-works/pi-ai/compat", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@earendil-works/pi-ai/compat")>()),
  anthropicMessagesApi: () => ({ stream: apiSpies.anthropic, streamSimple: apiSpies.anthropic }),
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

function native(id: string): Model<"anthropic-messages" | "openai-completions" | "openai-responses"> {
  return toNativeModels("litellm", "https://proxy.example/v1", discovered(id).models)[0];
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

  it("projects each protocol from one normalized proxy root", () => {
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
  it("restores stored models offline without discovery", async () => {
    const discover = vi.fn(async () => discovered("fresh"));
    const value = controller({ discover });

    await value.refreshModels?.(context([native("stored")], false));

    expect(value.getModels()).toEqual([native("stored")]);
    expect(discover).not.toHaveBeenCalled();
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
          },
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
    const legacyFallback: Model<Api> = {
      ...native("opus-5"),
      name: "opus-5 (no metadata)",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128_000,
      maxTokens: 16_384,
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

  it("reprojects matching cached roots and rejects stale or placeholder roots", () => {
    const baseModel = discovered("model").models[0];
    const models = toNativeModels("litellm", "https://proxy.example", [
      baseModel,
      { ...baseModel, id: "messages", api: "anthropic-messages", compat: {} },
    ]);
    const value = controller();

    expect(value.filterModels?.(models, credential).map(({ api, baseUrl }) => ({ api, baseUrl }))).toEqual([
      { api: "openai-completions", baseUrl: "https://proxy.example/v1" },
      { api: "anthropic-messages", baseUrl: "https://proxy.example" },
    ]);

    const stale = controller({ resolveCredentialRoot: () => "https://other.example" });
    expect(stale.filterModels?.(models, credential)).toEqual([]);

    const placeholder = controller({ resolveCredentialRoot: () => "https://litellm.example.com" });
    expect(placeholder.filterModels?.(models, credential)).toEqual([]);
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
    expect(() => value.stream(models[0], { messages: [] })).not.toThrow();
    expect(apiSpies.completions).toHaveBeenCalledWith(
      expect.objectContaining({ baseUrl: "https://proxy.example/team-a/v1" }),
      { messages: [] },
      undefined,
    );
  });

  it("keeps proxy-root paths case-sensitive", () => {
    const models = toNativeModels("litellm", "https://gateway.example/Team-A", discovered("model").models);
    const value = controller({ resolveCredentialRoot: () => "https://gateway.example/team-a" });

    expect(value.filterModels?.(models, credential)).toEqual([]);
    expect(() => value.stream(models[0], { messages: [] })).toThrow(/stale LiteLLM model root.*network refresh/i);
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
    expect(() => value.stream(models[0], { messages: [] })).toThrow(
      /stale LiteLLM model root https:\/\/gateway\.example\/team-a.*https:\/\/gateway\.example\/team-b.*network refresh/i,
    );
    expect(() => value.stream(models[1], { messages: [] })).toThrow(
      /stale LiteLLM model root https:\/\/gateway\.example\/team-a.*https:\/\/gateway\.example\/team-b.*network refresh/i,
    );
    expect(apiSpies.completions).not.toHaveBeenCalled();
    expect(apiSpies.anthropic).not.toHaveBeenCalled();
  });

  it("blocks stale roots before protocol dispatch", () => {
    const value = controller({ resolveCredentialRoot: () => "https://other.example" });

    expect(() => value.stream(native("stale"), { messages: [] })).toThrow(/stale LiteLLM model root.*network refresh/i);
    expect(() => value.streamSimple(native("stale"), { messages: [] })).toThrow(
      /stale LiteLLM model root.*network refresh/i,
    );
    expect(apiSpies.completions).not.toHaveBeenCalled();
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

    value.stream(models[0], { messages: [] });
    value.stream(models[1], { messages: [] });

    expect(models.map(({ api, baseUrl }) => ({ api, baseUrl }))).toEqual([
      { api: "anthropic-messages", baseUrl: "http://host.docker.internal" },
      { api: "openai-completions", baseUrl: "http://host.docker.internal/v1" },
    ]);
    expect(apiSpies.anthropic).toHaveBeenCalledWith(
      expect.objectContaining({ baseUrl: "http://host.docker.internal" }),
      { messages: [] },
      undefined,
    );
    expect(apiSpies.completions).toHaveBeenCalledWith(
      expect.objectContaining({ baseUrl: "http://host.docker.internal/v1" }),
      { messages: [] },
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
    expect(() => value.stream(model, { messages: [] })).toThrow(/invalid LiteLLM model URL.*network refresh/i);
    expect(apiSpies.completions).not.toHaveBeenCalled();
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

  it("blocks requests when active credentials have no model host", () => {
    const value = controller({ resolveCredentialRoot: () => undefined });

    expect(() => value.stream(native("missing-root"), { messages: [] })).toThrow(
      /Active credentials do not identify a LiteLLM model host.*network refresh/i,
    );
    expect(apiSpies.completions).not.toHaveBeenCalled();
  });

  it("blocks cached models with unsupported transports", () => {
    const model = { ...native("legacy"), api: "legacy-completions" } as unknown as Model<"openai-completions">;
    const value = controller();

    expect(() => value.stream(model, { messages: [] })).toThrow(/unsupported LiteLLM transport.*network refresh/i);
    expect(apiSpies.completions).not.toHaveBeenCalled();
  });

  it("blocks cached models with invalid URLs", () => {
    const model = { ...native("invalid-url"), baseUrl: "not a URL" };
    const value = controller();

    expect(value.filterModels?.([model], credential)).toEqual([]);
    expect(() => value.stream(model, { messages: [] })).toThrow(/invalid LiteLLM model URL.*network refresh/i);
    expect(apiSpies.completions).not.toHaveBeenCalled();
  });

  it("blocks invalid active credential roots", () => {
    const value = controller({ resolveCredentialRoot: () => "not a URL" });

    expect(value.filterModels?.([native("invalid-active-root")], credential)).toEqual([]);
    expect(() => value.stream(native("invalid-active-root"), { messages: [] })).toThrow(
      /Active credentials has an invalid LiteLLM model URL.*network refresh/i,
    );
    expect(apiSpies.completions).not.toHaveBeenCalled();
  });

  it("retains previous models when discovery rejects", async () => {
    const refreshContext = context([native("old")], true);
    const discover = vi.fn(async () => {
      throw new Error("rejected");
    });
    const value = controller({ discover });

    await expect(value.refreshModels?.(refreshContext)).rejects.toThrow("rejected");

    expect(value.getModels()).toEqual([native("old")]);
    expect(refreshContext.publications.every((publication) => publication.persist === undefined)).toBe(true);
  });

  it("retains previous models when discovery is aborted", async () => {
    const refreshContext = context([native("old")], true);
    const abort = new AbortController();
    const discover = vi.fn(async () => {
      abort.abort();
      return discovered("fresh");
    });
    const value = controller({ discover });

    await value.refreshModels?.({ ...refreshContext, signal: abort.signal });

    expect(value.getModels()).toEqual([native("old")]);
    expect(refreshContext.publications.every((publication) => publication.persist === undefined)).toBe(true);
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
    expect(apiSpies.anthropic).not.toHaveBeenCalled();
  });

  it("routes synthetic Messages models through the Anthropic API", () => {
    apiSpies.anthropic.mockReturnValueOnce({});
    const messagesModel = toNativeModels("litellm", "https://proxy.example/v1", [
      { ...discovered("messages").models[0], api: "anthropic-messages", compat: {} },
    ])[0];
    const value = controller();

    value.stream(messagesModel, { messages: [] });

    expect(messagesModel.baseUrl).toBe("https://proxy.example");
    expect(apiSpies.anthropic).toHaveBeenCalledOnce();
    expect(apiSpies.completions).not.toHaveBeenCalled();
    expect(apiSpies.responses).not.toHaveBeenCalled();
  });
});
