import type {
  Api,
  Credential,
  Model,
  ProviderAuth,
  ProviderModelsStore,
  RefreshModelsContext,
} from "@earendil-works/pi-ai";
import { beforeEach, describe, expect, expectTypeOf, it, vi } from "vitest";
import { createLiteLLMProvider, toNativeModels } from "../src/provider.js";
import type { DiscoveredModel, DiscoveryResult } from "../src/types.js";

const apiSpies = vi.hoisted(() => ({
  anthropic: vi.fn(),
  completions: vi.fn(),
  responses: vi.fn(),
}));
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

{
  const invalidMessagesModel: DiscoveredModel = {
    ...discovered("messages").models[0],
    api: "anthropic-messages",
    compat: {
      // @ts-expect-error OpenAI compatibility fields are invalid for Messages models.
      supportsStore: false,
    },
  };
  expectTypeOf(invalidMessagesModel).toMatchTypeOf<DiscoveredModel>();
}

function native(id: string): Model<"anthropic-messages" | "openai-completions" | "openai-responses"> {
  return toNativeModels("litellm", "https://proxy.example/v1", discovered(id).models)[0];
}

function store(initial?: readonly Model<Api>[]) {
  let entry = initial ? { models: initial, checkedAt: 1 } : undefined;
  const value: ProviderModelsStore = {
    read: vi.fn(async () => entry),
    write: vi.fn(async (next) => {
      entry = next;
    }),
    delete: vi.fn(async () => {
      entry = undefined;
    }),
  };
  return value;
}

function context(modelsStore: ProviderModelsStore, allowNetwork: boolean): RefreshModelsContext {
  return { store: modelsStore, allowNetwork, credential };
}

function controller(overrides: Partial<Parameters<typeof createLiteLLMProvider>[0]> = {}) {
  return createLiteLLMProvider({
    id: "litellm",
    name: "LiteLLM",
    baseUrl: "https://proxy.example/v1",
    auth,
    discover: vi.fn(async () => discovered("fresh")),
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

  it("derives each protocol's request base from URLs with or without /v1", () => {
    const baseModel = discovered("model").models[0];
    const models = toNativeModels("litellm", "https://proxy.example/v1/", [
      baseModel,
      { ...baseModel, id: "responses", api: "openai-responses", compat: {} },
      { ...baseModel, id: "messages", api: "anthropic-messages", compat: {} },
    ]);

    expect(models.map(({ api, baseUrl }) => ({ api, baseUrl }))).toEqual([
      { api: "openai-completions", baseUrl: "https://proxy.example/v1" },
      { api: "openai-responses", baseUrl: "https://proxy.example/v1" },
      { api: "anthropic-messages", baseUrl: "https://proxy.example" },
    ]);
  });

  it("keeps Messages compatibility metadata Anthropic-shaped", () => {
    const baseModel = discovered("messages").models[0];
    const messagesModel = {
      ...baseModel,
      api: "anthropic-messages",
      compat: { forceAdaptiveThinking: true },
    } as const;

    expect(toNativeModels("litellm", "https://proxy.example/v1", [messagesModel])[0]).toMatchObject({
      api: "anthropic-messages",
      compat: { forceAdaptiveThinking: true },
    });
  });
});

describe("createLiteLLMProvider", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("delegates native store restoration directly to createProvider", async () => {
    const modelsStore = store([native("stored")]);
    const value = controller();

    await value.refreshModels?.(context(modelsStore, false));

    expect(modelsStore.read).toHaveBeenCalledOnce();
  });

  it("restores stored models offline without discovery", async () => {
    const discover = vi.fn(async () => discovered("fresh"));
    const value = controller({ discover });

    await value.refreshModels?.(context(store([native("stored")]), false));

    expect(value.getModels()).toEqual([native("stored")]);
    expect(discover).not.toHaveBeenCalled();
  });

  it("publishes and persists successful discovery", async () => {
    const modelsStore = store([native("old")]);
    const value = controller({ discover: vi.fn(async () => discovered("fresh")) });

    await value.refreshModels?.(context(modelsStore, true));

    expect(value.getModels()).toEqual([native("fresh")]);
    expect(modelsStore.write).toHaveBeenCalledOnce();
    expect(modelsStore.write).toHaveBeenCalledWith(expect.objectContaining({ models: [native("fresh")] }));
  });

  it("publishes discovered models with the credential URL", async () => {
    const value = controller({
      discover: vi.fn(async () => ({
        ...discovered("fresh"),
        baseUrl: "https://credential.example/v1",
      })),
    });

    await value.refreshModels?.(context(store(), true));

    expect(value.getModels()[0]?.baseUrl).toBe("https://credential.example/v1");
  });

  it("projects credential-aware bases at the availability boundary", () => {
    const value = controller({
      credentialBaseUrl: (current) =>
        current.type === "oauth" && typeof current.baseUrl === "string" ? current.baseUrl : undefined,
    });
    const baseModel = discovered("model").models[0];
    const models = toNativeModels("litellm", "https://configured.example/v1", [
      baseModel,
      { ...baseModel, id: "messages", api: "anthropic-messages", compat: {} },
    ]);
    const oauthCredential: Credential = {
      type: "oauth",
      access: "token",
      refresh: "",
      expires: Number.MAX_SAFE_INTEGER,
      baseUrl: "https://credential.example/v1",
    };

    expect(value.filterModels?.(models, oauthCredential).map(({ api, baseUrl }) => ({ api, baseUrl }))).toEqual([
      { api: "openai-completions", baseUrl: "https://credential.example/v1" },
      { api: "anthropic-messages", baseUrl: "https://credential.example" },
    ]);
    expect(value.filterModels?.(models, credential)).toBe(models);
  });

  it("retains previous models when discovery rejects", async () => {
    const modelsStore = store([native("old")]);
    const discover = vi.fn(async () => {
      throw new Error("rejected");
    });
    const value = controller({ discover });

    await expect(value.refreshModels?.(context(modelsStore, true))).rejects.toThrow("rejected");

    expect(value.getModels()).toEqual([native("old")]);
    expect(modelsStore.write).not.toHaveBeenCalled();
  });

  it("retains previous models when discovery is aborted", async () => {
    const modelsStore = store([native("old")]);
    const abort = new AbortController();
    const discover = vi.fn(async () => {
      abort.abort();
      return discovered("fresh");
    });
    const value = controller({ discover });

    await value.refreshModels?.({ ...context(modelsStore, true), signal: abort.signal });

    expect(value.getModels()).toEqual([native("old")]);
    expect(modelsStore.write).not.toHaveBeenCalled();
  });

  it("shares one discovery across concurrent refreshes", async () => {
    let release!: (result: DiscoveryResult) => void;
    const pending = new Promise<DiscoveryResult>((resolve) => {
      release = resolve;
    });
    const discover = vi.fn(() => pending);
    const modelsStore = store([native("old")]);
    const value = controller({ discover });

    const first = value.refreshModels?.(context(modelsStore, true));
    const second = value.refreshModels?.(context(modelsStore, true));
    release(discovered("fresh"));
    await Promise.all([first, second]);

    expect(discover).toHaveBeenCalledOnce();
  });

  it("routes Responses models through the Responses API", async () => {
    apiSpies.responses.mockReturnValueOnce({});
    const responseModel = toNativeModels("litellm", "https://proxy.example/v1", [
      { ...discovered("responses").models[0], api: "openai-responses", compat: {} },
    ])[0];
    const value = controller();

    value.stream(responseModel, { messages: [] });

    expect(apiSpies.responses).toHaveBeenCalledOnce();
    expect(apiSpies.completions).not.toHaveBeenCalled();
    expect(apiSpies.anthropic).not.toHaveBeenCalled();
  });

  it("routes Messages models through the Anthropic API with the proxy root", () => {
    apiSpies.anthropic.mockReturnValueOnce({});
    const messagesModel = toNativeModels("litellm", "https://proxy.example/v1/", [
      { ...discovered("messages").models[0], api: "anthropic-messages", compat: {} },
    ])[0];
    const value = controller();

    value.stream(messagesModel, { messages: [] });

    expect(messagesModel.baseUrl).toBe("https://proxy.example");
    expect(`${messagesModel.baseUrl}/v1/messages`).toBe("https://proxy.example/v1/messages");
    expect(apiSpies.anthropic).toHaveBeenCalledOnce();
    expect(apiSpies.anthropic).toHaveBeenCalledWith(messagesModel, { messages: [] }, undefined);
    expect(apiSpies.completions).not.toHaveBeenCalled();
    expect(apiSpies.responses).not.toHaveBeenCalled();
  });
});
