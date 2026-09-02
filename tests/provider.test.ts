import type {
  Api,
  Credential,
  Model,
  ModelsPublication,
  ProviderAuth,
  RefreshModelsContext,
} from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, type MockInstance, vi } from "vitest";
import { discoverModels } from "../src/discover.js";
import { createLiteLLMProvider, toNativeModels } from "../src/provider.js";
import type { DiscoveryResult } from "../src/types.js";

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
    },
  ],
});

function native(id: string): Model<"openai-completions" | "openai-responses"> {
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

  it("preserves a discovered Responses API and defaults missing APIs to Completions", () => {
    const [responses, completions] = toNativeModels("litellm", "https://proxy.example/v1", [
      { ...discovered("responses").models[0], api: "openai-responses" },
      discovered("completions").models[0],
    ]);

    expect(responses.api).toBe("openai-responses");
    expect(completions.api).toBe("openai-completions");
  });
});

describe("createLiteLLMProvider", () => {
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

  it("routes Responses models through the Responses API", async () => {
    apiSpies.responses.mockReturnValueOnce({});
    const responseModel = toNativeModels("litellm", "https://proxy.example/v1", [
      { ...discovered("responses").models[0], api: "openai-responses" },
    ])[0];
    const value = controller();

    value.stream(responseModel, { messages: [] });

    expect(apiSpies.responses).toHaveBeenCalledOnce();
    expect(apiSpies.completions).not.toHaveBeenCalled();
  });
});

// Reduced groups deliberately use a permanent marker so offline cache reads
// cannot re-authorize metadata that discovery withheld.
describe("discovery and offline cache parity", () => {
  let fetchSpy: MockInstance<typeof fetch> | undefined;

  afterEach(() => {
    fetchSpy?.mockRestore();
    fetchSpy = undefined;
  });

  it("preserves withheld singleton and grouped catalog authority", async () => {
    const cases = [
      [
        {
          model_name: "openai/gpt-5.5",
          model_info: { id: "only", mode: "chat" },
          litellm_params: { model: "openai/gpt-5.5-internal-preview" },
        },
      ],
      [
        {
          model_name: "openai/gpt-5.5",
          model_info: { id: "only", mode: "chat" },
          litellm_params: { model: "internal/mystery" },
        },
      ],
      [
        { model_name: "openai/gpt-5.5", model_info: { id: "a", mode: "chat" } },
        {
          model_name: "openai/gpt-5.5",
          model_info: { id: "b", mode: "chat" },
          litellm_params: { model: "internal/mystery" },
        },
      ],
      [
        { model_name: "openai/gpt-5.5", model_info: { id: "same", mode: "chat" } },
        { model_name: "openai/gpt-5.5", model_info: { id: "same", mode: "chat", max_input_tokens: 64_000 } },
      ],
    ];

    for (const data of cases) {
      fetchSpy?.mockRestore();
      fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
        new Response(JSON.stringify({ data }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      );
      const onlineResult = await discoverModels("https://proxy.example/v1", "sk-test", {});
      const onlineModels = toNativeModels("litellm", "https://proxy.example/v1", onlineResult.models);

      expect(onlineModels).toHaveLength(1);
      expect(onlineModels[0]?.name).toBe("openai/gpt-5.5 (incomplete metadata)");

      const provider = controller();
      await provider.refreshModels?.(context(onlineModels, false));

      expect(provider.getModels()).toEqual(onlineModels);
    }
  });
});
