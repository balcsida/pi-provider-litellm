import type {
  Api,
  Credential,
  Model,
  ProviderAuth,
  ProviderModelsStore,
  RefreshModelsContext,
} from "@earendil-works/pi-ai";
import { describe, expect, it, vi } from "vitest";
import { createLiteLLMProvider, toNativeModels } from "../src/provider.js";
import type { DiscoveryResult } from "../src/types.js";

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

function native(id: string): Model<"openai-completions"> {
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
    legacyModels: vi.fn(async () => undefined),
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
});

describe("createLiteLLMProvider", () => {
  it("restores stored models offline without discovery", async () => {
    const discover = vi.fn(async () => discovered("fresh"));
    const value = controller({ discover });

    await value.provider.refreshModels?.(context(store([native("stored")]), false));

    expect(value.provider.getModels()).toEqual([native("stored")]);
    expect(discover).not.toHaveBeenCalled();
  });

  it("imports a valid legacy cache into an empty native store", async () => {
    const modelsStore = store();
    const legacyModels = vi.fn(async () => [native("legacy")]);
    const value = controller({ legacyModels });

    await value.provider.refreshModels?.(context(modelsStore, false));

    expect(legacyModels).toHaveBeenCalledOnce();
    expect(modelsStore.write).toHaveBeenCalledOnce();
    expect(value.provider.getModels()).toEqual([native("legacy")]);
  });

  it("publishes and persists successful discovery", async () => {
    const modelsStore = store([native("old")]);
    const value = controller({ discover: vi.fn(async () => discovered("fresh")) });

    await value.provider.refreshModels?.(context(modelsStore, true));

    expect(value.provider.getModels()).toEqual([native("fresh")]);
    expect(modelsStore.write).toHaveBeenCalledOnce();
    expect(modelsStore.write).toHaveBeenCalledWith(expect.objectContaining({ models: [native("fresh")] }));
  });

  it("retains previous models when discovery rejects", async () => {
    const modelsStore = store([native("old")]);
    const discover = vi.fn(async () => {
      throw new Error("rejected");
    });
    const value = controller({ discover });

    await expect(value.provider.refreshModels?.(context(modelsStore, true))).rejects.toThrow("rejected");

    expect(value.provider.getModels()).toEqual([native("old")]);
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

    await value.provider.refreshModels?.({ ...context(modelsStore, true), signal: abort.signal });

    expect(value.provider.getModels()).toEqual([native("old")]);
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

    const first = value.provider.refreshModels?.(context(modelsStore, true));
    const second = value.provider.refreshModels?.(context(modelsStore, true));
    release(discovered("fresh"));
    await Promise.all([first, second]);

    expect(discover).toHaveBeenCalledOnce();
  });

  it("force refreshes with the last Pi context and supplied signal", async () => {
    const discover = vi.fn(async () => discovered("fresh"));
    const onRefresh = vi.fn();
    const legacyModels = vi.fn(async () => undefined);
    const modelsStore = store([native("old")]);
    const value = controller({ discover, legacyModels, onRefresh });
    await value.provider.refreshModels?.(context(modelsStore, false));
    await modelsStore.delete();
    const abort = new AbortController();

    await expect(value.forceRefresh(abort.signal)).resolves.toEqual(discovered("fresh"));

    expect(discover).toHaveBeenCalledWith(credential, abort.signal);
    expect(legacyModels).toHaveBeenLastCalledWith(expect.objectContaining({ allowNetwork: true, force: true }));
    expect(onRefresh).toHaveBeenLastCalledWith([native("fresh")], credential);
    expect(modelsStore.write).toHaveBeenCalledOnce();
  });
});
