import {
  type Credential,
  createProvider,
  type Model,
  type Provider,
  type ProviderAuth,
  type RefreshModelsContext,
} from "@earendil-works/pi-ai";
import { openAICompletionsApi, openAIResponsesApi } from "@earendil-works/pi-ai/compat";
import { enrichCachedModel } from "./discover.js";
import type { DiscoveredModel, DiscoveryResult, LiteLLMApi } from "./types.js";

type ModelStore = RefreshModelsContext["store"];

export type LiteLLMProviderOptions = {
  id: string;
  name: string;
  baseUrl: string;
  auth: ProviderAuth;
  discover(credential: Credential, signal?: AbortSignal): Promise<DiscoveryResult & { baseUrl?: string }>;
};

export function toNativeModels(
  provider: string,
  baseUrl: string,
  models: readonly DiscoveredModel[],
): Model<LiteLLMApi>[] {
  return models.map((model) => ({
    ...model,
    provider,
    api: model.api ?? "openai-completions",
    baseUrl,
  })) as Model<LiteLLMApi>[];
}

export function createLiteLLMProvider(options: LiteLLMProviderOptions): Provider<LiteLLMApi> {
  const provider = createProvider<LiteLLMApi>({
    id: options.id,
    name: options.name,
    baseUrl: options.baseUrl,
    auth: options.auth,
    models: [],
    async fetchModels(context) {
      if (!context.credential) throw new Error("LiteLLM model discovery requires a credential");
      const result = await options.discover(context.credential, context.signal);
      return toNativeModels(options.id, result.baseUrl ?? options.baseUrl, result.models);
    },
    api: {
      "openai-completions": openAICompletionsApi(),
      "openai-responses": openAIResponsesApi(),
    },
  });
  const refreshModels = provider.refreshModels;
  if (!refreshModels) return provider;
  return {
    ...provider,
    refreshModels: (context) => refreshModels({ ...context, store: enrichingStore(context.store) }),
  };
}

/** Re-enriches the cached catalog on read so stale entries gain metadata later releases know about. */
function enrichingStore(store: ModelStore): ModelStore {
  return {
    read: async () => {
      const entry = await store.read();
      return entry && { ...entry, models: entry.models.map(enrichCachedModel) };
    },
    write: (entry) => store.write(entry),
    delete: () => store.delete(),
  };
}
