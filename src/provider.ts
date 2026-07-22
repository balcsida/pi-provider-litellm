import {
  type Credential,
  createProvider,
  type Model,
  type Provider,
  type ProviderAuth,
  type RefreshModelsContext,
} from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { openAIResponsesApi } from "@earendil-works/pi-ai/api/openai-responses.lazy";
import type { DiscoveredModel, DiscoveryResult, LiteLLMApi } from "./types.js";

export type LiteLLMProviderController = {
  provider: Provider<LiteLLMApi>;
  forceRefresh(signal?: AbortSignal): Promise<DiscoveryResult>;
};

export type LiteLLMProviderOptions = {
  id: string;
  name: string;
  baseUrl: string;
  auth: ProviderAuth;
  legacyModels(context: RefreshModelsContext): Promise<readonly Model<LiteLLMApi>[] | undefined>;
  discover(credential: Credential, signal?: AbortSignal): Promise<DiscoveryResult>;
  onRefresh?(models: readonly Model<LiteLLMApi>[], credential?: Credential): Promise<void> | void;
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

export function createLiteLLMProvider(options: LiteLLMProviderOptions): LiteLLMProviderController {
  let lastContext: RefreshModelsContext | undefined;
  let lastDiscovery: DiscoveryResult | undefined;
  let inflightRefresh: Promise<void> | undefined;
  const inner = createProvider<LiteLLMApi>({
    id: options.id,
    name: options.name,
    baseUrl: options.baseUrl,
    auth: options.auth,
    models: [],
    async fetchModels(context) {
      if (!context.credential) throw new Error("LiteLLM model discovery requires a credential");
      const result = await options.discover(context.credential, context.signal);
      lastDiscovery = result;
      return toNativeModels(options.id, options.baseUrl, result.models);
    },
    api: {
      "openai-completions": openAICompletionsApi(),
      "openai-responses": openAIResponsesApi(),
    },
  });
  const innerRefresh = inner.refreshModels!;
  const provider: Provider<LiteLLMApi> = {
    ...inner,
    refreshModels(context) {
      lastContext = context;
      inflightRefresh ??= (async () => {
        try {
          if (!(await context.store.read())) {
            const legacy = await options.legacyModels(context);
            if (legacy) await context.store.write({ models: legacy, checkedAt: Date.now() });
          }
          await innerRefresh(context);
          await options.onRefresh?.(provider.getModels(), context.credential);
        } finally {
          inflightRefresh = undefined;
        }
      })();
      return inflightRefresh;
    },
  };

  return {
    provider,
    async forceRefresh(signal) {
      if (!lastContext) throw new Error("LiteLLM provider has not been initialized");
      lastDiscovery = undefined;
      await provider.refreshModels?.({ ...lastContext, allowNetwork: true, force: true, signal });
      if (signal?.aborted) throw new Error("LiteLLM discovery was aborted");
      if (!lastDiscovery) throw new Error("LiteLLM discovery did not return a result");
      return lastDiscovery;
    },
  };
}
