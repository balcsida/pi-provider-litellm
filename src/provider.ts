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

export function createLiteLLMProvider(options: LiteLLMProviderOptions): LiteLLMProviderController {
  let lastContext: RefreshModelsContext | undefined;
  let lastDiscovery: DiscoveryResult | undefined;
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
      return toNativeModels(options.id, result.baseUrl ?? options.baseUrl, result.models);
    },
    api: {
      "openai-completions": openAICompletionsApi(),
      "openai-responses": openAIResponsesApi(),
    },
  });
  const innerRefresh = inner.refreshModels!;
  const provider: Provider<LiteLLMApi> = {
    ...inner,
    async refreshModels(context) {
      lastContext = context;
      await innerRefresh(context);
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
