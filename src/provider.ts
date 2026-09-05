import { type Credential, createProvider, type Provider, type ProviderAuth } from "@earendil-works/pi-ai";
import { openAICompletionsApi, openAIResponsesApi } from "@earendil-works/pi-ai/compat";
import { LITELLM_DISCOVERY_VERSION } from "./backend-identity.js";
import { enrichCachedModel } from "./discover.js";
import type { DiscoveredModel, DiscoveryResult, LiteLLMApi, LiteLLMModel } from "./types.js";

export type LiteLLMProviderOptions = {
  id: string;
  name: string;
  baseUrl: string;
  auth: ProviderAuth;
  /** Catalog discovered during activation. Pi's startup refresh never allows network access, so
   * without a seed the provider stays empty until the user opens /model. */
  models?: readonly LiteLLMModel[];
  discover(credential: Credential, signal?: AbortSignal): Promise<DiscoveryResult & { baseUrl?: string }>;
};

export function toNativeModels(provider: string, baseUrl: string, models: readonly DiscoveredModel[]): LiteLLMModel[] {
  return models.map((model) => ({
    ...model,
    provider,
    api: model.api ?? "openai-completions",
    baseUrl,
  })) as LiteLLMModel[];
}

export function createLiteLLMProvider(options: LiteLLMProviderOptions): Provider<LiteLLMApi> {
  const provider = createProvider<LiteLLMApi>({
    id: options.id,
    name: options.name,
    baseUrl: options.baseUrl,
    auth: options.auth,
    models: options.models ?? [],
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
  let warnedLegacy = false;
  return {
    ...provider,
    refreshModels: async (context) => {
      const isLegacyReasoningModel = (model: LiteLLMModel) =>
        model.litellmDiscoveryVersion !== LITELLM_DISCOVERY_VERSION &&
        (model.reasoning || model.thinkingLevelMap !== undefined);
      const storedModels = context.stored?.models as readonly LiteLLMModel[] | undefined;
      const hasLegacyReasoningModel = storedModels?.some(isLegacyReasoningModel) ?? false;
      const stored = context.stored && {
        ...context.stored,
        models: context.stored.models.map((model) =>
          isLegacyReasoningModel(model as LiteLLMModel) ? model : enrichCachedModel(model),
        ),
      };
      let replacedLegacyModels = false;
      try {
        await refreshModels({
          ...context,
          stored,
          publish: async (publication) => {
            const published = await context.publish(publication);
            if (published && publication.persist) replacedLegacyModels = true;
            return published;
          },
        });
      } finally {
        if (context.allowNetwork && hasLegacyReasoningModel && !replacedLegacyModels && !warnedLegacy) {
          warnedLegacy = true;
          process.stderr.write(
            "LiteLLM discovery: cached models predate discovery policy v2; network refresh required to update reasoning compatibility\n",
          );
        }
      }
    },
  };
}
