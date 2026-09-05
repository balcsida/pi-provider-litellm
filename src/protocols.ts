import type { ProviderStreams } from "@earendil-works/pi-ai";
import { openAICompletionsApi, openAIResponsesApi } from "@earendil-works/pi-ai/compat";
import { normalizeBaseUrl } from "./discover.js";
import type { LiteLLMApi } from "./types.js";

type LiteLLMProtocol = {
  createApi: () => ProviderStreams;
  modelBaseUrl(root: string): string;
};

export const LITELLM_PROTOCOLS = {
  "openai-completions": {
    createApi: openAICompletionsApi,
    modelBaseUrl: (root) => `${root}/v1`,
  },
  "openai-responses": {
    createApi: openAIResponsesApi,
    modelBaseUrl: (root) => `${root}/v1`,
  },
} satisfies Record<LiteLLMApi, LiteLLMProtocol>;

export const LITELLM_API_NAMES = Object.keys(LITELLM_PROTOCOLS) as LiteLLMApi[];

export function isLiteLLMApi(api: unknown): api is LiteLLMApi {
  return typeof api === "string" && Object.hasOwn(LITELLM_PROTOCOLS, api);
}

export function resolveModelBaseUrl(baseUrl: string, api: LiteLLMApi, allowInsecureHttp = false): string {
  return LITELLM_PROTOCOLS[api].modelBaseUrl(normalizeBaseUrl(baseUrl, allowInsecureHttp));
}

export function createLiteLLMProtocolApis(): Record<LiteLLMApi, ProviderStreams> {
  return {
    "openai-completions": LITELLM_PROTOCOLS["openai-completions"].createApi(),
    "openai-responses": LITELLM_PROTOCOLS["openai-responses"].createApi(),
  };
}
