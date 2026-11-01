import type { ProviderStreams } from "@earendil-works/pi-ai";
import { anthropicMessagesApi, openAICompletionsApi, openAIResponsesApi } from "@earendil-works/pi-ai/compat";
import { normalizeBaseUrl } from "./discover.js";
import type { LiteLLMApi } from "./types.js";

type LiteLLMProtocol = {
  createApi: () => ProviderStreams;
  modelBaseUrl(root: string): string;
};

export const LITELLM_PROTOCOLS = {
  "anthropic-messages": {
    createApi: anthropicMessagesApi,
    modelBaseUrl: (root) => root,
  },
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

// Pi's Azure Responses API is the OpenAI Responses wire format behind Azure URLs and `api-key` auth. A model can
// carry it from Pi's Azure catalog, but the proxy serves Responses itself at `<root>/v1` with its own credential.
const PROXY_API_ALIASES: Readonly<Record<string, LiteLLMApi>> = { "azure-openai-responses": "openai-responses" };

/** The protocol the proxy speaks for a model's API, or undefined when the proxy does not serve it. */
export function toLiteLLMApi(api: unknown): LiteLLMApi | undefined {
  if (isLiteLLMApi(api)) return api;
  return typeof api === "string" && Object.hasOwn(PROXY_API_ALIASES, api) ? PROXY_API_ALIASES[api] : undefined;
}

export function resolveModelBaseUrl(baseUrl: string, api: LiteLLMApi, allowInsecureHttp = false): string {
  return LITELLM_PROTOCOLS[api].modelBaseUrl(normalizeBaseUrl(baseUrl, allowInsecureHttp));
}

export function createLiteLLMProtocolApis(): Record<LiteLLMApi, ProviderStreams> {
  return {
    "anthropic-messages": LITELLM_PROTOCOLS["anthropic-messages"].createApi(),
    "openai-completions": LITELLM_PROTOCOLS["openai-completions"].createApi(),
    "openai-responses": LITELLM_PROTOCOLS["openai-responses"].createApi(),
  };
}
