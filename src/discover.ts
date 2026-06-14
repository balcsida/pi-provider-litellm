import type { Api, KnownProvider, Model } from "@earendil-works/pi-ai";
import { getModels, getProviders } from "@earendil-works/pi-ai";
import type { ProviderModelConfig } from "@earendil-works/pi-coding-agent";
import type {
  DiscoveryOptions,
  DiscoveryResult,
  HealthResponse,
  ModelInfoEntry,
  ModelInfoResponse,
  ModelsListEntry,
  ModelsListResponse,
} from "./types.js";

const DEFAULT_TIMEOUT_MS = 5000;
const DEFAULT_CONTEXT_WINDOW = 128_000;
const DEFAULT_MAX_TOKENS = 16_384;
// LiteLLM exposes Responses-API models (e.g. ChatGPT-subscription routes) with
// mode "responses". They are still callable via /chat/completions through the
// proxy, so treat them as selectable alongside "chat". Other modes (embedding,
// image_generation, ...) remain filtered out.
const SELECTABLE_MODES = new Set(["chat", "responses"]);
function isSelectableMode(mode: string | undefined): boolean {
  return !mode || SELECTABLE_MODES.has(mode);
}
const KNOWN_PROVIDER_SET = new Set<string>(getProviders());
const MODELS_DEV_URL = "https://models.dev/api.json";
let modelsDevCatalog: ModelsDevResponse | undefined;

interface ModelsDevModel {
  name?: string;
  reasoning?: boolean;
  modalities?: {
    input?: string[];
  };
  limit?: {
    context?: number;
    input?: number;
    output?: number;
  };
  cost?: {
    input?: number;
    output?: number;
    cache_read?: number;
    cache_write?: number;
  };
}

type ModelsDevResponse = Record<string, { models?: Record<string, ModelsDevModel> }>;

export function normalizeBaseUrl(input: string): string {
  return input.replace(/\/+$/, "").replace(/\/v1\/?$/i, "");
}

// Matches both the conventional `anthropic/...` prefix and aliases that
// LiteLLM deployments commonly assign to Anthropic-backed routes (e.g.
// `google/claude-sonnet-4-6`, `opus-4.7`, `sonnet-4.6`, `haiku-4.5`). Without
// the `cacheControlFormat: "anthropic"` flag, pi never relays cache_control
// markers through the proxy, so prompt caching silently no-ops on Claude models.
const ANTHROPIC_MODEL_PATTERN = /(?:^|[-_/.:])(?:anthropic\/|(?:claude|opus|sonnet|haiku)(?=$|[-_/.:]))/i;
const MOONSHOT_MODEL_PATTERN = /^(moonshotai\/|moonshot\/|kimi[-/])/i;
const FORCED_THINKING_MODEL_PATTERN = /(?:^|[-/])thinking(?:[-/]|$)/i;

export function isMoonshotModel(modelId: string): boolean {
  return MOONSHOT_MODEL_PATTERN.test(modelId);
}

export function shouldSuppressReasoningContent(modelId: string): boolean {
  return isMoonshotModel(modelId) && !FORCED_THINKING_MODEL_PATTERN.test(modelId);
}

export function buildCompat(modelId: string): ProviderModelConfig["compat"] {
  if (isMoonshotModel(modelId)) {
    return {
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsStrictMode: false,
      maxTokensField: "max_tokens",
    };
  }
  if (ANTHROPIC_MODEL_PATTERN.test(modelId)) {
    return { supportsStore: false, cacheControlFormat: "anthropic" };
  }
  return { supportsStore: false };
}

function toKnownProvider(provider: string | undefined): KnownProvider | undefined {
  if (!provider) return undefined;
  const normalized = provider.toLowerCase();
  return KNOWN_PROVIDER_SET.has(normalized) ? (normalized as KnownProvider) : undefined;
}

function findCatalogModel(id: string, ownedBy?: string): Model<Api> | undefined {
  const prefixProvider = toKnownProvider(id.split("/")[0]);
  const candidates = [toKnownProvider(ownedBy), prefixProvider].filter(
    (provider): provider is KnownProvider => provider !== undefined,
  );

  for (const provider of candidates) {
    const exact = getModels(provider).find((model) => model.id === id);
    if (exact) return exact;
    const providerQualified = getModels(provider).find((model) => model.id === `${provider}/${id}`);
    if (providerQualified) return providerQualified;
  }

  for (const provider of getProviders()) {
    const exact = getModels(provider).find((model) => model.id === id);
    if (exact) return exact;
  }

  return undefined;
}

function getFallbackProviderAndModel(id: string, ownedBy?: string): { provider?: string; modelId: string } {
  const [prefix, ...rest] = id.split("/");
  const prefixProvider = toKnownProvider(prefix);
  if (prefixProvider && rest.length > 0) {
    return { provider: prefixProvider, modelId: rest.join("/") };
  }
  return { provider: toKnownProvider(ownedBy), modelId: id };
}

function findModelsDevModel(
  catalog: ModelsDevResponse | undefined,
  id: string,
  ownedBy?: string,
): ModelsDevModel | undefined {
  const { provider, modelId } = getFallbackProviderAndModel(id, ownedBy);
  if (!provider) return undefined;
  return catalog?.[provider]?.models?.[modelId];
}

function withTimeout(timeoutMs: number, signal?: AbortSignal): { signal: AbortSignal; cancel: () => void } {
  const controller = new AbortController();
  const onAbort = () => controller.abort(signal?.reason);
  if (signal) {
    if (signal.aborted) controller.abort(signal.reason);
    else signal.addEventListener("abort", onAbort, { once: true });
  }
  const timer = setTimeout(() => controller.abort(new Error(`Timed out after ${timeoutMs}ms`)), timeoutMs);
  return {
    signal: controller.signal,
    cancel: () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    },
  };
}

async function fetchJson<T>(
  url: string,
  apiKey: string,
  options: DiscoveryOptions,
): Promise<{ ok: true; data: T } | { ok: false; status: number }> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const { signal, cancel } = withTimeout(timeoutMs, options.signal);
  try {
    const response = await fetch(url, {
      headers: { Authorization: `Bearer ${apiKey}`, Accept: "application/json" },
      signal,
    });
    if (!response.ok) return { ok: false, status: response.status };
    const data = (await response.json()) as T;
    return { ok: true, data };
  } finally {
    cancel();
  }
}

async function fetchPublicJson<T>(url: string, options: DiscoveryOptions): Promise<T> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const { signal, cancel } = withTimeout(timeoutMs, options.signal);
  try {
    const response = await fetch(url, {
      headers: { Accept: "application/json" },
      signal,
    });
    if (!response.ok) throw new Error(`${url} returned ${response.status}`);
    return (await response.json()) as T;
  } finally {
    cancel();
  }
}

async function getModelsDevCatalog(options: DiscoveryOptions): Promise<ModelsDevResponse | undefined> {
  if (modelsDevCatalog) return modelsDevCatalog;
  try {
    modelsDevCatalog = await fetchPublicJson<ModelsDevResponse>(MODELS_DEV_URL, options);
    return modelsDevCatalog;
  } catch {
    return undefined;
  }
}

function mapModelsDevMetadata(model: ModelsDevModel | undefined): Partial<ProviderModelConfig> {
  if (!model) return {};
  const metadata: Partial<ProviderModelConfig> = {};
  if (model.name) metadata.name = model.name;
  if (model.reasoning !== undefined) metadata.reasoning = model.reasoning;
  if (model.modalities?.input) {
    metadata.input = model.modalities.input.includes("image") ? ["text", "image"] : ["text"];
  }
  const contextWindow = model.limit?.context ?? model.limit?.input;
  if (contextWindow !== undefined) metadata.contextWindow = contextWindow;
  if (model.limit?.output !== undefined) metadata.maxTokens = model.limit.output;
  if (model.cost) {
    metadata.cost = {
      input: model.cost.input ?? 0,
      output: model.cost.output ?? 0,
      cacheRead: model.cost.cache_read ?? 0,
      cacheWrite: model.cost.cache_write ?? 0,
    };
  }
  return metadata;
}

function mapFromModelInfo(entry: ModelInfoEntry): ProviderModelConfig | undefined {
  const id = entry.model_name;
  if (!id) return undefined;
  const info = entry.model_info ?? {};
  if (!isSelectableMode(info.mode)) return undefined;
  // Borrow the thinking-level map from the catalog when the model id is known
  // (e.g. deepseek-v4-pro), so per-model levels like "xhigh" stay available.
  // The proxy's /model/info does not carry this mapping.
  const catalogModel = findCatalogModel(id, info.litellm_provider);
  return {
    id,
    name: id,
    reasoning: info.supports_reasoning ?? false,
    thinkingLevelMap: catalogModel?.thinkingLevelMap,
    input: info.supports_vision ? ["text", "image"] : ["text"],
    cost: {
      input: (info.input_cost_per_token ?? 0) * 1_000_000,
      output: (info.output_cost_per_token ?? 0) * 1_000_000,
      cacheRead: (info.cache_read_input_token_cost ?? 0) * 1_000_000,
      cacheWrite: (info.cache_creation_input_token_cost ?? 0) * 1_000_000,
    },
    contextWindow: info.max_input_tokens ?? DEFAULT_CONTEXT_WINDOW,
    maxTokens: info.max_output_tokens ?? DEFAULT_MAX_TOKENS,
    compat: buildCompat(id),
  };
}

function mapFromHealthModelInfo(
  entry: ModelInfoEntry,
  fallbackId: string | undefined,
): ProviderModelConfig | undefined {
  const model = mapFromModelInfo(entry);
  if (model || !fallbackId) return model;
  if (!isSelectableMode(entry.model_info?.mode)) return undefined;
  return {
    id: fallbackId,
    name: fallbackId,
    reasoning: entry.model_info?.supports_reasoning ?? false,
    input: entry.model_info?.supports_vision ? ["text", "image"] : ["text"],
    cost: {
      input: (entry.model_info?.input_cost_per_token ?? 0) * 1_000_000,
      output: (entry.model_info?.output_cost_per_token ?? 0) * 1_000_000,
      cacheRead: (entry.model_info?.cache_read_input_token_cost ?? 0) * 1_000_000,
      cacheWrite: (entry.model_info?.cache_creation_input_token_cost ?? 0) * 1_000_000,
    },
    contextWindow: entry.model_info?.max_input_tokens ?? DEFAULT_CONTEXT_WINDOW,
    maxTokens: entry.model_info?.max_output_tokens ?? DEFAULT_MAX_TOKENS,
    compat: buildCompat(fallbackId),
  };
}

function mapFromHealthEndpoint(entry: { model?: string }): ProviderModelConfig | undefined {
  const id = entry.model;
  if (!id) return undefined;
  const catalogModel = findCatalogModel(id);
  return {
    id,
    name: catalogModel?.name ?? id,
    reasoning: catalogModel?.reasoning ?? false,
    thinkingLevelMap: catalogModel?.thinkingLevelMap,
    input: catalogModel?.input ?? ["text"],
    cost: catalogModel?.cost ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: catalogModel?.contextWindow ?? DEFAULT_CONTEXT_WINDOW,
    maxTokens: catalogModel?.maxTokens ?? DEFAULT_MAX_TOKENS,
    compat: buildCompat(id),
  };
}

function mapFromModelsList(
  entry: ModelsListEntry,
  modelsDev: ModelsDevResponse | undefined,
): ProviderModelConfig | undefined {
  const id = entry.id;
  if (!id) return undefined;
  const catalogModel = findCatalogModel(id, entry.owned_by);
  const modelsDevMetadata = mapModelsDevMetadata(findModelsDevModel(modelsDev, id, entry.owned_by));
  return {
    id,
    name: modelsDevMetadata.name ?? catalogModel?.name ?? `${id} (no metadata)`,
    reasoning: modelsDevMetadata.reasoning ?? catalogModel?.reasoning ?? false,
    thinkingLevelMap: catalogModel?.thinkingLevelMap,
    input: modelsDevMetadata.input ?? catalogModel?.input ?? ["text"],
    cost: modelsDevMetadata.cost ?? catalogModel?.cost ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: modelsDevMetadata.contextWindow ?? catalogModel?.contextWindow ?? DEFAULT_CONTEXT_WINDOW,
    maxTokens: modelsDevMetadata.maxTokens ?? catalogModel?.maxTokens ?? DEFAULT_MAX_TOKENS,
    compat: buildCompat(id),
  };
}

async function discoverFromHealth(
  base: string,
  apiKey: string,
  options: DiscoveryOptions,
): Promise<ProviderModelConfig[]> {
  const healthResult = await fetchJson<HealthResponse>(`${base}/health`, apiKey, options);
  if (!healthResult.ok) return [];
  const endpoints = (healthResult.data.healthy_endpoints ?? []).filter((entry) => entry.model || entry.model_id);
  const models = await Promise.all(
    endpoints.map(async (endpoint) => {
      if (!endpoint.model_id) return mapFromHealthEndpoint(endpoint);
      const infoResult = await fetchJson<ModelInfoResponse>(
        `${base}/model/info?litellm_model_id=${encodeURIComponent(endpoint.model_id)}`,
        apiKey,
        options,
      );
      if (!infoResult.ok) return mapFromHealthEndpoint(endpoint);
      const entry = infoResult.data.data?.[0];
      return entry ? mapFromHealthModelInfo(entry, endpoint.model) : mapFromHealthEndpoint(endpoint);
    }),
  );
  return models.filter((model): model is ProviderModelConfig => model !== undefined);
}

export async function discoverModels(
  baseUrl: string,
  apiKey: string,
  options: DiscoveryOptions = {},
): Promise<DiscoveryResult> {
  const base = normalizeBaseUrl(baseUrl);
  const infoResult = await fetchJson<ModelInfoResponse>(`${base}/model/info`, apiKey, options);
  if (infoResult.ok) {
    const models = (infoResult.data.data ?? [])
      .map(mapFromModelInfo)
      .filter((m): m is ProviderModelConfig => m !== undefined);
    return { source: "model_info", models };
  }
  if (![401, 403, 404].includes(infoResult.status)) {
    throw new Error(`/model/info returned ${infoResult.status}`);
  }
  const listResult = await fetchJson<ModelsListResponse>(`${base}/v1/models`, apiKey, options);
  if (!listResult.ok) {
    if ([401, 403, 404].includes(listResult.status)) {
      const models = await discoverFromHealth(base, apiKey, options);
      if (models.length > 0) return { source: "health", models };
    }
    throw new Error(`/v1/models returned ${listResult.status}`);
  }
  const modelsDev = await getModelsDevCatalog(options);
  const models = (listResult.data.data ?? [])
    .map((entry) => mapFromModelsList(entry, modelsDev))
    .filter((m): m is ProviderModelConfig => m !== undefined);
  return { source: "models_list", models };
}
