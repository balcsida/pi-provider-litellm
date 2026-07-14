import type { Api, KnownProvider, Model } from "@earendil-works/pi-ai";
import { getModels, getProviders } from "@earendil-works/pi-ai/compat";
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

const RESPONSES_MODE_PATTERN = /^responses?$/i;

function isResponsesMode(mode: string | undefined): boolean {
  return mode !== undefined && RESPONSES_MODE_PATTERN.test(mode);
}

function isChatStyleMode(mode: string | undefined): boolean {
  return mode === undefined || mode === "chat" || isResponsesMode(mode);
}

// Matches both the conventional `anthropic/...` prefix and aliases that
// LiteLLM deployments commonly assign to Anthropic-backed routes (e.g.
// `google/claude-sonnet-4-6`, `opus-4.7`, `sonnet-4.6`, `haiku-4.5`). Without
// the `cacheControlFormat: "anthropic"` flag, pi never relays cache_control
// markers through the proxy, so prompt caching silently no-ops on Claude models.
const ANTHROPIC_MODEL_PATTERN = /(?:^|[-_/.:])(?:anthropic\/|(?:claude|opus|sonnet|haiku)(?=$|[-_/.:]))/i;
const MOONSHOT_MODEL_PATTERN = /^(moonshotai\/|moonshot\/|kimi[-/])/i;
const FORCED_THINKING_MODEL_PATTERN = /(?:^|[-/])thinking(?:[-/]|$)/i;
// Deployments expose the gpt-5.5 route under varying names (`llm-gateway/gpt-5.5`,
// bare `gpt-5.5`, dated ids like `gpt-5.5-20260504143601`); match them all so the
// tool+reasoning workaround survives route renames.
const GPT55_MODEL_PATTERN = /(?:^|\/)gpt-5\.5(?:$|[-.])/i;

export function isMoonshotModel(modelId: string): boolean {
  return MOONSHOT_MODEL_PATTERN.test(modelId);
}

export function isGpt55Model(modelId: string): boolean {
  return GPT55_MODEL_PATTERN.test(modelId);
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
  const lookupIds = catalogLookupIds(id);
  const candidates = [toKnownProvider(ownedBy), prefixProvider, lookupIds.length > 1 ? "anthropic" : undefined].filter(
    (provider): provider is KnownProvider => provider !== undefined,
  );

  for (const provider of candidates) {
    const match = findCatalogModelInProvider(provider, lookupIds);
    if (match) return match;
  }

  for (const provider of getProviders()) {
    const match = findCatalogModelInProvider(provider, lookupIds);
    if (match) return match;
  }

  return undefined;
}

function catalogLookupIds(id: string): string[] {
  const lookupIds = new Set([id]);
  const unprefixed = id.includes("/") ? id.slice(id.indexOf("/") + 1) : id;
  lookupIds.add(unprefixed);

  const anthropicAlias = unprefixed.toLowerCase().replaceAll(".", "-");
  const match = /^(?:claude-)?(opus|sonnet|haiku)-(\d+)-(\d+)$/.exec(anthropicAlias);
  if (match) lookupIds.add(`claude-${match[1]}-${match[2]}-${match[3]}`);

  return [...lookupIds];
}

function findCatalogModelInProvider(provider: KnownProvider, lookupIds: string[]): Model<Api> | undefined {
  for (const lookupId of lookupIds) {
    const exact = getModels(provider).find((model) => model.id === lookupId);
    if (exact) return exact;
    const providerQualified = getModels(provider).find((model) => model.id === `${provider}/${lookupId}`);
    if (providerQualified) return providerQualified;
  }
  return undefined;
}

function mapModelInfoCost(
  info: NonNullable<ModelInfoEntry["model_info"]>,
  fallback?: ProviderModelConfig["cost"],
): NonNullable<ProviderModelConfig["cost"]> {
  return {
    input: info.input_cost_per_token !== undefined ? info.input_cost_per_token * 1_000_000 : (fallback?.input ?? 0),
    output: info.output_cost_per_token !== undefined ? info.output_cost_per_token * 1_000_000 : (fallback?.output ?? 0),
    cacheRead:
      info.cache_read_input_token_cost !== undefined
        ? info.cache_read_input_token_cost * 1_000_000
        : (fallback?.cacheRead ?? 0),
    cacheWrite:
      info.cache_creation_input_token_cost !== undefined
        ? info.cache_creation_input_token_cost * 1_000_000
        : (fallback?.cacheWrite ?? 0),
  };
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

export function withTimeout(timeoutMs: number, signal?: AbortSignal): { signal: AbortSignal; cancel: () => void } {
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
      headers: { ...options.headers, Authorization: `Bearer ${apiKey}`, Accept: "application/json" },
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
  if (!isChatStyleMode(info.mode)) return undefined;
  const responsesMode = isResponsesMode(info.mode);
  const catalogModel = findCatalogModel(id);
  return {
    id,
    name: id,
    reasoning: info.supports_reasoning ?? false,
    input: info.supports_vision ? ["text", "image"] : ["text"],
    cost: mapModelInfoCost(info, catalogModel?.cost),
    contextWindow: info.max_input_tokens ?? DEFAULT_CONTEXT_WINDOW,
    maxTokens: info.max_output_tokens ?? DEFAULT_MAX_TOKENS,
    compat: buildCompat(id),
    ...(responsesMode ? { api: "openai-responses" as const } : {}),
  };
}

function mapFromHealthModelInfo(
  entry: ModelInfoEntry,
  fallbackId: string | undefined,
): ProviderModelConfig | undefined {
  const model = mapFromModelInfo(entry);
  if (model || !fallbackId) return model;
  if (!isChatStyleMode(entry.model_info?.mode)) return undefined;
  const info = entry.model_info ?? {};
  const responsesMode = isResponsesMode(info.mode);
  const catalogModel = findCatalogModel(fallbackId);
  return {
    id: fallbackId,
    name: fallbackId,
    reasoning: info.supports_reasoning ?? false,
    input: info.supports_vision ? ["text", "image"] : ["text"],
    cost: mapModelInfoCost(info, catalogModel?.cost),
    contextWindow: info.max_input_tokens ?? DEFAULT_CONTEXT_WINDOW,
    maxTokens: info.max_output_tokens ?? DEFAULT_MAX_TOKENS,
    compat: buildCompat(fallbackId),
    ...(responsesMode ? { api: "openai-responses" as const } : {}),
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
  options: DiscoveryOptions & { onProgress?: (message: string) => void },
): Promise<ProviderModelConfig[]> {
  options.onProgress?.("Querying /health endpoint...");
  const healthResult = await fetchJson<HealthResponse>(`${base}/health`, apiKey, options);
  if (!healthResult.ok) return [];
  const endpoints = (healthResult.data.healthy_endpoints ?? []).filter((entry) => entry.model || entry.model_id);
  options.onProgress?.(`Discovered ${endpoints.length} model endpoints, fetching details...`);
  let completed = 0;
  const models = await Promise.all(
    endpoints.map(async (endpoint) => {
      let model = mapFromHealthEndpoint(endpoint);
      if (endpoint.model_id) {
        const infoResult = await fetchJson<ModelInfoResponse>(
          `${base}/model/info?litellm_model_id=${encodeURIComponent(endpoint.model_id)}`,
          apiKey,
          options,
        );
        const entry = infoResult.ok ? infoResult.data.data?.[0] : undefined;
        if (entry) model = mapFromHealthModelInfo(entry, endpoint.model);
      }
      completed++;
      if (completed % 10 === 0) {
        options.onProgress?.(`Fetched ${completed}/${endpoints.length} models...`);
      }
      return model;
    }),
  );
  return models.filter((model): model is ProviderModelConfig => model !== undefined);
}

export async function discoverModels(
  baseUrl: string,
  apiKey: string,
  options: DiscoveryOptions & { onProgress?: (message: string) => void } = {},
): Promise<DiscoveryResult> {
  const base = normalizeBaseUrl(baseUrl);
  options.onProgress?.("Querying /model/info endpoint...");
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
  options.onProgress?.("/model/info unavailable, trying /v1/models...");
  const listResult = await fetchJson<ModelsListResponse>(`${base}/v1/models`, apiKey, options);
  if (!listResult.ok) {
    if ([401, 403, 404].includes(listResult.status)) {
      options.onProgress?.("/v1/models unavailable, falling back to /health endpoint...");
      const models = await discoverFromHealth(base, apiKey, options);
      if (models.length > 0) return { source: "health", models };
    }
    throw new Error(`/v1/models returned ${listResult.status}`);
  }
  options.onProgress?.("Fetching models.dev catalog for metadata enrichment...");
  const modelsDev = await getModelsDevCatalog(options);
  const models = (listResult.data.data ?? [])
    .map((entry) => mapFromModelsList(entry, modelsDev))
    .filter((m): m is ProviderModelConfig => m !== undefined);
  return { source: "models_list", models };
}
