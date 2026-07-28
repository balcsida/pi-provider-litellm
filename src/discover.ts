import { readFile } from "node:fs/promises";
import type { Api, Model, ThinkingLevelMap } from "@earendil-works/pi-ai";
import { getModels, getProviders } from "@earendil-works/pi-ai/compat";
import type { BuiltinProvider } from "@earendil-works/pi-ai/providers/all";
import { writeJsonAtomic } from "./cache.js";
import type {
  DiscoveredModel,
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
const MODELS_DEV_CACHE_TTL_MS = 28 * 24 * 60 * 60 * 1000;

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

interface ModelsDevCacheFile {
  fetchedAt: number;
  catalog: ModelsDevResponse;
}

const modelsDevCaches = new Map<string, ModelsDevCacheFile>();
const modelsDevRefreshes = new Map<string, Promise<ModelsDevResponse | undefined>>();

export function normalizeBaseUrl(input: string): string {
  return input.replace(/\/+$/, "").replace(/\/v1\/?$/i, "");
}

const RESPONSES_MODE_PATTERN = /^responses?$/i;

function isResponsesMode(mode: string | null | undefined): boolean {
  return mode != null && RESPONSES_MODE_PATTERN.test(mode);
}

function isChatStyleMode(mode: string | null | undefined): boolean {
  return mode == null || mode === "chat" || isResponsesMode(mode);
}

// Adapters that only front Anthropic models, so the adapter value alone is
// authoritative evidence of Messages support.
const ANTHROPIC_MESSAGES_ADAPTERS = new Set(["anthropic", "vertex_ai-anthropic_models"]);
// Adapters that front mixed catalogs (Bedrock Converse also serves Nova, Llama,
// and Mistral), so they need corroborating backend-model evidence.
const ANTHROPIC_BASE_MODEL_ADAPTERS = new Set(["azure_ai", "bedrock", "bedrock_converse"]);
// Routes addressed by their provider-qualified id carry the backend model in the
// route name and report no separate `base_model`, so the name is authoritative
// for them. A vanity alias is neither, and stays on Chat Completions.
const PROVIDER_QUALIFIED_ROUTE_PATTERN = /^(?:bedrock|azure_ai)\//i;
const ANTHROPIC_BASE_MODEL_PATTERN = /(?:^|[/_.:-])(?:anthropic|claude|opus|sonnet|haiku)(?:[/_.:-]|$)/i;

function selectApi(
  mode: string | null | undefined,
  litellmProvider?: string,
  baseModel?: string,
  modelId?: string,
): DiscoveredModel["api"] {
  if (isResponsesMode(mode)) return "openai-responses";
  const adapter = litellmProvider?.trim().toLowerCase();
  if (!adapter) return "openai-completions";
  if (ANTHROPIC_MESSAGES_ADAPTERS.has(adapter)) return "anthropic-messages";
  if (!ANTHROPIC_BASE_MODEL_ADAPTERS.has(adapter)) return "openai-completions";
  const backendModel = baseModel ?? (modelId && PROVIDER_QUALIFIED_ROUTE_PATTERN.test(modelId) ? modelId : undefined);
  return backendModel && ANTHROPIC_BASE_MODEL_PATTERN.test(backendModel) ? "anthropic-messages" : "openai-completions";
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
  if (FORCED_THINKING_MODEL_PATTERN.test(modelId)) return false;
  if (isMoonshotModel(modelId)) return true;

  const separator = modelId.indexOf("/");
  const unprefixed = separator === -1 ? modelId : modelId.slice(separator + 1);
  if (unprefixed === modelId || !isMoonshotModel(unprefixed)) return false;

  const catalogModel = findCatalogModel(modelId);
  return (
    catalogModel != null && isMoonshotModel(catalogModel.id) && !FORCED_THINKING_MODEL_PATTERN.test(catalogModel.id)
  );
}

function isMoonshotRoute(modelId: string, catalogModel?: Model<Api>): boolean {
  return isMoonshotModel(modelId) || (catalogModel != null && isMoonshotModel(catalogModel.id));
}

export function buildCompat(
  modelId: string,
  catalogModel?: Model<Api>,
  api: DiscoveredModel["api"] = "openai-completions",
): DiscoveredModel["compat"] {
  if (api === "anthropic-messages") {
    return catalogModel?.api === "anthropic-messages" ? catalogModel.compat : undefined;
  }
  if (api === "openai-responses") return undefined;
  if (isMoonshotRoute(modelId, catalogModel)) {
    return {
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsStrictMode: false,
      maxTokensField: "max_tokens",
      thinkingFormat: "deepseek",
    };
  }
  if (ANTHROPIC_MODEL_PATTERN.test(modelId)) {
    return { supportsStore: false, cacheControlFormat: "anthropic" };
  }
  return { supportsStore: false };
}

function toKnownProvider(provider: string | undefined): BuiltinProvider | undefined {
  if (!provider) return undefined;
  const normalized = provider.toLowerCase();
  return KNOWN_PROVIDER_SET.has(normalized) ? (normalized as BuiltinProvider) : undefined;
}

function findCatalogModel(id: string, ownedBy?: string): Model<Api> | undefined {
  const prefixProvider = toKnownProvider(id.split("/")[0]);
  const lookupIds = catalogLookupIds(id);
  const candidates = [toKnownProvider(ownedBy), prefixProvider, lookupIds.length > 1 ? "anthropic" : undefined].filter(
    (provider): provider is BuiltinProvider => provider !== undefined,
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

function findCatalogModelInProvider(provider: BuiltinProvider, lookupIds: string[]): Model<Api> | undefined {
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
  fallback?: DiscoveredModel["cost"],
): NonNullable<DiscoveredModel["cost"]> {
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
    ...(fallback?.tiers ? { tiers: fallback.tiers } : {}),
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
  const models = catalog?.[provider]?.models;
  return models && Object.hasOwn(models, modelId) ? models[modelId] : undefined;
}

const BOOLEAN_THINKING_LEVEL_MAP: ThinkingLevelMap = {
  minimal: null,
  low: null,
  medium: null,
  xhigh: null,
  max: null,
};

const ALWAYS_THINKING_LEVEL_MAP: ThinkingLevelMap = {
  off: null,
  ...BOOLEAN_THINKING_LEVEL_MAP,
};

type ReasoningCompat = Pick<
  NonNullable<Model<"openai-completions">["compat"]>,
  "thinkingFormat" | "supportsReasoningEffort"
>;

type ReasoningSignal = boolean | undefined;

function getReasoningCompat(catalogModel: Model<Api> | undefined): ReasoningCompat | undefined {
  if (catalogModel?.api !== "openai-completions") return undefined;
  const compat = catalogModel.compat as ReasoningCompat | undefined;
  const reasoningCompat: ReasoningCompat = {};
  if (compat?.thinkingFormat !== undefined) reasoningCompat.thinkingFormat = compat.thinkingFormat;
  if (compat?.supportsReasoningEffort !== undefined) {
    reasoningCompat.supportsReasoningEffort = compat.supportsReasoningEffort;
  }
  return Object.keys(reasoningCompat).length > 0 ? reasoningCompat : undefined;
}

function routeSupportsReasoning(routeSignal: ReasoningSignal, catalogModel: Model<Api> | undefined): boolean {
  if (routeSignal === false) return false;
  return catalogModel?.reasoning === true;
}

function buildThinkingLevelMap(modelId: string, catalogModel: Model<Api> | undefined): ThinkingLevelMap | undefined {
  if (!catalogModel?.reasoning) return undefined;
  if (catalogModel.thinkingLevelMap) {
    return isMoonshotRoute(modelId, catalogModel) && getReasoningCompat(catalogModel)?.thinkingFormat === "deepseek"
      ? { ...ALWAYS_THINKING_LEVEL_MAP, ...catalogModel.thinkingLevelMap }
      : catalogModel.thinkingLevelMap;
  }
  return isMoonshotRoute(modelId, catalogModel) ? BOOLEAN_THINKING_LEVEL_MAP : undefined;
}

function applyReasoningPolicy(
  model: Pick<DiscoveredModel, "api" | "compat" | "id"> &
    Partial<Pick<DiscoveredModel, "reasoning" | "thinkingLevelMap">>,
  catalogModel: Model<Api> | undefined,
  routeSignal: ReasoningSignal = catalogModel?.reasoning,
  effortCapabilities?: { xhigh?: boolean; max?: boolean },
): void {
  if (model.api === "anthropic-messages") {
    model.reasoning = routeSignal === false ? false : routeSignal === true || catalogModel?.reasoning === true;
    if (!model.reasoning) {
      delete model.thinkingLevelMap;
      return;
    }
    const thinkingLevelMap: ThinkingLevelMap = {};
    if (effortCapabilities?.xhigh === true) thinkingLevelMap.xhigh = "xhigh";
    if (effortCapabilities?.max === true) thinkingLevelMap.max = "max";
    model.thinkingLevelMap = thinkingLevelMap;
    if (effortCapabilities?.xhigh === true || effortCapabilities?.max === true) {
      model.compat = { ...(model.compat ?? {}), forceAdaptiveThinking: true };
    }
    return;
  }

  model.reasoning = routeSupportsReasoning(routeSignal, catalogModel);
  if (!model.reasoning) {
    delete model.thinkingLevelMap;
    return;
  }
  const thinkingLevelMap = buildThinkingLevelMap(model.id, catalogModel);
  if (thinkingLevelMap) model.thinkingLevelMap = thinkingLevelMap;
  else delete model.thinkingLevelMap;

  const reasoningCompat = model.api === "openai-completions" ? getReasoningCompat(catalogModel) : undefined;
  if (reasoningCompat) model.compat = { ...(model.compat ?? {}), ...reasoningCompat };
}

function awaitWithSignal<T>(promise: Promise<T>, signal?: AbortSignal): Promise<T> {
  if (!signal) return promise;
  if (signal.aborted) return Promise.reject(signal.reason);
  return new Promise<T>((resolve, reject) => {
    const cleanup = () => signal.removeEventListener("abort", onAbort);
    const onAbort = () => {
      cleanup();
      reject(signal.reason);
    };
    signal.addEventListener("abort", onAbort, { once: true });
    promise.then(
      (value) => {
        cleanup();
        resolve(value);
      },
      (error) => {
        cleanup();
        reject(error);
      },
    );
  });
}
async function fetchJson<T>(
  url: string,
  apiKey: string,
  options: DiscoveryOptions,
): Promise<{ ok: true; data: T } | { ok: false; status: number }> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const response = await fetch(url, {
    headers: { ...options.headers, Authorization: `Bearer ${apiKey}`, Accept: "application/json" },
    signal: options.signal
      ? AbortSignal.any([options.signal, AbortSignal.timeout(timeoutMs)])
      : AbortSignal.timeout(timeoutMs),
  });
  if (!response.ok) return { ok: false, status: response.status };
  const data = (await response.json()) as T;
  return { ok: true, data };
}

async function fetchPublicJson<T>(url: string, options: DiscoveryOptions): Promise<T> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const response = await fetch(url, {
    headers: { Accept: "application/json" },
    signal: options.signal
      ? AbortSignal.any([options.signal, AbortSignal.timeout(timeoutMs)])
      : AbortSignal.timeout(timeoutMs),
  });
  if (!response.ok) throw new Error(`${url} returned ${response.status}`);
  return (await response.json()) as T;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function finiteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function normalizeModelsDevCatalog(value: unknown): ModelsDevResponse | undefined {
  if (!isRecord(value)) return undefined;
  const catalog: ModelsDevResponse = {};
  for (const [providerId, providerValue] of Object.entries(value)) {
    if (!isRecord(providerValue) || !isRecord(providerValue.models)) continue;
    const models: Record<string, ModelsDevModel> = {};
    for (const [modelId, modelValue] of Object.entries(providerValue.models)) {
      if (!isRecord(modelValue)) continue;
      const model: ModelsDevModel = {};
      if (typeof modelValue.name === "string") model.name = modelValue.name;
      if (typeof modelValue.reasoning === "boolean") model.reasoning = modelValue.reasoning;
      if (isRecord(modelValue.modalities) && Array.isArray(modelValue.modalities.input)) {
        const input = modelValue.modalities.input.filter((entry): entry is string => typeof entry === "string");
        if (input.length > 0) model.modalities = { input };
      }
      if (isRecord(modelValue.limit)) {
        const limit: NonNullable<ModelsDevModel["limit"]> = {};
        const context = finiteNumber(modelValue.limit.context);
        const input = finiteNumber(modelValue.limit.input);
        const output = finiteNumber(modelValue.limit.output);
        if (context !== undefined) limit.context = context;
        if (input !== undefined) limit.input = input;
        if (output !== undefined) limit.output = output;
        if (Object.keys(limit).length > 0) model.limit = limit;
      }
      if (isRecord(modelValue.cost)) {
        const cost: NonNullable<ModelsDevModel["cost"]> = {};
        const input = finiteNumber(modelValue.cost.input);
        const output = finiteNumber(modelValue.cost.output);
        const cacheRead = finiteNumber(modelValue.cost.cache_read);
        const cacheWrite = finiteNumber(modelValue.cost.cache_write);
        if (input !== undefined) cost.input = input;
        if (output !== undefined) cost.output = output;
        if (cacheRead !== undefined) cost.cache_read = cacheRead;
        if (cacheWrite !== undefined) cost.cache_write = cacheWrite;
        if (Object.keys(cost).length > 0) model.cost = cost;
      }
      models[modelId] = model;
    }
    catalog[providerId] = { models };
  }
  return catalog;
}

async function readModelsDevCache(path: string): Promise<ModelsDevCacheFile | undefined> {
  try {
    const parsed: unknown = JSON.parse(await readFile(path, "utf8"));
    if (
      !isRecord(parsed) ||
      typeof parsed.fetchedAt !== "number" ||
      !Number.isFinite(parsed.fetchedAt) ||
      parsed.fetchedAt < 0 ||
      parsed.fetchedAt > Date.now()
    ) {
      return undefined;
    }
    const catalog = normalizeModelsDevCatalog(parsed.catalog);
    return catalog ? { fetchedAt: parsed.fetchedAt, catalog } : undefined;
  } catch {
    return undefined;
  }
}

function refreshModelsDevCatalog(key: string, options: DiscoveryOptions): Promise<ModelsDevResponse | undefined> {
  const active = modelsDevRefreshes.get(key);
  if (active) return active;
  const refresh = (async () => {
    try {
      const catalog = normalizeModelsDevCatalog(await fetchPublicJson<unknown>(MODELS_DEV_URL, options));
      if (!catalog) return undefined;
      const cache = { fetchedAt: Date.now(), catalog };
      modelsDevCaches.set(key, cache);
      if (options.modelsDevCachePath) {
        await writeJsonAtomic(options.modelsDevCachePath, cache).catch(() => undefined);
      }
      return catalog;
    } catch {
      return undefined;
    } finally {
      modelsDevRefreshes.delete(key);
    }
  })();
  modelsDevRefreshes.set(key, refresh);
  return refresh;
}

async function getModelsDevCatalog(options: DiscoveryOptions): Promise<ModelsDevResponse | undefined> {
  const key = options.modelsDevCachePath ?? MODELS_DEV_URL;
  const refreshOptions = { ...options, timeoutMs: DEFAULT_TIMEOUT_MS, signal: undefined };
  let cache = modelsDevCaches.get(key);
  if (!cache && options.modelsDevCachePath) {
    cache = await readModelsDevCache(options.modelsDevCachePath);
    if (cache) modelsDevCaches.set(key, cache);
  }
  if (!cache) {
    const timeout = AbortSignal.timeout(options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
    return awaitWithSignal(
      refreshModelsDevCatalog(key, refreshOptions),
      options.signal ? AbortSignal.any([options.signal, timeout]) : timeout,
    );
  }
  if (Date.now() - cache.fetchedAt < MODELS_DEV_CACHE_TTL_MS) return cache.catalog;
  void refreshModelsDevCatalog(key, refreshOptions);
  return cache.catalog;
}

function mapModelsDevMetadata(model: ModelsDevModel | undefined): Partial<DiscoveredModel> {
  if (!model) return {};
  const metadata: Partial<DiscoveredModel> = {};
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

function mapFromModelInfo(entry: ModelInfoEntry): DiscoveredModel | undefined {
  const id = entry.model_name;
  if (!id) return undefined;
  const info = entry.model_info ?? {};
  if (!isChatStyleMode(info.mode)) return undefined;
  const catalogModel = findCatalogModel(id);
  const api = selectApi(info.mode, info.litellm_provider, info.base_model, id);
  const model: DiscoveredModel = {
    id,
    name: id,
    reasoning: false,
    input: info.supports_vision ? ["text", "image"] : ["text"],
    cost: mapModelInfoCost(info, catalogModel?.cost),
    contextWindow: info.max_input_tokens ?? DEFAULT_CONTEXT_WINDOW,
    maxTokens: info.max_output_tokens ?? DEFAULT_MAX_TOKENS,
    api,
    compat: buildCompat(id, catalogModel, api),
  };
  applyReasoningPolicy(model, catalogModel, info.supports_reasoning, {
    xhigh: info.supports_xhigh_reasoning_effort,
    max: info.supports_max_reasoning_effort,
  });
  return model;
}

function mapFromHealthModelInfo(entry: ModelInfoEntry, fallbackId: string | undefined): DiscoveredModel | undefined {
  return entry.model_name || fallbackId
    ? mapFromModelInfo({ ...entry, model_name: entry.model_name ?? fallbackId })
    : undefined;
}

function mapFromHealthEndpoint(entry: { model?: string }): DiscoveredModel | undefined {
  const id = entry.model;
  if (!id) return undefined;
  const catalogModel = findCatalogModel(id);
  const api = selectApi(undefined);
  const model: DiscoveredModel = {
    id,
    name: catalogModel?.name ?? id,
    reasoning: false,
    input: catalogModel?.input ?? ["text"],
    cost: catalogModel?.cost ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: catalogModel?.contextWindow ?? DEFAULT_CONTEXT_WINDOW,
    maxTokens: catalogModel?.maxTokens ?? DEFAULT_MAX_TOKENS,
    api,
    compat: buildCompat(id, catalogModel, api),
  };
  applyReasoningPolicy(model, catalogModel);
  return model;
}

function mapFromModelsList(
  entry: ModelsListEntry,
  modelsDev: ModelsDevResponse | undefined,
): DiscoveredModel | undefined {
  const id = entry.id;
  if (!id) return undefined;
  const catalogModel = findCatalogModel(id, entry.owned_by);
  const modelsDevMetadata = mapModelsDevMetadata(findModelsDevModel(modelsDev, id, entry.owned_by));
  const api = selectApi(undefined);
  const model: DiscoveredModel = {
    id,
    name: modelsDevMetadata.name ?? catalogModel?.name ?? `${id} (no metadata)`,
    reasoning: false,
    input: modelsDevMetadata.input ?? catalogModel?.input ?? ["text"],
    cost: modelsDevMetadata.cost ?? catalogModel?.cost ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: modelsDevMetadata.contextWindow ?? catalogModel?.contextWindow ?? DEFAULT_CONTEXT_WINDOW,
    maxTokens: modelsDevMetadata.maxTokens ?? catalogModel?.maxTokens ?? DEFAULT_MAX_TOKENS,
    api,
    compat: buildCompat(id, catalogModel, api),
  };
  applyReasoningPolicy(model, catalogModel, catalogModel?.reasoning ?? modelsDevMetadata.reasoning);
  return model;
}

async function discoverFromHealth(
  base: string,
  apiKey: string,
  options: DiscoveryOptions & { onProgress?: (message: string) => void; silent?: boolean },
): Promise<DiscoveredModel[]> {
  const progress = options.silent ? undefined : options.onProgress;
  progress?.("Querying /health endpoint...");
  const healthResult = await fetchJson<HealthResponse>(`${base}/health`, apiKey, options);
  if (!healthResult.ok) return [];
  const endpoints = (healthResult.data.healthy_endpoints ?? []).filter((entry) => entry.model || entry.model_id);
  progress?.(`Discovered ${endpoints.length} model endpoints, fetching details...`);
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
      if (completed % 10 === 0 || completed === endpoints.length) {
        progress?.(`Fetched ${completed}/${endpoints.length} models...`);
      }
      return model;
    }),
  );
  return models.filter((model): model is DiscoveredModel => model !== undefined);
}

function deduplicateModels(models: DiscoveredModel[]): DiscoveredModel[] {
  const seen = new Set<string>();
  return models.filter((m) => {
    if (seen.has(m.id)) return false;
    seen.add(m.id);
    return true;
  });
}

export async function discoverModels(
  baseUrl: string,
  apiKey: string,
  options: DiscoveryOptions & { onProgress?: (message: string) => void; silent?: boolean } = {},
): Promise<DiscoveryResult> {
  const base = normalizeBaseUrl(baseUrl);
  const progress = options.silent ? undefined : options.onProgress;
  progress?.("Querying /model/info endpoint...");
  const infoResult = await fetchJson<ModelInfoResponse>(`${base}/model/info`, apiKey, options);
  if (infoResult.ok) {
    const entries = new Map<string, ModelInfoEntry>();
    for (const entry of infoResult.data.data ?? []) {
      if (!entry.model_name) continue;
      const previous = entries.get(entry.model_name);
      entries.set(entry.model_name, {
        ...previous,
        ...entry,
        model_info: { ...previous?.model_info, ...entry.model_info },
      });
    }
    const models = [...entries.values()].map(mapFromModelInfo).filter((m): m is DiscoveredModel => m !== undefined);
    return { source: "model_info", models };
  }
  if (![401, 403, 404].includes(infoResult.status)) {
    throw new Error(`/model/info returned ${infoResult.status}`);
  }
  progress?.("/model/info unavailable, trying /v1/models...");
  const listResult = await fetchJson<ModelsListResponse>(`${base}/v1/models`, apiKey, options);
  if (!listResult.ok) {
    if ([401, 403, 404].includes(listResult.status)) {
      progress?.("/v1/models unavailable, falling back to /health endpoint...");
      const models = await discoverFromHealth(base, apiKey, options);
      if (models.length > 0) return { source: "health", models: deduplicateModels(models) };
    }
    throw new Error(`/v1/models returned ${listResult.status}`);
  }
  let modelsDev: ModelsDevResponse | undefined;
  if (options.modelsDev !== false) {
    progress?.("Loading models.dev catalog for metadata enrichment...");
    modelsDev = await getModelsDevCatalog(options);
  }
  const models = (listResult.data.data ?? [])
    .map((entry) => mapFromModelsList(entry, modelsDev))
    .filter((m): m is DiscoveredModel => m !== undefined);
  return { source: "models_list", models: deduplicateModels(models) };
}
