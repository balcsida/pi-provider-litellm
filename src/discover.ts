import { isIP } from "node:net";
import type { Api, Model } from "@earendil-works/pi-ai";
import { getModels, getProviders } from "@earendil-works/pi-ai/compat";
import type { BuiltinProvider } from "@earendil-works/pi-ai/providers/all";
import type {
  DiscoveredModel,
  DiscoveredModelFor,
  DiscoveryOptions,
  DiscoveryResult,
  HealthResponse,
  ModelInfoEntry,
  ModelInfoResponse,
  ModelProtocol,
  ModelsListEntry,
  ModelsListResponse,
} from "./types.js";

const DEFAULT_TIMEOUT_MS = 5000;
const DEFAULT_CONTEXT_WINDOW = 128_000;
const DEFAULT_MAX_TOKENS = 16_384;
const KNOWN_PROVIDER_SET = new Set<string>(getProviders());

export function normalizeBaseUrl(input: string, allowInsecureHttp = false): string {
  const url = new URL(input);
  const hostname = url.hostname.toLowerCase();
  const loopback =
    hostname === "localhost" || hostname === "[::1]" || (isIP(hostname) === 4 && hostname.startsWith("127."));
  if (url.protocol !== "https:" && !(url.protocol === "http:" && (loopback || allowInsecureHttp))) {
    throw new Error("LiteLLM base URL must use HTTPS except for loopback hosts");
  }
  return input.replace(/\/+$/, "").replace(/\/v1\/?$/i, "");
}

const RESPONSES_MODE_PATTERN = /^responses?$/i;

function isResponsesMode(mode: string | null | undefined): boolean {
  return mode != null && RESPONSES_MODE_PATTERN.test(mode);
}

function isChatStyleMode(mode: string | null | undefined): boolean {
  return mode == null || mode === "chat" || isResponsesMode(mode);
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

const MOONSHOT_ROUTE_PROVIDERS = new Set(["moonshot", "moonshotai"]);

function isMoonshotRoute(entry: ModelInfoEntry): boolean {
  const params = entry.litellm_params;
  if (!params) return false;
  const model = params.model?.trim();
  if (!model) return false;
  const providers = [
    params.custom_llm_provider?.trim(),
    model.includes("/") ? model.split("/", 1)[0] : undefined,
  ].filter((provider): provider is string => Boolean(provider));
  return providers.length > 0 && providers.every((provider) => MOONSHOT_ROUTE_PROVIDERS.has(provider.toLowerCase()));
}

function shouldSuppressReasoningContent(modelId: string, entry: ModelInfoEntry): boolean {
  const routeModelId = entry.litellm_params?.model;
  return (
    isMoonshotRoute(entry) &&
    !FORCED_THINKING_MODEL_PATTERN.test(modelId) &&
    !(routeModelId && FORCED_THINKING_MODEL_PATTERN.test(routeModelId))
  );
}

function aggregateSuppressionEvidence(evidence: Iterable<boolean>): boolean {
  let hasEvidence = false;
  for (const suppress of evidence) {
    hasEvidence = true;
    if (!suppress) return false;
  }
  return hasEvidence;
}

export function emitsThinkTags(modelId: string): boolean {
  return isMoonshotModel(modelId) && !FORCED_THINKING_MODEL_PATTERN.test(modelId);
}

export function responsesCompat(modelId: string): DiscoveredModelFor<"openai-responses">["compat"] {
  // Pi's Responses transport has no cacheControlFormat setting and uses
  // Responses-native prompt-cache fields instead of Anthropic cache_control markers.
  return isMoonshotModel(modelId) ? { supportsDeveloperRole: false } : undefined;
}

export function completionsCompat(modelId: string): DiscoveredModelFor<"openai-completions">["compat"] {
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

export function modelProtocol(modelId: string, mode?: string | null): ModelProtocol {
  return isResponsesMode(mode)
    ? { api: "openai-responses", compat: responsesCompat(modelId) }
    : { api: "openai-completions", compat: completionsCompat(modelId) };
}

export function buildCompat(modelId: string): DiscoveredModelFor<"openai-completions">["compat"] {
  return completionsCompat(modelId);
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

export function enrichCachedModel(model: Model<Api>): Model<Api> {
  // ponytail: legacy cache lacks field provenance; add per-field cache provenance if strict preservation becomes necessary.
  if (
    !model.name.endsWith(" (no metadata)") ||
    model.reasoning ||
    model.thinkingLevelMap !== undefined ||
    model.input.length !== 1 ||
    model.input[0] !== "text" ||
    model.cost.input !== 0 ||
    model.cost.output !== 0 ||
    model.cost.cacheRead !== 0 ||
    model.cost.cacheWrite !== 0 ||
    model.cost.tiers !== undefined ||
    model.contextWindow !== DEFAULT_CONTEXT_WINDOW ||
    model.maxTokens !== DEFAULT_MAX_TOKENS
  ) {
    return model;
  }
  const catalogModel = findCatalogModel(model.id);
  if (!catalogModel) return model;
  return {
    ...model,
    name: catalogModel.name,
    reasoning: catalogModel.reasoning,
    thinkingLevelMap: catalogModel.thinkingLevelMap,
    input: catalogModel.input,
    cost: catalogModel.cost,
    contextWindow: catalogModel.contextWindow,
    maxTokens: catalogModel.maxTokens,
  };
}

function catalogLookupIds(id: string): string[] {
  const lookupIds = new Set([id]);
  const unprefixed = id.includes("/") ? id.slice(id.indexOf("/") + 1) : id;
  lookupIds.add(unprefixed);

  const anthropicAlias = unprefixed.toLowerCase().replaceAll(".", "-");
  const match = /^(?:claude-)?(opus|sonnet|haiku)-(\d+)-(\d+)$/.exec(anthropicAlias);
  if (match) lookupIds.add(`claude-${match[1]}-${match[2]}-${match[3]}`);
  if (anthropicAlias === "fable-5" || anthropicAlias === "opus-5") lookupIds.add(`claude-${anthropicAlias}`);

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

function mapReasoningEfforts(
  info: NonNullable<ModelInfoEntry["model_info"]>,
): NonNullable<DiscoveredModel["thinkingLevelMap"]> | undefined {
  const flags = [
    ["off", "none", info.supports_none_reasoning_effort],
    ["minimal", "minimal", info.supports_minimal_reasoning_effort],
    ["low", "low", info.supports_low_reasoning_effort],
    ["medium", "medium", info.supports_medium_reasoning_effort],
    ["high", "high", info.supports_high_reasoning_effort],
    ["xhigh", "xhigh", info.supports_xhigh_reasoning_effort],
    ["max", "max", info.supports_max_reasoning_effort],
  ] as const;
  const map = Object.fromEntries(
    flags
      .filter(([, , supported]) => supported !== undefined)
      .map(([level, value, supported]) => [level, supported ? value : null]),
  ) as NonNullable<DiscoveredModel["thinkingLevelMap"]>;
  return Object.keys(map).length > 0 ? map : undefined;
}

function mapFromModelInfo(
  entry: ModelInfoEntry,
  suppressReasoningContent = shouldSuppressReasoningContent(entry.model_name ?? "", entry),
): DiscoveredModel | undefined {
  const id = entry.model_name;
  if (!id) return undefined;
  const info = entry.model_info ?? {};
  if (!isChatStyleMode(info.mode)) return undefined;
  const responsesMode = isResponsesMode(info.mode);
  const catalogModel = findCatalogModel(id);
  const reasoningEffortMap = mapReasoningEfforts(info);
  const thinkingLevelMap =
    catalogModel?.thinkingLevelMap || reasoningEffortMap
      ? { ...catalogModel?.thinkingLevelMap, ...reasoningEffortMap }
      : undefined;
  return {
    id,
    name: id,
    reasoning: info.supports_reasoning ?? false,
    ...(thinkingLevelMap ? { thinkingLevelMap } : {}),
    input: info.supports_vision ? ["text", "image"] : ["text"],
    cost: mapModelInfoCost(info, catalogModel?.cost),
    contextWindow: info.max_input_tokens ?? DEFAULT_CONTEXT_WINDOW,
    maxTokens: info.max_output_tokens ?? DEFAULT_MAX_TOKENS,
    ...modelProtocol(id, responsesMode ? "responses" : "chat"),
    ...(suppressReasoningContent ? { suppressReasoningContent: true } : {}),
  };
}

function mapFromHealthModelInfo(entry: ModelInfoEntry, fallbackId: string | undefined): DiscoveredModel | undefined {
  if (entry.model_name || !fallbackId) return mapFromModelInfo(entry);
  const model = mapFromModelInfo({ ...entry, model_name: fallbackId });
  if (model) delete model.thinkingLevelMap;
  return model;
}

function mapFromHealthEndpoint(entry: { model?: string }): DiscoveredModel | undefined {
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
    ...modelProtocol(id),
  };
}

function mapFromModelsList(entry: ModelsListEntry): DiscoveredModel | undefined {
  const id = entry.id;
  if (!id) return undefined;
  const catalogModel = findCatalogModel(id, entry.owned_by);
  return {
    id,
    name: catalogModel?.name ?? `${id} (no metadata)`,
    reasoning: catalogModel?.reasoning ?? false,
    thinkingLevelMap: catalogModel?.thinkingLevelMap,
    input: catalogModel?.input ?? ["text"],
    cost: catalogModel?.cost ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: catalogModel?.contextWindow ?? DEFAULT_CONTEXT_WINDOW,
    maxTokens: catalogModel?.maxTokens ?? DEFAULT_MAX_TOKENS,
    ...modelProtocol(id),
  };
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
  const entries = new Map<string, { model: DiscoveredModel; suppressions: boolean[] }>();
  for (const model of models) {
    const existing = entries.get(model.id);
    if (existing) {
      existing.suppressions.push(model.suppressReasoningContent === true);
    } else {
      entries.set(model.id, { model, suppressions: [model.suppressReasoningContent === true] });
    }
  }
  return [...entries.values()].map(({ model, suppressions }) => {
    const deduplicated = { ...model };
    if (aggregateSuppressionEvidence(suppressions)) deduplicated.suppressReasoningContent = true;
    else delete deduplicated.suppressReasoningContent;
    return deduplicated;
  });
}

export async function discoverModels(
  baseUrl: string,
  apiKey: string,
  options: DiscoveryOptions & { onProgress?: (message: string) => void; silent?: boolean } = {},
): Promise<DiscoveryResult> {
  const base = normalizeBaseUrl(baseUrl, options.allowInsecureHttp);
  const progress = options.silent ? undefined : options.onProgress;
  progress?.("Querying /model/info endpoint...");
  const infoResult = await fetchJson<ModelInfoResponse>(`${base}/model/info`, apiKey, options);
  if (infoResult.ok) {
    const entries = new Map<string, ModelInfoEntry>();
    const suppressionEvidence = new Map<string, Set<boolean>>();
    for (const entry of infoResult.data.data ?? []) {
      if (!entry.model_name) continue;
      const previous = entries.get(entry.model_name);
      const suppressions = suppressionEvidence.get(entry.model_name) ?? new Set<boolean>();
      suppressions.add(shouldSuppressReasoningContent(entry.model_name, entry));
      suppressionEvidence.set(entry.model_name, suppressions);
      entries.set(entry.model_name, {
        ...previous,
        ...entry,
        model_info: { ...previous?.model_info, ...entry.model_info },
      });
    }
    let models = [...entries.entries()]
      .map(([id, entry]) => mapFromModelInfo(entry, aggregateSuppressionEvidence(suppressionEvidence.get(id)!)))
      .filter((m): m is DiscoveredModel => m !== undefined);
    // LiteLLM's /model/info does NOT expand wildcard model_name entries (e.g.
    // "lemonade/*" backed by model: openai/* + check_provider_endpoint: true)
    // — it returns the literal wildcard only. The discovered ids live in
    // /v1/models instead. When /model/info contains any wildcard id, also query
    // /v1/models and merge the expanded (non-wildcard) entries in, dropping the
    // raw wildcard row so it doesn't surface as a phantom model choice.
    // Ref: docs.litellm.ai/docs/proxy/model_discovery
    if (models.some((m) => m.id.includes("*"))) {
      progress?.("/model/info has wildcard entries, expanding via /v1/models...");
      const listResult = await fetchJson<ModelsListResponse>(`${base}/v1/models`, apiKey, options);
      if (listResult.ok) {
        const expanded = (listResult.data.data ?? [])
          .map(mapFromModelsList)
          .filter((m): m is DiscoveredModel => m !== undefined && !m.id.includes("*"));
        const seen = new Set<string>(models.map((m) => m.id));
        models = [...models.filter((m) => !m.id.includes("*")), ...expanded.filter((m) => !seen.has(m.id))];
      }
    }
    return { source: "model_info", models: deduplicateModels(models) };
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
  const models = (listResult.data.data ?? [])
    .map(mapFromModelsList)
    .filter((m): m is DiscoveredModel => m !== undefined);
  return { source: "models_list", models: deduplicateModels(models) };
}
