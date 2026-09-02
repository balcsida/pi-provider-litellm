import { isIP } from "node:net";
import type { Api, Model } from "@earendil-works/pi-ai";
import { getModels, getProviders } from "@earendil-works/pi-ai/compat";
import type { BuiltinProvider } from "@earendil-works/pi-ai/providers/all";
import {
  type CatalogResolution,
  catalogResolution,
  DEFAULT_CONTEXT_WINDOW,
  DEFAULT_MAX_TOKENS,
  type MessagesBackendCompat,
  normalizedMode,
  reduceModelGroup,
  type SemanticFamily,
  wireString,
} from "./model-groups.js";
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
  const model = wireString(params.model)?.trim();
  if (!model) return false;
  const providers = [
    wireString(params.custom_llm_provider)?.trim(),
    model.includes("/") ? model.split("/", 1)[0] : undefined,
  ].filter((provider): provider is string => Boolean(provider));
  return providers.length > 0 && providers.every((provider) => MOONSHOT_ROUTE_PROVIDERS.has(provider.toLowerCase()));
}

function shouldSuppressRouteReasoningContent(modelId: string, entry: ModelInfoEntry): boolean {
  const routeModelId = wireString(entry.litellm_params?.model);
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

export function buildCompat(
  modelId: string,
  api: DiscoveredModel["api"] = "openai-completions",
  semanticFamily?: SemanticFamily,
): DiscoveredModel["compat"] {
  if (api === "anthropic-messages") return undefined;
  if (isMoonshotModel(modelId)) {
    return {
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsStrictMode: false,
      maxTokensField: "max_tokens",
    };
  }
  if (semanticFamily === "claude" || ANTHROPIC_MODEL_PATTERN.test(modelId)) {
    return { supportsStore: false, cacheControlFormat: "anthropic" };
  }
  return { supportsStore: false };
}

function toKnownProvider(provider: string | undefined): BuiltinProvider | undefined {
  if (!provider) return undefined;
  const normalized = provider.trim().toLowerCase();
  return KNOWN_PROVIDER_SET.has(normalized) ? (normalized as BuiltinProvider) : undefined;
}

// Anthropic recognition is derived from the single `catalogLookupIds` rule so a
// second alias pattern cannot drift away from it. Every Anthropic catalog id and
// every alias that maps onto one is canonicalized to a `claude-` lookup id,
// including single-number names and dated snapshots.
function catalogProviderCandidates(
  lookupIds: readonly string[],
  id: string,
  ownedBy?: string,
  adapterProvider?: BuiltinProvider,
): BuiltinProvider[] {
  // A recognized adapter is authoritative provider evidence. If its catalog
  // misses, do not try a conflicting provider qualifier from model or base_model.
  if (adapterProvider) return [adapterProvider];
  const ownedByProvider = toKnownProvider(ownedBy);
  const prefixProvider = toKnownProvider(id.split("/")[0]);
  // Two explicit provider identities are contradictory even when the first
  // provider's catalog has no matching model. Do not borrow from the second.
  if (ownedByProvider && prefixProvider && ownedByProvider !== prefixProvider) return [];
  const candidates = [ownedByProvider, prefixProvider].filter(
    (provider): provider is BuiltinProvider => provider !== undefined,
  );
  if (lookupIds.some((lookupId) => lookupId.startsWith("claude-"))) candidates.push("anthropic");
  return [...new Set(candidates)];
}

function resolveCatalogModel(
  id: string,
  ownedBy?: string,
  adapterProvider?: BuiltinProvider,
): { provider: BuiltinProvider; model: Model<Api> } | undefined {
  const lookupIds = catalogLookupIds(id);
  for (const provider of catalogProviderCandidates(lookupIds, id, ownedBy, adapterProvider)) {
    const model = findCatalogModelInProvider(provider, lookupIds);
    if (model) return { provider, model };
  }
  return undefined;
}

function findCatalogModel(id: string, ownedBy?: string): Model<Api> | undefined {
  return resolveCatalogModel(id, ownedBy)?.model;
}

export function enrichCachedModel(model: Model<Api>): Model<Api> {
  // ponytail: legacy cache lacks field provenance; add per-field cache provenance if
  // strict preservation becomes necessary. The two-marker contract below is the
  // workaround for that absence, so this deferral is still open, not resolved.
  //
  // This sentinel is emitted only by the evidence-free `/v1/models` fallback, so
  // re-deriving catalog metadata from the model id here cannot re-authorize a
  // reduced `/model/info` group whose catalog authority was withheld. Reduced
  // groups carry the distinct ` (incomplete metadata)` marker instead.
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

function undecoratedBackendIds(id: string): string[] {
  const routed = (id.split("/").pop() ?? id).toLowerCase();
  const undecorated = routed.replace(/-v\d+(?::\d+)?$/, "").replace(/@[a-z0-9-]+$/, "");
  return [
    ...new Set([routed, routed.replace(/:\d+$/, ""), undecorated, undecorated.replace(/-\d{8}$/, "")].filter(Boolean)),
  ];
}

function catalogLookupIds(id: string): string[] {
  const lookupIds = new Set([id]);
  const unprefixed = id.includes("/") ? id.slice(id.indexOf("/") + 1) : id;
  lookupIds.add(unprefixed);
  for (const candidate of undecoratedBackendIds(id)) lookupIds.add(candidate);

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

function semanticFamily(id: string): SemanticFamily | undefined {
  const value = id.toLowerCase();
  if (/(?:^|[./_-])(?:anthropic|claude|opus|sonnet|haiku|fable)(?:$|[./_:-])/.test(value)) return "claude";
  if (/(?:^|[./_-])(?:moonshotai|moonshot|kimi)(?:$|[./_:-])/.test(value)) return "kimi";
  if (/(?:^|[./_-])deepseek(?:$|[./_:-])/.test(value)) return "deepseek";
  if (/(?:^|[./_-])gemini(?:$|[./_:-])/.test(value)) return "gemini";
  if (/(?:^|[./_-])(?:openai|gpt|o\d)(?:$|[./_:-])/.test(value)) return "openai";
  return undefined;
}

function messagesCompatOf(model: Model<Api>): MessagesBackendCompat | undefined {
  // Native Messages is safe only when Pi's Anthropic catalog supplies the exact
  // serializer policy for that generation; a provider adapter catalog describes
  // backend access and pricing, not the Anthropic wire contract LiteLLM accepts.
  if (model.api !== "anthropic-messages") return undefined;
  const compat = (model as Model<"anthropic-messages">).compat;
  const carried: MessagesBackendCompat = {};
  if (compat?.forceAdaptiveThinking !== undefined) carried.forceAdaptiveThinking = compat.forceAdaptiveThinking;
  if (compat?.supportsTemperature !== undefined) carried.supportsTemperature = compat.supportsTemperature;
  if (compat?.supportsStrictTools !== undefined) carried.supportsStrictTools = compat.supportsStrictTools;
  return carried;
}

function anthropicBackendLookupIds(id: string): string[] {
  const routed = (id.split("/").pop() ?? id).toLowerCase();
  const base = routed.replace(/^(?:[a-z0-9-]+\.)*anthropic[./]/, "");
  return undecoratedBackendIds(base);
}

function messagesCompatFromBackend(id: string): MessagesBackendCompat | undefined {
  const model = findCatalogModelInProvider("anthropic", anthropicBackendLookupIds(id));
  return model ? messagesCompatOf(model) : undefined;
}

// These are the LiteLLM adapters whose native request path can terminate at a
// Claude backend. Other adapters may expose Claude-like public aliases, but an
// alias alone is not evidence that LiteLLM accepts the Anthropic Messages schema.
const CLAUDE_CAPABLE_ADAPTERS = new Set(["anthropic", "bedrock", "bedrock_converse", "vertex_ai"]);
const CLAUDE_MODEL_PATTERN = /(?:^|[./_-])(?:claude|opus|sonnet|haiku|fable)(?:$|[./_:-])/i;

const ADAPTER_CATALOG_PROVIDERS: Readonly<Record<string, BuiltinProvider>> = {
  anthropic: "anthropic",
  azure: "azure-openai-responses",
  azure_ai: "azure-openai-responses",
  bedrock: "amazon-bedrock",
  bedrock_converse: "amazon-bedrock",
  deepseek: "deepseek",
  fireworks_ai: "fireworks",
  gemini: "google",
  moonshot: "moonshotai",
  nvidia_nim: "nvidia",
  openai: "openai",
  together_ai: "together",
  vertex_ai: "google-vertex",
};

function adapterCatalogProvider(adapter: unknown): BuiltinProvider | undefined {
  const normalized = wireString(adapter)?.trim().toLowerCase();
  return normalized ? (ADAPTER_CATALOG_PROVIDERS[normalized] ?? toKnownProvider(normalized)) : undefined;
}

export function resolveModelInfoCatalog(entry: ModelInfoEntry): CatalogResolution | undefined {
  const adapter = wireString(entry.model_info?.litellm_provider)?.trim().toLowerCase();
  const adapterProvider = adapterCatalogProvider(adapter);
  const routingModel = wireString(entry.litellm_params?.model)?.trim() || undefined;
  const baseModel = wireString(entry.model_info?.base_model)?.trim() || undefined;
  const routingFamily = routingModel ? semanticFamily(routingModel) : undefined;
  const baseFamily = baseModel ? semanticFamily(baseModel) : undefined;
  const routingCatalog = routingModel ? resolveCatalogModel(routingModel, undefined, adapterProvider) : undefined;
  const baseCatalog = baseModel ? resolveCatalogModel(baseModel, undefined, adapterProvider) : undefined;
  const conflictingFamilies =
    routingModel !== undefined &&
    baseFamily !== undefined &&
    ((routingFamily !== undefined && routingFamily !== baseFamily) ||
      (routingCatalog !== undefined && semanticFamily(routingCatalog.model.id) !== baseFamily));
  const conflictingCatalogModels =
    routingCatalog !== undefined &&
    baseCatalog !== undefined &&
    (routingCatalog.provider !== baseCatalog.provider || routingCatalog.model.id !== baseCatalog.model.id);
  if (conflictingFamilies) return undefined;
  const semantic =
    routingFamily ?? (routingCatalog ? semanticFamily(routingCatalog.model.id) : undefined) ?? baseFamily;
  const routingAllowsBaseWitness =
    routingModel === undefined ||
    routingFamily === "claude" ||
    (routingFamily === undefined && routingCatalog === undefined);
  const claudeEvidence =
    adapter !== undefined &&
    CLAUDE_CAPABLE_ADAPTERS.has(adapter) &&
    ((routingModel !== undefined && routingFamily === "claude" && CLAUDE_MODEL_PATTERN.test(routingModel)) ||
      (!conflictingFamilies &&
        routingAllowsBaseWitness &&
        baseModel !== undefined &&
        baseFamily === "claude" &&
        CLAUDE_MODEL_PATTERN.test(baseModel)));
  const candidates = [routingModel, baseModel].filter((candidate): candidate is string => candidate !== undefined);
  const candidateCompats = claudeEvidence ? candidates.map(messagesCompatFromBackend) : [];
  // Every declared backend identity that participates in Claude evidence must
  // resolve a Messages policy. Filtering unresolved entries would incorrectly
  // make the remaining policy look unanimous.
  const allCandidatesHaveCompat = candidateCompats.length === candidates.length && candidateCompats.every(Boolean);
  const compatIdentities = new Set(candidateCompats.map((value) => JSON.stringify(value)));
  const compat = allCandidatesHaveCompat && compatIdentities.size === 1 ? candidateCompats[0] : undefined;
  if (conflictingCatalogModels) return semantic ? { semanticFamily: semantic } : undefined;

  for (const candidate of candidates) {
    const resolved = resolveCatalogModel(candidate, undefined, adapterProvider);
    if (resolved) {
      return {
        ...catalogResolution(resolved.provider, semantic ?? semanticFamily(resolved.model.id), resolved.model),
        ...(compat ? { messagesCompat: compat } : {}),
      };
    }
  }
  return semantic ? { semanticFamily: semantic, ...(compat ? { messagesCompat: compat } : {}) } : undefined;
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

const AMBIGUOUS_AUTHORITY_SAMPLE = 3;

// Reported routes, so a persistent misconfiguration is announced once rather than on
// every background refresh and every `/model` open. Keyed by route rather than a
// single flag so a newly ambiguous route is still reported. Mirrors the once-per-
// process diagnostic set in src/index.ts.
const reportedAmbiguousRoutes = new Set<string>();

// Withholding catalog authority can be invisible in the model name when the router
// supplies complete prices; limits and other catalog-derived metadata may still use
// conservative defaults. Report that degradation regardless of
// LITELLM_VERBOSE_DISCOVERY, carrying only a count and bounded public route ids.
function reportAmbiguousCatalogAuthority(routes: readonly string[]): void {
  const unreported = routes.filter((route) => !reportedAmbiguousRoutes.has(route));
  if (unreported.length === 0) return;
  for (const route of unreported) reportedAmbiguousRoutes.add(route);
  const hidden = unreported.length - AMBIGUOUS_AUTHORITY_SAMPLE;
  const sample = unreported.slice(0, AMBIGUOUS_AUTHORITY_SAMPLE).join(", ");
  process.stderr.write(
    // "missing or conflicting": the group is also withheld when one deployment
    // resolves a provider and another supplies no usable backend evidence at all,
    // which points at a different fix than a genuine provider conflict.
    `LiteLLM discovery: ${unreported.length} route group(s) have missing or conflicting deployment provider ` +
      `evidence; catalog limits, pricing, and reasoning metadata are withheld: ${sample}` +
      `${hidden > 0 ? ` (+${hidden} more)` : ""}\n`,
  );
}

function mapFromModelInfoGroup(
  entries: readonly ModelInfoEntry[],
  ambiguousRoutes?: string[],
): DiscoveredModel | undefined {
  const reduced = reduceModelGroup(entries, resolveModelInfoCatalog);
  if (!reduced) return undefined;
  if (reduced.catalogAuthorityAmbiguous) ambiguousRoutes?.push(reduced.id);
  const suppressReasoningContent = aggregateSuppressionEvidence(
    entries
      .filter((entry) => normalizedMode(entry.model_info?.mode) !== "unsupported")
      .map((entry) => shouldSuppressRouteReasoningContent(reduced.id, entry)),
  );
  const shared = {
    id: reduced.id,
    // Reduced groups never borrow the ` (no metadata)` sentinel, which authorizes
    // catalog re-derivation from the model id during offline cache reads.
    name: reduced.hasCompleteMetadata ? reduced.id : `${reduced.id} (incomplete metadata)`,
    reasoning: reduced.reasoning,
    ...(reduced.thinkingLevelMap ? { thinkingLevelMap: reduced.thinkingLevelMap } : {}),
    input: (reduced.vision ? ["text", "image"] : ["text"]) as ("text" | "image")[],
    cost: reduced.cost,
    contextWindow: reduced.contextWindow,
    maxTokens: reduced.maxTokens,
    ...(suppressReasoningContent ? { suppressReasoningContent: true } : {}),
  };
  if (reduced.api === "anthropic-messages") {
    return { ...shared, api: "anthropic-messages", compat: reduced.messagesCompat };
  }
  if (reduced.api === "openai-responses") {
    return {
      ...shared,
      api: "openai-responses",
      compat: buildCompat(reduced.id, "openai-responses", reduced.semanticFamily),
    };
  }
  return {
    ...shared,
    api: "openai-completions",
    compat: buildCompat(reduced.id, "openai-completions", reduced.semanticFamily),
  };
}

interface HealthDeployment {
  entry: ModelInfoEntry;
  // Detail route presence is separate from the public `/health` route: missing
  // detail identity cannot authorize Messages-only request controls.
  hasDetailRoute: boolean;
}

function healthDeployment(
  detail: ModelInfoEntry | undefined,
  fallbackRoute: string | undefined,
  deploymentId: string | undefined,
): HealthDeployment | undefined {
  const detailRoute = wireString(detail?.model_name)?.trim() || undefined;
  const healthRoute = fallbackRoute?.trim() || undefined;
  const route = healthRoute ?? detailRoute;
  if (!route) return undefined;
  // A detail request is correlated by deployment ID, but its public route must
  // still agree with the route established by /health. Mismatched detail may
  // describe another route and cannot contribute metadata to this deployment.
  if (!detail || (healthRoute && detailRoute && detailRoute !== healthRoute)) {
    return {
      entry: { model_name: route, model_info: { ...(deploymentId ? { id: deploymentId } : {}), mode: "chat" } },
      hasDetailRoute: false,
    };
  }
  return {
    entry: {
      ...detail,
      model_name: route,
      model_info: {
        ...detail.model_info,
        ...(wireString(detail.model_info?.id)?.trim() || !deploymentId ? {} : { id: deploymentId }),
      },
    },
    hasDetailRoute: detailRoute !== undefined,
  };
}

function mapFromModelsList(entry: ModelsListEntry): DiscoveredModel | undefined {
  const id = wireString(entry.id);
  if (!id) return undefined;
  const ownedBy = wireString(entry.owned_by);
  const ownedByProvider = toKnownProvider(ownedBy);
  const prefixProvider = toKnownProvider(id.split("/")[0]);
  const conflictingProviderEvidence =
    ownedByProvider !== undefined && prefixProvider !== undefined && ownedByProvider !== prefixProvider;
  const catalogModel = conflictingProviderEvidence ? undefined : findCatalogModel(id, ownedBy);
  return {
    id,
    name: catalogModel?.name ?? `${id}${conflictingProviderEvidence ? " (incomplete metadata)" : " (no metadata)"}`,
    reasoning: catalogModel?.reasoning ?? false,
    thinkingLevelMap: catalogModel?.thinkingLevelMap,
    input: catalogModel?.input ?? ["text"],
    cost: catalogModel?.cost ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: catalogModel?.contextWindow ?? DEFAULT_CONTEXT_WINDOW,
    maxTokens: catalogModel?.maxTokens ?? DEFAULT_MAX_TOKENS,
    api: "openai-completions",
    compat: buildCompat(id),
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
  const deployments = await Promise.all(
    endpoints.map(async (endpoint) => {
      const route = wireString(endpoint.model)?.trim() || undefined;
      const deploymentId = wireString(endpoint.model_id)?.trim() || undefined;
      let detail: ModelInfoEntry | undefined;
      if (deploymentId) {
        const infoResult = await fetchJson<ModelInfoResponse>(
          `${base}/model/info?litellm_model_id=${encodeURIComponent(deploymentId)}`,
          apiKey,
          options,
        );
        detail = infoResult.ok ? infoResult.data.data?.[0] : undefined;
      }
      completed++;
      if (completed % 10 === 0 || completed === endpoints.length) {
        progress?.(`Fetched ${completed}/${endpoints.length} models...`);
      }
      return healthDeployment(detail, route, deploymentId);
    }),
  );
  // `/health` lists deployments, not route groups. Reduce every deployment for
  // a public route before selecting its transport or publishing its metadata.
  const groups = new Map<string, HealthDeployment[]>();
  for (const deployment of deployments) {
    const route = wireString(deployment?.entry.model_name);
    if (!deployment || !route) continue;
    const group = groups.get(route) ?? [];
    group.push(deployment);
    groups.set(route, group);
  }
  const ambiguousRoutes: string[] = [];
  const models = [...groups.values()]
    .map((group) => {
      let model = mapFromModelInfoGroup(
        group.map(({ entry }) => entry),
        ambiguousRoutes,
      );
      if (!model) return undefined;
      if (group.some(({ hasDetailRoute }) => !hasDetailRoute)) {
        delete model.thinkingLevelMap;
        if (model.api === "anthropic-messages") {
          model = {
            ...model,
            api: "openai-completions",
            compat: buildCompat(model.id, "openai-completions", "claude"),
          };
        }
      }
      return model;
    })
    .filter((model): model is DiscoveredModel => model !== undefined);
  reportAmbiguousCatalogAuthority(ambiguousRoutes);
  return models;
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
  const base = normalizeBaseUrl(baseUrl, options.allowInsecureHttp);
  const progress = options.silent ? undefined : options.onProgress;
  progress?.("Querying /model/info endpoint...");
  const infoResult = await fetchJson<ModelInfoResponse>(`${base}/model/info`, apiKey, options);
  if (infoResult.ok) {
    const groups = new Map<string, ModelInfoEntry[]>();
    for (const entry of infoResult.data.data ?? []) {
      // A route without a readable public name cannot be grouped or addressed.
      const route = wireString(entry.model_name);
      if (!route) continue;
      const group = groups.get(route) ?? [];
      group.push(entry);
      groups.set(route, group);
    }
    const ambiguousRoutes: string[] = [];
    let models = [...groups.values()]
      .map((group) => mapFromModelInfoGroup(group, ambiguousRoutes))
      .filter((m): m is DiscoveredModel => m !== undefined);
    reportAmbiguousCatalogAuthority(ambiguousRoutes);
    // LiteLLM's /model/info does NOT expand wildcard model_name entries (e.g.
    // "lemonade/*" backed by model: openai/* + check_provider_endpoint: true)
    // — it returns the literal wildcard only. The discovered ids live in
    // /v1/models instead. When /model/info contains any wildcard id, also query
    // /v1/models and merge the expanded (non-wildcard) entries in, dropping the
    // raw wildcard row so it doesn't surface as a phantom model choice.
    // Ref: docs.litellm.ai/docs/proxy/model_discovery
    if (models.some((m) => m.id.includes("*"))) {
      // A wildcard row is not addressable. Remove it before expansion so a failed
      // `/v1/models` request cannot leak the literal wildcard into the selector.
      models = models.filter((model) => !model.id.includes("*"));
      progress?.("/model/info has wildcard entries, expanding via /v1/models...");
      const listResult = await fetchJson<ModelsListResponse>(`${base}/v1/models`, apiKey, options);
      if (listResult.ok) {
        const expanded = (listResult.data.data ?? [])
          .map(mapFromModelsList)
          .filter((m): m is DiscoveredModel => m !== undefined && !m.id.includes("*"));
        const seen = new Set<string>(models.map((m) => m.id));
        models = [...models, ...expanded.filter((m) => !seen.has(m.id))];
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
