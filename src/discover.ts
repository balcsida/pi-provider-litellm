import { isIP } from "node:net";
import type { Api, Model } from "@earendil-works/pi-ai";
import { getModels, getProviders } from "@earendil-works/pi-ai/compat";
import type { BuiltinProvider } from "@earendil-works/pi-ai/providers/all";
import { LITELLM_DISCOVERY_VERSION, resolveBackendIdentity, resolveCatalogProvider } from "./backend-identity.js";
import {
  type CatalogResolution,
  catalogResolution,
  closeSerializerPolicy,
  conservativeCostTiers,
  DEFAULT_MAX_TOKENS,
  defaultContextWindow,
  hasMixedIncompatibleDeploymentModes,
  isFallbackContextWindow,
  type MessagesBackendCompat,
  meetVendorCompat,
  normalizedMode,
  reduceModelGroup,
  type SemanticFamily,
  type SemanticModel,
  wireString,
} from "./model-groups.js";
import { loadPublicCatalog, type PublicCatalog, type PublicCatalogRecord } from "./public-catalog.js";
import { intersectThinkingLevelMaps } from "./thinking-levels.js";
import type {
  DiscoveredModel,
  DiscoveredModelFor,
  DiscoveryOptions,
  DiscoveryResult,
  HealthResponse,
  LiteLLMModelPolicy,
  ModelInfoEntry,
  ModelInfoResponse,
  ModelProtocol,
  ModelsListEntry,
  ModelsListResponse,
} from "./types.js";

const DEFAULT_TIMEOUT_MS = 5000;
const HEALTH_DETAIL_CONCURRENCY = 8;
const reportedConflictingFamilyRoutes = new Set<string>();
const reportedWithheldRepairRoutes = new Set<string>();
interface HealthDeployment {
  entry: ModelInfoEntry;
  denyLevels: boolean;
  synthetic: boolean;
}
interface WildcardExpansionSource {
  model: DiscoveredModel;
  deploymentFamilies: readonly CatalogResolution["semanticFamily"][];
}
const GENERIC_TRANSPORT_ADAPTERS = new Set([
  "azure",
  "azure_ai",
  "custom_openai",
  "openai",
  "openai_like",
  "text-completion-openai",
]);
const PUBLIC_CATALOG_BUDGET_MS = 1000;
const PUBLIC_CATALOG_PROVIDER_BY_FAMILY = {
  claude: "anthropic",
  deepseek: "deepseek",
  gemini: "gemini",
  kimi: "moonshot",
  openai: "openai",
} as const;
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
const ANTHROPIC_MODEL_PATTERN = /(?:^|[-_/.:])(?:anthropic\/|(?:claude|fable|opus|sonnet|haiku)(?=$|[-_/.:]))/i;
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

function shouldSuppressReasoningContent(modelId: string): boolean {
  return isMoonshotModel(modelId) && !FORCED_THINKING_MODEL_PATTERN.test(modelId);
}

export function emitsThinkTags(modelId: string): boolean {
  return shouldSuppressReasoningContent(modelId);
}

export function responsesCompat(modelId: string): DiscoveredModelFor<"openai-responses">["compat"] {
  // Pi's Responses transport has no cacheControlFormat setting and uses
  // Responses-native prompt-cache fields instead of Anthropic cache_control markers.
  return isMoonshotModel(modelId) ? { supportsDeveloperRole: false } : undefined;
}

export function completionsCompat(
  modelId: string,
  semanticFamily?: CatalogResolution["semanticFamily"],
): DiscoveredModelFor<"openai-completions">["compat"] {
  // Conflicting deployment evidence must not re-enable route-name inference.
  if (semanticFamily === "conflicting") return { supportsStore: false };
  if (semanticFamily === "kimi" || (semanticFamily === undefined && isMoonshotModel(modelId))) {
    return {
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsStrictMode: false,
      maxTokensField: "max_tokens",
    };
  }
  if (semanticFamily === "claude" || (semanticFamily === undefined && ANTHROPIC_MODEL_PATTERN.test(modelId))) {
    return { supportsStore: false, cacheControlFormat: "anthropic" };
  }
  return { supportsStore: false };
}

function supportsResponses(entry: ModelInfoEntry): boolean {
  const endpoints = entry.model_info?.supported_endpoints;
  if (Array.isArray(endpoints)) return endpoints.some((endpoint) => endpoint === "/v1/responses");
  if (normalizedMode(entry.model_info?.mode) === "responses") return true;

  const identity = resolveBackendIdentity(entry);
  if (identity?.family !== "openai") return false;
  const adapter = wireString(entry.litellm_params?.custom_llm_provider)?.trim().toLowerCase();
  const configuredModel = wireString(entry.litellm_params?.model)?.trim().toLowerCase();
  const reportedProvider = wireString(entry.model_info?.litellm_provider)?.trim().toLowerCase();
  const azureAdapter =
    adapter === "azure" ||
    adapter === "azure_ai" ||
    /^azure(?:_ai)?\//.test(configuredModel ?? "") ||
    reportedProvider === "azure" ||
    reportedProvider === "azure_ai";
  // An Azure API version describes the API surface, not whether this deployment
  // serves Responses. Require the explicit mode/endpoint evidence above instead
  // of promoting a working Chat route to an unavailable endpoint.
  // Generic adapters remain eligible for LiteLLM's Responses-to-Chat bridge.
  return !azureAdapter;
}

export function modelProtocol(modelId: string, modeOrEntry?: string | null | ModelInfoEntry): ModelProtocol {
  const selectedEntry =
    typeof modeOrEntry === "object" && modeOrEntry !== null
      ? { ...modeOrEntry, model_name: undefined }
      : { model_name: modelId, model_info: { mode: modeOrEntry } };
  return supportsResponses(selectedEntry)
    ? { api: "openai-responses", compat: responsesCompat(modelId) }
    : { api: "openai-completions", compat: completionsCompat(modelId) };
}

export function buildCompat(modelId: string): DiscoveredModelFor<"openai-completions">["compat"] {
  return completionsCompat(modelId);
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
function catalogProviderCandidates(lookupIds: readonly string[], id: string, ownedBy?: string): BuiltinProvider[] {
  const candidates = [toKnownProvider(ownedBy), toKnownProvider(id.split("/")[0])];
  if (lookupIds.some((lookupId) => lookupId.startsWith("claude-"))) candidates.push("anthropic");
  return [...new Set(candidates.filter((provider): provider is BuiltinProvider => provider !== undefined))];
}

function resolveCatalogModel(
  id: string,
  ownedBy?: string,
): { provider: BuiltinProvider; model: Model<Api> } | undefined {
  const lookupIds = catalogLookupIds(id);
  for (const provider of catalogProviderCandidates(lookupIds, id, ownedBy)) {
    const model = findCatalogModelInProvider(provider, lookupIds);
    if (model) return { provider, model };
  }
  return undefined;
}

function findCatalogModel(id: string, ownedBy?: string): Model<Api> | undefined {
  return resolveCatalogModel(id, ownedBy)?.model;
}

export function enrichCachedModel(input: Model<Api>): Model<Api> {
  const restored = restoreCachedModelPolicy(input);
  // A model stored by a release that predates the transmissibility gate carries
  // whatever level map that release published, so the gate applies to the cached
  // map on the way in — not only to catalog metadata on the way out, which every
  // reasoning model skips via the guard below.
  const { thinkingLevelMap: cachedThinkingLevelMap, ...restoredWithoutLevels } = restored;
  const model = {
    ...restoredWithoutLevels,
    ...closeSerializerPolicy({
      api:
        restored.api === "anthropic-messages"
          ? "anthropic-messages"
          : restored.api === "openai-responses"
            ? "openai-responses"
            : "openai-completions",
      reasoning: restored.reasoning,
      vendorCompat: restored.compat,
      catalogLevels: cachedThinkingLevelMap,
      requireChatCarrier: true,
      allowInferredChatCarrier: false,
      acceptsResponsesReasoningControl: hasResponsesReasoningControl(restored),
    }),
  } as Model<Api>;
  // Reduced deployment groups use a distinct marker; this sentinel remains
  // exclusive to evidence-free fallback models that may be enriched safely. A window is
  // fallback evidence when it matches either default, so a cached entry still qualifies
  // after an operator configures LITELLM_DEFAULT_CONTEXT_WINDOW.
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
    !isFallbackContextWindow(model.contextWindow) ||
    model.maxTokens !== DEFAULT_MAX_TOKENS
  ) {
    return model;
  }
  const catalogModel = findCatalogModel(model.id);
  // The gate above proved this window is an assumption, so it is re-derived rather than
  // restored: the cache is seeded offline, where discovery never re-reads the setting.
  if (!catalogModel) return { ...model, contextWindow: defaultContextWindow() };
  // Match evidence-free discovery: only an explicit Responses catalog transport
  // changes the protocol; all other catalog APIs continue through Chat.
  const api = catalogModel.api === "openai-responses" ? "openai-responses" : "openai-completions";
  return {
    ...model,
    name: catalogModel.name,
    // The cached compat stays as stored, so catalog levels are closed against it
    // rather than trusted because the catalog offered them.
    ...closeSerializerPolicy({
      api,
      reasoning: catalogModel.reasoning,
      vendorCompat: catalogProtocol(model.id, catalogModel).compat,
      catalogLevels: catalogModel.thinkingLevelMap,
      requireChatCarrier: true,
      allowInferredChatCarrier: false,
      acceptsResponsesReasoningControl: hasResponsesReasoningControl(model),
    }),
    input: catalogModel.input,
    cost: catalogModel.cost,
    contextWindow: catalogModel.contextWindow,
    maxTokens: catalogModel.maxTokens,
    api,
  };
}

function undecoratedBackendIds(id: string): string[] {
  const normalized = id.toLowerCase();
  const unprefixed = normalized.includes("/") ? normalized.slice(normalized.indexOf("/") + 1) : normalized;
  const routed = normalized.split("/").pop() ?? normalized;
  const variants = (candidate: string): string[] => {
    const undecorated = candidate.replace(/-v\d+(?::\d+)?$/, "").replace(/@[a-z0-9-]+$/, "");
    return [candidate, candidate.replace(/:\d+$/, ""), undecorated, undecorated.replace(/-\d{8}$/, "")];
  };
  return [...new Set([...variants(unprefixed), ...variants(routed)].filter(Boolean))];
}

function catalogLookupIds(id: string): string[] {
  const normalized = id.toLowerCase();
  const lookupIds = new Set([normalized]);
  const unprefixed = normalized.includes("/") ? normalized.slice(normalized.indexOf("/") + 1) : normalized;
  lookupIds.add(unprefixed);
  for (const candidate of undecoratedBackendIds(id)) lookupIds.add(candidate);

  const anthropicAlias = unprefixed.replaceAll(".", "-");
  const match = /^(?:claude-)?(opus|sonnet|haiku)-(\d+)-(\d+)$/.exec(anthropicAlias);
  if (match) lookupIds.add(`claude-${match[1]}-${match[2]}-${match[3]}`);
  if (anthropicAlias === "fable-5" || anthropicAlias === "opus-5") lookupIds.add(`claude-${anthropicAlias}`);

  return [...lookupIds];
}

function findCatalogModelInProvider(provider: BuiltinProvider, lookupIds: string[]): Model<Api> | undefined {
  const models = getModels(provider);
  for (const lookupId of lookupIds) {
    const normalized = lookupId.toLowerCase();
    const exact = models.find((model) => model.id.toLowerCase() === normalized);
    if (exact) return exact;
    const qualified = `${provider}/${normalized}`;
    const providerQualified = models.find((model) => model.id.toLowerCase() === qualified);
    if (providerQualified) return providerQualified;
  }
  return undefined;
}

function messagesCompatOf(model: Model<Api>): MessagesBackendCompat | undefined {
  // Native Messages is safe only when Pi's Anthropic catalog supplies the
  // serializer policy for that generation. Lookup tries the exact id first, then
  // that generation's entry; provider adapter metadata is not a wire contract.
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
  const lookupIds = new Set(undecoratedBackendIds(base));
  for (const candidate of lookupIds) {
    const generation = /^(claude-[a-z]+-\d+)-\d+$/.exec(candidate)?.[1];
    if (generation) lookupIds.add(generation);
  }
  return [...lookupIds];
}

// These are the LiteLLM adapters whose native request path can terminate at a
// Claude backend. Other adapters may expose Claude-like public aliases, but an
// alias alone is not evidence that LiteLLM accepts the Anthropic Messages schema.
const CLAUDE_CAPABLE_ADAPTERS = new Set([
  "anthropic",
  "bedrock",
  "bedrock_converse",
  "vertex_ai",
  "vertex_ai-anthropic_models",
]);
const CLAUDE_MODEL_PATTERN = /(?:^|[./_-])(?:claude|opus|sonnet|haiku|fable)(?:$|[./_:-])/i;

function nativeMessagesCatalog(
  entry: ModelInfoEntry,
): Pick<CatalogResolution, "messagesCompat" | "messagesThinkingLevelMap"> {
  const adapter = wireString(entry.model_info?.litellm_provider)?.trim().toLowerCase();
  if (!adapter || !CLAUDE_CAPABLE_ADAPTERS.has(adapter) || deploymentFamily(entry) !== "claude") return {};
  const candidates = [entry.litellm_params?.model, entry.model_info?.base_model]
    .map((id) => wireString(id)?.trim())
    .filter((id): id is string => Boolean(id));
  if (candidates.length === 0 || !candidates.some((id) => CLAUDE_MODEL_PATTERN.test(id))) return {};
  const models = candidates.map((id) => findCatalogModelInProvider("anthropic", anthropicBackendLookupIds(id)));
  // Every declared backend must resolve to the same native serializer policy.
  // An unresolved identity cannot be discarded to make the remainder unanimous.
  if (models.some((model) => !model) || new Set(models.map((model) => model?.id)).size !== 1) return {};
  const model = models[0]!;
  const compat = messagesCompatOf(model);
  return compat ? { messagesCompat: compat, messagesThinkingLevelMap: model.thinkingLevelMap } : {};
}

const ADAPTER_CATALOG_PROVIDERS: Readonly<Record<string, BuiltinProvider>> = {
  anthropic: "anthropic",
  claude: "anthropic",
  azure: "azure-openai-responses",
  azure_ai: "azure-openai-responses",
  bedrock: "amazon-bedrock",
  bedrock_converse: "amazon-bedrock",
  chatgpt: "openai-codex",
  deepseek: "deepseek",
  "fireworks-ai": "fireworks",
  fireworks_ai: "fireworks",
  gemini: "google",
  kimi: "moonshotai",
  moonshot: "moonshotai",
  nvidia_nim: "nvidia",
  openai: "openai",
  together_ai: "together",
  vertex_ai: "google-vertex",
  "vertex_ai-anthropic_models": "google-vertex",
};

function adapterCatalogProvider(adapter: unknown): BuiltinProvider | undefined {
  const normalized = wireString(adapter)?.trim().toLowerCase();
  return normalized ? (ADAPTER_CATALOG_PROVIDERS[normalized] ?? toKnownProvider(normalized)) : undefined;
}

function adaptPublicCatalogRecord(
  provider: string,
  catalogModelId: string,
  record: PublicCatalogRecord | undefined,
): CatalogResolution {
  const piProvider = adapterCatalogProvider(record?.piProvider ?? record?.provider ?? provider);
  const resolved = piProvider
    ? (resolveCatalogModel(record?.modelId ?? catalogModelId, piProvider) ??
      resolveCatalogModel(catalogModelId, piProvider))
    : undefined;
  const piCatalog = resolved ? catalogResolution(resolved.provider, resolved.model) : undefined;
  const piCost = piCatalog?.cost;
  return {
    provider,
    // A models.dev hit names the model canonically even when Pi's catalog does not know it yet,
    // so two spellings of one backend model reduce to one identity instead of a conflict.
    catalogModelId: resolved?.model.id ?? record?.modelId ?? catalogModelId,
    ...(piCatalog?.reasoning !== undefined ? { reasoning: piCatalog.reasoning } : {}),
    ...(record?.effortLevels ? { reasoning: true, effortLevels: record.effortLevels } : {}),
    ...((record?.thinkingLevelMap ?? piCatalog?.thinkingLevelMap)
      ? { thinkingLevelMap: record?.thinkingLevelMap ?? piCatalog?.thinkingLevelMap }
      : {}),
    ...(record?.modalities
      ? { vision: record.modalities.includes("image") }
      : piCatalog?.vision !== undefined
        ? { vision: piCatalog.vision }
        : {}),
    ...(record?.limits?.context !== undefined
      ? { contextWindow: record.limits.context }
      : piCatalog?.contextWindow !== undefined
        ? { contextWindow: piCatalog.contextWindow }
        : {}),
    ...(record?.limits?.output !== undefined
      ? { maxTokens: record.limits.output }
      : piCatalog?.maxTokens !== undefined
        ? { maxTokens: piCatalog.maxTokens }
        : {}),
    ...(record?.cost || piCost
      ? {
          cost: {
            ...((record?.cost?.input ?? piCost?.input) !== undefined
              ? { input: record?.cost?.input ?? piCost?.input }
              : {}),
            ...((record?.cost?.output ?? piCost?.output) !== undefined
              ? { output: record?.cost?.output ?? piCost?.output }
              : {}),
            ...((record?.cost?.cacheRead ?? piCost?.cacheRead) !== undefined
              ? { cacheRead: record?.cost?.cacheRead ?? piCost?.cacheRead }
              : {}),
            ...((record?.cost?.cacheWrite ?? piCost?.cacheWrite) !== undefined
              ? { cacheWrite: record?.cost?.cacheWrite ?? piCost?.cacheWrite }
              : {}),
            ...(piCost?.tiers ? { tiers: piCost.tiers } : {}),
          },
        }
      : {}),
  };
}

export function resolveModelInfoCatalog(
  entry: ModelInfoEntry,
  publicCatalog?: PublicCatalog,
): CatalogResolution | undefined {
  const hasBackendIdentity = Boolean(
    wireString(entry.model_info?.base_model)?.trim() || wireString(entry.litellm_params?.model)?.trim(),
  );
  if (!hasBackendIdentity) return undefined;
  const identity = resolveBackendIdentity(entry);
  if (!identity) return undefined;
  const provider = resolveCatalogProvider(entry, identity);
  if (provider) {
    const record = publicCatalog?.lookup(provider, identity.modelId);
    return { ...adaptPublicCatalogRecord(provider, identity.qualifiedId, record), ...nativeMessagesCatalog(entry) };
  }
  if (!identity.family) return undefined;

  const publicProvider = PUBLIC_CATALOG_PROVIDER_BY_FAMILY[identity.family];
  const record = publicCatalog?.lookup(publicProvider, identity.modelId);
  return { ...adaptPublicCatalogRecord(publicProvider, identity.qualifiedId, record), ...nativeMessagesCatalog(entry) };
}

function awaitWithSignal<T>(promise: Promise<T>, signal: AbortSignal): Promise<T> {
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

async function awaitEnrichmentWithinBudget<T>(
  promise: Promise<T>,
  signal: AbortSignal | undefined,
  timeoutMs: number,
): Promise<T | undefined> {
  let timeout: ReturnType<typeof setTimeout> | undefined;
  const budget = new Promise<undefined>((resolve) => {
    timeout = setTimeout(() => resolve(undefined), timeoutMs);
  });
  try {
    return await Promise.race([signal ? awaitWithSignal(promise, signal) : promise, budget]);
  } finally {
    if (timeout) clearTimeout(timeout);
  }
}

function loadDiscoveryPublicCatalog(options: DiscoveryOptions): Promise<PublicCatalog | undefined> {
  const publicCatalogPromise = loadPublicCatalog({
    cachePath: options.modelsDevCachePath,
    offline: options.modelsDev === false ? true : undefined,
    timeoutMs: DEFAULT_TIMEOUT_MS,
  });
  return awaitEnrichmentWithinBudget(
    publicCatalogPromise,
    options.signal,
    Math.min(options.timeoutMs ?? DEFAULT_TIMEOUT_MS, PUBLIC_CATALOG_BUDGET_MS),
  );
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

const DIAGNOSTIC_ROUTE_SAMPLE = 3;

function reportBoundedRoutes(
  reported: Set<string>,
  routes: readonly string[],
  describe: (count: number) => string,
): void {
  const unreported = [...new Set(routes)].filter((route) => !reported.has(route));
  if (unreported.length === 0) return;
  for (const route of unreported) reported.add(route);
  const hidden = unreported.length - DIAGNOSTIC_ROUTE_SAMPLE;
  const sample = unreported.slice(0, DIAGNOSTIC_ROUTE_SAMPLE).join(", ");
  process.stderr.write(`${describe(unreported.length)}: ${sample}${hidden > 0 ? ` (+${hidden} more)` : ""}\n`);
}

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
  reportBoundedRoutes(
    reportedAmbiguousRoutes,
    routes,
    (count) =>
      `LiteLLM discovery: ${count} route group(s) have missing or conflicting deployment provider evidence; ` +
      "catalog limits, pricing, and reasoning metadata are withheld",
  );
}

const reportedIncompatibleModeRoutes = new Set<string>();

function reportIncompatibleDeploymentModes(routes: readonly string[]): void {
  reportBoundedRoutes(
    reportedIncompatibleModeRoutes,
    routes,
    (count) =>
      `LiteLLM discovery: ${count} route group(s) mix chat-style and explicitly incompatible deployment modes; ` +
      "the routes are withheld because not every deployment can accept chat requests",
  );
}

function mapFromModelInfoGroup(
  entries: readonly ModelInfoEntry[],
  publicCatalog: PublicCatalog | undefined,
  options: {
    ambiguousRoutes?: string[];
    conflictingFamilyRoutes?: string[];
    withheldRepairRoutes?: string[];
    denyLevels?: boolean;
    allowMessages?: boolean;
  } = {},
): DiscoveredModel | undefined {
  const reduced = reduceModelGroup(entries, (entry) => {
    const catalog = resolveModelInfoCatalog(entry, publicCatalog);
    const family = deploymentFamily(entry);
    const generations = [entry.litellm_params?.model, entry.model_info?.base_model]
      .map((id) => wireString(id)?.trim())
      .filter((id): id is string => Boolean(id))
      .map(semanticModel)
      .filter((model): model is SemanticModel => model !== undefined);
    const model = new Set(generations).size === 1 && family !== "conflicting" ? generations[0] : undefined;
    return {
      ...catalog,
      ...(options.allowMessages === false ? { messagesCompat: undefined } : {}),
      ...(family ? { semanticFamily: family } : {}),
      ...(model ? { semanticModel: model } : {}),
    };
  });
  if (!reduced) return undefined;
  if (reduced.catalogAuthorityAmbiguous) options.ambiguousRoutes?.push(reduced.id);
  if (reduced.deploymentFamilies.includes("conflicting")) options.conflictingFamilyRoutes?.push(reduced.id);
  const protocols = entries.map((entry) => modelProtocol(reduced.id, entry));
  const protocol = protocols.find((candidate) => candidate.api === "openai-completions") ?? protocols[0]!;
  const api = reduced.api === "anthropic-messages" ? reduced.api : protocol.api;
  const families = new Set(entries.map((entry) => resolveBackendIdentity({ ...entry, model_name: undefined })?.family));
  const [family] = families;
  const hasBackendEvidence = entries.some(hasReadableBackendEvidence);
  const reasoningPolicy =
    api !== "anthropic-messages" && reduced.semanticModel && hasBackendEvidence ? reduced.reasoningPolicy : undefined;
  const reasoning = reasoningPolicy?.reasoning ?? reduced.reasoning;
  // The semantic policy's compat only applies on Chat, so its level map must be
  // gated the same way. Vendor compatibility is reduced from deployment
  // evidence; route text is consulted only when no deployment identifies a
  // family. This gives vanity Kimi routes the complete strict-schema block while
  // withholding shape-changing fields from mixed or unidentified groups.
  const unlabeled = reduced.deploymentFamilies.every((family) => family === undefined);
  const vendorCompat = unlabeled
    ? buildCompat(reduced.id)
    : meetVendorCompat(
        reduced.deploymentFamilies.map((family) =>
          family === undefined ? undefined : completionsCompat(reduced.id, family),
        ),
      );
  const policy = closeSerializerPolicy({
    api,
    reasoning,
    vendorCompat: api === "anthropic-messages" ? reduced.messagesCompat : vendorCompat,
    semanticCompat: reduced.acceptsResponsesReasoningControl
      ? { ...reasoningPolicy?.compat, supportsReasoningEffort: true }
      : reasoningPolicy?.compat,
    semanticLevels: reasoningPolicy?.thinkingLevelMap,
    catalogLevels: reduced.thinkingLevelMap,
    requireChatCarrier: true,
    // Effort levels require a declared reasoning_effort carrier; public or LiteLLM
    // level evidence alone does not authorize adding an unsupported request field.
    allowInferredChatCarrier: false,
    acceptsResponsesReasoningControl: reduced.acceptsResponsesReasoningControl,
    denyLevels: options.denyLevels,
  });
  const hasConflictingFamily = reduced.deploymentFamilies.includes("conflicting");
  const unanimousMoonshot =
    reduced.deploymentFamilies.length > 0 && reduced.deploymentFamilies.every((family) => family === "kimi");
  const moonshotEvidence =
    !hasConflictingFamily &&
    (reduced.deploymentFamilies.includes("kimi") || (unlabeled && isMoonshotModel(reduced.id)));
  if (moonshotEvidence && !unanimousMoonshot) options.withheldRepairRoutes?.push(reduced.id);
  const modelPolicy = requestPolicy(
    reduced.id,
    reduced.deploymentFamilies,
    reduced.normalizeThinkTags,
    reduced.suppressReasoningVisibility,
    unanimousMoonshot,
  );
  return {
    id: reduced.id,
    // Reduced groups never borrow the ` (no metadata)` sentinel, which authorizes
    // catalog re-derivation from the model id during offline cache reads.
    name: reduced.hasCompleteMetadata ? reduced.id : `${reduced.id} (incomplete metadata)`,
    ...policy,
    input: reduced.vision ? ["text", "image"] : ["text"],
    cost: reduced.cost,
    contextWindow: reduced.contextWindow,
    maxTokens: reduced.maxTokens,
    api,
    litellmDiscoveryVersion: LITELLM_DISCOVERY_VERSION,
    ...(api === "openai-responses" && reduced.acceptsResponsesReasoningControl
      ? { litellmResponsesReasoningControl: true as const }
      : {}),
    ...(families.size === 1 && family ? { litellmBackendFamily: family } : {}),
    ...(modelPolicy ? { litellmPolicy: modelPolicy } : {}),
  };
}

// An evidence-free fallback entry has no deployment or adapter evidence. Its route name
// authorizes nothing; the Pi catalog entry for its id supplies its protocol and presentation
// metadata, while an unknown id stays on Chat Completions until /model/info can be read.
function catalogProtocol(modelId: string, catalogModel: Model<Api> | undefined): ModelProtocol {
  return catalogModel?.api === "openai-responses"
    ? { api: "openai-responses", compat: responsesCompat(modelId) }
    : { api: "openai-completions", compat: completionsCompat(modelId) };
}

function mapFromModelsList(entry: ModelsListEntry): DiscoveredModel | undefined {
  const id = wireString(entry.id);
  if (!id) return undefined;
  const ownedBy = wireString(entry.owned_by);
  const catalogModel = findCatalogModel(id, ownedBy);
  const api = catalogModel?.api === "openai-responses" ? "openai-responses" : "openai-completions";
  return {
    id,
    name: catalogModel?.name ?? `${id} (no metadata)`,
    ...closeSerializerPolicy({
      api,
      reasoning: catalogModel?.reasoning ?? false,
      vendorCompat: catalogProtocol(id, catalogModel).compat,
      catalogLevels: catalogModel?.thinkingLevelMap,
      requireChatCarrier: true,
      allowInferredChatCarrier: false,
    }),
    input: catalogModel?.input ?? ["text"],
    cost: catalogModel?.cost ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: catalogModel?.contextWindow ?? defaultContextWindow(),
    maxTokens: catalogModel?.maxTokens ?? DEFAULT_MAX_TOKENS,
    api,
    litellmDiscoveryVersion: LITELLM_DISCOVERY_VERSION,
    ...(isMoonshotModel(id) ? { litellmPolicy: moonshotPolicy(id) } : {}),
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
  const endpoints = (healthResult.data.healthy_endpoints ?? []).filter((entry) =>
    Boolean(wireString(entry.model)?.trim() || wireString(entry.model_id)?.trim()),
  );
  progress?.(`Discovered ${endpoints.length} model endpoints, fetching details...`);
  let completed = 0;
  let next = 0;
  const deployments: (HealthDeployment | undefined)[] = new Array(endpoints.length);
  const worker = async (): Promise<void> => {
    while (next < endpoints.length) {
      const index = next++;
      const endpoint = endpoints[index];
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
      deployments[index] = healthDeployment(detail, route, deploymentId);
    }
  };
  await Promise.all(Array.from({ length: Math.min(HEALTH_DETAIL_CONCURRENCY, endpoints.length) }, () => worker()));
  const groups = new Map<string, HealthDeployment[]>();
  for (const deployment of deployments) {
    const route = wireString(deployment?.entry.model_name);
    if (!deployment || !route) continue;
    const group = groups.get(route) ?? [];
    group.push(deployment);
    groups.set(route, group);
  }
  const publicCatalog = deployments.some((deployment) => deployment && !deployment.synthetic)
    ? await loadDiscoveryPublicCatalog(options)
    : undefined;
  const incompatibleModeRoutes: string[] = [];
  const ambiguousRoutes: string[] = [];
  const conflictingFamilyRoutes: string[] = [];
  const withheldRepairRoutes: string[] = [];
  const discovered = [...groups.values()]
    .map((group) => {
      const entries = group.map(({ entry }) => entry);
      if (hasMixedIncompatibleDeploymentModes(entries)) {
        const route = wireString(entries[0]?.model_name);
        if (route) incompatibleModeRoutes.push(route);
      }
      if (group.every((deployment) => deployment.synthetic)) return mapFromModelsList({ id: entries[0]!.model_name! });
      return mapFromModelInfoGroup(entries, publicCatalog, {
        ambiguousRoutes,
        conflictingFamilyRoutes,
        withheldRepairRoutes,
        denyLevels: group.some(({ denyLevels }) => denyLevels),
        allowMessages: false,
      });
    })
    .filter((model): model is DiscoveredModel => model !== undefined);
  reportIncompatibleDeploymentModes(incompatibleModeRoutes);
  reportAmbiguousCatalogAuthority(ambiguousRoutes);
  reportConflictingFamilyEvidence(conflictingFamilyRoutes);
  reportWithheldToolRepair(withheldRepairRoutes);
  reportWithheldToolRepairForModels(discovered);
  return discovered;
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

export function wildcardMatches(route: string, modelId: string): boolean {
  const segments = route.split("*");
  let offset = 0;
  for (const [index, segment] of segments.entries()) {
    if (segment === "") continue;
    const found = modelId.indexOf(segment, offset);
    if (found < 0 || (index === 0 && found !== 0)) return false;
    offset = found + segment.length;
  }
  const suffix = segments.at(-1);
  return suffix === "" || modelId.endsWith(suffix ?? "");
}

// LiteLLM serves a wildcard route by substituting the requested id's wildcard portion into
// the deployment's own `model` (`azure/*` + `azure/gpt-5.5` → `azure/gpt-5.5`), so the row a
// child actually runs on names that concrete backend model.
function resolveWildcardRow(row: ModelInfoEntry, modelId: string): ModelInfoEntry {
  const route = wireString(row.model_name) ?? "";
  const backend = wireString(row.litellm_params?.model);
  if (!route.includes("*") || !backend?.includes("*")) return { ...row, model_name: modelId };
  // Capture each `*` segment of the route against modelId, then substitute those captures
  // positionally into the backend's own wildcards, so a route with 2+ stars (e.g. "team/*-*")
  // resolves every star instead of only the first.
  const escaped = route.split("*").map((segment) => segment.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const match = modelId.match(new RegExp(`^${escaped.join("(.*)")}$`));
  if (!match) return { ...row, model_name: modelId };
  const captures = match.slice(1);
  let index = 0;
  return {
    ...row,
    model_name: modelId,
    litellm_params: { ...row.litellm_params, model: backend.replace(/\*/g, () => captures[index++] ?? "") },
  };
}

function wildcardPatternSpecificity(pattern: string): readonly [number, number] {
  const escaped = [...pattern]
    .map((character) => {
      if (character === "*") return "(.*)";
      return /[()[\]{}?+\-|^$\\.&~#\s]/.test(character) ? `\\${character}` : character;
    })
    .join("");
  const complexity = [...escaped].filter((character) => "*+?\\^$|()".includes(character)).length;
  return [escaped.length, complexity];
}

// A concrete id expanded from a wildcard route inherits the deployment evidence from the
// most-specific matching route, using LiteLLM's pattern order. Rows sharing that route still
// vote together, so any deployment needing Chat keeps the child on Chat.
function applyWildcardEvidence(
  model: DiscoveredModel,
  wildcardRows: readonly ModelInfoEntry[],
  publishedWildcardIds: ReadonlySet<string>,
  publicCatalog: PublicCatalog | undefined,
): DiscoveredModel | undefined {
  const matchingRows = wildcardRows.filter((row) => row.model_name && wildcardMatches(row.model_name, model.id));
  const selectedPattern = matchingRows
    .map((row) => row.model_name!)
    .sort((left, right) => {
      const [leftLength, leftComplexity] = wildcardPatternSpecificity(left);
      const [rightLength, rightComplexity] = wildcardPatternSpecificity(right);
      // Fewer wildcards (lower complexity) is more specific when escaped length ties.
      return rightLength - leftLength || leftComplexity - rightComplexity || left.localeCompare(right);
    })[0];
  if (!selectedPattern || !publishedWildcardIds.has(selectedPattern)) return undefined;
  const parents = matchingRows
    .filter((row) => row.model_name === selectedPattern)
    .map((row) => resolveWildcardRow(row, model.id));
  const selected = mapFromModelInfoGroup(parents, publicCatalog);
  if (!selected) return undefined;
  const policies = [model.litellmPolicy, selected.litellmPolicy].filter((policy) => policy !== undefined);
  const combinedPolicy =
    policies.length > 0
      ? {
          normalizeStrictToolMessages: policies.every((policy) => policy.normalizeStrictToolMessages),
          normalizeThinkTags: policies.every((policy) => policy.normalizeThinkTags),
          suppressReasoningVisibility: policies.every((policy) => policy.suppressReasoningVisibility),
          ...(policies.every((policy) => policy.normalizeGeminiReasoningEffort)
            ? { normalizeGeminiReasoningEffort: true as const }
            : {}),
        }
      : undefined;
  const {
    litellmBackendFamily: _family,
    litellmResponsesReasoningControl: _control,
    litellmPolicy: _policy,
    thinkingLevelMap: _levels,
    ...rest
  } = model;
  return {
    ...rest,
    api: selected.api,
    reasoning: selected.reasoning,
    compat: selected.compat,
    ...(selected.thinkingLevelMap ? { thinkingLevelMap: selected.thinkingLevelMap } : {}),
    ...(selected.litellmBackendFamily ? { litellmBackendFamily: selected.litellmBackendFamily } : {}),
    ...(combinedPolicy ? { litellmPolicy: combinedPolicy } : {}),
    ...(selected.litellmResponsesReasoningControl ? { litellmResponsesReasoningControl: true as const } : {}),
  };
}

function mapFromWildcardExpansion(
  entry: ModelsListEntry,
  wildcards: readonly WildcardExpansionSource[],
  withheldRepairRoutes?: string[],
): DiscoveredModel | undefined {
  const id = wireString(entry.id);
  if (!id || id.includes("*")) return undefined;
  const matchingSources = wildcards.filter(({ model }) => wildcardMatches(model.id, id));
  if (matchingSources.length === 0) return undefined;
  const matches = matchingSources.map(({ model }) => model);
  const api = matches.every((model) => model.api === "openai-responses") ? "openai-responses" : "openai-completions";
  const reasoning = matches.every((model) => model.reasoning);
  const thinkingLevelMap = reasoning
    ? intersectThinkingLevelMaps(matches.map((model) => model.thinkingLevelMap))
    : undefined;
  const hasFamilyEvidence = matchingSources.some(({ deploymentFamilies }) =>
    deploymentFamilies.some((family) => family !== undefined),
  );
  const vendorCompat = hasFamilyEvidence ? meetVendorCompat(matches.map((model) => model.compat)) : buildCompat(id);
  const modelPolicies = matches.map((model) => model.litellmPolicy);
  const hasModelPolicy = modelPolicies.some((policy) => policy !== undefined);
  const modelPolicy = hasModelPolicy
    ? {
        normalizeStrictToolMessages: modelPolicies.every((policy) => policy?.normalizeStrictToolMessages === true),
        normalizeThinkTags:
          !FORCED_THINKING_MODEL_PATTERN.test(id) &&
          modelPolicies.every((policy) => policy?.normalizeThinkTags === true),
        suppressReasoningVisibility:
          !FORCED_THINKING_MODEL_PATTERN.test(id) &&
          modelPolicies.every((policy) => policy?.suppressReasoningVisibility === true),
        ...(modelPolicies.every((policy) => policy?.normalizeGeminiReasoningEffort === true)
          ? { normalizeGeminiReasoningEffort: true as const }
          : {}),
      }
    : !hasFamilyEvidence && isMoonshotModel(id)
      ? moonshotPolicy(id)
      : undefined;
  const policy = closeSerializerPolicy({
    api,
    reasoning,
    vendorCompat,
    catalogLevels: thinkingLevelMap,
    requireChatCarrier: true,
    allowInferredChatCarrier: false,
    acceptsResponsesReasoningControl:
      api === "openai-responses" && matches.every((model) => model.litellmResponsesReasoningControl === true),
  });
  const hasKimiEvidence = matchingSources.some(({ deploymentFamilies }) => deploymentFamilies.includes("kimi"));
  const routeOnlyMoonshot = !hasFamilyEvidence && isMoonshotModel(id);
  if (modelPolicy?.normalizeStrictToolMessages === false && (hasKimiEvidence || routeOnlyMoonshot)) {
    withheldRepairRoutes?.push(id);
  }
  // A concrete /v1/models id may enrich presentation only. It never contributes
  // reasoning levels, compatibility carriers, or request policy to the child.
  const catalogModel = findCatalogModel(id, wireString(entry.owned_by));
  const parentIncomplete = matches.some((model) => model.name.endsWith(" (incomplete metadata)"));
  const base = catalogModel?.name ?? id;
  // Preserve every known tier even when a sibling has incomplete metadata. Omitting
  // a complete sibling's higher tier would understate the known worst-case rate;
  // the incomplete marker continues to signal that the resulting envelope is partial.
  const tiers = conservativeCostTiers(matches.map((model) => model.cost));
  return {
    id,
    name: parentIncomplete ? `${base} (incomplete metadata)` : base,
    ...policy,
    input: matches.every((model) => model.input.includes("image")) ? ["text", "image"] : ["text"],
    cost: {
      input: Math.max(...matches.map((model) => model.cost.input)),
      output: Math.max(...matches.map((model) => model.cost.output)),
      cacheRead: Math.max(...matches.map((model) => model.cost.cacheRead)),
      cacheWrite: Math.max(...matches.map((model) => model.cost.cacheWrite)),
      ...(tiers ? { tiers } : {}),
    },
    contextWindow: Math.min(...matches.map((model) => model.contextWindow)),
    maxTokens: Math.min(...matches.map((model) => model.maxTokens)),
    api,
    litellmDiscoveryVersion: LITELLM_DISCOVERY_VERSION,
    ...(api === "openai-responses" && matches.every((model) => model.litellmResponsesReasoningControl === true)
      ? { litellmResponsesReasoningControl: true as const }
      : {}),
    ...(modelPolicy ? { litellmPolicy: modelPolicy } : {}),
  };
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
    const publicCatalog = await loadDiscoveryPublicCatalog(options);
    const ambiguousRoutes: string[] = [];
    const incompatibleModeRoutes: string[] = [];
    const conflictingFamilyRoutes: string[] = [];
    const withheldRepairRoutes: string[] = [];
    const reducedGroups = [...groups.entries()].map(([route, group]) => {
      if (hasMixedIncompatibleDeploymentModes(group)) incompatibleModeRoutes.push(route);
      return {
        route,
        model: mapFromModelInfoGroup(group, publicCatalog, {
          ambiguousRoutes,
          conflictingFamilyRoutes,
          withheldRepairRoutes,
        }),
        deploymentFamilies: group.map(deploymentFamily),
      };
    });
    let models = reducedGroups
      .map(({ model }) => model)
      .filter((model): model is DiscoveredModel => model !== undefined);
    reportIncompatibleDeploymentModes(incompatibleModeRoutes);
    reportAmbiguousCatalogAuthority(ambiguousRoutes);
    reportConflictingFamilyEvidence(conflictingFamilyRoutes);
    reportWithheldToolRepair(withheldRepairRoutes);
    // LiteLLM's /model/info does NOT expand wildcard model_name entries (e.g.
    // "lemonade/*" backed by model: openai/* + check_provider_endpoint: true)
    // — it returns the literal wildcard only. The discovered ids live in
    // /v1/models instead. When /model/info contains any wildcard id, also query
    // /v1/models and merge the expanded (non-wildcard) entries in, dropping the
    // raw wildcard row so it doesn't surface as a phantom model choice.
    // Ref: docs.litellm.ai/docs/proxy/model_discovery
    const wildcardRoutes = reducedGroups.filter(({ route }) => route.includes("*"));
    if (wildcardRoutes.length > 0) {
      const wildcards = wildcardRoutes
        .filter((source): source is typeof source & { model: DiscoveredModel } => source.model !== undefined)
        .map(({ model, deploymentFamilies }) => ({ model, deploymentFamilies }));
      const droppedRoutes = reducedGroups.filter(({ model }) => model === undefined).map(({ route }) => route);
      const wildcardRows = wildcardRoutes.flatMap(({ route }) => groups.get(route)!);
      const publishedWildcardIds = new Set(wildcards.map(({ model }) => model.id));
      // Exact exclusions are bounded to the same public id: `/v1/models` lacks
      // deployment identity, so a differently named id for that deployment is unknowable.
      const droppedExactIds = new Set(droppedRoutes.filter((route) => !route.includes("*")));
      const droppedWildcards = droppedRoutes.filter((route) => route.includes("*"));
      // A wildcard row is not addressable. Remove it before expansion so a failed
      // `/v1/models` request cannot leak the literal wildcard into the selector.
      models = models.filter((model) => !model.id.includes("*"));
      progress?.("/model/info has wildcard entries, expanding via /v1/models...");
      const listResult = await fetchJson<ModelsListResponse>(`${base}/v1/models`, apiKey, options);
      if (listResult.ok && wildcards.length > 0) {
        const seen = new Set(models.map((model) => model.id));
        const expanded = (listResult.data.data ?? [])
          .filter((entry) => {
            const id = wireString(entry.id);
            return (
              id === undefined ||
              (!seen.has(id) &&
                !droppedExactIds.has(id) &&
                !droppedWildcards.some((route) => wildcardMatches(route, id)))
            );
          })
          .map((entry) => mapFromWildcardExpansion(entry, wildcards, withheldRepairRoutes))
          .filter((model): model is DiscoveredModel => model !== undefined)
          .map((model) => applyWildcardEvidence(model, wildcardRows, publishedWildcardIds, publicCatalog))
          .filter((model): model is DiscoveredModel => model !== undefined);
        models = [...models, ...expanded];
      }
    }
    reportWithheldToolRepair(withheldRepairRoutes);
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

export function moonshotPolicy(modelId: string, strictToolRepair = false): LiteLLMModelPolicy {
  return {
    normalizeStrictToolMessages: strictToolRepair,
    normalizeThinkTags: !FORCED_THINKING_MODEL_PATTERN.test(modelId),
    // Route-only fallback cannot authorize request-side reasoning suppression.
    suppressReasoningVisibility: false,
  };
}

function requestPolicy(
  modelId: string,
  deploymentFamilies: readonly CatalogResolution["semanticFamily"][],
  normalizeThinkTags: boolean,
  suppressReasoningVisibility: boolean,
  strictToolRepair: boolean,
): LiteLLMModelPolicy | undefined {
  // A contradiction within one deployment is evidence against applying any
  // family-specific request policy; do not discard it like a missing label.
  if (deploymentFamilies.includes("conflicting")) return undefined;
  const evidenced = deploymentFamilies.filter((family) => family !== undefined);
  if (evidenced.length === deploymentFamilies.length && evidenced.every((family) => family === "gemini")) {
    return {
      normalizeStrictToolMessages: false,
      normalizeThinkTags: false,
      suppressReasoningVisibility: false,
      normalizeGeminiReasoningEffort: true,
    };
  }
  if (evidenced.includes("kimi")) {
    return {
      normalizeStrictToolMessages: strictToolRepair,
      normalizeThinkTags,
      suppressReasoningVisibility,
    };
  }
  if (evidenced.length === 0 && isMoonshotModel(modelId)) return moonshotPolicy(modelId);
  return undefined;
}

function hasMoonshotCompatEvidence(compat: Model<Api>["compat"]): boolean {
  const openAICompat = compat as Model<"openai-completions">["compat"];
  return openAICompat?.maxTokensField === "max_tokens" && openAICompat.supportsStrictMode === false;
}

export function restoreCachedModelPolicy(model: Model<Api>): Model<Api> {
  const cached = model as Model<Api> & { litellmPolicy?: LiteLLMModelPolicy };
  if (cached.litellmPolicy || !hasMoonshotCompatEvidence(model.compat)) return model;
  const restored: typeof cached = { ...cached, litellmPolicy: moonshotPolicy(model.id) };
  return restored;
}

function hasResponsesReasoningControl(model: Model<Api>): boolean {
  return (
    (model as Model<Api> & Pick<DiscoveredModel, "litellmResponsesReasoningControl">)
      .litellmResponsesReasoningControl === true
  );
}

function semanticModel(id: string): SemanticModel | undefined {
  const value = resolveBackendIdentity({ litellm_params: { model: id } })?.modelId.toLowerCase() ?? "";
  if (/(?:^|[./_-])kimi[-_/]?k?2[._-]?[56](?:$|[./_:-])/.test(value)) return "kimi-k2.5-k2.6";
  if (/(?:^|[./_-])kimi[-_/]?k?2[._-]?7[./_-]?(?:code|highspeed)(?:$|[./_:-])/.test(value)) {
    return "kimi-k2.7-code";
  }
  if (/(?:^|[./_-])kimi[-_/]?k?3(?:$|[./_:-])/.test(value)) return "kimi-k3";
  if (/(?:^|[./_-])deepseek[-_/]?v?4(?:$|[./_:-])/.test(value)) return "deepseek-v4";
  return undefined;
}

function semanticFamily(id: string): SemanticFamily | undefined {
  return resolveBackendIdentity({ litellm_params: { model: id } })?.family;
}

function deploymentFamily(entry: ModelInfoEntry): CatalogResolution["semanticFamily"] {
  if (
    (wireString(entry.litellm_params?.model) || wireString(entry.model_info?.base_model)) &&
    !resolveBackendIdentity({ ...entry, model_name: undefined })
  )
    return "conflicting";
  const identityFamilies = [
    entry.litellm_params?.model,
    entry.litellm_params?.custom_llm_provider,
    entry.model_info?.base_model,
  ]
    .map((candidate) => wireString(candidate)?.trim())
    .filter((candidate): candidate is string => Boolean(candidate))
    .map(semanticFamily)
    .filter((family): family is SemanticFamily => family !== undefined);
  const distinctIdentityFamilies = new Set(identityFamilies);
  if (distinctIdentityFamilies.size > 1) return "conflicting";

  const identityFamily = identityFamilies[0];
  const adapter = wireString(entry.model_info?.litellm_provider)?.trim().toLowerCase();
  const adapterFamily = adapter
    ? GENERIC_TRANSPORT_ADAPTERS.has(adapter)
      ? "openai"
      : semanticFamily(adapter)
    : undefined;
  if (!identityFamily) return adapterFamily;
  // OpenAI-compatible and Azure adapters describe transport, not the backend
  // vendor. They remain useful fallback evidence for opaque identities, but must
  // not contradict a model/base identity such as `openai/kimi-k2.5`.
  if (!adapterFamily || (adapter && GENERIC_TRANSPORT_ADAPTERS.has(adapter))) return identityFamily;
  return adapterFamily === identityFamily ? identityFamily : "conflicting";
}

function reportConflictingFamilyEvidence(routes: readonly string[]): void {
  reportBoundedRoutes(
    reportedConflictingFamilyRoutes,
    routes,
    (count) =>
      `LiteLLM discovery: ${count} route group(s) have conflicting deployment family evidence; ` +
      "family-specific compatibility and request policy are withheld",
  );
}

function reportWithheldToolRepair(routes: readonly string[]): void {
  reportBoundedRoutes(
    reportedWithheldRepairRoutes,
    routes,
    (count) =>
      `LiteLLM discovery: ${count} route group(s) look Moonshot-backed but not every deployment evidences it; ` +
      "strict tool-message repair is withheld because it rewrites outbound messages and is unproven for a " +
      "deployment that has not identified its backend. Moonshot tool calls on these routes may fail until every " +
      "deployment declares its backend",
  );
}

function reportWithheldToolRepairForModels(models: readonly DiscoveredModel[]): void {
  reportWithheldToolRepair(
    models
      .filter(
        (model) =>
          model.litellmPolicy?.normalizeStrictToolMessages === false &&
          model.litellmPolicy.normalizeGeminiReasoningEffort !== true,
      )
      .map((model) => model.id),
  );
}

function hasReadableBackendEvidence(entry: ModelInfoEntry): boolean {
  return [
    entry.litellm_params?.model,
    entry.litellm_params?.custom_llm_provider,
    entry.model_info?.base_model,
    entry.model_info?.litellm_provider,
  ].some((candidate) => Boolean(wireString(candidate)?.trim()));
}

function healthDeployment(
  detail: ModelInfoEntry | undefined,
  fallbackRoute: string | undefined,
  deploymentId: string | undefined,
): HealthDeployment | undefined {
  const detailRoute = wireString(detail?.model_name)?.trim() || undefined;
  // /health exposes backend litellm_params.model values. Correlated detail owns
  // the public model_name; the health value is only a fallback when detail lacks it.
  const route = detailRoute || fallbackRoute?.trim();
  if (!route) return undefined;
  if (!detail) {
    return {
      entry: {
        model_name: route,
        model_info: {
          ...(deploymentId ? { id: deploymentId } : {}),
          mode: findCatalogModel(route)?.api === "openai-responses" ? "responses" : "chat",
        },
      },
      // The route name authorizes nothing but transport, which the Pi catalog
      // supplies for this evidence-free entry. Levels stay denied and no catalog
      // metadata is granted.
      denyLevels: true,
      synthetic: true,
    };
  }
  return {
    entry: {
      ...detail,
      model_name: route,
      model_info: {
        ...detail.model_info,
        ...(wireString(detail.model_info?.id)?.trim() ? {} : deploymentId ? { id: deploymentId } : {}),
      },
    },
    // Detail without a public route cannot authorize selectable levels.
    denyLevels: detailRoute === undefined,
    synthetic: false,
  };
}

function aggregateSuppressionEvidence(evidence: Iterable<boolean>): boolean {
  let hasEvidence = false;
  for (const suppress of evidence) {
    hasEvidence = true;
    if (!suppress) return false;
  }
  return hasEvidence;
}
