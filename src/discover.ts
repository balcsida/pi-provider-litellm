import { isIP } from "node:net";
import type { Api, Model } from "@earendil-works/pi-ai";
import { getModels, getProviders } from "@earendil-works/pi-ai/compat";
import type { BuiltinProvider } from "@earendil-works/pi-ai/providers/all";
import {
  type CatalogResolution,
  catalogResolution,
  closeSerializerPolicy,
  DEFAULT_CONTEXT_WINDOW,
  DEFAULT_MAX_TOKENS,
  hasMixedIncompatibleDeploymentModes,
  isResponsesMode,
  meetVendorCompat,
  reduceModelGroup,
  type SemanticFamily,
  type SemanticModel,
  wireString,
} from "./model-groups.js";
import type {
  DiscoveredModel,
  DiscoveryOptions,
  DiscoveryResult,
  HealthResponse,
  LiteLLMModelPolicy,
  ModelInfoEntry,
  ModelInfoResponse,
  ModelsListEntry,
  ModelsListResponse,
} from "./types.js";

const DEFAULT_TIMEOUT_MS = 5000;
const HEALTH_DETAIL_CONCURRENCY = 8;
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

// Response-only normalization is safe with route-name evidence, but outbound
// strict-tool repair is not. Fresh discovery enables that repair only when every
// routable deployment identifies Moonshot/Kimi.
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

export function buildCompat(
  modelId: string,
  semanticFamily?: CatalogResolution["semanticFamily"],
): DiscoveredModel["compat"] {
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

// The Moonshot strict-schema compat block is the only one that pins
// `maxTokensField`, so its presence identifies a model that discovery already
// judged to be Moonshot-backed. Releases before request policies existed wrote
// that same block, which makes it usable provenance.
function hasMoonshotCompatEvidence(compat: Model<Api>["compat"]): boolean {
  const openAICompat = compat as Model<"openai-completions">["compat"];
  return openAICompat?.maxTokensField === "max_tokens" && openAICompat.supportsStrictMode === false;
}

// A cached model written before request policies existed carries none. Its
// stored compatibility fingerprint can recover the response-only conclusion,
// but it cannot prove that every deployment identified Moonshot, so the
// outbound repair stays disabled until fresh discovery persists that authority.
function restoreCachedModelPolicy(model: Model<Api>): Model<Api> {
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
      api: restored.api === "openai-responses" ? "openai-responses" : "openai-completions",
      reasoning: restored.reasoning,
      vendorCompat: restored.compat,
      catalogLevels: cachedThinkingLevelMap,
      requireChatCarrier: true,
      allowInferredChatCarrier: false,
      acceptsResponsesReasoningControl: hasResponsesReasoningControl(restored),
    }),
  } as Model<Api>;
  // Reduced deployment groups use a distinct marker; this sentinel remains
  // exclusive to evidence-free fallback models that may be enriched safely.
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
    // The cached compat stays as stored, so catalog levels are closed against it
    // rather than trusted because the catalog offered them.
    ...closeSerializerPolicy({
      api: model.api === "openai-responses" ? "openai-responses" : "openai-completions",
      reasoning: catalogModel.reasoning,
      vendorCompat: model.compat,
      catalogLevels: catalogModel.thinkingLevelMap,
      requireChatCarrier: true,
      allowInferredChatCarrier: false,
      acceptsResponsesReasoningControl: hasResponsesReasoningControl(model),
    }),
    input: catalogModel.input,
    cost: catalogModel.cost,
    contextWindow: catalogModel.contextWindow,
    maxTokens: catalogModel.maxTokens,
  };
}

// LiteLLM backend ids may carry provider paths, Bedrock/Vertex version
// decoration, and dated release snapshots. Normalize those forms before catalog
// lookup so the same deployment evidence resolves consistently across providers.
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

function semanticModel(id: string): SemanticModel | undefined {
  const value = id.toLowerCase();
  if (/(?:^|[./_-])kimi[-_/]?k?2[._-]?[56](?:$|[./_:-])/.test(value)) return "kimi-k2.5-k2.6";
  if (/(?:^|[./_-])kimi[-_/]?k?2[._-]?7[./_-]?(?:code|highspeed)(?:$|[./_:-])/.test(value)) {
    return "kimi-k2.7-code";
  }
  if (/(?:^|[./_-])kimi[-_/]?k?3(?:$|[./_:-])/.test(value)) return "kimi-k3";
  if (/(?:^|[./_-])deepseek[-_/]?v?4(?:$|[./_:-])/.test(value)) return "deepseek-v4";
  return undefined;
}

function semanticFamily(id: string): SemanticFamily | undefined {
  const value = id.toLowerCase();
  if (/(?:^|[./_-])(?:anthropic|claude|opus|sonnet|haiku)(?:$|[./_:-])/.test(value)) return "claude";
  if (/(?:^|[./_-])(?:moonshotai|moonshot|kimi)(?:$|[./_:-])/.test(value)) return "kimi";
  if (/(?:^|[./_-])deepseek(?:$|[./_:-])/.test(value)) return "deepseek";
  if (/(?:^|[./_-])gemini(?:$|[./_:-])/.test(value)) return "gemini";
  if (/(?:^|[./_-])(?:openai|gpt|o\d)(?:$|[./_:-])/.test(value)) return "openai";
  return undefined;
}

const GENERIC_TRANSPORT_ADAPTERS = new Set([
  "azure",
  "azure_ai",
  "custom_openai",
  "openai",
  "openai_like",
  "text-completion-openai",
]);

function deploymentFamily(entry: ModelInfoEntry): CatalogResolution["semanticFamily"] {
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

function normalizedAdapter(adapter: unknown): string | undefined {
  return wireString(adapter)?.trim().toLowerCase() || undefined;
}

function isGenericTransportAdapter(adapter: unknown): boolean {
  const normalized = normalizedAdapter(adapter);
  return normalized !== undefined && GENERIC_TRANSPORT_ADAPTERS.has(normalized);
}

function adapterCatalogProvider(adapter: unknown): BuiltinProvider | undefined {
  const normalized = normalizedAdapter(adapter);
  return normalized ? (ADAPTER_CATALOG_PROVIDERS[normalized] ?? toKnownProvider(normalized)) : undefined;
}

function qualifiedIdentityProvider(model: string, family: SemanticFamily | undefined): BuiltinProvider | undefined {
  const separator = model.indexOf("/");
  if (separator < 1) return undefined;
  const qualifier = model.slice(0, separator);
  const qualifierFamily = semanticFamily(qualifier);
  if (family && qualifierFamily && qualifierFamily !== family && isGenericTransportAdapter(qualifier)) return undefined;
  return adapterCatalogProvider(qualifier);
}

export function resolveModelInfoCatalog(entry: ModelInfoEntry): CatalogResolution | undefined {
  const adapterProvider = adapterCatalogProvider(entry.model_info?.litellm_provider);
  const customProvider = adapterCatalogProvider(entry.litellm_params?.custom_llm_provider);
  const routingModel = wireString(entry.litellm_params?.model)?.trim() || undefined;
  const baseModel = wireString(entry.model_info?.base_model)?.trim() || undefined;
  // Two recognized generations are contradictory: applying one deployment
  // contract to the other would send the wrong control, so withhold both.
  const routingGeneration = routingModel ? semanticModel(routingModel) : undefined;
  const baseGeneration = baseModel ? semanticModel(baseModel) : undefined;
  const contradictoryGenerations =
    routingGeneration !== undefined && baseGeneration !== undefined && routingGeneration !== baseGeneration;
  const resolvedFamily = deploymentFamily(entry);
  const identityFamily = resolvedFamily === "conflicting" ? undefined : resolvedFamily;
  // Family disagreement invalidates generation-specific policy even when only one
  // side names a recognized generation. Otherwise OpenAI/Kimi or OpenAI/DeepSeek
  // evidence can inherit controls, replay, and suppression from one side.
  const model =
    contradictoryGenerations || resolvedFamily === "conflicting" ? undefined : (routingGeneration ?? baseGeneration);
  const candidates = [routingModel, baseModel].filter((candidate): candidate is string => candidate !== undefined);
  const resolutions = candidates
    .map((candidate) => resolveCatalogModel(candidate, adapterProvider))
    .filter((resolved): resolved is NonNullable<typeof resolved> => resolved !== undefined);
  const evidenceProviders = new Set([
    ...resolutions.map((resolved) => resolved.provider),
    ...(customProvider ? [customProvider] : []),
  ]);
  const catalogIdentities = new Set(resolutions.map((resolved) => `${resolved.provider}\0${resolved.model.id}`));
  for (const candidate of candidates) {
    const provider = qualifiedIdentityProvider(candidate, semanticFamily(candidate));
    if (provider) evidenceProviders.add(provider);
  }
  if (adapterProvider && (!identityFamily || !isGenericTransportAdapter(entry.model_info?.litellm_provider))) {
    evidenceProviders.add(adapterProvider);
  }
  if (evidenceProviders.size > 1) return undefined;
  if (catalogIdentities.size > 1) {
    return resolvedFamily || model
      ? { ...(resolvedFamily ? { semanticFamily: resolvedFamily } : {}), ...(model ? { semanticModel: model } : {}) }
      : undefined;
  }
  const resolved = resolutions[0];
  if (resolved) {
    return {
      ...catalogResolution(resolved.provider, resolved.model),
      ...(resolvedFamily ? { semanticFamily: resolvedFamily } : {}),
      ...(model ? { semanticModel: model } : {}),
    };
  }
  return resolvedFamily || model
    ? { ...(resolvedFamily ? { semanticFamily: resolvedFamily } : {}), ...(model ? { semanticModel: model } : {}) }
    : undefined;
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

// Keyed by route so persistent ambiguity is reported once per process while a
// newly observed route still receives a bounded diagnostic.
function reportBoundedRoutes(
  reported: Set<string>,
  routes: readonly string[],
  describe: (count: number) => string,
): void {
  const unreported = [...new Set(routes)].filter((route) => !reported.has(route));
  if (unreported.length === 0) return;
  for (const route of unreported) reported.add(route);
  const hidden = unreported.length - AMBIGUOUS_AUTHORITY_SAMPLE;
  const sample = unreported.slice(0, AMBIGUOUS_AUTHORITY_SAMPLE).join(", ");
  process.stderr.write(`${describe(unreported.length)}: ${sample}${hidden > 0 ? ` (+${hidden} more)` : ""}\n`);
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

const reportedAmbiguousRoutes = new Set<string>();

function reportAmbiguousCatalogAuthority(routes: readonly string[]): void {
  reportBoundedRoutes(
    reportedAmbiguousRoutes,
    routes,
    (count) =>
      `LiteLLM discovery: ${count} route group(s) have missing or conflicting deployment provider evidence; ` +
      "catalog limits, pricing, and reasoning metadata are withheld",
  );
}

const reportedConflictingFamilyRoutes = new Set<string>();

function reportConflictingFamilyEvidence(routes: readonly string[]): void {
  reportBoundedRoutes(
    reportedConflictingFamilyRoutes,
    routes,
    (count) =>
      `LiteLLM discovery: ${count} route group(s) have conflicting deployment family evidence; ` +
      "family-specific compatibility and request policy are withheld",
  );
}

const reportedWithheldRepairRoutes = new Set<string>();

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

function mapFromModelInfoGroup(
  entries: readonly ModelInfoEntry[],
  options: {
    ambiguousRoutes?: string[];
    conflictingFamilyRoutes?: string[];
    withheldRepairRoutes?: string[];
    allowRouteCatalogFallback?: boolean;
    denyLevels?: boolean;
  } = {},
): DiscoveredModel | undefined {
  const reduced = reduceModelGroup(entries, (entry, singleton) => {
    const resolved = resolveModelInfoCatalog(entry);
    if (resolved) return resolved;
    const semanticFamily = deploymentFamily(entry);
    if (semanticFamily) return { semanticFamily };
    if (options.allowRouteCatalogFallback === false || !singleton || hasReadableBackendEvidence(entry))
      return undefined;
    // Preserve upstream singleton enrichment only when LiteLLM provides no
    // readable backend identity. Route text never overrides opaque evidence.
    // `reduceModelGroup` only passes rows whose route name is a readable string.
    const id = entry.model_name;
    const catalog = id ? resolveCatalogModel(id) : undefined;
    return catalog ? catalogResolution(catalog.provider, catalog.model) : undefined;
  });
  if (!reduced) return undefined;
  if (reduced.catalogAuthorityAmbiguous) options.ambiguousRoutes?.push(reduced.id);
  if (reduced.deploymentFamilies.includes("conflicting")) options.conflictingFamilyRoutes?.push(reduced.id);
  const reasoningPolicy = reduced.semanticModel ? reduced.reasoningPolicy : undefined;
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
          family === undefined ? undefined : buildCompat(reduced.id, family),
        ),
      );
  const policy = closeSerializerPolicy({
    api: reduced.api,
    reasoning,
    vendorCompat,
    semanticCompat: reasoningPolicy?.compat,
    semanticLevels: reasoningPolicy?.thinkingLevelMap,
    catalogLevels: reduced.thinkingLevelMap,
    requireChatCarrier: true,
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
    name: reduced.hasCompleteCost ? reduced.id : `${reduced.id} (incomplete metadata)`,
    ...policy,
    input: reduced.vision ? ["text", "image"] : ["text"],
    cost: reduced.cost,
    contextWindow: reduced.contextWindow,
    maxTokens: reduced.maxTokens,
    api: reduced.api,
    ...(reduced.api === "openai-responses" && reduced.acceptsResponsesReasoningControl
      ? { litellmResponsesReasoningControl: true as const }
      : {}),
    ...(modelPolicy ? { litellmPolicy: modelPolicy } : {}),
  };
}

interface HealthDeployment {
  entry: ModelInfoEntry;
  allowRouteCatalogFallback: boolean;
  denyLevels: boolean;
  endpointOnly: boolean;
}

function healthDeployment(
  detail: ModelInfoEntry | undefined,
  fallbackRoute: string | undefined,
  deploymentId: string | undefined,
): HealthDeployment | undefined {
  const detailRoute = wireString(detail?.model_name)?.trim() || undefined;
  const route = fallbackRoute?.trim() || detailRoute;
  if (!route) return undefined;
  if (!detail) {
    return {
      entry: { model_name: route, model_info: { ...(deploymentId ? { id: deploymentId } : {}), mode: "chat" } },
      allowRouteCatalogFallback: true,
      // Route text may enrich display metadata, but without deployment detail
      // there is no backend evidence authorizing a reasoning control carrier.
      denyLevels: true,
      endpointOnly: true,
    };
  }
  const mode = isResponsesMode(detail.model_info?.mode) ? "chat" : detail.model_info?.mode;
  return {
    entry: {
      ...detail,
      model_name: route,
      model_info: {
        ...detail.model_info,
        ...(wireString(detail.model_info?.id)?.trim() ? {} : deploymentId ? { id: deploymentId } : {}),
        mode,
      },
    },
    // A detail row that omits its own route did not establish request-control
    // evidence. Route catalog metadata remains available as on the prior health
    // path, but selectable levels stay explicitly closed.
    allowRouteCatalogFallback: true,
    denyLevels: detailRoute === undefined,
    endpointOnly: false,
  };
}

function mapFromModelsList(entry: ModelsListEntry): DiscoveredModel | undefined {
  const id = wireString(entry.id);
  if (!id) return undefined;
  const ownedBy = wireString(entry.owned_by);
  const catalogModel = findCatalogModel(id, ownedBy);
  return {
    id,
    name: catalogModel?.name ?? `${id} (no metadata)`,
    ...closeSerializerPolicy({
      api: "openai-completions",
      reasoning: catalogModel?.reasoning ?? false,
      vendorCompat: buildCompat(id),
      catalogLevels: catalogModel?.thinkingLevelMap,
      requireChatCarrier: true,
    }),
    input: catalogModel?.input ?? ["text"],
    cost: catalogModel?.cost ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: catalogModel?.contextWindow ?? DEFAULT_CONTEXT_WINDOW,
    maxTokens: catalogModel?.maxTokens ?? DEFAULT_MAX_TOKENS,
    api: "openai-completions",
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
      const model = mapFromModelInfoGroup(entries, {
        ambiguousRoutes,
        conflictingFamilyRoutes,
        withheldRepairRoutes,
        allowRouteCatalogFallback: group.every(({ allowRouteCatalogFallback }) => allowRouteCatalogFallback),
        denyLevels: group.some(({ denyLevels }) => denyLevels),
      });
      if (!model || !group.every(({ endpointOnly }) => endpointOnly)) return model;
      const catalogName = findCatalogModel(model.id)?.name;
      return catalogName ? { ...model, name: catalogName } : model;
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
      const route = wireString(entry.model_name);
      if (!route) continue;
      const group = groups.get(route) ?? [];
      group.push(entry);
      groups.set(route, group);
    }
    const modelInfoRoutes = new Set(groups.keys());
    const incompatibleModeRoutes: string[] = [];
    const ambiguousRoutes: string[] = [];
    const conflictingFamilyRoutes: string[] = [];
    const withheldRepairRoutes: string[] = [];
    let models = [...groups.values()]
      .map((group) => {
        if (hasMixedIncompatibleDeploymentModes(group)) {
          const route = wireString(group[0]?.model_name);
          if (route) incompatibleModeRoutes.push(route);
        }
        return mapFromModelInfoGroup(group, { ambiguousRoutes, conflictingFamilyRoutes, withheldRepairRoutes });
      })
      .filter((m): m is DiscoveredModel => m !== undefined);
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
        models = [...models, ...expanded.filter((m) => !seen.has(m.id) && !modelInfoRoutes.has(m.id))];
      }
    }
    const deduplicated = deduplicateModels(models);
    reportWithheldToolRepairForModels(deduplicated);
    return { source: "model_info", models: deduplicated };
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
  const deduplicated = deduplicateModels(models);
  reportWithheldToolRepairForModels(deduplicated);
  return { source: "models_list", models: deduplicated };
}
