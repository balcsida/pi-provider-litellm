import type { Api, Model } from "@earendil-works/pi-ai";
import type { DiscoveredModel, ModelInfoEntry } from "./types.js";

export const DEFAULT_CONTEXT_WINDOW = 128_000;
export const DEFAULT_MAX_TOKENS = 16_384;

export type SemanticFamily = "claude" | "deepseek" | "gemini" | "kimi" | "openai";

export type MessagesBackendCompat = Pick<
  NonNullable<Model<"anthropic-messages">["compat"]>,
  "forceAdaptiveThinking" | "supportsTemperature" | "supportsStrictTools"
>;

export interface CatalogResolution {
  provider?: string;
  // Concrete catalog identity; provider unanimity alone cannot authorize one
  // model's limits or pricing for a different model from the same provider.
  catalogModelId?: string;
  semanticFamily?: SemanticFamily;
  messagesCompat?: MessagesBackendCompat;
  reasoning?: boolean;
  thinkingLevelMap?: DiscoveredModel["thinkingLevelMap"];
  vision?: boolean;
  contextWindow?: number;
  maxTokens?: number;
  cost?: DiscoveredModel["cost"];
}

export type CatalogResolver = (entry: ModelInfoEntry) => CatalogResolution | undefined;

export interface ReducedModelGroup {
  id: string;
  api: "anthropic-messages" | "openai-completions" | "openai-responses";
  reasoning: boolean;
  thinkingLevelMap?: DiscoveredModel["thinkingLevelMap"];
  vision: boolean;
  contextWindow: number;
  maxTokens: number;
  cost: DiscoveredModel["cost"];
  hasCompleteMetadata: boolean;
  catalogProvider?: string;
  semanticFamily?: SemanticFamily;
  messagesCompat?: MessagesBackendCompat;
  // Set when deployments disagreed on catalog provider identity, so catalog
  // limits, pricing, and reasoning metadata were withheld for the whole group.
  catalogAuthorityAmbiguous?: boolean;
}

const RESPONSES_MODE_PATTERN = /^responses?$/i;
const CHAT_STYLE_MODE_PATTERN = /^chat$/i;
const COST_FIELDS = ["input", "output", "cacheRead", "cacheWrite"] as const;
const REASONING_EFFORT_FLAGS = [
  ["off", "none", "supports_none_reasoning_effort"],
  ["minimal", "minimal", "supports_minimal_reasoning_effort"],
  ["low", "low", "supports_low_reasoning_effort"],
  ["medium", "medium", "supports_medium_reasoning_effort"],
  ["high", "high", "supports_high_reasoning_effort"],
  ["xhigh", "xhigh", "supports_xhigh_reasoning_effort"],
  ["max", "max", "supports_max_reasoning_effort"],
] as const;
type CostField = (typeof COST_FIELDS)[number];

// `/model/info` is parsed JSON from operator-authored proxy config, so a field
// declared as a string can arrive as a number. Reading it as one must withhold
// that row's evidence, not throw and lose every model in the response.
export function wireString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

// An unreadable capability flag must not be read as `true`. `"false"` and `"no"` are
// both truthy, so coercing would relax a group guarantee — the one direction that
// matters here, since a capability is only advertised when every deployment agrees.
function wireBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

export function normalizedMode(mode: unknown): "chat" | "responses" | "unknown" | "unsupported" {
  if (mode == null) return "unknown";
  const value = wireString(mode)?.trim();
  // An unreadable mode is not evidence that the deployment is non-chat. Treating it
  // as "unsupported" would drop the row from the reduction and discard its limits,
  // relaxing the group; "unknown" keeps it routable and conservative.
  if (value === undefined) return "unknown";
  if (RESPONSES_MODE_PATTERN.test(value)) return "responses";
  if (CHAT_STYLE_MODE_PATTERN.test(value)) return "chat";
  return "unsupported";
}

// Canonicalization is depth-bounded because deployment metadata is untrusted.
const MAX_CANONICAL_DEPTH = 12;

function sortValue(value: unknown, depth = 0): unknown {
  // Do not traverse the remaining untrusted subtree at the cap. Its contents
  // are deliberately outside the bounded canonical identity.
  if (depth >= MAX_CANONICAL_DEPTH) return "[depth-limit]";
  if (Array.isArray(value)) return value.map((child) => sortValue(child, depth + 1));
  if (typeof value !== "object" || value === null) return value;
  return Object.fromEntries(
    Object.entries(value)
      .sort(([left], [right]) => left.localeCompare(right))
      .map(([key, child]) => [key, sortValue(child, depth + 1)]),
  );
}

// Key-order-independent identity, so logically equal catalog metadata compares
// equal instead of silently failing unanimity because of property order.
function stableJson(value: unknown): string | undefined {
  return value === undefined ? undefined : JSON.stringify(sortValue(value));
}

function stableEntry(entry: ModelInfoEntry): string {
  return JSON.stringify(sortValue(entry));
}

// For rows with a deployment id, collapses only exact duplicates rather than all
// rows repeating that id. Id-less rows are never collapsed, even when identical,
// because there is no deployment identity proving that they describe one target.
// Conflicting variants therefore stay plural and fail closed.
function uniqueDeployments(entries: readonly ModelInfoEntry[]): ModelInfoEntry[] {
  const identified = new Map<string, Map<string, ModelInfoEntry>>();
  const anonymous: Array<{ signature: string; entry: ModelInfoEntry }> = [];
  for (const entry of entries) {
    const signature = stableEntry(entry);
    const id = wireString(entry.model_info?.id)?.trim();
    if (id) {
      const variants = identified.get(id) ?? new Map<string, ModelInfoEntry>();
      variants.set(signature, entry);
      identified.set(id, variants);
    } else {
      anonymous.push({ signature, entry });
    }
  }
  return [
    ...[...identified.entries()]
      .sort(([left], [right]) => left.localeCompare(right))
      .flatMap(([, variants]) =>
        [...variants.entries()].sort(([left], [right]) => left.localeCompare(right)).map(([, entry]) => entry),
      ),
    ...anonymous.sort((left, right) => left.signature.localeCompare(right.signature)).map(({ entry }) => entry),
  ];
}

// A router limit is usable only when it is a finite positive token count.
// Without this, one deployment reporting 0 would clamp the whole group to 0.
function explicitLimit(value: number | undefined): number | undefined {
  return value === undefined || !Number.isFinite(value) || value <= 0 ? undefined : value;
}

function routerThinkingLevelMap(entries: readonly ModelInfoEntry[]): DiscoveredModel["thinkingLevelMap"] {
  const map: NonNullable<DiscoveredModel["thinkingLevelMap"]> = {};
  for (const [level, effort, flag] of REASONING_EFFORT_FLAGS) {
    const reported = entries.map((entry) => entry.model_info?.[flag]);
    if (!reported.some((value) => value !== undefined)) continue;
    // Preserve upstream singleton behavior (including explicit false), while a
    // group advertises a router-reported effort only when every deployment says
    // true. Missing, false, or unreadable evidence suppresses that level.
    map[level] = reported.every((value) => wireBoolean(value) === true) ? effort : null;
  }
  return Object.keys(map).length > 0 ? map : undefined;
}

function explicitCost(entry: ModelInfoEntry, field: CostField): number | undefined {
  const info = entry.model_info;
  if (!info) return undefined;
  const perToken =
    field === "input"
      ? info.input_cost_per_token
      : field === "output"
        ? info.output_cost_per_token
        : field === "cacheRead"
          ? info.cache_read_input_token_cost
          : info.cache_creation_input_token_cost;
  return perToken === undefined || !Number.isFinite(perToken) ? undefined : perToken * 1_000_000;
}

function resolvedCost(
  entry: ModelInfoEntry,
  catalog: CatalogResolution | undefined,
  field: CostField,
): number | undefined {
  return explicitCost(entry, field) ?? catalog?.cost?.[field];
}

function min(values: readonly number[]): number {
  return Math.min(...values);
}

function unanimous<T>(values: readonly (T | undefined)[]): T | undefined {
  const first = values[0];
  return first !== undefined && values.every((value) => value === first) ? first : undefined;
}

export function reduceModelGroup(
  entries: readonly ModelInfoEntry[],
  resolveCatalog: CatalogResolver,
): ReducedModelGroup | undefined {
  // A group is addressed by its public route name, so a row without a readable one
  // cannot participate. Enforced here rather than at each caller, so no caller can
  // leak a non-string id into a discovered model.
  const candidates = uniqueDeployments(entries.filter((entry) => wireString(entry.model_name)));
  if (candidates.length === 0) return undefined;
  // Transport votes over every candidate row, so a row this reduction will not
  // otherwise use — an embedding sibling, say — can still force Chat. Capability,
  // limit, price, and identity evidence reduces only over routable rows, so such a
  // row cannot corrupt them or make the group look larger than it is.
  const candidateModes = candidates.map((entry) => normalizedMode(entry.model_info?.mode));
  const deployments = candidates.filter((_, index) => candidateModes[index] !== "unsupported");
  if (deployments.length === 0) return undefined;
  const catalogs = deployments.map((entry) => resolveCatalog(entry));
  const catalogProvider = unanimous(catalogs.map((catalog) => catalog?.provider));
  const catalogModelIds = catalogs.map((catalog) => catalog?.catalogModelId);
  const hasCatalogModelIdentity = catalogModelIds.some((id) => id !== undefined);
  const catalogModelId = unanimous(catalogModelIds);
  const semanticFamily = unanimous(catalogs.map((catalog) => catalog?.semanticFamily));
  const messagesCompat = unanimous(catalogs.map((catalog) => stableJson(catalog?.messagesCompat)));
  // Provider-only resolver fixtures preserve the pre-existing reducer contract;
  // production catalog resolutions always carry a concrete model identity.
  const hasCatalogAuthority =
    catalogProvider !== undefined && (!hasCatalogModelIdentity || catalogModelId !== undefined);
  const catalogAuthority = hasCatalogAuthority ? catalogs : catalogs.map(() => undefined);
  const catalogAuthorityAmbiguous =
    !hasCatalogAuthority &&
    catalogs.some((catalog) => catalog?.provider !== undefined || catalog?.catalogModelId !== undefined);
  const reasoningValues = deployments.map(
    (entry, index) => wireBoolean(entry.model_info?.supports_reasoning) ?? catalogAuthority[index]?.reasoning,
  );
  const visionValues = deployments.map(
    (entry, index) => wireBoolean(entry.model_info?.supports_vision) ?? catalogAuthority[index]?.vision,
  );
  const contextWindowValues = deployments.map(
    (entry, index) =>
      explicitLimit(entry.model_info?.max_input_tokens) ?? explicitLimit(catalogAuthority[index]?.contextWindow),
  );
  const maxTokensValues = deployments.map(
    (entry, index) =>
      explicitLimit(entry.model_info?.max_output_tokens) ?? explicitLimit(catalogAuthority[index]?.maxTokens),
  );
  const reasoning = reasoningValues.every((value) => value ?? false);
  const vision = visionValues.every((value) => value ?? false);
  const contextWindow = min(contextWindowValues.map((value) => value ?? DEFAULT_CONTEXT_WINDOW));
  const maxTokens = min(maxTokensValues.map((value) => value ?? DEFAULT_MAX_TOKENS));

  const costValues = COST_FIELDS.map((field) =>
    deployments.map((entry, index) => resolvedCost(entry, catalogAuthority[index], field)),
  );
  const completeCostFields = costValues.map((values) => values.every((value) => value !== undefined));
  const hasCompleteCost = completeCostFields.every(Boolean);
  const hasCompleteMetadata =
    hasCompleteCost &&
    candidateModes.every((mode) => mode !== "unknown") &&
    reasoningValues.every((value) => value !== undefined) &&
    visionValues.every((value) => value !== undefined) &&
    contextWindowValues.every((value) => value !== undefined) &&
    maxTokensValues.every((value) => value !== undefined);
  const cost: DiscoveredModel["cost"] = {
    input: completeCostFields[0] ? Math.max(...(costValues[0] as number[])) : 0,
    output: completeCostFields[1] ? Math.max(...(costValues[1] as number[])) : 0,
    cacheRead: completeCostFields[2] ? Math.max(...(costValues[2] as number[])) : 0,
    cacheWrite: completeCostFields[3] ? Math.max(...(costValues[3] as number[])) : 0,
  };
  if (hasCompleteCost && catalogProvider) {
    const tiers = unanimous(catalogAuthority.map((catalog) => stableJson(catalog?.cost?.tiers)));
    if (tiers) cost.tiers = JSON.parse(tiers);
  }
  const catalogThinkingLevelMap = catalogProvider
    ? unanimous(catalogAuthority.map((catalog) => stableJson(catalog?.thinkingLevelMap)))
    : undefined;
  const parsedCatalogThinkingLevelMap = catalogThinkingLevelMap ? JSON.parse(catalogThinkingLevelMap) : undefined;
  const routerMap = routerThinkingLevelMap(deployments);

  const id = wireString(deployments[0]?.model_name);
  if (id === undefined) return undefined;

  const api = candidateModes.every((mode) => mode === "responses")
    ? "openai-responses"
    : candidateModes.every((mode) => mode === "chat") && semanticFamily === "claude" && messagesCompat
      ? "anthropic-messages"
      : "openai-completions";
  let thinkingLevelMap = parsedCatalogThinkingLevelMap;
  if (api !== "anthropic-messages") {
    thinkingLevelMap =
      parsedCatalogThinkingLevelMap || routerMap ? { ...parsedCatalogThinkingLevelMap, ...routerMap } : undefined;
  } else if (parsedCatalogThinkingLevelMap && routerMap) {
    // LiteLLM's supports_*_reasoning_effort fields describe its OpenAI-compatible
    // surface. They may restrict a catalogued Messages effort, but must not add or
    // rename Anthropic effort values that the backend catalog does not authorize.
    thinkingLevelMap = Object.fromEntries(
      Object.entries(parsedCatalogThinkingLevelMap).map(([level, effort]) => [
        level,
        routerMap[level as keyof typeof routerMap] === null ? null : effort,
      ]),
    );
  }

  return {
    id,
    api,
    reasoning,
    ...(thinkingLevelMap ? { thinkingLevelMap } : {}),
    vision,
    contextWindow,
    maxTokens,
    cost,
    hasCompleteMetadata,
    ...(hasCatalogAuthority && catalogProvider ? { catalogProvider } : {}),
    ...(semanticFamily ? { semanticFamily } : {}),
    ...(messagesCompat ? { messagesCompat: JSON.parse(messagesCompat) } : {}),
    ...(catalogAuthorityAmbiguous ? { catalogAuthorityAmbiguous: true } : {}),
  };
}

export function catalogResolution(
  provider: string,
  semanticFamily: SemanticFamily | undefined,
  model: Model<Api>,
): CatalogResolution {
  return {
    provider,
    catalogModelId: model.id,
    semanticFamily,
    reasoning: model.reasoning,
    thinkingLevelMap: model.thinkingLevelMap,
    vision: model.input.includes("image"),
    contextWindow: model.contextWindow,
    maxTokens: model.maxTokens,
    cost: model.cost,
  };
}
