import type { Api, Model } from "@earendil-works/pi-ai";
import { intersectThinkingLevelMaps, THINKING_LEVEL_DEFINITIONS } from "./thinking-levels.js";
import type { DiscoveredModel, ModelInfoEntry } from "./types.js";

export const DEFAULT_CONTEXT_WINDOW = 128_000;
export const DEFAULT_MAX_TOKENS = 16_384;

export interface CatalogResolution {
  provider: string;
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
  api: "openai-completions" | "openai-responses";
  reasoning: boolean;
  thinkingLevelMap?: DiscoveredModel["thinkingLevelMap"];
  vision: boolean;
  contextWindow: number;
  maxTokens: number;
  cost: DiscoveredModel["cost"];
  hasCompleteCost: boolean;
  hasCompleteMetadata: boolean;
  catalogProvider?: string;
  // Set when deployments disagreed on catalog provider identity, so catalog
  // limits, pricing, and reasoning metadata were withheld for the whole group.
  catalogAuthorityAmbiguous?: boolean;
}

const RESPONSES_MODE_PATTERN = /^responses?$/i;
const CHAT_STYLE_MODE_PATTERN = /^chat$/i;
const COST_FIELDS = ["input", "output", "cacheRead", "cacheWrite"] as const;
type CostField = (typeof COST_FIELDS)[number];
type ModelCost = DiscoveredModel["cost"];
type ModelCostTier = NonNullable<ModelCost["tiers"]>[number];

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

function normalizedMode(mode: unknown): "chat" | "responses" | "unknown" | "unsupported" {
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
  for (const [level, effort, flag] of THINKING_LEVEL_DEFINITIONS) {
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
  return perToken === undefined || !Number.isFinite(perToken) || perToken < 0 ? undefined : perToken * 1_000_000;
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

function ratesAboveThreshold(cost: ModelCost, threshold: number): ModelCost {
  let matchedThreshold = -1;
  let matches: ModelCostTier[] = [];
  for (const tier of cost.tiers ?? []) {
    if (tier.inputTokensAbove < 0 || tier.inputTokensAbove > threshold) continue;
    if (tier.inputTokensAbove > matchedThreshold) {
      matchedThreshold = tier.inputTokensAbove;
      matches = [tier];
    } else if (tier.inputTokensAbove === matchedThreshold) {
      // Pi uses the first duplicate threshold. Taking the per-field maximum is
      // conservative if malformed catalog data supplies conflicting duplicates.
      matches.push(tier);
    }
  }
  if (matches.length === 0) return cost;
  return {
    input: Math.max(cost.input, ...matches.map((tier) => tier.input)),
    output: Math.max(cost.output, ...matches.map((tier) => tier.output)),
    cacheRead: Math.max(cost.cacheRead, ...matches.map((tier) => tier.cacheRead)),
    cacheWrite: Math.max(cost.cacheWrite, ...matches.map((tier) => tier.cacheWrite)),
  };
}

// Builds the per-field upper envelope of complete request-wide price ladders.
// Every source threshold is retained because crossing it can change which
// deployment is most expensive, even when the ladders use different breakpoints.
export function conservativeCostTiers(costs: readonly ModelCost[]): ModelCost["tiers"] {
  const thresholds = [
    ...new Set(
      costs.flatMap((cost) =>
        (cost.tiers ?? [])
          .map((tier) => tier.inputTokensAbove)
          .filter((threshold) => Number.isFinite(threshold) && threshold >= 0),
      ),
    ),
  ].sort((left, right) => left - right);
  if (thresholds.length === 0) return undefined;

  return thresholds.map((inputTokensAbove) => {
    const rates = costs.map((cost) => ratesAboveThreshold(cost, inputTokensAbove));
    return {
      inputTokensAbove,
      input: Math.max(...rates.map((rate) => rate.input)),
      output: Math.max(...rates.map((rate) => rate.output)),
      cacheRead: Math.max(...rates.map((rate) => rate.cacheRead)),
      cacheWrite: Math.max(...rates.map((rate) => rate.cacheWrite)),
    };
  });
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
  const catalogAuthority = catalogProvider ? catalogs : catalogs.map(() => undefined);
  const catalogAuthorityAmbiguous =
    catalogProvider === undefined && catalogs.some((catalog) => catalog?.provider !== undefined);
  const reasoningEvidence = deployments.map(
    (entry, index) => wireBoolean(entry.model_info?.supports_reasoning) ?? catalogAuthority[index]?.reasoning,
  );
  const visionEvidence = deployments.map(
    (entry, index) => wireBoolean(entry.model_info?.supports_vision) ?? catalogAuthority[index]?.vision,
  );
  const contextWindowEvidence = deployments.map(
    (entry, index) =>
      explicitLimit(entry.model_info?.max_input_tokens) ?? explicitLimit(catalogAuthority[index]?.contextWindow),
  );
  const maxTokensEvidence = deployments.map(
    (entry, index) =>
      explicitLimit(entry.model_info?.max_output_tokens) ?? explicitLimit(catalogAuthority[index]?.maxTokens),
  );
  // Explicit false is authoritative: reasoning controls cannot survive when any
  // deployment says the route does not support reasoning.
  const reasoning = reasoningEvidence.every((value) => value ?? false);
  const vision = visionEvidence.every((value) => value ?? false);
  const contextWindow = min(deployments.map((_entry, index) => contextWindowEvidence[index] ?? DEFAULT_CONTEXT_WINDOW));
  const maxTokens = min(deployments.map((_entry, index) => maxTokensEvidence[index] ?? DEFAULT_MAX_TOKENS));

  const costValues = COST_FIELDS.map((field) =>
    deployments.map((entry, index) => resolvedCost(entry, catalogAuthority[index], field)),
  );
  const completeCostFields = costValues.map((values) => values.every((value) => value !== undefined));
  const hasCompleteCost = completeCostFields.every(Boolean);
  const hasCompleteMetadata =
    hasCompleteCost &&
    reasoningEvidence.every((value) => value !== undefined) &&
    visionEvidence.every((value) => value !== undefined) &&
    contextWindowEvidence.every((value) => value !== undefined) &&
    maxTokensEvidence.every((value) => value !== undefined);
  const cost: DiscoveredModel["cost"] = {
    input: completeCostFields[0] ? Math.max(...(costValues[0] as number[])) : 0,
    output: completeCostFields[1] ? Math.max(...(costValues[1] as number[])) : 0,
    cacheRead: completeCostFields[2] ? Math.max(...(costValues[2] as number[])) : 0,
    cacheWrite: completeCostFields[3] ? Math.max(...(costValues[3] as number[])) : 0,
  };
  if (hasCompleteCost && catalogProvider) {
    const deploymentCosts = deployments.map((entry, index) => {
      const baseCost = {
        input: costValues[0][index] as number,
        output: costValues[1][index] as number,
        cacheRead: costValues[2][index] as number,
        cacheWrite: costValues[3][index] as number,
      };
      const explicitFields = new Set(COST_FIELDS.filter((field) => explicitCost(entry, field) !== undefined));
      const catalogTiers = catalogAuthority[index]?.cost?.tiers;
      // An explicit field replaces catalog pricing for that field at every
      // threshold. Unaffected fields retain their catalog ladder; otherwise a
      // partial router override could hide a known higher catalog rate.
      const tiers =
        catalogTiers && explicitFields.size < COST_FIELDS.length
          ? catalogTiers.map((tier) => ({
              inputTokensAbove: tier.inputTokensAbove,
              input: explicitFields.has("input") ? baseCost.input : tier.input,
              output: explicitFields.has("output") ? baseCost.output : tier.output,
              cacheRead: explicitFields.has("cacheRead") ? baseCost.cacheRead : tier.cacheRead,
              cacheWrite: explicitFields.has("cacheWrite") ? baseCost.cacheWrite : tier.cacheWrite,
            }))
          : undefined;
      return { ...baseCost, ...(tiers ? { tiers } : {}) };
    });
    const tiers = conservativeCostTiers(deploymentCosts);
    if (tiers) cost.tiers = tiers;
  }
  const catalogThinkingLevelMap = catalogProvider
    ? intersectThinkingLevelMaps(catalogAuthority.map((catalog) => catalog?.thinkingLevelMap))
    : undefined;
  const routerMap = routerThinkingLevelMap(deployments);
  const thinkingLevelMap =
    reasoning && (catalogThinkingLevelMap || routerMap) ? { ...catalogThinkingLevelMap, ...routerMap } : undefined;

  const id = wireString(deployments[0]?.model_name);
  if (id === undefined) return undefined;

  return {
    id,
    api: candidateModes.every((mode) => mode === "responses") ? "openai-responses" : "openai-completions",
    reasoning,
    ...(thinkingLevelMap ? { thinkingLevelMap } : {}),
    vision,
    contextWindow,
    maxTokens,
    cost,
    hasCompleteCost,
    hasCompleteMetadata,
    ...(catalogProvider ? { catalogProvider } : {}),
    ...(catalogAuthorityAmbiguous ? { catalogAuthorityAmbiguous: true } : {}),
  };
}

export function catalogResolution(provider: string, model: Model<Api>): CatalogResolution {
  return {
    provider,
    reasoning: model.reasoning,
    thinkingLevelMap: model.thinkingLevelMap,
    vision: model.input.includes("image"),
    contextWindow: model.contextWindow,
    maxTokens: model.maxTokens,
    cost: model.cost,
  };
}
