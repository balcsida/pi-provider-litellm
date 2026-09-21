import type { Api, Model } from "@earendil-works/pi-ai";
import { intersectThinkingLevelMaps, THINKING_LEVEL_DEFINITIONS } from "./thinking-levels.js";
import type { DiscoveredModel, ModelInfoEntry } from "./types.js";

const BUILTIN_CONTEXT_WINDOW = 128_000;

/**
 * Context window used when neither /model/info nor the catalog reports one.
 * LITELLM_DEFAULT_CONTEXT_WINDOW raises it for proxies whose model map cannot
 * populate `max_input_tokens`; an unset or unusable value keeps the 128K default.
 */
export function defaultContextWindow(): number {
  // Number, not parseInt: parseInt takes a numeric prefix, so `1.5` would become a
  // 1-token window and `922000junk` would pass as a limit instead of being rejected.
  const parsed = Number(process.env.LITELLM_DEFAULT_CONTEXT_WINDOW ?? "");
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : BUILTIN_CONTEXT_WINDOW;
}

/**
 * Whether a stored window is one this extension assumed rather than read. A model cached
 * under the built-in default must still read as evidence-free once an operator configures
 * LITELLM_DEFAULT_CONTEXT_WINDOW, or the setting would never reach cached entries.
 * ponytail: a window stored under a setting that was later lowered is indistinguishable
 * from measured evidence and stays until the next online discovery; persist a fallback
 * marker on the model if that ever matters.
 */
export function isFallbackContextWindow(contextWindow: number): boolean {
  return contextWindow === BUILTIN_CONTEXT_WINDOW || contextWindow === defaultContextWindow();
}

export const DEFAULT_MAX_TOKENS = 16_384;

export type SemanticFamily = "claude" | "deepseek" | "gemini" | "kimi" | "openai";
export type SemanticModel = "deepseek-v4" | "kimi-k2.5-k2.6" | "kimi-k2.7-code" | "kimi-k3";

type FamilyEvidence = SemanticFamily | "conflicting";

export type MessagesBackendCompat = Pick<
  NonNullable<Model<"anthropic-messages">["compat"]>,
  "forceAdaptiveThinking" | "supportsTemperature"
>;

type OpenAICompat = NonNullable<Model<"openai-completions">["compat"]>;

export interface ReasoningPolicy {
  reasoning: boolean;
  thinkingLevelMap?: DiscoveredModel["thinkingLevelMap"];
  compat?: Pick<
    OpenAICompat,
    "requiresReasoningContentOnAssistantMessages" | "supportsReasoningEffort" | "thinkingFormat"
  >;
}

export interface CatalogResolution {
  provider?: string;
  catalogModelId?: string;
  semanticFamily?: FamilyEvidence;
  semanticModel?: SemanticModel;
  messagesCompat?: MessagesBackendCompat;
  // Evidence for the actual Messages route, kept separate from model-generation
  // serializer policy so strict-tool disagreement cannot change the transport.
  messagesStrictTools?: boolean;
  messagesThinkingLevelMap?: DiscoveredModel["thinkingLevelMap"];
  reasoning?: boolean;
  effortLevels?: string[];
  thinkingLevelMap?: DiscoveredModel["thinkingLevelMap"];
  vision?: boolean;
  contextWindow?: number;
  maxTokens?: number;
  cost?: Partial<DiscoveredModel["cost"]>;
}

export type CatalogResolver = (entry: ModelInfoEntry) => CatalogResolution | undefined;

export interface ReducedModelGroup {
  id: string;
  api: "anthropic-messages" | "openai-completions" | "openai-responses";
  reasoning: boolean;
  acceptsResponsesReasoningControl: boolean;
  thinkingLevelMap?: DiscoveredModel["thinkingLevelMap"];
  vision: boolean;
  contextWindow: number;
  contextWindowDefaulted?: boolean;
  maxTokens: number;
  cost: DiscoveredModel["cost"];
  hasCompleteCost: boolean;
  hasCompleteMetadata: boolean;
  catalogProvider?: string;
  semanticModel?: SemanticModel;
  semanticFamily?: FamilyEvidence;
  messagesCompat?: MessagesBackendCompat;
  messagesStrictTools?: boolean;
  // Set when deployments disagreed on catalog provider identity, so catalog
  // limits, pricing, and reasoning metadata were withheld for the whole group.
  catalogAuthorityAmbiguous?: boolean;
  // One entry per routable deployment. Undefined means that deployment supplied
  // no usable family evidence; callers use this to gate outbound rewrites.
  deploymentFamilies: (FamilyEvidence | undefined)[];
  normalizeThinkTags: boolean;
  suppressReasoningVisibility: boolean;
  acceptedOpenAIParams: string[];
  reasoningPolicy: ReasoningPolicy;
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

// Unreadable capability flags are withheld rather than coerced. In particular,
// string values such as `"false"` must not become truthy capability evidence.
function wireBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

export function isResponsesMode(mode: unknown): boolean {
  const value = wireString(mode);
  return value !== undefined && RESPONSES_MODE_PATTERN.test(value.trim());
}

export function normalizedMode(mode: unknown): "chat" | "responses" | "unknown" | "unsupported" {
  if (mode == null) return "unknown";
  const value = wireString(mode)?.trim();
  // An unreadable mode remains in the conservative reduction instead of being
  // filtered as unsupported and silently relaxing the remaining group.
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

// Compare catalog policies without depending on object key order.
export function stableJson(value: unknown): string | undefined {
  return value === undefined ? undefined : JSON.stringify(sortValue(value));
}

function stableEntry(entry: ModelInfoEntry): string {
  return JSON.stringify(sortValue(entry));
}

// Exact identified duplicates collapse, conflicting variants sharing an id stay
// plural, and anonymous rows stay distinct because no identity proves equality.
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
  // Conflicting rows for one deployment remain in the reduction so their
  // disagreement fails closed, while exact repeats stay idempotent.
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

const EXTENDED_LEVELS = THINKING_LEVEL_DEFINITIONS.map(([level]) => level);

// Derived from the level table rather than restated: a second hand-written copy
// silently dropped `supports_medium_reasoning_effort` and
// `supports_high_reasoning_effort`, so those router flags were never read.
const LITELLM_LEVEL_FLAGS = Object.fromEntries(THINKING_LEVEL_DEFINITIONS.map(([level, , flag]) => [level, flag])) as {
  [Definition in (typeof THINKING_LEVEL_DEFINITIONS)[number] as Definition[0]]: Definition[2];
};

function normalizeEffort(level: string): (typeof EXTENDED_LEVELS)[number] | undefined {
  const normalized = level === "none" ? "off" : level;
  return EXTENDED_LEVELS.find((candidate) => candidate === normalized);
}

// LiteLLM reads a declared `reasoning_effort_levels` list whole, ahead of the
// per-level flags, because flags cannot state a set such as low/high/max: medium
// has no opt-out. A declared list therefore answers every level for its deployment.
function reportedLevel(entry: ModelInfoEntry, level: (typeof EXTENDED_LEVELS)[number]): boolean | undefined {
  const declared: unknown = entry.model_info?.reasoning_effort_levels;
  if (Array.isArray(declared)) {
    return declared.some((effort) => typeof effort === "string" && normalizeEffort(effort) === level);
  }
  return wireBoolean(entry.model_info?.[LITELLM_LEVEL_FLAGS[level]]);
}

// A public effort list is complete: listed standard levels are enabled and omitted
// ones are denied. Without a public list, standard levels stay absent so Pi keeps
// its defaults. LiteLLM flags may add or remove levels, while xhigh/max always
// require an explicit LiteLLM opt-in.
function reasoningLevelMap(
  entries: readonly ModelInfoEntry[],
  catalogEfforts: readonly (readonly string[] | undefined)[],
  catalogMaps: readonly DiscoveredModel["thinkingLevelMap"][],
): DiscoveredModel["thinkingLevelMap"] {
  const publicSets = catalogEfforts
    .filter((levels): levels is readonly string[] => levels !== undefined)
    .map((levels) => new Set(levels.map(normalizeEffort).filter((level) => level !== undefined)))
    .filter((levels) => levels.size > 0);
  const map: NonNullable<DiscoveredModel["thinkingLevelMap"]> = {};
  if (publicSets.length > 0) {
    for (const level of ["off", "minimal", "low", "medium", "high"] as const) {
      map[level] = publicSets.every((set) => set.has(level)) ? (level === "off" ? "none" : level) : null;
    }
  }
  // A catalog level map is not a complete list: a null denies, a value supplies the
  // wire spelling, and an omitted standard level keeps Pi's default. Flattening it
  // into a list denied every standard level a map left implicit, such as Codex
  // `gpt-5.6-sol`, which only states `xhigh`, `max`, and `minimal`.
  for (const catalogMap of catalogMaps) {
    for (const [level, value] of Object.entries(catalogMap ?? {}) as [keyof typeof map, string | null][]) {
      if (value === null || (value !== undefined && map[level] !== null)) map[level] = value;
    }
  }

  for (const level of EXTENDED_LEVELS) {
    const reported = entries.map((entry) => reportedLevel(entry, level));
    if (reported.some((value) => value === false)) map[level] = null;
    else if (reported.length > 0 && reported.every((value) => value === true)) {
      map[level] = level === "off" ? "none" : level;
    }
  }
  for (const level of ["xhigh", "max"] as const) {
    if (!entries.every((entry) => reportedLevel(entry, level) === true)) map[level] = null;
  }
  return map;
}

function deniedLevels(map: DiscoveredModel["thinkingLevelMap"]): Partial<Record<string, null>> {
  return Object.fromEntries(
    Object.entries(map ?? {})
      .filter(([, level]) => level === null)
      .map(([level]) => [level, null]),
  );
}

function normalizeParams(params: unknown): Set<string> {
  if (!Array.isArray(params)) return new Set();
  const named = params.map((param) => wireString(param)?.trim()).filter((param) => Boolean(param));
  return new Set(named as string[]);
}

function acceptedParams(entry: ModelInfoEntry): Set<string> {
  const params = normalizeParams(entry.model_info?.supported_openai_params);
  for (const param of normalizeParams(entry.litellm_params?.allowed_openai_params)) params.add(param);
  return params;
}

// LiteLLM omits supported_openai_params (or returns null) for a deployment its
// model map does not describe, so that absence says nothing about the carrier. An
// explicit `supports_reasoning: true` is the operator's opt-in there, as in
// LiteLLM's own effort resolution. Present but malformed data is not an absence.
function optsIntoEffortCarrier(entry: ModelInfoEntry): boolean {
  const supported: unknown = entry.model_info?.supported_openai_params;
  return (supported === undefined || supported === null) && wireBoolean(entry.model_info?.supports_reasoning) === true;
}

function intersectParams(entries: readonly ModelInfoEntry[]): string[] {
  const [first, ...rest] = entries.map(acceptedParams);
  if (!first) return [];
  return [...first].filter((param) => rest.every((params) => params.has(param))).sort();
}

const ONLY_HIGH = {
  off: null,
  minimal: null,
  low: null,
  medium: null,
  high: "high",
  xhigh: null,
  max: null,
} as const;

// pi-ai reads an ABSENT thinkingLevelMap as "every standard level supported"
// (getSupportedThinkingLevels), so a policy that cannot transmit any level has
// to deny each one explicitly instead of omitting the map.
export const NO_TRANSMISSIBLE_LEVELS = {
  off: null,
  minimal: null,
  low: null,
  medium: null,
  high: null,
  xhigh: null,
  max: null,
} as const;

// Efforts the Responses API accepts. pi-ai passes an unmapped level through
// verbatim and reads `thinkingLevelMap.off` as the disable value, so a Chat-shaped
// map would emit `off` as an effort. `none` is the disable spelling — pi-ai's own
// `openai/gpt-5.5` entry maps `off` to it, and its `openai/gpt-5.6-*` entries map
// `max` to itself, so `max` is a real Responses effort and not a Chat-only value.
const RESPONSES_EFFORTS = new Set(["none", "minimal", "low", "medium", "high", "xhigh", "max"]);

// A level map is only meaningful next to the compat that serializes it, and the
// two used to travel separately: five call sites each decided whether to copy
// one, the other, or both. `SerializerPolicy` closes them into one value so a
// consumer cannot take a level map without the conclusion that carries it.
export interface SerializerPolicy {
  reasoning: boolean;
  thinkingLevelMap?: DiscoveredModel["thinkingLevelMap"];
  compat: DiscoveredModel["compat"];
}

// Chat carries a level through `reasoning_effort` or an explicit `thinkingFormat`.
// pi-ai would default effort on for a litellm provider, but a closed policy must
// state its own carrier rather than inherit one, so this reports only explicit
// evidence and the caller sets the carrier it concluded.
function chatCarrier(compat: OpenAICompat | undefined): boolean {
  return compat?.thinkingFormat !== undefined || compat?.supportsReasoningEffort === true;
}

function deniesChatEffort(compat: OpenAICompat | undefined): boolean {
  return compat?.supportsReasoningEffort === false && compat?.thinkingFormat === undefined;
}

// Restrictions that only remove unsupported request fields are safe to apply as
// soon as one deployment requires them. Shape-changing fields need every
// deployment to agree because an unlabeled sibling may reject the alternate wire
// form (for example, `max_tokens` instead of `max_completion_tokens`).
const COMPAT_APPLY_IF_ANY = [
  "supportsStore",
  "supportsDeveloperRole",
  "supportsReasoningEffort",
  "supportsStrictMode",
] as const;
const COMPAT_REQUIRE_UNANIMITY = [
  "maxTokensField",
  "cacheControlFormat",
  "thinkingFormat",
  "requiresReasoningContentOnAssistantMessages",
] as const;

export function meetVendorCompat(
  perDeployment: readonly (DiscoveredModel["compat"] | undefined)[],
): DiscoveredModel["compat"] {
  const candidates = perDeployment.map((compat) => compat as OpenAICompat | undefined);
  if (candidates.length === 0) return undefined;
  const met: Record<string, unknown> = {};
  for (const field of COMPAT_APPLY_IF_ANY) {
    const stated = candidates.map((compat) => compat?.[field]).filter((value) => value !== undefined);
    if (stated.length > 0 && new Set(stated).size === 1) met[field] = stated[0];
  }
  for (const field of COMPAT_REQUIRE_UNANIMITY) {
    const stated = candidates.map((compat) => compat?.[field]);
    if (stated.every((value) => value !== undefined) && new Set(stated).size === 1) met[field] = stated[0];
  }
  return (Object.keys(met).length > 0 ? met : undefined) as DiscoveredModel["compat"];
}

// Translates a Chat-shaped map into Responses efforts, denying any level with no
// valid Responses value instead of letting it through verbatim. pi-ai treats an
// absent xhigh/max entry as unsupported (unlike the standard levels), so the
// translation must preserve that distinction rather than widening the map.
export function toResponsesLevels(
  levels: DiscoveredModel["thinkingLevelMap"] | undefined,
): NonNullable<DiscoveredModel["thinkingLevelMap"]> {
  const translated: Record<string, string | null> = {};
  for (const level of EXTENDED_LEVELS) {
    const chat = levels?.[level];
    if (level === "off") {
      // Disable is `none` on this API, never `off`.
      translated.off = chat === null ? null : "none";
      continue;
    }
    if (chat === undefined && (level === "xhigh" || level === "max")) {
      translated[level] = null;
      continue;
    }
    // pi-ai passes an absent standard level through as its own name.
    const candidate = chat === undefined ? level : chat;
    translated[level] = candidate !== null && RESPONSES_EFFORTS.has(candidate) ? candidate : null;
  }
  return translated as NonNullable<DiscoveredModel["thinkingLevelMap"]>;
}

// The one place a level map is paired with a serializer conclusion. Every
// discovery, cache, health and singleton path consumes the whole returned object.
export function closeSerializerPolicy(input: {
  api: "anthropic-messages" | "openai-completions" | "openai-responses";
  reasoning: boolean;
  vendorCompat: DiscoveredModel["compat"];
  semanticCompat?: ReasoningPolicy["compat"];
  semanticLevels?: DiscoveredModel["thinkingLevelMap"];
  catalogLevels?: DiscoveredModel["thinkingLevelMap"];
  requireChatCarrier?: boolean;
  allowInferredChatCarrier?: boolean;
  acceptsResponsesReasoningControl?: boolean;
  // For callers with no level evidence at all. Distinct from "no candidate map":
  // this states the no-level conclusion explicitly, so a caller spreading the
  // policy over an earlier model cannot leave a stale map behind.
  denyLevels?: boolean;
}): SerializerPolicy {
  const {
    api,
    reasoning,
    vendorCompat,
    semanticCompat,
    semanticLevels,
    catalogLevels,
    requireChatCarrier,
    allowInferredChatCarrier = true,
    acceptsResponsesReasoningControl,
    denyLevels,
  } = input;
  if (api === "anthropic-messages") {
    const messagesCompat = vendorCompat as Model<"anthropic-messages">["compat"];
    const fields = ["forceAdaptiveThinking", "supportsTemperature", "supportsStrictTools"] as const;
    const entries = fields
      .filter((field) => messagesCompat?.[field] !== undefined)
      .map((field) => [field, messagesCompat?.[field]]);
    return {
      reasoning,
      compat: entries.length > 0 ? Object.fromEntries(entries) : undefined,
      ...(reasoning && (denyLevels || catalogLevels)
        ? { thinkingLevelMap: denyLevels ? NO_TRANSMISSIBLE_LEVELS : catalogLevels }
        : {}),
    };
  }
  if (api === "openai-responses") {
    // Responses always serializes a selected level as `reasoning.effort`.
    // Chat-only evidence such as `thinking` cannot authorize that carrier.
    const responsesCompat = vendorCompat as Model<"openai-responses">["compat"];
    const allowedFields = [
      "supportsDeveloperRole",
      "sessionAffinityFormat",
      "supportsLongCacheRetention",
      "supportsStrictMode",
      "supportsOpenAIGrammarTools",
      "supportsAdditionalTools",
      "supportsToolSearch",
      "supportsExplicitPromptCacheMode",
    ] as const satisfies readonly (keyof NonNullable<typeof responsesCompat>)[];
    const entries = allowedFields
      .filter((field) => responsesCompat?.[field] !== undefined)
      .map((field) => [field, responsesCompat?.[field]]);
    const compat = entries.length > 0 ? Object.fromEntries(entries) : undefined;
    if (!reasoning) return { reasoning, compat };
    if (denyLevels) {
      return { reasoning, thinkingLevelMap: NO_TRANSMISSIBLE_LEVELS, compat };
    }
    if (!acceptsResponsesReasoningControl) {
      return { reasoning, thinkingLevelMap: NO_TRANSMISSIBLE_LEVELS, compat };
    }
    return { reasoning, thinkingLevelMap: toResponsesLevels(semanticLevels ?? catalogLevels), compat };
  }
  // A model with no compat at all keeps none: fabricating an empty object here
  // would rewrite every cached model that passes through.
  const stated = vendorCompat !== undefined || semanticCompat !== undefined;
  const merged = { ...(vendorCompat as OpenAICompat), ...semanticCompat } as OpenAICompat;
  const compat = (stated ? merged : undefined) as DiscoveredModel["compat"];
  const deniedCompat = { ...merged, supportsReasoningEffort: false } as DiscoveredModel["compat"];
  if (!reasoning) return { reasoning, compat: denyLevels ? deniedCompat : compat };
  if (denyLevels) {
    return { reasoning, thinkingLevelMap: NO_TRANSMISSIBLE_LEVELS, compat: deniedCompat };
  }
  if (deniesChatEffort(merged)) {
    // The vendor denies effort and named no format: nothing can carry a level.
    return { reasoning, thinkingLevelMap: NO_TRANSMISSIBLE_LEVELS, compat };
  }
  const candidateLevels = semanticLevels ?? catalogLevels;
  if (candidateLevels === undefined) {
    // An absent map enables pi-ai's standard levels, so freshly discovered
    // routes without an explicit carrier must be represented as a denial.
    return requireChatCarrier
      ? { reasoning, thinkingLevelMap: NO_TRANSMISSIBLE_LEVELS, compat }
      : { reasoning, compat };
  }
  if (chatCarrier(merged)) return { reasoning, thinkingLevelMap: candidateLevels, compat };
  if (!allowInferredChatCarrier) {
    return { reasoning, thinkingLevelMap: NO_TRANSMISSIBLE_LEVELS, compat: deniedCompat };
  }
  // Callers that explicitly permit inference may treat their own direct router
  // evidence as a carrier. Route, catalog, and legacy cache evidence must opt out.
  return {
    reasoning,
    thinkingLevelMap: candidateLevels,
    compat: { ...merged, supportsReasoningEffort: true } as DiscoveredModel["compat"],
  };
}

function failClosedReasoning(
  reasoning: boolean,
  replay?: Pick<OpenAICompat, "requiresReasoningContentOnAssistantMessages">,
): ReasoningPolicy {
  return {
    reasoning,
    ...(reasoning ? { thinkingLevelMap: NO_TRANSMISSIBLE_LEVELS } : {}),
    compat: { ...replay, supportsReasoningEffort: false },
  };
}

function buildReasoningPolicy(
  semanticModel: SemanticModel | undefined,
  acceptedOpenAIParams: readonly string[],
  reducedReasoning: boolean,
  explicitlyUnsupported: boolean,
): ReasoningPolicy {
  const acceptsThinking = acceptedOpenAIParams.includes("thinking");
  const acceptsEffort = acceptedOpenAIParams.includes("reasoning_effort");
  const hasAcceptedControl = acceptsThinking || acceptsEffort;
  const reasoning =
    !explicitlyUnsupported && (reducedReasoning || hasAcceptedControl || semanticModel === "kimi-k2.7-code");
  if (semanticModel === "kimi-k2.5-k2.6") {
    // Binary thinking rides the `thinking` param, so without that accepted
    // param there is no wire mechanism and every level must be denied.
    if (acceptsThinking && reasoning) {
      return {
        reasoning,
        thinkingLevelMap: { ...ONLY_HIGH, off: "off" },
        compat: { thinkingFormat: "deepseek", supportsReasoningEffort: false },
      };
    }
    return failClosedReasoning(reasoning);
  }
  if (semanticModel === "kimi-k2.7-code") {
    // K2.7 Code always reasons and cannot be switched off, so `off` stays
    // denied rather than inventing a disable control. The `high` level only
    // exists when a deployment accepts `thinking` to carry it.
    const replay = { requiresReasoningContentOnAssistantMessages: true } as const;
    if (acceptsThinking && reasoning) {
      return {
        reasoning,
        thinkingLevelMap: ONLY_HIGH,
        compat: { ...replay, thinkingFormat: "deepseek", supportsReasoningEffort: false },
      };
    }
    return failClosedReasoning(reasoning, replay);
  }
  if (semanticModel === "kimi-k3") {
    const replay = { requiresReasoningContentOnAssistantMessages: true } as const;
    if (acceptsEffort && reasoning) {
      return {
        reasoning,
        thinkingLevelMap: {
          off: null,
          minimal: null,
          low: "low",
          medium: null,
          high: "high",
          xhigh: null,
          max: "max",
        },
        compat: { ...replay, thinkingFormat: "openai", supportsReasoningEffort: true },
      };
    }
    return failClosedReasoning(reasoning, replay);
  }
  if (semanticModel === "deepseek-v4") {
    const replay = { requiresReasoningContentOnAssistantMessages: true } as const;
    if (acceptsThinking && acceptsEffort && reasoning) {
      return {
        reasoning,
        thinkingLevelMap: {
          off: "off",
          minimal: null,
          low: null,
          medium: null,
          high: "high",
          xhigh: null,
          max: "max",
        },
        compat: { ...replay, thinkingFormat: "deepseek", supportsReasoningEffort: true },
      };
    }
    if (acceptsEffort && reasoning) {
      return {
        reasoning,
        thinkingLevelMap: {
          off: null,
          minimal: null,
          low: null,
          medium: null,
          high: "high",
          xhigh: null,
          max: "max",
        },
        compat: { ...replay, thinkingFormat: "openai", supportsReasoningEffort: true },
      };
    }
    if (acceptsThinking && reasoning) {
      return {
        reasoning,
        thinkingLevelMap: { ...ONLY_HIGH, off: "off" },
        compat: { ...replay, thinkingFormat: "deepseek", supportsReasoningEffort: false },
      };
    }
    return failClosedReasoning(reasoning, replay);
  }
  // No identified semantic model means no known reasoning contract; the caller
  // ignores this policy entirely and keeps the reduced/catalog metadata.
  return { reasoning: false };
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

function conservativeLimit(explicit: number | undefined, catalog: number | undefined): number | undefined {
  const valid = [explicitLimit(explicit), explicitLimit(catalog)].filter(
    (value): value is number => value !== undefined,
  );
  return valid.length > 0 ? min(valid) : undefined;
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

interface CanonicalModelInfoDeployments {
  candidates: ModelInfoEntry[];
  candidateModes: ReturnType<typeof normalizedMode>[];
  routable: ModelInfoEntry[];
}

function canonicalModelInfoDeployments(entries: readonly ModelInfoEntry[]): CanonicalModelInfoDeployments {
  // A group is addressed by its public route name, so a row without a readable one
  // cannot participate. Enforce that here so every authority consumer observes the
  // same filtered and deduplicated set.
  const candidates = uniqueDeployments(entries.filter((entry) => wireString(entry.model_name)));
  const candidateModes = candidates.map((entry) => normalizedMode(entry.model_info?.mode));
  return {
    candidates,
    candidateModes,
    routable: candidates.filter((_, index) => candidateModes[index] !== "unsupported"),
  };
}

export function hasMixedIncompatibleDeploymentModes(entries: readonly ModelInfoEntry[]): boolean {
  const { candidateModes } = canonicalModelInfoDeployments(entries);
  return candidateModes.includes("unsupported") && candidateModes.some((mode) => mode !== "unsupported");
}

const KIMI_FAMILY_PATTERN = /(?:^|[./_-])(?:moonshotai|moonshot|kimi)(?:$|[./_:-])/i;
const FORCED_THINKING_PATTERN = /(?:^|[./_-])thinking(?:$|[./_:-])/i;

function kimiDeploymentEvidence(entry: ModelInfoEntry): {
  identified: boolean;
  forcedThinking: boolean;
  moonshotTransport: boolean;
} {
  const identities = [
    entry.litellm_params?.model,
    entry.litellm_params?.custom_llm_provider,
    entry.model_info?.base_model,
    entry.model_info?.litellm_provider,
  ]
    .map((candidate) => wireString(candidate)?.trim())
    .filter((candidate): candidate is string => Boolean(candidate));
  const kimi = identities.filter((identity) => KIMI_FAMILY_PATTERN.test(identity));
  const routingModel = wireString(entry.litellm_params?.model)?.trim();
  const routingProviders = [
    wireString(entry.litellm_params?.custom_llm_provider)?.trim(),
    routingModel?.includes("/") ? routingModel.split("/", 1)[0] : undefined,
  ]
    .filter((provider): provider is string => Boolean(provider))
    .map((provider) => provider.toLowerCase());
  return {
    identified: kimi.length > 0,
    forcedThinking: kimi.some((identity) => FORCED_THINKING_PATTERN.test(identity)),
    // Visibility parameters are accepted by Moonshot's API, not by every host
    // that serves a Kimi model. Both declared routing signals must name Moonshot.
    moonshotTransport:
      routingProviders.length > 0 &&
      routingProviders.every((provider) => provider === "moonshot" || provider === "moonshotai"),
  };
}

export function reduceModelGroup(
  entries: readonly ModelInfoEntry[],
  resolveCatalog: CatalogResolver,
): ReducedModelGroup | undefined {
  const { candidates, candidateModes, routable: deployments } = canonicalModelInfoDeployments(entries);
  if (candidates.length === 0) return undefined;
  // Every deployment behind a public route must accept a chat-style request.
  // Dropping an explicitly incompatible sibling would publish a route that can
  // still select an embedding or other non-chat deployment.
  if (candidateModes.includes("unsupported")) return undefined;
  const catalogs = deployments.map((entry) => resolveCatalog(entry));
  const catalogProvider = unanimous(catalogs.map((catalog) => catalog?.provider));
  const catalogModelIds = catalogs.map((catalog) => catalog?.catalogModelId);
  const hasCatalogModelIdentity = catalogModelIds.some((id) => id !== undefined);
  const catalogModelId = unanimous(catalogModelIds);
  // Provider-only test resolvers preserve the reducer's public unit-test contract;
  // production catalog resolutions always include the concrete model identity.
  const hasCatalogAuthority =
    catalogProvider !== undefined && (!hasCatalogModelIdentity || catalogModelId !== undefined);
  const catalogAuthority = hasCatalogAuthority ? catalogs : catalogs.map(() => undefined);
  const catalogAuthorityAmbiguous =
    !hasCatalogAuthority &&
    catalogs.some((catalog) => catalog?.provider !== undefined || catalog?.catalogModelId !== undefined);
  const semanticModel = unanimous(catalogs.map((catalog) => catalog?.semanticModel));
  const semanticFamily = unanimous(catalogs.map((catalog) => catalog?.semanticFamily));
  const messagesCompat = unanimous(catalogs.map((catalog) => stableJson(catalog?.messagesCompat)));
  // Strict tools require affirmative evidence from every routable deployment.
  const messagesStrictTools =
    catalogs.length > 0 && catalogs.every((catalog) => catalog?.messagesStrictTools === true) ? true : undefined;
  const messagesEndpointAllowed = deployments.every((entry) => {
    const endpoints = entry.model_info?.supported_endpoints;
    return endpoints === undefined || (Array.isArray(endpoints) && endpoints.includes("/v1/messages"));
  });
  const api = candidateModes.every((mode) => mode === "responses")
    ? "openai-responses"
    : candidateModes.every((mode) => mode === "chat") &&
        messagesEndpointAllowed &&
        semanticFamily === "claude" &&
        messagesCompat
      ? "anthropic-messages"
      : "openai-completions";
  const reasoningEvidence = deployments.map(
    (entry, index) => wireBoolean(entry.model_info?.supports_reasoning) ?? catalogAuthority[index]?.reasoning,
  );
  const visionEvidence = deployments.map(
    (entry, index) => wireBoolean(entry.model_info?.supports_vision) ?? catalogAuthority[index]?.vision,
  );
  const contextWindowEvidence = deployments.map((entry, index) =>
    conservativeLimit(entry.model_info?.max_input_tokens, catalogAuthority[index]?.contextWindow),
  );
  const maxTokensEvidence = deployments.map((entry, index) =>
    conservativeLimit(entry.model_info?.max_output_tokens, catalogAuthority[index]?.maxTokens),
  );
  const reasoning = reasoningEvidence.every((value) => value ?? false);
  const explicitlyUnsupported = deployments.some(
    (entry) => wireBoolean(entry.model_info?.supports_reasoning) === false,
  );
  const vision = visionEvidence.every((value) => value ?? false);
  const fallbackContextWindow = defaultContextWindow();
  const contextWindow = min(contextWindowEvidence.map((value) => value ?? fallbackContextWindow));
  const maxTokens = min(maxTokensEvidence.map((value) => value ?? DEFAULT_MAX_TOKENS));

  const costValues = COST_FIELDS.map((field) =>
    deployments.map((entry, index) => resolvedCost(entry, catalogAuthority[index], field)),
  );
  const completeCostFields = costValues.map((values) => values.every((value) => value !== undefined));
  const hasCompleteCost = completeCostFields.every(Boolean);
  // A defaulted capability or limit is as much a gap as a missing price: the
  // incomplete marker covers everything the reducer had to assume.
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
  if (hasCompleteCost && hasCatalogAuthority) {
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
  const acceptedOpenAIParams = intersectParams(deployments);
  // Kimi and DeepSeek generations name their own carriers, so an operator opt-in
  // cannot substitute for the parameter evidence their contracts require.
  const namedCarrierFamily = catalogs.some((catalog) =>
    ["kimi", "deepseek", "conflicting"].includes(catalog?.semanticFamily ?? ""),
  );
  const acceptsResponsesReasoningControl =
    acceptedOpenAIParams.includes("reasoning_effort") ||
    (!namedCarrierFamily &&
      deployments.length > 0 &&
      deployments.every((entry) => acceptedParams(entry).has("reasoning_effort") || optsIntoEffortCarrier(entry)));
  const publicEfforts = catalogAuthority.map((catalog) => catalog?.effortLevels);
  const evidenceLevelMap = reasoningLevelMap(deployments, publicEfforts, [
    intersectThinkingLevelMaps(catalogAuthority.map((catalog) => catalog?.thinkingLevelMap)),
  ]);
  let thinkingLevelMap = reasoning && acceptsResponsesReasoningControl ? evidenceLevelMap : undefined;
  if (api === "anthropic-messages") {
    thinkingLevelMap = intersectThinkingLevelMaps(catalogs.map((catalog) => catalog?.messagesThinkingLevelMap));
    // Native serializer restrictions apply even when catalog pricing is withheld.
    // Router flags can also deny Pi's implicit default levels, never add a level.
    for (const level of EXTENDED_LEVELS) {
      const reported = deployments.map((entry) => reportedLevel(entry, level));
      if (
        reported.some((value) => value === false) ||
        ((level === "xhigh" || level === "max") &&
          reported.some((value) => value !== undefined) &&
          !reported.every((value) => value === true))
      ) {
        thinkingLevelMap ??= {};
        thinkingLevelMap[level] = null;
      }
    }
  }
  const kimiEvidence = deployments.map((entry) => kimiDeploymentEvidence(entry));
  const unanimousNormalKimi = kimiEvidence.every((evidence) => evidence.identified && !evidence.forcedThinking);
  const unanimousMoonshotTransport = kimiEvidence.every((evidence) => evidence.moonshotTransport);

  const id = wireString(deployments[0]?.model_name);
  if (id === undefined) return undefined;

  const semanticReasoningPolicy = buildReasoningPolicy(
    semanticModel,
    acceptedOpenAIParams,
    reasoning,
    explicitlyUnsupported,
  );
  // A semantic policy that denies the effort carrier describes a generation with
  // binary or fixed thinking. When no public catalog states effort levels, that
  // contract stays in force even though a host accepts `reasoning_effort`;
  // otherwise Pi's defaults would turn a single on/off switch into five
  // selectable levels. Positive public level evidence still governs, as it does
  // for level-based generations whose accepted levels vary by host: Moonshot
  // rejects `medium` on K3 while Azure Foundry accepts it.
  const semanticMap = semanticReasoningPolicy.thinkingLevelMap;
  const hasEvidenceLevels = Object.values(evidenceLevelMap ?? {}).some((level) => level !== null);
  const semanticBinary = semanticReasoningPolicy.compat?.supportsReasoningEffort === false && !hasEvidenceLevels;
  const explicitOffDenial = deployments.some((entry) => reportedLevel(entry, "off") === false);
  const reasoningPolicy = semanticMap
    ? {
        ...semanticReasoningPolicy,
        thinkingLevelMap: !acceptsResponsesReasoningControl
          ? semanticMap
          : semanticBinary
            ? { ...semanticMap, ...deniedLevels(evidenceLevelMap) }
            : {
                ...evidenceLevelMap,
                ...(semanticMap.off !== undefined ? { off: explicitOffDenial ? null : semanticMap.off } : {}),
              },
      }
    : semanticReasoningPolicy;

  return {
    id,
    api,
    reasoning,
    acceptsResponsesReasoningControl,
    ...(thinkingLevelMap ? { thinkingLevelMap } : {}),
    vision,
    contextWindow,
    ...(contextWindowEvidence.includes(undefined) && contextWindow === fallbackContextWindow
      ? { contextWindowDefaulted: true }
      : {}),
    maxTokens,
    cost,
    hasCompleteCost,
    hasCompleteMetadata,
    ...(hasCatalogAuthority ? { catalogProvider } : {}),
    ...(semanticModel ? { semanticModel } : {}),
    ...(semanticFamily ? { semanticFamily } : {}),
    ...(messagesCompat ? { messagesCompat: JSON.parse(messagesCompat) } : {}),
    ...(messagesStrictTools !== undefined ? { messagesStrictTools } : {}),
    ...(catalogAuthorityAmbiguous ? { catalogAuthorityAmbiguous: true } : {}),
    deploymentFamilies: catalogs.map((catalog) => catalog?.semanticFamily),
    normalizeThinkTags: unanimousNormalKimi,
    suppressReasoningVisibility: unanimousNormalKimi && unanimousMoonshotTransport,
    acceptedOpenAIParams,
    reasoningPolicy,
  };
}

export function catalogResolution(provider: string, model: Model<Api>): CatalogResolution {
  return {
    provider,
    catalogModelId: model.id,
    reasoning: model.reasoning,
    thinkingLevelMap: model.thinkingLevelMap,
    vision: model.input.includes("image"),
    contextWindow: model.contextWindow,
    maxTokens: model.maxTokens,
    cost: model.cost,
  };
}
