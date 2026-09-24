export const LITELLM_DISCOVERY_VERSION = 3 as const;

export type BackendFamily = "claude" | "deepseek" | "gemini" | "kimi" | "openai";

export interface BackendIdentityRow {
  model_name?: string;
  litellm_params?: { model?: string; custom_llm_provider?: string };
  model_info?: { base_model?: string; litellm_provider?: string };
}

export interface BackendIdentity {
  provider?: string;
  modelId: string;
  qualifiedId: string;
  family?: BackendFamily;
}

const GENERIC_ADAPTERS = new Set(["azure", "azure_ai", "custom_openai", "openai_like"]);
const GENERIC_CATALOG_PROVIDERS = new Set([...GENERIC_ADAPTERS, "openai", "text-completion-openai"]);
// Providers whose name alone settles the family. `openai` is deliberately absent: LiteLLM
// uses that provider for any OpenAI-compatible server, so it says nothing about the model.
const PROVIDER_FAMILIES: Readonly<Record<string, BackendFamily>> = {
  anthropic: "claude",
  deepseek: "deepseek",
  gemini: "gemini",
  moonshot: "kimi",
  moonshotai: "kimi",
};
// Prefixes whose keyword match is trustworthy across the whole "prefix/modelId" string because
// the prefix itself names a vendor, not just some unrelated string that happens to contain one.
const KNOWN_VENDOR_PREFIXES = new Set([...Object.keys(PROVIDER_FAMILIES), "openai"]);
// First segments that are known to be model path, never a provider: Fireworks' account-scoped
// ids ("accounts/fireworks/models/x"). An allowlist fails closed: any other differing prefix,
// recognized provider or not, stays a conflict.
const MODEL_PATH_SEGMENTS = new Set(["accounts"]);
// `o\d` (OpenAI's o1/o3/o4-mini reasoning models) is only trustworthy at the start of the id or
// right after a provider path segment — an interior "-o1-" is as likely to be an unrelated
// product's own version marker (e.g. "custom-o1-clone").
const OPENAI_FAMILY_PATTERN = /(?:^|[./_-])(?:openai|gpt|codex)(?:$|[./_:-])|(?:^|\/)o\d(?:$|[./_:-])/i;

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function wireString(value: unknown): string | undefined {
  const trimmed = typeof value === "string" ? value.trim() : undefined;
  return trimmed && trimmed !== "undefined" ? trimmed : undefined;
}

function semanticFamily(id: string): BackendFamily | undefined {
  const value = id.toLowerCase();
  if (/(?:^|[./_-])(?:anthropic|claude|opus|sonnet|haiku|fable)(?:$|[./_:-])/.test(value)) return "claude";
  if (/(?:^|[./_-])(?:moonshotai|moonshot|kimi)(?:$|[./_:-])/.test(value)) return "kimi";
  if (/(?:^|[./_-])deepseek(?:$|[./_:-])/.test(value)) return "deepseek";
  if (/(?:^|[./_-])gemini(?:$|[./_:-])/.test(value)) return "gemini";
  if (isOpenAIBackend(value)) return "openai";
  return undefined;
}

export function isOpenAIBackend(id: string): boolean {
  return OPENAI_FAMILY_PATTERN.test(id);
}

export function resolveBackendIdentity(row: BackendIdentityRow): BackendIdentity | undefined {
  const raw =
    wireString(row.model_info?.base_model) ?? wireString(row.litellm_params?.model) ?? wireString(row.model_name);
  if (!raw) return undefined;

  let slash = raw.indexOf("/");
  const prefix = slash > 0 ? raw.slice(0, slash).trim().toLowerCase() : undefined;
  let prefixProvider = prefix && !GENERIC_ADAPTERS.has(prefix) ? prefix : undefined;
  // `custom_llm_provider` is how LiteLLM routes an unprefixed model. It is provider evidence
  // in its own right, so a differing prefix is a conflict, not a tiebreak. The exception is a
  // known model-path segment: LiteLLM routes by custom_llm_provider, so Fireworks'
  // "accounts/fireworks/models/x" is not a provider named "accounts".
  const custom = wireString(row.litellm_params?.custom_llm_provider)?.toLowerCase();
  const customProvider = custom && !GENERIC_ADAPTERS.has(custom) ? custom : undefined;
  if (prefixProvider && customProvider && prefixProvider !== customProvider) {
    if (!MODEL_PATH_SEGMENTS.has(prefixProvider)) return undefined;
    prefixProvider = undefined;
    slash = -1;
  }
  const provider = prefixProvider ?? customProvider;
  const modelId = slash > 0 ? raw.slice(slash + 1) : raw;
  // Scan the full "prefix/modelId" string only when the prefix is itself a whole known vendor
  // name (e.g. "openai/production"); otherwise scan modelId alone, so an unrelated custom
  // prefix that merely contains a vendor substring (e.g. "deepseek-proxy/gpt-4-turbo") can't
  // contaminate the keyword match with its own name.
  const scanTarget = !provider || KNOWN_VENDOR_PREFIXES.has(provider) ? raw : modelId;
  const family = semanticFamily(scanTarget) ?? (provider ? PROVIDER_FAMILIES[provider] : undefined);
  return {
    ...(provider ? { provider } : {}),
    modelId,
    qualifiedId: provider ? `${provider}/${modelId}` : modelId,
    ...(family ? { family } : {}),
  };
}

export function resolveCatalogProvider(
  row: BackendIdentityRow,
  identity = resolveBackendIdentity(row),
): string | undefined {
  if (identity?.provider && !GENERIC_CATALOG_PROVIDERS.has(identity.provider)) return identity.provider;
  const reported = wireString(row.model_info?.litellm_provider)?.toLowerCase();
  const custom = wireString(row.litellm_params?.custom_llm_provider)?.toLowerCase();
  if (custom && reported && GENERIC_CATALOG_PROVIDERS.has(reported)) {
    return custom;
  }
  return identity?.provider ?? reported ?? custom;
}
