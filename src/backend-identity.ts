export const LITELLM_DISCOVERY_VERSION = 2 as const;

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
// Providers whose name alone settles the family. `openai` is deliberately absent: LiteLLM
// uses that provider for any OpenAI-compatible server, so it says nothing about the model.
const PROVIDER_FAMILIES: Readonly<Record<string, BackendFamily>> = {
  anthropic: "claude",
  deepseek: "deepseek",
  gemini: "gemini",
  moonshot: "kimi",
  moonshotai: "kimi",
};
const OPENAI_FAMILY_PATTERN = /(?:^|[./_-])(?:openai|gpt|o\d)(?:$|[./_:-])/i;

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function wireString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function semanticFamily(id: string): BackendFamily | undefined {
  const value = id.toLowerCase();
  if (/(?:^|[./_-])(?:anthropic|claude|opus|sonnet|haiku)(?:$|[./_:-])/.test(value)) return "claude";
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

  const slash = raw.indexOf("/");
  const prefix = slash > 0 ? raw.slice(0, slash).trim().toLowerCase() : undefined;
  const prefixProvider = prefix && !GENERIC_ADAPTERS.has(prefix) ? prefix : undefined;
  // `custom_llm_provider` is how LiteLLM routes an unprefixed model. It is provider evidence
  // in its own right, so a prefix that names a different provider is a conflict, not a tiebreak.
  const custom = wireString(row.litellm_params?.custom_llm_provider)?.toLowerCase();
  const customProvider = custom && !GENERIC_ADAPTERS.has(custom) ? custom : undefined;
  if (prefixProvider && customProvider && prefixProvider !== customProvider) return undefined;
  const provider = prefixProvider ?? customProvider;
  const modelId = slash > 0 ? raw.slice(slash + 1) : raw;
  const family = semanticFamily(raw) ?? (provider ? PROVIDER_FAMILIES[provider] : undefined);
  return {
    ...(provider ? { provider } : {}),
    modelId,
    qualifiedId: provider ? `${provider}/${modelId}` : modelId,
    ...(family ? { family } : {}),
  };
}
