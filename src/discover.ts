import type { ProviderModelConfig } from "@earendil-works/pi-coding-agent";
import type {
  DiscoveryOptions,
  DiscoveryResult,
  ModelInfoEntry,
  ModelInfoResponse,
  ModelsListEntry,
  ModelsListResponse,
} from "./types.js";

const DEFAULT_TIMEOUT_MS = 5000;
const DEFAULT_CONTEXT_WINDOW = 128_000;
const DEFAULT_MAX_TOKENS = 16_384;

export function normalizeBaseUrl(input: string): string {
  return input.replace(/\/+$/, "").replace(/\/v1\/?$/i, "");
}

// Matches both the conventional `anthropic/...` prefix and bare aliases that
// LiteLLM deployments commonly assign to Anthropic-backed routes (e.g.
// `opus-4.7`, `sonnet-4.6`, `haiku-4.5`, `claude-3-5-sonnet`). Without the
// `cacheControlFormat: "anthropic"` flag, pi never relays cache_control markers
// through the proxy, so prompt caching silently no-ops on Claude models.
const ANTHROPIC_MODEL_PATTERN = /^(anthropic\/|claude|opus|sonnet|haiku)/i;

export function buildCompat(modelId: string): ProviderModelConfig["compat"] {
  if (ANTHROPIC_MODEL_PATTERN.test(modelId)) {
    return { supportsStore: false, cacheControlFormat: "anthropic" };
  }
  return { supportsStore: false };
}

function withTimeout(timeoutMs: number, signal?: AbortSignal): { signal: AbortSignal; cancel: () => void } {
  const controller = new AbortController();
  const onAbort = () => controller.abort(signal?.reason);
  if (signal) {
    if (signal.aborted) controller.abort(signal.reason);
    else signal.addEventListener("abort", onAbort, { once: true });
  }
  const timer = setTimeout(() => controller.abort(new Error(`Timed out after ${timeoutMs}ms`)), timeoutMs);
  return {
    signal: controller.signal,
    cancel: () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    },
  };
}

async function fetchJson<T>(
  url: string,
  apiKey: string,
  options: DiscoveryOptions,
): Promise<{ ok: true; data: T } | { ok: false; status: number }> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const { signal, cancel } = withTimeout(timeoutMs, options.signal);
  try {
    const response = await fetch(url, {
      headers: { Authorization: `Bearer ${apiKey}`, Accept: "application/json" },
      signal,
    });
    if (!response.ok) return { ok: false, status: response.status };
    const data = (await response.json()) as T;
    return { ok: true, data };
  } finally {
    cancel();
  }
}

function mapFromModelInfo(entry: ModelInfoEntry): ProviderModelConfig | undefined {
  const id = entry.model_name;
  if (!id) return undefined;
  const info = entry.model_info ?? {};
  if (info.mode && info.mode !== "chat") return undefined;
  return {
    id,
    name: id,
    reasoning: info.supports_reasoning ?? false,
    input: info.supports_vision ? ["text", "image"] : ["text"],
    cost: {
      input: (info.input_cost_per_token ?? 0) * 1_000_000,
      output: (info.output_cost_per_token ?? 0) * 1_000_000,
      cacheRead: (info.cache_read_input_token_cost ?? 0) * 1_000_000,
      cacheWrite: (info.cache_creation_input_token_cost ?? 0) * 1_000_000,
    },
    contextWindow: info.max_input_tokens ?? DEFAULT_CONTEXT_WINDOW,
    maxTokens: info.max_output_tokens ?? DEFAULT_MAX_TOKENS,
    compat: buildCompat(id),
  };
}

function mapFromModelsList(entry: ModelsListEntry): ProviderModelConfig | undefined {
  const id = entry.id;
  if (!id) return undefined;
  return {
    id,
    name: `${id} (no metadata)`,
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: DEFAULT_CONTEXT_WINDOW,
    maxTokens: DEFAULT_MAX_TOKENS,
    compat: buildCompat(id),
  };
}

export async function discoverModels(
  baseUrl: string,
  apiKey: string,
  options: DiscoveryOptions = {},
): Promise<DiscoveryResult> {
  const base = normalizeBaseUrl(baseUrl);
  const infoResult = await fetchJson<ModelInfoResponse>(`${base}/model/info`, apiKey, options);
  if (infoResult.ok) {
    const models = (infoResult.data.data ?? [])
      .map(mapFromModelInfo)
      .filter((m): m is ProviderModelConfig => m !== undefined);
    return { source: "model_info", models };
  }
  if (![401, 403, 404].includes(infoResult.status)) {
    throw new Error(`/model/info returned ${infoResult.status}`);
  }
  const listResult = await fetchJson<ModelsListResponse>(`${base}/v1/models`, apiKey, options);
  if (!listResult.ok) {
    throw new Error(`/v1/models returned ${listResult.status}`);
  }
  const models = (listResult.data.data ?? [])
    .map(mapFromModelsList)
    .filter((m): m is ProviderModelConfig => m !== undefined);
  return { source: "models_list", models };
}
