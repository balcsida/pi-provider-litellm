import { discoverModels, normalizeBaseUrl } from "../src/discover.js";
import type { DiscoverySource, LiteLLMApi } from "../src/types.js";

const DEFAULT_TIMEOUT_MS = 30_000;
const SMOKE_PROMPT = "Reply with one short word.";

export type SmokeCompletion = {
  modelId: string;
  api: LiteLLMApi;
  endpoint: string;
  content: string;
  hasResponseCost: boolean;
};

export type SmokeResult = {
  source: DiscoverySource;
  discoveredCount: number;
  completions: SmokeCompletion[];
};

export type SmokeOptions = {
  baseUrl: string;
  apiKey: string;
  modelIds: string[];
  timeoutMs?: number;
  expectedSource?: DiscoverySource;
  expectedApis?: ReadonlyMap<string, LiteLLMApi>;
  expectedResponseCost?: ReadonlyMap<string, boolean>;
  requireAllProtocols?: boolean;
};

const DISCOVERY_SOURCES: DiscoverySource[] = ["model_info", "models_list", "health"];

export function parseExpectedSource(raw: string | undefined): DiscoverySource | undefined {
  const trimmed = raw?.trim();
  if (!trimmed) return undefined;
  if (!(DISCOVERY_SOURCES as string[]).includes(trimmed)) {
    throw new Error(`LITELLM_SMOKE_EXPECT_SOURCE must be one of ${DISCOVERY_SOURCES.join(", ")}; got ${trimmed}`);
  }
  return trimmed as DiscoverySource;
}

type ChatCompletionResponse = {
  choices?: Array<{
    message?: {
      content?: unknown;
    };
  }>;
};

type ResponsesResponse = {
  output_text?: unknown;
  output?: Array<{ content?: Array<{ text?: unknown }> }>;
};

type MessagesResponse = {
  content?: Array<{ type?: unknown; text?: unknown }>;
};

export function parseSmokeModels(raw: string | undefined): string[] {
  return (raw ?? "")
    .split(/[,\s]+/)
    .map((model) => model.trim())
    .filter((model) => model.length > 0);
}

function parseExpectedMap<T extends string>(
  name: string,
  raw: string | undefined,
  allowed: readonly T[],
): Map<string, T> | undefined {
  const entries = parseSmokeModels(raw);
  if (entries.length === 0) return undefined;
  const result = new Map<string, T>();
  for (const entry of entries) {
    const separator = entry.lastIndexOf("=");
    const modelId = entry.slice(0, separator);
    const value = entry.slice(separator + 1) as T;
    if (separator <= 0 || !allowed.includes(value)) {
      throw new Error(`${name} entries must use model=value with value one of ${allowed.join(", ")}; got ${entry}`);
    }
    result.set(modelId, value);
  }
  return result;
}

export function parseExpectedApis(raw: string | undefined): Map<string, LiteLLMApi> | undefined {
  return parseExpectedMap("LITELLM_SMOKE_EXPECT_APIS", raw, [
    "anthropic-messages",
    "openai-completions",
    "openai-responses",
  ]);
}

export function parseExpectedResponseCost(raw: string | undefined): Map<string, boolean> | undefined {
  const parsed = parseExpectedMap("LITELLM_SMOKE_EXPECT_RESPONSE_COST", raw, ["present", "absent"]);
  return parsed && new Map([...parsed].map(([modelId, value]) => [modelId, value === "present"]));
}

export async function smokeCompletion(
  baseUrl: string,
  apiKey: string,
  modelId: string,
  api: LiteLLMApi,
  timeoutMs: number,
): Promise<SmokeCompletion> {
  const endpoint =
    api === "anthropic-messages"
      ? "/v1/messages"
      : api === "openai-responses"
        ? "/v1/responses"
        : "/v1/chat/completions";
  const body =
    api === "anthropic-messages"
      ? { model: modelId, messages: [{ role: "user", content: SMOKE_PROMPT }], max_tokens: 16 }
      : api === "openai-responses"
        ? { model: modelId, input: SMOKE_PROMPT, max_output_tokens: 16 }
        : {
            model: modelId,
            messages: [{ role: "user", content: SMOKE_PROMPT }],
            max_tokens: 16,
            temperature: 0,
          };
  const response = await fetch(`${baseUrl}${endpoint}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!response.ok) {
    throw new Error(`${endpoint} for ${modelId} returned ${response.status}`);
  }
  const data = (await response.json()) as ChatCompletionResponse & ResponsesResponse & MessagesResponse;
  const content =
    api === "anthropic-messages"
      ? data.content?.find((part) => part.type === "text")?.text
      : api === "openai-responses"
        ? (data.output_text ?? data.output?.flatMap((item) => item.content ?? []).find((part) => part.text)?.text)
        : data.choices?.[0]?.message?.content;
  if (typeof content !== "string" || content.trim().length === 0) {
    throw new Error(`${endpoint} for ${modelId} returned no assistant text`);
  }
  return {
    modelId,
    api,
    endpoint,
    content,
    hasResponseCost: response.headers.has("x-litellm-response-cost"),
  };
}

export function smokeChatCompletion(
  baseUrl: string,
  apiKey: string,
  modelId: string,
  timeoutMs: number,
): Promise<SmokeCompletion> {
  return smokeCompletion(baseUrl, apiKey, modelId, "openai-completions", timeoutMs);
}

export async function runSmoke(options: SmokeOptions): Promise<SmokeResult> {
  const baseUrl = normalizeBaseUrl(options.baseUrl);
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  if (options.modelIds.length === 0) {
    throw new Error("At least one smoke model must be configured in LITELLM_SMOKE_MODELS");
  }

  const discovery = await discoverModels(baseUrl, options.apiKey, { timeoutMs });
  if (options.expectedSource && discovery.source !== options.expectedSource) {
    throw new Error(`Discovery source mismatch: expected ${options.expectedSource}, got ${discovery.source}`);
  }
  const discovered = new Map(discovery.models.map((model) => [model.id, model]));
  const missing = options.modelIds.filter((modelId) => !discovered.has(modelId));
  if (missing.length > 0) {
    throw new Error(`Requested smoke models were not discovered: ${missing.join(", ")}`);
  }

  if (options.expectedApis) {
    for (const modelId of options.modelIds) {
      const expected = options.expectedApis.get(modelId);
      if (!expected) throw new Error(`No expected API configured for smoke model ${modelId}`);
      const actual = discovered.get(modelId)?.api;
      if (actual !== expected) throw new Error(`API mismatch for ${modelId}: expected ${expected}, got ${actual}`);
    }
  }

  const completions: SmokeCompletion[] = [];
  for (const modelId of options.modelIds) {
    const model = discovered.get(modelId);
    if (!model) continue;
    const completion = await smokeCompletion(baseUrl, options.apiKey, modelId, model.api, timeoutMs);
    const expectedCost = options.expectedResponseCost?.get(modelId);
    if (options.expectedResponseCost && expectedCost === undefined) {
      throw new Error(`No response-cost expectation configured for smoke model ${modelId}`);
    }
    if (expectedCost !== undefined && completion.hasResponseCost !== expectedCost) {
      const expected = expectedCost ? "present" : "absent";
      const actual = completion.hasResponseCost ? "present" : "absent";
      throw new Error(`Response-cost header mismatch for ${modelId}: expected ${expected}, got ${actual}`);
    }
    completions.push(completion);
  }
  if (options.requireAllProtocols) {
    const endpoints = new Set(completions.map(({ endpoint }) => endpoint));
    const required = ["/v1/messages", "/v1/chat/completions", "/v1/responses"];
    const absent = required.filter((endpoint) => !endpoints.has(endpoint));
    if (absent.length > 0) {
      throw new Error(
        `LITELLM_SMOKE_REQUIRE_ALL_PROTOCOLS is enabled, but smoke endpoint coverage is missing: ${absent.join(", ")}`,
      );
    }
  }

  return {
    source: discovery.source,
    discoveredCount: discovery.models.length,
    completions,
  };
}

export async function runSmokeFromEnv(env: NodeJS.ProcessEnv = process.env): Promise<SmokeResult> {
  const baseUrl = env.LITELLM_BASE_URL?.trim();
  const apiKey = env.LITELLM_API_KEY?.trim();
  if (!baseUrl || !apiKey) {
    throw new Error("LITELLM_BASE_URL and LITELLM_API_KEY must be set");
  }

  const timeoutMs = env.LITELLM_SMOKE_TIMEOUT_MS
    ? Number.parseInt(env.LITELLM_SMOKE_TIMEOUT_MS, 10)
    : DEFAULT_TIMEOUT_MS;
  return runSmoke({
    baseUrl,
    apiKey,
    modelIds: parseSmokeModels(env.LITELLM_SMOKE_MODELS),
    timeoutMs: Number.isNaN(timeoutMs) || timeoutMs <= 0 ? DEFAULT_TIMEOUT_MS : timeoutMs,
    expectedSource: parseExpectedSource(env.LITELLM_SMOKE_EXPECT_SOURCE),
    expectedApis: parseExpectedApis(env.LITELLM_SMOKE_EXPECT_APIS),
    expectedResponseCost: parseExpectedResponseCost(env.LITELLM_SMOKE_EXPECT_RESPONSE_COST),
    requireAllProtocols: env.LITELLM_SMOKE_REQUIRE_ALL_PROTOCOLS === "1",
  });
}

if (import.meta.main) {
  runSmokeFromEnv()
    .then((result) => {
      console.log(`Source: ${result.source}`);
      console.log(`Discovered ${result.discoveredCount} models.`);
      for (const completion of result.completions) {
        const cost = completion.hasResponseCost ? "cost header present" : "cost header absent; static estimate remains";
        console.log(
          `Smoke OK: ${completion.modelId} ${completion.endpoint} (${cost}) -> ${JSON.stringify(completion.content)}`,
        );
      }
    })
    .catch((error) => {
      console.error(error instanceof Error ? error.message : error);
      process.exitCode = 1;
    });
}
