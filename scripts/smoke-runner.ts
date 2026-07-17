import { discoverModels, normalizeBaseUrl } from "../src/discover.js";
import type { DiscoverySource } from "../src/types.js";

const DEFAULT_TIMEOUT_MS = 30_000;
const SMOKE_PROMPT = "Reply with one short word.";

export type SmokeCompletion = {
  modelId: string;
  content: string;
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

export function parseSmokeModels(raw: string | undefined): string[] {
  return (raw ?? "")
    .split(/[,\s]+/)
    .map((model) => model.trim())
    .filter((model) => model.length > 0);
}

async function readErrorBody(response: Response): Promise<string> {
  try {
    const body = await response.text();
    return body ? `: ${body.slice(0, 500)}` : "";
  } catch {
    return "";
  }
}

export async function smokeChatCompletion(
  baseUrl: string,
  apiKey: string,
  modelId: string,
  timeoutMs: number,
): Promise<SmokeCompletion> {
  const response = await fetch(`${baseUrl}/v1/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: modelId,
      messages: [{ role: "user", content: SMOKE_PROMPT }],
      max_tokens: 16,
      temperature: 0,
    }),
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!response.ok) {
    throw new Error(`/v1/chat/completions for ${modelId} returned ${response.status}${await readErrorBody(response)}`);
  }
  const data = (await response.json()) as ChatCompletionResponse;
  const content = data.choices?.[0]?.message?.content;
  if (typeof content !== "string" || content.trim().length === 0) {
    throw new Error(`/v1/chat/completions for ${modelId} returned no assistant text`);
  }
  return { modelId, content };
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
  const discovered = new Set(discovery.models.map((model) => model.id));
  const missing = options.modelIds.filter((modelId) => !discovered.has(modelId));
  if (missing.length > 0) {
    throw new Error(`Requested smoke models were not discovered: ${missing.join(", ")}`);
  }

  const completions: SmokeCompletion[] = [];
  for (const modelId of options.modelIds) {
    completions.push(await smokeChatCompletion(baseUrl, options.apiKey, modelId, timeoutMs));
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
  });
}

if (import.meta.main) {
  runSmokeFromEnv()
    .then((result) => {
      console.log(`Source: ${result.source}`);
      console.log(`Discovered ${result.discoveredCount} models.`);
      for (const completion of result.completions) {
        console.log(`Smoke OK: ${completion.modelId} -> ${JSON.stringify(completion.content)}`);
      }
    })
    .catch((error) => {
      console.error(error instanceof Error ? error.message : error);
      process.exitCode = 1;
    });
}
