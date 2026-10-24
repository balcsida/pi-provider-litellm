import type {
  AssistantImages,
  ImageContent,
  ImageModel,
  ModelCost,
  ProviderImages,
  Usage,
} from "@earendil-works/pi-ai";
import { LITELLM_DISCOVERY_VERSION } from "./backend-identity.js";
import { wireString } from "./model-groups.js";
import type { ModelInfoEntry } from "./types.js";

/** Image generation through the proxy's OpenAI-compatible `/v1/images/generations` endpoint. */
export const LITELLM_IMAGES_API = "litellm-images" as const;
export type LiteLLMImageApi = typeof LITELLM_IMAGES_API;

export type DiscoveredImageModel = Omit<ImageModel<LiteLLMImageApi>, "provider" | "baseUrl"> & {
  litellmDiscoveryVersion?: typeof LITELLM_DISCOVERY_VERSION;
};

export type LiteLLMImageModel = ImageModel<LiteLLMImageApi> & {
  litellmDiscoveryVersion?: typeof LITELLM_DISCOVERY_VERSION;
};

const IMAGE_GENERATION_MODE_PATTERN = /^image_generation$/i;
// An image response carries base64 payloads that can be several MiB each; an
// error body is only ever read for its message, so cap what is kept of it.
const MAX_ERROR_DETAIL = 300;

export function isImageGenerationMode(mode: unknown): boolean {
  const value = wireString(mode);
  return value !== undefined && IMAGE_GENERATION_MODE_PATTERN.test(value.trim());
}

export function perMillion(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value * 1_000_000 : undefined;
}

/**
 * Reduce one `/model/info` route group to an image model. Every deployment behind the
 * route must declare `mode: "image_generation"`: LiteLLM load-balances across them, so a
 * mixed group could answer an image request from a chat or embedding deployment. Groups
 * with no readable mode, or any other mode, are left to the chat reducer, which already
 * withholds every explicitly non-chat route.
 */
export function mapFromImageModelInfoGroup(
  route: string,
  entries: readonly ModelInfoEntry[],
): DiscoveredImageModel | undefined {
  // A wildcard row is not addressable, and `/v1/models` expansion carries no mode evidence.
  if (route.includes("*")) return undefined;
  const deployments = entries.filter((entry) => wireString(entry.model_name) === route);
  if (deployments.length === 0) return undefined;
  if (!deployments.every((entry) => isImageGenerationMode(entry.model_info?.mode))) return undefined;
  // Prices are estimates for Pi's usage display; LiteLLM's `x-litellm-response-cost` header,
  // when present, replaces them per request. Take the highest rate any deployment declares,
  // like the chat reducer, and report 0 when not every deployment declares one.
  const rate = (read: (entry: ModelInfoEntry) => number | undefined): number => {
    const values = deployments.map(read);
    return values.every((value) => value !== undefined) ? Math.max(...(values as number[])) : 0;
  };
  return {
    type: "image",
    id: route,
    name: route,
    api: LITELLM_IMAGES_API,
    // `/v1/images/generations` takes a text prompt only; reference images need `/v1/images/edits`.
    input: ["text"],
    output: ["image"],
    cost: {
      input: rate((entry) => perMillion(entry.model_info?.input_cost_per_token)),
      output: rate(
        (entry) =>
          perMillion(entry.model_info?.output_cost_per_image_token) ??
          perMillion(entry.model_info?.output_cost_per_token),
      ),
      cacheRead: 0,
      cacheWrite: 0,
    },
    litellmDiscoveryVersion: LITELLM_DISCOVERY_VERSION,
  };
}

type ImagesResponseEntry = { b64_json?: unknown; url?: unknown };
type ImagesResponse = {
  data?: unknown;
  output_format?: unknown;
  usage?: { input_tokens?: unknown; output_tokens?: unknown; prompt_tokens?: unknown; completion_tokens?: unknown };
};

const FORMAT_MIME_TYPES: Record<string, string> = {
  png: "image/png",
  jpeg: "image/jpeg",
  jpg: "image/jpeg",
  webp: "image/webp",
  gif: "image/gif",
};

// Base64 prefixes of each format's magic bytes, for responses without `output_format`
// (LiteLLM omits it for most non-OpenAI backends, e.g. Gemini returns JPEG).
const BASE64_SIGNATURES: [prefix: string, mimeType: string][] = [
  ["iVBORw0KGgo", "image/png"],
  ["/9j/", "image/jpeg"],
  ["UklGR", "image/webp"],
  ["R0lGOD", "image/gif"],
];

function imageMimeType(data: string, outputFormat: unknown): string {
  const declared = wireString(outputFormat)?.trim().toLowerCase();
  if (declared && FORMAT_MIME_TYPES[declared]) return FORMAT_MIME_TYPES[declared];
  return BASE64_SIGNATURES.find(([prefix]) => data.startsWith(prefix))?.[1] ?? "image/png";
}

function tokenCount(...values: unknown[]): number {
  for (const value of values) {
    if (typeof value === "number" && Number.isFinite(value) && value >= 0) return value;
  }
  return 0;
}

export function parseUsage(
  raw:
    | { input_tokens?: unknown; output_tokens?: unknown; prompt_tokens?: unknown; completion_tokens?: unknown }
    | undefined,
  cost: ModelCost,
  responseCost: number | undefined,
): Usage | undefined {
  if (!raw && responseCost === undefined) return undefined;
  const input = tokenCount(raw?.input_tokens, raw?.prompt_tokens);
  const output = tokenCount(raw?.output_tokens, raw?.completion_tokens);
  const estimated = {
    input: (cost.input / 1_000_000) * input,
    output: (cost.output / 1_000_000) * output,
  };
  // LiteLLM's own cost is authoritative: it knows per-image prices that a token rate cannot express.
  const total = responseCost ?? estimated.input + estimated.output;
  const estimatedTotal = estimated.input + estimated.output;
  const share = (part: number): number =>
    responseCost === undefined ? part : estimatedTotal > 0 ? (responseCost * part) / estimatedTotal : 0;
  const outputCost = responseCost !== undefined && estimatedTotal === 0 ? responseCost : share(estimated.output);
  return {
    input,
    output,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: input + output,
    cost: { input: share(estimated.input), output: outputCost, cacheRead: 0, cacheWrite: 0, total },
  };
}

/** LiteLLM's `x-litellm-response-cost` header, when it is a usable amount. */
export function responseCost(headers: Record<string, string>): number | undefined {
  const cost = Number.parseFloat(headers["x-litellm-response-cost"] ?? "");
  return Number.isFinite(cost) && cost >= 0 ? cost : undefined;
}

export function requestHeaders(
  apiKey: string,
  ...sources: (Record<string, string | null | undefined> | undefined)[]
): Record<string, string> {
  // Header names are case-insensitive: `fetch` would join `X-Tenant` and `x-tenant` into one
  // comma-separated value, and a capitalized `Authorization` would merge with the bearer below.
  const headers: Record<string, string> = {};
  for (const source of sources) {
    for (const [name, value] of Object.entries(source ?? {})) {
      const normalizedName = name.toLowerCase();
      if (typeof value === "string") headers[normalizedName] = value;
      else delete headers[normalizedName];
    }
  }
  headers.authorization = `Bearer ${apiKey}`;
  headers["content-type"] = "application/json";
  return headers;
}

export function errorDetail(body: string): string {
  try {
    const parsed = JSON.parse(body) as { error?: { message?: unknown } | unknown };
    const message =
      typeof parsed.error === "object" && parsed.error !== null
        ? wireString((parsed.error as { message?: unknown }).message)
        : wireString(parsed.error);
    if (message) return message.slice(0, MAX_ERROR_DETAIL);
  } catch {
    // Not JSON: fall through to the raw text.
  }
  return body.slice(0, MAX_ERROR_DETAIL);
}

export function requestSignal(
  options: { signal?: AbortSignal; timeoutMs?: number } | undefined,
): AbortSignal | undefined {
  const signals = [
    options?.signal,
    options?.timeoutMs !== undefined ? AbortSignal.timeout(options.timeoutMs) : undefined,
  ].filter((signal): signal is AbortSignal => signal !== undefined);
  return signals.length > 1 ? AbortSignal.any(signals) : signals[0];
}

/** Generate images through LiteLLM's `/v1/images/generations`. Never rejects. */
export const generateImages: ProviderImages["generateImages"] = async (model, context, options) => {
  const result: AssistantImages = {
    api: model.api,
    provider: model.provider,
    model: model.id,
    output: [],
    stopReason: "stop",
    timestamp: Date.now(),
  };
  try {
    const apiKey = options?.apiKey;
    if (!apiKey) throw new Error(`No API key for provider: ${model.provider}`);
    if (context.input.some((item) => item.type === "image")) {
      throw new Error(
        `LiteLLM image model ${model.id} takes a text prompt only; ` +
          "reference images need /v1/images/edits, which this provider does not call",
      );
    }
    const prompt = context.input
      .map((item) => (item.type === "text" ? item.text : ""))
      .filter((text) => text.trim().length > 0)
      .join("\n\n");
    if (!prompt) throw new Error(`LiteLLM image model ${model.id} needs a non-empty text prompt`);
    let params: Record<string, unknown> = { model: model.id, prompt, n: 1 };
    const nextParams = await options?.onPayload?.(params, model);
    if (nextParams !== undefined) params = nextParams as Record<string, unknown>;
    const response = await (options?.fetch ?? fetch)(`${model.baseUrl}/images/generations`, {
      method: "POST",
      headers: requestHeaders(apiKey, model.headers, options?.headers),
      body: JSON.stringify(params),
      signal: requestSignal(options),
    });
    const responseHeaders = Object.fromEntries(response.headers.entries());
    await options?.onResponse?.({ status: response.status, headers: responseHeaders }, model);
    const body = await response.text();
    if (!response.ok) throw new Error(`LiteLLM images request failed (${response.status}): ${errorDetail(body)}`);
    const parsed = JSON.parse(body) as ImagesResponse;
    const entries = Array.isArray(parsed.data) ? (parsed.data as ImagesResponseEntry[]) : [];
    for (const entry of entries) {
      const data = wireString(entry?.b64_json);
      if (!data) continue;
      const image: ImageContent = { type: "image", data, mimeType: imageMimeType(data, parsed.output_format) };
      result.output.push(image);
    }
    if (result.output.length === 0) {
      // Remote URLs are not fetched: the URL host is chosen by the backend, not the proxy.
      throw new Error(
        entries.some((entry) => wireString(entry?.url))
          ? `LiteLLM image model ${model.id} returned image URLs instead of base64 data`
          : `LiteLLM image model ${model.id} returned no images`,
      );
    }
    const usage = parseUsage(parsed.usage, model.cost, responseCost(responseHeaders));
    if (usage) result.usage = usage;
    return result;
  } catch (error) {
    result.stopReason = options?.signal?.aborted ? "aborted" : "error";
    result.errorMessage = error instanceof Error ? error.message : String(error);
    return result;
  }
};

export function createLiteLLMImageApis(): Record<LiteLLMImageApi, ProviderImages> {
  return { [LITELLM_IMAGES_API]: { generateImages } };
}
