import { readFile } from "node:fs/promises";
import { join } from "node:path";
import type { Api, AssistantMessage, Model, OAuthCredentials, OAuthLoginCallbacks } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ProviderModelConfig } from "@earendil-works/pi-coding-agent";
import { AuthStorage, getAgentDir } from "@earendil-works/pi-coding-agent";
import { fingerprint, readCache, writeCache } from "./cache.js";
import { setupLiteLLMCostTracking } from "./cost.js";
import { discoverModels, normalizeBaseUrl, shouldSuppressReasoningContent } from "./discover.js";
import {
  CLI_BASE_URL_FLAG,
  CLI_HOST_FLAG,
  getCliBaseUrl,
  getConfiguredBaseUrl,
  getSessionIdFromFile,
} from "./litellm.js";
import type { AuthFileEntry, CacheFile, DiscoveryOptions, DiscoveryResult, ResolvedCredentials } from "./types.js";

const PROVIDER_NAME = "litellm";
const ENV_BASE_URL = "LITELLM_BASE_URL";
const ENV_API_KEY = "LITELLM_API_KEY";
const ENV_TIMEOUT = "LITELLM_DISCOVERY_TIMEOUT_MS";
const ENV_OFFLINE = "LITELLM_OFFLINE";
const DEFAULT_TIMEOUT_MS = 5000;
const LOGIN_TIMEOUT_MS = 10_000;
const CACHE_FILENAME = "litellm-models.json";
const CACHE_STALE_MS = 24 * 60 * 60 * 1000;

type RefreshResult = { models: ProviderModelConfig[]; source: string };

function getAuthPath(): string {
  return join(getAgentDir(), "auth.json");
}

function getCachePath(): string {
  return join(getAgentDir(), CACHE_FILENAME);
}

async function readAuthEntry(): Promise<AuthFileEntry | undefined> {
  try {
    const raw = await readFile(getAuthPath(), "utf8");
    const parsed = JSON.parse(raw) as Record<string, AuthFileEntry>;
    return parsed?.[PROVIDER_NAME];
  } catch {
    return undefined;
  }
}

async function resolveCredentials(): Promise<ResolvedCredentials> {
  const entry = await readAuthEntry();
  const cliBase = getCliBaseUrl();
  const configBase = await getConfiguredBaseUrl();
  const envBase = process.env[ENV_BASE_URL]?.trim();
  const envKey = process.env[ENV_API_KEY]?.trim();
  const authBase = entry?.type === "oauth" ? entry.baseUrl?.trim() : undefined;
  const authKey =
    entry?.type === "oauth"
      ? entry.access?.trim()
      : entry?.type === "api_key"
        ? (await AuthStorage.create(getAuthPath()).getApiKey(PROVIDER_NAME, { includeFallback: false }))?.trim()
        : undefined;
  const rawBase = cliBase || configBase || authBase || envBase;
  return {
    baseUrl: rawBase ? normalizeBaseUrl(rawBase) : undefined,
    apiKey: authKey || envKey || undefined,
  };
}

function getDiscoveryTimeoutMs(): number {
  const raw = process.env[ENV_TIMEOUT];
  if (raw === undefined) return DEFAULT_TIMEOUT_MS;
  const parsed = Number.parseInt(raw, 10);
  if (Number.isNaN(parsed) || parsed < 0) return DEFAULT_TIMEOUT_MS;
  return parsed;
}

function isOffline(): boolean {
  return process.env[ENV_OFFLINE] === "1";
}

function isListModelsMode(): boolean {
  return process.argv.includes("--list-models");
}

async function discoverWithFallback(
  baseUrl: string,
  apiKey: string,
  options: DiscoveryOptions,
): Promise<{ result: DiscoveryResult; warning?: string }> {
  try {
    return { result: await discoverModels(baseUrl, apiKey, options) };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return {
      result: { models: [] as ProviderModelConfig[], source: "models_list" },
      warning: message,
    };
  }
}

async function loginLiteLLM(
  callbacks: OAuthLoginCallbacks,
  onCacheWrite?: (cache: CacheFile) => void,
): Promise<OAuthCredentials> {
  const rawBaseUrl = (
    await callbacks.onPrompt({
      message: "Enter LiteLLM proxy URL (no trailing /v1):",
      placeholder: "https://litellm.example.com",
    })
  ).trim();
  const apiKey = (await callbacks.onPrompt({ message: "Enter API key:" })).trim();
  if (!rawBaseUrl || !apiKey) throw new Error("Both base URL and API key are required");

  const baseUrl = normalizeBaseUrl(rawBaseUrl);
  const { models, source } = await discoverModels(baseUrl, apiKey, {
    timeoutMs: LOGIN_TIMEOUT_MS,
    signal: callbacks.signal,
  });

  const cache: CacheFile = {
    baseUrl,
    apiKeyFingerprint: fingerprint(apiKey),
    fetchedAt: Date.now(),
    source,
    models,
  };
  await writeCache(getCachePath(), cache);
  onCacheWrite?.(cache);
  callbacks.onProgress?.(`LiteLLM: ${models.length} models discovered (source: ${source})`);

  return {
    access: apiKey,
    refresh: "",
    expires: Number.MAX_SAFE_INTEGER,
    baseUrl,
  } as OAuthCredentials & { baseUrl: string };
}

async function refreshLiteLLM(credentials: OAuthCredentials): Promise<OAuthCredentials> {
  return credentials;
}

function modifyLiteLLMModels(models: Model<Api>[], cred: OAuthCredentials): Model<Api>[] {
  const baseUrl = (cred as { baseUrl?: string }).baseUrl;
  if (!baseUrl) return models;
  return models.map((m) => (m.provider === PROVIDER_NAME ? { ...m, baseUrl: `${baseUrl}/v1` } : m));
}

function prepareLiteLLMRequestPayload(
  payload: Record<string, unknown>,
  modelId: string | undefined,
  sessionId: string | undefined,
): Record<string, unknown> | undefined {
  let next: Record<string, unknown> | undefined;
  const update = (key: string, value: unknown): void => {
    if (payload[key] !== undefined) return;
    next ??= { ...payload };
    next[key] = value;
  };

  if (modelId && shouldSuppressReasoningContent(modelId)) {
    update("include_reasoning", false);
    update("reasoning_content", false);
    update("merge_reasoning_content_in_choices", true);
    update("thinking", { type: "disabled" });
  }

  if (sessionId) {
    next ??= { ...payload };
    next.litellm_session_id = sessionId;
  }

  return next;
}

function normalizeThinkTags(message: AssistantMessage): AssistantMessage | undefined {
  if (message.provider !== PROVIDER_NAME || !shouldSuppressReasoningContent(message.model)) return;

  let changed = false;
  const content: AssistantMessage["content"] = [];
  const appendText = (text: string): void => {
    if (!text) return;
    const last = content.at(-1);
    if (last?.type === "text") {
      last.text += text;
      return;
    }
    content.push({ type: "text", text });
  };
  const appendThinking = (thinking: string): void => {
    if (!thinking) return;
    const last = content.at(-1);
    if (last?.type === "thinking") {
      last.thinking += thinking;
      return;
    }
    content.push({ type: "thinking", thinking });
  };

  for (let blockIndex = 0; blockIndex < message.content.length; blockIndex++) {
    const block = message.content[blockIndex];
    if (block.type !== "text") {
      content.push(block);
      continue;
    }

    let index = 0;
    while (index < block.text.length) {
      const start = block.text.indexOf("<think>", index);
      if (start === -1) {
        appendText(block.text.slice(index));
        break;
      }

      changed = true;
      appendText(block.text.slice(index, start));
      const thinkingStart = start + "<think>".length;
      const end = block.text.indexOf("</think>", thinkingStart);
      if (end === -1) {
        const isBeforeNonTextContent = message.content
          .slice(blockIndex + 1)
          .some((nextBlock) => nextBlock.type !== "text");
        if (isBeforeNonTextContent) appendThinking(block.text.slice(thinkingStart));
        else appendText(block.text.slice(thinkingStart));
        index = block.text.length;
        break;
      }

      appendThinking(block.text.slice(thinkingStart, end));
      index = end + "</think>".length;
    }
  }

  if (!changed) return;
  return { ...message, content };
}

export default async function (pi: ExtensionAPI): Promise<void> {
  pi.registerFlag(CLI_HOST_FLAG, {
    description: "LiteLLM proxy host URL (alias for --litellm-base-url)",
    type: "string",
  });
  pi.registerFlag(CLI_BASE_URL_FLAG, {
    description: "LiteLLM proxy base URL",
    type: "string",
  });

  const creds = await resolveCredentials();
  const cache = await readCache(getCachePath());
  const fp = creds.apiKey ? fingerprint(creds.apiKey) : undefined;
  let cacheFetchedAt = cache?.fetchedAt ?? 0;

  const cacheValid =
    cache !== null &&
    creds.baseUrl !== undefined &&
    fp !== undefined &&
    cache.baseUrl === creds.baseUrl &&
    cache.apiKeyFingerprint === fp;

  let models: ProviderModelConfig[] = cacheValid && cache ? cache.models : [];
  const haveCreds = creds.baseUrl !== undefined && creds.apiKey !== undefined && fp !== undefined;
  const shouldFetch = haveCreds && !isOffline() && (!cacheValid || isListModelsMode());

  if (shouldFetch && creds.baseUrl && creds.apiKey && fp) {
    const timeoutMs = getDiscoveryTimeoutMs();
    if (timeoutMs > 0) {
      const { result, warning } = await discoverWithFallback(creds.baseUrl, creds.apiKey, {
        timeoutMs,
      });
      if (warning) {
        if (cacheValid && cache) {
          process.stderr.write(`LiteLLM: discovery failed (${warning}); using cached models.\n`);
          models = cache.models;
        } else {
          process.stderr.write(`LiteLLM: discovery failed (${warning}); registering provider with no models.\n`);
          models = [];
        }
      } else {
        models = result.models;
        const next: CacheFile = {
          baseUrl: creds.baseUrl,
          apiKeyFingerprint: fp,
          fetchedAt: Date.now(),
          source: result.source,
          models: result.models,
        };
        await writeCache(getCachePath(), next);
        cacheFetchedAt = next.fetchedAt;
        if (isListModelsMode()) {
          process.stderr.write(`LiteLLM: ${result.models.length} models discovered (source: ${result.source}).\n`);
        }
      }
    }
  }

  const oauth = {
    name: "LiteLLM",
    login: (callbacks: OAuthLoginCallbacks) =>
      loginLiteLLM(callbacks, (next) => {
        cacheFetchedAt = next.fetchedAt;
      }),
    refreshToken: refreshLiteLLM,
    getApiKey: (cred: OAuthCredentials) => cred.access,
    modifyModels: modifyLiteLLMModels,
  };

  function registerProvider(baseUrl: string | undefined, models: ProviderModelConfig[]): void {
    pi.registerProvider(PROVIDER_NAME, {
      baseUrl: baseUrl ? `${baseUrl}/v1` : "https://litellm.example.com/v1",
      apiKey: ENV_API_KEY,
      api: "openai-completions",
      models,
      oauth,
    });
  }

  registerProvider(creds.baseUrl, models);

  const updateCosts = setupLiteLLMCostTracking(pi, models);

  let refreshInProgress: Promise<RefreshResult> | null = null;

  function discoveryDisabledReason(): string | null {
    if (isOffline()) return `${ENV_OFFLINE}=1`;
    if (getDiscoveryTimeoutMs() === 0) return `${ENV_TIMEOUT}=0`;
    return null;
  }

  async function refreshModelsAndCosts(): Promise<RefreshResult> {
    const fresh = await resolveCredentials();
    const freshFp = fresh.apiKey ? fingerprint(fresh.apiKey) : undefined;
    if (!fresh.baseUrl || !fresh.apiKey || !freshFp) {
      throw new Error("no credentials. Run /login litellm or set env vars.");
    }
    const result = await discoverModels(fresh.baseUrl, fresh.apiKey, { timeoutMs: getDiscoveryTimeoutMs() });
    const now = Date.now();
    await writeCache(getCachePath(), {
      baseUrl: fresh.baseUrl,
      apiKeyFingerprint: freshFp,
      fetchedAt: now,
      source: result.source,
      models: result.models,
    });
    registerProvider(fresh.baseUrl, result.models);
    updateCosts(result.models);
    cacheFetchedAt = now;
    return { models: result.models, source: result.source };
  }

  function runRefresh(): Promise<RefreshResult> {
    refreshInProgress ??= refreshModelsAndCosts().finally(() => {
      refreshInProgress = null;
    });
    return refreshInProgress;
  }

  pi.registerCommand("litellm-refresh", {
    description: "Re-discover models from the LiteLLM proxy.",
    handler: async (_args, ctx) => {
      const disabledReason = discoveryDisabledReason();
      if (disabledReason) {
        ctx.ui.notify(`LiteLLM refresh disabled (${disabledReason})`, "warning");
        return;
      }
      try {
        const result = await runRefresh();
        ctx.ui.notify(`LiteLLM: ${result.models.length} models refreshed (source: ${result.source})`, "info");
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        ctx.ui.notify(`LiteLLM refresh failed: ${message}`, "error");
      }
    },
  });

  let sessionId: string | undefined;
  pi.on("session_start", (_event, ctx) => {
    sessionId = getSessionIdFromFile(ctx.sessionManager.getSessionFile());

    if (discoveryDisabledReason() || !cacheFetchedAt || Date.now() - cacheFetchedAt <= CACHE_STALE_MS) return;
    void runRefresh().catch(() => undefined);
  });

  pi.on("before_provider_request", (event, ctx) => {
    if (ctx.model?.provider !== PROVIDER_NAME) return;
    if (typeof event.payload !== "object" || event.payload === null) return;
    return prepareLiteLLMRequestPayload(event.payload as Record<string, unknown>, ctx.model?.id, sessionId);
  });

  pi.on("message_end", (event) => {
    if (event.message.role !== "assistant") return;
    const message = normalizeThinkTags(event.message as AssistantMessage);
    if (!message) return;
    return { message };
  });
}
