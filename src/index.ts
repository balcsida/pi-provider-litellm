import { readFile } from "node:fs/promises";
import { join } from "node:path";
import type { Api, Model, OAuthCredentials, OAuthLoginCallbacks } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ProviderModelConfig } from "@earendil-works/pi-coding-agent";
import { AuthStorage, getAgentDir } from "@earendil-works/pi-coding-agent";
import { fingerprint, readCache, writeCache } from "./cache.js";
import { setupLiteLLMCostTracking } from "./cost.js";
import { discoverModels, normalizeBaseUrl } from "./discover.js";
import { getSessionIdFromFile } from "./litellm.js";
import type { AuthFileEntry, CacheFile, DiscoveryOptions, DiscoveryResult, ResolvedCredentials } from "./types.js";

const PROVIDER_NAME = "litellm";
const ENV_BASE_URL = "LITELLM_BASE_URL";
const ENV_API_KEY = "LITELLM_API_KEY";
const ENV_TIMEOUT = "LITELLM_DISCOVERY_TIMEOUT_MS";
const ENV_OFFLINE = "LITELLM_OFFLINE";
const DEFAULT_TIMEOUT_MS = 5000;
const LOGIN_TIMEOUT_MS = 10_000;
const CACHE_FILENAME = "litellm-models.json";

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
  const envBase = process.env[ENV_BASE_URL]?.trim();
  const envKey = process.env[ENV_API_KEY]?.trim();
  const authBase = entry?.type === "oauth" ? entry.baseUrl?.trim() : undefined;
  const authKey =
    entry?.type === "oauth"
      ? entry.access?.trim()
      : entry?.type === "api_key"
        ? (await AuthStorage.create(getAuthPath()).getApiKey(PROVIDER_NAME, { includeFallback: false }))?.trim()
        : undefined;
  const rawBase = authBase || envBase;
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

async function loginLiteLLM(callbacks: OAuthLoginCallbacks): Promise<OAuthCredentials> {
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

  await writeCache(getCachePath(), {
    baseUrl,
    apiKeyFingerprint: fingerprint(apiKey),
    fetchedAt: Date.now(),
    source,
    models,
  });
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

export default async function (pi: ExtensionAPI): Promise<void> {
  const creds = await resolveCredentials();
  const cache = await readCache(getCachePath());
  const fp = creds.apiKey ? fingerprint(creds.apiKey) : undefined;

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
        if (isListModelsMode()) {
          process.stderr.write(`LiteLLM: ${result.models.length} models discovered (source: ${result.source}).\n`);
        }
      }
    }
  }

  const oauth = {
    name: "LiteLLM",
    login: loginLiteLLM,
    refreshToken: refreshLiteLLM,
    getApiKey: (cred: OAuthCredentials) => cred.access,
    modifyModels: modifyLiteLLMModels,
  };

  pi.registerProvider(PROVIDER_NAME, {
    baseUrl: creds.baseUrl ? `${creds.baseUrl}/v1` : "https://litellm.example.com/v1",
    apiKey: ENV_API_KEY,
    api: "openai-completions",
    models,
    oauth,
  });

  pi.registerCommand("litellm-refresh", {
    description: "Re-discover models from the LiteLLM proxy.",
    handler: async (_args, ctx) => {
      if (isOffline()) {
        ctx.ui.notify(`LiteLLM refresh disabled (${ENV_OFFLINE}=1)`, "warning");
        return;
      }
      const timeoutMs = getDiscoveryTimeoutMs();
      if (timeoutMs === 0) {
        ctx.ui.notify(`LiteLLM refresh disabled (${ENV_TIMEOUT}=0)`, "warning");
        return;
      }
      const fresh = await resolveCredentials();
      const freshFp = fresh.apiKey ? fingerprint(fresh.apiKey) : undefined;
      if (!fresh.baseUrl || !fresh.apiKey || !freshFp) {
        ctx.ui.notify("LiteLLM refresh failed: no credentials. Run /login litellm or set env vars.", "error");
        return;
      }
      try {
        const result = await discoverModels(fresh.baseUrl, fresh.apiKey, { timeoutMs });
        await writeCache(getCachePath(), {
          baseUrl: fresh.baseUrl,
          apiKeyFingerprint: freshFp,
          fetchedAt: Date.now(),
          source: result.source,
          models: result.models,
        });
        pi.registerProvider(PROVIDER_NAME, {
          baseUrl: `${fresh.baseUrl}/v1`,
          apiKey: ENV_API_KEY,
          api: "openai-completions",
          models: result.models,
          oauth,
        });
        ctx.ui.notify(`LiteLLM: ${result.models.length} models refreshed (source: ${result.source})`, "info");
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        ctx.ui.notify(`LiteLLM refresh failed: ${message}`, "error");
      }
    },
  });

  setupLiteLLMCostTracking(pi);

  let sessionId: string | undefined;
  pi.on("session_start", async (_event, ctx) => {
    sessionId = getSessionIdFromFile(ctx.sessionManager.getSessionFile());
  });

  pi.on("before_provider_request", (event) => {
    if (!sessionId) return;
    if (typeof event.payload !== "object" || event.payload === null) return;
    return {
      ...(event.payload as Record<string, unknown>),
      litellm_session_id: sessionId,
    };
  });
}
