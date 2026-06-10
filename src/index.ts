import { execSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import type { Api, AssistantMessage, Model, OAuthCredentials, OAuthLoginCallbacks } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext, ProviderModelConfig } from "@earendil-works/pi-coding-agent";
import { AuthStorage, getAgentDir } from "@earendil-works/pi-coding-agent";
import { fingerprint, readCache, writeCache } from "./cache.js";
import { setupLiteLLMCostTracking } from "./cost.js";
import { discoverModels, normalizeBaseUrl, shouldSuppressReasoningContent } from "./discover.js";
import {
  getGcloudToken,
  getGcloudTokenCacheKey,
  getGcloudTokenCommand,
  isGcloudTokenAuthEnabled,
} from "./gcloud-token.js";
import { getSessionIdFromFile } from "./litellm.js";
import { createMcpToolDefinitions } from "./mcp-tools.js";
import { createSkillsPromptSection, createSkillToolDefinitions, listSkills } from "./skills.js";
import type { AuthFileEntry, CacheFile, DiscoveryOptions, DiscoveryResult, ResolvedCredentials } from "./types.js";

const PROVIDER_NAME = "litellm";
const ENV_BASE_URL = "LITELLM_BASE_URL";
const ENV_API_KEY = "LITELLM_API_KEY";
const ENV_API_KEY_HELPER = "LITELLM_API_KEY_HELPER";
const ENV_TIMEOUT = "LITELLM_DISCOVERY_TIMEOUT_MS";
const ENV_OFFLINE = "LITELLM_OFFLINE";
const DEFAULT_TIMEOUT_MS = 5000;
const LOGIN_TIMEOUT_MS = 10_000;
const CACHE_FILENAME = "litellm-models.json";
const CACHE_STALE_MS = 24 * 60 * 60 * 1000;
const TOKEN_REFRESH_LEAD_MS = 5 * 60 * 1000;
const PERMANENT_TOKEN_EXPIRES_AT = Number.MAX_SAFE_INTEGER;
const EXPIRE_TOKEN_IMMEDIATELY = 0;

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

function cleanConfig(raw: string | undefined): string | undefined {
  const trimmed = raw?.trim();
  return trimmed && trimmed !== "undefined" ? trimmed : undefined;
}

function normalizeCommand(raw: string | undefined): string | undefined {
  const trimmed = cleanConfig(raw);
  if (!trimmed) return undefined;
  return trimmed.startsWith("!") ? trimmed : `!${trimmed}`;
}

function getApiKeyHelperCommand(): string | undefined {
  return normalizeCommand(process.env[ENV_API_KEY_HELPER]);
}

function executeApiKeyCommand(commandConfig: string): string {
  const command = commandConfig.startsWith("!") ? commandConfig.slice(1) : commandConfig;
  const output = execSync(command, { encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], timeout: 10_000 }).trim();
  if (!output) throw new Error(`LiteLLM API key helper produced no output: ${command}`);
  return output;
}

function tokenExpiresAt(apiKey: string, opaqueFallback = PERMANENT_TOKEN_EXPIRES_AT): number {
  const [, payload] = apiKey.split(".");
  if (!payload) return opaqueFallback;
  try {
    const claims = JSON.parse(Buffer.from(payload, "base64url").toString("utf8")) as { exp?: unknown };
    return typeof claims.exp === "number"
      ? Math.max(Date.now(), claims.exp * 1000 - TOKEN_REFRESH_LEAD_MS)
      : opaqueFallback;
  } catch {
    return opaqueFallback;
  }
}

function getLiteLLMApiKey(credentials: OAuthCredentials): string {
  return credentials.access;
}

function resolveOAuthApiKey(credentials: OAuthCredentials): string {
  return credentials.refresh.startsWith("!")
    ? executeApiKeyCommand(credentials.refresh)
    : getLiteLLMApiKey(credentials);
}

async function resolveCredentials({ executeHelpers = true } = {}): Promise<ResolvedCredentials> {
  const entry = await readAuthEntry();
  const envBase = cleanConfig(process.env[ENV_BASE_URL]);
  const envKey = cleanConfig(process.env[ENV_API_KEY]);
  const envHelperCommand = getApiKeyHelperCommand();
  const useGcloudToken = isGcloudTokenAuthEnabled();
  const authBase = entry?.type === "oauth" ? entry.baseUrl?.trim() : undefined;
  const gcloudCacheKey = useGcloudToken && !entry ? ((await getGcloudTokenCacheKey()) ?? undefined) : undefined;
  const authKey =
    entry?.type === "oauth"
      ? (executeHelpers ? resolveOAuthApiKey(entry) : entry.access).trim()
      : entry?.type === "api_key"
        ? (await AuthStorage.create(getAuthPath()).getApiKey(PROVIDER_NAME, { includeFallback: false }))?.trim()
        : undefined;
  const gcloudKey = executeHelpers && gcloudCacheKey ? (await getGcloudToken())?.trim() : undefined;
  const helperKey =
    !authKey && !gcloudKey && executeHelpers && envHelperCommand ? executeApiKeyCommand(envHelperCommand) : undefined;
  const apiKey = authKey || gcloudKey || helperKey || envKey;

  let apiKeyFingerprint: string | undefined;
  let apiKeyConfig: string | undefined;
  if (entry?.type === "oauth" && entry.refresh.startsWith("!")) {
    apiKeyFingerprint = fingerprint(entry.refresh);
  } else if (authKey) {
    apiKeyFingerprint = fingerprint(authKey);
  } else if (gcloudKey) {
    apiKeyFingerprint = fingerprint(gcloudCacheKey ?? gcloudKey);
    apiKeyConfig = getGcloudTokenCommand();
  } else if (!executeHelpers && gcloudCacheKey) {
    apiKeyFingerprint = fingerprint(gcloudCacheKey);
    apiKeyConfig = getGcloudTokenCommand();
  } else if (helperKey && envHelperCommand) {
    apiKeyFingerprint = fingerprint(envHelperCommand);
    apiKeyConfig = envHelperCommand;
  } else if (!executeHelpers && envHelperCommand) {
    apiKeyFingerprint = fingerprint(envHelperCommand);
    apiKeyConfig = envHelperCommand;
  } else if (envKey) {
    apiKeyFingerprint = fingerprint(envKey);
    apiKeyConfig = `$${ENV_API_KEY}`;
  }
  const rawBase = authBase || envBase;
  return {
    baseUrl: rawBase ? normalizeBaseUrl(rawBase) : undefined,
    apiKey: apiKey || undefined,
    apiKeyFingerprint,
    apiKeyConfig,
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
  const apiKeyInput = (await callbacks.onPrompt({ message: "Enter API key:" })).trim();
  if (!rawBaseUrl || !apiKeyInput) throw new Error("Both base URL and API key are required");

  const baseUrl = normalizeBaseUrl(rawBaseUrl);
  const refresh = apiKeyInput.startsWith("!") ? apiKeyInput : "";
  const apiKey = refresh ? executeApiKeyCommand(refresh) : apiKeyInput;
  const { models, source } = await discoverModels(baseUrl, apiKey, {
    timeoutMs: LOGIN_TIMEOUT_MS,
    signal: callbacks.signal,
  });

  const cache: CacheFile = {
    baseUrl,
    apiKeyFingerprint: fingerprint(refresh || apiKey),
    fetchedAt: Date.now(),
    source,
    models,
  };
  await writeCache(getCachePath(), cache);
  onCacheWrite?.(cache);
  callbacks.onProgress?.(`LiteLLM: ${models.length} models discovered (source: ${source})`);

  return {
    access: apiKey,
    refresh,
    expires: tokenExpiresAt(apiKey, refresh ? EXPIRE_TOKEN_IMMEDIATELY : PERMANENT_TOKEN_EXPIRES_AT),
    baseUrl,
  } as OAuthCredentials & { baseUrl: string };
}

async function refreshLiteLLM(credentials: OAuthCredentials): Promise<OAuthCredentials> {
  if (!credentials.refresh.startsWith("!")) return credentials;
  const access = executeApiKeyCommand(credentials.refresh);
  return { ...credentials, access, expires: tokenExpiresAt(access, EXPIRE_TOKEN_IMMEDIATELY) };
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
  let creds = await resolveCredentials({ executeHelpers: false });
  const cache = await readCache(getCachePath());
  let fp = creds.apiKeyFingerprint;
  let cacheFetchedAt = cache?.fetchedAt ?? 0;

  const cacheValid =
    cache !== null &&
    creds.baseUrl !== undefined &&
    fp !== undefined &&
    cache.baseUrl === creds.baseUrl &&
    cache.apiKeyFingerprint === fp;

  let models: ProviderModelConfig[] = cacheValid && cache ? cache.models : [];
  const shouldFetch =
    creds.baseUrl !== undefined &&
    fp !== undefined &&
    !isOffline() &&
    getDiscoveryTimeoutMs() > 0 &&
    (!cacheValid || isListModelsMode());

  let credentialWarning: string | undefined;
  let liveDiscoveryApiKey: string | undefined;
  if (shouldFetch) {
    try {
      creds = await resolveCredentials();
      fp = creds.apiKeyFingerprint;
    } catch (error) {
      if (!cacheValid || !cache) throw error;
      credentialWarning = error instanceof Error ? error.message : String(error);
      process.stderr.write(`LiteLLM: discovery failed (${credentialWarning}); using cached models.\n`);
      models = cache.models;
    }
  }

  if (shouldFetch && !credentialWarning && creds.baseUrl && creds.apiKey && fp) {
    const timeoutMs = getDiscoveryTimeoutMs();
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
      liveDiscoveryApiKey = creds.apiKey;
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

  let updateCosts: (models: ProviderModelConfig[]) => void = () => undefined;

  const oauth = {
    name: "LiteLLM",
    login: (callbacks: OAuthLoginCallbacks) =>
      loginLiteLLM(callbacks, (next) => {
        cacheFetchedAt = next.fetchedAt;
        registerProvider(next.baseUrl, next.models);
        updateCosts(next.models);
      }),
    refreshToken: refreshLiteLLM,
    getApiKey: getLiteLLMApiKey,
    modifyModels: modifyLiteLLMModels,
  };

  function registerProvider(
    baseUrl: string | undefined,
    models: ProviderModelConfig[],
    apiKeyConfig = creds.apiKeyConfig ?? getApiKeyHelperCommand() ?? `$${ENV_API_KEY}`,
  ): void {
    pi.registerProvider(PROVIDER_NAME, {
      baseUrl: baseUrl ? `${baseUrl}/v1` : "https://litellm.example.com/v1",
      // When LITELLM_API_KEY_HELPER is set we register the helper as a `!command` provider key.
      // Pi's per-request auth path (ModelRegistry.getApiKeyAndHeaders) resolves provider keys via
      // resolveConfigValueOrThrow -> resolveConfigValueUncached, i.e. the command is re-executed on
      // every request (it does NOT use the process-lifetime command cache, which only applies to
      // resolveConfigValue). So a short-lived/rotating helper token stays fresh. The OAuth hooks
      // remain registered for `/login litellm` users. See the regression test
      // "re-runs the helper command on every request" in tests/index.test.ts.
      apiKey: apiKeyConfig,
      api: "openai-completions",
      models,
      oauth,
    });
  }

  registerProvider(creds.baseUrl, models);

  updateCosts = setupLiteLLMCostTracking(pi, models);

  let refreshInProgress: Promise<RefreshResult> | null = null;

  function discoveryDisabledReason(): string | null {
    if (isOffline()) return `${ENV_OFFLINE}=1`;
    if (getDiscoveryTimeoutMs() === 0) return `${ENV_TIMEOUT}=0`;
    return null;
  }

  async function resolveRuntimeApiKey(): Promise<string> {
    const fresh = await resolveCredentials();
    if (!fresh.apiKey) throw new Error("no credentials. Run /login litellm or set env vars.");
    return fresh.apiKey;
  }

  function registerSkillTools(baseUrl: string | undefined): void {
    if (!baseUrl) return;
    for (const tool of createSkillToolDefinitions(baseUrl, resolveRuntimeApiKey)) {
      pi.registerTool(tool);
    }
  }

  function seededRuntimeApiKey(seed: string): () => Promise<string> {
    let first: string | undefined = seed;
    return async () => {
      if (first) {
        const value = first;
        first = undefined;
        return value;
      }
      return resolveRuntimeApiKey();
    };
  }

  async function registerMcpTools(baseUrl: string | undefined, discoveryApiKey: string | undefined): Promise<void> {
    if (!baseUrl || !discoveryApiKey || discoveryDisabledReason()) return;
    try {
      const tools = await createMcpToolDefinitions(baseUrl, seededRuntimeApiKey(discoveryApiKey));
      for (const tool of tools) {
        pi.registerTool(tool);
      }
    } catch (error) {
      process.stderr.write(
        `LiteLLM: MCP tool discovery failed (${error instanceof Error ? error.message : String(error)}).\n`,
      );
    }
  }

  registerSkillTools(creds.baseUrl);
  await registerMcpTools(creds.baseUrl, liveDiscoveryApiKey);

  async function refreshModelsAndCosts(): Promise<RefreshResult> {
    const fresh = await resolveCredentials();
    const freshFp = fresh.apiKeyFingerprint;
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
    registerProvider(fresh.baseUrl, result.models, fresh.apiKeyConfig);
    updateCosts(result.models);
    cacheFetchedAt = now;
    await registerMcpTools(fresh.baseUrl, fresh.apiKey);
    return { models: result.models, source: result.source };
  }

  function runRefresh(): Promise<RefreshResult> {
    refreshInProgress ??= refreshModelsAndCosts().finally(() => {
      refreshInProgress = null;
    });
    return refreshInProgress;
  }

  type LoginContext = Pick<ExtensionContext, "modelRegistry" | "signal" | "ui">;

  async function runLogin(ctx: LoginContext): Promise<void> {
    const credential = await loginLiteLLM(
      {
        onAuth: () => undefined,
        onDeviceCode: () => undefined,
        onPrompt: async ({ message, placeholder }) => {
          const value = await ctx.ui.input(message, placeholder);
          if (value === undefined) throw new Error("Login cancelled");
          return value;
        },
        onProgress: (message) => ctx.ui.notify(message, "info"),
        onSelect: async () => undefined,
        signal: ctx.signal,
      },
      (next) => {
        cacheFetchedAt = next.fetchedAt;
        registerProvider(next.baseUrl, next.models);
        updateCosts(next.models);
      },
    );

    ctx.modelRegistry.authStorage.set(PROVIDER_NAME, { type: "oauth", ...credential });
    ctx.modelRegistry.refresh();
    const credentialBaseUrl = (credential as { baseUrl?: string }).baseUrl;
    const credentialAccess = typeof credential.access === "string" ? credential.access : undefined;
    registerSkillTools(credentialBaseUrl);
    await registerMcpTools(credentialBaseUrl, credentialAccess);
    ctx.ui.notify(`Logged in to LiteLLM. Credentials saved to ${getAuthPath()}`, "info");
  }

  pi.on("input", async (event, ctx) => {
    if (event.text.trim() !== `/login ${PROVIDER_NAME}`) return { action: "continue" };
    if (!ctx.hasUI) return { action: "continue" };

    try {
      await runLogin(ctx);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (message !== "Login cancelled") ctx.ui.notify(`LiteLLM login failed: ${message}`, "error");
    }

    return { action: "handled" };
  });

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

  pi.on("before_agent_start", async (event) => {
    if (discoveryDisabledReason()) return;
    const fresh = await resolveCredentials();
    if (!fresh.baseUrl || !fresh.apiKey) return;
    const skills = await listSkills(fresh.baseUrl, fresh.apiKey);
    const section = createSkillsPromptSection(skills);
    if (!section) return;
    return { systemPrompt: `${event.systemPrompt}\n\n${section}` };
  });

  pi.on("message_end", (event) => {
    if (event.message.role !== "assistant") return;
    const message = normalizeThinkTags(event.message as AssistantMessage);
    if (!message) return;
    return { message };
  });
}
