import { execSync, spawn } from "node:child_process";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import type { Api, AssistantMessage, Model, OAuthCredentials, OAuthLoginCallbacks } from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext, ProviderModelConfig } from "@earendil-works/pi-coding-agent";
import { AuthStorage, getAgentDir } from "@earendil-works/pi-coding-agent";
import { fingerprint, readCache, writeCache } from "./cache.js";
import { setupLiteLLMCostTracking } from "./cost.js";
import {
  discoverModels,
  isGpt55Model,
  normalizeBaseUrl,
  shouldSuppressReasoningContent,
  withTimeout,
} from "./discover.js";
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

type ModelOverride = Partial<
  Pick<
    ProviderModelConfig,
    "name" | "reasoning" | "thinkingLevelMap" | "input" | "contextWindow" | "maxTokens" | "headers" | "compat"
  >
> & {
  cost?: Partial<ProviderModelConfig["cost"]>;
};

type RefreshResult = { models: ProviderModelConfig[]; source: string };

function getAuthPath(): string {
  return join(getAgentDir(), "auth.json");
}

function getCachePath(): string {
  return join(getAgentDir(), CACHE_FILENAME);
}

// Same tolerance as pi core's models.json loader (stripJsonComments in dist/utils/json.js):
// strip `//` line comments and trailing commas, leaving string literals untouched.
function stripJsonComments(input: string): string {
  return input
    .replace(/"(?:\\.|[^"\\])*"|\/\/[^\n]*/g, (m) => (m[0] === '"' ? m : ""))
    .replace(/"(?:\\.|[^"\\])*"|,(\s*[}\]])/g, (m, tail: string | undefined) => tail ?? m);
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// Mirrors pi core's ModelOverrideSchema: core rejects the whole models.json on invalid values,
// so anything dropped here would also have been flagged for built-in providers.
const MODEL_OVERRIDE_VALIDATORS: Record<keyof ModelOverride, (value: unknown) => boolean> = {
  name: (value) => typeof value === "string",
  reasoning: (value) => typeof value === "boolean",
  thinkingLevelMap: isPlainObject,
  input: Array.isArray,
  contextWindow: (value) => typeof value === "number",
  maxTokens: (value) => typeof value === "number",
  headers: isPlainObject,
  compat: isPlainObject,
  cost: (value) => isPlainObject(value) && Object.values(value).every((entry) => typeof entry === "number"),
};

function sanitizeModelOverride(modelId: string, raw: unknown): ModelOverride | undefined {
  if (!isPlainObject(raw)) {
    process.stderr.write(`LiteLLM: ignoring model override for ${modelId} in models.json (not an object).\n`);
    return undefined;
  }
  const override: Record<string, unknown> = {};
  const dropped: string[] = [];
  for (const [key, value] of Object.entries(raw)) {
    const isValid = MODEL_OVERRIDE_VALIDATORS[key as keyof ModelOverride];
    if (isValid?.(value)) override[key] = value;
    else dropped.push(key);
  }
  if (dropped.length > 0) {
    process.stderr.write(
      `LiteLLM: ignoring invalid model override field(s) for ${modelId} in models.json: ${dropped.join(", ")}.\n`,
    );
  }
  return override as ModelOverride;
}

async function readModelOverrides(): Promise<Map<string, ModelOverride>> {
  let raw: string;
  try {
    raw = await readFile(join(getAgentDir(), "models.json"), "utf8");
  } catch {
    return new Map();
  }
  try {
    const config = JSON.parse(stripJsonComments(raw)) as {
      providers?: Record<string, { modelOverrides?: Record<string, unknown> }>;
    };
    const overrides = new Map<string, ModelOverride>();
    for (const [id, rawOverride] of Object.entries(config.providers?.[PROVIDER_NAME]?.modelOverrides ?? {})) {
      const override = sanitizeModelOverride(id, rawOverride);
      if (override) overrides.set(id, override);
    }
    return overrides;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    process.stderr.write(`LiteLLM: ignoring model overrides (failed to parse models.json: ${message}).\n`);
    return new Map();
  }
}

// Merge semantics must match pi core's applyModelOverride/mergeCompat (dist/core/model-registry.js)
// so the same models.json entry behaves identically for litellm and built-in providers.
function mergeCompat(
  base: ProviderModelConfig["compat"],
  override: ProviderModelConfig["compat"],
): ProviderModelConfig["compat"] {
  if (!override) return base;
  const merged = { ...base, ...override } as Record<string, unknown>;
  for (const key of ["openRouterRouting", "vercelGatewayRouting", "chatTemplateKwargs"]) {
    const baseValue = (base as Record<string, unknown> | undefined)?.[key];
    const overrideValue = (override as Record<string, unknown>)[key];
    if (baseValue || overrideValue) {
      merged[key] = { ...(baseValue as object | undefined), ...(overrideValue as object | undefined) };
    }
  }
  return merged as ProviderModelConfig["compat"];
}

function applyModelOverride(model: ProviderModelConfig, override: ModelOverride): ProviderModelConfig {
  const next = { ...model };
  if (override.name !== undefined) next.name = override.name;
  if (override.reasoning !== undefined) next.reasoning = override.reasoning;
  if (override.thinkingLevelMap !== undefined) {
    next.thinkingLevelMap = { ...model.thinkingLevelMap, ...override.thinkingLevelMap };
  }
  if (override.input !== undefined) next.input = override.input;
  if (override.contextWindow !== undefined) next.contextWindow = override.contextWindow;
  if (override.maxTokens !== undefined) next.maxTokens = override.maxTokens;
  if (override.headers !== undefined) next.headers = override.headers;
  if (override.compat !== undefined) next.compat = mergeCompat(model.compat, override.compat);
  if (override.cost !== undefined) next.cost = { ...model.cost, ...override.cost };
  return next;
}

function applyModelOverrides(
  models: ProviderModelConfig[],
  overrides: Map<string, ModelOverride>,
): ProviderModelConfig[] {
  if (overrides.size === 0) return models;
  return models.map((model) => {
    const override = overrides.get(model.id);
    return override ? applyModelOverride(model, override) : model;
  });
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

function openInBrowser(url: string): void {
  // Never invoke a shell here: cmd.exe re-parses metacharacters before `start`
  // runs, which would make URL contents injectable. Launch is best-effort; the
  // URL is also shown to the user, so launcher failures must not crash the process.
  const [cmd, args]: [string, string[]] =
    process.platform === "darwin"
      ? ["open", [url]]
      : process.platform === "win32"
        ? ["rundll32", ["url.dll,FileProtocolHandler", url]]
        : ["xdg-open", [url]];
  spawn(cmd, args, { stdio: "ignore", detached: true })
    .on("error", () => undefined)
    .unref();
}

async function generateVirtualKey(
  baseUrl: string,
  userToken: string,
  signal?: AbortSignal,
): Promise<{ key: string; expiresAt?: number }> {
  const { signal: boundedSignal, cancel } = withTimeout(LOGIN_TIMEOUT_MS, signal);
  try {
    const response = await fetch(`${baseUrl}/key/generate`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${userToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({}),
      signal: boundedSignal,
    });
    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new Error(`Virtual key generation failed (${response.status}): ${text}`);
    }
    const data = (await response.json()) as { key?: unknown; expires?: unknown };
    if (typeof data.key !== "string" || !data.key) throw new Error("No key in response from /key/generate");
    const expiresMs = typeof data.expires === "string" ? Date.parse(data.expires) : Number.NaN;
    return { key: data.key, expiresAt: Number.isNaN(expiresMs) ? undefined : expiresMs };
  } finally {
    cancel();
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
  if (!rawBaseUrl) throw new Error("Base URL is required");

  const baseUrl = normalizeBaseUrl(rawBaseUrl);
  const method = (
    await callbacks.onPrompt({
      message: "Select login method (1 = API key / !command, 2 = SSO / Enterprise JWT):",
    })
  ).trim();

  let apiKey: string;
  let refresh: string;
  let expires: number;

  if (method === "2") {
    callbacks.onAuth?.({
      url: `${baseUrl}/sso/key/generate`,
      instructions: "Authenticate via SSO, then copy your token from the LiteLLM UI.",
    });
    const rawToken = (await callbacks.onPrompt({ message: "Paste your SSO token from the LiteLLM UI:" }))
      .trim()
      .replace(/^Bearer\s+/i, "")
      .trim();
    if (!rawToken) throw new Error("SSO token is required");

    const wantVirtualKey = (
      await callbacks.onPrompt({ message: "Generate a LiteLLM virtual key from this token? (y/n):" })
    )
      .trim()
      .toLowerCase();

    if (wantVirtualKey !== "n" && wantVirtualKey !== "no") {
      try {
        callbacks.onProgress?.("Generating virtual key...");
        const generated = await generateVirtualKey(baseUrl, rawToken, callbacks.signal);
        apiKey = generated.key;
        refresh = "";
        expires =
          generated.expiresAt === undefined
            ? PERMANENT_TOKEN_EXPIRES_AT
            : Math.max(Date.now(), generated.expiresAt - TOKEN_REFRESH_LEAD_MS);
        callbacks.onProgress?.("Virtual key generated and will be used for API calls.");
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        callbacks.onProgress?.(`LiteLLM: virtual key generation failed (${message}); using SSO token directly.`);
        apiKey = rawToken;
        refresh = "";
        expires = tokenExpiresAt(rawToken, PERMANENT_TOKEN_EXPIRES_AT);
      }
    } else {
      apiKey = rawToken;
      refresh = "";
      expires = tokenExpiresAt(rawToken, PERMANENT_TOKEN_EXPIRES_AT);
    }
  } else {
    const apiKeyInput = (await callbacks.onPrompt({ message: "Enter API key:" })).trim();
    if (!apiKeyInput) throw new Error("Both base URL and API key are required");
    refresh = apiKeyInput.startsWith("!") ? apiKeyInput : "";
    apiKey = refresh ? executeApiKeyCommand(refresh) : apiKeyInput;
    expires = tokenExpiresAt(apiKey, refresh ? EXPIRE_TOKEN_IMMEDIATELY : PERMANENT_TOKEN_EXPIRES_AT);
  }

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
    expires,
    baseUrl,
  } as OAuthCredentials & { baseUrl: string };
}

async function refreshLiteLLM(credentials: OAuthCredentials): Promise<OAuthCredentials> {
  if (!credentials.refresh.startsWith("!")) {
    if (credentials.expires < PERMANENT_TOKEN_EXPIRES_AT) {
      throw new Error("LiteLLM credential cannot be refreshed; run /login litellm again");
    }
    return credentials;
  }
  const access = executeApiKeyCommand(credentials.refresh);
  return { ...credentials, access, expires: tokenExpiresAt(access, EXPIRE_TOKEN_IMMEDIATELY) };
}

function modifyLiteLLMModels(models: Model<Api>[], cred: OAuthCredentials): Model<Api>[] {
  const baseUrl = (cred as { baseUrl?: string }).baseUrl;
  if (!baseUrl) return models;
  return models.map((m) => (m.provider === PROVIDER_NAME ? { ...m, baseUrl: `${baseUrl}/v1` } : m));
}

function isReasoningItem(item: unknown): boolean {
  return typeof item === "object" && item !== null && (item as { type?: unknown }).type === "reasoning";
}

// Reasoning fields LiteLLM forwards to chat-completions providers. The Moonshot
// path defaults them off; the gpt-5.5 tool path strips them entirely.
const REASONING_SUPPRESSION_DEFAULTS: Record<string, unknown> = {
  include_reasoning: false,
  reasoning_content: false,
  merge_reasoning_content_in_choices: true,
  thinking: { type: "disabled" },
};

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
    for (const [key, value] of Object.entries(REASONING_SUPPRESSION_DEFAULTS)) update(key, value);
  }

  // LiteLLM still routes gpt-5.5 tool+reasoning requests through chat completions.
  // Drop reasoning until the gateway honors /v1/responses for this route.
  if (modelId && isGpt55Model(modelId) && Array.isArray(payload.tools) && payload.tools.length > 0) {
    const reasoningKeys = ["reasoning", "reasoning_effort", ...Object.keys(REASONING_SUPPRESSION_DEFAULTS)];
    for (const key of reasoningKeys) {
      if (payload[key] === undefined) continue;
      next ??= { ...payload };
      delete next[key];
    }
    const include = (next ?? payload).include;
    if (Array.isArray(include) && include.includes("reasoning.encrypted_content")) {
      next ??= { ...payload };
      const filteredInclude = include.filter((value) => value !== "reasoning.encrypted_content");
      if (filteredInclude.length === 0) delete next.include;
      else next.include = filteredInclude;
    }
    // Prior turns may have replayed reasoning items (with encrypted_content)
    // into the input; they are rejected once reasoning is stripped.
    const input = (next ?? payload).input;
    if (Array.isArray(input) && input.some(isReasoningItem)) {
      next ??= { ...payload };
      next.input = input.filter((item) => !isReasoningItem(item));
    }
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
  const modelOverrides = await readModelOverrides();
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
        updateCosts(applyModelOverrides(next.models, modelOverrides));
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
    const registeredModels = applyModelOverrides(models, modelOverrides);
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
      models: registeredModels,
      oauth,
    });
  }

  registerProvider(creds.baseUrl, models);

  updateCosts = setupLiteLLMCostTracking(pi, applyModelOverrides(models, modelOverrides));

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
    updateCosts(applyModelOverrides(result.models, modelOverrides));
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
        onAuth: ({ url, instructions }) => {
          openInBrowser(url);
          ctx.ui.notify(instructions ? `${url} — ${instructions}` : url, "info");
        },
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
        updateCosts(applyModelOverrides(next.models, modelOverrides));
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
