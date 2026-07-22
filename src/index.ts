import { execSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import type {
  Api,
  ApiKeyCredential,
  AssistantMessage,
  AuthInteraction,
  Credential,
  OAuthCredential,
  OAuthCredentials,
  ProviderAuth,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import { setupLiteLLMCostTracking } from "./cost.js";
import { discoverModels, isGpt55Model, normalizeBaseUrl, shouldSuppressReasoningContent } from "./discover.js";
import {
  getGcloudToken,
  getGcloudTokenCacheKey,
  getGcloudTokenCommand,
  isGcloudTokenAuthEnabled,
} from "./gcloud-token.js";
import { getSessionIdFromFile } from "./litellm.js";
import { createMcpToolDefinitions } from "./mcp-tools.js";
import { createLiteLLMProvider } from "./provider.js";
import { createSkillsPromptSection, createSkillToolDefinitions, listSkills } from "./skills.js";
import type { DiscoveryOptions, LiteLLMRuntimeAuth, ResolvedCredentials } from "./types.js";

const PROVIDER_NAME = "litellm";
const SETTINGS_KEY = "litellm";
const ENV_BASE_URL = "LITELLM_BASE_URL";
const ENV_API_KEY = "LITELLM_API_KEY";
const ENV_API_KEY_HELPER = "LITELLM_API_KEY_HELPER";
const ENV_HEADERS = "LITELLM_HEADERS";
const ENV_TIMEOUT = "LITELLM_DISCOVERY_TIMEOUT_MS";
const ENV_OFFLINE = "LITELLM_OFFLINE";
const ENV_VERBOSE_DISCOVERY = "LITELLM_VERBOSE_DISCOVERY";
const ENV_MODELS_DEV = "LITELLM_MODELS_DEV";
const DEFAULT_TIMEOUT_MS = 5000;
const LOGIN_TIMEOUT_MS = 10_000;
const MODELS_DEV_CACHE_FILENAME = "litellm-models-dev.json";
const TOKEN_REFRESH_LEAD_MS = 5 * 60 * 1000;
const PERMANENT_TOKEN_EXPIRES_AT = Number.MAX_SAFE_INTEGER;
const EXPIRE_TOKEN_IMMEDIATELY = 0;

type RawProviderSettings = {
  displayName?: unknown;
  baseUrl?: unknown;
  apiKey?: unknown;
  headers?: unknown;
  enabled?: unknown;
};

type ProviderDefinition = {
  name: string;
  displayName: string;
  baseUrl?: string;
  apiKeyConfig?: string;
  headers?: unknown;
  useDefaultEnv: boolean;
  useGcloudTokenAuth: boolean;
  enableOAuth: boolean;
};

function getModelsDevDiscoveryOptions(): Pick<DiscoveryOptions, "modelsDev" | "modelsDevCachePath"> {
  return {
    modelsDev: process.env[ENV_MODELS_DEV] !== "0",
    modelsDevCachePath: join(getAgentDir(), MODELS_DEV_CACHE_FILENAME),
  };
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

async function readGlobalLiteLLMSettings(): Promise<Record<string, unknown> | undefined> {
  try {
    const raw = await readFile(join(getAgentDir(), "settings.json"), "utf8");
    const parsed = JSON.parse(raw) as Record<string, unknown>;
    const settings = parsed[SETTINGS_KEY];
    return settings && typeof settings === "object" && !Array.isArray(settings)
      ? (settings as Record<string, unknown>)
      : undefined;
  } catch {
    return undefined;
  }
}

function cleanConfig(raw: string | undefined): string | undefined {
  const trimmed = raw?.trim();
  return trimmed && trimmed !== "undefined" ? trimmed : undefined;
}

function stringSetting(value: unknown): string | undefined {
  return typeof value === "string" ? cleanConfig(value) : undefined;
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

async function generateVirtualKey(
  baseUrl: string,
  userToken: string,
  signal?: AbortSignal,
  headers?: Record<string, string>,
): Promise<{ key: string; expiresAt?: number }> {
  const boundedSignal = signal
    ? AbortSignal.any([signal, AbortSignal.timeout(LOGIN_TIMEOUT_MS)])
    : AbortSignal.timeout(LOGIN_TIMEOUT_MS);
  const response = await fetch(`${baseUrl}/key/generate`, {
    method: "POST",
    headers: {
      ...headers,
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
}

function resolveOAuthApiKey(credentials: OAuthCredentials): string {
  return credentials.refresh.startsWith("!") ? executeApiKeyCommand(credentials.refresh) : credentials.access;
}

function resolveTemplateConfigValue(config: string): string | undefined {
  let resolved = "";
  for (let index = 0; index < config.length; ) {
    const dollarIndex = config.indexOf("$", index);
    if (dollarIndex === -1) return resolved + config.slice(index);
    resolved += config.slice(index, dollarIndex);
    const nextChar = config[dollarIndex + 1];
    if (nextChar === "$" || nextChar === "!") {
      resolved += nextChar;
      index = dollarIndex + 2;
      continue;
    }
    if (nextChar === "{") {
      const endIndex = config.indexOf("}", dollarIndex + 2);
      if (endIndex === -1) {
        resolved += "$";
        index = dollarIndex + 1;
        continue;
      }
      const name = config.slice(dollarIndex + 2, endIndex);
      const envValue = process.env[name];
      if (envValue === undefined) return undefined;
      resolved += envValue;
      index = endIndex + 1;
      continue;
    }
    const match = config.slice(dollarIndex + 1).match(/^[A-Za-z_][A-Za-z0-9_]*/);
    if (!match) {
      resolved += "$";
      index = dollarIndex + 1;
      continue;
    }
    const envValue = process.env[match[0]];
    if (envValue === undefined) return undefined;
    resolved += envValue;
    index = dollarIndex + 1 + match[0].length;
  }
  return resolved;
}

async function resolveTemplateConfigValueFromContext(
  config: string,
  env: (name: string) => Promise<string | undefined>,
): Promise<string | undefined> {
  let resolved = "";
  for (let index = 0; index < config.length; ) {
    const dollarIndex = config.indexOf("$", index);
    if (dollarIndex === -1) return resolved + config.slice(index);
    resolved += config.slice(index, dollarIndex);
    const nextChar = config[dollarIndex + 1];
    if (nextChar === "$" || nextChar === "!") {
      resolved += nextChar;
      index = dollarIndex + 2;
      continue;
    }
    if (nextChar === "{") {
      const endIndex = config.indexOf("}", dollarIndex + 2);
      if (endIndex === -1) {
        resolved += "$";
        index = dollarIndex + 1;
        continue;
      }
      const envValue = await env(config.slice(dollarIndex + 2, endIndex));
      if (envValue === undefined) return undefined;
      resolved += envValue;
      index = endIndex + 1;
      continue;
    }
    const match = config.slice(dollarIndex + 1).match(/^[A-Za-z_][A-Za-z0-9_]*/);
    if (!match) {
      resolved += "$";
      index = dollarIndex + 1;
      continue;
    }
    const envValue = await env(match[0]);
    if (envValue === undefined) return undefined;
    resolved += envValue;
    index = dollarIndex + 1 + match[0].length;
  }
  return resolved;
}

function resolveConfigValue(config: string, { executeCommands }: { executeCommands: boolean }): string | undefined {
  if (config.startsWith("!")) return executeCommands ? executeApiKeyCommand(config) : undefined;
  return resolveTemplateConfigValue(config);
}

const warnedUnresolvedApiKeys = new Set<string>();

function warnUnresolvedApiKeyConfig(providerName: string, config: string): void {
  const key = `${providerName} ${config}`;
  if (warnedUnresolvedApiKeys.has(key)) return;
  warnedUnresolvedApiKeys.add(key);
  process.stderr.write(
    `LiteLLM (${providerName}): configured apiKey did not resolve (unset environment variable?); use $$ for a literal $.\n`,
  );
}

function parseHeaderRecord(value: unknown): Record<string, string> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const headers: Record<string, string> = {};
  for (const [key, raw] of Object.entries(value as Record<string, unknown>)) {
    if (!key.trim()) continue;
    let resolved: string | undefined;
    if (typeof raw === "string") resolved = resolveTemplateConfigValue(raw);
    else if (typeof raw === "number" || typeof raw === "boolean") resolved = String(raw);
    else {
      process.stderr.write(`LiteLLM: ignoring non-primitive header value for "${key}".\n`);
      continue;
    }
    if (resolved) headers[key] = resolved;
  }
  return Object.keys(headers).length > 0 ? headers : undefined;
}

function parseCustomHeaders(raw: string | undefined): Record<string, string> | undefined {
  const trimmed = cleanConfig(raw);
  if (!trimmed) return undefined;
  try {
    return parseHeaderRecord(JSON.parse(trimmed));
  } catch (error) {
    process.stderr.write(
      `LiteLLM: failed to parse custom headers (${error instanceof Error ? error.message : String(error)}).\n`,
    );
    return undefined;
  }
}

function resolveHeaders(definition: ProviderDefinition): Record<string, string> | undefined {
  if (typeof definition.headers === "string") return parseCustomHeaders(resolveTemplateConfigValue(definition.headers));
  return parseHeaderRecord(definition.headers);
}

async function resolveCredentials(
  definition: ProviderDefinition,
  { executeHelpers = true } = {},
): Promise<ResolvedCredentials> {
  const configuredBase =
    cleanConfig(definition.baseUrl) ?? (definition.useDefaultEnv ? cleanConfig(process.env[ENV_BASE_URL]) : undefined);
  const envKey = definition.useDefaultEnv ? cleanConfig(process.env[ENV_API_KEY]) : undefined;
  const envHelperCommand = definition.useDefaultEnv ? getApiKeyHelperCommand() : undefined;
  const useGcloudToken = definition.useGcloudTokenAuth && isGcloudTokenAuthEnabled();
  const gcloudCacheKey = useGcloudToken ? ((await getGcloudTokenCacheKey()) ?? undefined) : undefined;
  const gcloudKey = executeHelpers && gcloudCacheKey ? (await getGcloudToken())?.trim() : undefined;
  // Resolved lazily so a `!command` key is not executed when a
  // higher-precedence credential (saved auth, gcloud token) already won.
  let configuredKey: string | undefined;
  if (!gcloudKey && definition.apiKeyConfig) {
    configuredKey = resolveConfigValue(definition.apiKeyConfig, { executeCommands: executeHelpers });
    if (configuredKey === undefined && !definition.apiKeyConfig.startsWith("!")) {
      warnUnresolvedApiKeyConfig(definition.name, definition.apiKeyConfig);
    }
  }
  const helperKey =
    !gcloudKey && !configuredKey && executeHelpers && envHelperCommand
      ? executeApiKeyCommand(envHelperCommand)
      : undefined;
  const apiKey = gcloudKey || configuredKey || helperKey || envKey;

  let apiKeyConfig: string | undefined;
  if (gcloudKey) {
    apiKeyConfig = getGcloudTokenCommand();
  } else if (!executeHelpers && gcloudCacheKey) {
    apiKeyConfig = getGcloudTokenCommand();
  } else if (configuredKey && definition.apiKeyConfig) {
    apiKeyConfig = definition.apiKeyConfig;
  } else if (!executeHelpers && definition.apiKeyConfig?.startsWith("!")) {
    apiKeyConfig = definition.apiKeyConfig;
  } else if (helperKey && envHelperCommand) {
    apiKeyConfig = envHelperCommand;
  } else if (!executeHelpers && envHelperCommand) {
    apiKeyConfig = envHelperCommand;
  } else if (envKey) {
    apiKeyConfig = `$${ENV_API_KEY}`;
  }
  return {
    baseUrl: configuredBase ? normalizeBaseUrl(configuredBase) : undefined,
    apiKey: apiKey || undefined,
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

function isVerboseDiscovery(): boolean {
  return process.env[ENV_VERBOSE_DISCOVERY] === "1";
}

function normalizeProviderSettings(raw: unknown): RawProviderSettings | undefined {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return undefined;
  const record = raw as RawProviderSettings;
  if (record.enabled === false) return undefined;
  return record;
}

function isFeatureEnabled(settings: Record<string, unknown> | undefined, feature: "skills" | "mcp"): boolean {
  const raw = settings?.[feature];
  return !isPlainObject(raw) || raw.enabled !== false;
}

function getProviderDefinitions(settings: Record<string, unknown> | undefined): ProviderDefinition[] {
  const rawProviders = settings?.providers && typeof settings.providers === "object" ? settings.providers : undefined;
  const providerSettings = rawProviders as Record<string, unknown> | undefined;
  const defaultSettings = normalizeProviderSettings(providerSettings?.[PROVIDER_NAME]);

  const makeDefinition = (
    name: string,
    raw: RawProviderSettings | undefined,
    isDefault: boolean,
  ): ProviderDefinition => ({
    name,
    displayName: stringSetting(raw?.displayName) ?? (isDefault ? "LiteLLM" : name),
    baseUrl: stringSetting(raw?.baseUrl),
    apiKeyConfig: stringSetting(raw?.apiKey),
    headers: raw?.headers ?? (isDefault ? `$${ENV_HEADERS}` : undefined),
    useDefaultEnv: isDefault,
    useGcloudTokenAuth: isDefault,
    enableOAuth: isDefault,
  });

  const definitions = [makeDefinition(PROVIDER_NAME, defaultSettings, true)];
  for (const [name, raw] of Object.entries(providerSettings ?? {})) {
    if (name === PROVIDER_NAME) continue;
    const normalized = normalizeProviderSettings(raw);
    if (!normalized) continue;
    definitions.push(makeDefinition(name, normalized, false));
  }
  return definitions;
}

async function loginApiKey(interaction: AuthInteraction): Promise<ApiKeyCredential> {
  const rawBaseUrl = (
    await interaction.prompt({
      type: "text",
      message: "Enter LiteLLM proxy URL (no trailing /v1):",
      placeholder: "https://litellm.example.com",
    })
  ).trim();
  if (!rawBaseUrl) throw new Error("Base URL is required");
  const key = (await interaction.prompt({ type: "secret", message: "Enter API key:" })).trim();
  if (!key) throw new Error("Both base URL and API key are required");
  return { type: "api_key", key, env: { [ENV_BASE_URL]: normalizeBaseUrl(rawBaseUrl) } };
}

async function loginOAuth(interaction: AuthInteraction, headers?: Record<string, string>): Promise<OAuthCredential> {
  const rawBaseUrl = (
    await interaction.prompt({
      type: "text",
      message: "Enter LiteLLM proxy URL (no trailing /v1):",
      placeholder: "https://litellm.example.com",
    })
  ).trim();
  if (!rawBaseUrl) throw new Error("Base URL is required");
  const baseUrl = normalizeBaseUrl(rawBaseUrl);
  interaction.notify({
    type: "auth_url",
    url: `${baseUrl}/sso/key/generate`,
    instructions: "Authenticate via SSO, then copy your token from the LiteLLM UI.",
  });
  const rawToken = (await interaction.prompt({ type: "secret", message: "Paste your SSO token from the LiteLLM UI:" }))
    .trim()
    .replace(/^Bearer\s+/i, "")
    .trim();
  if (!rawToken) throw new Error("SSO token is required");
  const wantVirtualKey = (
    await interaction.prompt({ type: "text", message: "Generate a LiteLLM virtual key from this token? (y/n):" })
  )
    .trim()
    .toLowerCase();
  let access = rawToken;
  let expires = tokenExpiresAt(rawToken, PERMANENT_TOKEN_EXPIRES_AT);
  if (wantVirtualKey !== "n" && wantVirtualKey !== "no") {
    try {
      interaction.notify({ type: "progress", message: "Generating virtual key..." });
      const generated = await generateVirtualKey(baseUrl, rawToken, interaction.signal, headers);
      access = generated.key;
      expires =
        generated.expiresAt === undefined
          ? PERMANENT_TOKEN_EXPIRES_AT
          : Math.max(Date.now(), generated.expiresAt - TOKEN_REFRESH_LEAD_MS);
      interaction.notify({ type: "progress", message: "Virtual key generated and will be used for API calls." });
    } catch (error) {
      if (interaction.signal?.aborted) throw interaction.signal.reason;
      const message = error instanceof Error ? error.message : String(error);
      interaction.notify({
        type: "progress",
        message: `LiteLLM: virtual key generation failed (${message}); using SSO token directly.`,
      });
    }
  }
  return { type: "oauth", access, refresh: "", expires, baseUrl };
}

async function refreshLiteLLM(credentials: OAuthCredentials, _signal?: AbortSignal): Promise<OAuthCredentials> {
  if (!credentials.refresh.startsWith("!")) {
    if (credentials.expires < PERMANENT_TOKEN_EXPIRES_AT) {
      throw new Error("LiteLLM credential cannot be refreshed; run /login litellm again");
    }
    return credentials;
  }
  const access = executeApiKeyCommand(credentials.refresh);
  return { ...credentials, access, expires: tokenExpiresAt(access, EXPIRE_TOKEN_IMMEDIATELY) };
}

async function resolveApiKeyAuth(
  definition: ProviderDefinition,
  ctx: { env(name: string): Promise<string | undefined> },
  credential?: ApiKeyCredential,
  executeHelpers = true,
) {
  const baseUrl =
    cleanConfig(credential?.env?.[ENV_BASE_URL]) ??
    cleanConfig(definition.baseUrl) ??
    (definition.useDefaultEnv ? cleanConfig(await ctx.env(ENV_BASE_URL)) : undefined);
  const stored = credential?.key
    ? resolveConfigValue(credential.key, { executeCommands: executeHelpers })?.trim()
    : undefined;
  let source: string | undefined;
  let creds: ResolvedCredentials;
  if (stored) {
    source = "stored credential";
    creds = {
      baseUrl: baseUrl ? normalizeBaseUrl(baseUrl) : undefined,
      apiKey: stored,
    };
  } else {
    creds = await resolveCredentials(
      { ...definition, apiKeyConfig: undefined, useDefaultEnv: false },
      { executeHelpers },
    );
    if (!creds.apiKey && definition.apiKeyConfig) {
      const configured = definition.apiKeyConfig.startsWith("!")
        ? executeHelpers
          ? executeApiKeyCommand(definition.apiKeyConfig)
          : undefined
        : await resolveTemplateConfigValueFromContext(definition.apiKeyConfig, ctx.env);
      if (configured) {
        creds.apiKey = configured;
        creds.apiKeyConfig = definition.apiKeyConfig;
      } else if (!definition.apiKeyConfig.startsWith("!")) {
        warnUnresolvedApiKeyConfig(definition.name, definition.apiKeyConfig);
      }
    }
    if (!creds.apiKey && definition.useDefaultEnv) {
      const helper = normalizeCommand(await ctx.env(ENV_API_KEY_HELPER));
      if (helper) {
        creds.apiKey = executeHelpers ? executeApiKeyCommand(helper) : undefined;
        creds.apiKeyConfig = helper;
        source = ENV_API_KEY_HELPER;
      } else {
        const envKey = cleanConfig(await ctx.env(ENV_API_KEY));
        if (envKey) {
          creds.apiKey = envKey;
          creds.apiKeyConfig = ENV_API_KEY;
          source = ENV_API_KEY;
        }
      }
    }
    if (!creds.baseUrl && baseUrl) creds.baseUrl = normalizeBaseUrl(baseUrl);
  }
  if (!creds.apiKey) return undefined;
  return {
    auth: {
      apiKey: creds.apiKey,
      baseUrl: creds.baseUrl ? `${creds.baseUrl}/v1` : undefined,
      headers: resolveHeaders(definition),
    },
    env: baseUrl ? { [ENV_BASE_URL]: normalizeBaseUrl(baseUrl) } : undefined,
    source: source ?? creds.apiKeyConfig ?? ENV_API_KEY,
  };
}

function createProviderAuth(definition: ProviderDefinition): ProviderAuth {
  return {
    apiKey: {
      name: `${definition.displayName} API key`,
      login: definition.name === PROVIDER_NAME ? loginApiKey : undefined,
      check: async ({ ctx, credential }) => {
        const baseUrl =
          credential?.env?.[ENV_BASE_URL] ??
          definition.baseUrl ??
          (definition.useDefaultEnv ? await ctx.env(ENV_BASE_URL) : undefined);
        if (!cleanConfig(baseUrl)) return undefined;
        if (credential?.key) return { type: "api_key", source: "stored credential" };
        const configured = await resolveCredentials(
          { ...definition, apiKeyConfig: undefined, useDefaultEnv: false },
          { executeHelpers: false },
        );
        if (configured.apiKey || configured.apiKeyConfig)
          return {
            type: "api_key",
            source:
              definition.useGcloudTokenAuth && isGcloudTokenAuthEnabled()
                ? "gcloud ADC"
                : (configured.apiKeyConfig ?? ENV_API_KEY),
          };
        if (definition.apiKeyConfig) {
          const configuredKey = definition.apiKeyConfig.startsWith("!")
            ? definition.apiKeyConfig
            : await resolveTemplateConfigValueFromContext(definition.apiKeyConfig, ctx.env);
          if (configuredKey) return { type: "api_key", source: definition.apiKeyConfig };
        }
        if (definition.useDefaultEnv && cleanConfig(await ctx.env(ENV_API_KEY_HELPER)))
          return { type: "api_key", source: ENV_API_KEY_HELPER };
        return definition.useDefaultEnv && cleanConfig(await ctx.env(ENV_API_KEY))
          ? { type: "api_key", source: ENV_API_KEY }
          : undefined;
      },
      resolve: ({ ctx, credential }) => resolveApiKeyAuth(definition, ctx, credential),
    },
    oauth: definition.enableOAuth
      ? {
          name: "LiteLLM SSO",
          loginLabel: "Sign in with LiteLLM SSO",
          login: (interaction) => loginOAuth(interaction, resolveHeaders(definition)),
          refresh: async (credential, signal) => ({
            ...(await refreshLiteLLM(credential, signal)),
            type: "oauth" as const,
          }),
          toAuth: async (credential) => ({
            apiKey: credential.access,
            baseUrl: credential.baseUrl ? `${normalizeBaseUrl(String(credential.baseUrl))}/v1` : undefined,
            headers: resolveHeaders(definition),
          }),
        }
      : undefined,
  };
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
  api: Api | undefined,
  sessionId: string | undefined,
): Record<string, unknown> | undefined {
  let next: Record<string, unknown> | undefined;
  const update = (key: string, value: unknown): void => {
    if (payload[key] !== undefined) return;
    next ??= { ...payload };
    next[key] = value;
  };

  if (api !== "openai-responses" && modelId && shouldSuppressReasoningContent(modelId)) {
    for (const [key, value] of Object.entries(REASONING_SUPPRESSION_DEFAULTS)) update(key, value);
  }

  // LiteLLM still routes gpt-5.5 tool+reasoning requests through chat completions.
  // Drop reasoning until the gateway honors /v1/responses for this route.
  if (
    api !== "openai-responses" &&
    modelId &&
    isGpt55Model(modelId) &&
    Array.isArray(payload.tools) &&
    payload.tools.length > 0
  ) {
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

function normalizeThinkTags(
  message: AssistantMessage,
  litellmProviderNames: Set<string>,
): AssistantMessage | undefined {
  if (!litellmProviderNames.has(message.provider) || !shouldSuppressReasoningContent(message.model)) return;

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
  const settings = await readGlobalLiteLLMSettings();
  const definitions = getProviderDefinitions(settings);
  const skillsEnabled = isFeatureEnabled(settings, "skills");
  const mcpEnabled = isFeatureEnabled(settings, "mcp");
  const providerNames = new Set(definitions.map((definition) => definition.name));

  function discoveryDisabledReason(): string | null {
    if (isOffline()) return `${ENV_OFFLINE}=1`;
    if (getDiscoveryTimeoutMs() === 0) return `${ENV_TIMEOUT}=0`;
    return null;
  }

  function requestBaseUrl(definition: ProviderDefinition): string {
    const baseUrl =
      cleanConfig(definition.baseUrl) ??
      (definition.useDefaultEnv ? cleanConfig(process.env[ENV_BASE_URL]) : undefined) ??
      "https://litellm.example.com";
    return `${normalizeBaseUrl(baseUrl)}/v1`;
  }

  async function authForCredential(definition: ProviderDefinition, credential: Credential) {
    if (credential.type === "oauth") {
      const baseUrl =
        typeof credential.baseUrl === "string"
          ? normalizeBaseUrl(credential.baseUrl)
          : normalizeBaseUrl(requestBaseUrl(definition));
      return { baseUrl, apiKey: resolveOAuthApiKey(credential), headers: resolveHeaders(definition) };
    }
    const resolved = await resolveApiKeyAuth(definition, { env: async (name) => process.env[name] }, credential);
    if (!resolved?.auth.apiKey) {
      throw new Error(`no credentials for ${definition.name}. Run /login litellm or set env vars.`);
    }
    return {
      baseUrl: normalizeBaseUrl(resolved.auth.baseUrl ?? requestBaseUrl(definition)),
      apiKey: resolved.auth.apiKey,
      headers: resolved.auth.headers,
    };
  }

  let mcpRegistered = false;
  let defaultRuntimeAuth: LiteLLMRuntimeAuth | undefined;

  async function resolveDefaultRuntimeAuth(): Promise<LiteLLMRuntimeAuth> {
    if (!defaultRuntimeAuth) throw new Error("no credentials for litellm. Run /login litellm or set env vars.");
    return defaultRuntimeAuth;
  }

  async function registerMcpTools(signal?: AbortSignal): Promise<void> {
    if (!mcpEnabled || mcpRegistered || discoveryDisabledReason()) return;
    try {
      signal?.throwIfAborted();
      const tools = await createMcpToolDefinitions(
        resolveDefaultRuntimeAuth,
        isVerboseDiscovery() ? (message) => process.stderr.write(`LiteLLM MCP: ${message}\n`) : undefined,
        signal,
      );
      signal?.throwIfAborted();
      for (const tool of tools) pi.registerTool(tool);
      mcpRegistered = true;
    } catch (error) {
      if (signal?.aborted) throw signal.reason;
      process.stderr.write(
        `LiteLLM (${PROVIDER_NAME}): MCP tool discovery failed (${error instanceof Error ? error.message : String(error)}).\n`,
      );
    }
  }

  const controllers = new Map(
    definitions.map((definition) => {
      const controller = createLiteLLMProvider({
        id: definition.name,
        name: definition.displayName,
        baseUrl: requestBaseUrl(definition),
        auth: createProviderAuth(definition),
        discover: async (credential, signal) => {
          const disabledReason = discoveryDisabledReason();
          if (disabledReason) throw new Error(`discovery disabled (${disabledReason})`);
          const auth = await authForCredential(definition, credential);
          const result = await discoverModels(auth.baseUrl, auth.apiKey, {
            ...getModelsDevDiscoveryOptions(),
            timeoutMs: getDiscoveryTimeoutMs(),
            signal,
            headers: auth.headers,
            silent: !isVerboseDiscovery(),
            onProgress: isVerboseDiscovery() ? (message) => process.stderr.write(`LiteLLM: ${message}\n`) : undefined,
          });
          signal?.throwIfAborted();
          return result;
        },
      });
      Object.assign(controller.provider, { headers: resolveHeaders(definition) });
      const refreshModels = controller.provider.refreshModels!;
      controller.provider.refreshModels = async (context) => {
        try {
          await refreshModels(context);
        } finally {
          if (
            definition.name === PROVIDER_NAME &&
            context.allowNetwork &&
            !discoveryDisabledReason() &&
            context.credential
          ) {
            defaultRuntimeAuth = await authForCredential(definition, context.credential);
            await registerMcpTools(context.signal);
          }
        }
      };
      pi.registerProvider(controller.provider);
      return [definition.name, controller] as const;
    }),
  );

  setupLiteLLMCostTracking(pi, [...controllers.keys()]);

  if (skillsEnabled) {
    for (const tool of createSkillToolDefinitions(resolveDefaultRuntimeAuth)) {
      pi.registerTool(tool);
    }
  }

  pi.registerCommand("litellm-refresh", {
    description: "Re-discover models from the LiteLLM proxy.",
    handler: async (args, ctx) => {
      const disabledReason = discoveryDisabledReason();
      if (disabledReason) {
        ctx.ui.notify(`LiteLLM refresh disabled (${disabledReason})`, "warning");
        return;
      }
      const requestedProvider = args.trim();
      const entries = requestedProvider
        ? [...controllers].filter(([name]) => name === requestedProvider)
        : [...controllers];
      if (entries.length === 0) {
        ctx.ui.notify(`LiteLLM refresh failed: unknown provider ${requestedProvider}`, "error");
        return;
      }

      const settled = await Promise.allSettled(
        entries.map(async ([providerName, controller]) => {
          const result = await controller.forceRefresh(ctx.signal);
          return { providerName, models: controller.provider.getModels(), source: result.source };
        }),
      );
      const succeeded = settled.filter((result) => result.status === "fulfilled").map((result) => result.value);
      const failed = settled
        .map((result, index) => ({ result, name: entries[index][0] }))
        .filter(({ result }) => result.status === "rejected")
        .map(({ result, name }) => {
          const reason = (result as PromiseRejectedResult).reason;
          return { name, message: reason instanceof Error ? reason.message : String(reason) };
        });

      if (failed.length === 0) {
        if (succeeded.length === 1) {
          const result = succeeded[0];
          ctx.ui.notify(`LiteLLM: ${result.models.length} models refreshed (source: ${result.source})`, "info");
          return;
        }
        ctx.ui.notify(
          `LiteLLM: ${succeeded.length} providers refreshed (${succeeded
            .map((result) => `${result.providerName}: ${result.models.length} models`)
            .join(", ")})`,
          "info",
        );
        return;
      }
      const failures = failed.map(({ name, message }) => (settled.length === 1 ? message : `${name}: ${message}`));
      if (succeeded.length === 0) {
        ctx.ui.notify(`LiteLLM refresh failed: ${failures.join("; ")}`, "error");
        return;
      }
      ctx.ui.notify(
        `LiteLLM: refreshed ${succeeded
          .map((result) => `${result.providerName}: ${result.models.length} models`)
          .join(", ")}; failed ${failures.join("; ")}`,
        "warning",
      );
    },
  });

  let sessionId: string | undefined;
  pi.on("session_start", (_event, ctx) => {
    sessionId = getSessionIdFromFile(ctx.sessionManager.getSessionFile());
  });

  pi.on("before_provider_request", (event, ctx) => {
    if (!ctx.model?.provider || !providerNames.has(ctx.model.provider)) return;
    if (typeof event.payload !== "object" || event.payload === null) return;
    return prepareLiteLLMRequestPayload(
      event.payload as Record<string, unknown>,
      ctx.model?.id,
      ctx.model?.api,
      sessionId,
    );
  });

  pi.on("before_agent_start", async (event, ctx) => {
    defaultRuntimeAuth = undefined;
    if (!skillsEnabled || discoveryDisabledReason()) return;
    const auth = await ctx.modelRegistry.getProviderAuth(PROVIDER_NAME);
    const provider = ctx.modelRegistry.getProvider(PROVIDER_NAME);
    const baseUrl = auth?.auth.baseUrl ?? provider?.baseUrl;
    const apiKey = auth?.auth.apiKey;
    if (!baseUrl || !apiKey) return;
    const headers = Object.fromEntries(
      Object.entries(auth.auth.headers ?? provider?.headers ?? {}).filter(
        (entry): entry is [string, string] => typeof entry[1] === "string",
      ),
    );
    defaultRuntimeAuth = {
      baseUrl: normalizeBaseUrl(baseUrl),
      apiKey,
      headers: Object.keys(headers).length > 0 ? headers : undefined,
    };
    const skills = await listSkills(defaultRuntimeAuth.baseUrl, defaultRuntimeAuth.apiKey, defaultRuntimeAuth.headers);
    const section = createSkillsPromptSection(skills);
    if (!section) return;
    return { systemPrompt: `${event.systemPrompt}\n\n${section}` };
  });

  pi.on("message_end", (event) => {
    if (event.message.role !== "assistant") return;
    const message = normalizeThinkTags(event.message as AssistantMessage, providerNames);
    if (!message) return;
    return { message };
  });
}
