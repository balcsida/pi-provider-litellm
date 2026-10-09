import { execFileSync } from "node:child_process";
import { createHash, createHmac, randomBytes, timingSafeEqual } from "node:crypto";
import { readFile } from "node:fs/promises";
import { createServer } from "node:http";
import { join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import type {
  ApiKeyCredential,
  AssistantMessage,
  AuthInteraction,
  Credential,
  Model,
  OAuthCredential,
  OAuthCredentials,
  ProviderAuth,
} from "@earendil-works/pi-ai";
import type { ExtensionAPI, ExtensionContext, McpServerConfig } from "@earendil-works/pi-coding-agent";
// A namespace import, so a Pi without `VERSION` reaches the version check instead of failing to link.
import * as piCodingAgent from "@earendil-works/pi-coding-agent";
import { getAgentDir, readStoredCredential } from "@earendil-works/pi-coding-agent";
import { budgetDisplaySetting, setupLiteLLMBudget } from "./budget.js";
import { setupLiteLLMCostTracking } from "./cost.js";
import { discoverModels, isGpt6OrNewerModel, isGpt55OrNewerModel, normalizeBaseUrl } from "./discover.js";
import { getGcloudToken, hasGcloudAdcCredentials, isGcloudTokenAuthEnabled } from "./gcloud-token.js";
import { createLiteLLMProvider, DEFAULT_LITELLM_BASE_URL, isPlaceholderHost, toNativeModels } from "./provider.js";
import { parseProxyVersion, proxyVersionAtLeast } from "./proxy-version.js";
import { createSkillsPromptSection, createSkillToolDefinitions, listSkills } from "./skills.js";
import type {
  DiscoveryOptions,
  LiteLLMApi,
  LiteLLMModel,
  LiteLLMModelPolicy,
  LiteLLMRuntimeAuth,
  ResolvedCredentials,
} from "./types.js";

const PROVIDER_NAME = "litellm";
const SETTINGS_KEY = "litellm";
const ENV_BASE_URL = "LITELLM_BASE_URL";
const ENV_API_KEY = "LITELLM_API_KEY";
const ENV_DISPLAY_NAME = "LITELLM_DISPLAY_NAME";
const ENV_PROVIDERS = "LITELLM_PROVIDERS";
const ENV_PROVIDERS_JSON = "LITELLM_PROVIDERS_JSON";
const GCLOUD_ADC_SOURCE = "gcloud ADC";
const ENV_API_KEY_HELPER = "LITELLM_API_KEY_HELPER";
const ENV_HEADERS = "LITELLM_HEADERS";
const ENV_TIMEOUT = "LITELLM_DISCOVERY_TIMEOUT_MS";
const ENV_CLI_JWT_EXPIRATION_HOURS = "LITELLM_CLI_JWT_EXPIRATION_HOURS";
const ENV_OFFLINE = "LITELLM_OFFLINE";
const ENV_VERBOSE_DISCOVERY = "LITELLM_VERBOSE_DISCOVERY";
const ENV_MODELS_DEV = "LITELLM_MODELS_DEV";
const MODELS_DEV_CACHE_FILENAME = "litellm-models-dev.json";
const DEFAULT_TIMEOUT_MS = 5000;
const SEED_TIMEOUT_MS = 3000;
const LOGIN_TIMEOUT_MS = 10_000;
const CLI_SSO_POLL_INTERVAL_MS = 2_000;
const CLI_SSO_EXPIRES_IN_SECONDS = 600;
const CLI_AUTH_DISCOVERY_PATH = "/.well-known/litellm-cli-auth";
const PKCE_CALLBACK_TIMEOUT_MS = 10 * 60 * 1000;
const PKCE_FLOW = "litellm_cli_pkce";
// Persisted in auth.json; never rename.
const OIDC_FLOW = "oidc_pkce";
const OIDC_DISCOVERY_PATH = "/.well-known/openid-configuration";
const DEFAULT_CLI_JWT_EXPIRATION_HOURS = 24;
const TOKEN_REFRESH_LEAD_MS = 5 * 60 * 1000;
const PERMANENT_TOKEN_EXPIRES_AT = Number.MAX_SAFE_INTEGER;
const EXPIRE_TOKEN_IMMEDIATELY = 0;
const PKCE_TRANSIENT_REFRESH_BACKOFF_MS = 5_000;

type RawProviderSettings = {
  displayName?: unknown;
  baseUrl?: unknown;
  apiKey?: unknown;
  headers?: unknown;
  enabled?: unknown;
  allowInsecureHttp?: unknown;
  useGcloudTokenAuth?: unknown;
  enableOAuth?: unknown;
  oidc?: unknown;
  enableMcp?: unknown;
  mcp?: unknown;
};

export type ProviderDefinition = {
  name: string;
  displayName: string;
  baseUrl?: string;
  apiKeyConfig?: string;
  headers?: unknown;
  useDefaultEnv: boolean;
  useGcloudTokenAuth: boolean;
  enableOAuth: boolean;
  allowInsecureHttp: boolean;
  /** Raw `oidc` setting, validated at login so a bad value never breaks startup. */
  oidc?: unknown;
  envPrefix?: string;
  enableMcp?: boolean;
};

const RESERVED_PROVIDER_NAMES = new Set(["mcp", "skills", "codemode", "tool_search", "litellm"]);
const PROVIDER_NAME_REGEX = /^[a-z0-9][a-z0-9_-]{0,63}$/;
const PROVIDER_ENV_VAR_REGEX =
  /^LITELLM_PROVIDER_([A-Z0-9_]+?)_(BASE_URL|API_KEY_HELPER|API_KEY|HEADERS|DISPLAY_NAME|NAME|ALLOW_INSECURE_HTTP|USE_GCLOUD_AUTH|ENABLE_OAUTH|ENABLE_MCP|MCP_ENABLED)$/;

export function isValidProviderName(name: string): boolean {
  if (typeof name !== "string") return false;
  if (!PROVIDER_NAME_REGEX.test(name)) return false;
  if (RESERVED_PROVIDER_NAMES.has(name)) return false;
  return true;
}

export function parseCanonicalProviderList(raw: string | undefined): string[] {
  if (typeof raw !== "string") return [];
  const trimmed = raw.trim();
  if (!trimmed) return [];
  return trimmed
    .split(/[\s,]+/)
    .map((item) => item.trim())
    .filter(Boolean);
}

export function parseProvidersJson(raw: string | undefined): Record<string, RawProviderSettings> | undefined {
  if (typeof raw !== "string") return undefined;
  const trimmed = raw.trim();
  if (!trimmed) return undefined;
  try {
    const parsed = JSON.parse(trimmed);
    if (!isPlainObject(parsed)) return undefined;
    const result: Record<string, RawProviderSettings> = {};
    for (const [key, value] of Object.entries(parsed)) {
      if (isPlainObject(value)) {
        result[key] = value as RawProviderSettings;
      }
    }
    return Object.keys(result).length > 0 ? result : undefined;
  } catch {
    return undefined;
  }
}

function parseBooleanSetting(value: unknown): boolean {
  if (typeof value === "boolean") return value;
  if (typeof value === "string") {
    const trimmed = value.trim().toLowerCase();
    return trimmed === "1" || trimmed === "true";
  }
  return false;
}

export type ParsedProviderEnv = {
  token: string;
  envPrefix: string;
  name?: string;
  displayName?: string;
  baseUrl?: string;
  apiKey?: string;
  apiKeyHelper?: string;
  headers?: unknown;
  allowInsecureHttp?: boolean;
  useGcloudTokenAuth?: boolean;
  enableOAuth?: boolean;
  enableMcp?: boolean;
};

export function parseProviderEnvVars(env: NodeJS.ProcessEnv): Record<string, ParsedProviderEnv> {
  const result: Record<string, ParsedProviderEnv> = {};
  for (const key of Object.keys(env)) {
    const match = key.match(PROVIDER_ENV_VAR_REGEX);
    if (!match) continue;
    const token = match[1]!;
    const field = match[2]!;
    const val = env[key];
    if (val === undefined) continue;

    if (!result[token]) {
      result[token] = {
        token,
        envPrefix: `LITELLM_PROVIDER_${token}`,
      };
    }
    const entry = result[token]!;
    switch (field) {
      case "BASE_URL":
        entry.baseUrl = cleanConfig(val);
        break;
      case "API_KEY":
        entry.apiKey = cleanConfig(val);
        break;
      case "API_KEY_HELPER":
        entry.apiKeyHelper = cleanConfig(val);
        break;
      case "HEADERS":
        entry.headers = cleanConfig(val);
        break;
      case "DISPLAY_NAME":
        entry.displayName = cleanConfig(val);
        break;
      case "NAME":
        entry.name = cleanConfig(val);
        break;
      case "ALLOW_INSECURE_HTTP":
        entry.allowInsecureHttp = parseBooleanSetting(val);
        break;
      case "USE_GCLOUD_AUTH":
        entry.useGcloudTokenAuth = parseBooleanSetting(val);
        break;
      case "ENABLE_OAUTH":
        entry.enableOAuth = parseBooleanSetting(val);
        break;
      case "ENABLE_MCP":
      case "MCP_ENABLED":
        entry.enableMcp = parseBooleanSetting(val);
        break;
    }
  }
  return result;
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

function resolveCredentialRoot(
  definition: ProviderDefinition,
  credential?: Credential,
  requestBaseUrl?: string,
): string | undefined {
  const credentialBaseUrl =
    credential?.type === "oauth"
      ? cleanConfig(typeof credential.baseUrl === "string" ? credential.baseUrl : undefined)
      : credential?.type === "api_key"
        ? cleanConfig(credential.env?.[ENV_BASE_URL])
        : undefined;
  const baseUrl =
    credentialBaseUrl ??
    cleanConfig(requestBaseUrl) ??
    cleanConfig(definition.baseUrl) ??
    (definition.envPrefix ? cleanConfig(process.env[`${definition.envPrefix}_BASE_URL`]) : undefined) ??
    (definition.useDefaultEnv ? cleanConfig(process.env[ENV_BASE_URL]) : undefined);
  return baseUrl ? normalizeBaseUrl(baseUrl, definition.allowInsecureHttp) : undefined;
}

function requireCredentialRoot(root: string | undefined, providerName: string): string {
  if (!root) throw new Error(`no LiteLLM base URL for ${providerName}. Run /login litellm or set env vars.`);
  if (isPlaceholderHost(new URL(root).hostname)) {
    throw new Error(`placeholder LiteLLM base URL for ${providerName}. Run /login litellm or set env vars.`);
  }
  return root;
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

function parseApiKeyCommand(commandConfig: string): string[] {
  const command = commandConfig.startsWith("!") ? commandConfig.slice(1) : commandConfig;
  if (/[\n\r;&|<>`$(){}*?~%^!]|\[|\]/.test(command))
    throw new Error("LiteLLM API key helper shell syntax is not supported");

  const parts: string[] = [];
  let current = "";
  let quote: "'" | '"' | undefined;
  for (let index = 0; index < command.length; index += 1) {
    const character = command[index]!;
    const next = command[index + 1];
    if (character === "\\" && quote !== "'" && next && (next === "\\" || next === quote || /\s/.test(next))) {
      current += next;
      index += 1;
    } else if ((character === "'" || character === '"') && (!quote || quote === character)) {
      quote = quote ? undefined : character;
    } else if (!quote && /\s/.test(character)) {
      if (current) {
        parts.push(current);
        current = "";
      }
    } else {
      current += character;
    }
  }

  if (quote) throw new Error(`LiteLLM API key helper command has an unterminated quote: ${command}`);
  if (current) parts.push(current);
  if (parts.length === 0) throw new Error("LiteLLM API key helper command is empty");
  if (/\.(?:cmd|bat)$/i.test(parts[0]!))
    throw new Error("LiteLLM API key helper shell scripts are not supported; use an executable");
  return parts;
}

function executeApiKeyCommand(commandConfig: string): string {
  const [command, ...args] = parseApiKeyCommand(commandConfig);
  const output = execFileSync(command, args, {
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
    timeout: 10_000,
  }).trim();
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

function configuredCliJwtExpiresAt(): number {
  const configured = Number(process.env[ENV_CLI_JWT_EXPIRATION_HOURS]);
  const hours = Number.isFinite(configured) && configured > 0 ? configured : DEFAULT_CLI_JWT_EXPIRATION_HOURS;
  return Date.now() + hours * 60 * 60 * 1_000;
}

function boundedLoginSignal(signal?: AbortSignal): AbortSignal {
  return signal
    ? AbortSignal.any([signal, AbortSignal.timeout(LOGIN_TIMEOUT_MS)])
    : AbortSignal.timeout(LOGIN_TIMEOUT_MS);
}

async function generateVirtualKey(
  baseUrl: string,
  userToken: string,
  signal?: AbortSignal,
  headers?: Record<string, string>,
): Promise<{ key: string; expiresAt?: number }> {
  const response = await fetch(`${baseUrl}/key/generate`, {
    method: "POST",
    headers: {
      ...headers,
      Authorization: `Bearer ${userToken}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({}),
    signal: boundedLoginSignal(signal),
  });
  if (!response.ok) {
    throw new Error(`Virtual key generation failed (${response.status})`);
  }
  const data = (await response.json()) as { key?: unknown; expires?: unknown };
  if (typeof data.key !== "string" || !data.key) throw new Error("No key in response from /key/generate");
  const expiresMs = typeof data.expires === "string" ? Date.parse(data.expires) : Number.NaN;
  return { key: data.key, expiresAt: Number.isNaN(expiresMs) ? undefined : expiresMs };
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
  const rawHeaders =
    definition.headers ??
    (definition.envPrefix ? process.env[`${definition.envPrefix}_HEADERS`] : undefined) ??
    (definition.useDefaultEnv ? process.env[ENV_HEADERS] : undefined);
  if (typeof rawHeaders === "string") return parseCustomHeaders(resolveTemplateConfigValue(rawHeaders));
  return parseHeaderRecord(rawHeaders);
}

async function resolveHeadersFromContext(
  definition: ProviderDefinition,
  env: (name: string) => Promise<string | undefined>,
): Promise<Record<string, string> | undefined> {
  const rawHeaders =
    definition.headers ??
    (definition.envPrefix ? await env(`${definition.envPrefix}_HEADERS`) : undefined) ??
    (definition.useDefaultEnv ? await env(ENV_HEADERS) : undefined);
  if (typeof rawHeaders === "string")
    return parseCustomHeaders(await resolveTemplateConfigValueFromContext(rawHeaders, env));
  return parseHeaderRecord(rawHeaders);
}

export async function resolveCredentials(
  definition: ProviderDefinition,
  { executeHelpers = true } = {},
): Promise<ResolvedCredentials> {
  const configuredBase =
    cleanConfig(definition.baseUrl) ??
    (definition.envPrefix ? cleanConfig(process.env[`${definition.envPrefix}_BASE_URL`]) : undefined) ??
    (definition.useDefaultEnv ? cleanConfig(process.env[ENV_BASE_URL]) : undefined);
  const useGcloudToken = definition.useDefaultEnv
    ? definition.useGcloudTokenAuth && isGcloudTokenAuthEnabled()
    : definition.useGcloudTokenAuth;
  const gcloudKey = executeHelpers && useGcloudToken ? (await getGcloudToken())?.trim() : undefined;
  // Resolved lazily so a `!command` key is not executed when a
  // higher-precedence credential (saved auth, gcloud token) already won.
  let configuredKey: string | undefined;
  if (!gcloudKey && definition.apiKeyConfig) {
    configuredKey = resolveConfigValue(definition.apiKeyConfig, { executeCommands: executeHelpers });
    if (configuredKey === undefined && !definition.apiKeyConfig.startsWith("!")) {
      warnUnresolvedApiKeyConfig(definition.name, definition.apiKeyConfig);
    }
  }

  let helperKey: string | undefined;
  let helperCommand: string | undefined;
  let envKey: string | undefined;
  let envKeySource: string | undefined;

  if (!gcloudKey && !configuredKey && definition.envPrefix) {
    helperCommand = normalizeCommand(process.env[`${definition.envPrefix}_API_KEY_HELPER`]);
    if (helperCommand && executeHelpers) {
      helperKey = executeApiKeyCommand(helperCommand);
    }
    if (!helperCommand) {
      envKey = cleanConfig(process.env[`${definition.envPrefix}_API_KEY`]);
      if (envKey) envKeySource = `$${definition.envPrefix}_API_KEY`;
    }
  }

  if (!gcloudKey && !configuredKey && !helperKey && !helperCommand && !envKey && definition.useDefaultEnv) {
    helperCommand = getApiKeyHelperCommand();
    if (helperCommand && executeHelpers) {
      helperKey = executeApiKeyCommand(helperCommand);
    }
    if (!helperCommand) {
      envKey = cleanConfig(process.env[ENV_API_KEY]);
      if (envKey) envKeySource = `$${ENV_API_KEY}`;
    }
  }

  const apiKey = gcloudKey || configuredKey || helperKey || envKey;

  let apiKeyConfig: string | undefined;
  if (configuredKey && definition.apiKeyConfig) {
    apiKeyConfig = definition.apiKeyConfig;
  } else if (!executeHelpers && definition.apiKeyConfig?.startsWith("!")) {
    apiKeyConfig = definition.apiKeyConfig;
  } else if (helperKey && helperCommand) {
    apiKeyConfig = helperCommand;
  } else if (!executeHelpers && helperCommand) {
    apiKeyConfig = helperCommand;
  } else if (envKey) {
    apiKeyConfig = envKeySource;
  }
  return {
    baseUrl: configuredBase ? normalizeBaseUrl(configuredBase, definition.allowInsecureHttp) : undefined,
    apiKey: apiKey || undefined,
    apiKeyConfig,
    apiKeyFromGcloudAdc: Boolean(gcloudKey),
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

/** Pi disables all model network access when PI_OFFLINE is set; activation must honour it too. */
function isHostOffline(): boolean {
  return process.env.PI_OFFLINE !== undefined;
}

// Pi resolves `$NAME`, `${NAME}`, and a leading `!command` in MCP header values. Header values here
// are already resolved, so they are escaped back to literals: `$$` is a `$`, and `$!` a leading `!`.
function piLiteral(value: string): string {
  const escaped = value.replaceAll("$", () => "$$");
  return escaped.startsWith("!") ? `$${escaped}` : escaped;
}

/**
 * LiteLLM's /model/info is the metadata authority, so models.dev is an opt-in escape hatch.
 * Without a cache path and with refresh suppressed, discovery reads neither the models.dev
 * network nor a cache left by an earlier release; Pi's own catalog still applies.
 */
function modelsDevDiscoveryOptions(): Pick<DiscoveryOptions, "modelsDev" | "modelsDevCachePath"> {
  if (process.env[ENV_MODELS_DEV] !== "1") return { modelsDev: false };
  return { modelsDev: !isHostOffline(), modelsDevCachePath: join(getAgentDir(), MODELS_DEV_CACHE_FILENAME) };
}

/**
 * Absolute budget for activation-time discovery. Pi cannot paint its UI until every extension has
 * activated, so this is deliberately capped rather than following LITELLM_DISCOVERY_TIMEOUT_MS,
 * which is a per-request timeout that deployments raise to tens of seconds. Missing the budget only
 * costs the startup seed; Pi's own refresh and /model still populate the catalog.
 */
function getSeedTimeoutMs(): number {
  return Math.min(getDiscoveryTimeoutMs(), SEED_TIMEOUT_MS);
}

function isVerboseDiscovery(): boolean {
  return process.env[ENV_VERBOSE_DISCOVERY] === "1";
}

function normalizeProviderSettings(raw: unknown): RawProviderSettings | undefined {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return undefined;
  const record = raw as RawProviderSettings;
  if (record.enabled === false) return undefined;
  const enableMcp =
    record.enableMcp !== undefined
      ? parseBooleanSetting(record.enableMcp)
      : isPlainObject(record.mcp)
        ? record.mcp.enabled !== undefined
          ? parseBooleanSetting(record.mcp.enabled)
          : undefined
        : record.mcp !== undefined
          ? parseBooleanSetting(record.mcp)
          : undefined;
  return {
    ...record,
    ...(enableMcp !== undefined ? { enableMcp } : {}),
  };
}

function isFeatureEnabled(
  settings: Record<string, unknown> | undefined,
  feature: "skills" | "mcp" | "budget",
): boolean {
  const raw = settings?.[feature];
  return !isPlainObject(raw) || raw.enabled !== false;
}

type IntermediateProvider = {
  name: string;
  displayName?: string;
  baseUrl?: string;
  apiKeyConfig?: string;
  headers?: unknown;
  allowInsecureHttp?: boolean;
  useGcloudTokenAuth?: boolean;
  enableOAuth?: boolean;
  envPrefix?: string;
  enableMcp?: boolean;
};

export function getProviderDefinitions(settings: Record<string, unknown> | undefined): ProviderDefinition[] {
  const rawProviders = settings?.providers && typeof settings.providers === "object" ? settings.providers : undefined;
  const providerSettings = rawProviders as Record<string, unknown> | undefined;
  const defaultSettings = normalizeProviderSettings(providerSettings?.[PROVIDER_NAME]);
  const defaultDisplayName = cleanConfig(process.env[ENV_DISPLAY_NAME]) ?? "LiteLLM";
  const parsedEnv = parseProviderEnvVars(process.env);
  const primaryEnv = parsedEnv[PROVIDER_NAME.toUpperCase()];

  const primaryDefinition: ProviderDefinition = {
    name: PROVIDER_NAME,
    displayName: stringSetting(defaultSettings?.displayName) ?? defaultDisplayName,
    baseUrl: stringSetting(defaultSettings?.baseUrl),
    apiKeyConfig: stringSetting(defaultSettings?.apiKey),
    headers: defaultSettings?.headers ?? `$${ENV_HEADERS}`,
    useDefaultEnv: true,
    useGcloudTokenAuth: true,
    enableOAuth: true,
    allowInsecureHttp: defaultSettings?.allowInsecureHttp === true,
    oidc: defaultSettings?.oidc,
    enableMcp: typeof defaultSettings?.enableMcp === "boolean" ? defaultSettings.enableMcp : primaryEnv?.enableMcp,
  };

  const secondaryProviders = new Map<string, IntermediateProvider>();
  const disabledProviders = new Set<string>();

  // 1. Incorporate LITELLM_PROVIDERS_JSON
  const jsonProviders = parseProvidersJson(process.env[ENV_PROVIDERS_JSON]);
  if (jsonProviders) {
    for (const [name, raw] of Object.entries(jsonProviders)) {
      if (!isValidProviderName(name)) continue;
      if (raw.enabled === false) {
        disabledProviders.add(name);
        continue;
      }
      const normalized = normalizeProviderSettings(raw);
      if (!normalized) continue;
      secondaryProviders.set(name, {
        name,
        displayName: stringSetting(normalized.displayName),
        baseUrl: stringSetting(normalized.baseUrl),
        apiKeyConfig: stringSetting(normalized.apiKey),
        headers: normalized.headers,
        allowInsecureHttp: parseBooleanSetting(normalized.allowInsecureHttp),
        useGcloudTokenAuth: parseBooleanSetting(normalized.useGcloudTokenAuth),
        enableOAuth: parseBooleanSetting(normalized.enableOAuth),
        enableMcp: typeof normalized.enableMcp === "boolean" ? normalized.enableMcp : undefined,
      });
    }
  }

  // 2. Scan env vars
  const claimedTokens = new Set<string>([PROVIDER_NAME.toUpperCase()]);

  // 3. Incorporate canonical list LITELLM_PROVIDERS
  const canonicalList = parseCanonicalProviderList(process.env[ENV_PROVIDERS]);
  const tokenOwners = new Map<string, string>([[PROVIDER_NAME.toUpperCase(), PROVIDER_NAME]]);
  for (const canonicalId of canonicalList) {
    if (!isValidProviderName(canonicalId)) continue;
    if (disabledProviders.has(canonicalId)) continue;
    const token = canonicalId.toUpperCase().replace(/-/g, "_");
    const owner = tokenOwners.get(token);
    const hasCollision = owner !== undefined && owner !== canonicalId;
    if (!hasCollision && !tokenOwners.has(token)) {
      tokenOwners.set(token, canonicalId);
    }
    const envData = !hasCollision ? parsedEnv[token] : undefined;
    if (envData) claimedTokens.add(token);

    const existing = secondaryProviders.get(canonicalId);
    secondaryProviders.set(canonicalId, {
      name: canonicalId,
      displayName: envData?.displayName ?? existing?.displayName,
      baseUrl: existing?.baseUrl,
      apiKeyConfig: existing?.apiKeyConfig,
      headers: existing?.headers,
      allowInsecureHttp: envData?.allowInsecureHttp ?? existing?.allowInsecureHttp,
      useGcloudTokenAuth: envData?.useGcloudTokenAuth ?? existing?.useGcloudTokenAuth,
      enableOAuth: envData?.enableOAuth ?? existing?.enableOAuth,
      envPrefix: envData?.envPrefix ?? existing?.envPrefix,
      enableMcp: envData?.enableMcp ?? existing?.enableMcp,
    });
  }

  // 4. Incorporate remaining unlisted prefix-scanned tokens
  for (const [token, envData] of Object.entries(parsedEnv)) {
    if (claimedTokens.has(token)) continue;
    const providerId = envData.name ?? token.toLowerCase().replace(/_/g, "-");
    if (!isValidProviderName(providerId)) continue;
    if (disabledProviders.has(providerId) || disabledProviders.has(token.toLowerCase())) continue;

    tokenOwners.set(token, providerId);

    const existing = secondaryProviders.get(providerId);
    secondaryProviders.set(providerId, {
      name: providerId,
      displayName: envData.displayName ?? existing?.displayName,
      baseUrl: existing?.baseUrl,
      apiKeyConfig: existing?.apiKeyConfig,
      headers: existing?.headers,
      allowInsecureHttp: envData.allowInsecureHttp ?? existing?.allowInsecureHttp,
      useGcloudTokenAuth: envData.useGcloudTokenAuth ?? existing?.useGcloudTokenAuth,
      enableOAuth: envData.enableOAuth ?? existing?.enableOAuth,
      envPrefix: envData.envPrefix ?? existing?.envPrefix,
      enableMcp: envData.enableMcp ?? existing?.enableMcp,
    });
  }

  // 5. Merge with settings.json (disk settings override env vars; enabled === false excludes)
  for (const [name, raw] of Object.entries(providerSettings ?? {})) {
    if (name === PROVIDER_NAME) continue;
    if (typeof name !== "string" || name.length === 0 || RESERVED_PROVIDER_NAMES.has(name)) continue;

    const rawRecord = normalizeProviderSettings(raw);

    if (!rawRecord) {
      secondaryProviders.delete(name);
      continue;
    }

    const existing = secondaryProviders.get(name);
    const token = name.toUpperCase().replace(/-/g, "_");
    let matchingEnvPrefix = existing ? existing.envPrefix : undefined;
    if (!existing) {
      const owner = tokenOwners.get(token);
      if ((owner === undefined || owner === name) && !claimedTokens.has(token) && parsedEnv[token]) {
        matchingEnvPrefix = `LITELLM_PROVIDER_${token}`;
        tokenOwners.set(token, name);
        claimedTokens.add(token);
      }
    }

    secondaryProviders.set(name, {
      name,
      displayName: stringSetting(rawRecord.displayName) ?? existing?.displayName,
      baseUrl: stringSetting(rawRecord.baseUrl) ?? existing?.baseUrl,
      apiKeyConfig: stringSetting(rawRecord.apiKey) ?? existing?.apiKeyConfig,
      headers: rawRecord.headers ?? existing?.headers,
      allowInsecureHttp:
        typeof rawRecord.allowInsecureHttp === "boolean"
          ? rawRecord.allowInsecureHttp
          : (existing?.allowInsecureHttp ?? false),
      useGcloudTokenAuth:
        typeof rawRecord.useGcloudTokenAuth === "boolean"
          ? rawRecord.useGcloudTokenAuth
          : (existing?.useGcloudTokenAuth ?? false),
      enableOAuth:
        typeof rawRecord.enableOAuth === "boolean" ? rawRecord.enableOAuth : (existing?.enableOAuth ?? false),
      envPrefix: matchingEnvPrefix,
      enableMcp: typeof rawRecord.enableMcp === "boolean" ? rawRecord.enableMcp : existing?.enableMcp,
    });
  }

  const definitions: ProviderDefinition[] = [primaryDefinition];
  for (const [name, entry] of secondaryProviders) {
    definitions.push({
      name,
      displayName: entry.displayName ?? name,
      baseUrl: entry.baseUrl,
      apiKeyConfig: entry.apiKeyConfig,
      headers: entry.headers,
      useDefaultEnv: false,
      useGcloudTokenAuth: entry.useGcloudTokenAuth ?? false,
      enableOAuth: entry.enableOAuth ?? false,
      allowInsecureHttp: entry.allowInsecureHttp ?? false,
      oidc: undefined,
      envPrefix: entry.envPrefix,
      enableMcp: entry.enableMcp,
    });
  }
  return definitions;
}

const DIFFERENT_BASE_URL = "enter-a-different-url";

/**
 * The proxy URL is the one thing a user has to retype on every re-login, and an expired SSO token is
 * the common reason to re-login. Offer whatever we already know: settings, env, or the credential the
 * dead session was issued against.
 */
function knownBaseUrl(definition: ProviderDefinition): { url: string; source: string } | undefined {
  const stored = readStoredCredential(definition.name, join(getAgentDir(), "auth.json"));
  const candidates: [string | undefined, string][] = [
    [cleanConfig(definition.baseUrl), "configured"],
    [
      definition.envPrefix ? cleanConfig(process.env[`${definition.envPrefix}_BASE_URL`]) : undefined,
      `$${definition.envPrefix}_BASE_URL`,
    ],
    [definition.useDefaultEnv ? cleanConfig(process.env[ENV_BASE_URL]) : undefined, `$${ENV_BASE_URL}`],
    [
      stored?.type === "oauth"
        ? cleanConfig(typeof stored.baseUrl === "string" ? stored.baseUrl : undefined)
        : cleanConfig(stored?.env?.[ENV_BASE_URL]),
      "previous login",
    ],
  ];
  for (const [candidate, source] of candidates) {
    if (!candidate) continue;
    try {
      return { url: normalizeBaseUrl(candidate, definition.allowInsecureHttp), source };
    } catch {
      // A URL we could not use anyway is not worth offering; try the next candidate.
    }
  }
  return undefined;
}

async function promptBaseUrl(interaction: AuthInteraction, definition: ProviderDefinition): Promise<string> {
  const known = knownBaseUrl(definition);
  if (known) {
    const choice = await interaction.prompt({
      type: "select",
      message: "LiteLLM proxy URL:",
      // Pi's login selector renders labels only, so the source has to ride along in the label.
      options: [
        { id: known.url, label: `${known.url} (${known.source})` },
        { id: DIFFERENT_BASE_URL, label: "Enter a different URL…" },
      ],
    });
    if (choice === known.url) return known.url;
  }
  const rawBaseUrl = (
    await interaction.prompt({
      type: "text",
      message: "Enter LiteLLM proxy URL (no trailing /v1):",
      placeholder: "https://litellm.example.com",
    })
  ).trim();
  if (!rawBaseUrl) throw new Error("Base URL is required");
  return normalizeBaseUrl(rawBaseUrl, definition.allowInsecureHttp);
}

async function loginApiKey(interaction: AuthInteraction, definition: ProviderDefinition): Promise<ApiKeyCredential> {
  const baseUrl = await promptBaseUrl(interaction, definition);
  const key = (await interaction.prompt({ type: "secret", message: "Enter API key:" })).trim();
  if (!key) throw new Error("Both base URL and API key are required");
  return { type: "api_key", key, env: { [ENV_BASE_URL]: baseUrl } };
}

type CliSsoStart = { loginId: string; pollSecret: string; userCode: string; expiresInSeconds: number };

type CliAuthDiscovery = {
  issuer: string;
  authorizationEndpoint: string;
  tokenEndpoint: string;
  registrationEndpoint: string;
  resource: string;
};

type PkceCredentials = OAuthCredentials & {
  flow: typeof PKCE_FLOW;
  baseUrl: string;
  clientId: string;
  tokenEndpoint: string;
  resource: string;
  userId?: string;
  teamId?: string;
};

type PkceToken = {
  access: string;
  refresh: string;
  expires: number;
  userId?: string;
  teamId?: string;
};

type TokenFailure = { ok: false; transient: boolean; message: string };

type PkceTokenResult = { ok: true; token: PkceToken } | TokenFailure;

function authRequestHeaders(headers?: Record<string, string>, contentType?: string): Headers {
  const result = new Headers(headers);
  result.set("Accept", "application/json");
  if (contentType) result.set("Content-Type", contentType);
  return result;
}

function canonicalIssuer(value: string): string {
  const url = new URL(value);
  if (url.username || url.password || url.search || url.hash) throw new Error("LiteLLM CLI auth has invalid issuer");
  return `${url.origin}${url.pathname.replace(/\/+$/, "") || "/"}`;
}

function sameOriginUrl(value: unknown, issuer: URL, field: string): string {
  if (typeof value !== "string" || !value.trim()) throw new Error(`LiteLLM CLI auth discovery has invalid ${field}`);
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new Error(`LiteLLM CLI auth discovery has invalid ${field}`);
  }
  if (url.protocol !== issuer.protocol || url.origin !== issuer.origin)
    throw new Error(`LiteLLM CLI auth discovery has cross-origin ${field}`);
  if (url.username || url.password || url.hash) throw new Error(`LiteLLM CLI auth discovery has invalid ${field}`);
  return value.trim();
}

function isAuthToken(value: unknown): value is string {
  return typeof value === "string" && /^[\x21-\x7e]+$/.test(value);
}

/** The OAuth `error` code when it is a plain code; descriptions and other fields are never echoed. */
function oauthErrorCode(value: unknown): string | undefined {
  return typeof value === "string" && /^[a-z_]{1,64}$/.test(value) ? value : undefined;
}

async function readAuthJson(response: Response, signal: AbortSignal | undefined, stage: string): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    if (signal?.aborted) throw signal.reason;
    throw new Error(`${stage} returned invalid JSON`);
  }
}

async function discoverPkce(
  baseUrl: string,
  signal?: AbortSignal,
  headers?: Record<string, string>,
): Promise<CliAuthDiscovery | undefined> {
  canonicalIssuer(baseUrl);
  const response = await fetch(`${baseUrl}${CLI_AUTH_DISCOVERY_PATH}`, {
    headers: authRequestHeaders(headers),
    redirect: "manual",
    signal: boundedLoginSignal(signal),
  });
  if (response.status === 404) return undefined;
  if (!response.ok) throw new Error(`LiteLLM CLI auth discovery failed (HTTP ${response.status})`);
  const data = await readAuthJson(response, signal, "LiteLLM CLI auth discovery");
  if (!isPlainObject(data) || data.contract_version !== 1)
    throw new Error("LiteLLM CLI auth discovery has unsupported contract version");
  if (!Array.isArray(data.code_challenge_methods_supported) || !data.code_challenge_methods_supported.includes("S256"))
    throw new Error("LiteLLM CLI auth discovery does not support S256");
  if (typeof data.issuer !== "string" || !data.issuer.trim())
    throw new Error("LiteLLM CLI auth discovery has invalid issuer");
  let issuer: URL;
  try {
    issuer = new URL(data.issuer);
  } catch {
    throw new Error("LiteLLM CLI auth discovery has invalid issuer");
  }
  if (canonicalIssuer(data.issuer) !== canonicalIssuer(baseUrl))
    throw new Error("LiteLLM CLI auth discovery issuer does not match the proxy URL");
  return {
    issuer: data.issuer,
    authorizationEndpoint: sameOriginUrl(data.authorization_endpoint, issuer, "authorization endpoint"),
    tokenEndpoint: sameOriginUrl(data.token_endpoint, issuer, "token endpoint"),
    registrationEndpoint: sameOriginUrl(data.registration_endpoint, issuer, "registration endpoint"),
    resource: sameOriginUrl(data.resource, issuer, "resource"),
  };
}

/** POSTs a token form without following redirects; each flow validates its own success fields. */
async function postTokenForm(
  endpoint: string,
  form: URLSearchParams,
  signal: AbortSignal | undefined,
  headers: Record<string, string> | undefined,
  label: string,
): Promise<{ ok: true; data: Record<string, unknown> } | TokenFailure> {
  let response: Response;
  try {
    response = await fetch(endpoint, {
      method: "POST",
      headers: authRequestHeaders(headers, "application/x-www-form-urlencoded"),
      body: form,
      redirect: "manual",
      signal: boundedLoginSignal(signal),
    });
  } catch {
    if (signal?.aborted) throw signal.reason;
    return { ok: false, transient: true, message: `${label} token exchange failed (network error)` };
  }
  let data: Record<string, unknown> | undefined;
  try {
    const parsed = (await response.json()) as unknown;
    if (isPlainObject(parsed)) data = parsed;
  } catch (error) {
    if (signal?.aborted) throw signal.reason;
    if (response.ok && !(error instanceof SyntaxError)) {
      return { ok: false, transient: true, message: `${label} token exchange failed (network error)` };
    }
  }
  signal?.throwIfAborted();
  if (!response.ok) {
    const code = oauthErrorCode(data?.error);
    return {
      ok: false,
      transient: response.status === 429 || response.status >= 500,
      message: code
        ? `${label} token exchange rejected (${code})`
        : `${label} token exchange failed (HTTP ${response.status})`,
    };
  }
  return { ok: true, data: data ?? {} };
}

async function requestPkceToken(
  endpoint: string,
  form: URLSearchParams,
  signal: AbortSignal | undefined,
  headers?: Record<string, string>,
  existingRefreshToken?: string,
): Promise<PkceTokenResult> {
  const response = await postTokenForm(endpoint, form, signal, headers, "LiteLLM");
  if (!response.ok) return response;
  const { data } = response;
  const expires = typeof data.expires_in === "number" ? Date.now() + data.expires_in * 1_000 : NaN;
  if (
    !isAuthToken(data.access_token) ||
    (!isAuthToken(data.refresh_token) && !isAuthToken(existingRefreshToken)) ||
    typeof data.token_type !== "string" ||
    data.token_type.toLowerCase() !== "bearer" ||
    typeof data.expires_in !== "number" ||
    !Number.isFinite(data.expires_in) ||
    data.expires_in <= 0 ||
    !Number.isSafeInteger(expires)
  ) {
    return { ok: false, transient: false, message: "LiteLLM token exchange returned an invalid response" };
  }
  return {
    ok: true,
    token: {
      access: data.access_token,
      // A refresh may not rotate the refresh token; keep the existing one when the server omits it.
      refresh: isAuthToken(data.refresh_token) ? data.refresh_token : existingRefreshToken!,
      expires,
      userId: typeof data.user_id === "string" && data.user_id ? data.user_id : undefined,
      teamId: typeof data.team_id === "string" && data.team_id ? data.team_id : undefined,
    },
  };
}

function constantTimeEqual(value: unknown, expected: string): boolean {
  if (typeof value !== "string") return false;
  const left = Buffer.from(value);
  const right = Buffer.from(expected);
  return left.length === right.length && timingSafeEqual(left, right);
}

type LoopbackCallback = { redirectUri: string; state: string; code: () => Promise<string> };

/**
 * Serves an RFC 8252 loopback redirect on 127.0.0.1 for the lifetime of `run`. `code()` resolves with
 * the authorization code of the first callback carrying the expected state.
 */
async function withLoopbackCallback<T>(
  signal: AbortSignal | undefined,
  label: string,
  ports: readonly number[],
  run: (callback: LoopbackCallback) => Promise<T>,
): Promise<T> {
  const state = randomBytes(32).toString("base64url");
  const server = createServer();
  let settleCallback: ((result: { code?: string; error?: Error }) => void) | undefined;
  server.on("request", (request, response) => {
    response.setHeader("Cache-Control", "no-store");
    let url: URL;
    try {
      url = new URL(request.url ?? "/", "http://127.0.0.1");
    } catch {
      response.writeHead(400).end("Invalid callback URL");
      return;
    }
    if (url.pathname !== "/callback") {
      response.writeHead(404).end();
      return;
    }
    if (request.method !== "GET") {
      response.writeHead(405, { Allow: "GET" }).end();
      return;
    }
    if (!constantTimeEqual(url.searchParams.get("state"), state)) {
      response.writeHead(400).end("Invalid OAuth state");
      return;
    }
    const error = url.searchParams.get("error");
    if (error) {
      response.writeHead(400).end("OAuth login failed");
      const code = oauthErrorCode(error);
      settleCallback?.({ error: new Error(`${label} login was denied${code ? ` (${code})` : ""}`) });
      return;
    }
    const code = url.searchParams.get("code");
    if (!code) {
      response.writeHead(400).end("Missing OAuth code");
      return;
    }
    response.writeHead(200, {
      "Cache-Control": "no-store",
      "Content-Type": "text/html; charset=utf-8",
    });
    response.end("<!doctype html><title>LiteLLM login complete</title><p>You can close this window.</p>");
    settleCallback?.({ code });
  });
  try {
    // Try each pre-registered port in order; without any, the OS assigns one (RFC 8252 §7.3).
    const candidates = ports.length > 0 ? ports : [0];
    for (const [index, port] of candidates.entries()) {
      try {
        await new Promise<void>((resolve, reject) => {
          const onError = (error: Error) => reject(error);
          server.once("error", onError);
          server.listen(port, "127.0.0.1", () => {
            server.off("error", onError);
            resolve();
          });
        });
        break;
      } catch (error) {
        if (index === candidates.length - 1) throw error;
      }
    }
    const address = server.address();
    if (!address || typeof address === "string") throw new Error(`${label} callback failed to start`);
    const code = () =>
      new Promise<string>((resolve, reject) => {
        const timeout = setTimeout(() => finish(new Error(`${label} login timed out`)), PKCE_CALLBACK_TIMEOUT_MS);
        const onAbort = () => finish(signal?.reason ?? new Error(`${label} login cancelled`));
        const finish = (error?: Error, value?: string) => {
          clearTimeout(timeout);
          signal?.removeEventListener("abort", onAbort);
          settleCallback = undefined;
          if (error) reject(error);
          else resolve(value!);
        };
        settleCallback = ({ code, error }) => finish(error, code);
        signal?.addEventListener("abort", onAbort, { once: true });
        if (signal?.aborted) onAbort();
      });
    return await run({ redirectUri: `http://127.0.0.1:${address.port}/callback`, state, code });
  } finally {
    await new Promise<void>((resolve) => {
      server.close(() => resolve());
      server.closeAllConnections();
    });
  }
}

async function loginPkce(
  interaction: AuthInteraction,
  baseUrl: string,
  discovery: CliAuthDiscovery,
  headers?: Record<string, string>,
): Promise<OAuthCredential> {
  const verifier = randomBytes(32).toString("base64url");
  return withLoopbackCallback(
    interaction.signal,
    "LiteLLM PKCE",
    [],
    async ({ redirectUri, state, code: callbackCode }) => {
      const registrationResponse = await fetch(discovery.registrationEndpoint, {
        method: "POST",
        headers: authRequestHeaders(headers, "application/json"),
        body: JSON.stringify({
          client_name: "pi-provider-litellm",
          redirect_uris: [redirectUri],
          token_endpoint_auth_method: "none",
          grant_types: ["authorization_code", "refresh_token"],
          response_types: ["code"],
        }),
        redirect: "manual",
        signal: boundedLoginSignal(interaction.signal),
      });
      if (!registrationResponse.ok)
        throw new Error(`LiteLLM PKCE client registration failed (HTTP ${registrationResponse.status})`);
      const registration = await readAuthJson(
        registrationResponse,
        interaction.signal,
        "LiteLLM CLI auth registration",
      );
      if (
        !isPlainObject(registration) ||
        !isAuthToken(registration.client_id) ||
        !Array.isArray(registration.redirect_uris) ||
        !registration.redirect_uris.includes(redirectUri)
      ) {
        throw new Error("LiteLLM PKCE client registration returned an invalid response");
      }
      const authorizationUrl = new URL(discovery.authorizationEndpoint);
      for (const [key, value] of Object.entries({
        client_id: registration.client_id,
        redirect_uri: redirectUri,
        response_type: "code",
        resource: discovery.resource,
        code_challenge: createHash("sha256").update(verifier).digest("base64url"),
        code_challenge_method: "S256",
        state,
      }))
        authorizationUrl.searchParams.set(key, value);
      interaction.notify({
        type: "auth_url",
        url: authorizationUrl.toString(),
        instructions: "Open this URL in a browser to sign in with LiteLLM.",
      });
      const code = await callbackCode();
      const result = await requestPkceToken(
        discovery.tokenEndpoint,
        new URLSearchParams({
          grant_type: "authorization_code",
          code,
          redirect_uri: redirectUri,
          client_id: registration.client_id,
          code_verifier: verifier,
          resource: discovery.resource,
        }),
        interaction.signal,
        headers,
      );
      if (!result.ok) throw new Error(result.message);
      const credential: PkceCredentials = {
        type: "oauth",
        access: result.token.access,
        refresh: result.token.refresh,
        expires: result.token.expires,
        baseUrl,
        flow: PKCE_FLOW,
        clientId: registration.client_id,
        tokenEndpoint: discovery.tokenEndpoint,
        resource: discovery.resource,
        userId: result.token.userId,
        teamId: result.token.teamId,
      };
      return { ...credential, type: "oauth" };
    },
  );
}

type OidcConfig = { issuer: string; clientId: string; scope: string; redirectPorts: number[] };

type OidcDiscovery = { issuer: string; authorizationEndpoint: string; tokenEndpoint: string };

type OidcCredentials = OAuthCredentials & {
  flow: typeof OIDC_FLOW;
  baseUrl: string;
  issuer: string;
  clientId: string;
  tokenEndpoint: string;
  subject: string;
};

type OidcTokenResult =
  | { ok: true; token: { access: string; refresh: string; expires: number; subject: string } }
  | TokenFailure;

/** An https URL without credentials or fragment, returned verbatim. */
function httpsUrl(value: unknown, { allowQuery }: { allowQuery: boolean }): string | undefined {
  if (typeof value !== "string" || value.includes("#") || (!allowQuery && value.includes("?"))) return undefined;
  try {
    const url = new URL(value);
    return url.protocol === "https:" && !url.username && !url.password ? value : undefined;
  } catch {
    return undefined;
  }
}

function parseOidcConfig(raw: unknown): OidcConfig {
  const invalid = (field: string, expected: string) =>
    new Error(`Invalid LiteLLM ${field} setting: expected ${expected}`);
  if (!isPlainObject(raw)) throw invalid("oidc", "an object");
  const issuer = httpsUrl(raw.issuer, { allowQuery: false });
  if (!issuer) throw invalid("oidc.issuer", "an https URL without credentials, query, or fragment");
  if (!isAuthToken(raw.clientId)) throw invalid("oidc.clientId", "a non-empty string without spaces");
  const scope = raw.scope ?? "openid";
  if (typeof scope !== "string" || !scope.split(" ").includes("openid"))
    throw invalid("oidc.scope", 'a space-separated scope list that includes "openid"');
  const redirectPorts = raw.redirectPorts ?? [];
  if (
    !Array.isArray(redirectPorts) ||
    !redirectPorts.every((port) => Number.isInteger(port) && port >= 1 && port <= 65535)
  )
    throw invalid("oidc.redirectPorts", "an array of port numbers from 1 to 65535");
  return { issuer, clientId: raw.clientId, scope, redirectPorts };
}

// IdP requests never carry the proxy's headers: LITELLM_HEADERS and provider headers can hold gateway credentials.
async function discoverOidc(issuer: string, signal?: AbortSignal): Promise<OidcDiscovery> {
  const withoutSlash = (value: string) => value.replace(/\/+$/, "");
  const response = await fetch(`${withoutSlash(issuer)}${OIDC_DISCOVERY_PATH}`, {
    headers: authRequestHeaders(),
    redirect: "manual",
    signal: boundedLoginSignal(signal),
  });
  if (!response.ok) throw new Error(`OIDC discovery failed (HTTP ${response.status})`);
  const data = await readAuthJson(response, signal, "OIDC discovery");
  if (!isPlainObject(data) || typeof data.issuer !== "string" || withoutSlash(data.issuer) !== withoutSlash(issuer))
    throw new Error("OIDC discovery issuer does not match the configured issuer");
  const methods = data.code_challenge_methods_supported;
  if (methods !== undefined && (!Array.isArray(methods) || !methods.includes("S256")))
    throw new Error("OIDC discovery does not support S256");
  const authorizationEndpoint = httpsUrl(data.authorization_endpoint, { allowQuery: true });
  if (!authorizationEndpoint) throw new Error("OIDC discovery has invalid authorization_endpoint");
  const tokenEndpoint = httpsUrl(data.token_endpoint, { allowQuery: true });
  if (!tokenEndpoint) throw new Error("OIDC discovery has invalid token_endpoint");
  return { issuer: data.issuer, authorizationEndpoint, tokenEndpoint };
}

function jwtClaims(token: string): Record<string, unknown> | undefined {
  const [, payload, ...rest] = token.split(".");
  if (rest.length !== 1 || !payload || !/^[\w-]+$/.test(payload)) return undefined;
  try {
    const claims: unknown = JSON.parse(Buffer.from(payload, "base64url").toString("utf8"));
    return isPlainObject(claims) ? claims : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Exchanges a grant for an id_token and checks its claims (OIDC Core §3.1.3.7). The signature is
 * deliberately not verified here: the proxy verifies it against the IdP's JWKS before honouring the
 * bearer, so these checks only stop Pi from storing a token meant for another client or login.
 */
async function requestOidcToken(
  endpoint: string,
  form: URLSearchParams,
  signal: AbortSignal | undefined,
  expected: { issuer: string; clientId: string; nonce?: string; subject?: string },
  existingRefreshToken = "",
): Promise<OidcTokenResult> {
  const response = await postTokenForm(endpoint, form, signal, undefined, "OIDC");
  if (!response.ok) return response;
  const { id_token: idToken, refresh_token: refreshToken } = response.data;
  const claims = isAuthToken(idToken) ? jwtClaims(idToken) : undefined;
  if (!isAuthToken(idToken) || !claims)
    return { ok: false, transient: false, message: "OIDC token response has no valid id_token" };
  const invalid = (claim: string): TokenFailure => ({
    ok: false,
    transient: false,
    message: `OIDC id_token has invalid ${claim}`,
  });
  if (claims.iss !== expected.issuer) return invalid("iss");
  if (!(Array.isArray(claims.aud) ? claims.aud : [claims.aud]).includes(expected.clientId)) return invalid("aud");
  // OIDC Core §3.1.3.7 (errata set 2): azp is optional, even with several audiences, but when present it must name us.
  if (claims.azp !== undefined && claims.azp !== expected.clientId) return invalid("azp");
  if (typeof claims.sub !== "string" || !claims.sub) return invalid("sub");
  if (expected.subject !== undefined && claims.sub !== expected.subject) return invalid("sub");
  const expiresAt = typeof claims.exp === "number" ? Math.floor(claims.exp * 1_000) : NaN;
  if (!Number.isSafeInteger(expiresAt) || expiresAt <= Date.now()) return invalid("exp");
  if (expected.nonce !== undefined && !constantTimeEqual(claims.nonce, expected.nonce)) return invalid("nonce");
  return {
    ok: true,
    token: {
      access: idToken,
      // The refresh token is optional, and an IdP that does not rotate it keeps the existing one valid.
      refresh: isAuthToken(refreshToken) ? refreshToken : existingRefreshToken,
      // Refresh ahead of expiry, but by at most half the remaining lifetime, so a short-lived id_token is
      // still used rather than treated as already expired.
      expires: expiresAt - Math.ceil(Math.min(TOKEN_REFRESH_LEAD_MS, (expiresAt - Date.now()) / 2)),
      subject: claims.sub,
    },
  };
}

async function loginOidc(interaction: AuthInteraction, baseUrl: string, config: OidcConfig): Promise<OAuthCredential> {
  const discovery = await discoverOidc(config.issuer, interaction.signal);
  const verifier = randomBytes(32).toString("base64url");
  const nonce = randomBytes(32).toString("base64url");
  return withLoopbackCallback(
    interaction.signal,
    "OIDC",
    config.redirectPorts,
    async ({ redirectUri, state, code }) => {
      const authorizationUrl = new URL(discovery.authorizationEndpoint);
      for (const [key, value] of Object.entries({
        response_type: "code",
        client_id: config.clientId,
        redirect_uri: redirectUri,
        scope: config.scope,
        state,
        nonce,
        code_challenge: createHash("sha256").update(verifier).digest("base64url"),
        code_challenge_method: "S256",
      }))
        authorizationUrl.searchParams.set(key, value);
      interaction.notify({
        type: "auth_url",
        url: authorizationUrl.toString(),
        instructions: "Open this URL in a browser to sign in with your identity provider.",
      });
      const result = await requestOidcToken(
        discovery.tokenEndpoint,
        new URLSearchParams({
          grant_type: "authorization_code",
          code: await code(),
          redirect_uri: redirectUri,
          client_id: config.clientId,
          code_verifier: verifier,
        }),
        interaction.signal,
        { issuer: discovery.issuer, clientId: config.clientId, nonce },
      );
      if (!result.ok) throw new Error(result.message);
      const credential: OidcCredentials = {
        type: "oauth",
        access: result.token.access,
        refresh: result.token.refresh,
        expires: result.token.expires,
        baseUrl,
        flow: OIDC_FLOW,
        issuer: discovery.issuer,
        clientId: config.clientId,
        tokenEndpoint: discovery.tokenEndpoint,
        subject: result.token.subject,
      };
      return { ...credential, type: "oauth" };
    },
  );
}

async function startCliSso(
  baseUrl: string,
  signal?: AbortSignal,
  headers?: Record<string, string>,
): Promise<CliSsoStart | undefined> {
  const response = await fetch(`${baseUrl}/sso/cli/start`, {
    method: "POST",
    headers,
    signal: boundedLoginSignal(signal),
  });
  if (response.status === 404 || response.status === 405) return undefined;
  if (!response.ok) throw new Error(`LiteLLM CLI SSO start failed (HTTP ${response.status})`);
  const data = (await response.json()) as Record<string, unknown>;
  if (
    typeof data.login_id !== "string" ||
    !data.login_id ||
    typeof data.poll_secret !== "string" ||
    !data.poll_secret ||
    typeof data.user_code !== "string" ||
    !data.user_code
  ) {
    throw new Error("LiteLLM CLI SSO start returned an invalid response");
  }
  return {
    loginId: data.login_id,
    pollSecret: data.poll_secret,
    userCode: data.user_code,
    expiresInSeconds:
      typeof data.expires_in === "number" && Number.isFinite(data.expires_in) && data.expires_in > 0
        ? data.expires_in
        : CLI_SSO_EXPIRES_IN_SECONDS,
  };
}

async function waitForNextCliSsoPoll(interaction: AuthInteraction): Promise<void> {
  try {
    await delay(CLI_SSO_POLL_INTERVAL_MS, undefined, interaction.signal ? { signal: interaction.signal } : undefined);
  } catch (error) {
    if (interaction.signal?.aborted) throw interaction.signal.reason;
    throw error;
  }
}

async function pollCliSso(
  baseUrl: string,
  start: CliSsoStart,
  interaction: AuthInteraction,
  headers?: Record<string, string>,
): Promise<{ access: string; expiresInSeconds?: number }> {
  const deadline = Date.now() + start.expiresInSeconds * 1_000;
  const pollUrl = new URL(`${baseUrl}/sso/cli/poll/${encodeURIComponent(start.loginId)}`);
  let teamId: string | undefined;
  while (Date.now() < deadline) {
    if (teamId) pollUrl.searchParams.set("team_id", teamId);
    let response: Response;
    try {
      response = await fetch(pollUrl, {
        headers: { ...headers, "x-litellm-cli-poll-secret": start.pollSecret },
        signal: boundedLoginSignal(interaction.signal),
      });
    } catch {
      if (interaction.signal?.aborted) throw interaction.signal.reason;
      await waitForNextCliSsoPoll(interaction);
      continue;
    }
    if (response.status === 429 || response.status >= 500) {
      await waitForNextCliSsoPoll(interaction);
      continue;
    }
    if (response.status === 400) throw new Error("LiteLLM CLI SSO login expired or is invalid");
    if (!response.ok) throw new Error(`LiteLLM CLI SSO polling failed (HTTP ${response.status})`);
    const data = (await response.json()) as Record<string, unknown>;
    if (data.status === "ready" && typeof data.key === "string" && data.key) {
      return {
        access: data.key,
        expiresInSeconds:
          typeof data.expires_in === "number" && Number.isFinite(data.expires_in) && data.expires_in > 0
            ? data.expires_in
            : undefined,
      };
    }
    if (data.status === "ready" && data.requires_team_selection === true && !teamId) {
      const details = Array.isArray(data.team_details) ? data.team_details : [];
      const teams = details.flatMap((team) => {
        if (!team || typeof team !== "object") return [];
        const record = team as Record<string, unknown>;
        const id = typeof record.team_id === "string" ? record.team_id : record.id;
        const alias = record.team_alias;
        return typeof id === "string" && id ? [{ id, label: typeof alias === "string" && alias ? alias : id }] : [];
      });
      if (teams.length === 0 && Array.isArray(data.teams))
        teams.push(...data.teams.flatMap((id) => (typeof id === "string" && id ? [{ id, label: id }] : [])));
      if (teams.length === 0) throw new Error("LiteLLM CLI SSO requested team selection without any teams");
      const selected = await interaction.prompt({ type: "select", message: "Select a LiteLLM team:", options: teams });
      if (!teams.some((team) => team.id === selected)) throw new Error("Invalid LiteLLM team selection");
      teamId = selected;
      continue;
    }
    if (data.status !== "pending") throw new Error("LiteLLM CLI SSO polling returned an invalid response");
    await waitForNextCliSsoPoll(interaction);
  }
  throw new Error("LiteLLM CLI SSO login expired");
}

async function loginWithPastedToken(
  interaction: AuthInteraction,
  baseUrl: string,
  headers?: Record<string, string>,
): Promise<OAuthCredential> {
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

async function loginOAuth(interaction: AuthInteraction, definition: ProviderDefinition): Promise<OAuthCredential> {
  const oidc = definition.oidc === undefined ? undefined : parseOidcConfig(definition.oidc);
  const baseUrl = await promptBaseUrl(interaction, definition);
  // Direct OIDC sends nothing to the proxy. The IdP is never derived from it either: the proxy's own OAuth
  // metadata describes LiteLLM's authorization server, not the IdP that signs the JWTs it accepts.
  if (oidc) return loginOidc(interaction, baseUrl, oidc);
  const headers = resolveHeaders(definition);
  const discovery = await discoverPkce(baseUrl, interaction.signal, headers);
  if (discovery) return loginPkce(interaction, baseUrl, discovery, headers);
  const cliSso = await startCliSso(baseUrl, interaction.signal, headers);
  if (!cliSso) return loginWithPastedToken(interaction, baseUrl, headers);
  interaction.notify({
    type: "device_code",
    userCode: cliSso.userCode,
    verificationUri: `${baseUrl}/sso/key/generate?${new URLSearchParams({ source: "litellm-cli", key: cliSso.loginId })}`,
    expiresInSeconds: cliSso.expiresInSeconds,
  });
  const result = await pollCliSso(baseUrl, cliSso, interaction, headers);
  return {
    type: "oauth",
    access: result.access,
    refresh: "",
    expires: result.expiresInSeconds
      ? Date.now() + result.expiresInSeconds * 1_000
      : tokenExpiresAt(result.access, configuredCliJwtExpiresAt()),
    baseUrl,
  };
}

// Keyed by refresh token (so one credential's failures never delay another's refresh):
// how long to skip re-attempting a refresh after a transient failure, so callers
// serialized behind Pi's credential lock don't each fire another request at the
// still-failing token endpoint (e.g. during an outage or rate limit).
const pkceTransientRefreshBackoff = new Map<string, number>();

async function refreshWithBackoff(
  credentials: OAuthCredentials,
  request: () => Promise<PkceTokenResult | OidcTokenResult>,
): Promise<OAuthCredentials> {
  if (Date.now() < credentials.expires) {
    const backoffUntil = pkceTransientRefreshBackoff.get(credentials.refresh);
    if (backoffUntil !== undefined && Date.now() < backoffUntil) return credentials;
  }
  const result = await request();
  if (!result.ok && result.transient && Date.now() < credentials.expires) {
    pkceTransientRefreshBackoff.set(credentials.refresh, Date.now() + PKCE_TRANSIENT_REFRESH_BACKOFF_MS);
    return credentials;
  }
  pkceTransientRefreshBackoff.delete(credentials.refresh);
  // A transient failure leaves the stored credential intact, and a new login would need the same endpoint.
  if (!result.ok) throw new Error(result.transient ? result.message : `${result.message}; run /login litellm again`);
  return { ...credentials, ...result.token };
}

async function refreshLiteLLM(
  credentials: OAuthCredentials,
  definition: ProviderDefinition,
  signal?: AbortSignal,
): Promise<OAuthCredentials> {
  signal?.throwIfAborted();
  if (credentials.flow === PKCE_FLOW) {
    const { baseUrl, clientId } = credentials;
    if (
      typeof baseUrl !== "string" ||
      !isAuthToken(clientId) ||
      !isAuthToken(credentials.access) ||
      !isAuthToken(credentials.refresh) ||
      !Number.isSafeInteger(credentials.expires)
    ) {
      throw new Error("Invalid LiteLLM PKCE credential; run /login litellm again");
    }
    return refreshWithBackoff(credentials, () => {
      const root = requireCredentialRoot(normalizeBaseUrl(baseUrl, definition.allowInsecureHttp), definition.name);
      canonicalIssuer(root);
      const issuer = new URL(root);
      const tokenEndpoint = sameOriginUrl(credentials.tokenEndpoint, issuer, "token endpoint");
      const resource = sameOriginUrl(credentials.resource, issuer, "resource");
      return requestPkceToken(
        tokenEndpoint,
        new URLSearchParams({
          grant_type: "refresh_token",
          refresh_token: credentials.refresh,
          client_id: clientId,
          resource,
        }),
        signal,
        resolveHeaders(definition),
        credentials.refresh,
      );
    });
  }
  // Must precede the `!command` fallthrough below: an IdP refresh token is never executed.
  if (credentials.flow === OIDC_FLOW) {
    const { refresh, issuer, clientId, subject } = credentials;
    const tokenEndpoint = httpsUrl(credentials.tokenEndpoint, { allowQuery: true });
    if (!isAuthToken(refresh))
      throw new Error("LiteLLM OIDC credential has no refresh token; run /login litellm again");
    if (
      !tokenEndpoint ||
      typeof issuer !== "string" ||
      !isAuthToken(clientId) ||
      typeof subject !== "string" ||
      !subject ||
      !Number.isSafeInteger(credentials.expires)
    ) {
      throw new Error("Invalid LiteLLM OIDC credential; run /login litellm again");
    }
    return refreshWithBackoff(credentials, () =>
      requestOidcToken(
        tokenEndpoint,
        new URLSearchParams({ grant_type: "refresh_token", refresh_token: refresh, client_id: clientId }),
        signal,
        { issuer, clientId, subject },
        refresh,
      ),
    );
  }
  if (!credentials.refresh.startsWith("!")) {
    if (credentials.expires < PERMANENT_TOKEN_EXPIRES_AT) {
      throw new Error("LiteLLM credential cannot be refreshed; run /login litellm again");
    }
    return credentials;
  }
  const access = executeApiKeyCommand(credentials.refresh);
  return { ...credentials, access, expires: tokenExpiresAt(access, EXPIRE_TOKEN_IMMEDIATELY) };
}

async function configuredBaseUrl(
  definition: ProviderDefinition,
  ctx: { env(name: string): Promise<string | undefined> },
  credential?: ApiKeyCredential,
): Promise<string | undefined> {
  const scopedEnv = definition.envPrefix ? cleanConfig(await ctx.env(`${definition.envPrefix}_BASE_URL`)) : undefined;
  const defaultEnv = definition.useDefaultEnv ? cleanConfig(await ctx.env(ENV_BASE_URL)) : undefined;
  return cleanConfig(credential?.env?.[ENV_BASE_URL]) ?? cleanConfig(definition.baseUrl) ?? scopedEnv ?? defaultEnv;
}

async function resolveApiKeyAuth(
  definition: ProviderDefinition,
  ctx: { env(name: string): Promise<string | undefined> },
  credential?: ApiKeyCredential,
  executeHelpers = true,
) {
  const baseUrl = await configuredBaseUrl(definition, ctx, credential);
  const stored = credential?.key
    ? resolveConfigValue(credential.key, { executeCommands: executeHelpers })?.trim()
    : undefined;
  let source: string | undefined;
  let creds: ResolvedCredentials;
  if (stored) {
    source = "stored credential";
    creds = {
      baseUrl: baseUrl ? normalizeBaseUrl(baseUrl, definition.allowInsecureHttp) : undefined,
      apiKey: stored,
    };
  } else {
    // 1) Google ADC
    const useGcloudToken = definition.useDefaultEnv
      ? definition.useGcloudTokenAuth && isGcloudTokenAuthEnabled()
      : definition.useGcloudTokenAuth;
    const gcloudKey = executeHelpers && useGcloudToken ? (await getGcloudToken())?.trim() : undefined;
    let apiKey = gcloudKey;
    let apiKeyConfig: string | undefined;
    if (gcloudKey) {
      source = GCLOUD_ADC_SOURCE;
    }

    // 2) definition.apiKeyConfig
    if (!apiKey && definition.apiKeyConfig) {
      const configured = definition.apiKeyConfig.startsWith("!")
        ? executeHelpers
          ? executeApiKeyCommand(definition.apiKeyConfig)
          : undefined
        : await resolveTemplateConfigValueFromContext(definition.apiKeyConfig, ctx.env);
      if (configured) {
        apiKey = configured;
        apiKeyConfig = definition.apiKeyConfig;
        source = definition.apiKeyConfig;
      } else if (!definition.apiKeyConfig.startsWith("!")) {
        warnUnresolvedApiKeyConfig(definition.name, definition.apiKeyConfig);
      }
    }

    // 3) Scoped envPrefix: _API_KEY_HELPER then _API_KEY
    if (!apiKey && definition.envPrefix) {
      const helper = normalizeCommand(await ctx.env(`${definition.envPrefix}_API_KEY_HELPER`));
      if (helper) {
        apiKey = executeHelpers ? executeApiKeyCommand(helper) : undefined;
        apiKeyConfig = helper;
        source = `$${definition.envPrefix}_API_KEY_HELPER`;
      } else {
        const envKey = cleanConfig(await ctx.env(`${definition.envPrefix}_API_KEY`));
        if (envKey) {
          apiKey = envKey;
          apiKeyConfig = `$${definition.envPrefix}_API_KEY`;
          source = `$${definition.envPrefix}_API_KEY`;
        }
      }
    }

    // 4) Default env: LITELLM_API_KEY_HELPER then LITELLM_API_KEY
    if (!apiKey && definition.useDefaultEnv) {
      const helper = normalizeCommand(await ctx.env(ENV_API_KEY_HELPER));
      if (helper) {
        apiKey = executeHelpers ? executeApiKeyCommand(helper) : undefined;
        apiKeyConfig = helper;
        source = ENV_API_KEY_HELPER;
      } else {
        const envKey = cleanConfig(await ctx.env(ENV_API_KEY));
        if (envKey) {
          apiKey = envKey;
          apiKeyConfig = ENV_API_KEY;
          source = ENV_API_KEY;
        }
      }
    }

    creds = {
      baseUrl: baseUrl ? normalizeBaseUrl(baseUrl, definition.allowInsecureHttp) : undefined,
      apiKey: apiKey || undefined,
      apiKeyConfig,
      apiKeyFromGcloudAdc: Boolean(gcloudKey),
    };
  }
  if (!creds.apiKey) return undefined;
  const normalizedRoot = baseUrl ? normalizeBaseUrl(baseUrl, definition.allowInsecureHttp) : undefined;
  // Pin the request host to the root resolved from the credential. Pi's provider composer
  // routes an off-catalog model through a global API implementation that trusts model.baseUrl
  // verbatim, bypassing the provider's own host guard; Models.applyAuth overrides model.baseUrl
  // with auth.baseUrl on every path, so this keeps the credential from reaching a stale or
  // attacker-supplied host. Fail auth resolution when no usable root resolves rather than
  // returning a key with no pinned baseUrl, which would let the global fallback use a stale
  // or foreign model.baseUrl instead.
  const pinnedRoot = requireCredentialRoot(
    resolveCredentialRoot(definition, credential, normalizedRoot),
    definition.name,
  );
  return {
    auth: {
      apiKey: creds.apiKey,
      headers: await resolveHeadersFromContext(definition, ctx.env),
      baseUrl: pinnedRoot,
    },
    env: normalizedRoot ? { [ENV_BASE_URL]: normalizedRoot } : undefined,
    source: source ?? (creds.apiKeyFromGcloudAdc ? GCLOUD_ADC_SOURCE : undefined) ?? creds.apiKeyConfig ?? ENV_API_KEY,
  };
}

function createProviderAuth(
  definition: ProviderDefinition,
  clearOAuthRuntimeRoot?: () => void,
  loginHooks?: { start: () => void; complete: () => void },
  oauthRuntimeRoot?: () => { apiKey: string; root: string } | undefined,
): ProviderAuth {
  function completeLogin<T extends Credential>(credential: T): T {
    loginHooks?.complete();
    return credential;
  }
  return {
    apiKey: {
      name: `${definition.displayName} API key`,
      login: async (interaction) => {
        loginHooks?.start();
        const credential = await loginApiKey(interaction, definition);
        return completeLogin(credential);
      },
      check: async ({ ctx, credential }) => {
        const baseUrl =
          credential?.env?.[ENV_BASE_URL] ??
          definition.baseUrl ??
          (definition.envPrefix ? await ctx.env(`${definition.envPrefix}_BASE_URL`) : undefined) ??
          (definition.useDefaultEnv ? await ctx.env(ENV_BASE_URL) : undefined);
        if (!cleanConfig(baseUrl)) return undefined;
        if (credential?.key) return { type: "api_key", source: "stored credential" };

        // Credentials `resolve` would fall back to if ADC cannot mint a token.
        const fallbackSource = async (): Promise<string | undefined> => {
          if (definition.apiKeyConfig) {
            const configuredKey = definition.apiKeyConfig.startsWith("!")
              ? definition.apiKeyConfig
              : await resolveTemplateConfigValueFromContext(definition.apiKeyConfig, ctx.env);
            if (configuredKey) return definition.apiKeyConfig;
          }
          if (definition.envPrefix) {
            if (cleanConfig(await ctx.env(`${definition.envPrefix}_API_KEY_HELPER`))) {
              return `$${definition.envPrefix}_API_KEY_HELPER`;
            }
            if (cleanConfig(await ctx.env(`${definition.envPrefix}_API_KEY`))) {
              return `$${definition.envPrefix}_API_KEY`;
            }
          }
          if (definition.useDefaultEnv && cleanConfig(await ctx.env(ENV_API_KEY_HELPER))) return ENV_API_KEY_HELPER;
          if (definition.useDefaultEnv && cleanConfig(await ctx.env(ENV_API_KEY))) return ENV_API_KEY;
          return undefined;
        };

        // Mirror the precedence in `resolveCredentials`, where ADC outranks the config
        // key, the helper and the environment key. Whether the refresh token still mints
        // is only knowable at request time, and this must not make a network call; if it
        // fails, `resolve` falls back and reports the credential it actually used.
        const useGcloudToken = definition.useDefaultEnv
          ? definition.useGcloudTokenAuth && isGcloudTokenAuthEnabled()
          : definition.useGcloudTokenAuth;
        if (useGcloudToken && (await hasGcloudAdcCredentials())) return { type: "api_key", source: GCLOUD_ADC_SOURCE };
        const fallback = await fallbackSource();
        return fallback ? { type: "api_key", source: fallback } : undefined;
      },
      resolve: async ({ ctx, credential }) => {
        // Pi re-resolves auth through this api-key path whenever a caller passes an explicit
        // apiKey override — compaction and summarization both hand back the key they just
        // resolved — which bypasses the stored OAuth credential carrying the base URL. The
        // api-key path exports its root as env so it survives that round trip, but OAuth's
        // toAuth has no env to export, leaving an SSO session no record of its root. Read it
        // back from auth.json, the same fallback check() and seedModels() use, and export it
        // so the request carries the root rather than depending on in-memory state. auth.json is
        // shared, though: another Pi process logging in replaces the stored token while this one
        // still holds its own, so fall back to the root this process resolved for that exact
        // token. Only when nothing configures a root at all: a base URL that is set still
        // resolves, and still fails, on its own terms rather than silently rerouting.
        if (credential?.key && !(await configuredBaseUrl(definition, ctx, credential))) {
          const stored = readStoredCredential(definition.name, join(getAgentDir(), "auth.json"));
          const remembered = oauthRuntimeRoot?.();
          const root =
            stored?.type === "oauth" && stored.access === credential.key
              ? requireCredentialRoot(resolveCredentialRoot(definition, stored), definition.name)
              : remembered?.apiKey === credential.key
                ? remembered.root
                : undefined;
          if (root) {
            return {
              auth: { apiKey: credential.key, headers: resolveHeaders(definition), baseUrl: root },
              env: { [ENV_BASE_URL]: root },
              source: "OAuth",
            };
          }
        }
        clearOAuthRuntimeRoot?.();
        return resolveApiKeyAuth(definition, ctx, credential);
      },
    },
    oauth: definition.enableOAuth
      ? {
          name: "LiteLLM SSO",
          loginLabel: "Sign in with LiteLLM SSO",
          login: async (interaction) => {
            loginHooks?.start();
            const credential = await loginOAuth(interaction, definition);
            return completeLogin(credential);
          },
          refresh: async (credential, signal) => ({
            ...(await refreshLiteLLM(credential, definition, signal)),
            type: "oauth" as const,
          }),
          toAuth: async (credential) => ({
            apiKey: credential.access,
            headers: resolveHeaders(definition),
          }),
        }
      : undefined,
  };
}

function isReasoningItem(item: unknown): boolean {
  return typeof item === "object" && item !== null && (item as { type?: unknown }).type === "reasoning";
}

// Returns the original array reference when no normalization is needed.
function normalizeStrictToolMessages(messages: any[]): any[] {
  const normalized: any[] = [];
  let imageParts: any[] = [];
  let changed = false;
  const flushImages = (): void => {
    if (imageParts.length === 0) return;
    normalized.push({
      role: "user",
      content: [{ type: "text", text: "Attached image(s) from tool result:" }, ...imageParts],
    });
    imageParts = [];
  };

  for (const message of messages) {
    if (message.role !== "tool") flushImages();
    if (
      message.role === "assistant" &&
      Array.isArray(message.tool_calls) &&
      message.tool_calls.length > 0 &&
      message.content == null
    ) {
      normalized.push({ ...message, content: "" });
      changed = true;
    } else if (message.role === "tool" && Array.isArray(message.content)) {
      const previousImageCount = imageParts.length;
      const content = message.content
        .map((part: any) => {
          if (typeof part === "string") return part;
          if (part?.type === "text") return part.text || "";
          if (part?.type === "image_url") imageParts.push(part);
          return "";
        })
        .join("");
      normalized.push({
        ...message,
        content: content || (imageParts.length > previousImageCount ? "(see attached image)" : "(no tool output)"),
      });
      changed = true;
    } else {
      normalized.push(message);
    }
  }
  flushImages();
  return changed ? normalized : messages;
}

// Visibility fields suppress duplicate Kimi reasoning without changing the
// model-specific generation control selected by pi-ai.
const REASONING_VISIBILITY_DEFAULTS: Record<string, unknown> = {
  include_reasoning: false,
  reasoning_content: false,
  merge_reasoning_content_in_choices: true,
};
const REASONING_REQUEST_KEYS = [
  "reasoning",
  "reasoning_effort",
  ...Object.keys(REASONING_VISIBILITY_DEFAULTS),
  "thinking",
];
export function prepareLiteLLMRequestPayload(
  payload: Record<string, unknown>,
  model: LiteLLMModel | undefined,
  modelPolicy: LiteLLMModelPolicy | undefined,
): Record<string, unknown> | undefined {
  const modelId = model?.id;
  const api = model?.api;
  const openAIApi = api ?? "openai-completions";
  let next: Record<string, unknown> | undefined;
  const update = (key: string, value: unknown): void => {
    if (payload[key] !== undefined) return;
    next ??= { ...payload };
    next[key] = value;
  };

  if (openAIApi === "openai-completions" && modelPolicy?.suppressReasoningVisibility) {
    for (const [key, value] of Object.entries(REASONING_VISIBILITY_DEFAULTS)) update(key, value);
  }

  // GPT-5.5 and later reject reasoning alongside function tools on Chat Completions.
  // Drop it until the gateway serves /v1/responses for the route. Discovery decides
  // from deployment evidence; only a model with no backend evidence uses its route name.
  const routeOnlyId = model?.litellmBackendFamily === undefined ? modelId : undefined;
  if (
    openAIApi === "openai-completions" &&
    (modelPolicy?.dropToolReasoning === true || (routeOnlyId !== undefined && isGpt55OrNewerModel(routeOnlyId))) &&
    Array.isArray(payload.tools) &&
    payload.tools.length > 0
  ) {
    for (const key of REASONING_REQUEST_KEYS) {
      if (payload[key] === undefined) continue;
      next ??= { ...payload };
      delete next[key];
    }
    // Send the model's own off effort when it declares one. GPT-6 also rejects an
    // omitted effort (it falls back to its default), and its error names `none` as
    // the fix, so it gets `none` even without carrier evidence for an off level.
    const declaredOff = model?.thinkingLevelMap?.off;
    const explicitOff =
      modelPolicy?.explicitToolReasoningOff === true || (routeOnlyId !== undefined && isGpt6OrNewerModel(routeOnlyId));
    const off = typeof declaredOff === "string" ? declaredOff : explicitOff ? "none" : undefined;
    if (off !== undefined && (next ?? payload).reasoning_effort !== off) {
      next ??= { ...payload };
      next.reasoning_effort = off;
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

  // Moonshot/Kimi applies strict OpenAI schema validation: assistant tool calls
  // must carry string content, and tool results must be plain text. This outbound
  // rewrite requires the deployment-backed policy; route text is not evidence.
  if (openAIApi === "openai-completions" && modelPolicy?.normalizeStrictToolMessages) {
    const messages = (next ?? payload).messages;
    if (Array.isArray(messages)) {
      const normalized = normalizeStrictToolMessages(messages);
      if (normalized !== messages) {
        next ??= { ...payload };
        next.messages = normalized;
      }
    }
  }

  if (
    (openAIApi === "openai-completions" || openAIApi === "openai-responses") &&
    modelPolicy?.normalizeGeminiReasoningEffort
  ) {
    const currentPayload = next ?? payload;
    if (typeof currentPayload.reasoning_effort === "string") {
      const lower = currentPayload.reasoning_effort.toLowerCase();
      if (currentPayload.reasoning_effort !== lower) {
        next ??= { ...payload };
        next.reasoning_effort = lower;
      }
    }

    const reasoningPayload = next ?? payload;
    if (isPlainObject(reasoningPayload.reasoning) && typeof reasoningPayload.reasoning.effort === "string") {
      const lower = reasoningPayload.reasoning.effort.toLowerCase();
      if (reasoningPayload.reasoning.effort !== lower) {
        next ??= { ...payload };
        next.reasoning = {
          ...(next.reasoning as Record<string, unknown>),
          effort: lower,
        };
      }
    }
  }

  return next;
}

function normalizeThinkTags(
  message: AssistantMessage,
  litellmProviderNames: Set<string>,
  model: LiteLLMModel | undefined,
): AssistantMessage | undefined {
  if (
    !litellmProviderNames.has(message.provider) ||
    (model?.api !== undefined && model.api !== "openai-completions") ||
    !model?.litellmPolicy?.normalizeThinkTags
  )
    return;

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
  // Pi installs packages without resolving peerDependencies, so only this check keeps an older Pi from
  // loading the extension. It runs before anything registers; Pi reports the error and starts without it.
  // Pi's version parses like the proxy's: a pre-release orders before the release it names.
  if (!proxyVersionAtLeast(parseProxyVersion(piCodingAgent.VERSION), [0, 99, 2])) {
    throw new Error(
      `pi-provider-litellm needs Pi 0.99.2 or newer; this is Pi ${piCodingAgent.VERSION}. ` +
        "Update Pi, or install the last release for older Pi: pi install npm:pi-provider-litellm@3.4.0",
    );
  }
  const settings = await readGlobalLiteLLMSettings();
  const definitions = getProviderDefinitions(settings);
  const skillsEnabled = isFeatureEnabled(settings, "skills");
  const mcpEnabled = isFeatureEnabled(settings, "mcp");
  const budgetEnabled = isFeatureEnabled(settings, "budget");
  const providerNames = new Set(definitions.map((definition) => definition.name));
  const oauthRuntimeRoots = new Map<string, { apiKey: string; root: string }>();
  let mcpUI: ExtensionContext["ui"] | undefined;
  let sessionStarted = false;
  const pendingMcpMessages = new Map<string, "info" | "warning">();

  function notifyMcp(message: string, level: "info" | "warning" = "warning"): void {
    const text = message.trimEnd();
    if (mcpUI) mcpUI.notify(text, level);
    // Pi starts the terminal before supplying an ExtensionContext. Buffer until session_start.
    else if (!sessionStarted && process.stderr.isTTY) pendingMcpMessages.set(text, level);
    else process.stderr.write(`${text}\n`);
  }

  pi.on("session_start", (_event, ctx) => {
    sessionStarted = true;
    mcpUI = ctx.hasUI ? ctx.ui : undefined;
    for (const [message, level] of pendingMcpMessages) notifyMcp(message, level);
    pendingMcpMessages.clear();
  });
  pi.on("session_shutdown", () => {
    mcpUI = undefined;
    sessionStarted = false;
    pendingMcpMessages.clear();
  });

  function discoveryDisabledReason(): string | null {
    if (isOffline()) return `${ENV_OFFLINE}=1`;
    if (getDiscoveryTimeoutMs() === 0) return `${ENV_TIMEOUT}=0`;
    return null;
  }

  // LiteLLM serves MCP over streamable HTTP at `/mcp`, so each provider registers it as a Pi MCP
  // server and Pi's client does the rest. `auth.provider` makes Pi send the provider's current
  // credential on every request, so token refreshes and key helpers apply without registering
  // again; only a new proxy root or new headers change the registration.
  const mcpSettings = isPlainObject(settings?.mcp) ? settings.mcp : undefined;
  // The last config attempted for each provider, and whether Pi accepted it. Headers can carry
  // credentials, so the config is remembered as a keyed digest.
  const mcpAttempts = new Map<string, { identity: string; registered: boolean }>();
  const mcpIdentitySalt = randomBytes(32);

  function mcpServerConfig(definition: ProviderDefinition): (McpServerConfig & { url: string }) | undefined {
    if (discoveryDisabledReason() || isHostOffline()) return undefined;
    let root: string;
    try {
      const stored = readStoredCredential(definition.name, join(getAgentDir(), "auth.json"));
      root = requireCredentialRoot(resolveCredentialRoot(definition, stored), definition.name);
    } catch {
      return undefined;
    }
    const headers = resolveHeaders(definition);
    // Pi validates exposure and toolExposure; an unusable value is reported when registering.
    return {
      url: `${root}/mcp`,
      auth: { provider: definition.name },
      ...(headers ? { headers: Object.fromEntries(Object.entries(headers).map(([k, v]) => [k, piLiteral(v)])) } : {}),
      ...(mcpSettings?.exposure !== undefined ? { exposure: mcpSettings.exposure } : {}),
      ...(mcpSettings?.toolExposure !== undefined ? { toolExposure: mcpSettings.toolExposure } : {}),
    } as McpServerConfig & { url: string };
  }

  function dropMcpServer(definition: ProviderDefinition): void {
    const attempt = mcpAttempts.get(definition.name);
    mcpAttempts.delete(definition.name);
    if (attempt?.registered) pi.unregisterMcpServer(definition.name);
  }

  // LiteLLM 1.102.0 and later answer `initialize` with 403 when the key has no MCP servers granted,
  // and Pi warns at every start about a server that fails to connect. Asking first lets such a
  // server be registered disabled: Pi neither connects to it nor warns, and /mcp still lists it, so
  // the user can enable it once servers are granted.
  async function mcpAccessRefused(definition: ProviderDefinition, url: string): Promise<boolean> {
    try {
      const stored = readStoredCredential(definition.name, join(getAgentDir(), "auth.json"));
      // The check must send the credential Pi will send. Like seeding, it runs no key helper and
      // mints no ADC token, and resolving without them would fall through to another key.
      if (
        (stored?.type === "api_key" && stored.key?.startsWith("!")) ||
        definition.apiKeyConfig?.startsWith("!") ||
        (definition.useGcloudTokenAuth && isGcloudTokenAuthEnabled())
      ) {
        return false;
      }
      const auth = await authForCredential(definition, stored, false);
      // The credential goes only to its own root.
      if (`${auth.baseUrl}/mcp` !== url) return false;
      // The credential and headers Pi's client sends, so the proxy answers as it will answer Pi.
      const headers = { ...auth.headers, Authorization: `Bearer ${auth.apiKey}` };
      const response = await fetch(url, {
        method: "POST",
        headers: { ...headers, accept: "application/json, text/event-stream", "content-type": "application/json" },
        body: JSON.stringify({
          jsonrpc: "2.0",
          id: 1,
          method: "initialize",
          params: {
            protocolVersion: "2025-06-18",
            capabilities: {},
            clientInfo: { name: "pi", version: piCodingAgent.VERSION },
          },
        }),
        redirect: "manual",
        signal: AbortSignal.timeout(getSeedTimeoutMs()),
      });
      await response.body?.cancel().catch(() => undefined);
      // Pi opens its own session; end the one this check opened.
      const session = response.headers.get("mcp-session-id");
      if (session) {
        void fetch(url, {
          method: "DELETE",
          headers: { ...headers, "mcp-session-id": session },
          redirect: "manual",
          signal: AbortSignal.timeout(getSeedTimeoutMs()),
        })
          .then((closed) => closed.body?.cancel())
          .catch(() => undefined);
      }
      return response.status === 403;
    } catch {
      return false;
    }
  }

  async function syncMcpServer(definition: ProviderDefinition): Promise<void> {
    const config = mcpServerConfig(definition);
    const identity = config && createHmac("sha256", mcpIdentitySalt).update(JSON.stringify(config)).digest("hex");
    if (identity === mcpAttempts.get(definition.name)?.identity) return;
    // Close the old connection before opening one to a new root, and leave nothing behind when
    // Pi refuses the new config.
    dropMcpServer(definition);
    if (!config || !identity) return;
    // Claimed before the access check, so the next turn does not repeat it, and a login that starts
    // meanwhile drops the claim before this registration can land.
    const attempt = { identity, registered: false };
    mcpAttempts.set(definition.name, attempt);
    const refused = await mcpAccessRefused(definition, config.url);
    if (mcpAttempts.get(definition.name) !== attempt) return;
    try {
      pi.registerMcpServer(definition.name, refused ? { ...config, enabled: false } : config);
      attempt.registered = true;
    } catch (error) {
      // The claim stays, so the refusal is reported once rather than on every turn.
      notifyMcp(
        `LiteLLM MCP (${JSON.stringify(definition.name)}): ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  function requestBaseUrl(definition: ProviderDefinition): string {
    try {
      return `${resolveCredentialRoot(definition) ?? DEFAULT_LITELLM_BASE_URL}/v1`;
    } catch {
      return `${DEFAULT_LITELLM_BASE_URL}/v1`;
    }
  }

  async function authForCredential(definition: ProviderDefinition, credential?: Credential, executeHelpers = true) {
    if (credential?.type === "oauth") {
      const baseUrl = requireCredentialRoot(resolveCredentialRoot(definition, credential), definition.name);
      return {
        baseUrl,
        apiKey: credential.access,
        headers: resolveHeaders(definition),
        allowInsecureHttp: definition.allowInsecureHttp,
      };
    }
    const resolved = await resolveApiKeyAuth(
      definition,
      { env: async (name) => process.env[name] },
      credential,
      executeHelpers,
    );
    if (!resolved?.auth.apiKey || !resolved.auth.baseUrl) {
      throw new Error(`no credentials for ${definition.name}. Run /login litellm or set env vars.`);
    }
    return {
      baseUrl: resolved.auth.baseUrl,
      apiKey: resolved.auth.apiKey,
      headers: resolved.auth.headers,
      allowInsecureHttp: definition.allowInsecureHttp,
    };
  }

  let defaultRuntimeAuth: LiteLLMRuntimeAuth | undefined;
  // Bumped by /login litellm so a model refresh that started before the login cannot cache the old
  // credentials as the default auth.
  let loginGeneration = 0;

  async function getRuntimeAuth(
    ctx: ExtensionContext,
    definition = definitions[0]!,
  ): Promise<LiteLLMRuntimeAuth | undefined> {
    const auth = await ctx.modelRegistry.getProviderAuth(definition.name);
    if (!auth) return undefined;
    const provider = ctx.modelRegistry.getProvider(definition.name);
    const apiKey = auth.auth.apiKey;
    const baseUrl = cleanConfig(auth.auth.baseUrl) ?? cleanConfig(auth.env?.[ENV_BASE_URL]) ?? provider?.baseUrl;
    if (!baseUrl || !apiKey) return undefined;
    const runtimeRoot = requireCredentialRoot(normalizeBaseUrl(baseUrl, definition.allowInsecureHttp), definition.name);
    const headers = Object.fromEntries(
      Object.entries(auth.auth.headers ?? provider?.headers ?? {}).filter(
        (entry): entry is [string, string] => typeof entry[1] === "string",
      ),
    );
    return {
      baseUrl: runtimeRoot,
      apiKey,
      headers: Object.keys(headers).length > 0 ? headers : undefined,
      allowInsecureHttp: definition.allowInsecureHttp,
    };
  }

  function missingCredentials(definition: ProviderDefinition): string {
    const fix = definition.name === PROVIDER_NAME ? "Run /login litellm or set env vars" : "Set its baseUrl and apiKey";
    return `no credentials for ${definition.name}. ${fix}.`;
  }

  async function requireRuntimeAuth(ctx: ExtensionContext, definition = definitions[0]!): Promise<LiteLLMRuntimeAuth> {
    const auth = await getRuntimeAuth(ctx, definition);
    if (auth) return auth;
    throw new Error(missingCredentials(definition));
  }

  async function resolveDefaultRuntimeAuth(ctx?: ExtensionContext): Promise<LiteLLMRuntimeAuth> {
    if (ctx?.modelRegistry) return requireRuntimeAuth(ctx);
    if (!defaultRuntimeAuth) throw new Error("no credentials for litellm. Run /login litellm or set env vars.");
    return defaultRuntimeAuth;
  }

  async function runDiscovery(auth: LiteLLMRuntimeAuth, signal?: AbortSignal) {
    const result = await discoverModels(auth.baseUrl, auth.apiKey, {
      timeoutMs: getDiscoveryTimeoutMs(),
      signal,
      headers: auth.headers,
      allowInsecureHttp: auth.allowInsecureHttp,
      ...modelsDevDiscoveryOptions(),
      silent: !isVerboseDiscovery(),
      onProgress: isVerboseDiscovery() ? (message) => process.stderr.write(`LiteLLM: ${message}\n`) : undefined,
    });
    signal?.throwIfAborted();
    return { ...result, baseUrl: normalizeBaseUrl(auth.baseUrl, auth.allowInsecureHttp) };
  }

  // Pi's startup path only ever refreshes with allowNetwork:false, and it returns before the
  // network phase, so the provider would stay empty until the user opens /model. Discover here
  // instead, the way 1.x did, and hand the catalog over as the provider's baseline models.
  async function seedModels(definition: ProviderDefinition): Promise<Model<LiteLLMApi>[]> {
    const disabledReason = discoveryDisabledReason() ?? (isHostOffline() ? "PI_OFFLINE" : null);
    if (disabledReason) {
      if (isVerboseDiscovery()) {
        process.stderr.write(`LiteLLM (${definition.name}): startup discovery skipped (${disabledReason}).\n`);
      }
      return [];
    }
    try {
      const stored = readStoredCredential(definition.name, join(getAgentDir(), "auth.json"));
      // executeHelpers:false — activation must never run the user's key helper as a side effect,
      // so helper-backed setups keep waiting for Pi's own refresh.
      const auth = await authForCredential(definition, stored, false);
      const result = await runDiscovery(auth, AbortSignal.timeout(getSeedTimeoutMs()));
      return toNativeModels(definition.name, result.baseUrl, result.models, definition.allowInsecureHttp);
    } catch (error) {
      // No credentials yet is the normal unconfigured case; anything else is worth one line.
      if (isVerboseDiscovery()) {
        process.stderr.write(
          `LiteLLM (${definition.name}): startup discovery skipped (${error instanceof Error ? error.message : String(error)}).\n`,
        );
      }
      return [];
    }
  }

  // The MCP access check overlaps seeding rather than adding to startup. Servers registered while
  // the extension loads connect when the session starts.
  const [seeded] = await Promise.all([
    Promise.all(definitions.map(seedModels)),
    mcpEnabled ? Promise.all(definitions.map(syncMcpServer)) : undefined,
  ]);
  // Pi runs a provider's network phase only once a credential resolves, so /litellm-refresh reads an
  // unchanged count as "no credentials" rather than as a successful refresh.
  const networkRefreshAttempts = new Map<string, number>();

  for (const [index, definition] of definitions.entries()) {
    const loginHooks = {
      // Disconnect before the login can store a new credential: Pi's MCP client reads the
      // provider's current token on every request, including the session teardown, so a
      // connection still open to the old proxy root would send it the new credential.
      start: () => dropMcpServer(definition),
      complete: () => {
        loginGeneration++;
      },
    };
    const auth = createProviderAuth(
      definition,
      () => oauthRuntimeRoots.delete(definition.name),
      loginHooks,
      () => oauthRuntimeRoots.get(definition.name),
    );
    if (auth.oauth) {
      const toAuth = auth.oauth.toAuth;
      auth.oauth.toAuth = async (credential) => {
        const resolved = await toAuth(credential);
        const root = requireCredentialRoot(resolveCredentialRoot(definition, credential), definition.name);
        oauthRuntimeRoots.set(definition.name, { apiKey: credential.access, root });
        // Pin the request host to the credential root so Pi's global-API fallback for an
        // off-catalog model cannot send this credential to a stale model.baseUrl.
        return { ...resolved, baseUrl: root };
      };
    }
    const provider = createLiteLLMProvider({
      id: definition.name,
      name: definition.displayName,
      baseUrl: requestBaseUrl(definition),
      auth,
      models: seeded[index],
      allowInsecureHttp: definition.allowInsecureHttp,
      resolveCredentialRoot: (credential, requestRoot, apiKey) => {
        if (credential) return resolveCredentialRoot(definition, credential, requestRoot);
        const explicit = cleanConfig(requestRoot);
        if (explicit) return normalizeBaseUrl(explicit, definition.allowInsecureHttp);
        const oauthRuntimeRoot = oauthRuntimeRoots.get(definition.name);
        if (apiKey && oauthRuntimeRoot?.apiKey === apiKey) return oauthRuntimeRoot.root;
        return resolveCredentialRoot(definition);
      },
      discover: async (credential, signal) => {
        const disabledReason = discoveryDisabledReason();
        if (disabledReason) throw new Error(`discovery disabled (${disabledReason})`);
        return runDiscovery(await authForCredential(definition, credential), signal);
      },
    });
    Object.assign(provider, { headers: resolveHeaders(definition) });
    const refreshModels = provider.refreshModels!;
    provider.refreshModels = async (context) => {
      const refreshGeneration = loginGeneration;
      if (context.allowNetwork) {
        networkRefreshAttempts.set(definition.name, (networkRefreshAttempts.get(definition.name) ?? 0) + 1);
      }
      try {
        await refreshModels(context);
      } finally {
        if (
          definition.name === PROVIDER_NAME &&
          context.allowNetwork &&
          !discoveryDisabledReason() &&
          refreshGeneration === loginGeneration &&
          context.credential
        ) {
          // Best-effort: refreshing the cached default auth must not let a bad or placeholder
          // credential override refreshModels' own try/throw outcome via `finally`.
          try {
            const auth = await authForCredential(definition, context.credential);
            if (refreshGeneration === loginGeneration) defaultRuntimeAuth = auth;
          } catch {
            // ignored — authForCredential already reported/will report this via the paths that use it directly.
          }
        }
      }
    };
    pi.registerProvider(provider);
  }

  // Pi documents PI_OFFLINE as disabling model catalog refreshes, including the one /model starts, and a
  // provider cannot tell that refresh from an automatic one. This is the explicit refresh: it reaches only
  // the configured proxies, so LITELLM_OFFLINE=1 and a zero timeout still disable it and models.dev stays off.
  pi.registerCommand("litellm-refresh", {
    description: "Refresh LiteLLM model catalogs from the proxy, even with PI_OFFLINE set",
    getArgumentCompletions: (prefix) => {
      const matches = [...providerNames].filter((name) => name.startsWith(prefix));
      return matches.length > 0 ? matches.map((name) => ({ value: name, label: name })) : null;
    },
    handler: async (args, ctx) => {
      const notify = (message: string, level: "info" | "warning"): void => {
        if (ctx.hasUI) ctx.ui.notify(message, level);
        else process.stderr.write(`${message}\n`);
      };
      const disabledReason = discoveryDisabledReason();
      if (disabledReason) {
        notify(`LiteLLM: model refresh skipped (${disabledReason}).`, "warning");
        return;
      }
      const requested = args.trim();
      const selected = requested ? definitions.filter(({ name }) => name === requested) : definitions;
      if (selected.length === 0) {
        const configured = [...providerNames].map((name) => JSON.stringify(name)).join(", ");
        notify(`LiteLLM: unknown provider ${JSON.stringify(requested)}; configured: ${configured}.`, "warning");
        return;
      }
      const attempts = new Map(selected.map(({ name }) => [name, networkRefreshAttempts.get(name) ?? 0]));
      const result = await ctx.modelRegistry.refresh({
        providers: selected.map(({ name }) => name),
        allowNetwork: true,
        force: true,
      });
      const models = ctx.modelRegistry.getAll();
      for (const definition of selected) {
        const label = `LiteLLM (${JSON.stringify(definition.name)})`;
        const error = result.errors.get(definition.name);
        if (error) {
          notify(`${label}: model refresh failed (${error.message}).`, "warning");
        } else if ((networkRefreshAttempts.get(definition.name) ?? 0) === attempts.get(definition.name)) {
          notify(`${label}: model refresh skipped; ${missingCredentials(definition)}`, "warning");
        } else {
          const count = models.filter((model) => model.provider === definition.name).length;
          notify(`${label}: refreshed ${count} model${count === 1 ? "" : "s"}.`, "info");
        }
      }
    },
  });

  setupLiteLLMCostTracking(pi, [...providerNames]);

  if (budgetEnabled) {
    const { display, warning } = budgetDisplaySetting(settings?.budget);
    const definitionNamed = (name: string) => definitions.find((definition) => definition.name === name)!;
    setupLiteLLMBudget(pi, {
      providers: definitions.map(({ name, displayName }) => ({ name, displayName })),
      display,
      displayWarning: warning,
      resolveAuth: (ctx, name) => getRuntimeAuth(ctx, definitionNamed(name)),
      timeoutMs: getDiscoveryTimeoutMs,
      disabledReason: discoveryDisabledReason,
      hostOffline: isHostOffline,
      missingCredentials: (name) => missingCredentials(definitionNamed(name)),
    });
  }

  if (skillsEnabled) {
    for (const tool of createSkillToolDefinitions(resolveDefaultRuntimeAuth)) {
      pi.registerTool(tool);
    }
  }

  pi.on("before_provider_headers", (event, ctx) => {
    if (!ctx.model?.provider || !providerNames.has(ctx.model.provider)) return;
    event.headers["x-litellm-session-id"] = ctx.sessionManager.getSessionId();
  });

  pi.on("before_provider_request", (event, ctx) => {
    if (!ctx.model?.provider || !providerNames.has(ctx.model.provider)) return;
    if (typeof event.payload !== "object" || event.payload === null) return;
    return prepareLiteLLMRequestPayload(
      event.payload as Record<string, unknown>,
      ctx.model as LiteLLMModel,
      (ctx.model as LiteLLMModel | undefined)?.litellmPolicy,
    );
  });

  // Router fallbacks are invisible to discovery, so protocol and model policy were chosen for the
  // requested route even when another group answered. Name the route only; header text is proxy-supplied.
  const warnedFallbackRoutes = new Set<string>();
  pi.on("after_provider_response", (event, ctx) => {
    const model = ctx.model;
    if (!model?.provider || !providerNames.has(model.provider)) return;
    if (event.status < 200 || event.status >= 300) return;
    const attemptedFallbacks = Number(event.headers["x-litellm-attempted-fallbacks"]);
    if (!Number.isSafeInteger(attemptedFallbacks) || attemptedFallbacks <= 0) return;
    const key = `${model.provider}\0${model.id}`;
    if (warnedFallbackRoutes.has(key)) return;
    warnedFallbackRoutes.add(key);
    const route = JSON.stringify(model.id);
    const message =
      `LiteLLM (${JSON.stringify(model.provider)}): the request for ${route} was answered by a fallback, but protocol and model handling were chosen ` +
      `for ${route}'s own deployments, not the fallback's. If its fallbacks cross model families, pin the protocol with ` +
      "`model_info.supported_endpoints`.";
    if (ctx.hasUI) ctx.ui.notify(message, "warning");
    else process.stderr.write(`${message}\n`);
  });

  // Skills enrichment is best-effort: an expired credential or an unreachable proxy must not
  // report an extension error on every turn. The failing model request states the real problem.
  pi.on("before_agent_start", async (event, ctx) => {
    defaultRuntimeAuth = undefined;
    if (!skillsEnabled || discoveryDisabledReason()) return;
    let section: string | undefined;
    try {
      const auth = await getRuntimeAuth(ctx);
      if (!auth) return;
      defaultRuntimeAuth = auth;
      const skills = await listSkills(auth.baseUrl, auth.apiKey, auth.headers, auth.allowInsecureHttp);
      section = createSkillsPromptSection(skills);
    } catch (error) {
      if (isVerboseDiscovery()) {
        process.stderr.write(`LiteLLM: skipping Skills (${error instanceof Error ? error.message : String(error)}).\n`);
      }
      return;
    }
    if (!section) return;
    return { systemPrompt: `${event.systemPrompt}\n\n${section}` };
  });

  if (mcpEnabled) {
    // Each turn picks up a login that changed the proxy root, and a logout that removed it.
    pi.on("before_agent_start", async () => {
      await Promise.all(definitions.map(syncMcpServer));
    });
  }

  pi.on("message_end", (event, ctx) => {
    if (event.message.role !== "assistant") return;
    // Resolve the discovered model that produced this message so the display
    // conclusion comes from deployment evidence rather than the route name. An
    // unresolvable model carries no conclusion and is left untouched.
    const model = ctx?.modelRegistry?.find(event.message.provider, event.message.model) as LiteLLMModel | undefined;
    const message = normalizeThinkTags(event.message as AssistantMessage, providerNames, model);
    if (!message) return;
    return { message };
  });
}
