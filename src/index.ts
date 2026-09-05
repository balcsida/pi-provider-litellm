import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";
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
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { getAgentDir, readStoredCredential } from "@earendil-works/pi-coding-agent";
import { setupLiteLLMCostTracking } from "./cost.js";
import { discoverModels, emitsThinkTags, isGpt55Model, isMoonshotModel, normalizeBaseUrl } from "./discover.js";
import { getGcloudToken, hasGcloudAdcCredentials, isGcloudTokenAuthEnabled } from "./gcloud-token.js";
import { getSessionIdFromFile } from "./litellm.js";
import { createMcpToolDefinitions } from "./mcp-tools.js";
import { createLiteLLMProvider, toNativeModels } from "./provider.js";
import { createSkillsPromptSection, createSkillToolDefinitions, listSkills } from "./skills.js";
import type { LiteLLMApi, LiteLLMModel, LiteLLMRuntimeAuth, ResolvedCredentials } from "./types.js";

const PROVIDER_NAME = "litellm";
const SETTINGS_KEY = "litellm";
const ENV_BASE_URL = "LITELLM_BASE_URL";
const ENV_API_KEY = "LITELLM_API_KEY";
const GCLOUD_ADC_SOURCE = "gcloud ADC";
const ENV_API_KEY_HELPER = "LITELLM_API_KEY_HELPER";
const ENV_HEADERS = "LITELLM_HEADERS";
const ENV_TIMEOUT = "LITELLM_DISCOVERY_TIMEOUT_MS";
const ENV_CLI_JWT_EXPIRATION_HOURS = "LITELLM_CLI_JWT_EXPIRATION_HOURS";
const ENV_OFFLINE = "LITELLM_OFFLINE";
const ENV_VERBOSE_DISCOVERY = "LITELLM_VERBOSE_DISCOVERY";
const MODELS_DEV_CACHE_FILENAME = "litellm-models-dev.json";
const DEFAULT_TIMEOUT_MS = 5000;
const SEED_TIMEOUT_MS = 3000;
const LOGIN_TIMEOUT_MS = 10_000;
const CLI_SSO_POLL_INTERVAL_MS = 2_000;
const CLI_SSO_EXPIRES_IN_SECONDS = 600;
const DEFAULT_CLI_JWT_EXPIRATION_HOURS = 24;
const TOKEN_REFRESH_LEAD_MS = 5 * 60 * 1000;
const PERMANENT_TOKEN_EXPIRES_AT = Number.MAX_SAFE_INTEGER;
const EXPIRE_TOKEN_IMMEDIATELY = 0;

type RawProviderSettings = {
  displayName?: unknown;
  baseUrl?: unknown;
  apiKey?: unknown;
  headers?: unknown;
  enabled?: unknown;
  allowInsecureHttp?: unknown;
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
  allowInsecureHttp: boolean;
};

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
  if (typeof definition.headers === "string") return parseCustomHeaders(resolveTemplateConfigValue(definition.headers));
  return parseHeaderRecord(definition.headers);
}

async function resolveHeadersFromContext(
  definition: ProviderDefinition,
  env: (name: string) => Promise<string | undefined>,
): Promise<Record<string, string> | undefined> {
  if (typeof definition.headers === "string")
    return parseCustomHeaders(await resolveTemplateConfigValueFromContext(definition.headers, env));
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
  const helperKey =
    !gcloudKey && !configuredKey && executeHelpers && envHelperCommand
      ? executeApiKeyCommand(envHelperCommand)
      : undefined;
  const apiKey = gcloudKey || configuredKey || helperKey || envKey;

  let apiKeyConfig: string | undefined;
  if (configuredKey && definition.apiKeyConfig) {
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
    allowInsecureHttp: raw?.allowInsecureHttp === true,
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
  const headers = resolveHeaders(definition);
  const baseUrl = await promptBaseUrl(interaction, definition);
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
      baseUrl: baseUrl ? normalizeBaseUrl(baseUrl, definition.allowInsecureHttp) : undefined,
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
    if (!creds.baseUrl && baseUrl) creds.baseUrl = normalizeBaseUrl(baseUrl, definition.allowInsecureHttp);
  }
  if (!creds.apiKey) return undefined;
  return {
    auth: {
      apiKey: creds.apiKey,
      baseUrl: creds.baseUrl ? `${creds.baseUrl}/v1` : undefined,
      headers: await resolveHeadersFromContext(definition, ctx.env),
    },
    env: baseUrl ? { [ENV_BASE_URL]: normalizeBaseUrl(baseUrl, definition.allowInsecureHttp) } : undefined,
    source: source ?? (creds.apiKeyFromGcloudAdc ? GCLOUD_ADC_SOURCE : undefined) ?? creds.apiKeyConfig ?? ENV_API_KEY,
  };
}

function createProviderAuth(definition: ProviderDefinition): ProviderAuth {
  return {
    apiKey: {
      name: `${definition.displayName} API key`,
      login: definition.name === PROVIDER_NAME ? (interaction) => loginApiKey(interaction, definition) : undefined,
      check: async ({ ctx, credential }) => {
        const baseUrl =
          credential?.env?.[ENV_BASE_URL] ??
          definition.baseUrl ??
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
          if (definition.useDefaultEnv && cleanConfig(await ctx.env(ENV_API_KEY_HELPER))) return ENV_API_KEY_HELPER;
          if (definition.useDefaultEnv && cleanConfig(await ctx.env(ENV_API_KEY))) return ENV_API_KEY;
          return undefined;
        };

        // Mirror the precedence in `resolveCredentials`, where ADC outranks the config
        // key, the helper and the environment key. Whether the refresh token still mints
        // is only knowable at request time, and this must not make a network call; if it
        // fails, `resolve` falls back and reports the credential it actually used.
        if (definition.useGcloudTokenAuth && isGcloudTokenAuthEnabled() && (await hasGcloudAdcCredentials()))
          return { type: "api_key", source: GCLOUD_ADC_SOURCE };
        const fallback = await fallbackSource();
        return fallback ? { type: "api_key", source: fallback } : undefined;
      },
      resolve: ({ ctx, credential }) => resolveApiKeyAuth(definition, ctx, credential),
    },
    oauth: definition.enableOAuth
      ? {
          name: "LiteLLM SSO",
          loginLabel: "Sign in with LiteLLM SSO",
          login: (interaction) => loginOAuth(interaction, definition),
          refresh: async (credential, signal) => ({
            ...(await refreshLiteLLM(credential, signal)),
            type: "oauth" as const,
          }),
          toAuth: async (credential) => ({
            apiKey: credential.access,
            baseUrl: credential.baseUrl
              ? `${normalizeBaseUrl(String(credential.baseUrl), definition.allowInsecureHttp)}/v1`
              : undefined,
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

// Reasoning fields LiteLLM forwards to chat-completions providers. The Moonshot
// path defaults them off; the gpt-5.5 tool path strips them entirely.
const REASONING_SUPPRESSION_DEFAULTS: Record<string, unknown> = {
  include_reasoning: false,
  reasoning_content: false,
  merge_reasoning_content_in_choices: true,
  thinking: { type: "disabled" },
};
const GEMINI_MODEL_PATTERN = /(?:^|[-_/.:])gemini(?=$|[-_/.:])/i;

function prepareLiteLLMRequestPayload(
  payload: Record<string, unknown>,
  model: LiteLLMModel | undefined,
  sessionId: string | undefined,
): Record<string, unknown> | undefined {
  const modelId = model?.id;
  const api = model?.api;
  let next: Record<string, unknown> | undefined;
  const update = (key: string, value: unknown): void => {
    if (payload[key] !== undefined) return;
    next ??= { ...payload };
    next[key] = value;
  };

  if (api !== "openai-responses" && model?.suppressReasoningContent === true) {
    for (const [key, value] of Object.entries(REASONING_SUPPRESSION_DEFAULTS)) {
      if (key !== "thinking") update(key, value);
    }
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

  // Moonshot/Kimi applies strict OpenAI schema validation: assistant tool calls
  // must carry string content, and tool results must be plain text.
  if (modelId && isMoonshotModel(modelId)) {
    const messages = (next ?? payload).messages;
    if (Array.isArray(messages)) {
      const normalized = normalizeStrictToolMessages(messages);
      if (normalized !== messages) {
        next ??= { ...payload };
        next.messages = normalized;
      }
    }
  }

  if (modelId && GEMINI_MODEL_PATTERN.test(modelId)) {
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
  if (!litellmProviderNames.has(message.provider) || !emitsThinkTags(message.model)) return;

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
    return `${normalizeBaseUrl(baseUrl, definition.allowInsecureHttp)}/v1`;
  }

  async function authForCredential(definition: ProviderDefinition, credential?: Credential, executeHelpers = true) {
    if (credential?.type === "oauth") {
      const baseUrl =
        typeof credential.baseUrl === "string"
          ? normalizeBaseUrl(credential.baseUrl, definition.allowInsecureHttp)
          : normalizeBaseUrl(requestBaseUrl(definition), definition.allowInsecureHttp);
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
    if (!resolved?.auth.apiKey) {
      throw new Error(`no credentials for ${definition.name}. Run /login litellm or set env vars.`);
    }
    return {
      baseUrl: normalizeBaseUrl(resolved.auth.baseUrl ?? requestBaseUrl(definition), definition.allowInsecureHttp),
      apiKey: resolved.auth.apiKey,
      headers: resolved.auth.headers,
      allowInsecureHttp: definition.allowInsecureHttp,
    };
  }

  let registeredMcpIdentity: string | undefined;
  let mcpRegistration: Promise<void> | undefined;
  let defaultRuntimeAuth: LiteLLMRuntimeAuth | undefined;

  function mcpCatalogIdentity(auth: LiteLLMRuntimeAuth): string {
    return JSON.stringify({
      baseUrl: normalizeBaseUrl(auth.baseUrl, auth.allowInsecureHttp),
      apiKey: auth.apiKey,
      headers: Object.entries(auth.headers ?? {}).sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0)),
    });
  }

  async function getRuntimeAuth(ctx: ExtensionContext): Promise<LiteLLMRuntimeAuth | undefined> {
    const auth = await ctx.modelRegistry.getProviderAuth(PROVIDER_NAME);
    if (!auth) return undefined;
    const provider = ctx.modelRegistry.getProvider(PROVIDER_NAME);
    const baseUrl = auth.auth.baseUrl ?? provider?.baseUrl;
    const apiKey = auth.auth.apiKey;
    if (!baseUrl || !apiKey) return undefined;
    const headers = Object.fromEntries(
      Object.entries(auth.auth.headers ?? provider?.headers ?? {}).filter(
        (entry): entry is [string, string] => typeof entry[1] === "string",
      ),
    );
    return {
      baseUrl: normalizeBaseUrl(baseUrl, definitions[0]?.allowInsecureHttp),
      apiKey,
      headers: Object.keys(headers).length > 0 ? headers : undefined,
      allowInsecureHttp: definitions[0]?.allowInsecureHttp,
    };
  }

  async function resolveDefaultRuntimeAuth(ctx?: ExtensionContext): Promise<LiteLLMRuntimeAuth> {
    if (ctx?.modelRegistry) {
      const auth = await getRuntimeAuth(ctx);
      if (auth) return auth;
      throw new Error("no credentials for litellm. Run /login litellm or set env vars.");
    }
    if (!defaultRuntimeAuth) throw new Error("no credentials for litellm. Run /login litellm or set env vars.");
    return defaultRuntimeAuth;
  }

  function waitForMcpRegistration(registration: Promise<void>, signal?: AbortSignal): Promise<void> {
    if (!signal) return registration.catch(() => undefined);
    signal.throwIfAborted();
    return new Promise((resolve, reject) => {
      const onAbort = () => {
        signal.removeEventListener("abort", onAbort);
        reject(signal.reason);
      };
      const onComplete = () => {
        signal.removeEventListener("abort", onAbort);
        resolve();
      };
      signal.addEventListener("abort", onAbort, { once: true });
      if (signal.aborted) onAbort();
      else registration.then(onComplete, onComplete);
    });
  }

  async function registerMcpTools(auth: LiteLLMRuntimeAuth, signal?: AbortSignal): Promise<void> {
    if (!mcpEnabled || discoveryDisabledReason()) return;
    const identity = mcpCatalogIdentity(auth);
    while (mcpRegistration) {
      await waitForMcpRegistration(mcpRegistration, signal);
      signal?.throwIfAborted();
      if (registeredMcpIdentity === identity) return;
    }
    if (registeredMcpIdentity === identity) return;

    const registration = (async () => {
      try {
        signal?.throwIfAborted();
        const tools = await createMcpToolDefinitions(
          (ctx) => (ctx?.modelRegistry ? resolveDefaultRuntimeAuth(ctx) : Promise.resolve(auth)),
          isVerboseDiscovery() ? (message) => process.stderr.write(`LiteLLM MCP: ${message}\n`) : undefined,
          signal,
        );
        signal?.throwIfAborted();
        for (const tool of tools) pi.registerTool(tool);
        registeredMcpIdentity = identity;
      } catch (error) {
        if (signal?.aborted) throw signal.reason;
        process.stderr.write(
          `LiteLLM (${PROVIDER_NAME}): MCP tool discovery failed (${error instanceof Error ? error.message : String(error)}).\n`,
        );
      }
    })();
    mcpRegistration = registration;
    try {
      await registration;
    } finally {
      if (mcpRegistration === registration) mcpRegistration = undefined;
    }
  }

  async function runDiscovery(auth: LiteLLMRuntimeAuth, signal?: AbortSignal) {
    const result = await discoverModels(auth.baseUrl, auth.apiKey, {
      timeoutMs: getDiscoveryTimeoutMs(),
      signal,
      headers: auth.headers,
      allowInsecureHttp: auth.allowInsecureHttp,
      modelsDev: !isHostOffline(),
      modelsDevCachePath: join(getAgentDir(), MODELS_DEV_CACHE_FILENAME),
      silent: !isVerboseDiscovery(),
      onProgress: isVerboseDiscovery() ? (message) => process.stderr.write(`LiteLLM: ${message}\n`) : undefined,
    });
    signal?.throwIfAborted();
    return { ...result, baseUrl: `${normalizeBaseUrl(auth.baseUrl, auth.allowInsecureHttp)}/v1` };
  }

  // Pi's startup path only ever refreshes with allowNetwork:false, and it returns before the
  // network phase, so the provider would stay empty until the user opens /model. Discover here
  // instead, the way 1.x did, and hand the catalog over as the provider's baseline models.
  async function seedModels(definition: ProviderDefinition): Promise<Model<LiteLLMApi>[]> {
    if (discoveryDisabledReason() || isHostOffline()) return [];
    try {
      const stored = readStoredCredential(definition.name, join(getAgentDir(), "auth.json"));
      // executeHelpers:false — activation must never run the user's key helper as a side effect,
      // so helper-backed setups keep waiting for Pi's own refresh.
      const auth = await authForCredential(definition, stored, false);
      const result = await runDiscovery(auth, AbortSignal.timeout(getSeedTimeoutMs()));
      return toNativeModels(definition.name, result.baseUrl, result.models);
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

  const seeded = await Promise.all(definitions.map(seedModels));

  for (const [index, definition] of definitions.entries()) {
    const provider = createLiteLLMProvider({
      id: definition.name,
      name: definition.displayName,
      baseUrl: requestBaseUrl(definition),
      auth: createProviderAuth(definition),
      models: seeded[index],
      discover: async (credential, signal) => {
        const disabledReason = discoveryDisabledReason();
        if (disabledReason) throw new Error(`discovery disabled (${disabledReason})`);
        return runDiscovery(await authForCredential(definition, credential), signal);
      },
    });
    Object.assign(provider, { headers: resolveHeaders(definition) });
    const refreshModels = provider.refreshModels!;
    provider.refreshModels = async (context) => {
      try {
        await refreshModels(context);
      } finally {
        if (
          definition.name === PROVIDER_NAME &&
          context.allowNetwork &&
          !discoveryDisabledReason() &&
          context.credential
        ) {
          const auth = await authForCredential(definition, context.credential);
          defaultRuntimeAuth = auth;
          void registerMcpTools(auth, context.signal).catch(() => undefined);
        }
      }
    };
    pi.registerProvider(provider);
  }

  setupLiteLLMCostTracking(pi, [...providerNames]);

  if (skillsEnabled) {
    for (const tool of createSkillToolDefinitions(resolveDefaultRuntimeAuth)) {
      pi.registerTool(tool);
    }
  }

  let sessionId: string | undefined;
  pi.on("session_start", (_event, ctx) => {
    sessionId = getSessionIdFromFile(ctx.sessionManager.getSessionFile());
  });

  pi.on("before_provider_request", (event, ctx) => {
    if (!ctx.model?.provider || !providerNames.has(ctx.model.provider)) return;
    if (typeof event.payload !== "object" || event.payload === null) return;
    return prepareLiteLLMRequestPayload(event.payload as Record<string, unknown>, ctx.model as LiteLLMModel, sessionId);
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

  pi.on("message_end", (event) => {
    if (event.message.role !== "assistant") return;
    const message = normalizeThinkTags(event.message as AssistantMessage, providerNames);
    if (!message) return;
    return { message };
  });
}
