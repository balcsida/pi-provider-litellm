import { readFile } from "node:fs/promises";
import { join } from "node:path";
import type { OAuthCredentials } from "@earendil-works/pi-ai";
import { AuthStorage, getAgentDir } from "@earendil-works/pi-coding-agent";
import { normalizeBaseUrl } from "./discover.js";
import type { AuthFileEntry, ResolvedCredentials } from "./types.js";

export const PROVIDER_NAME = "litellm";
export const ENV_BASE_URL = "LITELLM_BASE_URL";
export const ENV_API_KEY = "LITELLM_API_KEY";
export const ENV_TIMEOUT = "LITELLM_DISCOVERY_TIMEOUT_MS";
export const ENV_OFFLINE = "LITELLM_OFFLINE";
export const CLI_HOST_FLAG = "litellm-host";
export const CLI_BASE_URL_FLAG = "litellm-base-url";
export const DEFAULT_TIMEOUT_MS = 5000;
export const LOGIN_TIMEOUT_MS = 10_000;
export const CACHE_FILENAME = "litellm-models.json";
const PACKAGE_NAME = "pi-provider-litellm";
const PROJECT_CONFIG_DIR = ".pi";

type PackageSettingsEntry = {
  source?: unknown;
  config?: unknown;
};

type SettingsFile = {
  packages?: unknown;
};

export function getAuthPath(): string {
  return join(getAgentDir(), "auth.json");
}

export function getCachePath(): string {
  return join(getAgentDir(), CACHE_FILENAME);
}

function getGlobalSettingsPath(): string {
  return join(getAgentDir(), "settings.json");
}

function getProjectSettingsPath(): string {
  return join(process.cwd(), PROJECT_CONFIG_DIR, "settings.json");
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

export function getCliFlagValue(name: string, argv = process.argv): string | undefined {
  const flag = `--${name}`;
  const prefix = `${flag}=`;
  let value: string | undefined;

  for (let index = 2; index < argv.length; index++) {
    const arg = argv[index];
    if (arg === "--") break;
    if (arg.startsWith(prefix)) {
      value = arg.slice(prefix.length);
      continue;
    }
    if (arg !== flag) continue;

    const next = argv[index + 1];
    if (next === undefined || next.startsWith("-")) continue;
    value = next;
    index++;
  }

  return value?.trim() || undefined;
}

export function getCliBaseUrl(): string | undefined {
  return getCliFlagValue(CLI_HOST_FLAG) ?? getCliFlagValue(CLI_BASE_URL_FLAG);
}

function getNpmPackageName(source: string): string {
  const spec = source.startsWith("npm:") ? source.slice("npm:".length).trim() : source.trim();
  const match = spec.match(/^(@?[^@]+(?:\/[^@]+)?)(?:@.+)?$/);
  return match?.[1] ?? spec;
}

function isLiteLLMPackageSource(source: unknown): boolean {
  return typeof source === "string" && getNpmPackageName(source) === PACKAGE_NAME;
}

function getPackageConfigBaseUrl(entry: PackageSettingsEntry): string | undefined {
  if (!isLiteLLMPackageSource(entry.source)) return undefined;
  if (typeof entry.config !== "object" || entry.config === null) return undefined;
  const config = entry.config as Record<string, unknown>;
  const raw = config.litellmHost ?? config.litellmBaseUrl;
  return typeof raw === "string" ? raw.trim() || undefined : undefined;
}

async function readPackageConfigBaseUrl(path: string): Promise<string | undefined> {
  try {
    const raw = await readFile(path, "utf8");
    const parsed = JSON.parse(raw) as SettingsFile;
    if (!Array.isArray(parsed.packages)) return undefined;
    for (const pkg of parsed.packages) {
      if (typeof pkg !== "object" || pkg === null || Array.isArray(pkg)) continue;
      const baseUrl = getPackageConfigBaseUrl(pkg as PackageSettingsEntry);
      if (baseUrl) return baseUrl;
    }
  } catch {
    return undefined;
  }
}

export async function getConfiguredBaseUrl(): Promise<string | undefined> {
  return (
    (await readPackageConfigBaseUrl(getProjectSettingsPath())) ?? readPackageConfigBaseUrl(getGlobalSettingsPath())
  );
}

export async function resolveCredentials(): Promise<ResolvedCredentials> {
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

export function getDiscoveryTimeoutMs(): number {
  const raw = process.env[ENV_TIMEOUT];
  if (raw === undefined) return DEFAULT_TIMEOUT_MS;
  const parsed = Number.parseInt(raw, 10);
  if (Number.isNaN(parsed) || parsed < 0) return DEFAULT_TIMEOUT_MS;
  return parsed;
}

export function isOffline(): boolean {
  return process.env[ENV_OFFLINE] === "1";
}

export function isListModelsMode(): boolean {
  return process.argv.includes("--list-models");
}

export function getSessionIdFromFile(sessionFile?: string): string | undefined {
  if (!sessionFile) return undefined;
  const filename = sessionFile
    .split("/")
    .pop()
    ?.replace(/\.jsonl$/i, "");
  if (!filename) return undefined;
  const uuidMatch = filename.match(/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$/i);
  return uuidMatch?.[1] ?? filename;
}

export type { OAuthCredentials };
