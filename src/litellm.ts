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
export const DEFAULT_TIMEOUT_MS = 5000;
export const LOGIN_TIMEOUT_MS = 10_000;
export const CACHE_FILENAME = "litellm-models.json";

export function getAuthPath(): string {
  return join(getAgentDir(), "auth.json");
}

export function getCachePath(): string {
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

export async function resolveCredentials(): Promise<ResolvedCredentials> {
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
