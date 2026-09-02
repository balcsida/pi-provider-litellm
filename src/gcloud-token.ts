import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";

export const CACHE_TTL_MS = 50 * 60 * 1000;
export const GCLOUD_TOKEN_CACHE_KEY = "gcloud-adc";

const ADC_FILENAME = "application_default_credentials.json";

let cachedToken: string | null = null;
let cachedAt = 0;
let cachedTokenKey: string | null = null;

interface AuthorizedUserCredentials {
  type: "authorized_user";
  client_id: string;
  client_secret: string;
  refresh_token: string;
}

interface ServiceAccountCredentials {
  type: "service_account";
}

type GoogleCredentials = AuthorizedUserCredentials | ServiceAccountCredentials | { type?: string };

function isAuthorizedUserCredentials(credentials: GoogleCredentials): credentials is AuthorizedUserCredentials {
  const authorizedUser = credentials as Partial<AuthorizedUserCredentials>;
  return (
    credentials.type === "authorized_user" &&
    typeof authorizedUser.client_id === "string" &&
    authorizedUser.client_id.trim() !== "" &&
    typeof authorizedUser.client_secret === "string" &&
    authorizedUser.client_secret.trim() !== "" &&
    typeof authorizedUser.refresh_token === "string" &&
    authorizedUser.refresh_token.trim() !== ""
  );
}

export function isGcloudTokenAuthEnabled(): boolean {
  const raw = process.env.LITELLM_GCLOUD_TOKEN_AUTH;
  return raw !== undefined && raw !== "" && raw !== "0";
}

function getAdcPath(): string | null {
  const envPath = process.env.GOOGLE_APPLICATION_CREDENTIALS;
  if (envPath) return envPath;

  const candidates = [join(homedir(), ".config", "gcloud", ADC_FILENAME)];
  if (process.env.APPDATA) candidates.push(join(process.env.APPDATA, "gcloud", ADC_FILENAME));

  return candidates.find((path) => existsSync(path)) ?? null;
}

function getCredentialsCacheKey(credentials: AuthorizedUserCredentials): string {
  return `${GCLOUD_TOKEN_CACHE_KEY}:authorized_user:${credentials.client_id}:${credentials.refresh_token}`;
}

async function readCredentials(path: string): Promise<GoogleCredentials | null> {
  try {
    return JSON.parse(await readFile(path, "utf8")) as GoogleCredentials;
  } catch {
    return null;
  }
}

// Single ADC resolution path: locate the file, read it, and either return usable
// authorized_user credentials or warn with the specific reason they are not.
async function resolveAuthorizedUserAdc(): Promise<AuthorizedUserCredentials | null> {
  const adcPath = getAdcPath();
  if (!adcPath) {
    console.warn(
      "LiteLLM gcloud auth: No Google ADC file found. Set GOOGLE_APPLICATION_CREDENTIALS or run `gcloud auth application-default login`.",
    );
    return null;
  }

  const credentials = await readCredentials(adcPath);
  if (!credentials) {
    console.warn(`LiteLLM gcloud auth: Failed to read ADC file: ${adcPath}`);
    return null;
  }

  if (isAuthorizedUserCredentials(credentials)) return credentials;

  if (credentials.type === "authorized_user") {
    console.warn("LiteLLM gcloud auth: authorized_user ADC has invalid or incomplete required fields.");
    return null;
  }

  if (credentials.type === "service_account") {
    console.warn("LiteLLM gcloud auth: Service account credentials are not supported; use authorized_user ADC.");
    return null;
  }

  console.warn(`LiteLLM gcloud auth: Unknown credential type: ${credentials.type ?? "missing"}`);
  return null;
}

// Reports whether a complete `authorized_user` ADC file is present, emitting the same
// diagnostics as the token path. This is a check on credential *shape*: whether the
// refresh token is still honored by Google is only knowable at mint time, and
// `apiKey.check` deliberately performs no network call.
export async function hasGcloudAdcCredentials(): Promise<boolean> {
  return (await resolveAuthorizedUserAdc()) !== null;
}

async function exchangeRefreshToken(credentials: AuthorizedUserCredentials): Promise<string | null> {
  const body = new URLSearchParams({
    grant_type: "refresh_token",
    client_id: credentials.client_id,
    client_secret: credentials.client_secret,
    refresh_token: credentials.refresh_token,
  }).toString();

  try {
    const response = await fetch("https://oauth2.googleapis.com/token", {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body,
      signal: AbortSignal.timeout(10_000),
    });

    if (!response.ok) {
      console.warn(`LiteLLM gcloud auth: Token exchange failed (${response.status})`);
      return null;
    }

    const data = (await response.json()) as { access_token?: unknown };
    return typeof data.access_token === "string" && data.access_token ? data.access_token : null;
  } catch (error) {
    console.warn(
      `LiteLLM gcloud auth: Token exchange failed: ${error instanceof Error ? error.message : String(error)}`,
    );
    return null;
  }
}

export async function getGcloudToken(): Promise<string | null> {
  const credentials = await resolveAuthorizedUserAdc();
  if (!credentials) return null;

  const cacheKey = getCredentialsCacheKey(credentials);
  if (cachedToken && cachedTokenKey === cacheKey && Date.now() - cachedAt < CACHE_TTL_MS) return cachedToken;

  const token = await exchangeRefreshToken(credentials);
  if (token) {
    cachedToken = token;
    cachedTokenKey = cacheKey;
    cachedAt = Date.now();
  }
  return token;
}

export function resetGcloudTokenCache(): void {
  cachedToken = null;
  cachedTokenKey = null;
  cachedAt = 0;
}
