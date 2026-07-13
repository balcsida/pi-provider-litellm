// CI auth smoke test against a real LiteLLM proxy.
// Reads LITELLM_BASE_URL, LITELLM_API_KEY, LITELLM_CLI_SMOKE_MODEL, and optional LITELLM_LICENSE.
// Run: npx tsx scripts/smoke-auth.ts

import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { normalizeBaseUrl } from "../src/discover.js";

const DEFAULT_TIMEOUT_MS = 30_000;
const BAD_SMOKE_KEY = "bad-smoke-key";
const SMOKE_PROMPT = "Reply with one short word.";

export type AuthSmokeResult = {
  enterprise: boolean;
  checks: string[];
};

export type AuthSmokeOptions = {
  baseUrl: string;
  masterKey: string;
  modelId: string;
  timeoutMs?: number;
  enterprise?: boolean;
};

type ChatCompletionResponse = {
  choices?: Array<{
    message?: {
      content?: unknown;
    };
  }>;
};

type KeyGenerateResponse = {
  key?: unknown;
};

type SmokeOAuthCredential = {
  access: string;
};

type SmokeProviderConfig = {
  oauth?: {
    login: (callbacks: {
      onPrompt: (options: { message: string; placeholder?: string }) => Promise<string>;
      onAuth?: (info: { url: string; instructions?: string }) => void;
      onProgress?: (message: string) => void;
      signal?: AbortSignal;
    }) => Promise<SmokeOAuthCredential>;
  };
};

type SmokePi = {
  providers: Array<{ name: string; config: SmokeProviderConfig }>;
  registerProvider: (name: string, config: SmokeProviderConfig) => void;
  registerCommand: () => void;
  registerTool: () => void;
  on: () => void;
};

function withTimeout(timeoutMs: number): { signal: AbortSignal; cancel: () => void } {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error(`Timed out after ${timeoutMs}ms`)), timeoutMs);
  return {
    signal: controller.signal,
    cancel: () => clearTimeout(timer),
  };
}

async function readErrorBody(response: Response): Promise<string> {
  try {
    const body = await response.text();
    return body ? `: ${body.slice(0, 500)}` : "";
  } catch {
    return "";
  }
}

async function fetchWithTimeout(url: string, init: RequestInit, timeoutMs: number): Promise<Response> {
  const { signal, cancel } = withTimeout(timeoutMs);
  try {
    return await fetch(url, { ...init, signal });
  } finally {
    cancel();
  }
}

function authHeaders(apiKey?: string): Record<string, string> {
  const headers: Record<string, string> = { Accept: "application/json" };
  if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
  return headers;
}

function jsonHeaders(apiKey: string): Record<string, string> {
  return {
    ...authHeaders(apiKey),
    "Content-Type": "application/json",
  };
}

function isAuthFailure(status: number): boolean {
  return status === 401 || status === 403;
}

async function expectAuthFailure(label: string, response: Response): Promise<void> {
  if (!isAuthFailure(response.status)) {
    throw new Error(`${label} should reject auth, got ${response.status}${await readErrorBody(response)}`);
  }
}

async function expectOk(label: string, response: Response): Promise<void> {
  if (!response.ok) {
    throw new Error(`${label} returned ${response.status}${await readErrorBody(response)}`);
  }
}

async function fetchModels(baseUrl: string, apiKey: string | undefined, timeoutMs: number): Promise<Response> {
  return fetchWithTimeout(
    `${baseUrl}/v1/models`,
    {
      method: "GET",
      headers: authHeaders(apiKey),
    },
    timeoutMs,
  );
}

async function smokeChat(baseUrl: string, apiKey: string, modelId: string, timeoutMs: number): Promise<void> {
  const response = await fetchWithTimeout(
    `${baseUrl}/v1/chat/completions`,
    {
      method: "POST",
      headers: jsonHeaders(apiKey),
      body: JSON.stringify({
        model: modelId,
        messages: [{ role: "user", content: SMOKE_PROMPT }],
        max_tokens: 16,
        temperature: 0,
      }),
    },
    timeoutMs,
  );
  await expectOk(`/v1/chat/completions for ${modelId}`, response);

  const data = (await response.json()) as ChatCompletionResponse;
  const content = data.choices?.[0]?.message?.content;
  if (typeof content !== "string" || content.trim().length === 0) {
    throw new Error(`/v1/chat/completions for ${modelId} returned no assistant text`);
  }
}

async function generateVirtualKey(
  baseUrl: string,
  masterKey: string,
  modelId: string,
  timeoutMs: number,
): Promise<string> {
  const response = await fetchWithTimeout(
    `${baseUrl}/key/generate`,
    {
      method: "POST",
      headers: jsonHeaders(masterKey),
      body: JSON.stringify({ models: [modelId], duration: "1h" }),
    },
    timeoutMs,
  );
  await expectOk("/key/generate with master key", response);

  const data = (await response.json()) as KeyGenerateResponse;
  if (typeof data.key !== "string" || data.key.length === 0) {
    throw new Error("/key/generate returned no key");
  }
  return data.key;
}

async function expectAdminOnlyKeyGenerate(
  baseUrl: string,
  apiKey: string,
  modelId: string,
  timeoutMs: number,
): Promise<void> {
  const response = await fetchWithTimeout(
    `${baseUrl}/key/generate`,
    {
      method: "POST",
      headers: jsonHeaders(apiKey),
      body: JSON.stringify({ models: [modelId], duration: "1h" }),
    },
    timeoutMs,
  );
  await expectAuthFailure("/key/generate with virtual key", response);
}

function createSmokePi(): SmokePi {
  return {
    providers: [],
    registerProvider(name, config) {
      this.providers.push({ name, config });
    },
    registerCommand: () => undefined,
    registerTool: () => undefined,
    on: () => undefined,
  };
}

export async function runSsoLoginSmoke(
  options: Required<Pick<AuthSmokeOptions, "baseUrl" | "masterKey" | "modelId">> & {
    timeoutMs: number;
  },
): Promise<void> {
  const baseUrl = normalizeBaseUrl(options.baseUrl);
  const previousAgentDir = process.env.PI_CODING_AGENT_DIR;
  const previousBaseUrl = process.env.LITELLM_BASE_URL;
  const previousApiKey = process.env.LITELLM_API_KEY;
  process.env.PI_CODING_AGENT_DIR ??= await mkdtemp(join(tmpdir(), "pi-litellm-sso-smoke-"));
  process.env.LITELLM_BASE_URL = baseUrl;
  process.env.LITELLM_API_KEY = options.masterKey;
  try {
    const extension = (await import("../src/index.js")).default as unknown as (pi: SmokePi) => Promise<void>;
    const pi = createSmokePi();
    await extension(pi);

    const oauth = pi.providers.find((provider) => provider.name === "litellm")?.config.oauth;
    if (!oauth) throw new Error("LiteLLM provider did not expose OAuth login");

    const authInfos: Array<{ url: string; instructions?: string }> = [];
    const credential = await oauth.login({
      onAuth: (info) => authInfos.push(info),
      onPrompt: async ({ message, placeholder }) => {
        if (placeholder) return baseUrl;
        if (message.includes("Select login method")) return "2";
        if (message.includes("SSO token")) return `Bearer ${options.masterKey}`;
        if (message.includes("Generate a LiteLLM virtual key")) return "y";
        return "";
      },
      signal: new AbortController().signal,
    });

    if (!authInfos.some((info) => info.url === `${baseUrl}/sso/key/generate`)) {
      throw new Error("SSO login did not request /sso/key/generate");
    }
    if (!credential.access || credential.access === options.masterKey) {
      throw new Error("SSO login did not return a generated virtual key");
    }

    await smokeChat(baseUrl, credential.access, options.modelId, options.timeoutMs);
  } finally {
    if (previousAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
    else process.env.PI_CODING_AGENT_DIR = previousAgentDir;
    if (previousBaseUrl === undefined) delete process.env.LITELLM_BASE_URL;
    else process.env.LITELLM_BASE_URL = previousBaseUrl;
    if (previousApiKey === undefined) delete process.env.LITELLM_API_KEY;
    else process.env.LITELLM_API_KEY = previousApiKey;
  }
}

function firstSmokeModel(env: NodeJS.ProcessEnv): string | undefined {
  const cliModel = env.LITELLM_CLI_SMOKE_MODEL?.trim();
  if (cliModel) return cliModel;
  return (env.LITELLM_SMOKE_MODELS ?? "")
    .split(/[,\s]+/)
    .map((model) => model.trim())
    .find((model) => model.length > 0);
}

export async function runAuthSmoke(options: AuthSmokeOptions): Promise<AuthSmokeResult> {
  const baseUrl = normalizeBaseUrl(options.baseUrl);
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const checks: string[] = [];

  await expectAuthFailure("missing-token /v1/models", await fetchModels(baseUrl, undefined, timeoutMs));
  checks.push("missing-token");

  await expectAuthFailure("bad-token /v1/models", await fetchModels(baseUrl, BAD_SMOKE_KEY, timeoutMs));
  checks.push("bad-token");

  await expectOk("master-key /v1/models", await fetchModels(baseUrl, options.masterKey, timeoutMs));
  checks.push("master-key-models");

  await smokeChat(baseUrl, options.masterKey, options.modelId, timeoutMs);
  checks.push("master-key-chat");

  if (options.enterprise) {
    const virtualKey = await generateVirtualKey(baseUrl, options.masterKey, options.modelId, timeoutMs);
    await smokeChat(baseUrl, virtualKey, options.modelId, timeoutMs);
    checks.push("virtual-key-chat");

    await expectAdminOnlyKeyGenerate(baseUrl, virtualKey, options.modelId, timeoutMs);
    checks.push("enterprise-admin-route");

    await runSsoLoginSmoke({
      baseUrl,
      masterKey: options.masterKey,
      modelId: options.modelId,
      timeoutMs,
    });
    checks.push("sso-login", "sso-virtual-key-chat");
  }

  return {
    enterprise: Boolean(options.enterprise),
    checks,
  };
}

export async function runAuthSmokeFromEnv(env: NodeJS.ProcessEnv = process.env): Promise<AuthSmokeResult> {
  const baseUrl = env.LITELLM_BASE_URL?.trim();
  const masterKey = env.LITELLM_API_KEY?.trim();
  const modelId = firstSmokeModel(env);
  if (!baseUrl || !masterKey || !modelId) {
    throw new Error("LITELLM_BASE_URL, LITELLM_API_KEY, and LITELLM_CLI_SMOKE_MODEL must be set");
  }

  const timeoutMs = env.LITELLM_SMOKE_TIMEOUT_MS
    ? Number.parseInt(env.LITELLM_SMOKE_TIMEOUT_MS, 10)
    : DEFAULT_TIMEOUT_MS;

  return runAuthSmoke({
    baseUrl,
    masterKey,
    modelId,
    timeoutMs: Number.isNaN(timeoutMs) || timeoutMs <= 0 ? DEFAULT_TIMEOUT_MS : timeoutMs,
    enterprise: Boolean(env.LITELLM_LICENSE?.trim()),
  });
}

async function main(): Promise<void> {
  const result = await runAuthSmokeFromEnv();
  console.log(`Enterprise auth smoke: ${result.enterprise ? "enabled" : "skipped"}`);
  for (const check of result.checks) {
    console.log(`Auth smoke OK: ${check}`);
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((err) => {
    console.error(err instanceof Error ? err.message : err);
    process.exit(1);
  });
}
