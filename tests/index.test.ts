import { execSync } from "node:child_process";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { fingerprint } from "../src/cache.js";

const ENV_KEYS = [
  "LITELLM_BASE_URL",
  "LITELLM_API_KEY",
  "LITELLM_API_KEY_HELPER",
  "LITELLM_DISCOVERY_TIMEOUT_MS",
  "STORED_LITELLM_KEY",
];
const ORIGINAL_ENV = new Map(ENV_KEYS.map((key) => [key, process.env[key]]));

type TestProviderConfig = {
  baseUrl?: string;
  apiKey?: string;
  models?: unknown[];
  oauth?: {
    login: (callbacks: {
      onPrompt: (options: { message: string; placeholder?: string }) => Promise<string>;
      onProgress?: (message: string) => void;
      signal?: AbortSignal;
    }) => Promise<{ access: string; refresh: string; expires: number; baseUrl?: string }>;
    refreshToken: (credential: { access: string; refresh: string; expires: number; baseUrl?: string }) => Promise<{
      access: string;
      refresh: string;
      expires: number;
      baseUrl?: string;
    }>;
    getApiKey: (credential: { access: string; refresh: string; expires: number; baseUrl?: string }) => string;
  };
};

type TestCommand = {
  description: string;
  handler: (args: string[], ctx: { ui: { notify: (message: string, type: string) => void } }) => Promise<void> | void;
};

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

async function makeAgentDir(): Promise<string> {
  return mkdtemp(join(tmpdir(), "pi-litellm-index-"));
}

function makeJwt(expSeconds: number): string {
  const encode = (value: unknown): string => Buffer.from(JSON.stringify(value)).toString("base64url");
  return `${encode({ alg: "none" })}.${encode({ exp: expSeconds })}.sig`;
}

async function writeHelper(
  agentDir: string,
  tokens: string[],
  helperPath = join(agentDir, "litellm-token-helper.sh"),
): Promise<string> {
  await writeFile(
    helperPath,
    `#!/usr/bin/env bash\ncount_file="${join(agentDir, "helper-count")}"\ncount=0\n[ -f "$count_file" ] && count=$(cat "$count_file")\ncase "$count" in\n${tokens.map((token, index) => `  ${index}) printf %s '${token}' ;;`).join("\n")}\n  *) printf %s '${tokens.at(-1)}' ;;\nesac\necho $((count + 1)) > "$count_file"\n`,
    { mode: 0o700 },
  );
  return helperPath;
}

async function writeFailingHelper(agentDir: string): Promise<string> {
  const helperPath = join(agentDir, "litellm-token-helper.sh");
  await writeFile(
    helperPath,
    `#!/usr/bin/env bash\ncount_file="${join(agentDir, "helper-count")}"\ncount=0\n[ -f "$count_file" ] && count=$(cat "$count_file")\necho $((count + 1)) > "$count_file"\nprintf 'idp offline' >&2\nexit 42\n`,
    { mode: 0o700 },
  );
  return helperPath;
}

async function readHelperCount(agentDir: string): Promise<number> {
  try {
    return Number(await readFile(join(agentDir, "helper-count"), "utf8"));
  } catch {
    return 0;
  }
}

const cachedModels = [{ id: "cached-model", name: "cached-model", provider: "litellm" }];

async function writeModelCache(agentDir: string, helperPath: string): Promise<void> {
  await writeFile(
    join(agentDir, "litellm-models.json"),
    JSON.stringify({
      baseUrl: "https://litellm.example.com",
      apiKeyFingerprint: fingerprint(`!${helperPath}`),
      fetchedAt: Date.now(),
      source: "model_info",
      models: cachedModels,
    }),
    "utf8",
  );
}

async function loadExtension(agentDir: string): Promise<(pi: TestPi) => Promise<void>> {
  vi.resetModules();
  vi.doMock("@earendil-works/pi-coding-agent", () => {
    class TestAuthStorage {
      constructor(private readonly authPath: string) {}

      static create(authPath: string): TestAuthStorage {
        return new TestAuthStorage(authPath);
      }

      async getApiKey(provider: string): Promise<string | undefined> {
        const parsed = JSON.parse(await readFile(this.authPath, "utf8")) as Record<
          string,
          { type: "api_key"; key: string } | { type: "oauth"; access: string; expires: number; refresh: string }
        >;
        const entry = parsed[provider];
        if (entry?.type === "api_key") return process.env[entry.key] || entry.key;
        if (entry?.type === "oauth") return entry.access;
        return undefined;
      }
    }

    return { AuthStorage: TestAuthStorage, getAgentDir: () => agentDir };
  });
  const mod = await import("../src/index.js");
  return mod.default as unknown as (pi: TestPi) => Promise<void>;
}

function createPi(): TestPi {
  return {
    providers: [],
    commands: new Map(),
    handlers: new Map(),
    registerProvider(name: string, config: TestProviderConfig) {
      this.providers.push({ name, config });
    },
    registerCommand(name: string, command: TestCommand) {
      this.commands.set(name, command);
    },
    on(event: string, handler: (event: any, ctx?: any) => Promise<any> | any) {
      this.handlers.set(event, [...(this.handlers.get(event) ?? []), handler]);
    },
  };
}

type TestPi = {
  providers: Array<{ name: string; config: TestProviderConfig }>;
  commands: Map<string, TestCommand>;
  handlers: Map<string, Array<(event: any, ctx?: any) => Promise<any> | any>>;
  registerProvider(name: string, config: TestProviderConfig): void;
  registerCommand(name: string, command: TestCommand): void;
  on(event: string, handler: (event: any, ctx?: any) => Promise<any> | any): void;
};

afterEach(() => {
  for (const key of ENV_KEYS) {
    const original = ORIGINAL_ENV.get(key);
    if (original === undefined) process.env[key] = undefined;
    else process.env[key] = original;
  }
  vi.restoreAllMocks();
  vi.resetModules();
  vi.unmock("@earendil-works/pi-coding-agent");
});

describe("extension startup", () => {
  it("discovers with the resolved stored auth key before LITELLM_API_KEY", async () => {
    const agentDir = await makeAgentDir();
    await writeFile(
      join(agentDir, "auth.json"),
      JSON.stringify({ litellm: { type: "api_key", key: "STORED_LITELLM_KEY" } }),
      "utf8",
    );
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    process.env.STORED_LITELLM_KEY = "stored-key";
    const seenAuthHeaders: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, init) => {
      seenAuthHeaders.push(new Headers(init?.headers).get("authorization") ?? "");
      return jsonResponse(200, {
        data: [{ model_name: "openai/gpt-4o", model_info: { mode: "chat" } }],
      });
    });

    const extension = await loadExtension(agentDir);
    await extension(createPi());

    expect(seenAuthHeaders).toEqual(["Bearer stored-key"]);
  });

  it("does not fetch on refresh when discovery timeout is zero", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        data: [{ model_name: "openai/gpt-4o", model_info: { mode: "chat" } }],
      }),
    );
    const notify = vi.fn();

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    await pi.commands.get("litellm-refresh")?.handler([], { ui: { notify } });

    expect(fetchMock).not.toHaveBeenCalled();
    expect(notify).toHaveBeenCalledWith("LiteLLM refresh disabled (LITELLM_DISCOVERY_TIMEOUT_MS=0)", "warning");
  });

  it("prompts during login and caches discovered models", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const seenRequests: Array<{ url: string; authorization: string }> = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      seenRequests.push({
        url,
        authorization: new Headers(init?.headers).get("authorization") ?? "",
      });
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [{ model_name: "vidaimock-openai", model_info: { mode: "chat" } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const promptMessages: string[] = [];
    const progress = vi.fn();
    const credential = await pi.providers[0]?.config.oauth?.login({
      onPrompt: async (options) => {
        promptMessages.push(options.message);
        return promptMessages.length === 1 ? " http://127.0.0.1:4000/v1 " : " sk-login ";
      },
      onProgress: progress,
      signal: new AbortController().signal,
    });

    expect(promptMessages).toEqual(["Enter LiteLLM proxy URL (no trailing /v1):", "Enter API key:"]);
    expect(seenRequests).toEqual([{ url: "http://127.0.0.1:4000/model/info", authorization: "Bearer sk-login" }]);
    expect(credential).toMatchObject({
      access: "sk-login",
      refresh: "",
      baseUrl: "http://127.0.0.1:4000",
    });
    const cache = JSON.parse(await readFile(join(agentDir, "litellm-models.json"), "utf8")) as {
      models: Array<{ id: string }>;
    };
    expect(cache.models.map((model) => model.id)).toEqual(["vidaimock-openai"]);
    expect(progress).toHaveBeenCalledWith("LiteLLM: 1 models discovered (source: model_info)");
  });

  it("uses the login cache timestamp for later stale auto-refresh", async () => {
    const agentDir = await makeAgentDir();
    delete process.env.LITELLM_BASE_URL;
    delete process.env.LITELLM_API_KEY;
    delete process.env.LITELLM_DISCOVERY_TIMEOUT_MS;
    const loginTime = new Date("2026-05-01T00:00:00.000Z").getTime();
    vi.spyOn(Date, "now").mockReturnValue(loginTime);

    let callCount = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        callCount++;
        return jsonResponse(200, {
          data: [{ model_name: `vidaimock-openai-${callCount}`, model_info: { mode: "chat" } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    expect(callCount).toBe(0);

    const credential = await pi.providers[0]?.config.oauth?.login({
      onPrompt: async (options) => (options.placeholder ? " http://127.0.0.1:4000/v1 " : " sk-login "),
      signal: new AbortController().signal,
    });
    expect(callCount).toBe(1);
    await writeFile(join(agentDir, "auth.json"), JSON.stringify({ litellm: { type: "oauth", ...credential } }), "utf8");

    vi.mocked(Date.now).mockReturnValue(loginTime + 25 * 60 * 60 * 1000);
    const sessionStartHandlers = pi.handlers.get("session_start") ?? [];
    for (const handler of sessionStartHandlers) {
      await handler({ reason: "start" }, { sessionManager: { getSessionFile: () => undefined } });
    }

    await vi.waitFor(() => {
      expect(callCount).toBe(2);
      expect(pi.providers).toHaveLength(2);
    });
  });

  it("registers the helper as a per-request `!command` provider key that re-runs each request", async () => {
    const agentDir = await makeAgentDir();
    const first = makeJwt(Math.floor(Date.now() / 1000) + 3600);
    const second = makeJwt(Math.floor(Date.now() / 1000) + 7200);
    const third = makeJwt(Math.floor(Date.now() / 1000) + 10800);
    const helperPath = await writeHelper(agentDir, [first, second, third]);
    process.env.LITELLM_BASE_URL = "https://litellm.example.com/v1";
    process.env.LITELLM_API_KEY = "stale-token";
    process.env.LITELLM_API_KEY_HELPER = helperPath;
    const seenAuthHeaders: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, init) => {
      seenAuthHeaders.push(new Headers(init?.headers).get("authorization") ?? "");
      return jsonResponse(200, { data: [{ model_name: "claude-opus-4-8", model_info: { mode: "chat" } }] });
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    // Startup discovery uses a fresh helper token (one helper invocation).
    expect(seenAuthHeaders).toEqual([`Bearer ${first}`]);
    expect(await readHelperCount(agentDir)).toBe(1);

    // The provider key is the `!helper` command. Pi's per-request auth path
    // (ModelRegistry.getApiKeyAndHeaders) resolves provider keys via resolveConfigValueUncached,
    // re-executing the command on every request — it does NOT use the process-lifetime command
    // cache. Simulate that by resolving the registered command twice and asserting fresh tokens.
    const registeredKey = pi.providers[0]?.config.apiKey;
    expect(registeredKey).toBe(`!${helperPath}`);
    expect(pi.providers[0]?.config.baseUrl).toBe("https://litellm.example.com/v1");

    const command = (registeredKey as string).slice(1);
    const resolveUncached = () => execSync(command, { encoding: "utf8" }).trim();
    expect(resolveUncached()).toBe(second);
    expect(resolveUncached()).toBe(third);
    expect(await readHelperCount(agentDir)).toBe(3);
  });

  it.each([
    { name: "discovery is disabled", timeout: "0", fetches: false, helperRuns: 0 },
    { name: "fresh cache avoids discovery", fetches: false, helperRuns: 0 },
    { name: "helper output rotates before discovery fails", listModels: true, fetches: true, helperRuns: 1 },
  ])("uses cached helper-backed models when $name", async ({ timeout, listModels, fetches, helperRuns }) => {
    const agentDir = await makeAgentDir();
    const helperPath = await writeHelper(agentDir, ["rotated-token"]);
    const originalArgv = process.argv;
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY_HELPER = helperPath;
    if (timeout) process.env.LITELLM_DISCOVERY_TIMEOUT_MS = timeout;
    if (listModels) process.argv = [...process.argv, "--list-models"];
    await writeModelCache(agentDir, helperPath);
    const fetchMock = vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("temporary outage"));

    try {
      const extension = await loadExtension(agentDir);
      const pi = createPi();
      await extension(pi);
      expect(fetchMock).toHaveBeenCalledTimes(fetches ? 1 : 0);
      expect(await readHelperCount(agentDir)).toBe(helperRuns);
      expect(pi.providers[0]?.config.models).toEqual(cachedModels);
    } finally {
      process.argv = originalArgv;
    }
  });

  it("uses cached helper-backed models when forced discovery cannot execute the helper", async () => {
    const agentDir = await makeAgentDir();
    const helperPath = await writeFailingHelper(agentDir);
    const originalArgv = process.argv;
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY_HELPER = helperPath;
    process.argv = [...process.argv, "--list-models"];
    await writeModelCache(agentDir, helperPath);
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { data: [] }));

    try {
      const extension = await loadExtension(agentDir);
      const pi = createPi();
      await extension(pi);
      expect(fetchMock).not.toHaveBeenCalled();
      expect(await readHelperCount(agentDir)).toBe(1);
      expect(pi.providers[0]?.config.models).toEqual(cachedModels);
    } finally {
      process.argv = originalArgv;
    }
  });

  it("does not re-run command-backed helpers after refreshing login credentials", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const now = new Date("2026-05-29T21:00:00.000Z").getTime();
    const first = makeJwt(Math.floor(now / 1000) + 60);
    const second = makeJwt(Math.floor(now / 1000) + 3600);
    const helperPath = await writeHelper(agentDir, [first, second, "unexpected-third-token"]);
    vi.spyOn(Date, "now").mockReturnValue(now);
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, { data: [{ model_name: "claude-opus-4-8", model_info: { mode: "chat" } }] }),
    );

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const credential = await pi.providers[0]?.config.oauth?.login({
      onPrompt: async (options) => (options.placeholder ? "https://litellm.example.com" : `!${helperPath}`),
    });
    const refreshed = await pi.providers[0]?.config.oauth?.refreshToken(credential!);
    const apiKey = pi.providers[0]?.config.oauth?.getApiKey(refreshed!);

    expect(credential?.access).toBe(first);
    expect(credential?.refresh).toBe(`!${helperPath}`);
    expect(credential?.expires).toBeLessThan((Math.floor(now / 1000) + 60) * 1000);
    expect(refreshed?.access).toBe(second);
    expect(apiKey).toBe(second);
    expect(await readHelperCount(agentDir)).toBe(2);
  });

  it("marks opaque command-backed tokens expired without re-running after refresh", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const helperPath = await writeHelper(agentDir, ["opaque-first", "opaque-second", "unexpected-third"]);
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, { data: [{ model_name: "claude-opus-4-8", model_info: { mode: "chat" } }] }),
    );

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const credential = await pi.providers[0]?.config.oauth?.login({
      onPrompt: async (options) => (options.placeholder ? "https://litellm.example.com" : `!${helperPath}`),
    });
    const refreshed = await pi.providers[0]?.config.oauth?.refreshToken(credential!);
    const apiKey = pi.providers[0]?.config.oauth?.getApiKey(refreshed!);

    expect(credential).toMatchObject({ access: "opaque-first", refresh: `!${helperPath}`, expires: 0 });
    expect(refreshed).toMatchObject({ access: "opaque-second", refresh: `!${helperPath}`, expires: 0 });
    expect(apiKey).toBe("opaque-second");
    expect(await readHelperCount(agentDir)).toBe(2);
  });

  it("uses a helper when stored OAuth credentials contain an expired token", async () => {
    const agentDir = await makeAgentDir();
    const now = new Date("2026-05-29T21:00:00.000Z").getTime();
    const expired = makeJwt(Math.floor(now / 1000) - 60);
    const fresh = makeJwt(Math.floor(now / 1000) + 3600);
    const helperPath = await writeHelper(agentDir, [fresh]);
    await writeFile(
      join(agentDir, "auth.json"),
      JSON.stringify({
        litellm: {
          type: "oauth",
          access: expired,
          refresh: `!${helperPath}`,
          expires: Number.MAX_SAFE_INTEGER,
          baseUrl: "https://litellm.example.com",
        },
      }),
      "utf8",
    );
    vi.spyOn(Date, "now").mockReturnValue(now);
    const seenAuthHeaders: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, init) => {
      seenAuthHeaders.push(new Headers(init?.headers).get("authorization") ?? "");
      return jsonResponse(200, { data: [{ model_name: "claude-opus-4-8", model_info: { mode: "chat" } }] });
    });

    const extension = await loadExtension(agentDir);
    await extension(createPi());

    expect(seenAuthHeaders).toEqual([`Bearer ${fresh}`]);
  });
});
