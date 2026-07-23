import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type {
  AuthInteraction,
  Credential,
  Provider,
  ProviderModelsStore,
  RefreshModelsContext,
} from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createPi, loadExtension } from "./test-helpers.js";

const ENV_KEYS = [
  "LITELLM_BASE_URL",
  "LITELLM_API_KEY",
  "LITELLM_API_KEY_HELPER",
  "LITELLM_HEADERS",
  "LITELLM_OFFLINE",
  "LITELLM_VERBOSE_DISCOVERY",
  "LITELLM_ANTHROPIC_API_KEY",
  "LITELLM_ANTHROPIC_HEADERS",
  "LITELLM_DISCOVERY_TIMEOUT_MS",
  "LITELLM_MODELS_DEV",
  "LITELLM_GCLOUD_TOKEN_AUTH",
  "GOOGLE_APPLICATION_CREDENTIALS",
  "STORED_LITELLM_KEY",
  "CUSTOM_LITELLM_KEY",
];
const ORIGINAL_ENV = new Map(ENV_KEYS.map((key) => [key, process.env[key]]));

vi.unmock("@earendil-works/pi-coding-agent");

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

async function readHelperCount(agentDir: string): Promise<number> {
  try {
    return Number(await readFile(join(agentDir, "helper-count"), "utf8"));
  } catch {
    return 0;
  }
}

function createModelsStore(models: readonly any[] = []): ProviderModelsStore {
  let entry: Awaited<ReturnType<ProviderModelsStore["read"]>> =
    models.length > 0 ? { models, checkedAt: Date.now() } : undefined;
  return {
    read: async () => entry,
    write: async (next) => {
      entry = next;
    },
    delete: async () => {
      entry = undefined;
    },
  };
}

async function refreshProvider(
  provider: Provider,
  options: Omit<RefreshModelsContext, "store"> & { store?: ProviderModelsStore },
): Promise<readonly unknown[]> {
  await provider.refreshModels?.({ ...options, store: options.store ?? createModelsStore() });
  return provider.getModels();
}

function resolveApiKey(provider: Provider, credential?: Extract<Credential, { type: "api_key" }>) {
  return provider.auth.apiKey?.resolve({
    credential,
    ctx: {
      env: async (name) => process.env[name],
      fileExists: async () => false,
    },
  });
}

function resolveApiKeyWithEnv(provider: Provider, env: Record<string, string | undefined>) {
  return provider.auth.apiKey?.resolve({
    ctx: { env: async (name) => env[name], fileExists: async () => false },
  });
}

function interaction(
  prompt: AuthInteraction["prompt"],
  notify: AuthInteraction["notify"] = vi.fn(),
  signal?: AbortSignal,
): AuthInteraction {
  return { prompt, notify, signal };
}

async function loginOAuth(
  provider: Provider,
  callbacks: {
    onPrompt: (prompt: { message: string; placeholder?: string }) => Promise<string>;
    onAuth?: (event: { url: string; instructions?: string }) => void;
    onProgress?: (message: string) => void;
    signal?: AbortSignal;
  },
) {
  return provider.auth.oauth?.login(
    interaction(
      (prompt) =>
        callbacks.onPrompt({
          message: prompt.message,
          placeholder: "placeholder" in prompt ? prompt.placeholder : undefined,
        }),
      (event) => {
        if (event.type === "auth_url") callbacks.onAuth?.(event);
        if (event.type === "progress") callbacks.onProgress?.(event.message);
      },
      callbacks.signal,
    ),
  );
}

afterEach(() => {
  for (const key of ENV_KEYS) {
    const original = ORIGINAL_ENV.get(key);
    if (original === undefined) delete process.env[key];
    else process.env[key] = original;
  }
  vi.restoreAllMocks();
  vi.resetModules();
});

describe("extension startup", () => {
  it("registers one complete native provider and one session handler", async () => {
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();

    await extension(pi);

    expect(pi.providers.map((provider) => provider.id)).toEqual(["litellm"]);
    expect(pi.providers[0]).toEqual(
      expect.objectContaining({
        name: "LiteLLM",
        stream: expect.any(Function),
        streamSimple: expect.any(Function),
        refreshModels: expect.any(Function),
      }),
    );
    expect(pi.handlers.get("session_start")).toHaveLength(1);
    expect(pi.commands.has("litellm-refresh")).toBe(false);
  });

  it("disables models.dev enrichment with LITELLM_MODELS_DEV=0", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "sk-test";
    process.env.LITELLM_MODELS_DEV = "0";
    const urls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      urls.push(url);
      if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
      if (url.endsWith("/v1/models")) {
        return jsonResponse(200, { data: [{ id: "gpt-5.5", owned_by: "openai" }] });
      }
      if (url.endsWith("/mcp-rest/tools/list")) return jsonResponse(200, { tools: [] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await refreshProvider(pi.providers[0]!, {
      allowNetwork: true,
      force: true,
      credential: { type: "api_key", key: "sk-test", env: { LITELLM_BASE_URL: "https://litellm.example.com" } },
    });

    expect(urls).not.toContain("https://models.dev/api.json");
    expect(pi.providers[0]?.getModels()[0]?.id).toBe("gpt-5.5");
  });

  it("keeps one provider registration across Pi-managed refresh", async () => {
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "sk-test";
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] }),
    );
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);

    await refreshProvider(pi.providers[0]!, {
      allowNetwork: true,
      force: true,
      credential: { type: "api_key", key: "sk-test", env: { LITELLM_BASE_URL: "https://litellm.example.com" } },
    });

    expect(pi.providers.map((provider) => provider.id)).toEqual(["litellm"]);
  });

  it("restores Pi-managed models offline without discovery", async () => {
    process.env.LITELLM_OFFLINE = "1";
    const fetchMock = vi.spyOn(globalThis, "fetch");
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);
    const stored = {
      id: "stored-model",
      name: "Stored model",
      provider: "litellm",
      api: "openai-completions",
      baseUrl: "https://litellm.example.com/v1",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128_000,
      maxTokens: 4096,
    };

    await refreshProvider(pi.providers[0]!, {
      allowNetwork: false,
      credential: { type: "api_key", key: "sk-test" },
      store: createModelsStore([stored]),
    });

    expect(pi.providers[0]?.getModels()).toEqual([stored]);
    expect(fetchMock).not.toHaveBeenCalled();
    expect(pi.providers).toHaveLength(1);
  });

  it("ignores legacy cache files without deleting them", async () => {
    process.env.LITELLM_OFFLINE = "1";
    const agentDir = await makeAgentDir();
    const cachePath = join(agentDir, "litellm-models.json");
    const legacyCache = JSON.stringify({ models: [{ id: "legacy-model" }] });
    await writeFile(cachePath, legacyCache, "utf8");
    const extension = await loadExtension(agentDir);
    const pi = createPi();

    await extension(pi);
    await refreshProvider(pi.providers[0]!, {
      allowNetwork: false,
      credential: { type: "api_key", key: "sk-test" },
      store: createModelsStore(),
    });

    expect(await readFile(cachePath, "utf8")).toBe(legacyCache);
    expect(pi.providers[0]?.getModels()).toEqual([]);
  });

  it("registers MCP tools after an online Pi-managed model restore", async () => {
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "sk-test";
    const requestedUrls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      requestedUrls.push(url);
      if (url.endsWith("/mcp-rest/tools/list")) {
        return jsonResponse(200, {
          tools: [
            {
              name: "search",
              description: "Search",
              inputSchema: { type: "object", properties: {} },
              mcp_info: { server_name: "brave", server_id: "brave-api" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);
    const stored = {
      id: "stored-model",
      name: "Stored model",
      provider: "litellm",
      api: "openai-completions",
      baseUrl: "https://litellm.example.com/v1",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128_000,
      maxTokens: 4096,
    };

    await expect(
      refreshProvider(pi.providers[0]!, {
        allowNetwork: true,
        credential: {
          type: "api_key",
          key: "sk-test",
          env: { LITELLM_BASE_URL: "https://litellm.example.com" },
        },
        store: createModelsStore([stored]),
      }),
    ).rejects.toThrow("unexpected URL");

    expect(requestedUrls).toEqual([
      "https://litellm.example.com/model/info",
      "https://litellm.example.com/mcp-rest/tools/list",
    ]);
    expect(pi.tools.map((tool) => tool.name)).toContain("mcp_brave_search");
    expect(pi.providers[0]?.getModels()).toEqual([stored]);
  });

  it("retains Pi-managed models when discovery fails", async () => {
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("offline"));
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);
    const store = createModelsStore([
      {
        id: "stored-model",
        name: "Stored model",
        provider: "litellm",
        api: "openai-completions",
        baseUrl: "https://litellm.example.com/v1",
        reasoning: false,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 128_000,
        maxTokens: 4096,
      },
    ]);
    const credential = {
      type: "api_key" as const,
      key: "sk-test",
      env: { LITELLM_BASE_URL: "https://litellm.example.com" },
    };
    await refreshProvider(pi.providers[0]!, { allowNetwork: false, credential, store });

    await expect(
      refreshProvider(pi.providers[0]!, { allowNetwork: true, force: true, credential, store }),
    ).rejects.toThrow("offline");
    expect(pi.providers[0]?.getModels().map((model) => model.id)).toEqual(["stored-model"]);
    expect(pi.providers).toHaveLength(1);
  });

  it("registers the API key as an explicit environment reference", async () => {
    const agentDir = await makeAgentDir();
    const extension = await loadExtension(agentDir);
    const pi = createPi();

    await extension(pi);

    expect(pi.providers[0]?.auth.apiKey).toMatchObject({ name: "LiteLLM API key", login: expect.any(Function) });
  });

  it('treats literal "undefined" env values as unset', async () => {
    process.env.LITELLM_BASE_URL = "undefined";
    process.env.LITELLM_API_KEY = "undefined";
    const agentDir = await makeAgentDir();
    const extension = await loadExtension(agentDir);
    const pi = createPi();

    await extension(pi);

    expect(pi.providers[0]?.baseUrl).toBe("https://litellm.example.com/v1");
  });

  it("applies LiteLLM request compatibility hooks to configured provider aliases", async () => {
    const agentDir = await makeAgentDir();
    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({
        litellm: {
          providers: {
            "litellm-anthropic": {
              baseUrl: "https://litellm-anthropic.example.com",
              apiKey: "$LITELLM_ANTHROPIC_API_KEY",
            },
          },
        },
      }),
      "utf8",
    );
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "openai-key";
    process.env.LITELLM_ANTHROPIC_API_KEY = "anthropic-key";
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const result = await pi.handlers.get("before_provider_request")?.[0]?.(
      { payload: { model: "kimi-k2.6" } },
      { model: { provider: "litellm-anthropic", id: "kimi-k2.6" } },
    );

    expect(result).toMatchObject({
      include_reasoning: false,
      reasoning_content: false,
      merge_reasoning_content_in_choices: true,
      thinking: { type: "disabled" },
    });
  });

  it("returns a native API-key credential without discovery side effects", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    process.env.LITELLM_VERBOSE_DISCOVERY = "1";
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

    const prompt = vi.fn().mockResolvedValueOnce(" http://127.0.0.1:4000/v1 ").mockResolvedValueOnce(" sk-login ");
    const credential = await pi.providers[0]?.auth.apiKey?.login?.(interaction(prompt));

    expect(prompt).toHaveBeenNthCalledWith(1, expect.objectContaining({ type: "text" }));
    expect(prompt).toHaveBeenNthCalledWith(2, expect.objectContaining({ type: "secret" }));
    expect(seenRequests).toEqual([]);
    expect(credential).toEqual({
      type: "api_key",
      key: "sk-login",
      env: { LITELLM_BASE_URL: "http://127.0.0.1:4000" },
    });
    delete process.env.LITELLM_VERBOSE_DISCOVERY;
  });

  it("checks command-backed auth without executing the helper", async () => {
    const agentDir = await makeAgentDir();
    const helperPath = await writeHelper(agentDir, ["helper-key"]);
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY_HELPER = helperPath;
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const result = await pi.providers[0]?.auth.apiKey?.check?.({
      ctx: { env: async (name) => process.env[name], fileExists: async () => false },
    });

    expect(result).toEqual({ type: "api_key", source: "LITELLM_API_KEY_HELPER" });
    expect(await readHelperCount(agentDir)).toBe(0);
  });

  it("resolves native auth from the injected context instead of process env", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_BASE_URL = "https://process.example.com";
    process.env.LITELLM_API_KEY = "process-key";
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await expect(
      resolveApiKeyWithEnv(pi.providers[0]!, {
        LITELLM_BASE_URL: "https://context.example.com",
        LITELLM_API_KEY: "context-key",
        LITELLM_HEADERS: '{"x-tenant":"context"}',
      }),
    ).resolves.toMatchObject({
      auth: {
        apiKey: "context-key",
        baseUrl: "https://context.example.com/v1",
        headers: { "x-tenant": "context" },
      },
      source: "LITELLM_API_KEY",
    });
  });

  it("executes only the helper supplied by the injected auth context", async () => {
    const agentDir = await makeAgentDir();
    const contextHelper = await writeHelper(agentDir, ["context-helper-key"]);
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await expect(
      resolveApiKeyWithEnv(pi.providers[0]!, {
        LITELLM_BASE_URL: "https://context.example.com",
        LITELLM_API_KEY_HELPER: contextHelper,
        LITELLM_API_KEY: "context-env-key",
      }),
    ).resolves.toMatchObject({ auth: { apiKey: "context-helper-key" }, source: "LITELLM_API_KEY_HELPER" });
    expect(await readHelperCount(agentDir)).toBe(1);
  });

  it("resolves configured key templates from the injected auth context", async () => {
    const agentDir = await makeAgentDir();
    const lowerPriorityHelper = await writeHelper(agentDir, ["unexpected-helper-key"]);
    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({ litellm: { providers: { litellm: { apiKey: "$CUSTOM_LITELLM_KEY" } } } }),
      "utf8",
    );
    process.env.CUSTOM_LITELLM_KEY = "process-configured-key";
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await expect(
      resolveApiKeyWithEnv(pi.providers[0]!, {
        LITELLM_BASE_URL: "https://context.example.com",
        CUSTOM_LITELLM_KEY: "context-configured-key",
        LITELLM_API_KEY_HELPER: lowerPriorityHelper,
        LITELLM_API_KEY: "context-default-key",
      }),
    ).resolves.toMatchObject({
      auth: { apiKey: "context-configured-key", baseUrl: "https://context.example.com/v1" },
      source: "$CUSTOM_LITELLM_KEY",
    });
    expect(await readHelperCount(agentDir)).toBe(0);
  });

  it("leaves model refresh to Pi after login", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
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

    expect(pi.providers).toHaveLength(1);
    expect(pi.providers[0]?.getModels()).toEqual([]);

    await loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => (options.placeholder ? " http://127.0.0.1:4000/v1 " : " sk-login "),
      signal: new AbortController().signal,
    });

    const registeredModels = pi.providers[1]?.getModels() as unknown as Array<{ id: string }> | undefined;
    expect(pi.providers).toHaveLength(1);
    expect(registeredModels).toBeUndefined();
    expect(vi.mocked(globalThis.fetch).mock.calls.every(([url]) => !String(url).endsWith("/model/info"))).toBe(true);
  });

  it("leaves /login litellm to Pi's registered OAuth provider", async () => {
    const agentDir = await makeAgentDir();
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(pi.commands.has("login")).toBe(false);
    expect(pi.providers[0]?.auth.oauth).toBeDefined();
    expect(pi.handlers.has("input")).toBe(false);
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

    const credential = await loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => (options.placeholder ? " http://127.0.0.1:4000/v1 " : " sk-login "),
      signal: new AbortController().signal,
    });
    expect(callCount).toBe(0);
    await writeFile(join(agentDir, "auth.json"), JSON.stringify({ litellm: { type: "oauth", ...credential } }), "utf8");

    vi.mocked(Date.now).mockReturnValue(loginTime + 25 * 60 * 60 * 1000);
    const sessionStartHandlers = pi.handlers.get("session_start") ?? [];
    for (const handler of sessionStartHandlers) {
      await handler({ reason: "start" }, { sessionManager: { getSessionFile: () => undefined } });
    }

    expect(callCount).toBe(0);
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
    const credential = await pi.providers[0]?.auth.apiKey?.login?.(
      interaction(vi.fn().mockResolvedValueOnce("https://litellm.example.com").mockResolvedValueOnce(`!${helperPath}`)),
    );
    const firstAuth = await resolveApiKey(pi.providers[0]!, credential);
    const secondAuth = await resolveApiKey(pi.providers[0]!, credential);

    expect(firstAuth?.auth.apiKey).toBe(first);
    expect(secondAuth?.auth.apiKey).toBe(second);
    expect(await readHelperCount(agentDir)).toBe(2);
  });

  it("resolves opaque command-backed API keys for each request", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const helperPath = await writeHelper(agentDir, ["opaque-first", "opaque-second", "unexpected-third"]);
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, { data: [{ model_name: "claude-opus-4-8", model_info: { mode: "chat" } }] }),
    );

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const credential = await pi.providers[0]?.auth.apiKey?.login?.(
      interaction(vi.fn().mockResolvedValueOnce("https://litellm.example.com").mockResolvedValueOnce(`!${helperPath}`)),
    );

    expect((await resolveApiKey(pi.providers[0]!, credential))?.auth.apiKey).toBe("opaque-first");
    expect((await resolveApiKey(pi.providers[0]!, credential))?.auth.apiKey).toBe("opaque-second");
    expect(await readHelperCount(agentDir)).toBe(2);
  });

  it("executes an OAuth refresh command only during refresh", async () => {
    const agentDir = await makeAgentDir();
    const helperPath = await writeHelper(agentDir, ["refreshed-token", "unexpected-second-run"]);
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const credential = {
      type: "oauth" as const,
      access: "expired-token",
      refresh: `!${helperPath}`,
      expires: 0,
      baseUrl: "https://litellm.example.com",
    };

    const refreshed = await pi.providers[0]?.auth.oauth?.refresh(credential);
    expect(await readHelperCount(agentDir)).toBe(1);
    await expect(pi.providers[0]?.auth.oauth?.toAuth(refreshed!)).resolves.toMatchObject({
      apiKey: "refreshed-token",
    });
    expect(await readHelperCount(agentDir)).toBe(1);
  });

  it("uses the refreshed OAuth access token during discovery", async () => {
    const agentDir = await makeAgentDir();
    const helperPath = await writeHelper(agentDir, ["unexpected-helper-run"]);
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    process.env.LITELLM_HEADERS = '{"x-tenant":"tenant-a"}';
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      if (String(input).endsWith("/model/info"))
        return jsonResponse(200, { data: [{ model_name: "claude-opus-4-8", model_info: { mode: "chat" } }] });
      return jsonResponse(200, { tools: [] });
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "5000";
    await refreshProvider(pi.providers[0]!, {
      allowNetwork: true,
      credential: {
        type: "oauth",
        access: "already-refreshed",
        refresh: `!${helperPath}`,
        expires: Date.now() + 60_000,
        baseUrl: "https://current.example.com",
      },
    });

    expect(fetchMock.mock.calls[0]?.[0]).toBe("https://current.example.com/model/info");
    expect(new Headers(fetchMock.mock.calls[0]?.[1]?.headers).get("authorization")).toBe("Bearer already-refreshed");
    expect(new Headers(fetchMock.mock.calls[0]?.[1]?.headers).get("x-tenant")).toBe("tenant-a");
    expect(await readHelperCount(agentDir)).toBe(0);
  });

  it("enterprise SSO login generates a virtual key and uses it as the access token", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const jwt = makeJwt(Math.floor(Date.now() / 1000) + 3600);
    const seenRequests: Array<{ url: string; method: string; authorization: string }> = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      seenRequests.push({
        url,
        method: String(init?.method ?? "GET"),
        authorization: new Headers(init?.headers).get("authorization") ?? "",
      });
      if (url.endsWith("/key/generate")) return jsonResponse(200, { key: "sk-virtual-abc" });
      if (url.endsWith("/model/info"))
        return jsonResponse(200, { data: [{ model_name: "gpt-4o", model_info: { mode: "chat" } }] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const authInfos: Array<{ url: string; instructions?: string }> = [];
    const credential = await loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => {
        if (options.placeholder) return "https://litellm.example.com";
        if (options.message.includes("Select login method")) return "2";
        if (options.message.includes("SSO token")) return `Bearer ${jwt}`;
        return "y";
      },
      onAuth: (info) => authInfos.push(info),
      signal: new AbortController().signal,
    });

    expect(authInfos).toEqual([
      {
        type: "auth_url",
        url: "https://litellm.example.com/sso/key/generate",
        instructions: "Authenticate via SSO, then copy your token from the LiteLLM UI.",
      },
    ]);
    expect(credential).toMatchObject({
      access: "sk-virtual-abc",
      refresh: "",
      expires: Number.MAX_SAFE_INTEGER,
      baseUrl: "https://litellm.example.com",
    });
    await expect(pi.providers[0]?.auth.oauth?.toAuth(credential!)).resolves.toMatchObject({
      apiKey: "sk-virtual-abc",
      baseUrl: "https://litellm.example.com/v1",
    });
    expect(seenRequests).toContainEqual(
      expect.objectContaining({
        url: "https://litellm.example.com/key/generate",
        method: "POST",
        authorization: `Bearer ${jwt}`,
      }),
    );
    expect(seenRequests.every(({ url }) => !url.endsWith("/model/info"))).toBe(true);
  });

  it("enterprise SSO login strips Bearer prefix from pasted SSO token", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const jwt = makeJwt(Math.floor(Date.now() / 1000) + 3600);
    const seenAuthorizations: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      seenAuthorizations.push(new Headers(init?.headers).get("authorization") ?? "");
      if (url.endsWith("/key/generate")) return jsonResponse(200, { key: "sk-stripped" });
      if (url.endsWith("/model/info"))
        return jsonResponse(200, { data: [{ model_name: "gpt-4o", model_info: { mode: "chat" } }] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => {
        if (options.placeholder) return "https://litellm.example.com";
        if (options.message.includes("Select login method")) return "2";
        if (options.message.includes("SSO token")) return `  Bearer  ${jwt}  `;
        return "y";
      },
      signal: new AbortController().signal,
    });

    expect(seenAuthorizations[0]).toBe(`Bearer ${jwt}`);
  });

  it("enterprise SSO login honors the expiry returned with a generated virtual key", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const jwt = makeJwt(Math.floor(Date.now() / 1000) + 3600);
    const keyExpiresAt = new Date(Date.now() + 60 * 60 * 1000);
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/key/generate"))
        return jsonResponse(200, { key: "sk-expiring", expires: keyExpiresAt.toISOString() });
      if (url.endsWith("/model/info"))
        return jsonResponse(200, { data: [{ model_name: "gpt-4o", model_info: { mode: "chat" } }] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const credential = await loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => {
        if (options.placeholder) return "https://litellm.example.com";
        if (options.message.includes("Select login method")) return "2";
        if (options.message.includes("SSO token")) return jwt;
        return "y";
      },
      signal: new AbortController().signal,
    });

    expect(credential).toMatchObject({ access: "sk-expiring", refresh: "" });
    expect(credential?.expires).toBe(keyExpiresAt.getTime() - 5 * 60 * 1000);
  });

  it("enterprise SSO login falls back to JWT when virtual key generation times out", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const jwt = makeJwt(Math.floor(Date.now() / 1000) + 3600);
    const progress = vi.fn();
    const timeoutController = new AbortController();
    const nativeTimeout = AbortSignal.timeout.bind(AbortSignal);
    vi.spyOn(AbortSignal, "timeout")
      .mockImplementationOnce(() => timeoutController.signal)
      .mockImplementation(nativeTimeout);
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
      const url = String(input);
      if (url.endsWith("/key/generate"))
        return new Promise<Response>((_, reject) => {
          if (init?.signal?.aborted) {
            reject(init.signal.reason);
            return;
          }
          init?.signal?.addEventListener("abort", () => reject(init.signal?.reason ?? new Error("aborted")), {
            once: true,
          });
        });
      if (url.endsWith("/model/info"))
        return Promise.resolve(jsonResponse(200, { data: [{ model_name: "gpt-4o", model_info: { mode: "chat" } }] }));
      return Promise.reject(new Error(`unexpected URL: ${url}`));
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const controller = new AbortController();
    const loginPromise = loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => {
        if (options.placeholder) return "https://litellm.example.com";
        if (options.message.includes("Select login method")) return "2";
        if (options.message.includes("SSO token")) return jwt;
        return "y";
      },
      onProgress: progress,
      signal: controller.signal,
    });
    await vi.waitFor(() =>
      expect(fetchSpy.mock.calls.some(([input]) => String(input).endsWith("/key/generate"))).toBe(true),
    );
    timeoutController.abort(new Error("test timeout"));

    const credential = await loginPromise;
    expect(credential).toMatchObject({ access: jwt, refresh: "" });
    expect(progress).toHaveBeenCalledWith(expect.stringContaining("virtual key generation failed"));
  });

  it("enterprise SSO login rejects when the caller cancels virtual key generation", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const jwt = makeJwt(Math.floor(Date.now() / 1000) + 3600);
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
      const url = String(input);
      return new Promise<Response>((resolve, reject) => {
        if (init?.signal?.aborted) {
          reject(init.signal.reason);
          return;
        }
        if (url.endsWith("/key/generate")) {
          init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
          return;
        }
        if (url.endsWith("/model/info")) {
          resolve(jsonResponse(200, { data: [{ model_name: "gpt-4o", model_info: { mode: "chat" } }] }));
          return;
        }
        reject(new Error(`unexpected URL: ${url}`));
      });
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const controller = new AbortController();
    const reason = new Error("caller cancelled login");
    const loginPromise = loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => {
        if (options.placeholder) return "https://litellm.example.com";
        if (options.message.includes("Select login method")) return "2";
        if (options.message.includes("SSO token")) return jwt;
        return "y";
      },
      signal: controller.signal,
    });
    await vi.waitFor(() =>
      expect(fetchSpy.mock.calls.some(([input]) => String(input).endsWith("/key/generate"))).toBe(true),
    );
    controller.abort(reason);

    await expect(loginPromise).rejects.toBe(reason);
  });

  it("enterprise SSO login uses JWT directly when user answers no to virtual key generation", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const jwt = makeJwt(Math.floor(Date.now() / 1000) + 3600);
    const seenRequests: Array<{ url: string; method: string }> = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      seenRequests.push({ url, method: String(init?.method ?? "GET") });
      if (url.endsWith("/model/info"))
        return jsonResponse(200, { data: [{ model_name: "gpt-4o", model_info: { mode: "chat" } }] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const credential = await loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => {
        if (options.placeholder) return "https://litellm.example.com";
        if (options.message.includes("Select login method")) return "2";
        if (options.message.includes("SSO token")) return jwt;
        return "no";
      },
      signal: new AbortController().signal,
    });

    expect(credential).toMatchObject({ access: jwt, refresh: "" });
    expect(credential?.expires).toBeLessThan(Number.MAX_SAFE_INTEGER);
    expect(seenRequests.every(({ url }) => !url.includes("key/generate"))).toBe(true);
  });

  it("enterprise SSO refresh rejects expiring generated virtual keys without a refresh path", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const jwt = makeJwt(Math.floor(Date.now() / 1000) + 3600);
    const keyExpiresAt = new Date(Date.now() + 60 * 60 * 1000);
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/key/generate"))
        return jsonResponse(200, { key: "sk-expiring", expires: keyExpiresAt.toISOString() });
      if (url.endsWith("/model/info"))
        return jsonResponse(200, { data: [{ model_name: "gpt-4o", model_info: { mode: "chat" } }] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const credential = await loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => {
        if (options.placeholder) return "https://litellm.example.com";
        if (options.message.includes("Select login method")) return "2";
        if (options.message.includes("SSO token")) return jwt;
        return "y";
      },
      signal: new AbortController().signal,
    });

    await expect(pi.providers[0]?.auth.oauth?.refresh(credential!)).rejects.toThrow(
      "LiteLLM credential cannot be refreshed; run /login litellm again",
    );
  });

  it("enterprise SSO login falls back to JWT when virtual key generation fails", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const jwt = makeJwt(Math.floor(Date.now() / 1000) + 3600);
    const progress = vi.fn();
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/key/generate")) return jsonResponse(403, { error: "forbidden" });
      if (url.endsWith("/model/info"))
        return jsonResponse(200, { data: [{ model_name: "gpt-4o", model_info: { mode: "chat" } }] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const credential = await loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => {
        if (options.placeholder) return "https://litellm.example.com";
        if (options.message.includes("Select login method")) return "2";
        if (options.message.includes("SSO token")) return jwt;
        return "y";
      },
      onProgress: progress,
      signal: new AbortController().signal,
    });

    expect(credential).toMatchObject({ access: jwt, refresh: "" });
    expect(progress).toHaveBeenCalledWith(expect.stringContaining("virtual key generation failed"));
  });

  it("enterprise SSO login throws when SSO token is empty", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { data: [] }));
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await expect(
      loginOAuth(pi.providers[0]!, {
        onPrompt: async (options) => {
          if (options.placeholder) return "https://litellm.example.com";
          if (options.message.includes("Select login method")) return "2";
          return "";
        },
        signal: new AbortController().signal,
      }),
    ).rejects.toThrow("SSO token is required");
  });
});

describe("multi-provider hardening", () => {
  it("does not register the default env key for an alias missing its apiKey", async () => {
    const agentDir = await makeAgentDir();
    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({
        litellm: {
          providers: {
            "litellm-anthropic": { baseUrl: "https://litellm-anthropic.example.com" },
          },
        },
      }),
      "utf8",
    );
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "openai-key";
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(await resolveApiKey(pi.providers[0]!)).toMatchObject({ auth: { apiKey: "openai-key" } });
    expect(pi.providers[1]?.id).toBe("litellm-anthropic");
    expect(await resolveApiKey(pi.providers[1]!)).toBeUndefined();
  });

  it("sends custom headers when generating a login virtual key", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({
        litellm: { providers: { litellm: { headers: { "x-litellm-customer-id": "team-a" } } } },
      }),
      "utf8",
    );
    const jwt = makeJwt(Math.floor(Date.now() / 1000) + 3600);
    const seenRequests: Array<{ url: string; customer: string | null }> = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      seenRequests.push({ url, customer: new Headers(init?.headers).get("x-litellm-customer-id") });
      if (url.endsWith("/key/generate")) return jsonResponse(200, { key: "sk-virtual-abc" });
      if (url.endsWith("/model/info"))
        return jsonResponse(200, { data: [{ model_name: "gpt-5", model_info: { mode: "chat" } }] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => {
        if (options.placeholder) return "https://litellm.example.com";
        if (options.message.includes("Select login method")) return "2";
        if (options.message.includes("SSO token")) return jwt;
        return "y";
      },
      signal: new AbortController().signal,
    });

    expect(seenRequests).toContainEqual({ url: "https://litellm.example.com/key/generate", customer: "team-a" });
  });

  it("drops non-primitive header values instead of stringifying them", async () => {
    const agentDir = await makeAgentDir();
    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({
        litellm: {
          providers: {
            "litellm-anthropic": {
              baseUrl: "https://litellm-anthropic.example.com",
              apiKey: "$LITELLM_ANTHROPIC_API_KEY",
              headers: { "x-obj": { team: "a" }, "x-null": null, "x-num": 30, "x-bool": false },
            },
          },
        },
      }),
      "utf8",
    );
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(pi.providers[1]?.headers).toEqual({ "x-num": "30", "x-bool": "false" });
    expect(stderrSpy).toHaveBeenCalledWith(expect.stringContaining("x-obj"));
    expect(stderrSpy).toHaveBeenCalledWith(expect.stringContaining("x-null"));
  });
});
