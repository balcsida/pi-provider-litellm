import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
  type AuthContext,
  type AuthInteraction,
  type Credential,
  createModels,
  InMemoryCredentialStore,
  InMemoryModelsStore,
  type ModelsStore,
  type ModelsStoreEntry,
  type Provider,
  type RefreshModelsContext,
} from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createPi, loadExtension, useHermeticEnv } from "./test-helpers.js";

const SUITE_ENV_VARS = [
  "LITELLM_ANTHROPIC_API_KEY",
  "LITELLM_ANTHROPIC_HEADERS",
  "LITELLM_DISCOVERY_TIMEOUT_MS",
  "LITELLM_CLI_JWT_EXPIRATION_HOURS",
  "LITELLM_GCLOUD_TOKEN_AUTH",
  "GOOGLE_APPLICATION_CREDENTIALS",
  "STORED_LITELLM_KEY",
  "CUSTOM_LITELLM_KEY",
] as const;

// Suite-specific names beyond the shared managed set.
useHermeticEnv(SUITE_ENV_VARS);

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

function createModelsStore(models: readonly any[] = []): ModelsStore {
  let entry: ModelsStoreEntry | undefined = models.length > 0 ? { models, checkedAt: Date.now() } : undefined;
  return {
    read: async () => entry,
    write: async (_providerId, next) => {
      entry = next;
    },
    delete: async () => {
      entry = undefined;
    },
  };
}

async function refreshProvider(
  provider: Provider,
  options: Omit<RefreshModelsContext, "publish" | "signal" | "stored"> & { store?: ModelsStore },
): Promise<readonly unknown[]> {
  const store = options.store ?? createModelsStore();
  await provider.refreshModels?.({
    ...options,
    stored: await store.read(provider.id),
    publish: async ({ persist, update }) => {
      if (persist === null) await store.delete(provider.id);
      else if (persist) await store.write(provider.id, persist);
      update?.();
      return true;
    },
    signal: new AbortController().signal,
  });
  return provider.getModels();
}

const TEST_SIGNAL = new AbortController().signal;

function resolveApiKey(provider: Provider, credential?: Extract<Credential, { type: "api_key" }>) {
  return provider.auth.apiKey?.resolve({
    credential,
    ctx: {
      env: async (name) => process.env[name],
      fileExists: async () => false,
    },
    signal: TEST_SIGNAL,
  });
}

function resolveApiKeyWithEnv(provider: Provider, env: Record<string, string | undefined>) {
  return provider.auth.apiKey?.resolve({
    ctx: { env: async (name) => env[name], fileExists: async () => false },
    signal: TEST_SIGNAL,
  });
}

function interaction(
  prompt: AuthInteraction["prompt"],
  notify: AuthInteraction["notify"] = vi.fn(),
  signal: AbortSignal = TEST_SIGNAL,
): AuthInteraction & { signal: AbortSignal } {
  return { prompt, notify, signal };
}

async function loginOAuth(
  provider: Provider,
  callbacks: {
    onPrompt: (prompt: {
      type: string;
      message: string;
      placeholder?: string;
      options?: readonly { id: string; label: string }[];
    }) => Promise<string>;
    onAuth?: (event: { url: string; instructions?: string }) => void;
    onDeviceCode?: (event: { userCode: string; verificationUri: string; expiresInSeconds?: number }) => void;
    onProgress?: (message: string) => void;
    signal?: AbortSignal;
  },
) {
  const fetchImpl = globalThis.fetch;
  if (!callbacks.onDeviceCode)
    globalThis.fetch = (input, init) =>
      String(input).endsWith("/sso/cli/start")
        ? Promise.resolve(jsonResponse(404, { detail: "Not Found" }))
        : fetchImpl(input, init);
  try {
    return await provider.auth.oauth?.login(
      interaction(
        (prompt) =>
          callbacks.onPrompt({
            type: prompt.type,
            message: prompt.message,
            placeholder: "placeholder" in prompt ? prompt.placeholder : undefined,
            options: "options" in prompt ? prompt.options : undefined,
          }),
        (event) => {
          if (event.type === "auth_url") callbacks.onAuth?.(event);
          if (event.type === "device_code") callbacks.onDeviceCode?.(event);
          if (event.type === "progress") callbacks.onProgress?.(event.message);
        },
        callbacks.signal ?? TEST_SIGNAL,
      ),
    );
  } finally {
    globalThis.fetch = fetchImpl;
  }
}

afterEach(() => {
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

  it("keeps one provider registration across Pi-managed refresh", async () => {
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";
    // A fresh Response per call: activation discovers too, and a body can only be read once.
    vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] }),
    );
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);

    await refreshProvider(pi.providers[0]!, {
      allowNetwork: true,
      force: true,
      credential: { type: "api_key", key: "sk-test", env: { LITELLM_BASE_URL: "https://proxy.example.com" } },
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
      baseUrl: "https://proxy.example.com/v1",
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
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
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
    // Activation attempts its own discovery; assert only on what the refresh requests.
    requestedUrls.length = 0;
    const stored = {
      id: "stored-model",
      name: "Stored model",
      provider: "litellm",
      api: "openai-completions",
      baseUrl: "https://proxy.example.com/v1",
      reasoning: false,
      input: ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 128_000,
      maxTokens: 4096,
      litellmDiscoveryVersion: 2,
    };

    await expect(
      refreshProvider(pi.providers[0]!, {
        allowNetwork: true,
        credential: {
          type: "api_key",
          key: "sk-test",
          env: { LITELLM_BASE_URL: "https://proxy.example.com" },
        },
        store: createModelsStore([stored]),
      }),
    ).rejects.toThrow("unexpected URL");

    expect(requestedUrls).toEqual([
      "https://proxy.example.com/model/info",
      "https://proxy.example.com/mcp-rest/tools/list",
    ]);
    // MCP registration runs in the background so a hanging /mcp-rest endpoint
    // cannot block model refresh; wait for it to finish before asserting.
    await vi.waitFor(() => {
      expect(pi.tools.map((tool) => tool.name)).toContain("mcp_brave_search");
    });
    expect(pi.providers[0]?.getModels()).toEqual([stored]);
  });

  it("does not block model refresh on MCP discovery", async () => {
    let mcpStarted!: () => void;
    let releaseMcp!: (response: Response) => void;
    const started = new Promise<void>((resolve) => {
      mcpStarted = resolve;
    });
    const pendingMcp = new Promise<Response>((resolve) => {
      releaseMcp = resolve;
    });
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/mcp-rest/tools/list")) {
        mcpStarted();
        return pendingMcp;
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);

    let refreshed = false;
    const refresh = refreshProvider(pi.providers[0]!, {
      allowNetwork: true,
      credential: {
        type: "api_key",
        key: "sk-test",
        env: { LITELLM_BASE_URL: "https://proxy.example.com" },
      },
    }).then(() => {
      refreshed = true;
    });
    await started;
    await new Promise<void>((resolve) => setImmediate(resolve));

    expect(refreshed).toBe(true);
    releaseMcp(
      jsonResponse(200, {
        tools: [
          {
            name: "search",
            description: "Search",
            inputSchema: { type: "object", properties: {} },
            mcp_info: { server_name: "brave" },
          },
        ],
      }),
    );
    await refresh;
    await vi.waitFor(() => expect(pi.tools.map((tool) => tool.name)).toContain("mcp_brave_search"));
  });

  it("retains Pi-managed models when discovery fails", async () => {
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
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
        baseUrl: "https://proxy.example.com/v1",
        reasoning: false,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 128_000,
        maxTokens: 4096,
        litellmDiscoveryVersion: 2,
      },
    ]);
    const credential = {
      type: "api_key" as const,
      key: "sk-test",
      env: { LITELLM_BASE_URL: "https://proxy.example.com" },
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

  it("uses explicitly allowed insecure HTTP for a provider", async () => {
    const agentDir = await makeAgentDir();
    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({
        litellm: {
          providers: {
            litellm: {
              baseUrl: "http://host.docker.internal",
              apiKey: "sk-local",
              allowInsecureHttp: true,
            },
          },
        },
      }),
      "utf8",
    );
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const extension = await loadExtension(agentDir);
    const pi = createPi();

    await extension(pi);

    await expect(resolveApiKey(pi.providers[0]!)).resolves.toMatchObject({
      auth: { apiKey: "sk-local" },
      env: { LITELLM_BASE_URL: "http://host.docker.internal" },
    });
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
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "openai-key";
    process.env.LITELLM_ANTHROPIC_API_KEY = "anthropic-key";
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const result = await pi.handlers.get("before_provider_request")?.[0]?.(
      {
        payload: {
          model: "kimi-k2.6",
          messages: [{ role: "tool", tool_call_id: "call_1", content: [{ type: "text", text: "tool output" }] }],
        },
      },
      { model: { provider: "litellm-anthropic", id: "kimi-k2.6" } },
    );

    expect(result).toMatchObject({
      messages: [{ role: "tool", tool_call_id: "call_1", content: "tool output" }],
    });

    const moonshotRoute = await pi.handlers.get("before_provider_request")?.[0]?.(
      { payload: { messages: [] } },
      {
        model: {
          provider: "litellm-anthropic",
          id: "k3-prod",
          api: "openai-completions",
          suppressReasoningContent: true,
        },
      },
    );
    const otherRoute = await pi.handlers.get("before_provider_request")?.[0]?.(
      { payload: { messages: [] } },
      {
        model: { provider: "litellm", id: "k3-prod", api: "openai-completions" },
      },
    );

    expect(moonshotRoute).toEqual({
      messages: [],
      include_reasoning: false,
      reasoning_content: false,
      merge_reasoning_content_in_choices: true,
    });
    expect(otherRoute).toBeUndefined();
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
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY_HELPER = helperPath;
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const result = await pi.providers[0]?.auth.apiKey?.check?.({
      ctx: { env: async (name) => process.env[name], fileExists: async () => false },
      signal: TEST_SIGNAL,
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

  it("rejects shell expressions in helper commands", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await expect(
      resolveApiKeyWithEnv(pi.providers[0]!, {
        LITELLM_BASE_URL: "https://context.example.com",
        LITELLM_API_KEY_HELPER: "printf safe-key; printf injected-key",
      }),
    ).rejects.toThrow("shell syntax is not supported");
  });

  it("preserves backslashes in helper executable paths", async () => {
    const agentDir = await makeAgentDir();
    const helperPath = await writeHelper(agentDir, ["backslash-key"], join(agentDir, "token\\helper.sh"));
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await expect(
      resolveApiKeyWithEnv(pi.providers[0]!, {
        LITELLM_BASE_URL: "https://context.example.com",
        LITELLM_API_KEY_HELPER: `"${helperPath}"`,
      }),
    ).resolves.toMatchObject({ auth: { apiKey: "backslash-key" } });
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
      auth: { apiKey: "context-configured-key" },
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

  it("rejects OAuth placeholder base URLs on non-default ports", async () => {
    delete process.env.LITELLM_BASE_URL;
    delete process.env.LITELLM_API_KEY;
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);

    await expect(
      pi.providers[0]?.auth.oauth?.toAuth({
        type: "oauth",
        access: "sk-oauth",
        refresh: "",
        expires: Number.MAX_SAFE_INTEGER,
        baseUrl: "https://litellm.example.com:8443",
      }),
    ).rejects.toThrow(/placeholder LiteLLM base URL/);
  });

  it("uses the OAuth credential base URL when ambient configuration is unset", async () => {
    delete process.env.LITELLM_BASE_URL;
    delete process.env.LITELLM_API_KEY;
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const requestedUrls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      requestedUrls.push(url);
      if (!url.endsWith("/chat/completions")) throw new Error(`unexpected URL: ${url}`);
      return new Response(
        'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\ndata: [DONE]\n\n',
        { headers: { "content-type": "text/event-stream" } },
      );
    });
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);
    const provider = pi.providers[0]!;
    const credentials = new InMemoryCredentialStore();
    await credentials.modify(provider.id, async () => ({
      type: "oauth",
      access: "sk-oauth",
      refresh: "",
      expires: Number.MAX_SAFE_INTEGER,
      baseUrl: "https://credential.example.com",
    }));
    const authContext: AuthContext = {
      env: async () => undefined,
      fileExists: async () => false,
    };
    const models = createModels({ credentials, modelsStore: new InMemoryModelsStore(), authContext });
    models.setProvider(provider);
    const model = {
      id: "oauth-model",
      name: "OAuth model",
      provider: "litellm",
      api: "openai-completions" as const,
      baseUrl: "https://credential.example.com/v1",
      reasoning: false,
      input: ["text"] as ("text" | "image")[],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 4096,
      maxTokens: 1024,
    };

    const result = await models.complete(model, { messages: [] });

    expect(result.stopReason).toBe("stop");
    expect(requestedUrls).toEqual(["https://credential.example.com/v1/chat/completions"]);
  });

  it("clears the remembered OAuth base URL when API-key auth resolves", async () => {
    process.env.LITELLM_BASE_URL = "https://api-key.example.com";
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);
    const provider = pi.providers[0]!;
    const oauthCredential = {
      type: "oauth" as const,
      access: "shared-key",
      refresh: "",
      expires: Number.MAX_SAFE_INTEGER,
      baseUrl: "https://oauth.example.com",
    };

    await provider.auth.oauth?.toAuth(oauthCredential);
    await resolveApiKey(provider, { type: "api_key", key: "shared-key" });

    expect(() =>
      provider.stream(
        {
          id: "api-key-model",
          name: "API-key model",
          provider: "litellm",
          api: "openai-completions",
          baseUrl: "https://api-key.example.com/v1",
          reasoning: false,
          input: ["text"],
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
          contextWindow: 4096,
          maxTokens: 1024,
        },
        { messages: [] },
        { apiKey: "shared-key" },
      ),
    ).not.toThrow();
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
      interaction(vi.fn().mockResolvedValueOnce("https://proxy.example.com").mockResolvedValueOnce(`!${helperPath}`)),
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
      interaction(vi.fn().mockResolvedValueOnce("https://proxy.example.com").mockResolvedValueOnce(`!${helperPath}`)),
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
      baseUrl: "https://proxy.example.com",
    };

    const refreshed = await pi.providers[0]?.auth.oauth?.refresh(credential, TEST_SIGNAL);
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
        if (options.placeholder) return "https://proxy.example.com";
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
        url: "https://proxy.example.com/sso/key/generate",
        instructions: "Authenticate via SSO, then copy your token from the LiteLLM UI.",
      },
    ]);
    expect(credential).toMatchObject({
      access: "sk-virtual-abc",
      refresh: "",
      expires: Number.MAX_SAFE_INTEGER,
      baseUrl: "https://proxy.example.com",
    });
    await expect(pi.providers[0]?.auth.oauth?.toAuth(credential!)).resolves.toEqual({
      apiKey: "sk-virtual-abc",
    });
    expect(seenRequests).toContainEqual(
      expect.objectContaining({
        url: "https://proxy.example.com/key/generate",
        method: "POST",
        authorization: `Bearer ${jwt}`,
      }),
    );
    expect(seenRequests.every(({ url }) => !url.endsWith("/model/info"))).toBe(true);
  });

  it("completes CLI SSO with the server lifetime and selected team", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const requests: Array<{ url: string; method: string; pollSecret: string | null }> = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      requests.push({
        url,
        method: String(init?.method ?? "GET"),
        pollSecret: new Headers(init?.headers).get("x-litellm-cli-poll-secret"),
      });
      if (url.endsWith("/sso/cli/start"))
        return jsonResponse(200, {
          login_id: "cli-login",
          poll_secret: "poll-secret",
          user_code: "ABCD-EFGH",
          expires_in: 600,
        });
      if (url.endsWith("/sso/cli/poll/cli-login"))
        return jsonResponse(200, {
          status: "ready",
          requires_team_selection: true,
          team_details: [{ id: "team-b", team_alias: "Beta" }],
        });
      if (url.endsWith("/sso/cli/poll/cli-login?team_id=team-b"))
        return jsonResponse(200, { status: "ready", key: "opaque-cli-token", expires_in: 7200 });
      throw new Error(`unexpected URL: ${url} (${String(init?.method ?? "GET")})`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const startedAt = Date.now();
    const deviceCodes: Array<{ userCode: string; verificationUri: string }> = [];

    const credential = await loginOAuth(pi.providers[0]!, {
      onPrompt: async (prompt) => (prompt.placeholder ? "https://proxy.example.com" : "team-b"),
      onDeviceCode: (event) => deviceCodes.push(event),
      signal: new AbortController().signal,
    });

    expect(deviceCodes).toEqual([
      {
        type: "device_code",
        userCode: "ABCD-EFGH",
        verificationUri: "https://proxy.example.com/sso/key/generate?source=litellm-cli&key=cli-login",
        expiresInSeconds: 600,
      },
    ]);
    expect(credential).toMatchObject({
      access: "opaque-cli-token",
      refresh: "",
      baseUrl: "https://proxy.example.com",
    });
    expect(credential?.expires).toBeGreaterThanOrEqual(startedAt + 7200 * 1000);
    expect(requests).toEqual([
      { url: "https://proxy.example.com/sso/cli/start", method: "POST", pollSecret: null },
      { url: "https://proxy.example.com/sso/cli/poll/cli-login", method: "GET", pollSecret: "poll-secret" },
      {
        url: "https://proxy.example.com/sso/cli/poll/cli-login?team_id=team-b",
        method: "GET",
        pollSecret: "poll-secret",
      },
    ]);
  }, 15_000);

  it("retries pending and transient CLI SSO polls", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    let polls = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/sso/cli/start"))
        return jsonResponse(200, { login_id: "cli-login", poll_secret: "poll-secret", user_code: "ABCD-EFGH" });
      if (url.endsWith("/sso/cli/poll/cli-login")) {
        polls += 1;
        if (polls === 1) return jsonResponse(200, { status: "pending" });
        if (polls === 2) return jsonResponse(503, { detail: "unavailable" });
        return jsonResponse(200, { status: "ready", key: "opaque-cli-token" });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await expect(
      loginOAuth(pi.providers[0]!, {
        onPrompt: async (prompt) => (prompt.placeholder ? "https://proxy.example.com" : ""),
        onDeviceCode: () => undefined,
        signal: new AbortController().signal,
      }),
    ).resolves.toMatchObject({ access: "opaque-cli-token" });
    expect(polls).toBe(3);
  }, 15_000);

  it("preserves cancellation while waiting for a CLI SSO poll", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    let polled = false;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/sso/cli/start"))
        return jsonResponse(200, { login_id: "cli-login", poll_secret: "poll-secret", user_code: "ABCD-EFGH" });
      if (url.endsWith("/sso/cli/poll/cli-login")) {
        polled = true;
        return jsonResponse(200, { status: "pending" });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const controller = new AbortController();
    const reason = new Error("caller cancelled login");
    const login = loginOAuth(pi.providers[0]!, {
      onPrompt: async (prompt) => (prompt.placeholder ? "https://proxy.example.com" : ""),
      onDeviceCode: () => undefined,
      signal: controller.signal,
    });
    await vi.waitFor(() => expect(polled).toBe(true));
    controller.abort(reason);
    await expect(login).rejects.toBe(reason);
  }, 15_000);

  it("uses legacy token paste only when CLI SSO start is unavailable", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    let status = 404;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/sso/cli/start")) return jsonResponse(status, {});
      if (url.endsWith("/key/generate")) return jsonResponse(200, { key: "sk-legacy" });
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const login = () =>
      loginOAuth(pi.providers[0]!, {
        onPrompt: async (prompt) => (prompt.placeholder ? "https://proxy.example.com" : "Bearer legacy-token"),
        onDeviceCode: () => undefined,
        signal: new AbortController().signal,
      });

    await expect(login()).resolves.toMatchObject({ access: "sk-legacy" });
    status = 500;
    await expect(login()).rejects.toThrow("LiteLLM CLI SSO start failed (HTTP 500)");
  }, 15_000);

  it("uses configured and default CLI token lifetimes when poll expiry is absent", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/sso/cli/start"))
        return jsonResponse(200, { login_id: "cli-login", poll_secret: "poll-secret", user_code: "ABCD-EFGH" });
      if (url.endsWith("/sso/cli/poll/cli-login"))
        return jsonResponse(200, { status: "ready", key: "opaque-cli-token" });
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const login = () =>
      loginOAuth(pi.providers[0]!, {
        onPrompt: async (prompt) => (prompt.placeholder ? "https://proxy.example.com" : ""),
        onDeviceCode: () => undefined,
        signal: new AbortController().signal,
      });

    process.env.LITELLM_CLI_JWT_EXPIRATION_HOURS = "48";
    const configuredAt = Date.now();
    const configured = await login();
    expect(configured?.expires).toBeGreaterThanOrEqual(configuredAt + 48 * 60 * 60 * 1000);
    delete process.env.LITELLM_CLI_JWT_EXPIRATION_HOURS;
    const defaultAt = Date.now();
    const fallback = await login();
    expect(fallback?.expires).toBeGreaterThanOrEqual(defaultAt + 24 * 60 * 60 * 1000);
  }, 15_000);

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
        if (options.placeholder) return "https://proxy.example.com";
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
        if (options.placeholder) return "https://proxy.example.com";
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
      .mockImplementationOnce(nativeTimeout)
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
        if (options.placeholder) return "https://proxy.example.com";
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
        if (options.placeholder) return "https://proxy.example.com";
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
        if (options.placeholder) return "https://proxy.example.com";
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
        if (options.placeholder) return "https://proxy.example.com";
        if (options.message.includes("Select login method")) return "2";
        if (options.message.includes("SSO token")) return jwt;
        return "y";
      },
      signal: new AbortController().signal,
    });

    await expect(pi.providers[0]?.auth.oauth?.refresh(credential!, TEST_SIGNAL)).rejects.toThrow(
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
      if (url.endsWith("/key/generate")) return jsonResponse(403, { error: "forbidden", token: "remote-secret" });
      if (url.endsWith("/model/info"))
        return jsonResponse(200, { data: [{ model_name: "gpt-4o", model_info: { mode: "chat" } }] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const credential = await loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => {
        if (options.placeholder) return "https://proxy.example.com";
        if (options.message.includes("Select login method")) return "2";
        if (options.message.includes("SSO token")) return jwt;
        return "y";
      },
      onProgress: progress,
      signal: new AbortController().signal,
    });

    expect(credential).toMatchObject({ access: jwt, refresh: "" });
    expect(progress).toHaveBeenCalledWith(expect.stringContaining("virtual key generation failed"));
    expect(progress.mock.calls.flat().join(" ")).not.toContain("remote-secret");
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
          if (options.placeholder) return "https://proxy.example.com";
          if (options.message.includes("Select login method")) return "2";
          return "";
        },
        signal: new AbortController().signal,
      }),
    ).rejects.toThrow("SSO token is required");
  });
});

describe("login base URL reuse", () => {
  const STORED_URL = "https://stored.example.com";

  async function agentDirWithStoredOAuth(): Promise<string> {
    const agentDir = await makeAgentDir();
    await writeFile(
      join(agentDir, "auth.json"),
      JSON.stringify({
        litellm: { type: "oauth", access: "expired-token", refresh: "", expires: 0, baseUrl: STORED_URL },
      }),
      "utf8",
    );
    return agentDir;
  }

  it("offers the stored credential's base URL instead of asking for it again", async () => {
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    delete process.env.LITELLM_BASE_URL;
    const extension = await loadExtension(await agentDirWithStoredOAuth());
    const pi = createPi();
    await extension(pi);

    const messages: string[] = [];
    const credential = await loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => {
        messages.push(options.message);
        if (options.options) return options.options.find((option) => option.id === STORED_URL)!.id;
        return options.type === "secret" ? "sk-sso-token" : "n";
      },
      signal: new AbortController().signal,
    });

    expect(credential?.baseUrl).toBe(STORED_URL);
    expect(messages.some((message) => message.includes("Enter LiteLLM proxy URL"))).toBe(false);
  });

  it("names where the offered base URL came from", async () => {
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    delete process.env.LITELLM_BASE_URL;
    const extension = await loadExtension(await agentDirWithStoredOAuth());
    const pi = createPi();
    await extension(pi);

    let offered: readonly { id: string; label: string; description?: string }[] | undefined;
    await loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => {
        if (options.options) {
          offered = options.options;
          return options.options.find((option) => option.id === STORED_URL)!.id;
        }
        return options.type === "secret" ? "sk-sso-token" : "n";
      },
      signal: new AbortController().signal,
    });

    expect(offered?.map((option) => option.label)).toEqual([
      `${STORED_URL} (previous login)`,
      "Enter a different URL…",
    ]);
    expect(offered?.[0]?.id).toBe(STORED_URL);
  });

  it("still asks for a URL when the offered one is declined", async () => {
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    delete process.env.LITELLM_BASE_URL;
    const extension = await loadExtension(await agentDirWithStoredOAuth());
    const pi = createPi();
    await extension(pi);

    const types: string[] = [];
    const credential = await loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => {
        types.push(options.type);
        if (options.options) return options.options.find((option) => option.id !== STORED_URL)!.id;
        if (options.placeholder) return "https://other.example.com";
        return options.type === "secret" ? "sk-sso-token" : "n";
      },
      signal: new AbortController().signal,
    });

    expect(types.slice(0, 2)).toEqual(["select", "text"]);
    expect(credential?.baseUrl).toBe("https://other.example.com");
  });

  it("offers LITELLM_BASE_URL to the API-key login", async () => {
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    process.env.LITELLM_BASE_URL = "https://env.example.com";
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);

    const prompt = vi.fn(async (options: Parameters<AuthInteraction["prompt"]>[0]) =>
      "options" in options ? options.options.find((option) => option.id === "https://env.example.com")!.id : "sk-typed",
    );
    const credential = await pi.providers[0]?.auth.apiKey?.login?.(interaction(prompt));

    expect(credential?.env?.LITELLM_BASE_URL).toBe("https://env.example.com");
    expect(prompt.mock.calls.map(([options]) => options.type)).toEqual(["select", "secret"]);
  });

  it("ignores a stored base URL that is no longer usable", async () => {
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    delete process.env.LITELLM_BASE_URL;
    const agentDir = await makeAgentDir();
    await writeFile(
      join(agentDir, "auth.json"),
      JSON.stringify({
        litellm: { type: "api_key", key: "sk-old", env: { LITELLM_BASE_URL: "http://insecure.example.com" } },
      }),
      "utf8",
    );
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const prompt = vi.fn(async (options: Parameters<AuthInteraction["prompt"]>[0]) =>
      "placeholder" in options && options.placeholder ? "https://typed.example.com" : "sk-typed",
    );
    const credential = await pi.providers[0]?.auth.apiKey?.login?.(interaction(prompt));

    expect(prompt.mock.calls.map(([options]) => options.type)).toEqual(["text", "secret"]);
    expect(credential?.env?.LITELLM_BASE_URL).toBe("https://typed.example.com");
  });

  it("asks for the proxy URL when nothing is configured yet", async () => {
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    delete process.env.LITELLM_BASE_URL;
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);

    const prompt = vi.fn(async (options: Parameters<AuthInteraction["prompt"]>[0]) =>
      "placeholder" in options && options.placeholder ? "https://typed.example.com" : "sk-typed",
    );
    const credential = await pi.providers[0]?.auth.apiKey?.login?.(interaction(prompt));

    expect(prompt.mock.calls.map(([options]) => options.type)).toEqual(["text", "secret"]);
    expect(credential?.env?.LITELLM_BASE_URL).toBe("https://typed.example.com");
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
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
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
        if (options.placeholder) return "https://proxy.example.com";
        if (options.message.includes("Select login method")) return "2";
        if (options.message.includes("SSO token")) return jwt;
        return "y";
      },
      signal: new AbortController().signal,
    });

    expect(seenRequests).toContainEqual({ url: "https://proxy.example.com/key/generate", customer: "team-a" });
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
