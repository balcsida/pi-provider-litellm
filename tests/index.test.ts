import { createHash } from "node:crypto";
import { mkdtemp, readdir, readFile, writeFile } from "node:fs/promises";
import { request as httpRequest } from "node:http";
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

// Generated tool names carry a 10-hex identity hash; assert the contract, not a literal digest.
const named = (base: string) => expect.stringMatching(new RegExp(`^${base}_[a-f0-9]{10}$`));

async function refreshProvider(
  provider: Provider,
  options: Omit<RefreshModelsContext, "publish" | "stored" | "signal"> & {
    store?: ModelsStore;
    signal?: AbortSignal;
  },
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
    signal: options.signal ?? new AbortController().signal,
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
    pkce?: true;
    signal?: AbortSignal;
  },
) {
  const fetchImpl = globalThis.fetch;
  globalThis.fetch = (input, init) => {
    const url = String(input);
    if (!callbacks.pkce && url.endsWith("/.well-known/litellm-cli-auth")) {
      return Promise.resolve(jsonResponse(404, { detail: "Not Found" }));
    }
    if (!callbacks.onDeviceCode && url.endsWith("/sso/cli/start")) {
      return Promise.resolve(jsonResponse(404, { detail: "Not Found" }));
    }
    return fetchImpl(input, init);
  };
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
  it("registers one complete native provider and the request hooks", async () => {
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
    expect(pi.handlers.get("before_provider_headers")).toHaveLength(1);
    expect(pi.handlers.get("before_provider_request")).toHaveLength(1);
    expect(pi.commands.has("litellm-refresh")).toBe(true);
  });

  it("warns once per provider and route when a LiteLLM fallback serves the request", async () => {
    const agentDir = await makeAgentDir();
    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({ litellm: { providers: { "litellm-alias": {} } } }),
    );
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const notify = vi.fn();
    const respond = (provider: string, id: string, attempted?: string): void => {
      const headers = attempted === undefined ? {} : { "x-litellm-attempted-fallbacks": attempted };
      for (const handler of pi.handlers.get("after_provider_response") ?? []) {
        handler(
          { type: "after_provider_response", status: 200, headers },
          { model: { provider, id }, hasUI: true, ui: { notify } },
        );
      }
    };

    respond("openai", "high", "1");
    respond("litellm-unregistered", "high", "1");
    respond("litellm", "high");
    respond("litellm", "high", "0");
    expect(notify).not.toHaveBeenCalled();
    respond("litellm", "high", "1");
    respond("litellm", "high", "2");
    respond("litellm", "low", "1");
    respond("litellm-alias", "high", "1");
    respond("litellm-alias", "high", "2");

    expect(notify.mock.calls).toEqual([
      [expect.stringContaining('LiteLLM ("litellm"): a fallback served "high"'), "warning"],
      [expect.stringContaining('LiteLLM ("litellm"): a fallback served "low"'), "warning"],
      [expect.stringContaining('LiteLLM ("litellm-alias"): a fallback served "high"'), "warning"],
    ]);
  });

  it.each([
    [199, "1"],
    [300, "1"],
    [503, "1"],
    [200, ""],
    [200, "-1"],
    [200, "0.5"],
    [200, "Infinity"],
    [200, "not-a-count"],
  ])("ignores a fallback header on status %s with count %s", async (status, attempted) => {
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);
    const notify = vi.fn();
    for (const handler of pi.handlers.get("after_provider_response") ?? []) {
      handler(
        { type: "after_provider_response", status, headers: { "x-litellm-attempted-fallbacks": attempted } },
        { model: { provider: "litellm", id: "high" }, hasUI: true, ui: { notify } },
      );
    }
    expect(notify).not.toHaveBeenCalled();
  });

  it("escapes provider and route names in a headless fallback warning", async () => {
    const provider = 'alias"\n\u001b[31m';
    const agentDir = await makeAgentDir();
    await writeFile(join(agentDir, "settings.json"), JSON.stringify({ litellm: { providers: { [provider]: {} } } }));
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    for (const handler of pi.handlers.get("after_provider_response") ?? []) {
      handler(
        {
          type: "after_provider_response",
          status: 200,
          headers: {
            "x-litellm-attempted-fallbacks": "1",
            "x-litellm-model-name": "private-backend",
            "x-litellm-model-api-base": "https://private-backend.example.com",
          },
        },
        { model: { provider, id: 'high"\n\u001b[31m' }, hasUI: false },
      );
    }
    expect(stderr).toHaveBeenCalledTimes(1);
    const output = String(stderr.mock.calls[0]?.[0]);
    expect(output).toContain('a fallback served "high\\"\\n\\u001b[31m"');
    expect(output.trimEnd()).not.toContain("\n");
    expect(output).not.toContain("\u001b");
    expect(output).toContain('LiteLLM ("alias\\"\\n\\u001b[31m"):');
    expect(output).not.toContain("private-backend");
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
      litellmDiscoveryVersion: 3,
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
      expect(pi.tools.map((tool) => tool.name)).toContainEqual(named("mcp_brave_search"));
    });
    expect(pi.providers[0]?.getModels()).toEqual([stored]);
  });

  it.each([
    ["unexpected_error", "MCP discovery reported an unexpected proxy error"],
    ["proxy-owned-tag", "MCP discovery reported an error"],
  ])("does not expose the %s discovery envelope in stderr", async (proxyTag, expectedText) => {
    process.env.LITELLM_MODELS_DEV = "0";
    const proxyMessage = "private proxy details";
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/mcp-rest/tools/list")) {
        return jsonResponse(200, { tools: [], error: proxyTag, message: proxyMessage });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);

    await refreshProvider(pi.providers[0]!, {
      allowNetwork: true,
      credential: {
        type: "api_key",
        key: "sk-test",
        env: { LITELLM_BASE_URL: "https://proxy.example.com" },
      },
    });
    await vi.waitFor(() =>
      expect(stderr.mock.calls.map(([message]) => String(message)).join("")).toContain(expectedText),
    );

    const output = stderr.mock.calls.map(([message]) => String(message)).join("");
    expect(output).not.toContain(proxyTag);
    expect(output).not.toContain(proxyMessage);
  });

  it.each(["api_key", "oauth"] as const)(
    "keeps denied MCP discovery paused across restarts until %s login succeeds",
    async (loginType) => {
      process.env.LITELLM_MODELS_DEV = "0";
      const agentDir = await makeAgentDir();
      let listCalls = 0;
      vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
        const url = String(input);
        if (url.endsWith("/token"))
          return jsonResponse(200, {
            access_token: "rotated-access",
            refresh_token: "rotated-refresh",
            token_type: "bearer",
            expires_in: 3600,
          });
        if (url.endsWith("/model/info"))
          return jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
        if (url.endsWith("/mcp-rest/tools/list")) {
          listCalls += 1;
          return listCalls === 1
            ? jsonResponse(200, {
                tools: [],
                error: "unexpected_error",
                message: "The key is not allowed to access any MCP servers.",
              })
            : jsonResponse(200, {
                tools: [{ name: "search", server_name: "server", inputSchema: { type: "object", properties: {} } }],
              });
        }
        throw new Error(`unexpected URL: ${url}`);
      });
      const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
      let pi = createPi();
      await (await loadExtension(agentDir))(pi);
      const credential: Credential = {
        type: "oauth",
        access: "old-access",
        refresh: "old-refresh",
        expires: Date.now() + 3600000,
        baseUrl: "https://proxy.example.com",
        flow: "litellm_cli_pkce",
        clientId: "llm_dcrc_client",
        tokenEndpoint: "https://proxy.example.com/token",
        resource: "https://proxy.example.com",
      };
      const refresh = (current: Credential = credential) =>
        refreshProvider(pi.providers[0]!, { allowNetwork: true, credential: current });

      await refresh();
      await vi.waitFor(() => expect(stderr.mock.calls.flat().join("")).toContain("MCP"));
      await refresh();
      expect(listCalls).toBe(1);

      const rotated = await pi.providers[0]!.auth.oauth!.refresh(credential, TEST_SIGNAL);
      pi = createPi();
      await (await loadExtension(agentDir))(pi);
      await refresh(rotated);
      expect(listCalls).toBe(1);
      expect(pi.providers[0]?.getModels().map((model) => model.id)).toEqual(["fresh-model"]);

      await expect(
        pi.providers[0]?.auth[loginType === "oauth" ? "oauth" : "apiKey"]?.login?.(
          interaction(async () => {
            throw new Error("login cancelled");
          }),
        ),
      ).rejects.toThrow("login cancelled");
      await refresh();
      expect(listCalls).toBe(1);

      const current =
        loginType === "oauth"
          ? await loginOAuth(pi.providers[0]!, {
              onPrompt: vi
                .fn()
                .mockResolvedValueOnce("https://proxy.example.com")
                .mockResolvedValueOnce("new-access")
                .mockResolvedValueOnce("n"),
            })
          : await pi.providers[0]?.auth.apiKey?.login?.(
              interaction(vi.fn().mockResolvedValueOnce("https://proxy.example.com").mockResolvedValueOnce("sk-new")),
            );
      await refresh(current);
      await vi.waitFor(() => expect(pi.tools.map((tool) => tool.name)).toContainEqual(named("mcp_server_search")));
      expect(listCalls).toBe(2);

      // Other processes can still use the old login after this one signs in successfully.
      pi = createPi();
      await (await loadExtension(agentDir))(pi);
      await refresh(rotated);
      expect(listCalls).toBe(2);
    },
  );

  it.each(["key change", "header change", "same-key login"] as const)(
    "isolates persisted MCP pauses across processes for %s",
    async (changed) => {
      process.env.LITELLM_MODELS_DEV = "0";
      const agentDir = await makeAgentDir();
      let releaseOld!: (response: Response) => void;
      const pendingOld = new Promise<Response>((resolve) => {
        releaseOld = resolve;
      });
      const calls: string[] = [];
      vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
        const url = String(input);
        if (url.endsWith("/model/info"))
          return jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
        if (url.endsWith("/mcp-rest/tools/list")) {
          const headers = new Headers(init?.headers);
          const scope = `${headers.get("Authorization")}:${headers.get("x-auth")}`;
          calls.push(scope);
          return calls.length === 1
            ? pendingOld
            : jsonResponse(200, {
                tools: [{ name: "search", server_name: "server", inputSchema: { type: "object", properties: {} } }],
              });
        }
        throw new Error(`unexpected URL: ${url}`);
      });
      const oldCredential: Credential = {
        type: "api_key",
        key: "sk-old",
        env: { LITELLM_BASE_URL: "https://proxy.example.com" },
      };
      const oldPi = createPi();
      await (await loadExtension(agentDir))(oldPi);
      await refreshProvider(oldPi.providers[0]!, { allowNetwork: true, credential: oldCredential });
      await vi.waitFor(() => expect(calls).toHaveLength(1));
      // Module isolation models independent Pi processes with independent per-process hash salts.
      const newPi = createPi();
      await (await loadExtension(agentDir))(newPi);
      const newCredential =
        changed === "same-key login"
          ? await newPi.providers[0]?.auth.apiKey?.login?.(
              interaction(vi.fn().mockResolvedValueOnce("https://proxy.example.com").mockResolvedValueOnce("sk-old")),
            )
          : { ...oldCredential, key: changed === "key change" ? "sk-new" : "sk-old" };
      releaseOld(jsonResponse(403, {}));
      const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
      await vi.waitFor(() => expect(stderr.mock.calls.flat().join("")).toContain("discovery paused"));
      if (changed === "header change") process.env.LITELLM_HEADERS = JSON.stringify({ "x-auth": "new-header-secret" });
      // Start again after the old process writes its denial, before discovering the new catalog.
      const restartedPi = createPi();
      await (await loadExtension(agentDir))(restartedPi);
      await refreshProvider(restartedPi.providers[0]!, { allowNetwork: true, credential: newCredential });
      await vi.waitFor(() =>
        expect(restartedPi.tools.map((tool) => tool.name)).toContainEqual(named("mcp_server_search")),
      );
      expect(calls).toHaveLength(2);
      delete process.env.LITELLM_HEADERS;
      await refreshProvider(oldPi.providers[0]!, { allowNetwork: true, credential: oldCredential });
      expect(calls).toHaveLength(2);
      const entries = await readdir(join(agentDir, "litellm-mcp-pauses"));
      expect(entries.join(" ")).not.toMatch(/sk-old|sk-new|new-header-secret/);
      for (const name of entries.filter((name) => name.startsWith("paused-")))
        expect(await readFile(join(agentDir, "litellm-mcp-pauses", name), "utf8")).toBe("");
    },
  );

  it.each([true, false])("preserves a late MCP denial across OAuth refresh (persistence: %s)", async (persistent) => {
    process.env.LITELLM_MODELS_DEV = "0";
    const agentDir = await makeAgentDir();
    if (!persistent) await writeFile(join(agentDir, "litellm-mcp-pauses"), "unavailable directory");
    let release!: (response: Response) => void;
    const pending = new Promise<Response>((resolve) => {
      release = resolve;
    });
    let listCalls = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/token"))
        return jsonResponse(200, {
          access_token: "new-access",
          refresh_token: "new-refresh",
          expires_in: 3600,
          token_type: "bearer",
        });
      if (url.endsWith("/mcp-rest/tools/list")) {
        listCalls += 1;
        return pending;
      }
      return jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
    });
    const oldCredential: Credential = {
      type: "oauth",
      access: "old-access",
      refresh: "old-refresh",
      expires: Date.now() + 3600000,
      flow: "litellm_cli_pkce",
      clientId: "client",
      tokenEndpoint: "https://proxy.example.com/token",
      resource: "https://proxy.example.com",
      baseUrl: "https://proxy.example.com",
    };
    const pi = createPi();
    await (await loadExtension(agentDir))(pi);
    await refreshProvider(pi.providers[0]!, { allowNetwork: true, credential: oldCredential });
    await vi.waitFor(() => expect(listCalls).toBe(1));
    const refreshed = await pi.providers[0]!.auth.oauth!.refresh(oldCredential, TEST_SIGNAL);
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    release(jsonResponse(403, {}));
    await vi.waitFor(() => expect(stderr.mock.calls.flat().join("")).toContain("discovery paused"));
    const restartedPi = persistent ? createPi() : pi;
    if (persistent) await (await loadExtension(agentDir))(restartedPi);
    await refreshProvider(restartedPi.providers[0]!, { allowNetwork: true, credential: refreshed });
    expect(listCalls).toBe(1);

    await restartedPi.providers[0]!.auth.apiKey!.login!(
      interaction(vi.fn().mockResolvedValueOnce("https://proxy.example.com").mockResolvedValueOnce("sk-new")),
    );
    await refreshProvider(restartedPi.providers[0]!, { allowNetwork: true, credential: refreshed });
    expect(listCalls).toBe(1);
  });

  it("keeps keyless helper credentials paused when the helper rotates tokens", async () => {
    process.env.LITELLM_MODELS_DEV = "0";
    const agentDir = await makeAgentDir();
    process.env.LITELLM_API_KEY_HELPER = await writeHelper(agentDir, ["first", "second", "third", "fourth"]);
    let listCalls = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      if (String(input).endsWith("/mcp-rest/tools/list")) {
        listCalls += 1;
        return jsonResponse(403, {});
      }
      return jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
    });
    const pi = createPi();
    await (await loadExtension(agentDir))(pi);
    const credential: Credential = { type: "api_key", env: { LITELLM_BASE_URL: "https://proxy.example.com" } };
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    await refreshProvider(pi.providers[0]!, { allowNetwork: true, credential });
    await vi.waitFor(() => expect(stderr.mock.calls.flat().join("")).toContain("discovery paused"));
    await refreshProvider(pi.providers[0]!, { allowNetwork: true, credential });
    expect(listCalls).toBe(1);
    expect(await readHelperCount(agentDir)).toBeGreaterThan(2);
  });

  it("ignores an MCP denial from a request started before a successful login", async () => {
    process.env.LITELLM_MODELS_DEV = "0";
    let release!: (response: Response) => void;
    const pending = new Promise<Response>((resolve) => {
      release = resolve;
    });
    let listCalls = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info"))
        return jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
      if (url.endsWith("/mcp-rest/tools/list")) {
        listCalls += 1;
        return listCalls === 1
          ? pending
          : jsonResponse(200, {
              tools: [{ name: "search", server_name: "server", inputSchema: { type: "object", properties: {} } }],
            });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const pi = createPi();
    await (await loadExtension(await makeAgentDir()))(pi);
    await refreshProvider(pi.providers[0]!, {
      allowNetwork: true,
      credential: { type: "api_key", key: "sk-old", env: { LITELLM_BASE_URL: "https://proxy.example.com" } },
    });
    await vi.waitFor(() => expect(listCalls).toBe(1));
    const credential = await pi.providers[0]?.auth.apiKey?.login?.(
      interaction(vi.fn().mockResolvedValueOnce("https://proxy.example.com").mockResolvedValueOnce("sk-new")),
    );
    release(jsonResponse(403, { detail: "access denied" }));
    await refreshProvider(pi.providers[0]!, { allowNetwork: true, credential });
    await vi.waitFor(() => expect(pi.tools.map((tool) => tool.name)).toContainEqual(named("mcp_server_search")));
    expect(listCalls).toBe(2);
  });

  it("does not start MCP discovery from a model refresh that predates login", async () => {
    process.env.LITELLM_MODELS_DEV = "0";
    let release!: (response: Response) => void;
    const pending = new Promise<Response>((resolve) => {
      release = resolve;
    });
    let modelCalls = 0;
    const mcpKeys: Array<string | null> = [];
    const models = () => jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return ++modelCalls === 1 ? pending : models();
      if (url.endsWith("/mcp-rest/tools/list")) {
        mcpKeys.push(new Headers(init?.headers).get("Authorization"));
        return mcpKeys.at(-1) === "Bearer sk-old"
          ? jsonResponse(403, {})
          : jsonResponse(200, {
              tools: [{ name: "search", server_name: "server", inputSchema: { type: "object", properties: {} } }],
            });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const pi = createPi();
    await (await loadExtension(await makeAgentDir()))(pi);
    const oldRefresh = refreshProvider(pi.providers[0]!, {
      allowNetwork: true,
      credential: { type: "api_key", key: "sk-old", env: { LITELLM_BASE_URL: "https://proxy.example.com" } },
    });
    await vi.waitFor(() => expect(modelCalls).toBe(1));
    const credential = await pi.providers[0]?.auth.apiKey?.login?.(
      interaction(vi.fn().mockResolvedValueOnce("https://proxy.example.com").mockResolvedValueOnce("sk-new")),
    );
    release(models());
    await oldRefresh;
    expect(mcpKeys).toEqual([]);
    await refreshProvider(pi.providers[0]!, { allowNetwork: true, credential });
    await vi.waitFor(() => expect(pi.tools.map((tool) => tool.name)).toContainEqual(named("mcp_server_search")));
    expect(mcpKeys).toEqual(["Bearer sk-new"]);
  });

  it.each([
    [403, {}, "discovery paused until /login litellm"],
    [200, { tools: [], error: "unexpected_error", message: "private-proxy-text" }, "unexpected proxy error"],
    [200, { tools: [] }, "no MCP tools were registered"],
    [200, "x".repeat(5 * 1024 * 1024 + 1), "exceeded its 5242880-byte limit"],
    [
      200,
      {
        tools: [
          {
            name: "search",
            server_name: "server",
            inputSchema: { type: "object", properties: { query: { type: "string", pattern: "private-proxy-text" } } },
          },
        ],
      },
      "safe args envelope",
    ],
  ])("routes MCP diagnostics through the active UI %#", async (status, body, expected) => {
    process.env.LITELLM_MODELS_DEV = "0";
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      if (String(input).endsWith("/mcp-rest/tools/list")) return jsonResponse(status, body);
      return jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
    });
    const pi = createPi();
    await (await loadExtension(await makeAgentDir()))(pi);
    const notify = vi.fn();
    for (const handler of pi.handlers.get("session_start") ?? [])
      await handler({ type: "session_start" }, { hasUI: true, ui: { notify } });
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    process.env.LITELLM_VERBOSE_DISCOVERY = "1";
    await refreshProvider(pi.providers[0]!, {
      allowNetwork: true,
      credential: { type: "api_key", key: "sk-test", env: { LITELLM_BASE_URL: "https://proxy.example.com" } },
    });
    await vi.waitFor(() => expect(notify.mock.calls.flat().join("\n")).toContain(expected));
    expect(notify.mock.calls.flat().join("\n")).toContain("Querying MCP tools/list endpoint");
    expect(notify.mock.calls.flat().join("\n")).not.toContain("private-proxy-text");
    expect(stderr.mock.calls.flat().join("\n")).not.toContain("MCP");
  });

  it("routes MCP tool-call safety diagnostics through the active UI", async () => {
    process.env.LITELLM_MODELS_DEV = "0";
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/mcp-rest/tools/call")) return new Response("x".repeat(5 * 1024 * 1024 + 1));
      if (url.endsWith("/mcp-rest/tools/list"))
        return jsonResponse(200, {
          tools: [{ name: "search", server_name: "server", inputSchema: { type: "object", properties: {} } }],
        });
      return jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
    });
    const pi = createPi();
    await (await loadExtension(await makeAgentDir()))(pi);
    const notify = vi.fn();
    for (const handler of pi.handlers.get("session_start") ?? [])
      await handler({ type: "session_start" }, { hasUI: true, ui: { notify } });
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    await refreshProvider(pi.providers[0]!, {
      allowNetwork: true,
      credential: { type: "api_key", key: "sk-test", env: { LITELLM_BASE_URL: "https://proxy.example.com" } },
    });
    await vi.waitFor(() => expect(pi.tools.map((tool) => tool.name)).toContainEqual(named("mcp_server_search")));
    const tool = pi.tools.find((tool) => tool.name.startsWith("mcp_"));
    await expect(tool?.execute?.("test-call", {}, new AbortController().signal)).rejects.toThrow(
      "exceeds its 5242880-byte limit",
    );
    expect(notify).toHaveBeenCalledWith(expect.stringContaining("MCP tool call response exceeded"), "warning");
    expect(stderr).not.toHaveBeenCalled();
  });

  it("buffers MCP warnings before the TUI context is available", async () => {
    process.env.LITELLM_MODELS_DEV = "0";
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) =>
      String(input).endsWith("/mcp-rest/tools/list")
        ? jsonResponse(403, {})
        : jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] }),
    );
    const agentDir = await makeAgentDir();
    const pi = createPi();
    await (await loadExtension(agentDir))(pi);
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const isTTY = Object.getOwnPropertyDescriptor(process.stderr, "isTTY");
    Object.defineProperty(process.stderr, "isTTY", { configurable: true, value: true });
    try {
      await refreshProvider(pi.providers[0]!, {
        allowNetwork: true,
        credential: { type: "api_key", key: "sk-test", env: { LITELLM_BASE_URL: "https://proxy.example.com" } },
      });
      await vi.waitFor(async () =>
        expect((await readdir(join(agentDir, "litellm-mcp-pauses"))).some((name) => name.startsWith("paused-"))).toBe(
          true,
        ),
      );
      expect(stderr).not.toHaveBeenCalled();
      const notify = vi.fn();
      for (const handler of pi.handlers.get("session_start") ?? [])
        await handler({ type: "session_start" }, { hasUI: true, ui: { notify } });
      expect(notify).toHaveBeenCalledWith(expect.stringContaining("discovery paused until /login litellm"), "warning");
      expect(stderr).not.toHaveBeenCalled();
    } finally {
      if (isTTY) Object.defineProperty(process.stderr, "isTTY", isTTY);
      else Reflect.deleteProperty(process.stderr, "isTTY");
    }
  });

  it("retries a partial-failure catalog and registers tools from the clean refresh", async () => {
    process.env.LITELLM_MODELS_DEV = "0";
    let listCalls = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/mcp-rest/tools/list")) {
        listCalls += 1;
        const tools = [
          { name: "search", server_name: "server", inputSchema: { type: "object", properties: {} } },
          { name: "browse", server_name: "server", inputSchema: { type: "object", properties: {} } },
        ];
        return jsonResponse(
          200,
          listCalls === 1
            ? { tools: tools.slice(0, 1), error: "partial_failure", message: "private proxy details" }
            : { tools },
        );
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);

    const refresh = () =>
      refreshProvider(pi.providers[0]!, {
        allowNetwork: true,
        credential: {
          type: "api_key",
          key: "sk-test",
          env: { LITELLM_BASE_URL: "https://proxy.example.com" },
        },
      });

    await refresh();
    await vi.waitFor(() => expect(pi.tools.map((tool) => tool.name)).toContainEqual(named("mcp_server_search")));
    await refresh();
    await vi.waitFor(() => {
      expect(listCalls).toBe(2);
      expect(pi.tools.map((tool) => tool.name).filter((name) => name.startsWith("mcp_"))).toEqual([
        named("mcp_server_search"),
        named("mcp_server_browse"),
      ]);
    });
    const diagnostics = stderr.mock.calls
      .map(([message]) => String(message))
      .filter((message) => message.includes("partial server failure"));
    expect(diagnostics).toEqual(["LiteLLM MCP: proxy reported a partial server failure; 1 tool registered.\n"]);
    expect(stderr.mock.calls.map(([message]) => String(message)).join("")).not.toContain("private proxy details");
  });

  it("does not settle the identity when a catalog yields no registrable tool", async () => {
    process.env.LITELLM_MODELS_DEV = "0";
    let listCalls = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/mcp-rest/tools/list")) {
        listCalls += 1;
        return jsonResponse(200, { tools: [] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    await extension(pi);

    const refresh = () =>
      refreshProvider(pi.providers[0]!, {
        allowNetwork: true,
        credential: { type: "api_key", key: "sk-test", env: { LITELLM_BASE_URL: "https://proxy.example.com" } },
        signal: new AbortController().signal,
      });

    await refresh();
    await refresh();

    // Not settled: discovery is retried, and the silence is explained rather than silent.
    expect(listCalls).toBe(2);
    expect(stderr.mock.calls.map(([message]) => String(message)).join("")).toContain(
      "no MCP tools were registered from 0 raw entries",
    );
  });

  it("coalesces concurrent MCP refreshes when registration becomes fatal", async () => {
    process.env.LITELLM_MODELS_DEV = "0";
    const proxyText = "proxy-description-must-not-leak";
    const refusal = `extension stale ${"x".repeat(500)}`;
    let listCalls = 0;
    let mcpStarted!: () => void;
    let releaseMcp!: (response: Response) => void;
    const started = new Promise<void>((resolve) => {
      mcpStarted = resolve;
    });
    const pendingMcp = new Promise<Response>((resolve) => {
      releaseMcp = resolve;
    });
    const catalogResponse = () =>
      jsonResponse(200, {
        tools: ["first", "second", "third"].map((name) => ({
          name,
          server_name: "server",
          description: proxyText,
          inputSchema: { type: "object", properties: {} },
        })),
      });
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/mcp-rest/tools/list")) {
        listCalls += 1;
        if (listCalls === 1) {
          mcpStarted();
          return pendingMcp;
        }
        return catalogResponse();
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    await extension(pi);
    const registeredNames: string[] = [];
    pi.registerTool = (tool) => {
      if (registeredNames.length === 1) throw new Error(refusal);
      registeredNames.push(tool.name);
      pi.tools.push(tool);
    };

    const refresh = () =>
      refreshProvider(pi.providers[0]!, {
        allowNetwork: true,
        credential: {
          type: "api_key",
          key: "sk-test",
          env: { LITELLM_BASE_URL: "https://proxy.example.com" },
        },
        signal: new AbortController().signal,
      });

    const fatalLines = () =>
      stderr.mock.calls
        .map(([message]) => String(message))
        .filter((message) => message.includes("registration stopped"));

    const firstRefresh = refresh();
    await started;
    const secondRefresh = refresh();
    await Promise.all([firstRefresh, secondRefresh]);
    releaseMcp(catalogResponse());
    await vi.waitFor(() => expect(registeredNames).toEqual([named("mcp_server_first")]));
    await vi.waitFor(() => expect(fatalLines()).toHaveLength(1));

    // A later refresh must also observe the instance-fatal state rather than retry discovery.
    await refresh();
    await new Promise<void>((resolve) => setImmediate(resolve));

    expect(listCalls).toBe(1);
    expect(pi.tools.map((tool) => tool.name).filter((name) => name.startsWith("mcp_"))).toEqual([
      named("mcp_server_first"),
    ]);
    expect(fatalLines()).toHaveLength(1);
    expect(fatalLines()[0]).toContain("registration stopped after 1 of 3 MCP tools");
    expect(fatalLines()[0]).toContain("no further attempts will be made by this extension instance");
    expect(fatalLines()[0]).toContain("…");
    expect(Buffer.byteLength(fatalLines()[0] ?? "", "utf8")).toBeLessThan(400);
    expect(fatalLines()[0]).not.toContain(proxyText);
    expect(fatalLines()[0]).not.toContain(refusal);
  });

  it.each([
    ["registered tools", [{ name: "good", server_name: "server", inputSchema: { type: "object", properties: {} } }]],
    ["an empty catalog", []],
  ])("clears a prior instance's fatal diagnostic after a successful pass with %s", async (_label, tools) => {
    process.env.LITELLM_MODELS_DEV = "0";
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/mcp-rest/tools/list")) return jsonResponse(200, { tools });
      throw new Error(`unexpected URL: ${url}`);
    });
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const extension = await loadExtension(await makeAgentDir());
    const fatalReporter = await import("../src/mcp-tools.js");
    const diagnosticLines = () => stderr.mock.calls.map(([message]) => String(message));
    const fatalLines = () => diagnosticLines().filter((message) => message.includes("registration stopped"));
    const emptyCatalogLines = () =>
      diagnosticLines().filter((message) => message.includes("no MCP tools were registered from 0 raw entries"));

    fatalReporter.reportMcpRegistrationFatal(0, 1, new Error("extension stale"));
    fatalReporter.reportMcpRegistrationFatal(0, 1, new Error("extension stale"));
    expect(fatalLines()).toHaveLength(1);

    const pi = createPi();
    await extension(pi);
    const refresh = () =>
      refreshProvider(pi.providers[0]!, {
        allowNetwork: true,
        credential: { type: "api_key", key: "sk-test", env: { LITELLM_BASE_URL: "https://proxy.example.com" } },
        signal: new AbortController().signal,
      });
    await refresh();
    if (tools.length > 0) {
      await vi.waitFor(() => expect(pi.tools.map((tool) => tool.name)).toContainEqual(named("mcp_server_good")));
    } else {
      await vi.waitFor(() => expect(emptyCatalogLines()).toHaveLength(1));
      await refresh();
      expect(emptyCatalogLines()).toHaveLength(1);
    }

    fatalReporter.reportMcpRegistrationFatal(0, 1, new Error("extension stale"));
    expect(fatalLines()).toHaveLength(2);
  });

  it("registers an alias provider's MCP tools when Pi refreshes that alias", async () => {
    process.env.LITELLM_MODELS_DEV = "0";
    const agentDir = await makeAgentDir();
    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({ litellm: { providers: { team: { baseUrl: "https://team.example.com", apiKey: "team-key" } } } }),
    );
    const listed: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/mcp-rest/tools/list")) {
        listed.push(url);
        return jsonResponse(200, {
          tools: [{ name: "good", inputSchema: { type: "object", properties: {} }, server_name: "server" }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const pi = createPi();
    await (await loadExtension(agentDir))(pi);
    const alias = pi.providers.find((provider) => provider.id === "team");

    await refreshProvider(alias!, {
      allowNetwork: true,
      credential: { type: "api_key", key: "team-key" },
      signal: new AbortController().signal,
    });

    await vi.waitFor(() => expect(pi.tools.map((tool) => tool.name)).toContainEqual(named("mcp_team_server_good")));
    expect(listed).toEqual(["https://team.example.com/mcp-rest/tools/list"]);
  });

  it("keeps an in-flight alias MCP registration when the default provider logs in", async () => {
    process.env.LITELLM_MODELS_DEV = "0";
    const agentDir = await makeAgentDir();
    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({ litellm: { providers: { team: { baseUrl: "https://team.example.com", apiKey: "team-key" } } } }),
    );
    let releaseList!: () => void;
    const listReleased = new Promise<void>((resolve) => {
      releaseList = resolve;
    });
    let listRequested = false;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
      }
      if (url === "https://team.example.com/mcp-rest/tools/list") {
        listRequested = true;
        await listReleased;
        return jsonResponse(200, {
          tools: [{ name: "good", inputSchema: { type: "object", properties: {} }, server_name: "server" }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const pi = createPi();
    await (await loadExtension(agentDir))(pi);
    const alias = pi.providers.find((provider) => provider.id === "team");

    await refreshProvider(alias!, {
      allowNetwork: true,
      credential: { type: "api_key", key: "team-key" },
      signal: new AbortController().signal,
    });
    await vi.waitFor(() => expect(listRequested).toBe(true));
    await loginOAuth(pi.providers[0]!, {
      onPrompt: async (options) => (options.placeholder ? "https://proxy.example.com" : "sk-login"),
      signal: new AbortController().signal,
    });
    releaseList();

    await vi.waitFor(() => expect(pi.tools.map((tool) => tool.name)).toContainEqual(named("mcp_team_server_good")));
  });

  it("skips re-registration for an unchanged identity after a fully successful pass", async () => {
    process.env.LITELLM_MODELS_DEV = "0";
    let listCalls = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/mcp-rest/tools/list")) {
        listCalls += 1;
        return jsonResponse(200, {
          tools: [{ name: "good", server_name: "server", inputSchema: { type: "object", properties: {} } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);

    const refresh = () =>
      refreshProvider(pi.providers[0]!, {
        allowNetwork: true,
        credential: {
          type: "api_key",
          key: "sk-test",
          env: { LITELLM_BASE_URL: "https://proxy.example.com" },
        },
        signal: new AbortController().signal,
      });

    await refresh();
    await refresh();

    expect(listCalls).toBe(1);
    expect(pi.tools.map((tool) => tool.name).filter((name) => name.startsWith("mcp_"))).toEqual([
      named("mcp_server_good"),
    ]);
  });

  it("re-registers when the credential identity changes", async () => {
    process.env.LITELLM_MODELS_DEV = "0";
    const listedHosts: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/mcp-rest/tools/list")) {
        listedHosts.push(new URL(url).host);
        return jsonResponse(200, {
          tools: [{ name: "good", server_name: "server", inputSchema: { type: "object", properties: {} } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);

    const refreshWith = (host: string) =>
      refreshProvider(pi.providers[0]!, {
        allowNetwork: true,
        credential: {
          type: "api_key",
          key: "sk-test",
          env: { LITELLM_BASE_URL: `https://${host}` },
        },
        signal: new AbortController().signal,
      });

    await refreshWith("first.example.com");
    await refreshWith("second.example.com");

    expect(listedHosts).toEqual(["first.example.com", "second.example.com"]);
  });

  it("reports an empty catalog again after a refresh that registered tools", async () => {
    process.env.LITELLM_MODELS_DEV = "0";
    let listCalls = 0;
    const catalogs: unknown[][] = [
      [],
      [{ name: "good", server_name: "server", inputSchema: { type: "object", properties: {} } }],
      [],
    ];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/mcp-rest/tools/list")) {
        const tools = catalogs[Math.min(listCalls, catalogs.length - 1)] ?? [];
        listCalls += 1;
        return jsonResponse(200, { tools });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    await extension(pi);

    const refresh = (host: string) =>
      refreshProvider(pi.providers[0]!, {
        allowNetwork: true,
        credential: { type: "api_key", key: "sk-test", env: { LITELLM_BASE_URL: `https://${host}` } },
        signal: new AbortController().signal,
      });
    const emptyCatalogLines = () =>
      stderr.mock.calls
        .map(([message]) => String(message))
        .filter((message) => message.includes("no MCP tools were registered"));

    await refresh("a.example.com");
    await vi.waitFor(() => expect(emptyCatalogLines()).toHaveLength(1));
    await refresh("b.example.com");
    await vi.waitFor(() => expect(pi.tools.map((tool) => tool.name)).toContainEqual(named("mcp_server_good")));
    await refresh("c.example.com");
    await vi.waitFor(() => expect(emptyCatalogLines()).toHaveLength(2));
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
    await vi.waitFor(() => expect(pi.tools.map((tool) => tool.name)).toContainEqual(named("mcp_brave_search")));
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
        litellmDiscoveryVersion: 3,
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
      auth: { apiKey: "sk-local", baseUrl: "http://host.docker.internal" },
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
      {
        model: {
          provider: "litellm-anthropic",
          id: "kimi-k2.6",
          api: "openai-completions",
          litellmPolicy: {
            normalizeStrictToolMessages: true,
            normalizeThinkTags: true,
            suppressReasoningVisibility: false,
          },
        },
      },
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
          litellmPolicy: {
            normalizeStrictToolMessages: true,
            normalizeThinkTags: true,
            suppressReasoningVisibility: true,
          },
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

  it("does not infer reasoning controls from configured provider alias route text", async () => {
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
      { payload: { model: "kimi-k2.6" } },
      { model: { provider: "litellm-anthropic", id: "kimi-k2.6", api: "openai-completions" } },
    );

    expect(result).toBeUndefined();
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
      litellmMcpSession: expect.stringMatching(/^[a-f0-9]{32}$/),
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
        // Pinned to the injected-context base URL, not the process-env one.
        baseUrl: "https://context.example.com",
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

  it.each(["https://litellm.example.com:8443", "https://litellm.example.com./v1"])(
    "rejects OAuth placeholder base URL %s",
    async (baseUrl) => {
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
          baseUrl,
        }),
      ).rejects.toThrow(/placeholder LiteLLM base URL/);
    },
  );

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

  async function ssoAgentDir(): Promise<string> {
    const agentDir = await makeAgentDir();
    await writeFile(
      join(agentDir, "auth.json"),
      JSON.stringify({
        litellm: {
          type: "oauth",
          access: "sk-sso",
          refresh: "",
          expires: Number.MAX_SAFE_INTEGER,
          baseUrl: "https://oauth.example.com",
        },
      }),
      "utf8",
    );
    return agentDir;
  }

  it("keeps the OAuth base URL when Pi re-resolves auth with the session's own token", async () => {
    delete process.env.LITELLM_BASE_URL;
    delete process.env.LITELLM_API_KEY;
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const extension = await loadExtension(await ssoAgentDir());
    const pi = createPi();
    await extension(pi);
    const provider = pi.providers[0]!;

    // No toAuth() first: the root has to come from auth.json, not from in-memory state that a
    // later api-key resolve would clear. Compaction hands the token it just resolved back as an
    // apiKey override, which sends Pi down the api-key path with no base URL of its own.
    await expect(resolveApiKey(provider, { type: "api_key", key: "sk-sso" })).resolves.toMatchObject({
      auth: { apiKey: "sk-sso", baseUrl: "https://oauth.example.com" },
      env: { LITELLM_BASE_URL: "https://oauth.example.com" },
    });

    // The exported env carries the root, so the request stands on its own.
    expect(() =>
      provider.stream(
        {
          id: "sso-model",
          name: "SSO model",
          provider: "litellm",
          api: "openai-completions",
          baseUrl: "https://oauth.example.com/v1",
          reasoning: false,
          input: ["text"],
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
          contextWindow: 4096,
          maxTokens: 1024,
        },
        { messages: [] },
        { apiKey: "sk-sso", env: { LITELLM_BASE_URL: "https://oauth.example.com" } },
      ),
    ).not.toThrow();
  });

  it("keeps this process's root after another process replaces the stored credential", async () => {
    delete process.env.LITELLM_BASE_URL;
    delete process.env.LITELLM_API_KEY;
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const agentDir = await ssoAgentDir();
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const provider = pi.providers[0]!;

    // This process resolved its own OAuth credential...
    await provider.auth.oauth?.toAuth({
      type: "oauth" as const,
      access: "sk-live",
      refresh: "",
      expires: Number.MAX_SAFE_INTEGER,
      baseUrl: "https://live.example.com",
    });
    // ...and then another Pi process logged in, replacing the shared auth.json token. This
    // process keeps its own login, so its root must still resolve for its own token.
    await writeFile(
      join(agentDir, "auth.json"),
      JSON.stringify({
        litellm: {
          type: "oauth",
          access: "sk-other",
          refresh: "",
          expires: Number.MAX_SAFE_INTEGER,
          baseUrl: "https://other.example.com",
        },
      }),
      "utf8",
    );

    await expect(resolveApiKey(provider, { type: "api_key", key: "sk-live" })).resolves.toMatchObject({
      auth: { apiKey: "sk-live", baseUrl: "https://live.example.com" },
      env: { LITELLM_BASE_URL: "https://live.example.com" },
    });
  });

  it.each([
    ["placeholder", "https://litellm.example.com", /placeholder LiteLLM base URL/],
    ["insecure", "http://insecure.example.com", /http/i],
  ])("still rejects a %s base URL when the session's own token is re-resolved", async (_label, baseUrl, expected) => {
    delete process.env.LITELLM_API_KEY;
    process.env.LITELLM_BASE_URL = baseUrl;
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const extension = await loadExtension(await ssoAgentDir());
    const pi = createPi();
    await extension(pi);
    const provider = pi.providers[0]!;

    // A configured base URL must fail on its own terms — never fall back to the stored OAuth
    // root, which would silently reroute the request past the guard that rejected it.
    await expect(resolveApiKey(provider, { type: "api_key", key: "sk-sso" })).rejects.toThrow(expected);
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

  describe("PKCE refresh", () => {
    const pkceCredential = () => ({
      type: "oauth" as const,
      litellmMcpSession: "existing-mcp-session",
      access: "access-old",
      refresh: "refresh-old",
      expires: Date.now() + 60_000,
      flow: "litellm_cli_pkce",
      baseUrl: "https://proxy.example.com",
      clientId: "llm_dcrc_client",
      tokenEndpoint: "https://proxy.example.com/token",
      resource: "https://proxy.example.com",
    });

    it("rotates PKCE refresh credentials", async () => {
      const now = 1_800_000_000_000;
      process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
      process.env.LITELLM_HEADERS = '{"x-tenant":"tenant-a"}';
      const extension = await loadExtension(await makeAgentDir());
      const pi = createPi();
      await extension(pi);
      vi.spyOn(Date, "now").mockReturnValue(now);
      const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
        jsonResponse(200, {
          access_token: "access-new",
          refresh_token: "refresh-new",
          token_type: "bearer",
          expires_in: 3600,
        }),
      );
      const credential = pkceCredential();

      await expect(pi.providers[0]?.auth.oauth?.refresh(credential, TEST_SIGNAL)).resolves.toEqual({
        ...credential,
        access: "access-new",
        refresh: "refresh-new",
        expires: now + 3_600_000,
      });
      expect(fetchMock).toHaveBeenCalledOnce();
      const [, init] = fetchMock.mock.calls[0]!;
      expect(Object.fromEntries(new URLSearchParams(String(init?.body)))).toEqual({
        grant_type: "refresh_token",
        refresh_token: "refresh-old",
        client_id: "llm_dcrc_client",
        resource: "https://proxy.example.com",
      });
      expect(init?.redirect).toBe("manual");
      expect(new Headers(init?.headers).get("x-tenant")).toBe("tenant-a");
    });

    it("keeps the existing refresh token when the server does not rotate it", async () => {
      const now = 1_800_000_000_000;
      process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
      const extension = await loadExtension(await makeAgentDir());
      const pi = createPi();
      await extension(pi);
      vi.spyOn(Date, "now").mockReturnValue(now);
      vi.spyOn(globalThis, "fetch").mockResolvedValue(
        jsonResponse(200, {
          access_token: "access-new",
          token_type: "bearer",
          expires_in: 3600,
        }),
      );
      const credential = pkceCredential();

      await expect(pi.providers[0]?.auth.oauth?.refresh(credential, TEST_SIGNAL)).resolves.toEqual({
        ...credential,
        access: "access-new",
        refresh: "refresh-old",
        expires: now + 3_600_000,
      });
    });

    it.each([429, 500, 503, "network", "body"])("keeps fresh PKCE credentials after %s", async (failure) => {
      const now = 1_800_000_000_000;
      process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
      const extension = await loadExtension(await makeAgentDir());
      const pi = createPi();
      await extension(pi);
      vi.spyOn(Date, "now").mockReturnValue(now);
      vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
        if (failure === "network") throw new TypeError("network unavailable");
        if (failure === "body") {
          return new Response(
            new ReadableStream({
              start(controller) {
                controller.error(new TypeError("connection closed"));
              },
            }),
          );
        }
        return jsonResponse(failure as number, {});
      });
      const credential = pkceCredential();

      await expect(pi.providers[0]?.auth.oauth?.refresh(credential, TEST_SIGNAL)).resolves.toEqual(credential);
    });

    it("keeps fresh PKCE credentials when the internal refresh deadline elapses", async () => {
      process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
      const extension = await loadExtension(await makeAgentDir());
      const pi = createPi();
      await extension(pi);
      vi.spyOn(AbortSignal, "timeout").mockImplementation(() =>
        AbortSignal.abort(new DOMException("timed out", "TimeoutError")),
      );
      vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, init) => {
        init?.signal?.throwIfAborted();
        return jsonResponse(200, {});
      });
      const credential = pkceCredential();

      await expect(pi.providers[0]?.auth.oauth?.refresh(credential, undefined as never)).resolves.toEqual(credential);
    });

    it("scopes the transient PKCE refresh backoff to one credential", async () => {
      process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
      const extension = await loadExtension(await makeAgentDir());
      const pi = createPi();
      await extension(pi);
      const fetchMock = vi
        .spyOn(globalThis, "fetch")
        .mockResolvedValueOnce(jsonResponse(503, {}))
        .mockResolvedValueOnce(
          jsonResponse(200, {
            access_token: "access-new",
            refresh_token: "refresh-new",
            token_type: "Bearer",
            expires_in: 3600,
          }),
        );
      const refresh = pi.providers[0]!.auth.oauth!.refresh;
      const failing = pkceCredential();
      const other = { ...pkceCredential(), clientId: "llm_dcrc_other", refresh: "refresh-other" };

      await expect(refresh(failing, TEST_SIGNAL)).resolves.toEqual(failing);
      await expect(refresh(other, TEST_SIGNAL)).resolves.toMatchObject({
        access: "access-new",
        refresh: "refresh-new",
      });
      expect(fetchMock).toHaveBeenCalledTimes(2);
    });

    it("rejects a transient PKCE refresh failure after expiry", async () => {
      const now = 1_800_000_000_000;
      process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
      const extension = await loadExtension(await makeAgentDir());
      const pi = createPi();
      await extension(pi);
      vi.spyOn(Date, "now").mockReturnValue(now);
      vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(503, {}));
      const credential = { ...pkceCredential(), expires: now };

      await expect(pi.providers[0]?.auth.oauth?.refresh(credential, TEST_SIGNAL)).rejects.toThrow(
        "LiteLLM token exchange failed (HTTP 503); run /login litellm again",
      );
    });

    it("sanitizes PKCE refresh OAuth errors", async () => {
      process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
      const extension = await loadExtension(await makeAgentDir());
      const pi = createPi();
      await extension(pi);
      vi.spyOn(globalThis, "fetch").mockResolvedValue(
        jsonResponse(400, {
          error: "invalid_grant",
          error_description: "refresh-old\naccess-old\u001b[31m",
          token: "unrelated-secret",
        }),
      );

      await expect(pi.providers[0]?.auth.oauth?.refresh(pkceCredential(), TEST_SIGNAL)).rejects.toThrow(
        "LiteLLM token exchange rejected (invalid_grant); run /login litellm again",
      );
    });

    it.each([
      { tokenEndpoint: "https://attacker.example.com/token" },
      { resource: "https://attacker.example.com" },
      { tokenEndpoint: "https://secret@proxy.example.com/token" },
      { tokenEndpoint: "https://proxy.example.com/token#fragment" },
      { tokenEndpoint: "blob:https://proxy.example.com/token" },
      { baseUrl: "https://secret@proxy.example.com" },
      { baseUrl: "http://proxy.example.com" },
      { clientId: "" },
      { refresh: "" },
    ])("rejects malformed PKCE refresh metadata before fetching: %j", async (override) => {
      process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
      const extension = await loadExtension(await makeAgentDir());
      const pi = createPi();
      await extension(pi);
      const fetchMock = vi.spyOn(globalThis, "fetch");

      await expect(
        pi.providers[0]?.auth.oauth?.refresh({ ...pkceCredential(), ...override }, TEST_SIGNAL),
      ).rejects.toThrow();
      expect(fetchMock).not.toHaveBeenCalled();
    });

    it("preserves cancellation while reading the PKCE token response", async () => {
      process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
      const extension = await loadExtension(await makeAgentDir());
      const pi = createPi();
      await extension(pi);
      const controller = new AbortController();
      const reason = new Error("cancelled token body");
      vi.spyOn(globalThis, "fetch").mockResolvedValue(
        Object.assign(jsonResponse(200, {}), {
          json: async () => {
            controller.abort(reason);
            throw reason;
          },
        }),
      );
      await expect(pi.providers[0]?.auth.oauth?.refresh(pkceCredential(), controller.signal)).rejects.toBe(reason);
    });

    it.each([false, true])("uses Pi's credential lock and persistence (transient: %s)", async (transient) => {
      process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
      const extension = await loadExtension(await makeAgentDir());
      const pi = createPi();
      await extension(pi);
      const stored = pkceCredential();
      const credentials = new InMemoryCredentialStore();
      await credentials.modify("litellm", async () => stored);
      const models = createModels({
        credentials,
        modelsStore: new InMemoryModelsStore(),
        authContext: { env: async () => undefined, fileExists: async () => false },
      });
      models.setProvider(pi.providers[0]!);
      const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
        transient
          ? jsonResponse(503, {})
          : jsonResponse(200, {
              access_token: "access-new",
              refresh_token: "refresh-new",
              token_type: "Bearer",
              expires_in: 3600,
            }),
      );
      const results = await Promise.all([models.getAuth("litellm"), models.getAuth("litellm")]);
      // A transient failure backs off instead of letting the second, lock-serialized
      // caller immediately retry against the still-failing token endpoint.
      expect(fetchMock).toHaveBeenCalledTimes(1);
      const access = transient ? "access-old" : "access-new";
      expect(results.map((result) => result?.auth.apiKey)).toEqual([access, access]);
      expect(await credentials.read("litellm")).toMatchObject({
        ...stored,
        access,
        refresh: transient ? "refresh-old" : "refresh-new",
        expires: expect.any(Number),
      });
    });
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
      baseUrl: "https://proxy.example.com",
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

  it("completes native PKCE login", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const requests: Array<{ url: string; method: string; redirect?: RequestInit["redirect"]; body?: string }> = [];
    let registrationBody: Record<string, unknown> | undefined;
    let authorizationUrl: URL | undefined;
    let tokenForm: URLSearchParams | undefined;
    let discoveryRequest: { redirect?: RequestInit["redirect"] } | undefined;
    let tokenRequest: { redirect?: RequestInit["redirect"] } | undefined;
    let callbackResponse: Promise<Response> | undefined;
    const nativeFetch = globalThis.fetch;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      const method = String(init?.method ?? "GET");
      if (url.startsWith("http://127.0.0.1:")) return nativeFetch(input, init);
      requests.push({
        url,
        method,
        redirect: init?.redirect,
        body: typeof init?.body === "string" ? init.body : undefined,
      });
      if (url.endsWith("/.well-known/litellm-cli-auth")) {
        discoveryRequest = { redirect: init?.redirect };
        return jsonResponse(200, {
          contract_version: 1,
          issuer: "https://litellm.example.com",
          authorization_endpoint: "https://litellm.example.com/authorize?tenant=alpha&client_id=wrong",
          token_endpoint: "https://litellm.example.com/token",
          registration_endpoint: "https://litellm.example.com/register",
          resource: "https://litellm.example.com",
          code_challenge_methods_supported: ["S256"],
        });
      }
      if (url.endsWith("/register")) {
        registrationBody = JSON.parse(String(init?.body)) as Record<string, unknown>;
        return jsonResponse(201, {
          client_id: "llm_dcrc_client",
          redirect_uris: registrationBody.redirect_uris,
        });
      }
      if (url.endsWith("/token")) {
        tokenRequest = { redirect: init?.redirect };
        tokenForm = new URLSearchParams(String(init?.body));
        return jsonResponse(200, {
          access_token: "access-new",
          refresh_token: "refresh-new",
          token_type: "Bearer",
          expires_in: 3600,
          user_id: "user@example.com",
          team_id: "team-a",
        });
      }
      throw new Error(`unexpected URL: ${url} (${method})`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const startedAt = Date.now();
    const credential = await loginOAuth(pi.providers[0]!, {
      onPrompt: async (prompt) => (prompt.placeholder ? "https://litellm.example.com" : ""),
      onAuth: ({ url }) => {
        authorizationUrl = new URL(url);
        callbackResponse = fetch(
          `${new URL(url).searchParams.get("redirect_uri")}?code=authorization-code&state=${new URL(url).searchParams.get("state")}`,
        );
      },
      pkce: true,
      signal: new AbortController().signal,
    });
    const endedAt = Date.now();

    expect(discoveryRequest?.redirect).toBe("manual");
    expect(registrationBody).toMatchObject({
      client_name: "pi-provider-litellm",
      redirect_uris: [expect.stringMatching(/^http:\/\/127\.0\.0\.1:\d+\/callback$/)],
      token_endpoint_auth_method: "none",
      grant_types: ["authorization_code", "refresh_token"],
      response_types: ["code"],
    });
    expect(authorizationUrl?.searchParams.get("resource")).toBe("https://litellm.example.com");
    expect(authorizationUrl?.searchParams.get("tenant")).toBe("alpha");
    expect(authorizationUrl?.searchParams.getAll("client_id")).toEqual(["llm_dcrc_client"]);
    const verifier = tokenForm?.get("code_verifier");
    expect(verifier).toEqual(expect.any(String));
    expect(
      createHash("sha256")
        .update(verifier ?? "")
        .digest("base64url"),
    ).toBe(authorizationUrl?.searchParams.get("code_challenge"));
    const registeredRedirectUris = Array.isArray(registrationBody?.redirect_uris) ? registrationBody.redirect_uris : [];
    expect(tokenForm?.get("redirect_uri")).toBe(registeredRedirectUris[0]);
    expect(tokenRequest?.redirect).toBe("manual");
    expect((await callbackResponse)?.status).toBe(200);
    expect((await callbackResponse)?.headers.get("cache-control")).toBe("no-store");
    expect(await (await callbackResponse)?.text()).toContain("close this window");
    expect(credential).toMatchObject({
      type: "oauth",
      access: "access-new",
      refresh: "refresh-new",
      baseUrl: "https://litellm.example.com",
      flow: "litellm_cli_pkce",
      clientId: "llm_dcrc_client",
      tokenEndpoint: "https://litellm.example.com/token",
      resource: "https://litellm.example.com",
      userId: "user@example.com",
      teamId: "team-a",
    });
    expect(credential?.expires).toBeGreaterThanOrEqual(startedAt + 3_600_000);
    expect(credential?.expires).toBeLessThanOrEqual(endedAt + 3_600_000);
    expect(requests.some((request) => request.url.endsWith("/sso/cli/start"))).toBe(false);
  }, 15_000);

  it.each([
    ["an unsupported contract version", { contract_version: 2 }],
    ["no S256 support", { code_challenge_methods_supported: ["plain"] }],
    ["a different issuer", { issuer: "https://other.example.com" }],
    ["issuer credentials", { issuer: "https://secret@litellm.example.com" }],
    ["an issuer query", { issuer: "https://litellm.example.com?tenant=other" }],
    ["token endpoint credentials", { token_endpoint: "https://secret@litellm.example.com/token" }],
    ["a token endpoint fragment", { token_endpoint: "https://litellm.example.com/token#other" }],
    ["a cross-origin authorization endpoint", { authorization_endpoint: "https://other.example.com/authorize" }],
    ["a cross-origin token endpoint", { token_endpoint: "https://other.example.com/token" }],
    ["a cross-origin registration endpoint", { registration_endpoint: "https://other.example.com/register" }],
    ["a cross-origin resource", { resource: "https://other.example.com" }],
  ])("rejects native PKCE discovery with %s", async (_name, override) => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      if (String(input).endsWith("/.well-known/litellm-cli-auth"))
        return jsonResponse(200, {
          contract_version: 1,
          issuer: "https://litellm.example.com",
          authorization_endpoint: "https://litellm.example.com/authorize",
          token_endpoint: "https://litellm.example.com/token",
          registration_endpoint: "https://litellm.example.com/register",
          resource: "https://litellm.example.com",
          code_challenge_methods_supported: ["S256"],
          ...override,
        });
      throw new Error(`unexpected URL: ${String(input)}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await expect(
      loginOAuth(pi.providers[0]!, {
        onPrompt: async (prompt) => (prompt.placeholder ? "https://litellm.example.com" : ""),
        pkce: true,
        signal: new AbortController().signal,
      }),
    ).rejects.toThrow();
    expect(fetchMock.mock.calls.some(([input]) => String(input).endsWith("/register"))).toBe(false);
  });

  it.each([
    "https://secret@proxy.example.com",
    "https://proxy.example.com?tenant=other",
    "https://proxy.example.com#fragment",
  ])("rejects an invalid PKCE proxy URL before discovery: %s", async (baseUrl) => {
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(404, {}));
    await expect(
      loginOAuth(pi.providers[0]!, {
        onPrompt: async (prompt) => (prompt.placeholder ? baseUrl : ""),
        pkce: true,
      }),
    ).rejects.toThrow();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it.each([302, 200])("rejects invalid PKCE discovery responses (HTTP %s)", async (status) => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("untrusted-secret", {
        status,
        headers: { location: "https://other.example.com/.well-known/litellm-cli-auth" },
      }),
    );
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await expect(
      loginOAuth(pi.providers[0]!, {
        onPrompt: async (prompt) => (prompt.placeholder ? "https://litellm.example.com" : ""),
        pkce: true,
        signal: new AbortController().signal,
      }),
    ).rejects.toThrow(
      status === 302
        ? "LiteLLM CLI auth discovery failed (HTTP 302)"
        : "LiteLLM CLI auth discovery returned invalid JSON",
    );
    expect(fetchMock.mock.calls[0]?.[1]?.redirect).toBe("manual");
  });

  it.each(["non-GET", "malformed URL"])("rejects %s native PKCE callbacks", async (kind) => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const nativeFetch = globalThis.fetch;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.startsWith("http://127.0.0.1:")) return nativeFetch(input, init);
      if (url.endsWith("/.well-known/litellm-cli-auth"))
        return jsonResponse(200, {
          contract_version: 1,
          issuer: "https://litellm.example.com",
          authorization_endpoint: "https://litellm.example.com/authorize",
          token_endpoint: "https://litellm.example.com/token",
          registration_endpoint: "https://litellm.example.com/register",
          resource: "https://litellm.example.com",
          code_challenge_methods_supported: ["S256"],
        });
      if (url.endsWith("/register"))
        return jsonResponse(200, {
          client_id: "llm_dcrc_client",
          redirect_uris: JSON.parse(String(init?.body)).redirect_uris,
        });
      if (url.endsWith("/token"))
        return jsonResponse(200, {
          access_token: "access-new",
          refresh_token: "refresh-new",
          token_type: "Bearer",
          expires_in: 3600,
        });
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const controller = new AbortController();
    const reason = new Error("caller cancelled login");
    let callbackResponse: Promise<{ status: number }> | undefined;
    const login = loginOAuth(pi.providers[0]!, {
      onPrompt: async (prompt) => (prompt.placeholder ? "https://litellm.example.com" : ""),
      onAuth: ({ url }) => {
        const authorizationUrl = new URL(url);
        const redirectUri = new URL(authorizationUrl.searchParams.get("redirect_uri")!);
        callbackResponse =
          kind === "non-GET"
            ? fetch(`${redirectUri}?state=${authorizationUrl.searchParams.get("state")}&code=authorization-code`, {
                method: "POST",
              })
            : new Promise((resolve, reject) => {
                const request = httpRequest(
                  { hostname: redirectUri.hostname, port: redirectUri.port, path: "http://[" },
                  (response) => {
                    response.resume();
                    resolve({ status: response.statusCode! });
                  },
                );
                request.on("error", reject);
                request.setTimeout(1000, () => request.destroy(new Error("Callback did not reject malformed URL")));
                request.end();
              });
      },
      pkce: true,
      signal: controller.signal,
    });
    await vi.waitFor(() => expect(callbackResponse).toBeDefined());
    expect((await callbackResponse)?.status).toBe(kind === "non-GET" ? 405 : 400);
    controller.abort(reason);
    await expect(login).rejects.toBe(reason);
  });

  it("surfaces an OAuth denial from the native PKCE callback", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const nativeFetch = globalThis.fetch;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.startsWith("http://127.0.0.1:")) return nativeFetch(input, init);
      if (url.endsWith("/.well-known/litellm-cli-auth"))
        return jsonResponse(200, {
          contract_version: 1,
          issuer: "https://litellm.example.com",
          authorization_endpoint: "https://litellm.example.com/authorize",
          token_endpoint: "https://litellm.example.com/token",
          registration_endpoint: "https://litellm.example.com/register",
          resource: "https://litellm.example.com",
          code_challenge_methods_supported: ["S256"],
        });
      if (url.endsWith("/register"))
        return jsonResponse(200, {
          client_id: "llm_dcrc_client",
          redirect_uris: JSON.parse(String(init?.body)).redirect_uris,
        });
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await expect(
      loginOAuth(pi.providers[0]!, {
        onPrompt: async (prompt) => (prompt.placeholder ? "https://litellm.example.com" : ""),
        onAuth: ({ url }) => {
          const authorizationUrl = new URL(url);
          void fetch(
            `${authorizationUrl.searchParams.get("redirect_uri")}?state=${authorizationUrl.searchParams.get("state")}&error=access_denied&error_description=User%20declined`,
          );
        },
        pkce: true,
        signal: new AbortController().signal,
      }),
    ).rejects.toThrow("LiteLLM PKCE login was denied");
  });

  it.each([
    ["registration", "redirect"],
    ["token", "redirect"],
    ["registration", "invalid JSON"],
  ])("rejects native PKCE %s %s", async (stage, failure) => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const nativeFetch = globalThis.fetch;
    const redirects: NonNullable<RequestInit["redirect"]>[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.startsWith("http://127.0.0.1:")) return nativeFetch(input, init);
      if (url.endsWith("/.well-known/litellm-cli-auth"))
        return jsonResponse(200, {
          contract_version: 1,
          issuer: "https://litellm.example.com",
          authorization_endpoint: "https://litellm.example.com/authorize",
          token_endpoint: "https://litellm.example.com/token",
          registration_endpoint: "https://litellm.example.com/register",
          resource: "https://litellm.example.com",
          code_challenge_methods_supported: ["S256"],
        });
      if (url.endsWith("/register")) {
        redirects.push(init?.redirect ?? "follow");
        if (stage === "registration")
          return new Response("untrusted-secret", {
            status: failure === "redirect" ? 302 : 200,
            headers: { location: "/register-next" },
          });
        return jsonResponse(200, {
          client_id: "llm_dcrc_client",
          redirect_uris: JSON.parse(String(init?.body)).redirect_uris,
        });
      }
      if (url.endsWith("/token")) {
        redirects.push(init?.redirect ?? "follow");
        return new Response(null, { status: 302, headers: { location: "/token-next" } });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await expect(
      loginOAuth(pi.providers[0]!, {
        onPrompt: async (prompt) => (prompt.placeholder ? "https://litellm.example.com" : ""),
        onAuth: ({ url }) => {
          const authorizationUrl = new URL(url);
          void fetch(
            `${authorizationUrl.searchParams.get("redirect_uri")}?state=${authorizationUrl.searchParams.get("state")}&code=authorization-code`,
          );
        },
        pkce: true,
        signal: new AbortController().signal,
      }),
    ).rejects.toThrow(
      failure === "invalid JSON"
        ? "LiteLLM CLI auth registration returned invalid JSON"
        : stage === "registration"
          ? "client registration failed (HTTP 302)"
          : "token exchange failed (HTTP 302)",
    );
    expect(redirects).toEqual(stage === "registration" ? ["manual"] : ["manual", "manual"]);
  });

  it("reports a transient native PKCE token network failure", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const nativeFetch = globalThis.fetch;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.startsWith("http://127.0.0.1:")) return nativeFetch(input, init);
      if (url.endsWith("/.well-known/litellm-cli-auth"))
        return jsonResponse(200, {
          contract_version: 1,
          issuer: "https://litellm.example.com",
          authorization_endpoint: "https://litellm.example.com/authorize",
          token_endpoint: "https://litellm.example.com/token",
          registration_endpoint: "https://litellm.example.com/register",
          resource: "https://litellm.example.com",
          code_challenge_methods_supported: ["S256"],
        });
      if (url.endsWith("/register"))
        return jsonResponse(200, {
          client_id: "llm_dcrc_client",
          redirect_uris: JSON.parse(String(init?.body)).redirect_uris,
        });
      if (url.endsWith("/token")) throw new Error("network offline");
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await expect(
      loginOAuth(pi.providers[0]!, {
        onPrompt: async (prompt) => (prompt.placeholder ? "https://litellm.example.com" : ""),
        onAuth: ({ url }) => {
          const authorizationUrl = new URL(url);
          void fetch(
            `${authorizationUrl.searchParams.get("redirect_uri")}?state=${authorizationUrl.searchParams.get("state")}&code=authorization-code`,
          );
        },
        pkce: true,
        signal: new AbortController().signal,
      }),
    ).rejects.toThrow("LiteLLM token exchange failed (network error)");
  });

  it.each<[string, number, unknown, string]>([
    ["a malformed response", 200, {}, "LiteLLM token exchange returned an invalid response"],
    ...[
      { access_token: " " },
      { refresh_token: "refresh\nsecret" },
      { expires_in: 1e308 },
      { expires_in: { toString: null } },
    ].map((override): [string, number, unknown, string] => [
      `an invalid ${Object.keys(override)[0]}`,
      200,
      { access_token: "access", refresh_token: "refresh", token_type: "Bearer", expires_in: 3600, ...override },
      "LiteLLM token exchange returned an invalid response",
    ]),
    [
      "an OAuth error",
      400,
      { error: "invalid_grant", error_description: "authorization-code\nsecret\u001b[31m" },
      "LiteLLM token exchange rejected (invalid_grant)",
    ],
    ["a 429 response", 429, {}, "LiteLLM token exchange failed (HTTP 429)"],
    ["a 500 response", 500, {}, "LiteLLM token exchange failed (HTTP 500)"],
  ])("surfaces native PKCE token %s", async (_name, status, body, message) => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const nativeFetch = globalThis.fetch;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.startsWith("http://127.0.0.1:")) return nativeFetch(input, init);
      if (url.endsWith("/.well-known/litellm-cli-auth"))
        return jsonResponse(200, {
          contract_version: 1,
          issuer: "https://litellm.example.com",
          authorization_endpoint: "https://litellm.example.com/authorize",
          token_endpoint: "https://litellm.example.com/token",
          registration_endpoint: "https://litellm.example.com/register",
          resource: "https://litellm.example.com",
          code_challenge_methods_supported: ["S256"],
        });
      if (url.endsWith("/register"))
        return jsonResponse(200, {
          client_id: "llm_dcrc_client",
          redirect_uris: JSON.parse(String(init?.body)).redirect_uris,
        });
      if (url.endsWith("/token")) return jsonResponse(status, body);
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await expect(
      loginOAuth(pi.providers[0]!, {
        onPrompt: async (prompt) => (prompt.placeholder ? "https://litellm.example.com" : ""),
        onAuth: ({ url }) => {
          const authorizationUrl = new URL(url);
          void fetch(
            `${authorizationUrl.searchParams.get("redirect_uri")}?state=${authorizationUrl.searchParams.get("state")}&code=authorization-code`,
          );
        },
        pkce: true,
        signal: new AbortController().signal,
      }),
    ).rejects.toThrow(message);
  });

  it("keeps waiting after a native PKCE callback with the wrong state", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const nativeFetch = globalThis.fetch;
    let wrongStateResponse: Promise<Response> | undefined;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.startsWith("http://127.0.0.1:")) return nativeFetch(input, init);
      if (url.endsWith("/.well-known/litellm-cli-auth"))
        return jsonResponse(200, {
          contract_version: 1,
          issuer: "https://litellm.example.com",
          authorization_endpoint: "https://litellm.example.com/authorize",
          token_endpoint: "https://litellm.example.com/token",
          registration_endpoint: "https://litellm.example.com/register",
          resource: "https://litellm.example.com",
          code_challenge_methods_supported: ["S256"],
        });
      if (url.endsWith("/register"))
        return jsonResponse(200, {
          client_id: "llm_dcrc_client",
          redirect_uris: JSON.parse(String(init?.body)).redirect_uris,
        });
      if (url.endsWith("/token"))
        return jsonResponse(200, {
          access_token: "access-new",
          refresh_token: "refresh-new",
          token_type: "Bearer",
          expires_in: 3600,
        });
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const credential = await loginOAuth(pi.providers[0]!, {
      onPrompt: async (prompt) => (prompt.placeholder ? "https://litellm.example.com" : ""),
      onAuth: ({ url }) => {
        const authorizationUrl = new URL(url);
        const redirectUri = authorizationUrl.searchParams.get("redirect_uri");
        wrongStateResponse = fetch(`${redirectUri}?state=wrong&code=wrong-code`);
        void fetch(`${redirectUri}?state=${authorizationUrl.searchParams.get("state")}&code=authorization-code`);
      },
      pkce: true,
      signal: new AbortController().signal,
    });
    expect((await wrongStateResponse)?.status).toBe(400);
    expect(credential).toMatchObject({ access: "access-new" });
  });

  it("closes the native PKCE callback listener when login is aborted", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith("/.well-known/litellm-cli-auth"))
        return jsonResponse(200, {
          contract_version: 1,
          issuer: "https://litellm.example.com",
          authorization_endpoint: "https://litellm.example.com/authorize",
          token_endpoint: "https://litellm.example.com/token",
          registration_endpoint: "https://litellm.example.com/register",
          resource: "https://litellm.example.com",
          code_challenge_methods_supported: ["S256"],
        });
      if (url.endsWith("/register"))
        return jsonResponse(200, {
          client_id: "llm_dcrc_client",
          redirect_uris: JSON.parse(String(init?.body)).redirect_uris,
        });
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const controller = new AbortController();
    const reason = new Error("caller cancelled login");
    let callbackUrl: string | undefined;
    const login = loginOAuth(pi.providers[0]!, {
      onPrompt: async (prompt) => (prompt.placeholder ? "https://litellm.example.com" : ""),
      onAuth: ({ url }) => {
        callbackUrl = new URL(url).searchParams.get("redirect_uri")!;
      },
      pkce: true,
      signal: controller.signal,
    });
    await vi.waitFor(() => expect(callbackUrl).toBeDefined());
    controller.abort(reason);
    await expect(login).rejects.toBe(reason);
    await expect(fetch(`${callbackUrl}?code=authorization-code`)).rejects.toThrow();
  });

  it.each([
    ["success", undefined],
    ["OAuth denial", "LiteLLM PKCE login was denied"],
    ["registration failure", "LiteLLM PKCE client registration failed (HTTP 500)"],
    ["token failure", "LiteLLM token exchange failed (HTTP 500)"],
  ])("closes the native PKCE callback listener after %s", async (outcome, expectedError) => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const nativeFetch = globalThis.fetch;
    let callbackUrl: string | undefined;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.startsWith("http://127.0.0.1:")) return nativeFetch(input, init);
      if (url.endsWith("/.well-known/litellm-cli-auth"))
        return jsonResponse(200, {
          contract_version: 1,
          issuer: "https://litellm.example.com",
          authorization_endpoint: "https://litellm.example.com/authorize",
          token_endpoint: "https://litellm.example.com/token",
          registration_endpoint: "https://litellm.example.com/register",
          resource: "https://litellm.example.com",
          code_challenge_methods_supported: ["S256"],
        });
      if (url.endsWith("/register")) {
        callbackUrl = JSON.parse(String(init?.body)).redirect_uris[0];
        if (outcome === "registration failure") return jsonResponse(500, {});
        return jsonResponse(200, { client_id: "llm_dcrc_client", redirect_uris: [callbackUrl] });
      }
      if (url.endsWith("/token")) {
        if (outcome === "token failure") return jsonResponse(500, {});
        return jsonResponse(200, {
          access_token: "access-new",
          refresh_token: "refresh-new",
          token_type: "Bearer",
          expires_in: 3600,
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const login = loginOAuth(pi.providers[0]!, {
      onPrompt: async (prompt) => (prompt.placeholder ? "https://litellm.example.com" : ""),
      onAuth: ({ url }) => {
        const authorizationUrl = new URL(url);
        const response =
          outcome === "OAuth denial"
            ? "error=access_denied&error_description=User%20declined"
            : "code=authorization-code";
        void fetch(
          `${authorizationUrl.searchParams.get("redirect_uri")}?state=${authorizationUrl.searchParams.get("state")}&${response}`,
        );
      },
      pkce: true,
      signal: new AbortController().signal,
    });
    if (expectedError) await expect(login).rejects.toThrow(expectedError);
    else await expect(login).resolves.toMatchObject({ access: "access-new" });
    await expect(nativeFetch(`${callbackUrl}?code=authorization-code`)).rejects.toThrow();
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

  it("re-registers when the API key rotates on the same host", async () => {
    process.env.LITELLM_MODELS_DEV = "0";
    const listedKeys: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/mcp-rest/tools/list")) {
        listedKeys.push(String(new Headers(init?.headers).get("authorization")));
        return jsonResponse(200, {
          tools: [{ name: "good", server_name: "server", inputSchema: { type: "object", properties: {} } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const extension = await loadExtension(await makeAgentDir());
    const pi = createPi();
    await extension(pi);

    const refreshWithKey = (key: string) =>
      refreshProvider(pi.providers[0]!, {
        allowNetwork: true,
        credential: { type: "api_key", key, env: { LITELLM_BASE_URL: "https://proxy.example.com" } },
        signal: new AbortController().signal,
      });

    await refreshWithKey("sk-first");
    await refreshWithKey("sk-first");
    await refreshWithKey("sk-second");

    // Fingerprinting the credential must not cost the change detection it exists to provide:
    // the unchanged key is skipped, the rotated one triggers a fresh catalog pass.
    expect(listedKeys).toEqual(["Bearer sk-first", "Bearer sk-second"]);
  });
  it("streams OAuth Messages requests to the credential host over a conflicting environment host", async () => {
    process.env.LITELLM_BASE_URL = "https://environment.example.com";
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const requestedUrls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      requestedUrls.push(url);
      if (new URL(url).pathname !== "/v1/messages") throw new Error(`unexpected URL: ${url}`);
      return new Response(
        'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-opus-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}\n\nevent: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\nevent: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}\n\nevent: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\nevent: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}\n\nevent: message_stop\ndata: {"type":"message_stop"}\n\n',
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
      env: async (name) => process.env[name],
      fileExists: async () => false,
    };
    const models = createModels({ credentials, modelsStore: new InMemoryModelsStore(), authContext });
    models.setProvider(provider);
    const model = {
      id: "oauth-messages",
      name: "OAuth Messages",
      provider: "litellm",
      api: "anthropic-messages" as const,
      baseUrl: "https://credential.example.com",
      reasoning: false,
      input: ["text"] as ("text" | "image")[],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 4096,
      maxTokens: 1024,
      compat: {},
    };

    const result = await models.complete(model, { messages: [] });

    expect(result.stopReason).toBe("stop");
    expect(requestedUrls).toEqual(["https://credential.example.com/v1/messages?beta=true"]);
  });
});
