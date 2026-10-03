import { createHash } from "node:crypto";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { createServer, request as httpRequest, type Server } from "node:http";
import type { AddressInfo } from "node:net";
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
  normalizeContext,
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

  it("keeps Pi-managed models when an online refresh fails", async () => {
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";
    const requestedUrls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      requestedUrls.push(String(input));
      throw new Error(`unexpected URL: ${String(input)}`);
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
      litellmDiscoveryVersion: 4,
    };

    await expect(
      refreshProvider(pi.providers[0]!, {
        allowNetwork: true,
        credential: { type: "api_key", key: "sk-test", env: { LITELLM_BASE_URL: "https://proxy.example.com" } },
        store: createModelsStore([stored]),
      }),
    ).rejects.toThrow("unexpected URL");

    expect(requestedUrls).toEqual(["https://proxy.example.com/model/info"]);
    expect(pi.providers[0]?.getModels()).toEqual([stored]);
  });

  it("does not cache the credential of a model refresh that predates login", async () => {
    process.env.LITELLM_MODELS_DEV = "0";
    let release!: (response: Response) => void;
    const pending = new Promise<Response>((resolve) => {
      release = resolve;
    });
    let modelCalls = 0;
    const skillKeys: Array<string | null> = [];
    const models = () => jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return ++modelCalls === 1 ? pending : models();
      if (url.endsWith("/claude-code/marketplace.json")) {
        skillKeys.push(new Headers(init?.headers).get("Authorization"));
        return jsonResponse(200, { plugins: [] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const pi = createPi();
    await (await loadExtension(await makeAgentDir()))(pi);
    // Without a context, the Skills tools use the credential the last model refresh cached.
    const listSkills = pi.tools.find((tool) => tool.name === "litellm_skill_list")!;
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
    await expect(listSkills.execute?.("test-call", {}, TEST_SIGNAL)).rejects.toThrow("no credentials for litellm");

    await refreshProvider(pi.providers[0]!, { allowNetwork: true, credential });
    await listSkills.execute?.("test-call", {}, TEST_SIGNAL);
    expect(skillKeys).toEqual(["Bearer sk-new"]);
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
        litellmDiscoveryVersion: 4,
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
        normalizeContext({ messages: [] }),
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
        normalizeContext({ messages: [] }),
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

    it("rejects a transient PKCE refresh failure after expiry without asking for a new login", async () => {
      const now = 1_800_000_000_000;
      process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
      const extension = await loadExtension(await makeAgentDir());
      const pi = createPi();
      await extension(pi);
      vi.spyOn(Date, "now").mockReturnValue(now);
      vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(503, {}));
      const credential = { ...pkceCredential(), expires: now };

      await expect(pi.providers[0]?.auth.oauth?.refresh(credential, TEST_SIGNAL)).rejects.toThrow(
        /^LiteLLM token exchange failed \(HTTP 503\)$/,
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

describe("direct OIDC login", () => {
  const issuer = "https://idp.example.com";
  const clientId = "example-client-id";
  const discoveryUrl = `${issuer}/.well-known/openid-configuration`;
  const tokenEndpoint = `${issuer}/token`;
  const proxyUrl = "https://proxy.example.com";

  type OidcLoginOptions = {
    oidc?: unknown;
    providerSettings?: Record<string, unknown>;
    discovery?: Record<string, unknown>;
    discoveryResponse?: () => Response;
    claims?: Record<string, unknown>;
    tokenBody?: Record<string, unknown>;
    token?: () => Response;
    callbackQuery?: (state: string) => string;
  };

  function idToken(claims: Record<string, unknown>): string {
    const encode = (value: unknown): string => Buffer.from(JSON.stringify(value)).toString("base64url");
    return `${encode({ alg: "RS256", typ: "JWT" })}.${encode(claims)}.sig`;
  }

  async function runOidcLogin(options: OidcLoginOptions = {}) {
    const agentDir = await makeAgentDir();
    const oidc = "oidc" in options ? options.oidc : { issuer, clientId };
    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({ litellm: { providers: { litellm: { ...options.providerSettings, oidc } } } }),
    );
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    process.env.LITELLM_HEADERS = '{"x-gateway-secret":"gateway-secret"}';
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const requests: Array<{ url: string; method: string; redirect?: string; headers: Headers; body?: string }> = [];
    const issued: string[] = [];
    let authorizationUrl: URL | undefined;
    let callbackResponse: Promise<Response> | undefined;
    const nativeFetch = globalThis.fetch;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      requests.push({
        url,
        method: String(init?.method ?? "GET"),
        redirect: init?.redirect,
        headers: new Headers(init?.headers),
        body: init?.body === undefined ? undefined : String(init.body),
      });
      if (url === discoveryUrl)
        return (
          options.discoveryResponse?.() ??
          jsonResponse(200, {
            issuer,
            authorization_endpoint: `${issuer}/authorize?tenant=alpha&client_id=wrong`,
            token_endpoint: tokenEndpoint,
            code_challenge_methods_supported: ["S256"],
            ...options.discovery,
          })
        );
      if (url === tokenEndpoint) {
        if (options.token) return options.token();
        const token = idToken({
          iss: issuer,
          aud: clientId,
          sub: "user-123",
          exp: Math.floor(Date.now() / 1000) + 3600,
          nonce: authorizationUrl?.searchParams.get("nonce"),
          ...options.claims,
        });
        issued.push(token);
        return jsonResponse(200, {
          id_token: token,
          refresh_token: "refresh-new",
          access_token: "idp-access-token",
          token_type: "Bearer",
          expires_in: 3600,
          ...options.tokenBody,
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const login = pi.providers[0]!.auth.oauth!.login(
      interaction(
        async (prompt) => ("placeholder" in prompt && prompt.placeholder ? proxyUrl : ""),
        (event) => {
          if (event.type !== "auth_url") return;
          authorizationUrl = new URL(event.url);
          const state = authorizationUrl.searchParams.get("state")!;
          const query = options.callbackQuery?.(state) ?? `code=authorization-code&state=${state}`;
          callbackResponse = nativeFetch(`${authorizationUrl.searchParams.get("redirect_uri")}?${query}`);
        },
        new AbortController().signal,
      ),
    );
    const outcome = await login.then(
      (credential) => ({ credential, error: undefined }),
      (error: Error) => ({ credential: undefined, error }),
    );
    return { ...outcome, pi, requests, issued, authorizationUrl, callbackResponse };
  }

  async function listenOnFreePort(): Promise<Server> {
    const server = createServer();
    await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
    return server;
  }

  function portOf(server: Server): number {
    return (server.address() as AddressInfo).port;
  }

  it("signs in with the identity provider and stores its id_token", async () => {
    const exp = Math.floor(Date.now() / 1000) + 3600;
    const { credential, error, pi, requests, issued, authorizationUrl, callbackResponse } = await runOidcLogin({
      claims: { exp },
    });

    expect(error).toBeUndefined();
    // Only the IdP is contacted, without redirects: no CLI-auth discovery, /sso/* or /key/generate on the proxy.
    expect(requests.map(({ url, method, redirect }) => ({ url, method, redirect }))).toEqual([
      { url: discoveryUrl, method: "GET", redirect: "manual" },
      { url: tokenEndpoint, method: "POST", redirect: "manual" },
    ]);
    for (const { headers } of requests) {
      expect(headers.get("x-gateway-secret")).toBeNull();
      expect(headers.get("authorization")).toBeNull();
    }
    const redirectUri = authorizationUrl!.searchParams.get("redirect_uri")!;
    expect(redirectUri).toMatch(/^http:\/\/127\.0\.0\.1:\d+\/callback$/);
    expect(`${authorizationUrl!.origin}${authorizationUrl!.pathname}`).toBe(`${issuer}/authorize`);
    expect(Object.fromEntries(authorizationUrl!.searchParams)).toEqual({
      tenant: "alpha",
      response_type: "code",
      client_id: clientId,
      redirect_uri: redirectUri,
      scope: "openid",
      state: expect.stringMatching(/^[\w-]{43}$/),
      nonce: expect.stringMatching(/^[\w-]{43}$/),
      code_challenge: expect.stringMatching(/^[\w-]{43}$/),
      code_challenge_method: "S256",
    });
    expect(authorizationUrl!.searchParams.getAll("client_id")).toEqual([clientId]);
    const tokenForm = Object.fromEntries(new URLSearchParams(requests[1]!.body));
    // No client_secret: the exact form proves a public client.
    expect(tokenForm).toEqual({
      grant_type: "authorization_code",
      code: "authorization-code",
      redirect_uri: redirectUri,
      client_id: clientId,
      code_verifier: expect.stringMatching(/^[\w-]{43}$/),
    });
    expect(createHash("sha256").update(tokenForm.code_verifier!).digest("base64url")).toBe(
      authorizationUrl!.searchParams.get("code_challenge"),
    );
    expect((await callbackResponse)?.status).toBe(200);
    expect((await callbackResponse)?.headers.get("cache-control")).toBe("no-store");
    expect(credential).toEqual({
      type: "oauth",
      access: issued[0],
      refresh: "refresh-new",
      expires: exp * 1000 - 5 * 60 * 1000,
      baseUrl: proxyUrl,
      flow: "oidc_pkce",
      issuer,
      clientId,
      tokenEndpoint,
      subject: "user-123",
    });
    await expect(pi.providers[0]!.auth.oauth!.toAuth(credential!)).resolves.toMatchObject({ apiKey: issued[0] });
  }, 15_000);

  it("keeps provider headers off IdP requests", async () => {
    const { error, requests } = await runOidcLogin({
      providerSettings: { headers: { "x-provider-secret": "provider-secret" } },
    });

    expect(error).toBeUndefined();
    expect(requests).toHaveLength(2);
    for (const { headers } of requests) expect(headers.get("x-provider-secret")).toBeNull();
  });

  it("stores an empty refresh token when the IdP issues none", async () => {
    const { credential, error } = await runOidcLogin({ tokenBody: { refresh_token: undefined } });

    expect(error).toBeUndefined();
    expect(credential).toMatchObject({ flow: "oidc_pkce", refresh: "" });
  });

  it("uses the first free configured redirect port", async () => {
    const occupied = await listenOnFreePort();
    const free = await listenOnFreePort();
    const freePort = portOf(free);
    await new Promise((resolve) => free.close(resolve));
    try {
      const { error, authorizationUrl } = await runOidcLogin({
        oidc: { issuer, clientId, redirectPorts: [portOf(occupied), freePort] },
      });

      expect(error).toBeUndefined();
      expect(authorizationUrl?.searchParams.get("redirect_uri")).toBe(`http://127.0.0.1:${freePort}/callback`);
    } finally {
      await new Promise((resolve) => occupied.close(resolve));
    }
  });

  it("accepts an audience array that contains the client ID and a configured scope", async () => {
    const { error, authorizationUrl } = await runOidcLogin({
      oidc: { issuer: `${issuer}/`, clientId, scope: "openid offline_access" },
      claims: { aud: ["other-client", clientId] },
    });

    expect(error).toBeUndefined();
    expect(authorizationUrl?.searchParams.get("scope")).toBe("openid offline_access");
  });

  it.each<[string, OidcLoginOptions, string]>([
    ["a wrong nonce", { claims: { nonce: "other-nonce" } }, "OIDC id_token has invalid nonce"],
    ["a wrong issuer", { claims: { iss: "https://other.example.com" } }, "OIDC id_token has invalid iss"],
    ["a non-exact issuer", { claims: { iss: `${issuer}/` } }, "OIDC id_token has invalid iss"],
    ["an audience mismatch", { claims: { aud: "other-client" } }, "OIDC id_token has invalid aud"],
    ["an audience array mismatch", { claims: { aud: ["other-client"] } }, "OIDC id_token has invalid aud"],
    [
      "an authorized party for another client",
      { claims: { aud: [clientId, "other-client"], azp: "other-client" } },
      "OIDC id_token has invalid azp",
    ],
    ["an expired token", { claims: { exp: Math.floor(Date.now() / 1000) - 60 } }, "OIDC id_token has invalid exp"],
    ["a non-numeric expiry", { claims: { exp: "4102444800" } }, "OIDC id_token has invalid exp"],
    ["a missing subject", { claims: { sub: undefined } }, "OIDC id_token has invalid sub"],
    ["an empty subject", { claims: { sub: "" } }, "OIDC id_token has invalid sub"],
    ["a missing id_token", { tokenBody: { id_token: undefined } }, "OIDC token response has no valid id_token"],
    ["a two-segment id_token", { tokenBody: { id_token: "e30.e30" } }, "OIDC token response has no valid id_token"],
    [
      "a non-JSON payload",
      { tokenBody: { id_token: "e30.bm90LWpzb24.sig" } },
      "OIDC token response has no valid id_token",
    ],
    ["a non-object payload", { tokenBody: { id_token: "e30.MTIz.sig" } }, "OIDC token response has no valid id_token"],
    [
      "a non-base64url payload",
      { tokenBody: { id_token: "e30.e30+.sig" } },
      "OIDC token response has no valid id_token",
    ],
    [
      "a discovery issuer mismatch",
      { discovery: { issuer: "https://other.example.com" } },
      "OIDC discovery issuer does not match the configured issuer",
    ],
    [
      "a non-https authorization endpoint",
      { discovery: { authorization_endpoint: "http://idp.example.com/authorize" } },
      "OIDC discovery has invalid authorization_endpoint",
    ],
    [
      "a non-https token endpoint",
      { discovery: { token_endpoint: "http://idp.example.com/token" } },
      "OIDC discovery has invalid token_endpoint",
    ],
    [
      "no S256 support",
      { discovery: { code_challenge_methods_supported: ["plain"] } },
      "OIDC discovery does not support S256",
    ],
    [
      "a discovery redirect",
      {
        discoveryResponse: () =>
          new Response(null, { status: 302, headers: { location: "https://other.example.com/.well-known" } }),
      },
      "OIDC discovery failed (HTTP 302)",
    ],
    [
      "a token redirect",
      { token: () => new Response(null, { status: 302, headers: { location: "https://other.example.com/token" } }) },
      "OIDC token exchange failed (HTTP 302)",
    ],
    [
      "a callback error with a description",
      { callbackQuery: (state) => `state=${state}&error=access_denied&error_description=secret-description` },
      "OIDC login was denied (access_denied)",
    ],
    [
      "a callback error with an unsafe code",
      { callbackQuery: (state) => `state=${state}&error=Secret%20Code` },
      "OIDC login was denied",
    ],
    [
      "a token error with a description",
      {
        token: () =>
          jsonResponse(400, { error: "invalid_grant", error_description: "secret-description authorization-code" }),
      },
      "OIDC token exchange rejected (invalid_grant)",
    ],
    [
      "a token error with an unsafe code",
      { token: () => jsonResponse(401, { error: "Secret\nCode", error_description: "secret-description" }) },
      "OIDC token exchange failed (HTTP 401)",
    ],
  ])("rejects %s", async (_name, options, message) => {
    const { credential, error, requests } = await runOidcLogin(options);

    expect(credential).toBeUndefined();
    // Exact messages: no code, verifier, token, query string, or error_description is echoed.
    expect(error?.message).toBe(message);
    expect(requests.every(({ url }) => new URL(url).origin === issuer)).toBe(true);
  });

  it.each<[string, unknown, string]>([
    ["a string", issuer, "oidc"],
    ["null", null, "oidc"],
    ["a missing issuer", { clientId }, "oidc.issuer"],
    ["an http issuer", { issuer: "http://idp.example.com", clientId }, "oidc.issuer"],
    ["issuer credentials", { issuer: "https://user:pass@idp.example.com", clientId }, "oidc.issuer"],
    ["an issuer query", { issuer: `${issuer}?tenant=alpha`, clientId }, "oidc.issuer"],
    ["an issuer fragment", { issuer: `${issuer}#alpha`, clientId }, "oidc.issuer"],
    ["a missing client ID", { issuer }, "oidc.clientId"],
    ["an empty client ID", { issuer, clientId: "" }, "oidc.clientId"],
    ["a scope without openid", { issuer, clientId, scope: "profile email" }, "oidc.scope"],
    ["a non-array redirectPorts", { issuer, clientId, redirectPorts: 8400 }, "oidc.redirectPorts"],
    ["a zero port", { issuer, clientId, redirectPorts: [0] }, "oidc.redirectPorts"],
    ["an out-of-range port", { issuer, clientId, redirectPorts: [8400, 65536] }, "oidc.redirectPorts"],
    ["a fractional port", { issuer, clientId, redirectPorts: [8400.5] }, "oidc.redirectPorts"],
  ])("fails login without reaching LiteLLM when oidc is %s", async (_name, oidc, field) => {
    const { error, requests } = await runOidcLogin({ oidc });

    expect(error?.message).toMatch(new RegExp(`^Invalid LiteLLM ${field.replace(".", "\\.")} setting: `));
    expect(requests).toEqual([]);
  });

  describe("refresh", () => {
    const now = 1_800_000_000_000;
    const exp = now / 1000 + 3600;
    const freshIdToken = (sub = "user-123") => idToken({ iss: issuer, aud: clientId, sub, exp });
    const oidcCredential = (overrides: Record<string, unknown> = {}) => ({
      type: "oauth" as const,
      access: idToken({ iss: issuer, aud: clientId, sub: "user-123", exp: now / 1000 + 60 }),
      refresh: "refresh-old",
      expires: now + 60_000,
      baseUrl: proxyUrl,
      flow: "oidc_pkce",
      issuer,
      clientId,
      tokenEndpoint,
      subject: "user-123",
      ...overrides,
    });

    async function refreshOidc(
      credential: ReturnType<typeof oidcCredential>,
      respond: () => Response | Promise<Response>,
      agentDir?: string,
    ) {
      process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
      process.env.LITELLM_HEADERS = '{"x-gateway-secret":"gateway-secret"}';
      const extension = await loadExtension(agentDir ?? (await makeAgentDir()));
      const pi = createPi();
      await extension(pi);
      vi.spyOn(Date, "now").mockReturnValue(now);
      const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async () => respond());
      const result = await pi.providers[0]!.auth.oauth!.refresh(credential, TEST_SIGNAL).then(
        (refreshed) => ({ refreshed, error: undefined }),
        (error: Error) => ({ refreshed: undefined, error }),
      );
      return { ...result, fetchMock };
    }

    it("rotates the id_token and refresh token", async () => {
      const credential = oidcCredential();
      const fresh = freshIdToken();
      const { refreshed, error, fetchMock } = await refreshOidc(credential, () =>
        jsonResponse(200, { id_token: fresh, refresh_token: "refresh-new", token_type: "Bearer", expires_in: 3600 }),
      );

      expect(error).toBeUndefined();
      expect(refreshed).toEqual({
        ...credential,
        access: fresh,
        refresh: "refresh-new",
        expires: exp * 1000 - 300_000,
      });
      expect(fetchMock).toHaveBeenCalledOnce();
      const [input, init] = fetchMock.mock.calls[0]!;
      expect(String(input)).toBe(tokenEndpoint);
      expect(Object.fromEntries(new URLSearchParams(String(init?.body)))).toEqual({
        grant_type: "refresh_token",
        refresh_token: "refresh-old",
        client_id: clientId,
      });
      expect(init?.redirect).toBe("manual");
      expect(new Headers(init?.headers).get("x-gateway-secret")).toBeNull();
    });

    it("keeps the refresh token when the IdP does not rotate it", async () => {
      const credential = oidcCredential();
      const fresh = freshIdToken();
      const { refreshed } = await refreshOidc(credential, () => jsonResponse(200, { id_token: fresh }));

      expect(refreshed).toEqual({ ...credential, access: fresh, expires: exp * 1000 - 300_000 });
    });

    it.each([
      ["a long-lived", 3600, now + 3_600_000 - 300_000],
      // Less than twice the five-minute lead left: refresh halfway rather than treat it as already expired.
      ["a short-lived", 120, now + 60_000],
      ["a fractional-expiry", 3600.0004, now + 3_600_000 - 300_000],
    ])("schedules the next refresh of %s id_token before it expires", async (_name, lifetime, expires) => {
      const fresh = idToken({ iss: issuer, aud: clientId, sub: "user-123", exp: now / 1000 + lifetime });
      const { refreshed, error } = await refreshOidc(oidcCredential(), () => jsonResponse(200, { id_token: fresh }));

      expect(error).toBeUndefined();
      expect(refreshed?.expires).toBe(expires);
    });

    it.each<[string, () => Response, string]>([
      ["a changed subject", () => jsonResponse(200, { id_token: freshIdToken("user-456") }), "invalid sub"],
      [
        "a missing id_token",
        () => jsonResponse(200, { access_token: "idp-access-token", token_type: "Bearer" }),
        "no valid id_token",
      ],
      [
        "an id_token for another client",
        () => jsonResponse(200, { id_token: idToken({ iss: issuer, aud: "other-client", sub: "user-123", exp }) }),
        "invalid aud",
      ],
      [
        "invalid_grant",
        () => jsonResponse(400, { error: "invalid_grant", error_description: "refresh-old secret-description" }),
        "OIDC token exchange rejected (invalid_grant)",
      ],
      ["a redirect", () => new Response(null, { status: 302, headers: { location: "/next" } }), "(HTTP 302)"],
    ])("requires a new login after %s", async (_name, respond, message) => {
      const { refreshed, error } = await refreshOidc(oidcCredential(), respond);

      expect(refreshed).toBeUndefined();
      expect(error?.message).toContain(message);
      expect(error?.message).toMatch(/; run \/login litellm again$/);
      expect(error?.message).not.toContain("refresh-old");
      expect(error?.message).not.toContain("secret-description");
    });

    it.each([429, 503, "network"])("keeps the credential before expiry after %s", async (failure) => {
      const credential = oidcCredential();
      const { refreshed } = await refreshOidc(credential, () => {
        if (failure === "network") throw new TypeError("network unavailable");
        return jsonResponse(failure as number, {});
      });

      expect(refreshed).toEqual(credential);
    });

    it.each([
      [429, "HTTP 429"],
      [503, "HTTP 503"],
      ["network", "network error"],
    ])("fails without asking for a new login after %s once the id_token has expired", async (failure, reason) => {
      const { error } = await refreshOidc(oidcCredential({ expires: now }), () => {
        if (failure === "network") throw new TypeError("network unavailable");
        return jsonResponse(failure as number, {});
      });

      expect(error?.message).toBe(`OIDC token exchange failed (${reason})`);
    });

    it.each<Record<string, unknown>>([
      { refresh: "" },
      { tokenEndpoint: "http://idp.example.com/token" },
      { tokenEndpoint: "https://user@idp.example.com/token" },
      { clientId: "" },
      { issuer: undefined },
      { subject: "" },
    ])("requires a new login without contacting the IdP when stored %j", async (override) => {
      const { error, fetchMock } = await refreshOidc(oidcCredential(override), () => jsonResponse(200, {}));

      expect(error?.message).toMatch(/; run \/login litellm again$/);
      expect(fetchMock).not.toHaveBeenCalled();
    });

    it("never executes a refresh token that starts with !", async () => {
      const agentDir = await makeAgentDir();
      const helperPath = await writeHelper(agentDir, ["executed-token"]);
      const credential = oidcCredential({ refresh: `!${helperPath}` });
      const fresh = freshIdToken();

      const refreshed = await refreshOidc(credential, () => jsonResponse(200, { id_token: fresh }), agentDir);
      const rejected = await refreshOidc(credential, () => jsonResponse(400, { error: "invalid_grant" }), agentDir);

      expect(refreshed.refreshed).toMatchObject({ access: fresh, refresh: `!${helperPath}` });
      expect(new URLSearchParams(String(refreshed.fetchMock.mock.calls[0]?.[1]?.body)).get("refresh_token")).toBe(
        `!${helperPath}`,
      );
      expect(rejected.error?.message).toBe("OIDC token exchange rejected (invalid_grant); run /login litellm again");
      expect(await readHelperCount(agentDir)).toBe(0);
    });
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
