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
  "LITELLM_HEADERS",
  "LITELLM_ANTHROPIC_API_KEY",
  "LITELLM_ANTHROPIC_HEADERS",
  "LITELLM_DISCOVERY_TIMEOUT_MS",
  "LITELLM_GCLOUD_TOKEN_AUTH",
  "GOOGLE_APPLICATION_CREDENTIALS",
  "STORED_LITELLM_KEY",
  "CUSTOM_LITELLM_KEY",
];
const ORIGINAL_ENV = new Map(ENV_KEYS.map((key) => [key, process.env[key]]));

vi.unmock("@earendil-works/pi-coding-agent");

type TestProviderConfig = {
  baseUrl?: string;
  apiKey?: string;
  headers?: Record<string, string>;
  models?: unknown[];
  oauth?: {
    login: (callbacks: {
      onPrompt: (options: { message: string; placeholder?: string }) => Promise<string>;
      onAuth?: (info: { url: string; instructions?: string }) => void;
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
  handler: (args: string, ctx: TestCommandContext) => Promise<void> | void;
};

type TestCommandContext = {
  ui: {
    input?: (title: string, placeholder?: string) => Promise<string | undefined>;
    notify: (message: string, type: string) => void;
  };
  modelRegistry?: {
    authStorage: {
      set: (provider: string, credential: unknown) => void;
    };
    refresh?: () => void;
  };
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

async function writeAdcFile(agentDir: string, name: string, refreshToken: string): Promise<string> {
  const adcPath = join(agentDir, `${name}.json`);
  await writeFile(
    adcPath,
    JSON.stringify({
      type: "authorized_user",
      client_id: `${name}-client`,
      client_secret: `${name}-secret`,
      refresh_token: refreshToken,
    }),
    "utf8",
  );
  return adcPath;
}

async function readHelperCount(agentDir: string): Promise<number> {
  try {
    return Number(await readFile(join(agentDir, "helper-count"), "utf8"));
  } catch {
    return 0;
  }
}

const cachedModels = [{ id: "cached-model", name: "cached-model", provider: "litellm" }];

async function writeModelsConfig(agentDir: string, modelOverrides: Record<string, unknown>): Promise<void> {
  await writeFile(
    join(agentDir, "models.json"),
    JSON.stringify({ providers: { litellm: { modelOverrides } } }),
    "utf8",
  );
}

async function writeModelCache(agentDir: string, helperPath: string, models: unknown[] = cachedModels): Promise<void> {
  await writeFile(
    join(agentDir, "litellm-models.json"),
    JSON.stringify({
      baseUrl: "https://litellm.example.com",
      apiKeyFingerprint: fingerprint(`!${helperPath}`),
      fetchedAt: Date.now(),
      source: "model_info",
      models,
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

    return { AuthStorage: TestAuthStorage, defineTool: (tool: unknown) => tool, getAgentDir: () => agentDir };
  });
  const mod = await import("../src/index.js");
  return mod.default as unknown as (pi: TestPi) => Promise<void>;
}

function createPi(): TestPi {
  return {
    providers: [],
    commands: new Map(),
    handlers: new Map(),
    tools: [],
    registerProvider(name: string, config: TestProviderConfig) {
      this.providers.push({ name, config });
    },
    registerCommand(name: string, command: TestCommand) {
      this.commands.set(name, command);
    },
    registerTool(tool: { name: string; description: string; execute?: (...args: any[]) => Promise<any> | any }) {
      this.tools.push(tool);
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
  tools: Array<{ name: string; description: string; execute?: (...args: any[]) => Promise<any> | any }>;
  registerProvider(name: string, config: TestProviderConfig): void;
  registerCommand(name: string, command: TestCommand): void;
  registerTool(tool: { name: string; description: string; execute?: (...args: any[]) => Promise<any> | any }): void;
  on(event: string, handler: (event: any, ctx?: any) => Promise<any> | any): void;
};

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
  it("registers the API key as an explicit environment reference", async () => {
    const agentDir = await makeAgentDir();
    const extension = await loadExtension(agentDir);
    const pi = createPi();

    await extension(pi);

    expect(pi.providers[0]?.config.apiKey).toBe("$LITELLM_API_KEY");
  });

  it("registers the env key when gcloud ADC auth falls back during discovery", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    process.env.LITELLM_GCLOUD_TOKEN_AUTH = "1";
    process.env.GOOGLE_APPLICATION_CREDENTIALS = join(agentDir, "missing-adc.json");
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined);
    const seenAuthHeaders: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        seenAuthHeaders.push(new Headers(init?.headers).get("authorization") ?? "");
        return jsonResponse(200, {
          data: [{ model_name: "openai/gpt-4o", model_info: { mode: "chat" } }],
        });
      }
      if (url.endsWith("/mcp-rest/tools/list")) return jsonResponse(200, { tools: [] });
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining("Failed to read ADC file"));
    expect(seenAuthHeaders).toEqual(["Bearer env-key"]);
    expect(pi.providers[0]?.config.apiKey).toBe("$LITELLM_API_KEY");
    const cache = JSON.parse(await readFile(join(agentDir, "litellm-models.json"), "utf8")) as {
      apiKeyFingerprint: string;
    };
    expect(cache.apiKeyFingerprint).toBe(fingerprint("env-key"));
  });

  it("refreshes the model cache when the gcloud ADC identity changes", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_GCLOUD_TOKEN_AUTH = "1";
    const seenModelAuthHeaders: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url === "https://oauth2.googleapis.com/token") {
        const body = String(init?.body ?? "");
        return jsonResponse(200, {
          access_token: body.includes("refresh_token=second-refresh") ? "second-token" : "first-token",
        });
      }
      if (url.endsWith("/model/info")) {
        const authorization = new Headers(init?.headers).get("authorization") ?? "";
        seenModelAuthHeaders.push(authorization);
        return jsonResponse(200, {
          data: [
            {
              model_name: authorization === "Bearer second-token" ? "second-model" : "first-model",
              model_info: { mode: "chat" },
            },
          ],
        });
      }
      if (url.endsWith("/mcp-rest/tools/list")) return jsonResponse(200, { tools: [] });
      throw new Error(`unexpected URL: ${url}`);
    });

    process.env.GOOGLE_APPLICATION_CREDENTIALS = await writeAdcFile(agentDir, "first-adc", "first-refresh");
    let extension = await loadExtension(agentDir);
    let pi = createPi();
    await extension(pi);
    expect((pi.providers[0].config.models as Array<{ id: string }>).map((model) => model.id)).toEqual(["first-model"]);

    process.env.GOOGLE_APPLICATION_CREDENTIALS = await writeAdcFile(agentDir, "second-adc", "second-refresh");
    extension = await loadExtension(agentDir);
    pi = createPi();
    await extension(pi);

    expect(seenModelAuthHeaders).toEqual(["Bearer first-token", "Bearer second-token"]);
    expect((pi.providers[0].config.models as Array<{ id: string }>).map((model) => model.id)).toEqual(["second-model"]);
  });

  it('treats literal "undefined" env values as unset', async () => {
    process.env.LITELLM_BASE_URL = "undefined";
    process.env.LITELLM_API_KEY = "undefined";
    const agentDir = await makeAgentDir();
    const extension = await loadExtension(agentDir);
    const pi = createPi();

    await extension(pi);

    expect(pi.providers[0]?.config.baseUrl).toBe("https://litellm.example.com/v1");
  });

  it("applies models.json overrides to discovered models", async () => {
    const agentDir = await makeAgentDir();
    await writeModelsConfig(agentDir, {
      "llm-gateway/gpt-5.5": { contextWindow: 272_000, maxTokens: 64_000 },
    });
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "llm-gateway/gpt-5.5",
              model_info: { mode: "chat", max_input_tokens: 128_000, max_output_tokens: 16_384 },
            },
          ],
        });
      }
      if (url.endsWith("/mcp-rest/tools/list")) return jsonResponse(200, { tools: [] });
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(pi.providers[0]?.config.models).toEqual([
      expect.objectContaining({ id: "llm-gateway/gpt-5.5", contextWindow: 272_000, maxTokens: 64_000 }),
    ]);
  });

  it("applies models.json overrides to cached models", async () => {
    const agentDir = await makeAgentDir();
    const helperPath = await writeHelper(agentDir, [makeJwt(Math.floor(Date.now() / 1000) + 3600)]);
    await writeModelCache(agentDir, helperPath);
    await writeModelsConfig(agentDir, { "cached-model": { contextWindow: 272_000 } });
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY_HELPER = helperPath;
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(pi.providers[0]?.config.models).toEqual([
      expect.objectContaining({ id: "cached-model", contextWindow: 272_000 }),
    ]);
  });

  it("merges partial thinkingLevelMap overrides with existing mappings", async () => {
    const agentDir = await makeAgentDir();
    const helperPath = await writeHelper(agentDir, [makeJwt(Math.floor(Date.now() / 1000) + 3600)]);
    await writeModelCache(agentDir, helperPath, [
      {
        id: "cached-model",
        name: "cached-model",
        provider: "litellm",
        thinkingLevelMap: { low: "low", high: "high" },
      },
    ]);
    await writeModelsConfig(agentDir, { "cached-model": { thinkingLevelMap: { high: "xhigh" } } });
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY_HELPER = helperPath;
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(pi.providers[0]?.config.models).toEqual([
      expect.objectContaining({ id: "cached-model", thinkingLevelMap: { low: "low", high: "xhigh" } }),
    ]);
  });

  it("deep-merges nested compat overrides", async () => {
    const agentDir = await makeAgentDir();
    const helperPath = await writeHelper(agentDir, [makeJwt(Math.floor(Date.now() / 1000) + 3600)]);
    await writeModelCache(agentDir, helperPath, [
      {
        id: "cached-model",
        name: "cached-model",
        provider: "litellm",
        compat: { supportsStore: false, chatTemplateKwargs: { enable_thinking: true } },
      },
    ]);
    await writeModelsConfig(agentDir, { "cached-model": { compat: { chatTemplateKwargs: { max_thinking: 4 } } } });
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY_HELPER = helperPath;
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(pi.providers[0]?.config.models).toEqual([
      expect.objectContaining({
        id: "cached-model",
        compat: { supportsStore: false, chatTemplateKwargs: { enable_thinking: true, max_thinking: 4 } },
      }),
    ]);
  });

  it("applies overrides from models.json with comments and trailing commas", async () => {
    const agentDir = await makeAgentDir();
    const helperPath = await writeHelper(agentDir, [makeJwt(Math.floor(Date.now() / 1000) + 3600)]);
    await writeModelCache(agentDir, helperPath);
    await writeFile(
      join(agentDir, "models.json"),
      `{
  // raise the context window
  "providers": {
    "litellm": {
      "modelOverrides": {
        "cached-model": { "contextWindow": 272000, },
      },
    },
  },
}`,
      "utf8",
    );
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY_HELPER = helperPath;
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(pi.providers[0]?.config.models).toEqual([
      expect.objectContaining({ id: "cached-model", contextWindow: 272_000 }),
    ]);
  });

  it("warns and ignores overrides when models.json is malformed", async () => {
    const agentDir = await makeAgentDir();
    const helperPath = await writeHelper(agentDir, [makeJwt(Math.floor(Date.now() / 1000) + 3600)]);
    await writeModelCache(agentDir, helperPath);
    await writeFile(join(agentDir, "models.json"), "{ not json", "utf8");
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY_HELPER = helperPath;
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const stderrSpy = vi.spyOn(process.stderr, "write").mockImplementation(() => true);

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(pi.providers[0]?.config.models).toEqual([expect.objectContaining({ id: "cached-model" })]);
    expect(stderrSpy).toHaveBeenCalledWith(expect.stringContaining("models.json"));
  });

  it("warns and drops override fields with invalid types", async () => {
    const agentDir = await makeAgentDir();
    const helperPath = await writeHelper(agentDir, [makeJwt(Math.floor(Date.now() / 1000) + 3600)]);
    await writeModelCache(agentDir, helperPath, [
      { id: "cached-model", name: "cached-model", provider: "litellm", contextWindow: 128_000 },
    ]);
    await writeModelsConfig(agentDir, { "cached-model": { contextWindow: "272000", maxTokens: 64_000 } });
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY_HELPER = helperPath;
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const stderrSpy = vi.spyOn(process.stderr, "write").mockImplementation(() => true);

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(pi.providers[0]?.config.models).toEqual([
      expect.objectContaining({ id: "cached-model", contextWindow: 128_000, maxTokens: 64_000 }),
    ]);
    expect(stderrSpy).toHaveBeenCalledWith(expect.stringContaining("contextWindow"));
  });

  it("drops override fields whose values violate the core schema", async () => {
    const agentDir = await makeAgentDir();
    const helperPath = await writeHelper(agentDir, [makeJwt(Math.floor(Date.now() / 1000) + 3600)]);
    await writeModelCache(agentDir, helperPath, [
      { id: "cached-model", name: "cached-model", provider: "litellm", contextWindow: 128_000 },
    ]);
    await writeModelsConfig(agentDir, {
      "cached-model": {
        headers: { "X-Team": "core", "X-Retries": 3 },
        thinkingLevelMap: { low: 42 },
        input: ["text", "audio"],
        maxTokens: 64_000,
      },
    });
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY_HELPER = helperPath;
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const stderrSpy = vi.spyOn(process.stderr, "write").mockImplementation(() => true);

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const model = pi.providers[0]?.config.models?.[0] as Record<string, unknown>;
    expect(model).toEqual(expect.objectContaining({ id: "cached-model", maxTokens: 64_000 }));
    expect(model.headers).toBeUndefined();
    expect(model.thinkingLevelMap).toBeUndefined();
    expect(model.input).toBeUndefined();
    expect(stderrSpy).toHaveBeenCalledWith(expect.stringContaining("headers, thinkingLevelMap, input"));
  });

  it("applies models.json overrides written after startup on refresh", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "llm-gateway/gpt-5.5",
              model_info: { mode: "chat", max_input_tokens: 128_000, max_output_tokens: 16_384 },
            },
          ],
        });
      }
      if (url.endsWith("/mcp-rest/tools/list")) return jsonResponse(200, { tools: [] });
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    expect(pi.providers[0]?.config.models).toEqual([
      expect.objectContaining({ id: "llm-gateway/gpt-5.5", contextWindow: 128_000 }),
    ]);

    await writeModelsConfig(agentDir, { "llm-gateway/gpt-5.5": { contextWindow: 272_000 } });
    const notify = vi.fn();
    await pi.commands.get("litellm-refresh")?.handler("", { ui: { notify } });

    expect(pi.providers.at(-1)?.config.models).toEqual([
      expect.objectContaining({ id: "llm-gateway/gpt-5.5", contextWindow: 272_000 }),
    ]);
  });

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

    expect(seenAuthHeaders).toEqual(["Bearer stored-key", "Bearer stored-key"]);
  });

  it("registers multiple configured LiteLLM provider aliases with separate credentials and caches", async () => {
    const agentDir = await makeAgentDir();
    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({
        litellm: {
          providers: {
            "litellm-anthropic": {
              baseUrl: "https://litellm-anthropic.example.com",
              apiKey: "$LITELLM_ANTHROPIC_API_KEY",
              headers: "$LITELLM_ANTHROPIC_HEADERS",
            },
          },
        },
      }),
      "utf8",
    );
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "openai-key";
    process.env.LITELLM_HEADERS = JSON.stringify({ "x-litellm-customer-id": "openai-customer" });
    process.env.LITELLM_ANTHROPIC_API_KEY = "anthropic-key";
    process.env.LITELLM_ANTHROPIC_HEADERS = JSON.stringify({ "x-litellm-customer-id": "anthropic-customer" });

    const seenModelInfoRequests: Array<{ url: string; authorization: string; customer: string }> = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        const headers = new Headers(init?.headers);
        seenModelInfoRequests.push({
          url,
          authorization: headers.get("authorization") ?? "",
          customer: headers.get("x-litellm-customer-id") ?? "",
        });
        return jsonResponse(200, {
          data: [
            {
              model_name: url.includes("anthropic") ? "claude-sonnet" : "gpt-5",
              model_info: { mode: "chat" },
            },
          ],
        });
      }
      if (url.endsWith("/mcp-rest/tools/list")) return jsonResponse(200, { tools: [] });
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(pi.providers.map((provider) => provider.name)).toEqual(["litellm", "litellm-anthropic"]);
    expect(pi.providers[0]?.config).toMatchObject({
      baseUrl: "https://litellm.example.com/v1",
      apiKey: "$LITELLM_API_KEY",
      headers: { "x-litellm-customer-id": "openai-customer" },
    });
    expect(pi.providers[1]?.config).toMatchObject({
      baseUrl: "https://litellm-anthropic.example.com/v1",
      apiKey: "$LITELLM_ANTHROPIC_API_KEY",
      headers: { "x-litellm-customer-id": "anthropic-customer" },
    });
    expect((pi.providers[0].config.models as Array<{ id: string }>).map((model) => model.id)).toEqual(["gpt-5"]);
    expect((pi.providers[1].config.models as Array<{ id: string }>).map((model) => model.id)).toEqual([
      "claude-sonnet",
    ]);
    // Providers load in parallel, so cross-provider request order is not deterministic.
    expect(seenModelInfoRequests).toHaveLength(2);
    expect(seenModelInfoRequests).toContainEqual({
      url: "https://litellm.example.com/model/info",
      authorization: "Bearer openai-key",
      customer: "openai-customer",
    });
    expect(seenModelInfoRequests).toContainEqual({
      url: "https://litellm-anthropic.example.com/model/info",
      authorization: "Bearer anthropic-key",
      customer: "anthropic-customer",
    });
    const defaultCache = JSON.parse(await readFile(join(agentDir, "litellm-models.json"), "utf8")) as {
      models: Array<{ id: string }>;
    };
    const aliasCache = JSON.parse(await readFile(join(agentDir, "litellm-models-litellm-anthropic.json"), "utf8")) as {
      models: Array<{ id: string }>;
    };
    expect(defaultCache.models.map((model) => model.id)).toEqual(["gpt-5"]);
    expect(aliasCache.models.map((model) => model.id)).toEqual(["claude-sonnet"]);
  });

  it("does not let the default helper override an alias-specific API key env", async () => {
    const agentDir = await makeAgentDir();
    const helperPath = await writeHelper(agentDir, ["default-helper-key"]);
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
    process.env.LITELLM_API_KEY_HELPER = helperPath;
    process.env.LITELLM_ANTHROPIC_API_KEY = "anthropic-key";

    const seenModelInfoRequests: Array<{ url: string; authorization: string }> = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        seenModelInfoRequests.push({
          url,
          authorization: new Headers(init?.headers).get("authorization") ?? "",
        });
        return jsonResponse(200, {
          data: [{ model_name: url.includes("anthropic") ? "claude-sonnet" : "gpt-5", model_info: { mode: "chat" } }],
        });
      }
      if (url.endsWith("/mcp-rest/tools/list")) return jsonResponse(200, { tools: [] });
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    // Providers load in parallel, so cross-provider request order is not deterministic.
    expect(seenModelInfoRequests).toHaveLength(2);
    expect(seenModelInfoRequests).toContainEqual({
      url: "https://litellm.example.com/model/info",
      authorization: "Bearer default-helper-key",
    });
    expect(seenModelInfoRequests).toContainEqual({
      url: "https://litellm-anthropic.example.com/model/info",
      authorization: "Bearer anthropic-key",
    });
    expect(pi.providers[0]?.config.apiKey).toBe(`!${helperPath}`);
    expect(pi.providers[1]?.config.apiKey).toBe("$LITELLM_ANTHROPIC_API_KEY");
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
    await pi.commands.get("litellm-refresh")?.handler("", { ui: { notify } });

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
        if (options.placeholder) return " http://127.0.0.1:4000/v1 ";
        if (options.message.includes("Select login method")) return "1";
        return " sk-login ";
      },
      onProgress: progress,
      signal: new AbortController().signal,
    });

    expect(promptMessages).toEqual([
      "Enter LiteLLM proxy URL (no trailing /v1):",
      "Select login method (1 = API key / !command, 2 = SSO / Enterprise JWT):",
      "Enter API key:",
    ]);
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

  it("re-registers models discovered during login", async () => {
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
    expect(pi.providers[0]?.config.models).toEqual([]);

    await pi.providers[0]?.config.oauth?.login({
      onPrompt: async (options) => (options.placeholder ? " http://127.0.0.1:4000/v1 " : " sk-login "),
      signal: new AbortController().signal,
    });

    const registeredModels = pi.providers[1]?.config.models as Array<{ id: string }> | undefined;
    expect(pi.providers).toHaveLength(2);
    expect(pi.providers[1]?.config.baseUrl).toBe("http://127.0.0.1:4000/v1");
    expect(registeredModels?.map((model) => model.id)).toEqual(["vidaimock-openai"]);
  });

  it("handles /login litellm without registering a conflicting command", async () => {
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

    expect(pi.commands.has("login")).toBe(false);
    const savedCredentials: Record<string, unknown> = {};
    const promptMessages: string[] = [];
    const notifications: Array<{ message: string; type: string }> = [];
    const result = await pi.handlers.get("input")?.[0]?.(
      { type: "input", text: "/login litellm", source: "interactive" },
      {
        hasUI: true,
        ui: {
          input: async (message: string, placeholder?: string) => {
            promptMessages.push(message);
            if (placeholder) return " http://127.0.0.1:4000/v1 ";
            if (message.includes("Select login method")) return "1";
            return " sk-login ";
          },
          notify: (message: string, type: string) => notifications.push({ message, type }),
        },
        modelRegistry: {
          authStorage: {
            set: (provider: string, credential: unknown) => {
              savedCredentials[provider] = credential;
            },
          },
          refresh: vi.fn(),
        },
      },
    );

    expect(result).toEqual({ action: "handled" });
    expect(promptMessages).toEqual([
      "Enter LiteLLM proxy URL (no trailing /v1):",
      "Select login method (1 = API key / !command, 2 = SSO / Enterprise JWT):",
      "Enter API key:",
    ]);
    expect(savedCredentials.litellm).toMatchObject({
      type: "oauth",
      access: "sk-login",
      refresh: "",
      baseUrl: "http://127.0.0.1:4000",
    });
    const registeredModels = pi.providers[1]?.config.models as Array<{ id: string }> | undefined;
    expect(pi.providers[1]?.config.baseUrl).toBe("http://127.0.0.1:4000/v1");
    expect(registeredModels?.map((model) => model.id)).toEqual(["vidaimock-openai"]);
    expect(notifications).toContainEqual({
      message: "LiteLLM: 1 models discovered (source: model_info)",
      type: "info",
    });
  });

  it("continues input handling for non-LiteLLM login arguments", async () => {
    const agentDir = await makeAgentDir();
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const result = await pi.handlers.get("input")?.[0]?.(
      { type: "input", text: "/login other", source: "interactive" },
      {
        ui: {
          notify: vi.fn(),
        },
      },
    );

    expect(result).toEqual({ action: "continue" });
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
      expect(pi.providers).toHaveLength(3);
      expect((pi.providers.at(-1)?.config.models as Array<{ id: string }> | undefined)?.[0]?.id).toBe(
        "vidaimock-openai-2",
      );
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

    // Startup model and MCP discovery use the same fresh helper token (one helper invocation).
    expect(seenAuthHeaders).toEqual([`Bearer ${first}`, `Bearer ${first}`]);
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

    expect(seenAuthHeaders).toEqual([`Bearer ${fresh}`, `Bearer ${fresh}`]);
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
    const credential = await pi.providers[0]?.config.oauth?.login({
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
    expect(seenRequests).toContainEqual(
      expect.objectContaining({
        url: "https://litellm.example.com/key/generate",
        method: "POST",
        authorization: `Bearer ${jwt}`,
      }),
    );
    expect(seenRequests).toContainEqual(
      expect.objectContaining({
        url: "https://litellm.example.com/model/info",
        authorization: "Bearer sk-virtual-abc",
      }),
    );
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

    await pi.providers[0]?.config.oauth?.login({
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

    const credential = await pi.providers[0]?.config.oauth?.login({
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

  it("enterprise SSO login falls back to JWT when virtual key generation stalls", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    const jwt = makeJwt(Math.floor(Date.now() / 1000) + 3600);
    const progress = vi.fn();
    vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
      const url = String(input);
      if (url.endsWith("/key/generate"))
        return new Promise<Response>((_, reject) => {
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

    vi.useFakeTimers();
    try {
      const loginPromise = pi.providers[0]?.config.oauth?.login({
        onPrompt: async (options) => {
          if (options.placeholder) return "https://litellm.example.com";
          if (options.message.includes("Select login method")) return "2";
          if (options.message.includes("SSO token")) return jwt;
          return "y";
        },
        onProgress: progress,
        signal: new AbortController().signal,
      });
      await vi.advanceTimersByTimeAsync(10_000);
      const credential = await loginPromise;
      expect(credential).toMatchObject({ access: jwt, refresh: "" });
      expect(progress).toHaveBeenCalledWith(expect.stringContaining("virtual key generation failed"));
    } finally {
      vi.useRealTimers();
    }
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

    const credential = await pi.providers[0]?.config.oauth?.login({
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

    const credential = await pi.providers[0]?.config.oauth?.login({
      onPrompt: async (options) => {
        if (options.placeholder) return "https://litellm.example.com";
        if (options.message.includes("Select login method")) return "2";
        if (options.message.includes("SSO token")) return jwt;
        return "y";
      },
      signal: new AbortController().signal,
    });

    await expect(pi.providers[0]?.config.oauth?.refreshToken(credential!)).rejects.toThrow(
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

    const credential = await pi.providers[0]?.config.oauth?.login({
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
      pi.providers[0]?.config.oauth?.login({
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
  it("registers remaining providers when one alias's credential helper fails at startup", async () => {
    const agentDir = await makeAgentDir();
    const failingHelper = join(agentDir, "broken-alias-helper.sh");
    await writeFile(failingHelper, "#!/usr/bin/env bash\nexit 42\n", { mode: 0o700 });
    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({
        litellm: {
          providers: {
            "litellm-anthropic": {
              baseUrl: "https://litellm-anthropic.example.com",
              apiKey: `!${failingHelper}`,
            },
          },
        },
      }),
      "utf8",
    );
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "openai-key";
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url === "https://litellm.example.com/model/info") {
        return jsonResponse(200, { data: [{ model_name: "gpt-5", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/mcp-rest/tools/list")) return jsonResponse(200, { tools: [] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(pi.providers.map((provider) => provider.name)).toEqual(["litellm", "litellm-anthropic"]);
    expect((pi.providers[0].config.models as Array<{ id: string }>).map((model) => model.id)).toEqual(["gpt-5"]);
    expect(pi.providers[1]?.config.models).toEqual([]);
    expect(stderrSpy).toHaveBeenCalledWith(expect.stringContaining("litellm-anthropic"));
  });

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

    expect(pi.providers[0]?.config.apiKey).toBe("$LITELLM_API_KEY");
    expect(pi.providers[1]?.name).toBe("litellm-anthropic");
    expect(pi.providers[1]?.config.apiKey).toBeUndefined();
  });

  it("escapes resolved header values so Pi core does not re-resolve them per request", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "openai-key";
    process.env.LITELLM_HEADERS = JSON.stringify({ "x-secret": "v$$X", "x-bang": "!not-a-command" });
    const seenSecretHeaders: Array<string | null> = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        seenSecretHeaders.push(new Headers(init?.headers).get("x-secret"));
        return jsonResponse(200, { data: [{ model_name: "gpt-5", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/mcp-rest/tools/list")) return jsonResponse(200, { tools: [] });
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(seenSecretHeaders).toEqual(["v$X"]);
    expect(pi.providers[0]?.config.headers).toEqual({ "x-secret": "v$$X", "x-bang": "$!not-a-command" });
  });

  it("prefers a configured default-provider apiKey over the env helper command", async () => {
    const agentDir = await makeAgentDir();
    const helperPath = await writeHelper(agentDir, ["default-helper-key"]);
    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({
        litellm: { providers: { litellm: { apiKey: "$CUSTOM_LITELLM_KEY" } } },
      }),
      "utf8",
    );
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY_HELPER = helperPath;
    process.env.CUSTOM_LITELLM_KEY = "custom-key";
    const seenAuthHeaders: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        seenAuthHeaders.push(new Headers(init?.headers).get("authorization") ?? "");
        return jsonResponse(200, { data: [{ model_name: "gpt-5", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/mcp-rest/tools/list")) return jsonResponse(200, { tools: [] });
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(seenAuthHeaders).toEqual(["Bearer custom-key"]);
    expect(pi.providers[0]?.config.apiKey).toBe("$CUSTOM_LITELLM_KEY");
    expect(await readHelperCount(agentDir)).toBe(0);
  });

  it("does not execute an alias key command when a stored auth entry wins", async () => {
    const agentDir = await makeAgentDir();
    const countingHelper = await writeHelper(agentDir, ["alias-command-key"]);
    await writeFile(
      join(agentDir, "auth.json"),
      JSON.stringify({ "litellm-anthropic": { type: "api_key", key: "stored-alias-key" } }),
      "utf8",
    );
    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({
        litellm: {
          providers: {
            "litellm-anthropic": {
              baseUrl: "https://litellm-anthropic.example.com",
              apiKey: `!${countingHelper}`,
            },
          },
        },
      }),
      "utf8",
    );
    const seenAuthHeaders: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        seenAuthHeaders.push(new Headers(init?.headers).get("authorization") ?? "");
        return jsonResponse(200, { data: [{ model_name: "claude-sonnet", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/mcp-rest/tools/list")) return jsonResponse(200, { tools: [] });
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(seenAuthHeaders).toEqual(["Bearer stored-alias-key"]);
    expect(await readHelperCount(agentDir)).toBe(0);
  });

  it("warns when a configured apiKey resolves to nothing", async () => {
    const agentDir = await makeAgentDir();
    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({
        litellm: {
          providers: {
            "litellm-anthropic": {
              baseUrl: "https://litellm-anthropic.example.com",
              apiKey: "sk-abc$UNSET_LITELLM_VAR",
            },
          },
        },
      }),
      "utf8",
    );
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    delete process.env.UNSET_LITELLM_VAR;
    const stderrSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);

    const extension = await loadExtension(agentDir);
    await extension(createPi());

    expect(stderrSpy).toHaveBeenCalledWith(
      expect.stringContaining("LiteLLM (litellm-anthropic): configured apiKey did not resolve"),
    );
  });

  it("reports both successes and failures when /litellm-refresh spans providers", async () => {
    const agentDir = await makeAgentDir();
    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({
        litellm: {
          providers: {
            "litellm-anthropic": {
              baseUrl: "https://litellm-anthropic.example.com",
              apiKey: "$UNSET_ALIAS_KEY",
            },
          },
        },
      }),
      "utf8",
    );
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "openai-key";
    delete process.env.UNSET_ALIAS_KEY;
    vi.spyOn(process.stderr, "write").mockReturnValue(true);
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url === "https://litellm.example.com/model/info") {
        return jsonResponse(200, { data: [{ model_name: "gpt-5", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/mcp-rest/tools/list")) return jsonResponse(200, { tools: [] });
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const notify = vi.fn();
    await pi.commands.get("litellm-refresh")?.handler("", { ui: { notify } });

    expect(notify).toHaveBeenCalledTimes(1);
    const [message, type] = notify.mock.calls[0] ?? [];
    expect(type).toBe("warning");
    expect(message).toContain("litellm: 1 models");
    expect(message).toContain("litellm-anthropic");
  });

  it("sends custom headers when generating a login virtual key", async () => {
    const agentDir = await makeAgentDir();
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";
    process.env.LITELLM_HEADERS = JSON.stringify({ "x-litellm-customer-id": "team-a" });
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

    await pi.providers[0]?.config.oauth?.login({
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

  it("skips an alias whose cache file would collide with another provider", async () => {
    const agentDir = await makeAgentDir();
    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({
        litellm: {
          providers: {
            "team-claude": { baseUrl: "https://a.example.com", apiKey: "$LITELLM_ANTHROPIC_API_KEY" },
            "Team Claude": { baseUrl: "https://b.example.com", apiKey: "$LITELLM_ANTHROPIC_API_KEY" },
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

    expect(pi.providers.map((provider) => provider.name)).toEqual(["litellm", "team-claude"]);
    expect(stderrSpy).toHaveBeenCalledWith(expect.stringContaining("Team Claude"));
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

    expect(pi.providers[1]?.config.headers).toEqual({ "x-num": "30", "x-bool": "false" });
    expect(stderrSpy).toHaveBeenCalledWith(expect.stringContaining("x-obj"));
    expect(stderrSpy).toHaveBeenCalledWith(expect.stringContaining("x-null"));
  });

  it("invalidates the cache when configured headers change even if the base URL and key match", async () => {
    const agentDir = await makeAgentDir();
    await writeFile(
      join(agentDir, "litellm-models.json"),
      JSON.stringify({
        baseUrl: "https://litellm.example.com",
        apiKeyFingerprint: fingerprint("env-key"),
        fetchedAt: Date.now(),
        source: "model_info",
        models: cachedModels,
      }),
      "utf8",
    );
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    process.env.LITELLM_HEADERS = JSON.stringify({ "x-litellm-customer-id": "team-b" });
    const seenRequests: Array<{ customer: string | null }> = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        seenRequests.push({ customer: new Headers(init?.headers).get("x-litellm-customer-id") });
        return jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/mcp-rest/tools/list")) return jsonResponse(200, { tools: [] });
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(seenRequests).toEqual([{ customer: "team-b" }]);
    expect((pi.providers[0].config.models as Array<{ id: string }>).map((model) => model.id)).toEqual(["fresh-model"]);
  });
});
