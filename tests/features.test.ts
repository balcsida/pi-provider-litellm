import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createPi, loadExtension, type TestPi, useHermeticEnv } from "./test-helpers.js";

useHermeticEnv();

vi.unmock("@earendil-works/pi-coding-agent");

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

async function refreshProvider(pi: TestPi, allowNetwork = true, signal?: AbortSignal): Promise<void> {
  await pi.providers[0]?.refreshModels?.({
    allowNetwork,
    stored: undefined,
    publish: async () => true,
    credential: {
      type: "api_key",
      key: process.env.LITELLM_API_KEY ?? "sk-test",
      env: { LITELLM_BASE_URL: process.env.LITELLM_BASE_URL ?? "https://proxy.example.com" },
    },
    signal: signal ?? new AbortController().signal,
  });
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.resetModules();
});

const VALID_ADC = {
  type: "authorized_user",
  client_id: "client-id",
  client_secret: "client-secret",
  refresh_token: "refresh-token",
};

// Builds a provider whose only credential source is Google ADC, so `check` and
// `resolve` can be compared directly for one ADC file shape.
async function setupAdcProvider(adc: unknown | undefined): Promise<{
  check: () => Promise<unknown>;
  resolve: () => Promise<unknown>;
}> {
  const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
  const adcPath = join(agentDir, "adc.json");
  if (adc !== undefined) await writeFile(adcPath, JSON.stringify(adc), "utf8");
  process.env.GOOGLE_APPLICATION_CREDENTIALS = adcPath;
  process.env.LITELLM_BASE_URL = "https://litellm.example.com";
  process.env.LITELLM_GCLOUD_TOKEN_AUTH = "1";
  process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";

  const extension = await loadExtension(agentDir);
  const pi = createPi();
  await extension(pi);
  const ctx = {
    env: async (name: string) => process.env[name],
    fileExists: async () => false,
  };
  const signal = new AbortController().signal;

  // Without these, `pi.providers[0]?.auth...?.()` yields undefined when registration
  // never happened -- the same value the negative cases assert, so they would pass
  // against a provider that was never registered at all.
  const provider = pi.providers[0];
  expect(pi.providers).toHaveLength(1);
  expect(typeof provider?.auth.apiKey?.check).toBe("function");
  expect(typeof provider?.auth.apiKey?.resolve).toBe("function");

  return {
    check: async () => provider?.auth.apiKey?.check?.({ ctx, signal }),
    resolve: async () => provider?.auth.apiKey?.resolve?.({ ctx, signal }),
  };
}

describe("gcloud ADC provider auth", () => {
  it("resolves an in-process ADC token and labels both auth paths consistently", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input instanceof Request ? input.url : input);
      if (url === "https://oauth2.googleapis.com/token") {
        return jsonResponse(200, { access_token: "ya29.minted", expires_in: 3600 });
      }
      throw new Error(`unexpected fetch ${url}`);
    });
    const provider = await setupAdcProvider(VALID_ADC);

    await expect(provider.check()).resolves.toEqual({ type: "api_key", source: "gcloud ADC" });
    const resolved = (await provider.resolve()) as { auth: { apiKey: string }; source: string };
    expect(resolved.auth.apiKey).toBe("ya29.minted");
    expect(resolved.source).toBe("gcloud ADC");
  });

  // ADC outranks the environment key in resolveCredentials, so check must name ADC here.
  // Nothing covered this state before, and reporting the fallback instead looked correct
  // in isolation while contradicting what every request actually sends.
  it("names ADC when it outranks a configured fallback", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input instanceof Request ? input.url : input);
      if (url === "https://oauth2.googleapis.com/token") {
        return jsonResponse(200, { access_token: "ya29.minted", expires_in: 3600 });
      }
      throw new Error(`unexpected fetch ${url}`);
    });
    process.env.LITELLM_API_KEY = "sk-env-fallback";
    const provider = await setupAdcProvider(VALID_ADC);

    const checked = (await provider.check()) as { source: string };
    const resolved = (await provider.resolve()) as { auth: { apiKey: string }; source: string };

    expect(resolved.auth.apiKey).toBe("ya29.minted");
    expect(resolved.source).toBe("gcloud ADC");
    expect(checked.source).toBe(resolved.source);
  });

  // The one state where the two legitimately differ: check cannot know the refresh token
  // is dead without a network call, so resolve reports the credential it fell back to.
  it("falls back to the environment key when ADC cannot mint", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input instanceof Request ? input.url : input);
      if (url === "https://oauth2.googleapis.com/token") return jsonResponse(400, { error: "invalid_grant" });
      throw new Error(`unexpected fetch ${url}`);
    });
    vi.spyOn(console, "warn").mockImplementation(() => {});
    process.env.LITELLM_API_KEY = "sk-env-fallback";
    const provider = await setupAdcProvider(VALID_ADC);

    const resolved = (await provider.resolve()) as { auth: { apiKey: string }; source: string };

    expect(resolved.auth.apiKey).toBe("sk-env-fallback");
    expect(resolved.source).toBe("LITELLM_API_KEY");
  });

  it.for([
    ["an unreadable ADC file", undefined],
    ["service_account credentials", { type: "service_account", client_email: "a@b.iam.gserviceaccount.com" }],
    ["external_account credentials", { type: "external_account", audience: "//iam.googleapis.com/x" }],
    ["a malformed authorized_user file", { type: "authorized_user", client_id: "client-id" }],
  ] as const)("reports the provider unconfigured for %s", async ([, adc]) => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    const provider = await setupAdcProvider(adc);

    await expect(provider.check()).resolves.toBeUndefined();
    await expect(provider.resolve()).resolves.toBeUndefined();
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

describe("feature parity", () => {
  it("registers discovered LiteLLM MCP tools as Pi tools", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return jsonResponse(200, { data: [] });
      if (url.endsWith("/mcp-rest/tools/list")) {
        return jsonResponse(200, {
          tools: [
            {
              name: "search",
              description: "Search the web",
              inputSchema: {
                type: "object",
                properties: { query: { type: "string" } },
                required: ["query"],
              },
              mcp_info: { server_name: "brave", server_id: "brave-api" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    await refreshProvider(pi);

    await vi.waitFor(() => expect(pi.tools.map((tool) => tool.name)).toContain("mcp_brave_search"));
  });

  it("refreshes the MCP catalog when default-provider auth changes", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return jsonResponse(200, { data: [] });
      if (url === "https://first.example.com/mcp-rest/tools/list") {
        return jsonResponse(200, {
          tools: [
            {
              name: "first",
              inputSchema: { type: "object", properties: {} },
              mcp_info: { server_name: "first" },
            },
          ],
        });
      }
      if (url === "https://second.example.com/mcp-rest/tools/list") {
        return jsonResponse(200, {
          tools: [
            {
              name: "second",
              inputSchema: { type: "object", properties: {} },
              mcp_info: { server_name: "second" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    process.env.LITELLM_BASE_URL = "https://first.example.com";
    process.env.LITELLM_API_KEY = "first-token";
    process.env.LITELLM_HEADERS = '{"x-tenant":"first"}';
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    await refreshProvider(pi);

    process.env.LITELLM_BASE_URL = "https://second.example.com";
    process.env.LITELLM_API_KEY = "second-token";
    process.env.LITELLM_HEADERS = '{"x-tenant":"second"}';
    await refreshProvider(pi);
    await vi.waitFor(() => expect(pi.tools.map((tool) => tool.name)).toContain("mcp_second_second"));

    expect(fetchMock).toHaveBeenCalledWith(
      "https://first.example.com/mcp-rest/tools/list",
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer first-token", "x-tenant": "first" }),
      }),
    );
    expect(fetchMock).toHaveBeenCalledWith(
      "https://second.example.com/mcp-rest/tools/list",
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer second-token", "x-tenant": "second" }),
      }),
    );
  });

  it("shares in-flight MCP discovery between default-provider refreshes", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    let resolveCatalog: (response: Response) => void = () => {};
    const catalog = new Promise<Response>((resolve) => {
      resolveCatalog = resolve;
    });
    let resolveSecondModel: (response: Response) => void = () => {};
    const secondModel = new Promise<Response>((resolve) => {
      resolveSecondModel = resolve;
    });
    let modelInfoRequests = 0;
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        modelInfoRequests++;
        return modelInfoRequests === 1 ? jsonResponse(200, { data: [] }) : secondModel;
      }
      if (url === "https://proxy.example.com/mcp-rest/tools/list") return catalog;
      throw new Error(`unexpected URL: ${url}`);
    });

    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    // Activation seeds the catalog; count only the refreshes this test drives.
    modelInfoRequests = 0;
    fetchMock.mockClear();

    const firstRefresh = refreshProvider(pi);
    await vi.waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith("https://proxy.example.com/mcp-rest/tools/list", expect.anything()),
    );
    const secondRefresh = refreshProvider(pi);
    await vi.waitFor(() => expect(modelInfoRequests).toBe(2));
    resolveSecondModel(jsonResponse(200, { data: [] }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(
      fetchMock.mock.calls.filter(([url]) => url === "https://proxy.example.com/mcp-rest/tools/list"),
    ).toHaveLength(1);
    resolveCatalog(jsonResponse(200, []));
    await Promise.all([firstRefresh, secondRefresh]);
  });

  it("continues a different-identity background refresh after the active refresh aborts", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    const firstAbort = new AbortController();
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return jsonResponse(200, { data: [] });
      if (url === "https://first.example.com/mcp-rest/tools/list") {
        return new Promise<Response>((_resolve, reject) => {
          const signal = init?.signal;
          signal?.addEventListener("abort", () => reject(signal.reason), { once: true });
        });
      }
      if (url === "https://second.example.com/mcp-rest/tools/list") {
        return jsonResponse(200, {
          tools: [
            {
              name: "second",
              inputSchema: { type: "object", properties: {} },
              mcp_info: { server_name: "second" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    process.env.LITELLM_BASE_URL = "https://first.example.com";
    process.env.LITELLM_API_KEY = "first-token";
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    // Activation seeds the catalog; count only the refreshes this test drives.
    fetchMock.mockClear();
    const firstRefresh = refreshProvider(pi, true, firstAbort.signal);
    await vi.waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith("https://first.example.com/mcp-rest/tools/list", expect.anything()),
    );
    await expect(firstRefresh).resolves.toBeUndefined();

    process.env.LITELLM_BASE_URL = "https://second.example.com";
    process.env.LITELLM_API_KEY = "second-token";
    const secondRefresh = refreshProvider(pi);
    await vi.waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith("https://second.example.com/model/info", expect.anything()),
    );
    firstAbort.abort(new Error("first refresh cancelled"));

    await expect(secondRefresh).resolves.toBeUndefined();
    await vi.waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith("https://second.example.com/mcp-rest/tools/list", expect.anything()),
    );
    await vi.waitFor(() => expect(pi.tools.map((tool) => tool.name)).toContain("mcp_second_second"));
  });

  it("does not block model refresh on an aborted same-identity MCP waiter", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    const secondAbort = new AbortController();
    let resolveCatalog: (response: Response) => void = () => {};
    const catalog = new Promise<Response>((resolve) => {
      resolveCatalog = resolve;
    });
    let modelInfoRequests = 0;
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        modelInfoRequests++;
        return jsonResponse(200, { data: [] });
      }
      if (url === "https://proxy.example.com/mcp-rest/tools/list") return catalog;
      throw new Error(`unexpected URL: ${url}`);
    });

    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    // Activation seeds the catalog; count only the refreshes this test drives.
    modelInfoRequests = 0;
    fetchMock.mockClear();
    const firstRefresh = refreshProvider(pi);
    await vi.waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith("https://proxy.example.com/mcp-rest/tools/list", expect.anything()),
    );
    const secondRefresh = refreshProvider(pi, true, secondAbort.signal);
    await vi.waitFor(() => expect(modelInfoRequests).toBe(2));

    const reason = new Error("second refresh cancelled");
    secondAbort.abort(reason);
    try {
      await expect(secondRefresh).resolves.toBeUndefined();
    } finally {
      resolveCatalog(jsonResponse(200, []));
      await firstRefresh;
      await secondRefresh.catch(() => undefined);
    }
  });

  it("uses fresh Pi auth when a discovered MCP tool executes", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://cached.example.com";
    process.env.LITELLM_API_KEY = "cached-token";

    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return jsonResponse(200, { data: [] });
      if (url === "https://cached.example.com/mcp-rest/tools/list") {
        return jsonResponse(200, {
          tools: [
            {
              name: "search",
              description: "Search the web",
              inputSchema: { type: "object", properties: { query: { type: "string" } }, required: ["query"] },
              mcp_info: { server_name: "brave", server_id: "brave-api" },
            },
          ],
        });
      }
      if (url === "https://fresh.example.com/mcp-rest/tools/call") return jsonResponse(200, { result: "fresh" });
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    await refreshProvider(pi);
    await vi.waitFor(() => expect(pi.tools.map((candidate) => candidate.name)).toContain("mcp_brave_search"));
    const tool = pi.tools.find((candidate) => candidate.name === "mcp_brave_search");

    await tool?.execute?.("call-1", { query: "Pi" }, undefined, undefined, {
      modelRegistry: {
        getProviderAuth: async () => ({
          auth: {
            apiKey: "fresh-token",
            headers: { "x-tenant": "fresh" },
          },
          env: { LITELLM_BASE_URL: "https://fresh.example.com/v1" },
        }),
        getProvider: () => pi.providers[0],
      },
    });

    expect(fetchMock).toHaveBeenLastCalledWith(
      "https://fresh.example.com/mcp-rest/tools/call",
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer fresh-token", "x-tenant": "fresh" }),
      }),
    );
  });

  it("uses fresh Pi auth when a registered Skills tool executes", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://cached.example.com";
    process.env.LITELLM_API_KEY = "cached-token";

    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return jsonResponse(200, { data: [] });
      if (url === "https://cached.example.com/mcp-rest/tools/list") return jsonResponse(200, []);
      if (url.startsWith("https://fresh.example.com/")) return jsonResponse(200, []);
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    await refreshProvider(pi);
    const tool = pi.tools.find((candidate) => candidate.name === "litellm_skill_list");

    await tool?.execute?.("call-1", {}, undefined, undefined, {
      modelRegistry: {
        getProviderAuth: async () => ({
          auth: {
            apiKey: "fresh-token",
            headers: { "x-tenant": "fresh" },
          },
          env: { LITELLM_BASE_URL: "https://fresh.example.com/v1" },
        }),
        getProvider: () => pi.providers[0],
      },
    });

    expect(fetchMock).toHaveBeenCalledWith(
      "https://fresh.example.com/claude-code/marketplace.json",
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer fresh-token", "x-tenant": "fresh" }),
      }),
    );
  });

  it("injects enabled LiteLLM skills into the system prompt", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return jsonResponse(200, { data: [] });
      if (url.endsWith("/mcp-rest/tools/list")) return jsonResponse(200, []);
      if (url.endsWith("/claude-code/marketplace.json")) return jsonResponse(404, {});
      if (url.endsWith("/v1/skills")) {
        return jsonResponse(200, {
          data: [{ id: "skill-1", name: "terraform", description: "Terraform conventions", enabled: true }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const beforeAgentStart = pi.handlers.get("before_agent_start")?.[0];
    const result = await beforeAgentStart?.(
      { systemPrompt: "Base prompt" },
      {
        modelRegistry: {
          getProviderAuth: async () => ({
            auth: { apiKey: "sk-test" },
            env: { LITELLM_BASE_URL: "https://skills.example.com/v1" },
          }),
          getProvider: () => pi.providers[0],
        },
      },
    );

    expect(result.systemPrompt).toContain("Base prompt");
    expect(result.systemPrompt).toContain("<litellm_skills>");
    expect(result.systemPrompt).toContain("Terraform conventions");
  });

  it("clears cached Skills auth when Pi reports revoked credentials", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, []));
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const beforeAgentStart = pi.handlers.get("before_agent_start")?.[0];

    await beforeAgentStart?.(
      { systemPrompt: "Base prompt" },
      {
        modelRegistry: {
          getProviderAuth: async () => ({
            auth: { apiKey: "active-key" },
            env: { LITELLM_BASE_URL: "https://active.example.com/v1" },
          }),
          getProvider: () => pi.providers[0],
        },
      },
    );
    vi.mocked(globalThis.fetch).mockClear();
    await beforeAgentStart?.(
      { systemPrompt: "Base prompt" },
      {
        modelRegistry: {
          getProviderAuth: async () => undefined,
          getProvider: () => pi.providers[0],
        },
      },
    );

    const listTool = pi.tools.find((tool) => tool.name === "litellm_skill_list");
    await expect(listTool?.execute?.("call-1", {}, undefined, undefined, {})).rejects.toThrow(
      "no credentials for litellm",
    );
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  it("disables LiteLLM skills through settings", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({ litellm: { skills: { enabled: false } } }),
      "utf8",
    );
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";
    const requestedUrls: string[] = [];
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      requestedUrls.push(String(input));
      return jsonResponse(200, { data: [] });
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    fetchMock.mockClear();

    expect(pi.tools.map((tool) => tool.name)).not.toContain("litellm_skill_list");
    const beforeAgentStart = pi.handlers.get("before_agent_start")?.[0];
    await expect(beforeAgentStart?.({ systemPrompt: "Base prompt" }, {})).resolves.toBeUndefined();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(requestedUrls.some((url) => url.includes("/skills"))).toBe(false);
  });

  it("disables LiteLLM MCP discovery through settings", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    await writeFile(join(agentDir, "settings.json"), JSON.stringify({ litellm: { mcp: { enabled: false } } }), "utf8");
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    const requestedUrls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      requestedUrls.push(url);
      if (url.endsWith("/model/info")) return jsonResponse(200, { data: [] });
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    await refreshProvider(pi);

    // Activation seeds the catalog, then the refresh discovers again; neither may touch MCP.
    expect(new Set(requestedUrls)).toEqual(new Set(["https://proxy.example.com/model/info"]));
    expect(pi.tools.map((tool) => tool.name)).toContain("litellm_skill_list");
    expect(pi.tools.some((tool) => tool.name.startsWith("mcp_"))).toBe(false);
  });

  it("registers cost tracking and session grouping handlers", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "anthropic/claude-3-5-sonnet",
              model_info: {
                mode: "chat",
                input_cost_per_token: 0.000003,
                output_cost_per_token: 0.000015,
                cache_read_input_token_cost: 0.0000003,
                cache_creation_input_token_cost: 0.00000375,
              },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(pi.handlers.has("before_provider_request")).toBe(true);
    expect(pi.handlers.has("after_provider_response")).toBe(true);
    expect(pi.handlers.has("message_end")).toBe(true);
  });

  it("does not inject LiteLLM session ids into non-LiteLLM provider requests", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "anthropic/claude-3-5-sonnet",
              model_info: {
                mode: "chat",
                input_cost_per_token: 0.000003,
                output_cost_per_token: 0.000015,
                cache_read_input_token_cost: 0.0000003,
                cache_creation_input_token_cost: 0.00000375,
              },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const sessionStartHandlers = pi.handlers.get("session_start") ?? [];
    for (const handler of sessionStartHandlers) {
      await handler(
        { reason: "reload" },
        {
          sessionManager: {
            getSessionFile: () => join(agentDir, "2026-05-11T16-00-00-000Z_123e4567-e89b-12d3-a456-426614174000.jsonl"),
          },
        },
      );
    }

    const beforeRequest = pi.handlers.get("before_provider_request")?.[0];
    const updated = beforeRequest?.(
      { payload: { messages: [] } },
      { model: { provider: "openai-codex", id: "gpt-5.5" } },
    );
    expect(updated).toBeUndefined();
  });

  it("injects LiteLLM session ids into LiteLLM provider requests", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "anthropic/claude-3-5-sonnet",
              model_info: {
                mode: "chat",
                input_cost_per_token: 0.000003,
                output_cost_per_token: 0.000015,
                cache_read_input_token_cost: 0.0000003,
                cache_creation_input_token_cost: 0.00000375,
              },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const sessionStartHandlers = pi.handlers.get("session_start") ?? [];
    for (const handler of sessionStartHandlers) {
      await handler(
        { reason: "reload" },
        {
          sessionManager: {
            getSessionFile: () => join(agentDir, "2026-05-11T16-00-00-000Z_123e4567-e89b-12d3-a456-426614174000.jsonl"),
          },
        },
      );
    }

    const beforeRequest = pi.handlers.get("before_provider_request")?.[0];
    const updated = beforeRequest?.(
      { payload: { messages: [] } },
      { model: { provider: "litellm", id: "kimi-k2.6", suppressReasoningContent: true } },
    );
    expect(updated).toMatchObject({
      messages: [],
      include_reasoning: false,
      reasoning_content: false,
      merge_reasoning_content_in_choices: true,
      litellm_session_id: "123e4567-e89b-12d3-a456-426614174000",
    });
  });

  it("does not send thinking for Kimi OpenAI completions requests", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "kimi-k3",
              model_info: { mode: "chat" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const beforeRequest = pi.handlers.get("before_provider_request")?.[0];
    const updated = beforeRequest?.(
      { payload: { messages: [] } },
      {
        model: {
          provider: "litellm",
          id: "kimi-k3",
          api: "openai-completions",
          suppressReasoningContent: true,
        },
      },
    );
    expect(updated).toEqual({
      messages: [],
      include_reasoning: false,
      reasoning_content: false,
      merge_reasoning_content_in_choices: true,
    });
  });

  it("uses backend suppression evidence instead of a public route alias", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { data: [] }));

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const beforeRequest = pi.handlers.get("before_provider_request")?.[0];
    const suppressed = beforeRequest?.(
      { payload: { messages: [] } },
      {
        model: {
          provider: "litellm",
          id: "neutral-route",
          api: "openai-completions",
          suppressReasoningContent: true,
        },
      },
    );
    expect(suppressed).toMatchObject({
      include_reasoning: false,
      reasoning_content: false,
      merge_reasoning_content_in_choices: true,
    });

    const untouched = beforeRequest?.(
      { payload: { messages: [] } },
      { model: { provider: "litellm", id: "kimi-looking-route", api: "openai-completions" } },
    );
    expect(untouched).toBeUndefined();
  });

  it("leaves synthetic Messages payloads unchanged", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { data: [] }));

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const payload = {
      messages: [{ role: "user", content: "hi" }],
      reasoning_effort: "HIGH",
      thinking: { type: "enabled", budget_tokens: 1024 },
    };
    const beforeRequest = pi.handlers.get("before_provider_request")?.[0];
    const updated = beforeRequest?.(
      { payload },
      {
        model: {
          provider: "litellm",
          id: "kimi-k2.6",
          api: "anthropic-messages",
          suppressReasoningContent: true,
        },
      },
    );

    expect(updated).toBeUndefined();
    expect(payload).toEqual({
      messages: [{ role: "user", content: "hi" }],
      reasoning_effort: "HIGH",
      thinking: { type: "enabled", budget_tokens: 1024 },
    });
  });

  it("leaves Kimi Responses requests unchanged", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { data: [] }));

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const beforeRequest = pi.handlers.get("before_provider_request")?.[0];
    const updated = beforeRequest?.(
      { payload: { input: [{ type: "message", role: "user", content: "hi" }] } },
      { model: { provider: "litellm", id: "kimi-k2.6", api: "openai-responses" } },
    );

    expect(updated).toBeUndefined();
  });

  it("fills empty assistant content on Kimi tool calls", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { data: [] }));

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const beforeRequest = pi.handlers.get("before_provider_request")?.[0];
    const updated = beforeRequest?.(
      {
        payload: {
          messages: [
            { role: "user", content: "hi" },
            {
              role: "assistant",
              content: null,
              tool_calls: [{ id: "call_1", type: "function", function: { name: "noop", arguments: "{}" } }],
            },
            { role: "tool", tool_call_id: "call_1", content: "tool output" },
          ],
        },
      },
      { model: { provider: "litellm", id: "kimi-k3", suppressReasoningContent: true } },
    );

    expect(updated).toEqual({
      messages: [
        { role: "user", content: "hi" },
        {
          role: "assistant",
          content: "",
          tool_calls: [{ id: "call_1", type: "function", function: { name: "noop", arguments: "{}" } }],
        },
        { role: "tool", tool_call_id: "call_1", content: "tool output" },
      ],
      include_reasoning: false,
      reasoning_content: false,
      merge_reasoning_content_in_choices: true,
    });
  });

  it("preserves image blocks after Kimi tool results", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { data: [] }));

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const beforeRequest = pi.handlers.get("before_provider_request")?.[0];
    const updated = beforeRequest?.(
      {
        payload: {
          messages: [
            {
              role: "tool",
              tool_call_id: "call_1",
              content: [{ type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } }],
            },
            {
              role: "tool",
              tool_call_id: "call_2",
              content: [{ type: "text", text: "second output" }],
            },
            { role: "tool", tool_call_id: "call_3", content: [] },
          ],
        },
      },
      { model: { provider: "litellm", id: "kimi-k3", suppressReasoningContent: true } },
    );

    expect(updated).toEqual({
      messages: [
        { role: "tool", tool_call_id: "call_1", content: "(see attached image)" },
        { role: "tool", tool_call_id: "call_2", content: "second output" },
        { role: "tool", tool_call_id: "call_3", content: "(no tool output)" },
        {
          role: "user",
          content: [
            { type: "text", text: "Attached image(s) from tool result:" },
            { type: "image_url", image_url: { url: "data:image/png;base64,AAAA" } },
          ],
        },
      ],
      include_reasoning: false,
      reasoning_content: false,
      merge_reasoning_content_in_choices: true,
    });
  });

  it("leaves well-formed Kimi tool messages untouched", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { data: [] }));

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const messages = [
      {
        role: "assistant",
        content: "calling the tool",
        tool_calls: [{ id: "call_1", type: "function", function: { name: "noop", arguments: "{}" } }],
      },
      { role: "tool", tool_call_id: "call_1", content: "tool output" },
    ];
    const beforeRequest = pi.handlers.get("before_provider_request")?.[0];
    const updated = beforeRequest?.(
      { payload: { messages } },
      { model: { provider: "litellm", id: "kimi-k3", suppressReasoningContent: true } },
    );

    expect(updated?.messages).toBe(messages);
  });

  it("leaves strict-schema tool messages untouched for non-Moonshot models", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { data: [] }));

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const beforeRequest = pi.handlers.get("before_provider_request")?.[0];
    const updated = beforeRequest?.(
      {
        payload: {
          messages: [
            {
              role: "assistant",
              content: null,
              tool_calls: [{ id: "call_1", type: "function", function: { name: "noop", arguments: "{}" } }],
            },
            { role: "tool", tool_call_id: "call_1", content: [{ type: "text", text: "tool output" }] },
          ],
        },
      },
      { model: { provider: "litellm", id: "anthropic/claude-3-5-sonnet" } },
    );

    expect(updated).toBeUndefined();
  });

  it("drops reasoning fields for llm-gateway/gpt-5.5 tool requests", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "llm-gateway/gpt-5.5",
              model_info: { mode: "chat" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const beforeRequest = pi.handlers.get("before_provider_request")?.[0];
    const updated = beforeRequest?.(
      {
        payload: {
          input: [
            { type: "reasoning", id: "rs_1", encrypted_content: "opaque" },
            { type: "message", role: "user", content: "hi" },
          ],
          tools: [{ type: "function", function: { name: "noop", parameters: { type: "object" } } }],
          reasoning: { effort: "high", summary: "auto" },
          reasoning_effort: "high",
          include: ["reasoning.encrypted_content", "other"],
          include_reasoning: true,
          reasoning_content: true,
          merge_reasoning_content_in_choices: false,
          thinking: { type: "enabled" },
        },
      },
      { model: { provider: "litellm", id: "llm-gateway/gpt-5.5" } },
    );
    expect(updated).toEqual({
      input: [{ type: "message", role: "user", content: "hi" }],
      tools: [{ type: "function", function: { name: "noop", parameters: { type: "object" } } }],
      include: ["other"],
    });
  });

  it("leaves gpt-5.5 tool requests without reasoning fields unchanged", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "llm-gateway/gpt-5.5",
              model_info: { mode: "chat" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const beforeRequest = pi.handlers.get("before_provider_request")?.[0];
    const updated = beforeRequest?.(
      {
        payload: {
          input: [{ type: "message", role: "user", content: "hi" }],
          tools: [{ type: "function", function: { name: "noop", parameters: { type: "object" } } }],
          include: ["other"],
        },
      },
      { model: { provider: "litellm", id: "llm-gateway/gpt-5.5" } },
    );
    expect(updated).toBeUndefined();
  });

  it("keeps reasoning fields for gpt-5.5 Responses tool requests", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { data: [] }));

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const beforeRequest = pi.handlers.get("before_provider_request")?.[0];
    const updated = beforeRequest?.(
      {
        payload: {
          input: [{ type: "reasoning", id: "rs_1", encrypted_content: "opaque" }],
          tools: [{ type: "function", function: { name: "noop", parameters: { type: "object" } } }],
          reasoning: { effort: "high", summary: "auto" },
          reasoning_effort: "high",
        },
      },
      { model: { provider: "litellm", id: "llm-gateway/gpt-5.5", api: "openai-responses" } },
    );

    expect(updated).toBeUndefined();
  });

  it("drops reasoning fields for gpt-5.5 route aliases", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "gpt-5.5-20260504143601",
              model_info: { mode: "chat" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const beforeRequest = pi.handlers.get("before_provider_request")?.[0];
    for (const id of ["gpt-5.5", "openai/gpt-5.5", "gpt-5.5-20260504143601"]) {
      const updated = beforeRequest?.(
        {
          payload: {
            messages: [],
            tools: [{ type: "function", function: { name: "noop", parameters: { type: "object" } } }],
            reasoning: { effort: "high" },
          },
        },
        { model: { provider: "litellm", id } },
      );
      expect(updated, id).toEqual({
        messages: [],
        tools: [{ type: "function", function: { name: "noop", parameters: { type: "object" } } }],
      });
    }
  });

  it("normalizes capitalized reasoning effort values for Gemini models only", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { data: [] }));

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const beforeRequest = pi.handlers.get("before_provider_request")?.[0];
    const updated = beforeRequest?.(
      {
        payload: {
          messages: [{ role: "user", content: "hello" }],
          reasoning_effort: "HIGH",
          reasoning: { effort: "HIGH", summary: "auto" },
        },
      },
      { model: { provider: "litellm", id: "google/gemini-3.1-pro-preview" } },
    );

    expect(updated).toEqual({
      messages: [{ role: "user", content: "hello" }],
      reasoning_effort: "high",
      reasoning: { effort: "high", summary: "auto" },
    });

    expect(
      beforeRequest?.(
        {
          payload: {
            messages: [{ role: "user", content: "hello" }],
            reasoning_effort: "MAX_THINKING",
            reasoning: { effort: "MAX_THINKING", summary: "auto" },
          },
        },
        { model: { provider: "litellm", id: "custom/reasoner" } },
      ),
    ).toBeUndefined();
  });

  it("normalizes Kimi think tags into Pi thinking blocks", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "kimi-k2.6",
              model_info: { mode: "chat" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    let message: any = {
      role: "assistant",
      provider: "litellm",
      model: "kimi-k2.6",
      content: [{ type: "text", text: "<think>internal reasoning</think>DONE" }],
      usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0 },
    };
    for (const handler of pi.handlers.get("message_end") ?? []) {
      const result = await handler({ message });
      if (result?.message) message = result.message;
    }

    expect(message.content).toEqual([
      { type: "thinking", thinking: "internal reasoning" },
      { type: "text", text: "DONE" },
    ]);
  });

  it("keeps final Kimi text visible when a dangling think tag prefixes it", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "kimi-k2.6",
              model_info: { mode: "chat" },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    let message: any = {
      role: "assistant",
      provider: "litellm",
      model: "kimi-k2.6",
      content: [{ type: "text", text: "<think>DONE" }],
      usage: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0 },
    };
    for (const handler of pi.handlers.get("message_end") ?? []) {
      const result = await handler({ message });
      if (result?.message) message = result.message;
    }

    expect(message.content).toEqual([{ type: "text", text: "DONE" }]);
  });

  it("overrides assistant cost from LiteLLM response metadata", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "anthropic/claude-3-5-sonnet",
              model_info: {
                mode: "chat",
                input_cost_per_token: 0.000003,
                output_cost_per_token: 0.000015,
                cache_read_input_token_cost: 0.0000003,
                cache_creation_input_token_cost: 0.00000375,
              },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const sessionStartHandlers = pi.handlers.get("session_start") ?? [];
    for (const handler of sessionStartHandlers) {
      await handler({ reason: "start" }, { sessionManager: { getSessionFile: () => undefined } });
    }

    const responseHandler = pi.handlers.get("after_provider_response")?.[0];
    responseHandler?.(
      { headers: { "x-litellm-response-cost": "0.42" } },
      { model: { provider: "litellm", id: "anthropic/claude-3-5-sonnet" } },
    );

    const endHandler = pi.handlers.get("message_end")?.[0];
    const result = await endHandler?.({
      message: {
        role: "assistant",
        provider: "litellm",
        model: "anthropic/claude-3-5-sonnet",
        usage: { input: 100, output: 50, cacheRead: 10, cacheWrite: 5 },
      },
    });

    expect(result).toMatchObject({
      message: {
        usage: {
          cost: {
            total: 0.42,
          },
        },
      },
    });
  });

  it("does not apply LiteLLM model costs to other providers' messages", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "anthropic/claude-3-5-sonnet",
              model_info: {
                mode: "chat",
                input_cost_per_token: 0.000003,
                output_cost_per_token: 0.000015,
              },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    // Same model id discovered through LiteLLM, but the message came from
    // a direct provider — LiteLLM pricing must not overwrite its cost.
    const endHandler = pi.handlers.get("message_end")?.[0];
    const result = await endHandler?.({
      message: {
        role: "assistant",
        provider: "anthropic",
        model: "anthropic/claude-3-5-sonnet",
        usage: { input: 100, output: 50, cacheRead: 10, cacheWrite: 5 },
      },
    });

    expect(result).toBeUndefined();
  });

  it("ignores LiteLLM cost headers captured from non-LiteLLM responses", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "anthropic/claude-3-5-sonnet",
              model_info: {
                mode: "chat",
                input_cost_per_token: 0.000003,
                output_cost_per_token: 0.000015,
                cache_read_input_token_cost: 0.0000003,
                cache_creation_input_token_cost: 0.00000375,
              },
            },
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    await refreshProvider(pi);

    const responseHandler = pi.handlers.get("after_provider_response")?.[0];
    responseHandler?.(
      { headers: { "x-litellm-response-cost": "0.42" } },
      { model: { provider: "openai-codex", id: "gpt-5.5" } },
    );

    const endHandler = pi.handlers.get("message_end")?.[0];
    const usage = {
      input: 100,
      output: 50,
      cacheRead: 10,
      cacheWrite: 5,
      cost: { input: 0.0003, output: 0.00075, cacheRead: 0.000003, cacheWrite: 0.00001875, total: 0.00107175 },
    };
    const result = await endHandler?.({
      message: {
        role: "assistant",
        provider: "litellm",
        model: "anthropic/claude-3-5-sonnet",
        usage,
      },
    });

    expect(result).toBeUndefined();
    expect(usage.cost.total).toBeCloseTo(0.00107175, 10);
  });
});
