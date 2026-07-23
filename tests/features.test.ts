import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ProviderModelsStore } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createPi, loadExtension, type TestPi } from "./test-helpers.js";

vi.unmock("@earendil-works/pi-coding-agent");

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

async function refreshProvider(pi: TestPi, allowNetwork = true, signal?: AbortSignal): Promise<void> {
  const store: ProviderModelsStore = {
    read: async () => undefined,
    write: async () => undefined,
    delete: async () => undefined,
  };
  await pi.providers[0]?.refreshModels?.({
    allowNetwork,
    store,
    credential: {
      type: "api_key",
      key: process.env.LITELLM_API_KEY ?? "sk-test",
      env: { LITELLM_BASE_URL: process.env.LITELLM_BASE_URL ?? "https://litellm.example.com" },
    },
    signal,
  });
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.resetModules();
  delete process.env.LITELLM_BASE_URL;
  delete process.env.LITELLM_API_KEY;
  delete process.env.LITELLM_HEADERS;
  delete process.env.LITELLM_DISCOVERY_TIMEOUT_MS;
  delete process.env.LITELLM_GCLOUD_TOKEN_AUTH;
  delete process.env.GOOGLE_APPLICATION_CREDENTIALS;
});

describe("feature parity", () => {
  it("registers a command-backed gcloud token provider key when ADC auth is enabled", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    const adcPath = join(agentDir, "adc.json");
    await writeFile(
      adcPath,
      JSON.stringify({
        type: "authorized_user",
        client_id: "client-id",
        client_secret: "client-secret",
        refresh_token: "refresh-token",
      }),
      "utf8",
    );
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_GCLOUD_TOKEN_AUTH = "1";
    process.env.GOOGLE_APPLICATION_CREDENTIALS = adcPath;
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "0";

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    await expect(
      pi.providers[0]?.auth.apiKey?.check?.({
        ctx: { env: async (name) => process.env[name], fileExists: async () => false },
      }),
    ).resolves.toEqual({ type: "api_key", source: "gcloud ADC" });
  });

  it("registers discovered LiteLLM MCP tools as Pi tools", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
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
    expect(pi.tools.map((tool) => tool.name)).toContain("mcp_second_second");
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
      if (url === "https://litellm.example.com/mcp-rest/tools/list") return catalog;
      throw new Error(`unexpected URL: ${url}`);
    });

    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "sk-test";
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    const firstRefresh = refreshProvider(pi);
    await vi.waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith("https://litellm.example.com/mcp-rest/tools/list", expect.anything()),
    );
    const secondRefresh = refreshProvider(pi);
    await vi.waitFor(() => expect(modelInfoRequests).toBe(2));
    resolveSecondModel(jsonResponse(200, { data: [] }));
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(
      fetchMock.mock.calls.filter(([url]) => url === "https://litellm.example.com/mcp-rest/tools/list"),
    ).toHaveLength(1);
    resolveCatalog(jsonResponse(200, []));
    await Promise.all([firstRefresh, secondRefresh]);
  });

  it("lets a live different-identity refresh continue after the active refresh aborts", async () => {
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
    const firstRefresh = refreshProvider(pi, true, firstAbort.signal);
    await vi.waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith("https://first.example.com/mcp-rest/tools/list", expect.anything()),
    );

    process.env.LITELLM_BASE_URL = "https://second.example.com";
    process.env.LITELLM_API_KEY = "second-token";
    const secondRefresh = refreshProvider(pi);
    await vi.waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith("https://second.example.com/model/info", expect.anything()),
    );
    firstAbort.abort(new Error("first refresh cancelled"));

    await expect(firstRefresh).rejects.toThrow("first refresh cancelled");
    await expect(secondRefresh).resolves.toBeUndefined();
    expect(fetchMock).toHaveBeenCalledWith("https://second.example.com/mcp-rest/tools/list", expect.anything());
    expect(pi.tools.map((tool) => tool.name)).toContain("mcp_second_second");
  });

  it("rejects an aborted same-identity MCP waiter with its own reason", async () => {
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
      if (url === "https://litellm.example.com/mcp-rest/tools/list") return catalog;
      throw new Error(`unexpected URL: ${url}`);
    });

    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "sk-test";
    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);
    const firstRefresh = refreshProvider(pi);
    await vi.waitFor(() =>
      expect(fetchMock).toHaveBeenCalledWith("https://litellm.example.com/mcp-rest/tools/list", expect.anything()),
    );
    const secondRefresh = refreshProvider(pi, true, secondAbort.signal);
    await vi.waitFor(() => expect(modelInfoRequests).toBe(2));

    const reason = new Error("second refresh cancelled");
    secondAbort.abort(reason);
    try {
      const outcome = await Promise.race([
        secondRefresh.then(
          () => "resolved",
          (error: unknown) => error,
        ),
        new Promise((resolve) => setTimeout(() => resolve("pending"), 50)),
      ]);
      expect(outcome).toBe(reason);
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
    const tool = pi.tools.find((candidate) => candidate.name === "mcp_brave_search");

    await tool?.execute?.("call-1", { query: "Pi" }, undefined, undefined, {
      modelRegistry: {
        getProviderAuth: async () => ({
          auth: {
            apiKey: "fresh-token",
            baseUrl: "https://fresh.example.com/v1",
            headers: { "x-tenant": "fresh" },
          },
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
            baseUrl: "https://fresh.example.com/v1",
            headers: { "x-tenant": "fresh" },
          },
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
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
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
            auth: { apiKey: "sk-test", baseUrl: "https://litellm.example.com/v1" },
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
            auth: { apiKey: "active-key", baseUrl: "https://active.example.com/v1" },
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
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "sk-test";
    const fetchMock = vi.spyOn(globalThis, "fetch");

    const extension = await loadExtension(agentDir);
    const pi = createPi();
    await extension(pi);

    expect(pi.tools.map((tool) => tool.name)).not.toContain("litellm_skill_list");
    const beforeAgentStart = pi.handlers.get("before_agent_start")?.[0];
    await expect(beforeAgentStart?.({ systemPrompt: "Base prompt" }, {})).resolves.toBeUndefined();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("disables LiteLLM MCP discovery through settings", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    await writeFile(join(agentDir, "settings.json"), JSON.stringify({ litellm: { mcp: { enabled: false } } }), "utf8");
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
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

    expect(requestedUrls).toEqual(["https://litellm.example.com/model/info"]);
    expect(pi.tools.map((tool) => tool.name)).toContain("litellm_skill_list");
    expect(pi.tools.some((tool) => tool.name.startsWith("mcp_"))).toBe(false);
  });

  it("registers cost tracking and session grouping handlers", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
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
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
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
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
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
    const updated = beforeRequest?.({ payload: { messages: [] } }, { model: { provider: "litellm", id: "kimi-k2.6" } });
    expect(updated).toMatchObject({
      messages: [],
      include_reasoning: false,
      reasoning_content: false,
      merge_reasoning_content_in_choices: true,
      thinking: { type: "disabled" },
      litellm_session_id: "123e4567-e89b-12d3-a456-426614174000",
    });
  });

  it("suppresses separate Kimi reasoning streams before session ids are available", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
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

    const beforeRequest = pi.handlers.get("before_provider_request")?.[0];
    const updated = beforeRequest?.({ payload: { messages: [] } }, { model: { provider: "litellm", id: "kimi-k2.6" } });
    expect(updated).toEqual({
      messages: [],
      include_reasoning: false,
      reasoning_content: false,
      merge_reasoning_content_in_choices: true,
      thinking: { type: "disabled" },
    });
  });

  it("leaves Kimi Responses requests unchanged", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
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

  it("drops reasoning fields for llm-gateway/gpt-5.5 tool requests", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
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
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
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
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
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
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
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

  it("normalizes Kimi think tags into Pi thinking blocks", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
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
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
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
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
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
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
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
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
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

  it("shares concurrent litellm-refresh requests", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "sk-test";

    let releaseFetch!: () => void;
    const fetchGate = new Promise<void>((resolve) => {
      releaseFetch = resolve;
    });
    let callCount = 0;
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        callCount++;
        if (callCount > 1) throw new Error("overlapping discovery");
        await fetchGate;
        return jsonResponse(200, {
          data: [
            {
              model_name: "test-model",
              model_info: {
                mode: "chat",
                input_cost_per_token: 0.000006,
                output_cost_per_token: 0.00003,
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
    await refreshProvider(pi, false);

    const refreshCmd = pi.commands.get("litellm-refresh");
    const notifications: Array<{ message: string; type: string }> = [];
    const ctx = {
      ui: {
        notify: (message: string, type: string) => {
          notifications.push({ message, type });
        },
      },
    };

    const firstRefresh = refreshCmd!.handler("", ctx);
    await vi.waitFor(() => expect(callCount).toBe(1));
    const secondRefresh = refreshCmd!.handler("", ctx);
    releaseFetch();
    await Promise.all([firstRefresh, secondRefresh]);

    expect(callCount).toBe(1);
    expect(notifications.filter(({ message }) => message.endsWith("models refreshed (source: model_info)"))).toEqual([
      { message: "LiteLLM: 1 models refreshed (source: model_info)", type: "info" },
      { message: "LiteLLM: 1 models refreshed (source: model_info)", type: "info" },
    ]);
  });
});
