import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AuthInteraction } from "@earendil-works/pi-ai";
import { ModelRegistry, ModelRuntime } from "@earendil-works/pi-coding-agent";
import { afterEach, describe, expect, it, vi } from "vitest";
// Pi does not export its MCP header resolver; import it so the escaping is tested against the real one.
import { resolveHeadersOrThrow } from "../node_modules/@earendil-works/pi-coding-agent/dist/core/resolve-config-value.js";
import { createPi, loadExtension, type TestPi, useHermeticEnv } from "./test-helpers.js";

vi.unmock("@earendil-works/pi-coding-agent");

useHermeticEnv();

afterEach(() => {
  vi.restoreAllMocks();
  vi.resetModules();
});

// Without `mcp`, the MCP access check fails, which registers the server enabled.
function mockProxy(mcp?: (request: Request) => Response | Promise<Response>): void {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const url = String(input);
    if (url.endsWith("/model/info")) return Response.json({ data: [] });
    if (mcp && url.endsWith("/mcp")) return mcp(new Request(url, init));
    throw new Error(`unexpected URL: ${url}`);
  });
}

// LiteLLM 1.102.0's answer to `initialize` for a key with no MCP servers granted.
function noServersGranted(): Response {
  const error =
    "The key has no MCP servers granted, or none of its granted servers is loaded and allowed for this client IP. " +
    "Grant servers or access groups to the key, its team, or its organization (object_permission.mcp_servers), " +
    "check the server's allowed IPs, and reconnect.";
  return Response.json({ detail: { error } }, { status: 403 });
}

async function makeAgentDir(litellm: Record<string, unknown> = {}): Promise<string> {
  const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-mcp-"));
  const settings = { skills: { enabled: false }, ...litellm };
  await writeFile(join(agentDir, "settings.json"), JSON.stringify({ litellm: settings }), "utf8");
  return agentDir;
}

async function load(agentDir: string, pi: TestPi = createPi(), piVersion?: string): Promise<TestPi> {
  await (await loadExtension(agentDir, { piVersion }))(pi);
  return pi;
}

async function startTurn(pi: TestPi): Promise<void> {
  for (const handler of pi.handlers.get("before_agent_start") ?? []) await handler({ systemPrompt: "" }, {});
}

function stderrText(stderr: { mock: { calls: unknown[][] } }): string {
  return stderr.mock.calls.map(([message]) => String(message)).join("");
}

describe("LiteLLM MCP server registration", () => {
  it("registers each provider's /mcp endpoint, authenticated by the provider itself", async () => {
    mockProxy();
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-default";
    const agentDir = await makeAgentDir({
      providers: {
        team: { baseUrl: "https://team.example.com/root", apiKey: "sk-team", headers: { "x-team": "blue" } },
      },
    });

    const pi = await load(agentDir);

    expect(Object.fromEntries(pi.mcpServers)).toEqual({
      litellm: { url: "https://proxy.example.com/mcp", auth: { provider: "litellm" } },
      team: { url: "https://team.example.com/root/mcp", auth: { provider: "team" }, headers: { "x-team": "blue" } },
    });
    expect(JSON.stringify(Object.fromEntries(pi.mcpServers))).not.toContain("sk-");
  });

  it("passes the exposure settings to Pi unchanged", async () => {
    mockProxy();
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-default";
    const toolExposure = { "limble-get_*": "direct", "*delete*": "hidden" };
    const agentDir = await makeAgentDir({ mcp: { exposure: "deferred", toolExposure } });

    const pi = await load(agentDir);

    expect(pi.mcpServers.get("litellm")).toMatchObject({ exposure: "deferred", toolExposure });
  });

  it("lets Pi resolve the credential that auth.provider sends", async () => {
    mockProxy();
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-default";
    const agentDir = await makeAgentDir();
    const runtime = await ModelRuntime.create({
      authPath: join(agentDir, "auth.json"),
      modelsPath: join(agentDir, "models.json"),
    });
    const pi = createPi();
    pi.registerProvider = (provider) => runtime.registerNativeProvider(provider);

    await load(agentDir, pi);

    // Pi's MCP client reads this on every request to the server registered with `auth.provider`.
    expect(await new ModelRegistry(runtime).getApiKeyForProvider("litellm")).toBe("sk-default");
  });

  it("registers nothing when MCP is disabled, and keeps Skills", async () => {
    mockProxy();
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-default";
    const pi = await load(await makeAgentDir({ skills: { enabled: true }, mcp: { enabled: false } }));
    await startTurn(pi);

    expect(pi.mcpServers.size).toBe(0);
    expect(pi.tools.map((tool) => tool.name)).toContain("litellm_skill_list");
  });

  it.each([
    ["LITELLM_OFFLINE", "1"],
    ["LITELLM_DISCOVERY_TIMEOUT_MS", "0"],
    ["PI_OFFLINE", "1"],
    ["PI_OFFLINE", "0"],
    ["PI_OFFLINE", ""],
  ])("registers nothing when %s=%j disables network access", async (name, value) => {
    mockProxy();
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-default";
    process.env[name] = value;
    const pi = await load(await makeAgentDir());
    await startTurn(pi);

    expect(pi.mcpServers.size).toBe(0);
  });

  it("follows a login that supplies the proxy root and a logout that removes it", async () => {
    mockProxy();
    const agentDir = await makeAgentDir();
    const pi = await load(agentDir);
    expect(pi.mcpServers.size).toBe(0);

    await writeFile(
      join(agentDir, "auth.json"),
      JSON.stringify({
        litellm: { type: "api_key", key: "sk-login", env: { LITELLM_BASE_URL: "https://login.example.com" } },
      }),
      "utf8",
    );
    await startTurn(pi);
    expect(pi.mcpServers.get("litellm")).toEqual({
      url: "https://login.example.com/mcp",
      auth: { provider: "litellm" },
    });

    await rm(join(agentDir, "auth.json"));
    await startTurn(pi);
    expect(pi.mcpServers.size).toBe(0);
  });

  it("registers again only when the registration changes", async () => {
    mockProxy();
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-default";
    const pi = createPi();
    const register = vi.spyOn(pi, "registerMcpServer");
    await load(await makeAgentDir(), pi);
    await startTurn(pi);
    await startTurn(pi);
    expect(register).toHaveBeenCalledTimes(1);

    process.env.LITELLM_BASE_URL = "https://moved.example.com";
    await startTurn(pi);
    expect(register).toHaveBeenCalledTimes(2);
    expect(pi.mcpServers.get("litellm")).toMatchObject({ url: "https://moved.example.com/mcp" });
  });

  it("reports a registration Pi refuses once, without retrying it every turn", async () => {
    mockProxy();
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-default";
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const pi = createPi();
    const register = vi.spyOn(pi, "registerMcpServer").mockImplementation(() => {
      throw new Error('server "litellm": exposure must be one of "codemode", "deferred", "direct", "hidden"');
    });
    await load(await makeAgentDir({ mcp: { exposure: "everywhere" } }), pi);
    await startTurn(pi);

    expect(register).toHaveBeenCalledTimes(1);
    expect(stderrText(stderr).match(/LiteLLM MCP \("litellm"\): server "litellm": exposure must be/g)).toHaveLength(1);
  });

  it("holds MCP notices until the session UI can show them", async () => {
    mockProxy();
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-default";
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const isTTY = Object.getOwnPropertyDescriptor(process.stderr, "isTTY");
    Object.defineProperty(process.stderr, "isTTY", { configurable: true, value: true });
    try {
      const pi = createPi();
      vi.spyOn(pi, "registerMcpServer").mockImplementation(() => {
        throw new Error("refused");
      });
      await load(await makeAgentDir(), pi);
      expect(stderr).not.toHaveBeenCalled();

      const notify = vi.fn();
      for (const handler of pi.handlers.get("session_start") ?? [])
        await handler({ type: "session_start" }, { hasUI: true, ui: { notify } });

      expect(notify).toHaveBeenCalledWith('LiteLLM MCP ("litellm"): refused', "warning");
      expect(stderr).not.toHaveBeenCalled();
    } finally {
      if (isTTY) Object.defineProperty(process.stderr, "isTTY", isTTY);
      else Reflect.deleteProperty(process.stderr, "isTTY");
    }
  });

  it("keeps resolved header values literal through Pi's header resolver", async () => {
    mockProxy();
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-default";
    process.env.LITELLM_HEADERS = JSON.stringify({ "x-price": "$$HOME", "x-token": "$!echo pwned", "x-plain": "tok" });

    const pi = await load(await makeAgentDir());
    const headers = pi.mcpServers.get("litellm")?.headers as Record<string, string>;

    expect(resolveHeadersOrThrow(headers, "test")).toEqual({
      "x-price": "$HOME",
      "x-token": "!echo pwned",
      "x-plain": "tok",
    });
  });

  it("disconnects when a login starts, before it can store a new credential", async () => {
    mockProxy();
    process.env.LITELLM_BASE_URL = "https://old.example.com";
    process.env.LITELLM_API_KEY = "sk-old";
    const agentDir = await makeAgentDir();
    const pi = await load(agentDir);
    expect(pi.mcpServers.has("litellm")).toBe(true);

    const registeredWhenPrompted: boolean[] = [];
    const answers: Record<string, string> = { text: "https://new.example.com", secret: "sk-new" };
    const prompt = vi.fn(async ({ type, options }: { type: string; options?: Array<{ id: string }> }) => {
      registeredWhenPrompted.push(pi.mcpServers.has("litellm"));
      // The select offers LITELLM_BASE_URL first and "enter a different URL" second.
      return type === "select" ? options![1]!.id : answers[type]!;
    });
    const credential = await pi.providers[0]?.auth.apiKey?.login?.({
      prompt,
      notify: vi.fn(),
      signal: new AbortController().signal,
    } as unknown as AuthInteraction & { signal: AbortSignal });
    expect(registeredWhenPrompted).toEqual([false, false, false]);

    await writeFile(join(agentDir, "auth.json"), JSON.stringify({ litellm: credential }), "utf8");
    await startTurn(pi);
    expect(pi.mcpServers.get("litellm")).toMatchObject({ url: "https://new.example.com/mcp" });
  });

  it("withdraws the previous server when Pi refuses its replacement", async () => {
    mockProxy();
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-default";
    vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const pi = createPi();
    const register = pi.registerMcpServer.bind(pi);
    vi.spyOn(pi, "registerMcpServer").mockImplementation((name, config) => {
      if (String(config.url).startsWith("https://moved.")) throw new Error("refused");
      register(name, config);
    });
    await load(await makeAgentDir(), pi);
    expect(pi.mcpServers.has("litellm")).toBe(true);

    process.env.LITELLM_BASE_URL = "https://moved.example.com";
    await startTurn(pi);
    await startTurn(pi);

    expect(pi.mcpServers.has("litellm")).toBe(false);
  });

  it("registers the server disabled, without a notice, when the proxy refuses the key at initialize", async () => {
    const checks: Request[] = [];
    mockProxy((request) => {
      checks.push(request);
      return noServersGranted();
    });
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-default";
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const pi = await load(await makeAgentDir());
    await startTurn(pi);
    await startTurn(pi);

    expect(pi.mcpServers.get("litellm")).toEqual({
      url: "https://proxy.example.com/mcp",
      auth: { provider: "litellm" },
      enabled: false,
    });
    expect(checks).toHaveLength(1);
    expect(stderr).not.toHaveBeenCalled();
  });

  it("checks with the credential and headers Pi sends, and ends the session the check opened", async () => {
    const requests: Request[] = [];
    mockProxy((request) => {
      requests.push(request);
      if (request.method !== "POST") return new Response(null, { status: 200 });
      return Response.json({ jsonrpc: "2.0", id: 1, result: {} }, { headers: { "mcp-session-id": "check-session" } });
    });
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-default";
    process.env.LITELLM_HEADERS = JSON.stringify({ "x-mcp-client": "pi" });
    const pi = await load(await makeAgentDir());

    expect(pi.mcpServers.get("litellm")).toEqual({
      url: "https://proxy.example.com/mcp",
      auth: { provider: "litellm" },
      headers: { "x-mcp-client": "pi" },
    });
    const check = requests[0]!;
    expect(check.method).toBe("POST");
    expect(check.redirect).toBe("manual");
    expect(check.headers.get("authorization")).toBe("Bearer sk-default");
    expect(check.headers.get("x-mcp-client")).toBe("pi");
    expect(await check.json()).toMatchObject({ method: "initialize" });
    await vi.waitFor(() => expect(requests).toHaveLength(2));
    const close = requests[1]!;
    expect(close.method).toBe("DELETE");
    expect(close.headers.get("mcp-session-id")).toBe("check-session");
    expect(close.headers.get("authorization")).toBe("Bearer sk-default");
  });

  it.each([401, 404, 500])("registers the server enabled when the check gets %i", async (status) => {
    mockProxy(() => new Response(null, { status }));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-default";
    const pi = await load(await makeAgentDir());

    expect(pi.mcpServers.get("litellm")).toEqual({
      url: "https://proxy.example.com/mcp",
      auth: { provider: "litellm" },
    });
  });

  it("gives up a check the proxy does not answer within the discovery budget", async () => {
    mockProxy(
      (request) =>
        new Promise((_, reject) => request.signal.addEventListener("abort", () => reject(request.signal.reason))),
    );
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-default";
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "50";
    const pi = await load(await makeAgentDir());

    expect(pi.mcpServers.get("litellm")).toEqual({
      url: "https://proxy.example.com/mcp",
      auth: { provider: "litellm" },
    });
  });

  it.each([
    ["a stored key helper", { type: "api_key", key: "!print-key" }, {}],
    ["Google ADC", undefined, { LITELLM_GCLOUD_TOKEN_AUTH: "1" }],
  ])("skips the check when %s supplies the credential Pi sends", async (_source, stored, env) => {
    const checks: Request[] = [];
    mockProxy((request) => {
      checks.push(request);
      return noServersGranted();
    });
    // Without the helper or ADC, resolution would fall through to this key.
    process.env.LITELLM_API_KEY = "sk-env";
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    Object.assign(process.env, env);
    const agentDir = await makeAgentDir();
    if (stored) await writeFile(join(agentDir, "auth.json"), JSON.stringify({ litellm: stored }), "utf8");
    const pi = await load(agentDir);

    expect(checks).toHaveLength(0);
    expect(pi.mcpServers.get("litellm")).toEqual({
      url: "https://proxy.example.com/mcp",
      auth: { provider: "litellm" },
    });
  });

  it("drops a check that a login overtakes, and checks again on the next turn", async () => {
    let answer: ((response: Response) => void) | undefined;
    mockProxy(() => {
      if (answer) return new Response(null, { status: 200 });
      return new Promise((resolve) => {
        answer = resolve;
      });
    });
    const agentDir = await makeAgentDir();
    const pi = await load(agentDir);
    await writeFile(
      join(agentDir, "auth.json"),
      JSON.stringify({
        litellm: { type: "api_key", key: "sk-old", env: { LITELLM_BASE_URL: "https://old.example.com" } },
      }),
      "utf8",
    );
    const turn = startTurn(pi);
    await vi.waitFor(() => expect(answer).toBeDefined());

    // The login may store a credential for another root, which Pi would send to this server.
    void pi.providers[0]?.auth.apiKey?.login?.({
      prompt: () => new Promise(() => {}),
      notify: vi.fn(),
      signal: new AbortController().signal,
    } as unknown as AuthInteraction & { signal: AbortSignal });
    answer?.(new Response(null, { status: 200 }));
    await turn;
    expect(pi.mcpServers.size).toBe(0);

    await startTurn(pi);
    expect(pi.mcpServers.get("litellm")).toEqual({ url: "https://old.example.com/mcp", auth: { provider: "litellm" } });
  });
});
