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

function mockProxy(): void {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input);
    if (url.endsWith("/model/info")) return Response.json({ data: [] });
    throw new Error(`unexpected URL: ${url}`);
  });
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

  it("says once that Pi 0.99.1 gets no MCP tools, since it lacks auth.provider", async () => {
    mockProxy();
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-default";
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);

    const pi = await load(await makeAgentDir(), createPi(), "0.99.1");
    await startTurn(pi);

    expect(pi.mcpServers.size).toBe(0);
    expect(stderrText(stderr).match(/MCP tools need Pi 0\.99\.2 or newer/g)).toHaveLength(1);
  });

  it("says once that Pi before 0.99 gets no MCP tools", async () => {
    mockProxy();
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "sk-default";
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const pi = createPi() as Omit<TestPi, "registerMcpServer"> & Partial<Pick<TestPi, "registerMcpServer">>;
    delete pi.registerMcpServer;

    await load(await makeAgentDir(), pi as TestPi);
    await startTurn(pi as TestPi);

    expect(stderrText(stderr).match(/MCP tools need Pi 0\.99\.2 or newer/g)).toHaveLength(1);
  });
});
