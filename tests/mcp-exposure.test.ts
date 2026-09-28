import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Static, TSchema } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  createMcpGatewayDefinition,
  createMcpToolDefinitions,
  gatewayToolName,
  MAX_AUTO_DIRECT_BYTES,
  type McpCatalogEntry,
  measureToolDefinitions,
  searchCatalog,
} from "../src/mcp-tools.js";
import type { LiteLLMMcpTool } from "../src/types.js";
import { createPi, loadExtension, type TestPi, useHermeticEnv } from "./test-helpers.js";

useHermeticEnv(["PI_CODING_AGENT_DIR"]);

vi.unmock("@earendil-works/pi-coding-agent");

afterEach(() => {
  vi.restoreAllMocks();
});

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

const auth = async () => ({ baseUrl: "https://litellm.example.com", apiKey: "sk-test" });

function entry(name: string, tool: Partial<LiteLLMMcpTool> = {}): McpCatalogEntry {
  return {
    name,
    tool: { name, server_name: "server", description: "", input_schema: {}, ...tool },
  };
}

type Params = Static<TSchema>;

async function runGateway(catalog: McpCatalogEntry[], params: Record<string, unknown>, getAuth = auth) {
  const gateway = createMcpGatewayDefinition(getAuth, () => catalog);
  return gateway.execute("call-1", params as Params, undefined, undefined, {} as never);
}

describe("MCP gateway tool", () => {
  it("names one gateway per provider, hashing aliases so they stay distinct and bounded", () => {
    expect(gatewayToolName()).toBe("litellm_mcp");
    expect(gatewayToolName("team-a")).toMatch(/^litellm_mcp_team_a_[a-f0-9]{10}$/);
    expect(gatewayToolName("team_a")).toMatch(/^litellm_mcp_team_a_[a-f0-9]{10}$/);
    expect(gatewayToolName("team-a")).not.toBe(gatewayToolName("team_a"));
    expect(gatewayToolName("a".repeat(200)).length).toBeLessThanOrEqual(64);
  });

  it("returns the prepared catalog alongside the direct definitions", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [
        { name: "a", server_name: "one", input_schema: { type: "object", properties: {} } },
        { name: "b", server_name: "two", input_schema: { type: "object", properties: {} } },
      ]),
    );

    const { definitions, catalog } = await createMcpToolDefinitions(auth);

    expect(catalog.map((item) => item.name)).toEqual(definitions.map((definition) => definition.name));
    expect(catalog.map((item) => item.tool.server_name)).toEqual(["one", "two"]);
  });

  it("measures the definition bytes direct registration would send", async () => {
    const small = entry("small");
    const gateway = createMcpGatewayDefinition(auth, () => [small]);
    const bytes = measureToolDefinitions([gateway]);

    expect(bytes).toBeGreaterThan(Buffer.byteLength(gateway.description));
    expect(measureToolDefinitions([gateway, gateway])).toBe(bytes * 2);
    expect(measureToolDefinitions([])).toBe(0);
  });

  it("ranks name and server matches ahead of description matches and keeps ties in catalog order", () => {
    const catalog = [
      entry("mcp_docs_read", { description: "Read a catalog entity" }),
      entry("mcp_catalog_query", { server_name: "catalog", description: "Query entities" }),
      entry("mcp_catalog_get", { server_name: "catalog", description: "Get one entity" }),
      entry("mcp_mail_send", { description: "Send mail" }),
    ];

    expect(searchCatalog(catalog, "Catalog").map((item) => item.name)).toEqual([
      "mcp_catalog_query",
      "mcp_catalog_get",
      "mcp_docs_read",
    ]);
    expect(searchCatalog(catalog, "  ").map((item) => item.name)).toEqual(catalog.map((item) => item.name));
    expect(searchCatalog(catalog, "invoice")).toEqual([]);
  });

  it("inlines the best two matches with their schemas and lists the rest as single lines", async () => {
    const catalog = Array.from({ length: 25 }, (_, index) =>
      entry(`mcp_srv_tool_${index}`, {
        description: `Tool ${index}\nsecond line`,
        input_schema: { type: "object", properties: { [`arg_${index}`]: { type: "string" } } },
      }),
    );

    const result = await runGateway(catalog, { action: "search", query: "tool" });
    const text = (result.content[0] as { text: string }).text;
    const [heading, first, second, list] = text.split("\n\n---\n\n");

    expect(heading).toBe(
      "25 of 25 MCP tools match; showing the first 20. The best 2 follow with their input schemas, ready to call. " +
        'Use action "describe" for any other tool before calling it.',
    );
    expect(first).toContain("mcp_srv_tool_0\nServer: server");
    expect(first).toContain('Input schema: {"type":"object","properties":{"arg_0":{"type":"string"}}}');
    expect(second).toContain("mcp_srv_tool_1\nServer: server");
    expect(list?.split("\n")[0]).toBe("- mcp_srv_tool_2: server: Tool 2 second line");
    expect(list).not.toContain("mcp_srv_tool_0:");
    expect(list).not.toContain("mcp_srv_tool_20");
  });

  it("names a single match ready to call without pointing at describe", async () => {
    const result = await runGateway([entry("mcp_srv_ping", { description: "Ping" })], {
      action: "search",
      query: "ping",
    });
    const text = (result.content[0] as { text: string }).text;

    expect(text.split("\n\n---\n\n")[0]).toBe(
      "1 of 1 MCP tools match. The best match follows with its input schema, ready to call.",
    );
  });

  it("leaves a best match whose schema would not fit whole to describe", async () => {
    const properties: Record<string, unknown> = {};
    for (let index = 0; JSON.stringify({ type: "object", properties }).length < 60 * 1024; index += 1) {
      properties[`field_${index}`] = { type: "string", description: "d".repeat(200) };
    }
    const catalog = [
      entry("mcp_srv_wide", { description: "Wide tool", input_schema: { type: "object", properties } }),
      entry("mcp_srv_narrow", { description: "Narrow tool" }),
    ];

    const result = await runGateway(catalog, { action: "search", query: "tool" });
    const text = (result.content[0] as { text: string }).text;

    expect(text.split("\n\n---\n\n")[0]).toBe(
      '2 of 2 MCP tools match. Use action "describe" for any other tool before calling it.',
    );
    expect(text).toContain("- mcp_srv_wide: server: Wide tool");
    expect(text).toContain("- mcp_srv_narrow: server: Narrow tool");
    expect(text).not.toContain("Input schema:");
  });

  it("describes a tool's schema as text without compiling proxy regexes into the gateway", async () => {
    const schema = {
      type: "object",
      properties: { id: { type: "string", pattern: "^(a+)+$" } },
      required: ["id"],
    };
    const catalog = [entry("mcp_srv_lookup", { description: "Look up a record", input_schema: schema })];
    const gateway = createMcpGatewayDefinition(auth, () => catalog);

    const result = await runGateway(catalog, { action: "describe", tool: "mcp_srv_lookup" });
    const text = (result.content[0] as { text: string }).text;

    expect(text).toContain("Server: server");
    expect(text).toContain("Look up a record");
    expect(text).toContain(`Input schema: ${JSON.stringify(schema)}`);
    expect(JSON.stringify(gateway.parameters)).not.toContain("(a+)+");
  });

  it("returns a near-limit schema whole, with a full-length description", async () => {
    const properties: Record<string, unknown> = {};
    for (let index = 0; JSON.stringify({ type: "object", properties }).length < 63 * 1024; index += 1) {
      properties[`field_${index}`] = { type: "string", description: "d".repeat(200) };
    }
    const schema = { type: "object", properties };
    const catalog = [entry("mcp_srv_wide", { description: "x".repeat(8 * 1024), input_schema: schema })];

    const result = await runGateway(catalog, { action: "describe", tool: "mcp_srv_wide" });
    const text = (result.content[0] as { text: string }).text;

    expect(JSON.parse(text.slice(text.indexOf("Input schema: ") + "Input schema: ".length))).toEqual(schema);
    expect(text).toContain("\u2026 [truncated]");
  });

  it("never compiles a proxy-supplied expression when Pi validates gateway calls", async () => {
    const schema = { type: "object", properties: { id: { type: "string", pattern: "^(a+)+$" } } };
    const catalog = [entry("mcp_srv_lookup", { input_schema: schema })];
    const gateway = createMcpGatewayDefinition(auth, () => catalog);
    const { Compile } = await import("typebox/compile");
    const compiled: string[] = [];
    const NativeRegExp = globalThis.RegExp;
    class RecordingRegExp extends NativeRegExp {
      constructor(source: string | RegExp, flags?: string) {
        compiled.push(String(source));
        super(source as never, flags);
      }
    }
    globalThis.RegExp = RecordingRegExp as never;
    try {
      const validator = Compile(gateway.parameters as never);
      expect(validator.Check({ action: "describe", tool: "mcp_srv_lookup" })).toBe(true);
      expect(validator.Check({ action: "call", tool: "mcp_srv_lookup", arguments: { id: "aaaa" } })).toBe(true);
      await runGateway(catalog, { action: "describe", tool: "mcp_srv_lookup" });
    } finally {
      globalThis.RegExp = NativeRegExp;
    }

    for (const source of compiled) expect(source).not.toContain("(a+)+");
  });

  it("tells the model when a tool has no schema", async () => {
    const result = await runGateway([entry("mcp_srv_ping")], { action: "describe", tool: "mcp_srv_ping" });

    expect((result.content[0] as { text: string }).text).toContain(
      "Input schema: none supplied by the server; pass arguments as an object",
    );
  });

  it("rejects an unknown tool without calling the proxy", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");

    await expect(runGateway([entry("mcp_srv_ping")], { action: "call", tool: "mcp_srv_missing" })).rejects.toThrow(
      'Unknown litellm_mcp tool; use action "search" to find a tool name.',
    );
    await expect(runGateway([entry("mcp_srv_ping")], { action: "describe" })).rejects.toThrow("Unknown litellm_mcp");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("calls the proxy exactly once with fresh auth and the tool's server identity", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { result: "ok" }));
    const getAuth = vi.fn(async () => ({
      baseUrl: "https://new.example.com",
      apiKey: "execution-token",
      headers: { "x-tenant": "new" },
    }));
    const catalog = [entry("mcp_brave_search", { name: "search", server_name: "brave", server_id: "brave-api" })];

    const result = await runGateway(
      catalog,
      { action: "call", tool: "mcp_brave_search", arguments: { query: "pi" } },
      getAuth,
    );

    expect(result.content).toEqual([{ type: "text", text: JSON.stringify("ok", null, 2) }]);
    expect(result.details).toEqual({ server: "brave", serverId: "brave-api", tool: "search" });
    expect(getAuth).toHaveBeenCalledTimes(1);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0]?.[0]).toBe("https://new.example.com/mcp-rest/tools/call");
    expect(fetchMock.mock.calls[0]?.[1]).toMatchObject({
      headers: expect.objectContaining({ Authorization: "Bearer execution-token", "x-tenant": "new" }),
      body: JSON.stringify({ server_id: "brave-api", name: "search", arguments: { query: "pi" } }),
    });
  });
});

// Enough description text that the catalog's definitions exceed the `auto` budget.
function largeCatalog(count: number, server = "big") {
  const description = "x".repeat(Math.ceil((MAX_AUTO_DIRECT_BYTES * 2) / count));
  return Array.from({ length: count }, (_, index) => ({
    name: `tool_${index}`,
    description,
    inputSchema: { type: "object", properties: {} },
    mcp_info: { server_name: server },
  }));
}

const smallCatalog = [
  {
    name: "search",
    description: "Search",
    inputSchema: { type: "object", properties: {} },
    mcp_info: { server_name: "brave" },
  },
];

async function refreshProvider(pi: TestPi): Promise<void> {
  await pi.providers[0]?.refreshModels?.({
    allowNetwork: true,
    stored: undefined,
    publish: async () => true,
    credential: {
      type: "api_key",
      key: process.env.LITELLM_API_KEY ?? "sk-test",
      env: { LITELLM_BASE_URL: process.env.LITELLM_BASE_URL ?? "https://proxy.example.com" },
    },
    signal: new AbortController().signal,
  });
}

// A catalog is a tool list, or a full discovery body such as a partial-failure envelope.
async function loadWithCatalogs(catalogs: Record<string, unknown>, mcpSettings?: Record<string, unknown>) {
  const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-"));
  if (mcpSettings) {
    await writeFile(join(agentDir, "settings.json"), JSON.stringify({ litellm: { mcp: mcpSettings } }), "utf8");
  }
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input);
    if (url.endsWith("/model/info")) return jsonResponse(200, { data: [] });
    for (const [origin, tools] of Object.entries(catalogs)) {
      if (url === `${origin}/mcp-rest/tools/list`) return jsonResponse(200, Array.isArray(tools) ? { tools } : tools);
    }
    throw new Error(`unexpected URL: ${url}`);
  });
  process.env.LITELLM_BASE_URL = Object.keys(catalogs)[0];
  process.env.LITELLM_API_KEY = "sk-test";
  const pi = createPi();
  await (await loadExtension(agentDir))(pi);
  return pi;
}

const mcpNames = (names: string[]) => names.filter((name) => name.startsWith("mcp_"));
const stderrText = (stderr: { mock: { calls: unknown[][] } }) =>
  stderr.mock.calls.map(([text]) => String(text)).join("");

async function refreshWith(pi: TestPi, baseUrl: string, apiKey: string): Promise<void> {
  process.env.LITELLM_BASE_URL = baseUrl;
  process.env.LITELLM_API_KEY = apiKey;
  await refreshProvider(pi);
}

describe("MCP exposure modes", () => {
  it("keeps a small catalog as direct tools in auto mode", async () => {
    const pi = await loadWithCatalogs({ "https://proxy.example.com": smallCatalog });
    await refreshProvider(pi);

    await vi.waitFor(() => expect(mcpNames(pi.activeTools)).toHaveLength(1));
    expect(pi.tools.map((tool) => tool.name)).not.toContain("litellm_mcp");
  });

  it("puts a catalog over the budget behind one gateway tool in auto mode", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const pi = await loadWithCatalogs({ "https://proxy.example.com": largeCatalog(40) });
    await refreshProvider(pi);

    await vi.waitFor(() => expect(pi.tools.map((tool) => tool.name)).toContain("litellm_mcp"));
    expect(mcpNames(pi.tools.map((tool) => tool.name))).toEqual([]);
    expect(stderr.mock.calls.map(([text]) => String(text)).join("")).toMatch(
      /LiteLLM MCP: 40 MCP tools would add about \d+k tokens to every request, so they are available through litellm_mcp instead\./,
    );
  });

  it('registers every tool directly when mode is "direct", whatever the catalog size', async () => {
    const pi = await loadWithCatalogs({ "https://proxy.example.com": largeCatalog(40) }, { mode: "direct" });
    await refreshProvider(pi);

    await vi.waitFor(() => expect(mcpNames(pi.activeTools)).toHaveLength(40));
    expect(pi.tools.map((tool) => tool.name)).not.toContain("litellm_mcp");
  });

  it('uses the gateway for a small catalog when mode is "gateway"', async () => {
    const pi = await loadWithCatalogs({ "https://proxy.example.com": smallCatalog }, { mode: "gateway" });
    await refreshProvider(pi);

    await vi.waitFor(() => expect(pi.activeTools).toContain("litellm_mcp"));
    expect(mcpNames(pi.tools.map((tool) => tool.name))).toEqual([]);
  });

  it("warns once and falls back to auto for an unknown mode", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const pi = await loadWithCatalogs({ "https://proxy.example.com": smallCatalog }, { mode: "lazy" });
    await refreshProvider(pi);

    await vi.waitFor(() => expect(mcpNames(pi.activeTools)).toHaveLength(1));
    const warnings = stderr.mock.calls
      .map(([text]) => String(text))
      .filter((text) => text.includes("litellm.mcp.mode"));
    expect(warnings).toEqual(['LiteLLM MCP: litellm.mcp.mode must be "auto", "direct", or "gateway"; using "auto".\n']);
  });

  it("hides direct tools when a refreshed catalog outgrows the budget, and restores them when it shrinks", async () => {
    vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const pi = await loadWithCatalogs({
      "https://small.example.com": smallCatalog,
      "https://large.example.com": largeCatalog(40),
    });
    pi.activeTools.push("read");
    pi.tools.push({ name: "read", description: "Read a file" });
    await refreshProvider(pi);
    await vi.waitFor(() => expect(mcpNames(pi.activeTools)).toHaveLength(1));
    const [directName] = mcpNames(pi.activeTools);

    process.env.LITELLM_BASE_URL = "https://large.example.com";
    process.env.LITELLM_API_KEY = "sk-large";
    await refreshProvider(pi);
    await vi.waitFor(() => expect(pi.activeTools).toContain("litellm_mcp"));
    expect(pi.activeTools).not.toContain(directName);
    expect(pi.activeTools).toContain("read");

    process.env.LITELLM_BASE_URL = "https://small.example.com";
    process.env.LITELLM_API_KEY = "sk-small";
    await refreshProvider(pi);
    await vi.waitFor(() => expect(pi.activeTools).toContain(directName));
    expect(pi.activeTools).not.toContain("litellm_mcp");
    expect(pi.activeTools).toContain("read");
  });

  it("leaves a direct tool the user turned off inactive across a gateway round trip", async () => {
    vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const pi = await loadWithCatalogs({
      "https://small.example.com": [...smallCatalog, { ...smallCatalog[0], name: "fetch" }],
      "https://large.example.com": largeCatalog(40),
    });
    await refreshProvider(pi);
    await vi.waitFor(() => expect(mcpNames(pi.activeTools)).toHaveLength(2));
    const [kept, turnedOff] = mcpNames(pi.activeTools);
    pi.setActiveTools(pi.activeTools.filter((name) => name !== turnedOff));

    await refreshWith(pi, "https://large.example.com", "sk-large");
    await vi.waitFor(() => expect(pi.activeTools).toContain("litellm_mcp"));
    await refreshWith(pi, "https://small.example.com", "sk-small");
    await vi.waitFor(() => expect(pi.activeTools).toContain(kept));

    expect(pi.activeTools).not.toContain(turnedOff);
  });

  it("leaves a gateway the user turned off inactive across a direct round trip", async () => {
    vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const pi = await loadWithCatalogs({
      "https://large.example.com": largeCatalog(40),
      "https://small.example.com": smallCatalog,
    });
    await refreshProvider(pi);
    await vi.waitFor(() => expect(pi.activeTools).toContain("litellm_mcp"));
    pi.setActiveTools(pi.activeTools.filter((name) => name !== "litellm_mcp"));

    await refreshWith(pi, "https://small.example.com", "sk-small");
    await vi.waitFor(() => expect(mcpNames(pi.activeTools)).toHaveLength(1));
    await refreshWith(pi, "https://large.example.com", "sk-large-2");
    await vi.waitFor(() => expect(mcpNames(pi.activeTools)).toEqual([]));

    expect(pi.activeTools).not.toContain("litellm_mcp");
    expect(pi.tools.map((tool) => tool.name)).toContain("litellm_mcp");
  });

  it("counts a refused gateway registration as one attempted tool", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const pi = await loadWithCatalogs({ "https://proxy.example.com": largeCatalog(40) });
    const registerTool = pi.registerTool.bind(pi);
    pi.registerTool = (tool) => {
      if (tool.name === "litellm_mcp") throw new Error("extension instance is stale");
      registerTool(tool);
    };
    await refreshProvider(pi);

    await vi.waitFor(() => expect(stderrText(stderr)).toContain("registration stopped after 0 of 1 MCP tool;"));
  });

  it("reports a partial-failure catalog as available through the gateway, not registered", async () => {
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const pi = await loadWithCatalogs({
      "https://proxy.example.com": { tools: largeCatalog(40), error: "partial_failure" },
    });
    await refreshProvider(pi);

    await vi.waitFor(() =>
      expect(stderrText(stderr)).toContain(
        "proxy reported a partial server failure; 40 tools available through litellm_mcp.",
      ),
    );
  });
});
