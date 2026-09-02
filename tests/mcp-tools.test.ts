import { createHash } from "node:crypto";
import type { Static, TSchema } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  createMcpToolDefinitions as createMcpToolDefinitionsRaw,
  credentialFingerprint,
  discoverMcpTools as discoverMcpToolsRaw,
  executeMcpTool,
  findSchemaHazard,
} from "../src/mcp-tools.js";

// Thin wrappers so the bulk of the suite keeps asserting on the shapes it cares about.
// Tests that need the reconciliation report import the raw functions directly.
const createMcpToolDefinitions = async (
  ...args: Parameters<typeof createMcpToolDefinitionsRaw>
): Promise<Awaited<ReturnType<typeof createMcpToolDefinitionsRaw>>["definitions"]> =>
  (await createMcpToolDefinitionsRaw(...args)).definitions;
const discoverMcpTools = async (
  ...args: Parameters<typeof discoverMcpToolsRaw>
): Promise<Awaited<ReturnType<typeof discoverMcpToolsRaw>>["tools"]> => (await discoverMcpToolsRaw(...args)).tools;

import type { LiteLLMMcpTool } from "../src/types.js";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

// Tool names are `mcp_<server>_<tool>` plus a 10-hex identity hash. Assert that contract rather
// than a literal digest, so the tests do not re-implement the naming function they are checking.
const named = (base: string) => expect.stringMatching(new RegExp(`^${base}_[a-f0-9]{10}$`));

describe("discoverMcpTools", () => {
  it("uses an explicitly allowed insecure endpoint", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { tools: [] }));

    await expect(
      discoverMcpTools("http://host.docker.internal", "sk-test", undefined, undefined, undefined, true),
    ).resolves.toEqual([]);

    expect(fetchMock).toHaveBeenCalledWith("http://host.docker.internal/mcp-rest/tools/list", expect.any(Object));
  });

  it("returns tools from LiteLLM MCP REST discovery", async () => {
    const inputSchema = {
      type: "object",
      properties: { query: { type: "string" } },
      required: ["query"],
    };
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        tools: [
          {
            name: "web-search",
            description: "Search the web",
            inputSchema,
            mcp_info: { server_name: "Brave API", server_id: "brave-api" },
          },
        ],
      }),
    );

    await expect(discoverMcpTools("https://litellm.example.com", "sk-test")).resolves.toEqual([
      {
        name: "web-search",
        server_name: "Brave API",
        server_id: "brave-api",
        description: "Search the web",
        input_schema: inputSchema,
      },
    ]);

    expect(fetchMock).toHaveBeenCalledWith(
      "https://litellm.example.com/mcp-rest/tools/list",
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer sk-test" }),
        signal: expect.any(AbortSignal),
      }),
    );
  });

  it("keeps compatibility with older array-shaped discovery responses", async () => {
    const tools: LiteLLMMcpTool[] = [
      {
        name: "web-search",
        server_name: "Brave API",
        server_id: "Brave API",
        description: "Search the web",
        input_schema: {
          type: "object",
          properties: { query: { type: "string" } },
          required: ["query"],
        },
      },
    ];
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, tools));

    await expect(discoverMcpTools("https://litellm.example.com", "sk-test")).resolves.toEqual(tools);

    expect(fetchMock).toHaveBeenCalledWith(
      "https://litellm.example.com/mcp-rest/tools/list",
      expect.objectContaining({
        headers: expect.objectContaining({ Authorization: "Bearer sk-test" }),
        signal: expect.any(AbortSignal),
      }),
    );
  });

  it("rejects malformed and oversized discovery bodies with actionable errors", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response("<html>nope</html>", { status: 200 }))
      .mockResolvedValueOnce(new Response("x".repeat(5 * 1024 * 1024 + 1), { status: 200 }));

    await expect(discoverMcpTools("https://litellm.example.com", "sk-test")).rejects.toThrow("invalid JSON");
    await expect(discoverMcpTools("https://litellm.example.com", "sk-test")).rejects.toThrow(
      "exceeds its 5242880-byte limit",
    );
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("returns the full bounded discovery list before registration filtering", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        tools: Array.from({ length: 513 }, (_, index) => ({
          name: `tool-${index}`,
          server_name: "server",
          input_schema: { type: "object", properties: {} },
        })),
      }),
    );

    const tools = await discoverMcpTools("https://litellm.example.com", "sk-test");

    expect(tools).toHaveLength(513);
    expect(tools.at(-1)?.name).toBe("tool-512");
  });

  it.each([
    ["array response", `[${"0,".repeat(10_000)}0]`],
    ["object response", `{"ignored":{"tools":[0]},"tools":[${"0,".repeat(10_000)}0]}`],
    ["escaped tools key", `{"to\\u006fls":[${"0,".repeat(10_000)}0]}`],
    ["multi-megabyte primitive array", `[${"0,".repeat(1_000_000)}0]`],
  ])("rejects excessive entries in a %s before JSON.parse materializes them", async (_label, body) => {
    const parse = vi.spyOn(JSON, "parse");
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(body, { status: 200 }));
    const progress: string[] = [];

    await expect(
      discoverMcpToolsRaw("https://litellm.example.com", "sk-test", undefined, (message) => progress.push(message)),
    ).rejects.toThrow("MCP discovery exceeds its 10000-entry limit");
    expect(parse.mock.calls).not.toContainEqual([body]);
    expect(progress).not.toContain("Found 10001 raw MCP tools, normalizing...");
  });

  it("propagates the original discovery cancellation reason", async () => {
    const controller = new AbortController();
    const reason = new Error("refresh cancelled");
    vi.spyOn(globalThis, "fetch").mockImplementation(
      (_input, init) =>
        new Promise((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
        }),
    );

    const discovery = discoverMcpTools(
      "https://litellm.example.com",
      "sk-test",
      undefined,
      undefined,
      controller.signal,
    );
    controller.abort(reason);

    await expect(discovery).rejects.toBe(reason);
  });

  it("rejects when discovery fails", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("offline"));

    await expect(discoverMcpTools("https://litellm.example.com", "sk-test")).rejects.toThrow("offline");
  });
});

describe("executeMcpTool", () => {
  it("uses an explicitly allowed insecure endpoint", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { result: "ok" }));

    await expect(
      executeMcpTool("http://host.docker.internal", "sk-test", "server", "tool", {}, undefined, true),
    ).resolves.toBe(JSON.stringify("ok", null, 2));

    expect(fetchMock).toHaveBeenCalledWith(
      "http://host.docker.internal/mcp-rest/tools/call",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("calls LiteLLM MCP REST execution and formats the result", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(200, { result: { content: [{ type: "text", text: "found" }] } }));

    await expect(
      executeMcpTool(
        "https://litellm.example.com",
        "sk-test",
        "brave",
        "search",
        { query: "pi" },
        { "X-Team": "agent" },
      ),
    ).resolves.toBe(JSON.stringify({ content: [{ type: "text", text: "found" }] }, null, 2));

    expect(fetchMock).toHaveBeenCalledWith(
      "https://litellm.example.com/mcp-rest/tools/call",
      expect.objectContaining({
        method: "POST",
        headers: expect.objectContaining({ Authorization: "Bearer sk-test", "X-Team": "agent" }),
        body: JSON.stringify({ server_id: "brave", name: "search", arguments: { query: "pi" } }),
      }),
    );
  });

  it.each([408, 425, 429, 500, 502, 503, 504])("attempts HTTP %i exactly once", async (status) => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(status, { error: "busy" }))
      .mockResolvedValueOnce(jsonResponse(200, { result: "must not run" }));

    await expect(
      executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", { query: "pi" }),
    ).rejects.toThrow(`HTTP ${status}`);

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it.each([
    new TypeError("fetch failed"),
    Object.assign(new Error("reset"), { code: "ECONNRESET" }),
    new DOMException("timed out", "TimeoutError"),
  ])("attempts transport failure $name exactly once", async (failure) => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockRejectedValue(failure);

    await expect(
      executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", { query: "pi" }),
    ).rejects.toThrow();

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("performs zero requests for a pre-aborted signal and propagates its reason", async () => {
    const controller = new AbortController();
    const reason = new Error("tool cancelled");
    controller.abort(reason);
    const fetchMock = vi.spyOn(globalThis, "fetch");

    await expect(
      executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", {}, undefined, controller.signal),
    ).rejects.toBe(reason);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("propagates the original reason when cancelled during execution", async () => {
    const controller = new AbortController();
    const reason = new Error("tool cancelled");
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(
      (_input, init) =>
        new Promise((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(new TypeError("fetch failed")), { once: true });
        }),
    );

    const execution = executeMcpTool(
      "https://litellm.example.com",
      "sk-test",
      "brave",
      "search",
      {},
      undefined,
      controller.signal,
    );
    controller.abort(reason);

    await expect(execution).rejects.toBe(reason);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("propagates the original reason when cancelled while reading the response body", async () => {
    const controller = new AbortController();
    const reason = new Error("tool cancelled after headers");
    const encoder = new TextEncoder();
    vi.spyOn(globalThis, "fetch").mockImplementation(async (_input, init) => {
      const body = new ReadableStream<Uint8Array>({
        start(streamController) {
          streamController.enqueue(encoder.encode('{"result":"partial'));
          init?.signal?.addEventListener("abort", () => streamController.error(new TypeError("body read failed")), {
            once: true,
          });
        },
      });
      return new Response(body, { status: 200 });
    });

    const execution = executeMcpTool(
      "https://litellm.example.com",
      "sk-test",
      "brave",
      "search",
      {},
      undefined,
      controller.signal,
    );
    await vi.waitFor(() => expect(globalThis.fetch).toHaveBeenCalledTimes(1));
    controller.abort(reason);

    await expect(execution).rejects.toBe(reason);
  });

  it("rejects response parse and MCP result errors", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response("not json", { status: 200 }))
      .mockResolvedValueOnce(jsonResponse(200, { error: { message: "denied" } }))
      .mockResolvedValueOnce(jsonResponse(200, { result: { isError: true, content: [{ text: "failed" }] } }));

    await expect(executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", {})).rejects.toThrow(
      "invalid JSON",
    );
    await expect(executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", {})).rejects.toThrow(
      "denied",
    );
    await expect(executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", {})).rejects.toThrow(
      "failed",
    );
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it("accepts an explicit null MCP error as success", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { error: null, result: "ok" }));

    await expect(executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", {})).resolves.toBe(
      JSON.stringify("ok", null, 2),
    );
  });

  it("bounds tool-call bodies before parsing", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("x".repeat(5 * 1024 * 1024 + 1), { status: 200 }));

    await expect(executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", {})).rejects.toThrow(
      "MCP tool call response exceeds its 5242880-byte limit",
    );
  });

  it("attempts a malformed successful body exactly once", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response("not json", { status: 200 }))
      .mockResolvedValueOnce(jsonResponse(200, { result: "must not run" }));

    await expect(executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", {})).rejects.toThrow(
      "invalid JSON",
    );
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("truncates returned result text to 64 KiB with a marker without splitting multibyte text", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { result: "😀".repeat(20 * 1024) }));

    const text = await executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", {});

    expect(Buffer.byteLength(text)).toBeLessThanOrEqual(64 * 1024);
    expect(text).toContain("[truncated by pi-provider-litellm]");
    expect(text).not.toContain("�");
  });

  it.each([
    { error: { message: "x".repeat(70 * 1024) } },
    { result: { isError: true, content: [{ text: "x".repeat(70 * 1024) }] } },
  ])("truncates MCP error text to 64 KiB", async (body) => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, body));

    const error = await executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", {}).catch(
      (failure: unknown) => failure,
    );

    expect(error).toBeInstanceOf(Error);
    expect(Buffer.byteLength((error as Error).message)).toBeLessThanOrEqual(64 * 1024);
    expect((error as Error).message).toContain("[truncated by pi-provider-litellm]");
  });

  it("bounds nested MCP error traversal", async () => {
    let error: unknown = "too deep";
    for (let depth = 0; depth < 100; depth += 1) error = [error];
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { error }));

    await expect(executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", {})).rejects.toThrow(
      "MCP error",
    );
  });
});

describe("createMcpToolDefinitions", () => {
  it("reports each drop class separately with counts and sanitized names, and no credential or schema content", async () => {
    vi.resetModules();
    const {
      createMcpToolDefinitions: createDefinitionsRaw,
      discoverMcpTools: discoverToolsRaw,
      executeMcpTool: executeTool,
    } = await import("../src/mcp-tools.js");
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const oversizedBody = "x".repeat(5 * 1024 * 1024 + 1);
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(async () => new Response(oversizedBody, { status: 200 }));
    await expect(discoverToolsRaw("https://litellm.example.com", "sk-secret-must-not-leak")).rejects.toThrow();
    await expect(discoverToolsRaw("https://litellm.example.com", "sk-secret-must-not-leak")).rejects.toThrow();
    await expect(
      executeTool("https://litellm.example.com", "sk-secret-must-not-leak", "server", "tool", {}),
    ).rejects.toThrow();
    await expect(
      executeTool("https://litellm.example.com", "sk-secret-must-not-leak", "server", "tool", {}),
    ).rejects.toThrow();

    const privateSchemaText = "private-schema-value";
    const duplicate = {
      name: "duplicate",
      server_name: "server",
      server_id: "server-id",
      input_schema: { type: "object", properties: {} },
    };
    fetchMock.mockImplementation(async () =>
      jsonResponse(200, [
        duplicate,
        duplicate,
        {
          name: "bad-one",
          server_name: "server",
          input_schema: { type: "object", properties: { secret: privateSchemaText } },
        },
        {
          name: "bad-two",
          server_name: "server",
          input_schema: { type: "object", properties: { secret: "also-private" } },
        },
        ...Array.from({ length: 513 }, (_, index) => ({
          name: `valid-${index}`,
          server_name: "server",
          input_schema: { type: "object", properties: {} },
        })),
      ]),
    );

    await createDefinitionsRaw(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-secret-must-not-leak",
    }));

    const diagnostics = stderr.mock.calls.map(([message]) => String(message));
    expect(diagnostics.filter((message) => message.includes("MCP discovery response exceeded"))).toHaveLength(1);
    expect(diagnostics.filter((message) => message.includes("MCP tool call response exceeded"))).toHaveLength(1);

    // Each class is reported on its own line, with a count and the sanitized generated names.
    const duplicateLine = diagnostics.find((message) => message.includes("a repeated valid identity"));
    expect(duplicateLine).toMatch(
      /^LiteLLM MCP: dropped 1 MCP tool with a repeated valid identity \(the first valid occurrence of each is the one registered\): mcp_server_duplicate_[a-f0-9]{10}\.\n$/,
    );
    const schemaLine = diagnostics.find((message) => message.includes("invalid or oversized input schema"));
    expect(schemaLine).toMatch(
      /^LiteLLM MCP: dropped 2 MCP tools with an invalid or oversized input schema: mcp_server_bad_one_[a-f0-9]{10}, mcp_server_bad_two_[a-f0-9]{10}\.\n$/,
    );
    // 513 valid + the surviving duplicate = 514 accepted, so exactly 2 exceed the cap.
    expect(diagnostics.filter((message) => message.includes("512-tool limit"))).toEqual([
      "LiteLLM MCP: ignoring 2 MCP tools beyond the 512-tool limit.\n",
    ]);

    expect(diagnostics.join("\n")).not.toContain("sk-secret-must-not-leak");
    expect(diagnostics.join("\n")).not.toContain(privateSchemaText);
    expect(diagnostics.join("\n")).not.toContain("also-private");
  });

  it("suppresses an unchanged drop report across refreshes but reports a changed one", async () => {
    vi.resetModules();
    const { createMcpToolDefinitions: createDefinitionsRaw } = await import("../src/mcp-tools.js");
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const invalid = (name: string) => ({ name, server_name: "server", input_schema: { type: "array" } });
    const auth = async () => ({ baseUrl: "https://litellm.example.com", apiKey: "sk-test" });
    const fetchMock = vi.spyOn(globalThis, "fetch");

    const schemaLines = () =>
      stderr.mock.calls
        .map(([message]) => String(message))
        .filter((message) => message.includes("invalid or oversized input schema"));

    fetchMock.mockImplementation(async () => jsonResponse(200, [invalid("bad-one")]));
    await createDefinitionsRaw(auth);
    expect(schemaLines()).toHaveLength(1);

    // Identical incident on the next refresh: stays quiet.
    await createDefinitionsRaw(auth);
    expect(schemaLines()).toHaveLength(1);

    // Count changed: reported again, because the operator's picture changed.
    fetchMock.mockImplementation(async () => jsonResponse(200, [invalid("bad-one"), invalid("bad-two")]));
    await createDefinitionsRaw(auth);
    expect(schemaLines()).toHaveLength(2);
    expect(schemaLines()[1]).toContain("dropped 2 MCP tools");

    // A clean refresh clears the class, so its recurrence is reported rather than suppressed as unchanged.
    fetchMock.mockImplementation(async () => jsonResponse(200, []));
    await createDefinitionsRaw(auth);
    expect(schemaLines()).toHaveLength(2);
    fetchMock.mockImplementation(async () => jsonResponse(200, [invalid("bad-one"), invalid("bad-two")]));
    await createDefinitionsRaw(auth);
    expect(schemaLines()).toHaveLength(3);
  });

  it("exposes prepared and dropped counts through verbose discovery progress", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [
        { name: "good", server_name: "server", input_schema: { type: "object", properties: {} } },
        { name: "bad", server_name: "server", input_schema: { type: "array" } },
      ]),
    );
    const progress: string[] = [];
    await createMcpToolDefinitions(
      async () => ({ baseUrl: "https://litellm.example.com", apiKey: "sk-test" }),
      (message) => progress.push(message),
    );

    expect(progress).toContain(
      "Prepared 1 of 2 raw MCP tools (0 kept with a safe args envelope); lost 1 (invalid-schema=1)",
    );
  });

  it("still disambiguates with a hash when both colliding tools survive the cap", async () => {
    const collidingTool = {
      name: "search",
      server_name: "shared",
      input_schema: { type: "object", properties: {} },
    };
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [
        { ...collidingTool, server_id: "server-one" },
        { ...collidingTool, server_id: "server-two" },
      ]),
    );

    const definitions = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));

    expect(definitions).toHaveLength(2);
    for (const definition of definitions) expect(definition.name).toMatch(/^mcp_shared_search_[a-f0-9]{10}$/);
    expect(new Set(definitions.map((definition) => definition.name)).size).toBe(2);
  });

  it("preserves a valid duplicate after an invalid occurrence of the same identity", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [
        { name: "duplicate", server_name: "server", input_schema: { type: "array" } },
        { name: "duplicate", server_name: "server", input_schema: { type: "object", properties: {} } },
      ]),
    );

    const { definitions, report } = await createMcpToolDefinitionsRaw(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));

    expect(definitions.map((tool) => tool.name)).toEqual([named("mcp_server_duplicate")]);
    expect(report.dropped.map(({ reason }) => reason)).toEqual(["invalid-schema"]);
    expect(report.prepared + report.dropped[0]!.tools.length + report.overflow).toBe(report.discovered);
  });

  it("caps registered tools after invalid entries are isolated", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [
        { name: "invalid", server_name: "server", input_schema: { type: "array" } },
        ...Array.from({ length: 513 }, (_, index) => ({
          name: `valid-${index}`,
          server_name: "server",
          input_schema: { type: "object", properties: {} },
        })),
      ]),
    );

    const definitions = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));

    expect(definitions).toHaveLength(512);
    expect(definitions.at(-1)?.name).toEqual(named("mcp_server_valid_511"));
  });

  it("creates sanitized Pi tool definitions with mapped parameter schemas", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [
        {
          name: "web-search",
          server_name: "Brave API",
          server_id: "brave-api",
          description: "Search the web",
          input_schema: {
            type: "object",
            properties: {
              query: { type: "string" },
              limit: { type: "integer" },
              safe: { type: "boolean" },
              tags: { type: "array", items: { type: "string" } },
            },
            required: ["query"],
          },
        },
      ] satisfies LiteLLMMcpTool[]),
    );

    const definitions = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));

    expect(definitions.map((tool) => tool.name)).toEqual([named("mcp_brave_api_web_search")]);
    expect(definitions[0]?.executionMode).toBe("parallel");
    expect(definitions[0]?.description).toBe("Search the web (via Brave API MCP server)");
    const parameters = definitions[0]?.parameters as { required?: string[]; properties?: Record<string, unknown> };
    expect(parameters.required).toEqual(["query"]);
    expect(Object.keys(parameters.properties ?? {})).toEqual(["query", "limit", "safe", "tags"]);
  });

  it("uses unique bounded names across server IDs and deduplicates exact identities", async () => {
    const longServerName = `shared-${"server".repeat(10)}`;
    const longToolName = `search-${"tool".repeat(12)}`;
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [
        {
          name: longToolName,
          server_name: longServerName,
          server_id: "server-one",
          input_schema: { type: "object", properties: {} },
        },
        {
          name: longToolName,
          server_name: longServerName,
          server_id: "server-two",
          input_schema: { type: "object", properties: {} },
        },
        {
          name: longToolName,
          server_name: longServerName,
          server_id: "server-one",
          input_schema: { type: "object", properties: {} },
        },
      ]),
    );

    const definitions = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));

    expect(definitions).toHaveLength(2);
    expect(new Set(definitions.map((tool) => tool.name)).size).toBe(2);
    expect(definitions.every((tool) => tool.name.length <= 64)).toBe(true);
    expect(definitions.every((tool) => /^[a-z0-9_]+$/.test(tool.name))).toBe(true);
  });

  it("isolates invalid schemas and preserves valid siblings", async () => {
    const tooDeep: Record<string, unknown> = { type: "string" };
    for (let depth = 0; depth < 17; depth += 1) {
      Object.assign(tooDeep, { type: "object", properties: { nested: { ...tooDeep } } });
    }
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [
        { name: "valid", server_name: "schema", input_schema: { type: "object", properties: {} } },
        {
          name: "properties-argument",
          server_name: "schema",
          input_schema: {
            type: "object",
            properties: { properties: { type: "object", additionalProperties: { type: "string" } } },
          },
        },
        {
          name: "required-argument",
          server_name: "schema",
          input_schema: { type: "object", properties: { required: { type: "boolean" } } },
        },
        { name: "array-root", server_name: "schema", input_schema: { type: "array", items: {} } },
        { name: "array-properties", server_name: "schema", input_schema: { type: "object", properties: [] } },
        {
          name: "invalid-property",
          server_name: "schema",
          input_schema: { type: "object", properties: { query: "not-a-schema" } },
        },
        {
          name: "nested-invalid-property",
          server_name: "schema",
          input_schema: { type: "object", properties: { query: { type: "object", properties: [] } } },
        },
        {
          name: "invalid-required",
          server_name: "schema",
          input_schema: { type: "object", properties: {}, required: "query" },
        },
        {
          name: "non-string-required",
          server_name: "schema",
          input_schema: { type: "object", properties: {}, required: ["query", 1] },
        },
        { name: "deep", server_name: "schema", input_schema: tooDeep },
        {
          name: "large",
          server_name: "schema",
          input_schema: { type: "object", properties: { value: { description: "x".repeat(64 * 1024) } } },
        },
      ]),
    );

    const definitions = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));

    expect(definitions.map((tool) => tool.name)).toEqual([
      named("mcp_schema_valid"),
      named("mcp_schema_properties_argument"),
      named("mcp_schema_required_argument"),
    ]);
  });

  it("accepts schemas at the depth and size boundaries", async () => {
    let acceptedDepth: Record<string, unknown> = { type: "string" };
    for (let depth = 0; depth < 7; depth += 1) {
      acceptedDepth = { type: "object", properties: { nested: acceptedDepth } };
    }
    const prefix = JSON.stringify({ type: "object", properties: { value: { description: "" } } }).length;
    const acceptedSize = {
      type: "object",
      properties: { value: { description: "x".repeat(64 * 1024 - prefix) } },
    };
    expect(Buffer.byteLength(JSON.stringify(acceptedSize))).toBe(64 * 1024);
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [
        { name: "depth-boundary", server_name: "schema", input_schema: acceptedDepth },
        { name: "size-boundary", server_name: "schema", input_schema: acceptedSize },
      ]),
    );

    const definitions = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));

    expect(definitions.map((tool) => tool.name)).toEqual([
      named("mcp_schema_depth_boundary"),
      named("mcp_schema_size_boundary"),
    ]);
  });

  it("truncates descriptions and prompt snippets to 4 KiB", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [
        {
          name: "verbose",
          server_name: "server",
          description: "x".repeat(5 * 1024),
          input_schema: { type: "object", properties: {} },
        },
      ]),
    );

    const [definition] = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));

    expect(Buffer.byteLength(definition?.description ?? "")).toBeLessThanOrEqual(4 * 1024);
    expect(definition?.description).toContain("[truncated]");
    expect(Buffer.byteLength(definition?.promptSnippet ?? "")).toBeLessThanOrEqual(4 * 1024);
  });

  it("passes complex object schemas through to Pi tools", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [
        {
          name: "complex",
          server_name: "Schema",
          description: "Complex input",
          input_schema: {
            type: "object",
            properties: { nested: { type: "object", properties: { value: { type: "string" } } } },
            required: ["nested"],
          },
        },
      ] satisfies LiteLLMMcpTool[]),
    );

    const definitions = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));

    expect(definitions[0]?.parameters).toMatchObject({
      properties: { nested: { type: "object", properties: { value: { type: "string" } } } },
      required: ["nested"],
    });
  });

  it.each([
    {
      label: "object",
      argsSchema: { type: "object", properties: { value: { type: "string" } } },
      value: { value: "kept" },
    },
    { label: "array", argsSchema: { type: "array", items: { type: "string" } }, value: ["kept"] },
  ])("does not unwrap legitimate $label-valued args properties", async ({ argsSchema, value }) => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        jsonResponse(200, [
          {
            name: "takes-args",
            server_name: "server",
            input_schema: { type: "object", properties: { args: argsSchema } },
          },
        ]),
      )
      .mockResolvedValueOnce(jsonResponse(200, { result: "ok" }));

    const [definition] = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));
    await definition?.execute("call-1", { args: value }, undefined, undefined, {} as never);

    expect(vi.mocked(globalThis.fetch).mock.calls[1]?.[1]).toMatchObject({
      body: JSON.stringify({
        server_id: "server",
        name: "takes-args",
        arguments: { args: value },
      }),
    });
  });

  it("unwraps a synthetic args envelope for an absent schema", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(200, [{ name: "unknown-schema", server_name: "server" }]))
      .mockResolvedValueOnce(jsonResponse(200, { result: "ok" }));

    const [definition] = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));
    await definition?.execute("call-1", { args: { value: "unwrapped" } }, undefined, undefined, {} as never);

    expect(vi.mocked(globalThis.fetch).mock.calls[1]?.[1]).toMatchObject({
      body: JSON.stringify({ server_id: "server", name: "unknown-schema", arguments: { value: "unwrapped" } }),
    });
  });

  it.each([{ args: "bad" }, { args: ["bad"] }, {}])(
    "rejects malformed synthetic args without issuing a tool call",
    async (params) => {
      const fetchMock = vi
        .spyOn(globalThis, "fetch")
        .mockResolvedValueOnce(jsonResponse(200, [{ name: "unknown-schema", server_name: "server" }]));
      const [definition] = await createMcpToolDefinitions(async () => ({
        baseUrl: "https://litellm.example.com",
        apiKey: "sk-test",
      }));

      await expect(definition?.execute("call-1", params, undefined, undefined, {} as never)).rejects.toThrow(
        "object-valued args property",
      );
      expect(fetchMock).toHaveBeenCalledTimes(1);
    },
  );

  it("propagates definition-level execution failures", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        jsonResponse(200, [
          { name: "failure", server_name: "server", input_schema: { type: "object", properties: {} } },
        ]),
      )
      .mockResolvedValueOnce(jsonResponse(500, { error: "failed" }));
    const [definition] = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));

    await expect(definition?.execute("call-1", {}, undefined, undefined, {} as never)).rejects.toThrow("HTTP 500");
  });

  it("passes Pi cancellation through generated tool execution", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        jsonResponse(200, [
          { name: "cancel", server_name: "server", input_schema: { type: "object", properties: {} } },
        ]),
      )
      .mockImplementationOnce(
        (_input, init) =>
          new Promise((_resolve, reject) => {
            init?.signal?.addEventListener("abort", () => reject(init.signal?.reason), { once: true });
          }),
      );
    const [definition] = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));
    const controller = new AbortController();
    const reason = new Error("cancelled by Pi");

    const execution = definition?.execute("call-1", {}, controller.signal, undefined, {} as never);
    controller.abort(reason);

    await expect(execution).rejects.toBe(reason);
  });

  it("uses a fresh token when a generated tool executes", async () => {
    vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(
        jsonResponse(200, {
          tools: [
            {
              name: "search",
              description: "Search",
              inputSchema: {
                type: "object",
                properties: { query: { type: "string" } },
                required: ["query"],
              },
              mcp_info: { server_name: "brave", server_id: "brave-api" },
            },
          ],
        }),
      )
      .mockResolvedValueOnce(jsonResponse(200, { result: "ok" }));
    const getAuth = vi
      .fn()
      .mockResolvedValueOnce({ baseUrl: "https://old.example.com", apiKey: "discovery-token" })
      .mockResolvedValueOnce({
        baseUrl: "https://new.example.com",
        apiKey: "execution-token",
        headers: { "x-tenant": "new" },
      });

    const definitions = await createMcpToolDefinitions(getAuth);
    type Params = Static<TSchema>;
    const result = await definitions[0]?.execute(
      "call-1",
      { query: "pi" } as Params,
      undefined,
      undefined,
      {} as never,
    );

    expect(result?.content).toEqual([{ type: "text", text: JSON.stringify("ok", null, 2) }]);
    expect(getAuth).toHaveBeenCalledTimes(2);
    expect(vi.mocked(globalThis.fetch).mock.calls[1]?.[0]).toBe("https://new.example.com/mcp-rest/tools/call");
    expect(vi.mocked(globalThis.fetch).mock.calls[1]?.[1]).toMatchObject({
      headers: expect.objectContaining({ Authorization: "Bearer execution-token", "x-tenant": "new" }),
      body: JSON.stringify({ server_id: "brave-api", name: "search", arguments: { query: "pi" } }),
    });
  });
});

describe("schema keyword validation positions", () => {
  const auth = async () => ({ baseUrl: "https://litellm.example.com", apiKey: "sk-test" });

  async function prepare(inputSchema: unknown): Promise<string[]> {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [{ name: "probe", server_name: "srv", input_schema: inputSchema }]),
    );
    const definitions = await createMcpToolDefinitions(auth);
    return definitions.map((definition) => definition.name);
  }

  // Each newly allowed keyword is validated where the JSON Schema spec puts a schema, and must be
  // rejected when that position holds a non-schema value. `patternProperties` is covered separately
  // because it is refused outright.
  const directKeywords = [
    "additionalProperties",
    "contains",
    "contentSchema",
    "else",
    "if",
    "not",
    "propertyNames",
    "then",
    "unevaluatedItems",
    "unevaluatedProperties",
  ] as const;
  const arrayKeywords = ["allOf", "anyOf", "oneOf", "prefixItems"] as const;
  const mapKeywords = ["$defs", "definitions", "dependentSchemas", "properties"] as const;

  it.each(directKeywords)("accepts a schema at the %s position and rejects a non-schema there", async (keyword) => {
    expect(await prepare({ type: "object", properties: { x: { [keyword]: { type: "string" } } } })).toEqual([
      named("mcp_srv_probe"),
    ]);
    expect(await prepare({ type: "object", properties: { x: { [keyword]: 5 } } })).toEqual([]);
  });

  it.each(arrayKeywords)("accepts a schema array at the %s position and rejects a non-array there", async (keyword) => {
    expect(await prepare({ type: "object", properties: { x: { [keyword]: [{ type: "string" }] } } })).toEqual([
      named("mcp_srv_probe"),
    ]);
    expect(await prepare({ type: "object", properties: { x: { [keyword]: { type: "string" } } } })).toEqual([]);
    expect(await prepare({ type: "object", properties: { x: { [keyword]: [5] } } })).toEqual([]);
  });

  it.each(mapKeywords)(
    "accepts a schema map at the %s position and rejects a non-schema value there",
    async (keyword) => {
      expect(await prepare({ type: "object", properties: { x: { [keyword]: { k: { type: "string" } } } } })).toEqual([
        named("mcp_srv_probe"),
      ]);
      expect(await prepare({ type: "object", properties: { x: { [keyword]: 5 } } })).toEqual([]);
      expect(await prepare({ type: "object", properties: { x: { [keyword]: { k: 5 } } } })).toEqual([]);
    },
  );

  it("accepts boolean subschemas and both items forms", async () => {
    expect(await prepare({ type: "object", properties: { x: { additionalProperties: false } } })).toEqual([
      named("mcp_srv_probe"),
    ]);
    expect(await prepare({ type: "object", properties: { x: { items: { type: "string" } } } })).toEqual([
      named("mcp_srv_probe"),
    ]);
    expect(await prepare({ type: "object", properties: { x: { items: [{ type: "string" }, true] } } })).toEqual([
      named("mcp_srv_probe"),
    ]);
    expect(await prepare({ type: "object", properties: { x: { items: 5 } } })).toEqual([]);
  });

  it("rejects a non-string-array required at any schema position", async () => {
    expect(await prepare({ type: "object", required: ["a"], properties: { a: { type: "string" } } })).toEqual([
      named("mcp_srv_probe"),
    ]);
    expect(await prepare({ type: "object", required: "a" })).toEqual([]);
    expect(await prepare({ type: "object", properties: { x: { type: "object", required: [5] } } })).toEqual([]);
  });

  it("does not validate keyword names that appear at data positions", async () => {
    // These are argument names, not constraints, so they must survive untouched.
    const dataPositions = {
      type: "object",
      properties: {
        if: { type: "string" },
        not: { type: "string" },
        required: { type: "string" },
        properties: { type: "string" },
        allOf: { type: "string" },
        pattern: { type: "string" },
        patternProperties: { type: "string" },
      },
    };
    expect(await prepare(dataPositions)).toEqual([named("mcp_srv_probe")]);
  });
});

describe("untrusted schema hazards", () => {
  const auth = async () => ({ baseUrl: "https://litellm.example.com", apiKey: "sk-test" });
  const CATASTROPHIC = "^(a+)+$";
  const REMOTE_REF = "https://evil.example/x.json";

  // Positions are chosen from the JSON Schema specification and from what TypeBox's validator
  // actually walks. Deliberately NOT derived from the implementation's own keyword tables: a list
  // copied from src cannot detect a position src forgot.
  const hazardCatalogs: Array<[string, unknown]> = [
    ["root property", { type: "object", properties: { s: { type: "string", pattern: CATASTROPHIC } } }],
    [
      "deeply nested property",
      { type: "object", properties: { o: { type: "object", properties: { s: { pattern: CATASTROPHIC } } } } },
    ],
    ["allOf branch", { type: "object", properties: { s: { allOf: [{ pattern: CATASTROPHIC }] } } }],
    ["items schema", { type: "object", properties: { a: { type: "array", items: { pattern: CATASTROPHIC } } } }],
    ["$defs entry", { type: "object", $defs: { s: { pattern: CATASTROPHIC } }, properties: {} }],
    ["schema root", { type: "object", pattern: CATASTROPHIC, properties: {} }],
    ["patternProperties keys", { type: "object", patternProperties: { [CATASTROPHIC]: { type: "string" } } }],
    // The four escapes that a position-table walk misses.
    [
      "dependencies subschema (draft-07)",
      {
        type: "object",
        properties: { k: { type: "string" } },
        dependencies: { k: { properties: { s: { pattern: CATASTROPHIC } } } },
      },
    ],
    [
      "additionalItems subschema",
      { type: "object", properties: { a: { type: "array", additionalItems: { pattern: CATASTROPHIC } } } },
    ],
    [
      "local ref into a data position (default)",
      { type: "object", default: { e: { pattern: CATASTROPHIC } }, properties: { s: { $ref: "#/default/e" } } },
    ],
    [
      "local ref into a data position (examples)",
      { type: "object", examples: [{ pattern: CATASTROPHIC }], properties: { s: { $ref: "#/examples/0" } } },
    ],
    // Reference forms whose target is not in the document we were handed.
    ["remote $ref", { type: "object", properties: { s: { $ref: REMOTE_REF } } }],
    ["$dynamicRef", { type: "object", properties: { s: { $dynamicRef: "#meta" } } }],
    ["$recursiveRef", { type: "object", properties: { s: { $recursiveRef: "#" } } }],
    [
      "nested $id self-cycle",
      {
        type: "object",
        $defs: { nested: { $id: "nested", $ref: "#" } },
        properties: { s: { $ref: "#/$defs/nested" } },
      },
    ],
    ["root $anchor alias", { $anchor: "root", type: "object", properties: { s: { $ref: "#root" } } }],
  ];

  it.each(hazardCatalogs)(
    "replaces the schema with the trusted envelope and keeps the sibling: %s",
    async (_label, inputSchema) => {
      vi.spyOn(globalThis, "fetch").mockResolvedValue(
        jsonResponse(200, {
          tools: [
            { name: "risky", server_name: "srv", inputSchema },
            {
              name: "benign",
              server_name: "srv",
              inputSchema: { type: "object", properties: { q: { type: "string" } } },
            },
          ],
        }),
      );
      const definitions = await createMcpToolDefinitions(auth);

      // Both tools remain usable; the hazard costs the schema, not the tool.
      expect(definitions.map((definition) => definition.name)).toEqual([
        named("mcp_srv_risky"),
        named("mcp_srv_benign"),
      ]);

      const risky = definitions[0];
      const serialized = JSON.stringify(risky?.parameters);
      // The exact proxy-supplied regex and ref must be absent from what Pi will compile.
      expect(serialized).not.toContain(CATASTROPHIC);
      expect(serialized).not.toContain(REMOTE_REF);
      expect(serialized).not.toContain("$ref");
      expect(serialized).not.toContain("$dynamicRef");
      expect(serialized).not.toContain("$recursiveRef");
      // It is the extension-owned envelope, which requires an object-valued `args`.
      expect(risky?.parameters).toMatchObject({ type: "object", required: ["args"] });

      // The untouched sibling still carries its own schema.
      expect(JSON.stringify(definitions[1]?.parameters)).toContain('"q"');
    },
  );

  it("refuses a percent-encoded ref decoy before the validator can decode it", async () => {
    const decoy = {
      type: "object",
      $defs: {
        "a%2Fdependencies": { type: "string" },
        a: { dependencies: { pattern: CATASTROPHIC } },
      },
      properties: { s: { $ref: "#/$defs/a%2Fdependencies" } },
    };
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, { tools: [{ name: "decoy", server_name: "srv", inputSchema: decoy }] }),
    );

    const { definitions, report } = await createMcpToolDefinitionsRaw(auth);

    expect(report.enveloped).toBe(0);
    expect(definitions).toEqual([]);
    expect(Object.fromEntries(report.dropped.map((entry) => [entry.reason, entry.tools.length]))).toEqual({
      "invalid-schema": 1,
    });
  });

  it("reports envelope degradation under its own class, without echoing the schema", async () => {
    vi.resetModules();
    const { createMcpToolDefinitions: createDefinitionsRaw } = await import("../src/mcp-tools.js");
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        tools: [
          {
            name: "lookup",
            server_name: "srv",
            inputSchema: { type: "object", properties: { s: { type: "string", pattern: CATASTROPHIC } } },
          },
          { name: "broken", server_name: "srv", inputSchema: { type: "array" } },
        ],
      }),
    );

    const { report } = await createDefinitionsRaw(auth);
    const diagnostics = stderr.mock.calls.map(([message]) => String(message));

    expect(report.enveloped).toBe(1);
    expect(diagnostics.join("")).toContain("kept 1 MCP tool but replaced its schema with a safe args envelope");
    // Kept distinct from a structural rejection, which is a real loss.
    expect(diagnostics.join("")).toContain("dropped 1 MCP tool with an invalid or oversized input schema");
    expect(diagnostics.join("")).not.toContain(CATASTROPHIC);
  });

  describe("negative controls: these must keep their supplied schema", () => {
    const clean: Array<[string, unknown]> = [
      [
        "a property literally named pattern",
        { type: "object", properties: { pattern: { type: "string" }, patternProperties: { type: "string" } } },
      ],
      [
        "a local ref to a real subschema",
        { type: "object", $defs: { s: { type: "string" } }, properties: { s: { $ref: "#/$defs/s" } } },
      ],
      [
        "an escaped local ref to a real subschema",
        { type: "object", $defs: { "a/b": { type: "string" } }, properties: { s: { $ref: "#/$defs/a~1b" } } },
      ],
      ["an enum containing the word pattern", { type: "object", properties: { s: { enum: ["pattern", "other"] } } }],
      [
        "a default whose value is the string pattern",
        { type: "object", properties: { s: { type: "string", default: "pattern" } } },
      ],
      ["a description mentioning pattern", { type: "object", description: "matches a pattern", properties: {} }],
      [
        "nested objects with no regex at all",
        { type: "object", properties: { a: { type: "object", properties: { b: { type: "number" } } } } },
      ],
    ];

    it.each(clean)("keeps the proxy schema for %s", async (_label, inputSchema) => {
      vi.spyOn(globalThis, "fetch").mockResolvedValue(
        jsonResponse(200, { tools: [{ name: "clean", server_name: "srv", inputSchema }] }),
      );
      const { definitions, report } = await createMcpToolDefinitionsRaw(auth);
      expect(definitions).toHaveLength(1);
      expect(report.enveloped).toBe(0);
      // Passed through as supplied, not swapped for the envelope.
      expect(JSON.stringify(definitions[0]?.parameters)).toBe(JSON.stringify(inputSchema));
    });
  });
});

describe("bounded untrusted display strings", () => {
  it("truncates oversized label and details values with a marker", async () => {
    const hugeServer = "s".repeat(4096);
    const hugeTool = "t".repeat(4096);
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [{ name: hugeTool, server_name: hugeServer, server_id: "i".repeat(4096), input_schema: {} }]),
    );

    const [definition] = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));

    expect(definition).toBeDefined();
    expect(Buffer.byteLength(definition?.label ?? "", "utf8")).toBeLessThanOrEqual(256);
    expect(definition?.label).toMatch(/…$/);

    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { result: "ok" }));
    const result = await definition?.execute?.(
      "call-1",
      { args: {} } as never,
      undefined as never,
      () => {},
      {} as never,
    );
    const details = (result as { details: Record<string, string> }).details;
    for (const value of Object.values(details)) {
      expect(Buffer.byteLength(value, "utf8")).toBeLessThanOrEqual(256);
      expect(value).toMatch(/…$/);
    }
  });

  it("leaves label and details untouched when they are already short", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [{ name: "search", server_name: "brave", server_id: "brave-id", input_schema: {} }]),
    );
    const [definition] = await createMcpToolDefinitions(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));
    expect(definition?.label).toBe("brave: search");
  });
});

describe("body cap diagnostics under a failing cancel", () => {
  it("still emits the diagnostic and the cap error when reader.cancel() rejects", async () => {
    vi.resetModules();
    const { discoverMcpTools: discoverToolsRaw } = await import("../src/mcp-tools.js");
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);

    // A body that breaches the cap and whose cancel() rejects, which is what an errored stream does.
    const body = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new Uint8Array(5 * 1024 * 1024 + 1));
      },
      cancel() {
        return Promise.reject(new Error("upstream reset"));
      },
    });
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response(body, { status: 200 }));

    await expect(discoverToolsRaw("https://litellm.example.com", "sk-test")).rejects.toThrow(
      "MCP discovery response exceeds its 5242880-byte limit",
    );
    expect(stderr.mock.calls.map(([message]) => String(message))).toContain(
      "LiteLLM MCP: MCP discovery response exceeded its 5242880-byte limit.\n",
    );
  });
});

describe("loss accounting reconciles to the raw catalog", () => {
  const auth = async () => ({ baseUrl: "https://litellm.example.com", apiKey: "sk-test" });

  it("accounts for 4 raw entries yielding 2 registered tools", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        tools: [
          { name: "ok", server_name: "srv", inputSchema: { type: "object", properties: {} } },
          {
            name: "enveloped",
            server_name: "srv",
            inputSchema: { type: "object", properties: { s: { pattern: "^a+$" } } },
          },
          { name: "no-server" }, // normalization loss: no server identity
          { name: "garbage", server_name: "srv", inputSchema: "not-an-object" }, // malformed, not schemaless
        ],
      }),
    );

    const { definitions, report } = await createMcpToolDefinitionsRaw(auth);

    expect(report.discovered).toBe(4);
    expect(definitions).toHaveLength(2);
    expect(report.prepared).toBe(2);
    expect(report.enveloped).toBe(1);

    const byReason = Object.fromEntries(report.dropped.map((entry) => [entry.reason, entry.tools.length]));
    expect(byReason).toEqual({ "invalid-tool": 1, "invalid-schema": 1 });

    // Every raw entry is accounted for exactly once.
    const droppedTotal = report.dropped.reduce((total, entry) => total + entry.tools.length, 0);
    expect(report.prepared + droppedTotal + report.overflow).toBe(report.discovered);
  });

  it("labels a normalization loss safely when the entry has no usable name", async () => {
    vi.resetModules();
    const { createMcpToolDefinitions: createDefinitionsRaw } = await import("../src/mcp-tools.js");
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, { tools: [{ description: "\u001b[31mred\u001b[0m nonsense" }, {}] }),
    );

    await createDefinitionsRaw(auth);

    const diagnostics = stderr.mock.calls.map(([message]) => String(message)).join("");
    expect(diagnostics).toContain("dropped 2 MCP tools with a missing name or server identity: entry_0, entry_1.");
    // No proxy-supplied text, and no escape sequences, reach the diagnostic.
    expect(diagnostics).not.toContain("nonsense");
    expect(diagnostics).not.toContain("\u001b");
  });

  it("only reports degradation for hazardous tools that survive the registration cap", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        tools: [
          ...Array.from({ length: 512 }, (_, index) => ({
            name: `safe-${index}`,
            server_name: "srv",
            inputSchema: { type: "object", properties: {} },
          })),
          {
            name: "overflow-hazard",
            server_name: "srv",
            inputSchema: { type: "object", properties: { s: { pattern: "^(a+)+$" } } },
          },
        ],
      }),
    );

    const { definitions, report } = await createMcpToolDefinitionsRaw(auth);

    expect(definitions).toHaveLength(512);
    expect(report.overflow).toBe(1);
    expect(report.degraded).toEqual([]);
    expect(report.enveloped).toBe(0);
  });

  it("treats a malformed non-object schema as invalid rather than schemaless", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        tools: [
          { name: "arr", server_name: "srv", inputSchema: ["evil"] },
          { name: "str", server_name: "srv", inputSchema: "evil" },
          { name: "num", server_name: "srv", inputSchema: 7 },
          { name: "absent", server_name: "srv" },
        ],
      }),
    );
    const { definitions, report } = await createMcpToolDefinitionsRaw(auth);

    // Only the genuinely schemaless tool survives, and it uses the trusted envelope.
    expect(definitions.map((definition) => definition.name)).toEqual([named("mcp_srv_absent")]);
    expect(definitions[0]?.parameters).toMatchObject({ type: "object", required: ["args"] });
    expect(Object.fromEntries(report.dropped.map((e) => [e.reason, e.tools.length]))).toEqual({ "invalid-schema": 3 });
    // The schemaless tool carries the envelope, so it is counted...
    expect(report.enveloped).toBe(1);
    // ...but it is not a hazard degradation, so no schema-envelope class is reported.
    expect(report.degraded).toEqual([]);
  });
});

describe("naming is membership-independent", () => {
  const auth = async () => ({ baseUrl: "https://litellm.example.com", apiKey: "sk-test" });
  const tool = (name: string, serverId: string) => ({
    name,
    server_name: "shared",
    server_id: serverId,
    inputSchema: { type: "object", properties: {} },
  });

  async function namesFor(catalog: unknown[]): Promise<string[]> {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { tools: catalog }));
    const definitions = await createMcpToolDefinitions(auth);
    vi.restoreAllMocks();
    return definitions.map((definition) => definition.name);
  }

  it("does not rename a survivor when a colliding sibling appears or disappears", async () => {
    const alone = await namesFor([tool("search", "one")]);
    const together = await namesFor([tool("search", "one"), tool("search", "two")]);
    const aloneAgain = await namesFor([tool("search", "one")]);

    expect(alone).toEqual(aloneAgain);
    // The first tool keeps exactly the same name whether or not its twin is present.
    expect(together[0]).toBe(alone[0]);
    expect(together[1]).not.toBe(together[0]);
    for (const name of together) expect(name.length).toBeLessThanOrEqual(64);
  });

  it("does not rename a survivor when a sibling is dropped or falls beyond the cap", async () => {
    const clean = await namesFor([tool("search", "one")]);
    const withDroppedSibling = await namesFor([
      tool("search", "one"),
      { name: "broken", server_name: "shared", inputSchema: { type: "array" } },
    ]);
    const beyondCap = await namesFor([
      tool("search", "one"),
      ...Array.from({ length: 600 }, (_, index) => tool(`bulk-${index}`, `bulk-${index}`)),
    ]);

    expect(withDroppedSibling[0]).toBe(clean[0]);
    expect(beyondCap[0]).toBe(clean[0]);
  });

  it("keeps names within the length limit and accepted charset for very long identities", async () => {
    const names = await namesFor([
      { name: "t".repeat(200), server_name: "s".repeat(200), server_id: "i", inputSchema: {} },
    ]);
    expect(names[0]).toMatch(/^mcp_[a-z0-9_]*_[a-f0-9]{10}$/);
    expect(names[0]?.length).toBe(64);
  });
});

describe("incident identity covers full membership", () => {
  const auth = async () => ({ baseUrl: "https://litellm.example.com", apiKey: "sk-test" });

  it("re-reports when membership changes beyond the printed sample", async () => {
    vi.resetModules();
    const { createMcpToolDefinitions: createDefinitionsRaw } = await import("../src/mcp-tools.js");
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const fetchMock = vi.spyOn(globalThis, "fetch");
    const broken = (name: string) => ({ name, server_name: "srv", inputSchema: { type: "array" } });
    const lines = () =>
      stderr.mock.calls.map(([m]) => String(m)).filter((m) => m.includes("invalid or oversized input schema"));

    // Seven drops: only the first five names are printed.
    const first = ["a1", "a2", "a3", "a4", "a5", "f6", "f7"];
    fetchMock.mockImplementation(async () => jsonResponse(200, { tools: first.map(broken) }));
    await createDefinitionsRaw(auth);
    expect(lines()).toHaveLength(1);
    expect(lines()[0]).toContain("(+2 more)");

    // Identical membership: stays quiet.
    await createDefinitionsRaw(auth);
    expect(lines()).toHaveLength(1);

    // Same count, same first five, different tail. Must still re-report.
    const swapped = ["a1", "a2", "a3", "a4", "a5", "z6", "z7"];
    fetchMock.mockImplementation(async () => jsonResponse(200, { tools: swapped.map(broken) }));
    await createDefinitionsRaw(auth);
    expect(lines()).toHaveLength(2);
  });

  it("re-reports when tool-cap membership changes without changing the overflow count", async () => {
    vi.resetModules();
    const { createMcpToolDefinitions: createDefinitionsRaw } = await import("../src/mcp-tools.js");
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const fetchMock = vi.spyOn(globalThis, "fetch");
    const tool = (name: string) => ({
      name,
      server_name: "srv",
      inputSchema: { type: "object", properties: {} },
    });
    const lines = () =>
      stderr.mock.calls.map(([message]) => String(message)).filter((message) => message.includes("512-tool limit"));
    const firstCatalog = Array.from({ length: 513 }, (_, index) => tool(`tool-${index}`));

    fetchMock.mockImplementation(async () => jsonResponse(200, { tools: firstCatalog }));
    await createDefinitionsRaw(auth);
    expect(lines()).toHaveLength(1);

    await createDefinitionsRaw(auth);
    expect(lines()).toHaveLength(1);

    fetchMock.mockImplementation(async () =>
      jsonResponse(200, { tools: [...firstCatalog.slice(0, 512), tool("replacement-overflow")] }),
    );
    await createDefinitionsRaw(auth);
    expect(lines()).toHaveLength(2);
    expect(lines()[1]).toBe("LiteLLM MCP: ignoring 1 MCP tool beyond the 512-tool limit.\n");
  });

  it("reports a body-cap breach again after a clean response on the same surface", async () => {
    vi.resetModules();
    const { discoverMcpTools: discoverToolsRaw } = await import("../src/mcp-tools.js");
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const fetchMock = vi.spyOn(globalThis, "fetch");
    const oversized = () => new Response("x".repeat(5 * 1024 * 1024 + 1), { status: 200 });
    const capLines = () =>
      stderr.mock.calls.map(([m]) => String(m)).filter((m) => m.includes("MCP discovery response exceeded"));

    fetchMock.mockImplementation(async () => oversized());
    await expect(discoverToolsRaw("https://litellm.example.com", "sk-test")).rejects.toThrow();
    expect(capLines()).toHaveLength(1);

    // A clean response clears the incident...
    fetchMock.mockImplementation(async () => jsonResponse(200, { tools: [] }));
    await discoverToolsRaw("https://litellm.example.com", "sk-test");

    // ...so the same breach is reported again rather than suppressed as unchanged.
    fetchMock.mockImplementation(async () => oversized());
    await expect(discoverToolsRaw("https://litellm.example.com", "sk-test")).rejects.toThrow();
    expect(capLines()).toHaveLength(2);
  });
});

describe("findSchemaHazard", () => {
  it("reports a regex wherever the validator could compile one", () => {
    expect(findSchemaHazard({ type: "object", properties: { s: { pattern: "^a+$" } } })).toBe("regex");
    expect(findSchemaHazard({ type: "object", patternProperties: { "^a+$": { type: "string" } } })).toBe("regex");
  });

  it("reports references whose target is outside the supplied document", () => {
    expect(findSchemaHazard({ $ref: "https://evil.example/x.json" })).toBe("nonlocal-ref");
    expect(findSchemaHazard({ $dynamicRef: "#meta" })).toBe("nonlocal-ref");
    expect(findSchemaHazard({ $recursiveRef: "#" })).toBe("nonlocal-ref");
  });

  it.each(["$ref", "$dynamicRef", "$recursiveRef"])("reports a non-string %s keyword", (keyword) => {
    expect(findSchemaHazard({ type: "object", properties: { s: { [keyword]: 7 } } })).toBe("malformed-ref");
  });

  it.each(["$id", "$anchor", "$dynamicAnchor", "$recursiveAnchor"])(
    "degrades schemas that define the TypeBox resolver scope with %s",
    (keyword) => {
      expect(findSchemaHazard({ type: "object", [keyword]: keyword === "$recursiveAnchor" ? true : "alias" })).toBe(
        "resolver-scope",
      );
    },
  );

  it("degrades a nested $id self-cycle and a root $anchor alias", () => {
    expect(
      findSchemaHazard({
        type: "object",
        $defs: { nested: { $id: "nested", $ref: "#" } },
        properties: { s: { $ref: "#/$defs/nested" } },
      }),
    ).toBe("resolver-scope");
    expect(findSchemaHazard({ $anchor: "root", type: "object", properties: { s: { $ref: "#root" } } })).toBe(
      "resolver-scope",
    );
  });

  it("rejects percent-encoded local refs before TypeBox can decode them to a different target", () => {
    expect(
      findSchemaHazard({
        type: "object",
        $defs: {
          "a%2Fdependencies": { type: "string" },
          a: { dependencies: { pattern: "^(a+)+$" } },
        },
        properties: { s: { $ref: "#/$defs/a%2Fdependencies" } },
      }),
    ).toBe("unresolvable-ref");
    expect(
      findSchemaHazard({
        type: "object",
        $defs: { s: { type: "string" } },
        properties: { s: { $ref: "#/$defs/%73" } },
      }),
    ).toBe("unresolvable-ref");
  });

  it("only resolves local references to independently valid schemas", () => {
    expect(
      findSchemaHazard({
        type: "object",
        properties: { s: { $ref: "#/dependencies" } },
        dependencies: { pattern: "^(a+)+$" },
      }),
    ).toBe("unresolvable-ref");
    expect(
      findSchemaHazard({
        type: "object",
        properties: { s: { $ref: "#/dependencies" } },
        dependencies: { patternProperties: { "^(a+)+$": { type: "string" } } },
      }),
    ).toBe("unresolvable-ref");
    expect(
      findSchemaHazard({
        type: "object",
        default: { type: "object", properties: 5 },
        properties: { s: { $ref: "#/default" } },
      }),
    ).toBe("unresolvable-ref");
    expect(
      findSchemaHazard({
        type: "object",
        default: { type: "object", properties: {} },
        properties: { s: { $ref: "#/default" } },
      }),
    ).toBe("unresolvable-ref");
    expect(
      findSchemaHazard({
        type: "object",
        examples: [{ type: "array", items: 5 }],
        properties: { s: { $ref: "#/examples/0" } },
      }),
    ).toBe("unresolvable-ref");

    for (const schemaPositionFirst of [false, true]) {
      const sharedSchema = { type: "object", properties: {} };
      const sharedPositions = schemaPositionFirst
        ? { $defs: { sharedSchema }, default: sharedSchema }
        : { default: sharedSchema, $defs: { sharedSchema } };
      expect(
        findSchemaHazard({
          type: "object",
          ...sharedPositions,
          properties: { s: { $ref: "#/default" } },
        }),
      ).toBeUndefined();

      const sharedHazard = { pattern: "^(a+)+$" };
      const hazardousPositions = schemaPositionFirst
        ? { $defs: { sharedHazard }, default: sharedHazard }
        : { default: sharedHazard, $defs: { sharedHazard } };
      expect(findSchemaHazard({ type: "object", ...hazardousPositions, properties: {} })).toBe("regex");
    }
    expect(
      findSchemaHazard({ type: "object", $defs: { s: { type: "string" } }, properties: { s: { $ref: "#/$defs/s" } } }),
    ).toBeUndefined();
    expect(findSchemaHazard({ type: "object", properties: { s: { $ref: "#" } } })).toBeUndefined();
  });

  it("degrades rather than recursing without bound on a deep graph", () => {
    let deep: Record<string, unknown> = { type: "string" };
    for (let level = 0; level < 40; level += 1) deep = { wrapper: deep };
    expect(findSchemaHazard(deep)).toBe("budget");
  });

  it("degrades rather than walking without bound on a wide graph", () => {
    // Well inside the depth and serialized-size limits, but past the node budget.
    const wide = { type: "object", enum: Array.from({ length: 25_000 }, (_, index) => index) };
    expect(findSchemaHazard(wide)).toBe("budget");
  });

  it("degrades on a cyclic graph instead of looping forever", () => {
    const cyclic: Record<string, unknown> = { type: "object" };
    cyclic.self = cyclic;
    expect(findSchemaHazard(cyclic)).toBe("cycle");
  });

  it("returns no hazard for ordinary schemas, including keyword names used as argument names", () => {
    expect(findSchemaHazard({ type: "object", properties: { q: { type: "string" } } })).toBeUndefined();
    expect(
      findSchemaHazard({
        type: "object",
        properties: { pattern: { type: "string" }, patternProperties: { type: "string" } },
      }),
    ).toBeUndefined();
    expect(findSchemaHazard({ type: "object", properties: { s: { enum: ["pattern"] } } })).toBeUndefined();
  });
});

// The guard's own string-absence assertions cannot catch a hazard class nobody anticipated: a
// pointer to a non-schema, or a ref cycle, contains no forbidden substring. This suite instead
// exercises the consumer being protected — it compiles every registered schema with the same
// TypeBox the runtime uses and watches the RegExp constructor — so a future escape fails here
// regardless of what shape it takes.
describe("every registered schema is safe for the validator that consumes it", () => {
  const auth = async () => ({ baseUrl: "https://litellm.example.com", apiKey: "sk-test" });
  const EVIL = "^(a+)+$";
  const REMOTE = "https://evil.example/x.json";

  const hostileCatalog = [
    { name: "clean", server_name: "srv", inputSchema: { type: "object", properties: { q: { type: "string" } } } },
    { name: "bare", server_name: "srv", inputSchema: { type: "object" } },
    { name: "schemaless", server_name: "srv" },
    { name: "direct", server_name: "srv", inputSchema: { type: "object", properties: { s: { pattern: EVIL } } } },
    { name: "patternprops", server_name: "srv", inputSchema: { type: "object", patternProperties: { [EVIL]: {} } } },
    {
      name: "viadeps",
      server_name: "srv",
      inputSchema: {
        type: "object",
        properties: { k: {} },
        dependencies: { k: { properties: { s: { pattern: EVIL } } } },
      },
    },
    {
      name: "refdependenciespattern",
      server_name: "srv",
      inputSchema: {
        type: "object",
        properties: { s: { $ref: "#/dependencies" } },
        dependencies: { pattern: EVIL },
      },
    },
    {
      name: "refdependenciespatternproperties",
      server_name: "srv",
      inputSchema: {
        type: "object",
        properties: { s: { $ref: "#/dependencies" } },
        dependencies: { patternProperties: { [EVIL]: { type: "string" } } },
      },
    },
    {
      name: "refpercentdecoy",
      server_name: "srv",
      inputSchema: {
        type: "object",
        $defs: {
          "a%2Fdependencies": { type: "string" },
          a: { dependencies: { pattern: EVIL } },
        },
        properties: { s: { $ref: "#/$defs/a%2Fdependencies" } },
      },
    },
    {
      name: "viaadditems",
      server_name: "srv",
      inputSchema: { type: "object", properties: { a: { type: "array", additionalItems: { pattern: EVIL } } } },
    },
    {
      name: "refintodefault",
      server_name: "srv",
      inputSchema: { type: "object", default: { e: { pattern: EVIL } }, properties: { s: { $ref: "#/default/e" } } },
    },
    {
      name: "refintoexamples",
      server_name: "srv",
      inputSchema: { type: "object", examples: [{ pattern: EVIL }], properties: { s: { $ref: "#/examples/0" } } },
    },
    {
      name: "refmalformeddefault",
      server_name: "srv",
      inputSchema: {
        type: "object",
        default: { type: "object", properties: 5 },
        properties: { s: { $ref: "#/default" } },
      },
    },
    {
      name: "refschemalookingdefault",
      server_name: "srv",
      inputSchema: {
        type: "object",
        default: { type: "object", properties: {} },
        properties: { s: { $ref: "#/default" } },
      },
    },
    { name: "remoteref", server_name: "srv", inputSchema: { type: "object", properties: { s: { $ref: REMOTE } } } },
    { name: "dynref", server_name: "srv", inputSchema: { type: "object", properties: { s: { $dynamicRef: "#m" } } } },
    { name: "malformedref", server_name: "srv", inputSchema: { type: "object", properties: { s: { $ref: 7 } } } },
    // Pointers that resolve to something that is not a subschema, or not at all.
    {
      name: "refnonschema",
      server_name: "srv",
      inputSchema: { type: "object", description: "hi", properties: { s: { $ref: "#/description" } } },
    },
    {
      name: "refmissing",
      server_name: "srv",
      inputSchema: { type: "object", properties: { s: { $ref: "#/$defs/missing" } } },
    },
    {
      name: "refcycle",
      server_name: "srv",
      inputSchema: {
        type: "object",
        $defs: { a: { $ref: "#/$defs/b" }, b: { $ref: "#/$defs/a" } },
        properties: { s: { $ref: "#/$defs/a" } },
      },
    },
    {
      name: "refcombinatorcycle",
      server_name: "srv",
      inputSchema: {
        type: "object",
        $defs: {
          a: { allOf: [{ $ref: "#/$defs/b" }] },
          b: { allOf: [{ $ref: "#/$defs/a" }] },
        },
        properties: { s: { $ref: "#/$defs/a" } },
      },
    },
    {
      name: "refanchor",
      server_name: "srv",
      inputSchema: { type: "object", properties: { s: { $ref: "#someAnchor" } } },
    },
    {
      name: "nestedidselfcycle",
      server_name: "srv",
      inputSchema: {
        type: "object",
        $defs: { nested: { $id: "nested", $ref: "#" } },
        properties: { s: { $ref: "#/$defs/nested" } },
      },
    },
    {
      name: "rootanchoralias",
      server_name: "srv",
      inputSchema: { $anchor: "root", type: "object", properties: { s: { $ref: "#root" } } },
    },
    // Legitimate local refs, which must survive as passthrough.
    {
      name: "goodref",
      server_name: "srv",
      $comment: "resolves to a real subschema",
      inputSchema: { type: "object", $defs: { s: { type: "string" } }, properties: { s: { $ref: "#/$defs/s" } } },
    },
    { name: "selfref", server_name: "srv", inputSchema: { type: "object", properties: { s: { $ref: "#" } } } },
  ];

  it("compiles and runs without throwing, and never compiles a proxy-supplied expression", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { tools: hostileCatalog }));
    const definitions = await createMcpToolDefinitions(auth);
    expect(definitions.length).toBeGreaterThan(10);

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
      for (const definition of definitions) {
        // Compiling is where TypeBox resolves refs and builds regexes; Check is where it runs them.
        const validator = Compile(definition.parameters as never);
        expect(() => validator.Check({ args: {}, q: "x", s: "x", k: "x", a: ["x"] })).not.toThrow();
      }
    } finally {
      globalThis.RegExp = NativeRegExp;
    }

    // No proxy-supplied expression or reference may have been compiled or embedded.
    for (const source of compiled) {
      expect(source).not.toContain("(a+)+");
      expect(source).not.toContain("evil.example");
    }
    const serialized = JSON.stringify(definitions.map((definition) => definition.parameters));
    expect(serialized).not.toContain(EVIL);
    expect(serialized).not.toContain(REMOTE);
    expect(serialized).not.toContain("$dynamicRef");
    expect(serialized).not.toContain('"$id":"nested"');
    expect(serialized).not.toContain('"$anchor":"root"');
  });

  it("keeps legitimate $defs and document-root references as passthrough", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        tools: [
          {
            name: "goodref",
            server_name: "srv",
            inputSchema: { type: "object", $defs: { s: { type: "string" } }, properties: { s: { $ref: "#/$defs/s" } } },
          },
          {
            name: "selfref",
            server_name: "srv",
            inputSchema: { type: "object", properties: { s: { $ref: "#" } } },
          },
          {
            name: "escapedref",
            server_name: "srv",
            inputSchema: {
              type: "object",
              $defs: { "a/b": { type: "string" } },
              properties: { s: { $ref: "#/$defs/a~1b" } },
            },
          },
          {
            name: "dependencyref",
            server_name: "srv",
            inputSchema: {
              type: "object",
              dependencies: { enabled: { type: "object", properties: { token: { type: "string" } } } },
              properties: { enabled: { type: "boolean" }, guarded: { $ref: "#/dependencies/enabled" } },
            },
          },
        ],
      }),
    );
    const { definitions, report } = await createMcpToolDefinitionsRaw(auth);
    expect(report.enveloped).toBe(0);
    expect(definitions).toHaveLength(4);
    expect(definitions.map((definition) => JSON.stringify(definition.parameters))).toEqual([
      expect.stringContaining('"$ref":"#/$defs/s"'),
      expect.stringContaining('"$ref":"#"'),
      expect.stringContaining('"$ref":"#/$defs/a~1b"'),
      expect.stringContaining('"$ref":"#/dependencies/enabled"'),
    ]);
  });
});

describe("reference resolution hazards", () => {
  it("reports a local pointer that does not resolve to a subschema", () => {
    expect(findSchemaHazard({ type: "object", description: "hi", properties: { s: { $ref: "#/description" } } })).toBe(
      "unresolvable-ref",
    );
    expect(findSchemaHazard({ type: "object", properties: { s: { $ref: "#/$defs/missing" } } })).toBe(
      "unresolvable-ref",
    );
    // `#name` is an $anchor reference, which cannot be resolved without an anchor index.
    expect(findSchemaHazard({ type: "object", properties: { s: { $ref: "#someAnchor" } } })).toBe("unresolvable-ref");
  });

  it("does not resolve JSON pointers through the prototype chain", () => {
    const inherited = Object.create({ hidden: { type: "string" } }) as Record<string, unknown>;
    inherited.type = "object";
    inherited.properties = { value: { $ref: "#/hidden" } };

    expect(findSchemaHazard(inherited)).toBe("unresolvable-ref");
  });

  it.each(["__proto__", "constructor", "prototype"])("rejects the dangerous pointer token %s like TypeBox", (token) => {
    const definitions = Object.create(null) as Record<string, unknown>;
    definitions[token] = { type: "string" };
    expect(
      findSchemaHazard({
        type: "object",
        $defs: definitions,
        properties: { value: { $ref: `#/$defs/${token}` } },
      }),
    ).toBe("unresolvable-ref");
  });

  it("reports reference cycles hidden behind same-instance schema combinators", () => {
    expect(
      findSchemaHazard({
        type: "object",
        $defs: { a: { $ref: "#/$defs/b" }, b: { $ref: "#/$defs/a" } },
        properties: { s: { $ref: "#/$defs/a" } },
      }),
    ).toBe("ref-cycle");
    expect(
      findSchemaHazard({
        type: "object",
        $defs: {
          a: { allOf: [{ $ref: "#/$defs/b" }] },
          b: { allOf: [{ $ref: "#/$defs/a" }] },
        },
        properties: { s: { $ref: "#/$defs/a" } },
      }),
    ).toBe("ref-cycle");
  });

  it("accepts productive recursive references through nested values", () => {
    expect(
      findSchemaHazard({
        type: "object",
        $defs: {
          node: {
            type: "object",
            properties: { next: { anyOf: [{ type: "null" }, { $ref: "#/$defs/node" }] } },
          },
        },
        properties: { root: { $ref: "#/$defs/node" } },
      }),
    ).toBeUndefined();
  });

  it("accepts local pointers that resolve to a usable subschema", () => {
    expect(
      findSchemaHazard({ type: "object", $defs: { s: { type: "string" } }, properties: { s: { $ref: "#/$defs/s" } } }),
    ).toBeUndefined();
    // A boolean is a valid subschema, and `#` is the document itself.
    expect(
      findSchemaHazard({ type: "object", $defs: { s: true }, properties: { s: { $ref: "#/$defs/s" } } }),
    ).toBeUndefined();
    expect(findSchemaHazard({ type: "object", properties: { s: { $ref: "#" } } })).toBeUndefined();
    // Escaped pointer tokens and canonical array indices resolve correctly.
    expect(
      findSchemaHazard({
        type: "object",
        $defs: { "a/b": { type: "string" } },
        properties: { s: { $ref: "#/$defs/a~1b" } },
      }),
    ).toBeUndefined();
    expect(
      findSchemaHazard({
        type: "object",
        allOf: [{ type: "string" }],
        properties: { s: { $ref: "#/allOf/0" } },
      }),
    ).toBeUndefined();
    expect(
      findSchemaHazard({
        type: "object",
        dependencies: { enabled: { type: "object", properties: { token: { type: "string" } } } },
        properties: { guarded: { $ref: "#/dependencies/enabled" } },
      }),
    ).toBeUndefined();
    expect(
      findSchemaHazard({
        type: "object",
        allOf: [{ type: "string" }],
        properties: { s: { $ref: "#/allOf/00" } },
      }),
    ).toBe("unresolvable-ref");
  });

  it("follows a chain of local references and rejects one that leaves the document", () => {
    expect(
      findSchemaHazard({
        type: "object",
        $defs: { a: { $ref: "#/$defs/b" }, b: { type: "string" } },
        properties: { s: { $ref: "#/$defs/a" } },
      }),
    ).toBeUndefined();
    expect(
      findSchemaHazard({
        type: "object",
        $defs: { a: { $ref: "https://evil.example/x.json" } },
        properties: { s: { $ref: "#/$defs/a" } },
      }),
    ).toBe("nonlocal-ref");
  });
});

describe("unrecognized discovery body shapes", () => {
  it.each([
    ["an object without tools", { detail: "proxy-owned-error" }],
    ["a non-array tools container", { tools: { a: { name: "x" } } }],
  ])("rejects %s instead of reporting an empty catalog", async (_label, body) => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, body));
    await expect(discoverMcpToolsRaw("https://litellm.example.com", "sk-test")).rejects.toThrow(
      "MCP discovery returned an unexpected body shape",
    );
  });

  it("still accepts both documented shapes", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, []));
    expect((await discoverMcpToolsRaw("https://litellm.example.com", "sk-test")).raw).toBe(0);
    vi.restoreAllMocks();
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { tools: [] }));
    expect((await discoverMcpToolsRaw("https://litellm.example.com", "sk-test")).raw).toBe(0);
  });
});

describe("bounded diagnostic labels", () => {
  it("bounds a normalization label built from a huge proxy-supplied name", async () => {
    vi.resetModules();
    const { createMcpToolDefinitions: createDefinitionsRaw } = await import("../src/mcp-tools.js");
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { tools: [{ name: "A".repeat(200_000) }] }));

    await createDefinitionsRaw(async () => ({ baseUrl: "https://litellm.example.com", apiKey: "sk-test" }));

    const line = stderr.mock.calls.map(([m]) => String(m)).find((m) => m.includes("missing name or server identity"));
    expect(line).toBeDefined();
    // Well under the 200 KB a charset-only sanitizer would have emitted.
    expect(Buffer.byteLength(line ?? "", "utf8")).toBeLessThan(400);
    expect(line).toContain("…");
  });
});

describe("regex keywords are refused regardless of value type", () => {
  it("degrades a non-string pattern so its text never reaches the registered schema", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        tools: [
          {
            name: "arraypattern",
            server_name: "srv",
            inputSchema: { type: "object", properties: { s: { type: "string", pattern: ["^(a+)+$"] } } },
          },
          {
            name: "objpattern",
            server_name: "srv",
            inputSchema: { type: "object", properties: { s: { pattern: { regex: "^(a+)+$" } } } },
          },
        ],
      }),
    );
    const { definitions, report } = await createMcpToolDefinitionsRaw(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));

    expect(definitions).toHaveLength(2);
    expect(report.enveloped).toBe(2);
    expect(report.degraded).toEqual([{ reason: "schema-envelope", tools: expect.any(Array) }]);
    // This TypeBox would not compile either, but the expression must not be forwarded regardless.
    expect(JSON.stringify(definitions.map((definition) => definition.parameters))).not.toContain("(a+)+");
  });

  it("drops structurally invalid schema maps and dependencies before the hazard scan sees them", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        tools: [
          { name: "arr", server_name: "srv", inputSchema: { type: "object", patternProperties: [{ "^a+$": {} }] } },
          {
            name: "scalar-dependency",
            server_name: "srv",
            inputSchema: { type: "object", dependencies: { enabled: "^(a+)+$" } },
          },
          {
            name: "mixed-dependency-array",
            server_name: "srv",
            inputSchema: { type: "object", dependencies: { enabled: ["other", 7] } },
          },
        ],
      }),
    );
    const { definitions, report } = await createMcpToolDefinitionsRaw(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));
    expect(definitions).toEqual([]);
    expect(Object.fromEntries(report.dropped.map((e) => [e.reason, e.tools.length]))).toEqual({ "invalid-schema": 3 });
  });

  it("accepts schema-valued and string-array dependencies", async () => {
    const schemas = [
      { type: "object", dependencies: { enabled: { type: "object", properties: { token: { type: "string" } } } } },
      { type: "object", dependencies: { enabled: ["token"] } },
    ];
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, {
        tools: schemas.map((inputSchema, index) => ({
          name: `valid-dependency-${index}`,
          server_name: "srv",
          inputSchema,
        })),
      }),
    );
    const { definitions, report } = await createMcpToolDefinitionsRaw(async () => ({
      baseUrl: "https://litellm.example.com",
      apiKey: "sk-test",
    }));
    expect(report.enveloped).toBe(0);
    expect(definitions.map((definition) => definition.parameters)).toEqual(schemas);
  });

  it("still leaves a property literally named pattern alone", () => {
    expect(
      findSchemaHazard({
        type: "object",
        properties: { pattern: { type: "string" }, patternProperties: { type: "number" } },
      }),
    ).toBeUndefined();
  });
});

describe("upstream assumption: TypeBox escapes proxy-supplied property names", () => {
  // Property names reach `new RegExp` via additionalProperties, and are safe only because TypeBox
  // escapes them. Nothing in this repo controls that, so pin the assumption: if a future TypeBox
  // stops escaping, this fails here rather than becoming an unbounded regex at runtime.
  it("does not treat a regex-metacharacter property name as an expression", async () => {
    const { Compile } = await import("typebox/compile");
    const schema = {
      type: "object",
      properties: { "(a+)+$": { type: "string" } },
      additionalProperties: false,
    };
    const compiled: string[] = [];
    const NativeRegExp = globalThis.RegExp;
    class RecordingRegExp extends NativeRegExp {
      constructor(source: string | RegExp, flags?: string) {
        compiled.push(String(source));
        super(source as never, flags);
      }
    }
    globalThis.RegExp = RecordingRegExp as never;
    let validator: { Check: (value: unknown) => boolean };
    try {
      validator = Compile(schema as never) as never;
      // A name that would match the unescaped expression must not be accepted as that property.
      expect(validator.Check({ aaaa: "x" })).toBe(false);
      expect(validator.Check({ "(a+)+$": "x" })).toBe(true);
    } finally {
      globalThis.RegExp = NativeRegExp;
    }
    // Every compiled source must contain the escaped literal, never the raw quantifier group.
    for (const source of compiled) expect(source).not.toContain("(a+)+$");
  });
});

// Two upstream behaviors keep proxy-supplied schemas safe that this repo does not control. Pin them
// so a dependency bump that changes either one fails here rather than at runtime.
describe("upstream assumptions the guard depends on", () => {
  const auth = async () => ({ baseUrl: "https://litellm.example.com", apiKey: "sk-test" });

  async function registeredParametersFor(inputSchema: unknown): Promise<unknown> {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, { tools: [{ name: "probe", server_name: "srv", inputSchema }] }),
    );
    const definitions = await createMcpToolDefinitions(auth);
    expect(definitions).toHaveLength(1);
    return definitions[0]?.parameters;
  }

  it("builds no regex from a passthrough schema's property names", async () => {
    // typebox's value/convert/from_object.mjs turns `properties` keys into `new RegExp(`^${key}$`)`
    // with NO escaping, and pi-ai calls Value.Convert on tool parameters. That is only harmless
    // because Convert walks recognized TypeBox types and no-ops on a raw JSON Schema, so a
    // proxy-supplied property name never reaches it. If that changes, a name like `(a+)+$` becomes
    // an executable backtracking regex tested against model-supplied argument keys.
    const parameters = await registeredParametersFor({
      type: "object",
      properties: { "(a+)+$": { type: "string" }, "(x+x+)+y": { type: "string" } },
    });
    // Not a public export, so reached by file path: the point is to exercise the real consumer.
    const { validateToolArguments } = await import("../node_modules/@earendil-works/pi-ai/dist/utils/validation.js");

    const built: string[] = [];
    const NativeRegExp = globalThis.RegExp;
    class RecordingRegExp extends NativeRegExp {
      constructor(source: string | RegExp, flags?: string) {
        built.push(String(source));
        super(source as never, flags);
      }
    }
    globalThis.RegExp = RecordingRegExp as never;
    try {
      // Long non-matching argument keys: the shape that would backtrack if a regex were built.
      const args = { [`${"a".repeat(30)}!`]: "v", [`${"x".repeat(30)}!`]: "v" };
      expect(() =>
        validateToolArguments({ name: "probe", parameters } as never, { name: "probe", arguments: args } as never),
      ).not.toThrow();
    } finally {
      globalThis.RegExp = NativeRegExp;
    }

    for (const source of built) {
      expect(source).not.toContain("(a+)+");
      expect(source).not.toContain("(x+x+)+y");
    }

    // The assertion above is only meaningful if the recorder actually captures constructions, so
    // prove it on a path that is known to build them: our own envelope is a real TypeBox type, and
    // Convert does walk it.
    const envelopeParameters = await registeredParametersFor({});
    const envelopeBuilt: string[] = [];
    class RecordingEnvelopeRegExp extends NativeRegExp {
      constructor(source: string | RegExp, flags?: string) {
        envelopeBuilt.push(String(source));
        super(source as never, flags);
      }
    }
    globalThis.RegExp = RecordingEnvelopeRegExp as never;
    try {
      validateToolArguments(
        { name: "probe", parameters: envelopeParameters } as never,
        { name: "probe", arguments: { args: { k: 1 } } } as never,
      );
    } finally {
      globalThis.RegExp = NativeRegExp;
    }
    expect(envelopeBuilt).toContain("^args$");
  });

  it("coerces supported primitive values and honors extra-property constraints", async () => {
    const { validateToolArguments } = await import("../node_modules/@earendil-works/pi-ai/dist/utils/validation.js");
    const validate = (parameters: unknown, args: unknown) =>
      validateToolArguments({ name: "probe", parameters } as never, { name: "probe", arguments: args } as never);

    const typed = await registeredParametersFor({
      type: "object",
      properties: { n: { type: "number" } },
      required: ["n"],
    });
    expect(validate(typed, { n: "42" })).toEqual({ n: 42 });
    expect(() => validate(typed, { n: "not-a-number" })).toThrow();

    for (const constraint of [{ additionalProperties: false }, { unevaluatedProperties: false }]) {
      const constrained = await registeredParametersFor({
        type: "object",
        properties: { known: { type: "string" } },
        ...constraint,
      });
      expect(() => validate(constrained, { known: "yes", extra: true })).toThrow();
    }

    const typedExtras = await registeredParametersFor({ type: "object", additionalProperties: { type: "number" } });
    expect(validate(typedExtras, { extra: "7" })).toEqual({ extra: 7 });
    const unconstrained = await registeredParametersFor({ type: "object" });
    expect(validate(unconstrained, { extra: true })).toEqual({ extra: true });
  });

  it("evaluates `format` on a passthrough schema, so its regexes are library-supplied and not inert", async () => {
    // Recorded deliberately: `format` is a proxy-chosen selector of typebox's own regexes, executed
    // against model-supplied strings. The shipped formats are well-anchored, so this is a residual
    // dependency on upstream regex quality rather than a hole — but it is live, not ignored, and
    // anyone assuming otherwise would mis-assess it.
    const parameters = await registeredParametersFor({
      type: "object",
      properties: { s: { type: "string", format: "email" } },
      required: ["s"],
    });
    const { validateToolArguments } = await import("../node_modules/@earendil-works/pi-ai/dist/utils/validation.js");
    const call = (value: string) =>
      validateToolArguments(
        { name: "probe", parameters } as never,
        { name: "probe", arguments: { s: value } } as never,
      );

    expect(() => call("definitely not an email")).toThrow();
    expect(() => call("a@b.co")).not.toThrow();
  });
});

describe("credentialFingerprint", () => {
  const KEY = "sk-super-secret-value-must-not-appear";

  it("holds no credential material and is not reversible by inspection", () => {
    const fingerprint = credentialFingerprint(KEY, { "x-proxy-auth": "header-secret-value" });
    expect(fingerprint).not.toContain(KEY);
    expect(fingerprint).not.toContain("super-secret");
    expect(fingerprint).not.toContain("header-secret-value");
    expect(fingerprint).toMatch(/^[a-f0-9]{32}$/);
  });

  it("changes when any credential material changes, so a catalog notices", () => {
    const base = credentialFingerprint(KEY);
    expect(credentialFingerprint(`${KEY}-rotated`)).not.toBe(base);
    // Headers carry authorization material too, so they must participate.
    expect(credentialFingerprint(KEY, { "x-proxy-auth": "a" })).not.toBe(base);
    expect(credentialFingerprint(KEY, { "x-proxy-auth": "a" })).not.toBe(
      credentialFingerprint(KEY, { "x-proxy-auth": "b" }),
    );
    // An empty header record adds no credential material.
    expect(credentialFingerprint(KEY, {})).toBe(base);
  });

  it("is stable for identical material regardless of header ordering", () => {
    const left = credentialFingerprint(KEY, { b: "2", a: "1" });
    const right = credentialFingerprint(KEY, { a: "1", b: "2" });
    expect(left).toBe(right);
  });

  it("is salted per process, so an identical key does not yield a predictable digest", async () => {
    const own = credentialFingerprint(KEY);
    // A bare digest of the key would be reproducible by anyone holding a candidate key; this must not be.
    const bareDigest = createHash("sha256").update(KEY).digest("hex").slice(0, 32);
    expect(own).not.toBe(bareDigest);
    // A freshly loaded module has a different salt, confirming the salt is real and per process.
    vi.resetModules();
    const { credentialFingerprint: fresh } = await import("../src/mcp-tools.js");
    expect(fresh(KEY)).not.toBe(own);
  });
});

describe("review hardening regressions", () => {
  it("rejects an explicitly null input schema", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [{ name: "bad", server_name: "srv", inputSchema: null }]),
    );
    const result = await createMcpToolDefinitionsRaw(async () => ({
      baseUrl: "https://proxy.example.com",
      apiKey: "sk-test",
    }));
    expect(result.definitions).toEqual([]);
    expect(result.report.dropped).toEqual([
      { reason: "invalid-schema", tools: [expect.stringMatching(/^mcp_srv_bad_[a-f0-9]{10}$/)] },
    ]);
  });

  it("keeps NUL-bearing identity tuples distinct", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, [
        { name: "c", server_name: "b", server_id: "a\0b", inputSchema: {} },
        { name: "b\0c", server_name: "b", server_id: "a", inputSchema: {} },
      ]),
    );
    const result = await createMcpToolDefinitionsRaw(async () => ({
      baseUrl: "https://proxy.example.com",
      apiKey: "sk-test",
    }));
    expect(result.definitions).toHaveLength(2);
    expect(result.report.dropped).toEqual([]);
  });
});
