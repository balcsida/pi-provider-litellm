import type { Static, TSchema } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createMcpToolDefinitions, discoverMcpTools, executeMcpTool } from "../src/mcp-tools.js";
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

describe("discoverMcpTools", () => {
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

  it("rejects when discovery fails", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("offline"));

    await expect(discoverMcpTools("https://litellm.example.com", "sk-test")).rejects.toThrow("offline");
  });
});

describe("executeMcpTool", () => {
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

  it("retries retryable HTTP failures once", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(jsonResponse(503, { error: "busy" }))
      .mockResolvedValueOnce(jsonResponse(200, { result: "ok" }));

    await expect(
      executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", { query: "pi" }),
    ).resolves.toBe(JSON.stringify("ok", null, 2));

    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("does not retry non-retryable HTTP failures", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(401, { error: "nope" }));

    await expect(
      executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", { query: "pi" }),
    ).resolves.toBe("Error calling search on brave: HTTP 401");

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("does not retry response parse failures after a successful HTTP response", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(new Response("not json", { status: 200, headers: { "content-type": "application/json" } }));

    await expect(
      executeMcpTool("https://litellm.example.com", "sk-test", "brave", "search", { query: "pi" }),
    ).resolves.toMatch(/^Error calling search on brave:/);

    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});

describe("createMcpToolDefinitions", () => {
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

    const definitions = await createMcpToolDefinitions("https://litellm.example.com", async () => "sk-test");

    expect(definitions.map((tool) => tool.name)).toEqual(["mcp_brave_api_web_search"]);
    expect(definitions[0]?.executionMode).toBe("parallel");
    expect(definitions[0]?.description).toBe("Search the web (via Brave API MCP server)");
    const parameters = definitions[0]?.parameters as { required?: string[]; properties?: Record<string, unknown> };
    expect(parameters.required).toEqual(["query"]);
    expect(Object.keys(parameters.properties ?? {})).toEqual(["query", "limit", "safe", "tags"]);
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

    const definitions = await createMcpToolDefinitions("https://litellm.example.com", async () => "sk-test");

    expect(definitions[0]?.parameters).toMatchObject({
      properties: { nested: { type: "object", properties: { value: { type: "string" } } } },
      required: ["nested"],
    });
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
    const getApiKey = vi.fn().mockResolvedValueOnce("discovery-token").mockResolvedValueOnce("execution-token");

    const definitions = await createMcpToolDefinitions("https://litellm.example.com", getApiKey);
    type Params = Static<TSchema>;
    const result = await definitions[0]?.execute(
      "call-1",
      { query: "pi" } as Params,
      undefined,
      undefined,
      {} as never,
    );

    expect(result?.content).toEqual([{ type: "text", text: JSON.stringify("ok", null, 2) }]);
    expect(getApiKey).toHaveBeenCalledTimes(2);
    expect(vi.mocked(globalThis.fetch).mock.calls[1]?.[1]).toMatchObject({
      headers: expect.objectContaining({ Authorization: "Bearer execution-token" }),
      body: JSON.stringify({ server_id: "brave-api", name: "search", arguments: { query: "pi" } }),
    });
  });
});
