import type { Static, TSchema } from "@earendil-works/pi-ai";
import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ToolDefinition } from "@earendil-works/pi-coding-agent";
import { normalizeBaseUrl } from "./discover.js";
import type { LiteLLMMcpTool } from "./types.js";

const LIST_TIMEOUT_MS = 10_000;
const CALL_TIMEOUT_MS = 30_000;
const MCP_RETRY_DELAY_MS = 350;
const MCP_RETRY_ATTEMPTS = 1;
const RETRYABLE_HTTP_STATUS = new Set([408, 425, 429, 500, 502, 503, 504]);

interface McpExecutionResult {
  text: string;
  retryable: boolean;
}

interface RawLiteLLMMcpTool {
  name?: unknown;
  description?: unknown;
  inputSchema?: unknown;
  input_schema?: unknown;
  server_id?: unknown;
  server_name?: unknown;
  mcp_info?: {
    server_id?: unknown;
    server_name?: unknown;
  };
}

function withTimeout(timeoutMs: number, parentSignal?: AbortSignal): { signal: AbortSignal; cancel: () => void } {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error(`Timed out after ${timeoutMs}ms`)), timeoutMs);
  const abort = () => controller.abort(parentSignal?.reason);
  if (parentSignal?.aborted) abort();
  else parentSignal?.addEventListener("abort", abort, { once: true });
  return {
    signal: controller.signal,
    cancel: () => {
      clearTimeout(timer);
      parentSignal?.removeEventListener("abort", abort);
    },
  };
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : undefined;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function formatMcpToolError(toolName: string, serverId: string, error: unknown): string {
  return `Error calling ${toolName} on ${serverId}: ${error instanceof Error ? error.message : String(error)}`;
}

function isRetryableTransportError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  if (error.name === "AbortError" || error.name === "TimeoutError") return true;
  if (error.message.startsWith("Timed out after ")) return true;
  if (error.name !== "TypeError") return false;
  return /fetch failed|failed to fetch|network|terminated|ECONNRESET|ECONNREFUSED|ETIMEDOUT|EAI_AGAIN|ENOTFOUND|socket/i.test(
    error.message,
  );
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function normalizeMcpTool(value: unknown): LiteLLMMcpTool | undefined {
  const raw = asRecord(value) as RawLiteLLMMcpTool | undefined;
  if (!raw) return undefined;
  const name = stringValue(raw.name);
  const serverName =
    stringValue(raw.mcp_info?.server_name) ??
    stringValue(raw.server_name) ??
    stringValue(raw.mcp_info?.server_id) ??
    stringValue(raw.server_id);
  if (!name || !serverName) return undefined;
  const inputSchema = asRecord(raw.inputSchema) ?? asRecord(raw.input_schema) ?? { type: "object", properties: {} };
  return {
    name,
    server_name: serverName,
    server_id: stringValue(raw.mcp_info?.server_id) ?? stringValue(raw.server_id) ?? serverName,
    description: stringValue(raw.description) ?? name,
    input_schema: inputSchema,
  };
}

export async function discoverMcpTools(
  baseUrl: string,
  apiKey: string,
  headers?: Record<string, string>,
  onProgress?: (message: string) => void,
  parentSignal?: AbortSignal,
): Promise<LiteLLMMcpTool[]> {
  onProgress?.("Discovering MCP tools from server...");
  const { signal, cancel } = withTimeout(LIST_TIMEOUT_MS, parentSignal);
  try {
    onProgress?.("Querying MCP tools/list endpoint...");
    const response = await fetch(`${normalizeBaseUrl(baseUrl)}/mcp-rest/tools/list`, {
      headers: {
        ...headers,
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      signal,
    });
    if (!response.ok) {
      onProgress?.(`MCP tools discovery failed with status ${response.status}`);
      throw new Error(`HTTP ${response.status}`);
    }
    const body = (await response.json()) as unknown;
    const bodyRecord = asRecord(body);
    const rawTools = Array.isArray(body) ? body : Array.isArray(bodyRecord?.tools) ? bodyRecord.tools : [];
    onProgress?.(`Found ${rawTools.length} raw MCP tools, normalizing...`);
    const tools = rawTools.map(normalizeMcpTool).filter((tool): tool is LiteLLMMcpTool => tool !== undefined);
    onProgress?.(`Successfully discovered ${tools.length} MCP tools`);
    return tools;
  } catch (error) {
    onProgress?.("MCP tools discovery encountered an error");
    throw error;
  } finally {
    cancel();
  }
}

export async function executeMcpTool(
  baseUrl: string,
  apiKey: string,
  serverId: string,
  toolName: string,
  args: Record<string, unknown>,
  headers?: Record<string, string>,
): Promise<string> {
  for (let attempt = 0; attempt <= MCP_RETRY_ATTEMPTS; attempt++) {
    const result = await executeMcpToolOnce(baseUrl, apiKey, serverId, toolName, args, headers);
    if (attempt < MCP_RETRY_ATTEMPTS && result.retryable) {
      await sleep(MCP_RETRY_DELAY_MS);
      continue;
    }
    return result.text;
  }
  return `Error calling ${toolName} on ${serverId}: retry attempts exhausted`;
}

async function executeMcpToolOnce(
  baseUrl: string,
  apiKey: string,
  serverId: string,
  toolName: string,
  args: Record<string, unknown>,
  headers?: Record<string, string>,
): Promise<McpExecutionResult> {
  const { signal, cancel } = withTimeout(CALL_TIMEOUT_MS);
  try {
    let response: Response;
    try {
      response = await fetch(`${normalizeBaseUrl(baseUrl)}/mcp-rest/tools/call`, {
        method: "POST",
        headers: {
          ...headers,
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ server_id: serverId, name: toolName, arguments: args }),
        signal,
      });
    } catch (error) {
      return {
        text: formatMcpToolError(toolName, serverId, error),
        retryable: isRetryableTransportError(error),
      };
    }

    if (!response.ok) {
      return {
        text: `Error calling ${toolName} on ${serverId}: HTTP ${response.status}`,
        retryable: RETRYABLE_HTTP_STATUS.has(response.status),
      };
    }

    try {
      const body = (await response.json()) as unknown;
      const bodyRecord = asRecord(body);
      return {
        text: JSON.stringify(bodyRecord && "result" in bodyRecord ? bodyRecord.result : body, null, 2),
        retryable: false,
      };
    } catch (error) {
      return { text: formatMcpToolError(toolName, serverId, error), retryable: false };
    }
  } catch (error) {
    return { text: formatMcpToolError(toolName, serverId, error), retryable: false };
  } finally {
    cancel();
  }
}

function sanitizeName(name: string): string {
  const sanitized = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return sanitized || "tool";
}

function buildParameters(inputSchema: Record<string, unknown>): TSchema {
  const properties = inputSchema.properties as Record<string, unknown> | undefined;
  if (!properties || typeof properties !== "object") {
    return Type.Object({
      args: Type.Record(Type.String(), Type.Unknown(), { description: "Tool arguments as key-value pairs" }),
    });
  }

  return Type.Unsafe(inputSchema);
}

export async function createMcpToolDefinitions(
  baseUrl: string,
  getApiKey: () => Promise<string>,
  headers?: Record<string, string>,
  onProgress?: (message: string) => void,
  signal?: AbortSignal,
): Promise<ToolDefinition[]> {
  const discoveryApiKey = await getApiKey();
  const tools = await discoverMcpTools(baseUrl, discoveryApiKey, headers, onProgress, signal);

  return tools.map((mcpTool) => {
    const safeServer = sanitizeName(mcpTool.server_name);
    const safeTool = sanitizeName(mcpTool.name);
    const parameters = buildParameters(mcpTool.input_schema);

    return defineTool({
      name: `mcp_${safeServer}_${safeTool}`,
      label: `${mcpTool.server_name}: ${mcpTool.name}`,
      description: `${mcpTool.description} (via ${mcpTool.server_name} MCP server)`,
      promptSnippet: `${mcpTool.description} via ${mcpTool.server_name} MCP server`,
      executionMode: "parallel",
      parameters,
      async execute(_toolCallId, params: Static<typeof parameters>) {
        const apiKey = await getApiKey();
        const rawParams = params as Record<string, unknown>;
        const args =
          Object.keys(rawParams).length === 1 && rawParams.args && typeof rawParams.args === "object"
            ? (rawParams.args as Record<string, unknown>)
            : rawParams;
        const text = await executeMcpTool(
          baseUrl,
          apiKey,
          mcpTool.server_id ?? mcpTool.server_name,
          mcpTool.name,
          args,
          headers,
        );
        return {
          content: [{ type: "text", text }],
          details: { server: mcpTool.server_name, serverId: mcpTool.server_id, tool: mcpTool.name },
        };
      },
    });
  });
}
