import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

type TestProviderConfig = {
  baseUrl?: string;
  apiKey?: string;
  api?: string;
  models?: Array<{ id: string; compat?: unknown }>;
  oauth?: unknown;
};

type TestCommand = {
  description: string;
  handler: (args: string[], ctx: { ui: { notify: (message: string, type: string) => void } }) => Promise<void> | void;
};

type TestPi = {
  providers: Array<{ name: string; config: TestProviderConfig }>;
  commands: Map<string, TestCommand>;
  handlers: Map<string, Array<(event: any, ctx?: any) => Promise<any> | any>>;
  registerProvider(name: string, config: TestProviderConfig): void;
  registerCommand(name: string, command: TestCommand): void;
  on(event: string, handler: (event: any, ctx?: any) => Promise<any> | any): void;
};

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
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

    return { AuthStorage: TestAuthStorage, getAgentDir: () => agentDir };
  });
  const mod = await import("../src/index.js");
  return mod.default as unknown as (pi: TestPi) => Promise<void>;
}

function createPi(): TestPi {
  return {
    providers: [],
    commands: new Map(),
    handlers: new Map(),
    registerProvider(name: string, config: TestProviderConfig) {
      this.providers.push({ name, config });
    },
    registerCommand(name: string, command: TestCommand) {
      this.commands.set(name, command);
    },
    on(event: string, handler: (event: any, ctx?: any) => Promise<any> | any) {
      this.handlers.set(event, [...(this.handlers.get(event) ?? []), handler]);
    },
  };
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.resetModules();
  vi.unmock("@earendil-works/pi-coding-agent");
  process.env.LITELLM_BASE_URL = undefined;
  process.env.LITELLM_API_KEY = undefined;
  process.env.LITELLM_DISCOVERY_TIMEOUT_MS = undefined;
});

describe("feature parity", () => {
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
    const updated = beforeRequest?.({ payload: { messages: [] } });
    expect(updated).toMatchObject({
      messages: [],
      litellm_session_id: "123e4567-e89b-12d3-a456-426614174000",
    });
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
    responseHandler?.({ headers: { "x-litellm-response-cost": "0.42" } });

    const endHandler = pi.handlers.get("message_end")?.[0];
    const result = await endHandler?.({
      message: {
        role: "assistant",
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
});
