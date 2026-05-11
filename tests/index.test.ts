import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

const ENV_KEYS = ["LITELLM_BASE_URL", "LITELLM_API_KEY", "LITELLM_DISCOVERY_TIMEOUT_MS", "STORED_LITELLM_KEY"];
const ORIGINAL_ENV = new Map(ENV_KEYS.map((key) => [key, process.env[key]]));

type TestProviderConfig = {
  baseUrl?: string;
  models?: unknown[];
};

type TestCommand = {
  description: string;
  handler: (args: string[], ctx: { ui: { notify: (message: string, type: string) => void } }) => Promise<void> | void;
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

type TestPi = {
  providers: Array<{ name: string; config: TestProviderConfig }>;
  commands: Map<string, TestCommand>;
  handlers: Map<string, Array<(event: any, ctx?: any) => Promise<any> | any>>;
  registerProvider(name: string, config: TestProviderConfig): void;
  registerCommand(name: string, command: TestCommand): void;
  on(event: string, handler: (event: any, ctx?: any) => Promise<any> | any): void;
};

afterEach(() => {
  for (const key of ENV_KEYS) {
    const original = ORIGINAL_ENV.get(key);
    if (original === undefined) process.env[key] = undefined;
    else process.env[key] = original;
  }
  vi.restoreAllMocks();
  vi.resetModules();
  vi.unmock("@earendil-works/pi-coding-agent");
});

describe("extension startup", () => {
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

    expect(seenAuthHeaders).toEqual(["Bearer stored-key"]);
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
    await pi.commands.get("litellm-refresh")?.handler([], { ui: { notify } });

    expect(fetchMock).not.toHaveBeenCalled();
    expect(notify).toHaveBeenCalledWith("LiteLLM refresh disabled (LITELLM_DISCOVERY_TIMEOUT_MS=0)", "warning");
  });
});
