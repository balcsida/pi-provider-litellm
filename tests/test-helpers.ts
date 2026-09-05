import { readFileSync } from "node:fs";
import type { Provider } from "@earendil-works/pi-ai";
import { afterEach, beforeEach, vi } from "vitest";

// Every environment variable src/ reads. Tests that assert credential or discovery
// behavior must start from a known-empty environment, not from the developer's
// shell, and must not leak into the next test.
export const MANAGED_ENV_VARS = [
  "LITELLM_API_KEY",
  "LITELLM_API_KEY_HELPER",
  "LITELLM_BASE_URL",
  "LITELLM_HEADERS",
  "LITELLM_OFFLINE",
  "LITELLM_DISCOVERY_TIMEOUT_MS",
  "LITELLM_VERBOSE_DISCOVERY",
  "LITELLM_GCLOUD_TOKEN_AUTH",
  "GOOGLE_APPLICATION_CREDENTIALS",
  "PI_OFFLINE",
  "APPDATA",
] as const;

// Returns a subprocess environment with every source-read variable cleared before
// applying explicit overrides.
export function hermeticChildEnv(overrides: NodeJS.ProcessEnv = {}): NodeJS.ProcessEnv {
  const env = { ...process.env };
  for (const name of MANAGED_ENV_VARS) delete env[name];
  return { ...env, ...overrides };
}

// Snapshots and clears the managed variables before each test and restores the
// original values afterwards, so a result never depends on ambient environment or
// on which test ran first in the file.
//
// `extra` adds suite-specific names. HOME is deliberately not in the shared set: clearing
// it for every suite would break the npm and git subprocesses tests/package.test.ts spawns,
// so only the suites that exercise ADC path discovery opt into it.
export function useHermeticEnv(extra: readonly string[] = []): void {
  const managed = [...MANAGED_ENV_VARS, ...extra];
  let saved: Array<[string, string | undefined]> = [];

  beforeEach(() => {
    saved = managed.map((name) => [name, process.env[name]]);
    for (const name of managed) delete process.env[name];
  });

  afterEach(() => {
    for (const [name, value] of saved) {
      if (value === undefined) delete process.env[name];
      else process.env[name] = value;
    }
  });
}

type TestCommandContext = {
  ui: {
    input?: (title: string, placeholder?: string) => Promise<string | undefined>;
    notify: (message: string, type: string) => void;
  };
  modelRegistry?: {
    authStorage: {
      set: (provider: string, credential: unknown) => void;
    };
    refresh?: () => void;
  };
};

type TestCommand = {
  description: string;
  handler: (args: string, ctx: TestCommandContext) => Promise<void> | void;
};

export type TestPi = {
  providers: Provider[];
  commands: Map<string, TestCommand>;
  handlers: Map<string, Array<(event: any, ctx?: any) => Promise<any> | any>>;
  tools: Array<{ name: string; description: string; execute?: (...args: any[]) => Promise<any> | any }>;
  registerProvider(provider: Provider): void;
  registerCommand(name: string, command: TestCommand): void;
  registerTool(tool: { name: string; description: string; execute?: (...args: any[]) => Promise<any> | any }): void;
  on(event: string, handler: (event: any, ctx?: any) => Promise<any> | any): void;
};

export async function loadExtension(agentDir: string): Promise<(pi: TestPi) => Promise<void>> {
  vi.resetModules();
  vi.doMock("@earendil-works/pi-coding-agent", () => ({
    defineTool: (tool: unknown) => tool,
    getAgentDir: () => agentDir,
    readStoredCredential: (provider: string, authPath: string) => {
      try {
        return (JSON.parse(readFileSync(authPath, "utf8")) as Record<string, unknown>)[provider];
      } catch {
        return undefined;
      }
    },
  }));
  const mod = await import("../src/index.js");
  return mod.default as unknown as (pi: TestPi) => Promise<void>;
}

export function createPi(): TestPi {
  return {
    providers: [],
    commands: new Map(),
    handlers: new Map(),
    tools: [],
    registerProvider(provider) {
      this.providers.push(provider);
    },
    registerCommand(name, command) {
      this.commands.set(name, command);
    },
    registerTool(tool) {
      this.tools.push(tool);
    },
    on(event, handler) {
      this.handlers.set(event, [...(this.handlers.get(event) ?? []), handler]);
    },
  };
}

export async function createLoadedPi(agentDir: string): Promise<TestPi> {
  const pi = createPi();
  await (await loadExtension(agentDir))(pi);
  return pi;
}
