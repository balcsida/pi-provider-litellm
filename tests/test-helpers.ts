import { readFileSync } from "node:fs";
import { vi } from "vitest";

type TestProviderConfig = {
  baseUrl?: string;
  apiKey?: string;
  headers?: Record<string, string>;
  api?: string;
  models?: unknown[];
  refreshModels?: (context: {
    allowNetwork: boolean;
    force?: boolean;
    signal?: AbortSignal;
    credential?:
      | { type: "api_key"; key?: string }
      | { type: "oauth"; access: string; refresh: string; expires: number; baseUrl?: string };
  }) => Promise<unknown[]>;
  oauth?: {
    login: (callbacks: {
      onPrompt: (options: { message: string; placeholder?: string }) => Promise<string>;
      onAuth?: (info: { url: string; instructions?: string }) => void;
      onProgress?: (message: string) => void;
      signal?: AbortSignal;
    }) => Promise<{ access: string; refresh: string; expires: number; baseUrl?: string }>;
    refreshToken: (credential: { access: string; refresh: string; expires: number; baseUrl?: string }) => Promise<{
      access: string;
      refresh: string;
      expires: number;
      baseUrl?: string;
    }>;
    getApiKey: (credential: { access: string; refresh: string; expires: number; baseUrl?: string }) => string;
  };
};

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
  providers: Array<{ name: string; config: TestProviderConfig }>;
  commands: Map<string, TestCommand>;
  handlers: Map<string, Array<(event: any, ctx?: any) => Promise<any> | any>>;
  tools: Array<{ name: string; description: string; execute?: (...args: any[]) => Promise<any> | any }>;
  registerProvider(name: string, config: TestProviderConfig): void;
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
    registerProvider(name, config) {
      this.providers.push({ name, config });
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
