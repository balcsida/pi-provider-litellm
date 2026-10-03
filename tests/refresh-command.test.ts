import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ModelRegistry, ModelRuntime } from "@earendil-works/pi-coding-agent";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createPi, loadExtension, type TestPi, useHermeticEnv } from "./test-helpers.js";

vi.unmock("@earendil-works/pi-coding-agent");

useHermeticEnv(["LITELLM_MODELS_DEV"]);

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

/** Records every request and answers each proxy with one route named after its host. */
function mockProxies(overrides: Record<string, () => Response> = {}): string[] {
  const seen: string[] = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input);
    seen.push(url);
    const override = overrides[url];
    if (override) return override();
    if (url.endsWith("/model/info")) {
      const route = new URL(url).hostname.split(".")[0];
      return jsonResponse(200, { data: [{ model_name: `${route}-gpt-4o`, model_info: { mode: "chat" } }] });
    }
    throw new Error(`unexpected URL: ${url}`);
  });
  return seen;
}

/** Replays Pi's startup: ModelRuntime.create reads PI_OFFLINE, and every startup refresh is offline. */
async function startup(agentDir: string): Promise<{ pi: TestPi; runtime: ModelRuntime }> {
  const runtime = await ModelRuntime.create({
    authPath: join(agentDir, "auth.json"),
    modelsPath: join(agentDir, "models.json"),
  });
  const pi = createPi();
  pi.registerProvider = (provider) => runtime.registerNativeProvider(provider);
  await (await loadExtension(agentDir))(pi);
  await runtime.refresh({ allowNetwork: false });
  return { pi, runtime };
}

async function refreshCommand(
  pi: TestPi,
  runtime: ModelRuntime,
  args = "",
  hasUI = true,
): Promise<Array<[string, string]>> {
  const command = pi.commands.get("litellm-refresh");
  if (!command) throw new Error("litellm-refresh is not registered");
  const notify = vi.fn<(message: string, level: string) => void>();
  await command.handler(args, { hasUI, ui: { notify }, modelRegistry: new ModelRegistry(runtime) } as never);
  return notify.mock.calls;
}

function ids(runtime: ModelRuntime, provider: string): string[] {
  return runtime.getModels(provider).map((model) => model.id);
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.resetModules();
});

describe("/litellm-refresh (issue #155)", () => {
  it("discovers under PI_OFFLINE, where Pi's own /model refresh stays offline", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-refresh-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    process.env.PI_OFFLINE = "1";
    const seen = mockProxies();

    const { pi, runtime } = await startup(agentDir);
    // The refresh /model starts takes Pi's default network setting, which PI_OFFLINE turns off.
    await runtime.refresh();
    expect(seen).toEqual([]);
    expect(ids(runtime, "litellm")).toEqual([]);

    const notices = await refreshCommand(pi, runtime);

    expect(seen).toContain("https://proxy.example.com/model/info");
    expect(ids(runtime, "litellm")).toEqual(["proxy-gpt-4o"]);
    expect(notices).toEqual([['LiteLLM ("litellm"): refreshed 1 model.', "info"]]);
  });

  it("keeps models.dev off under PI_OFFLINE even when enrichment is enabled", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-refresh-modelsdev-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    process.env.LITELLM_MODELS_DEV = "1";
    process.env.PI_OFFLINE = "1";
    const seen = mockProxies();

    const { pi, runtime } = await startup(agentDir);
    await refreshCommand(pi, runtime);

    expect(seen).toContain("https://proxy.example.com/model/info");
    expect(seen.filter((url) => new URL(url).hostname !== "proxy.example.com")).toEqual([]);
  });

  it.each([
    ["LITELLM_OFFLINE", "1", "LITELLM_OFFLINE=1"],
    ["LITELLM_DISCOVERY_TIMEOUT_MS", "0", "LITELLM_DISCOVERY_TIMEOUT_MS=0"],
  ])("stays offline when %s=%s", async (variable, value, reason) => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-refresh-disabled-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    process.env[variable] = value;
    const seen = mockProxies();

    const { pi, runtime } = await startup(agentDir);
    const notices = await refreshCommand(pi, runtime);

    expect(seen).toEqual([]);
    expect(notices).toEqual([[`LiteLLM: model refresh skipped (${reason}).`, "warning"]]);
  });

  it("refreshes every configured provider, or only the one named", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-refresh-alias-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    process.env.PI_OFFLINE = "1";
    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({ litellm: { providers: { team: { baseUrl: "https://team.example.com", apiKey: "team-key" } } } }),
      "utf8",
    );
    const seen = mockProxies();
    const { pi, runtime } = await startup(agentDir);

    const aliasNotices = await refreshCommand(pi, runtime, " team ");

    expect(seen.filter((url) => url.endsWith("/model/info"))).toEqual(["https://team.example.com/model/info"]);
    expect(ids(runtime, "team")).toEqual(["team-gpt-4o"]);
    expect(ids(runtime, "litellm")).toEqual([]);
    expect(aliasNotices).toEqual([['LiteLLM ("team"): refreshed 1 model.', "info"]]);

    const allNotices = await refreshCommand(pi, runtime);

    expect(ids(runtime, "litellm")).toEqual(["proxy-gpt-4o"]);
    expect(allNotices).toEqual([
      ['LiteLLM ("litellm"): refreshed 1 model.', "info"],
      ['LiteLLM ("team"): refreshed 1 model.', "info"],
    ]);
  });

  it("rejects a provider it does not own without refreshing", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-refresh-unknown-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    const seen = mockProxies();
    const { pi, runtime } = await startup(agentDir);
    seen.length = 0;

    const notices = await refreshCommand(pi, runtime, "openai");

    expect(seen).toEqual([]);
    expect(notices).toEqual([['LiteLLM: unknown provider "openai"; configured: "litellm".', "warning"]]);
  });

  it("reports missing credentials instead of an empty refresh", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-refresh-nocreds-"));
    process.env.PI_OFFLINE = "1";
    const seen = mockProxies();
    const { pi, runtime } = await startup(agentDir);

    const notices = await refreshCommand(pi, runtime);

    expect(seen).toEqual([]);
    expect(notices).toEqual([
      [
        'LiteLLM ("litellm"): model refresh skipped; no credentials for litellm. Run /login litellm or set env vars.',
        "warning",
      ],
    ]);
  });

  it("reports a failed refresh and keeps the cached catalog", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-refresh-fail-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    mockProxies();
    const { pi, runtime } = await startup(agentDir);
    expect(ids(runtime, "litellm")).toEqual(["proxy-gpt-4o"]);
    vi.restoreAllMocks();
    mockProxies({ "https://proxy.example.com/model/info": () => jsonResponse(500, {}) });

    const notices = await refreshCommand(pi, runtime);

    expect(ids(runtime, "litellm")).toEqual(["proxy-gpt-4o"]);
    expect(notices).toEqual([['LiteLLM ("litellm"): model refresh failed (/model/info returned 500).', "warning"]]);
  });

  it("writes to stderr without a UI", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-refresh-noui-"));
    process.env.LITELLM_OFFLINE = "1";
    const { pi, runtime } = await startup(agentDir);
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);

    const notices = await refreshCommand(pi, runtime, "", false);

    expect(notices).toEqual([]);
    expect(stderr).toHaveBeenCalledExactlyOnceWith("LiteLLM: model refresh skipped (LITELLM_OFFLINE=1).\n");
  });

  it("completes configured provider names", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-refresh-complete-"));
    await writeFile(
      join(agentDir, "settings.json"),
      JSON.stringify({ litellm: { providers: { "litellm-team": { baseUrl: "https://team.example.com" } } } }),
      "utf8",
    );
    const pi = createPi();
    await (await loadExtension(agentDir))(pi);
    const command = pi.commands.get("litellm-refresh") as unknown as {
      getArgumentCompletions: (prefix: string) => Array<{ value: string }> | null;
    };

    expect(command.getArgumentCompletions("litellm-")?.map(({ value }) => value)).toEqual(["litellm-team"]);
    expect(command.getArgumentCompletions("")?.map(({ value }) => value)).toEqual(["litellm", "litellm-team"]);
    expect(command.getArgumentCompletions("x")).toBeNull();
  });
});
