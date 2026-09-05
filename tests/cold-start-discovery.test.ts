import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createPi, loadExtension, useHermeticEnv } from "./test-helpers.js";

vi.unmock("@earendil-works/pi-coding-agent");

useHermeticEnv();

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

/** Replays createAgentSessionServices(): every startup refresh runs with allowNetwork:false. */
async function startup(agentDir: string): Promise<ModelRuntime> {
  const runtime = await ModelRuntime.create({
    authPath: join(agentDir, "auth.json"),
    modelsPath: join(agentDir, "models.json"),
  });
  const pi = createPi();
  pi.registerProvider = (provider) => runtime.registerNativeProvider(provider);
  await (await loadExtension(agentDir))(pi);
  await runtime.refresh({ allowNetwork: false });
  return runtime;
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.resetModules();
});

describe("cold start discovery (issue #137)", () => {
  it("populates the catalog from env credentials during Pi's no-network startup", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-cold-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "gpt-4o", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/mcp-rest/tools/list")) return jsonResponse(200, { tools: [] });
      throw new Error(`unexpected URL: ${url}`);
    });

    const runtime = await startup(agentDir);

    expect(runtime.getModels("litellm").map((model) => model.id)).toEqual(["gpt-4o"]);
  });

  it("populates the catalog from a stored /login credential", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-cold-stored-"));
    await writeFile(
      join(agentDir, "auth.json"),
      JSON.stringify({
        litellm: { type: "api_key", key: "stored-key", env: { LITELLM_BASE_URL: "https://stored.example.com" } },
      }),
      "utf8",
    );
    const seen: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      seen.push(url);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "sonnet", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/mcp-rest/tools/list")) return jsonResponse(200, { tools: [] });
      throw new Error(`unexpected URL: ${url}`);
    });

    const runtime = await startup(agentDir);

    expect(runtime.getModels("litellm").map((model) => model.id)).toEqual(["sonnet"]);
    expect(seen).toContain("https://stored.example.com/model/info");
  });

  it("stays offline when PI_OFFLINE is set", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-cold-offline-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    process.env.PI_OFFLINE = "1";
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      throw new Error("network must not be used while offline");
    });

    const runtime = await startup(agentDir);

    expect(fetchSpy).not.toHaveBeenCalled();
    expect(runtime.getModels("litellm")).toEqual([]);
  });

  it("stays offline when LITELLM_OFFLINE is set", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-cold-lloffline-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    process.env.LITELLM_OFFLINE = "1";
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      throw new Error("network must not be used while offline");
    });

    await startup(agentDir);

    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("registers the provider anyway when the proxy is unreachable", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-cold-fail-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("connect ECONNREFUSED"));

    const runtime = await startup(agentDir);

    expect(runtime.getProvider("litellm")).toBeDefined();
    expect(runtime.getModels("litellm")).toEqual([]);
  });

  it("gives up quickly when the proxy hangs, whatever the discovery timeout is", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-cold-hang-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    // Deployments raise this per-request timeout; activation must not inherit it, because Pi
    // cannot paint its UI until every extension has activated.
    process.env.LITELLM_DISCOVERY_TIMEOUT_MS = "60000";
    vi.spyOn(globalThis, "fetch").mockImplementation(
      (_input, init) =>
        new Promise((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(new Error("aborted")), { once: true });
        }),
    );

    const startedAt = Date.now();
    await startup(agentDir);

    expect(Date.now() - startedAt).toBeLessThan(15_000);
  }, 70_000);

  it("does not discover when no credentials are configured", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-cold-nocreds-"));
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      throw new Error("must not reach the network without credentials");
    });

    await startup(agentDir);

    expect(fetchSpy).not.toHaveBeenCalled();
  });
});
