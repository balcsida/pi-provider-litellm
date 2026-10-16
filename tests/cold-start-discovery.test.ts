import { mkdir, mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import { afterEach, describe, expect, it, vi } from "vitest";
import { LITELLM_DISCOVERY_VERSION } from "../src/backend-identity.js";
import { createPi, loadExtension, useHermeticEnv } from "./test-helpers.js";

/** A catalog in the shape Pi persists: `{ <provider>: { models, checkedAt } }`, with per-protocol baseUrls. */
function storedCatalog(overrides: { baseUrl?: string; version?: number; checkedAt?: number } = {}) {
  return {
    litellm: {
      checkedAt: overrides.checkedAt ?? Date.now(),
      models: [
        {
          id: "stored-model",
          name: "stored-model",
          api: "openai-completions",
          provider: "litellm",
          baseUrl: overrides.baseUrl ?? "https://proxy.example.com/v1",
          reasoning: false,
          input: ["text"],
          contextWindow: 128000,
          maxTokens: 16384,
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
          litellmDiscoveryVersion: overrides.version ?? LITELLM_DISCOVERY_VERSION,
        },
      ],
    },
  };
}

vi.unmock("@earendil-works/pi-coding-agent");

useHermeticEnv();

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

/** Replays createAgentSessionServices(): every startup refresh runs with allowNetwork:false. */
async function startup(agentDir: string, modelsStorePath?: string): Promise<ModelRuntime> {
  const runtime = await ModelRuntime.create({
    authPath: join(agentDir, "auth.json"),
    modelsPath: join(agentDir, "models.json"),
    ...(modelsStorePath ? { modelsStorePath } : {}),
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
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "gpt-4o", model_info: { mode: "chat" } }] });
      }
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
      throw new Error(`unexpected URL: ${url}`);
    });

    const runtime = await startup(agentDir);

    expect(runtime.getModels("litellm").map((model) => model.id)).toEqual(["sonnet"]);
    expect(seen).toContain("https://stored.example.com/model/info");
  });

  it("skips discovery when Pi already has a stored catalog for the provider", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-stored-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    // Pi persists its own catalog. Once it has one, the session gets its models from Pi, so the seed's round trip is
    // not needed; this is the 1.8 s the change removes.
    await writeFile(join(agentDir, "models-store.json"), JSON.stringify(storedCatalog()), "utf8");
    const requested: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      requested.push(String(input));
      throw new Error("no discovery should be needed when Pi has a stored catalog");
    });

    const runtime = await startup(agentDir);

    expect(requested.filter((url) => url.includes("/model/info") || url.includes("/v1/models"))).toEqual([]);
    expect(runtime.getModels("litellm").map((model) => model.id)).toEqual(["stored-model"]);
  });

  it("still supplies the models when Pi's active store is somewhere else", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-other-store-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    // ModelRuntime takes a modelsStorePath. Every caller inside Pi passes an agent-directory path, but an SDK embedder
    // need not. The catalog in the default file must then not make the seed hand back nothing, or the provider would be
    // empty -- the fault the seed exists to prevent.
    await writeFile(join(agentDir, "models-store.json"), JSON.stringify(storedCatalog()), "utf8");
    const elsewhere = join(agentDir, "elsewhere", "models-store.json");
    await mkdir(dirname(elsewhere), { recursive: true });
    const requested: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      requested.push(String(input));
      throw new Error("no discovery should be needed when a stored catalog is readable");
    });

    const runtime = await startup(agentDir, elsewhere);

    expect(runtime.getModels("litellm").map((model) => model.id)).toEqual(["stored-model"]);
    expect(requested.filter((url) => url.includes("/model/info") || url.includes("/v1/models"))).toEqual([]);
  });

  it("hands stored image models over too, without discovering", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-stored-images-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    // Pi persists every model type, so a proxy with image routes stores them beside its chat models. The store is read
    // from elsewhere so that only the seed, not Pi's own restore, can supply them.
    const catalog = storedCatalog();
    (catalog.litellm.models as Record<string, unknown>[]).push({
      type: "image",
      id: "stored-image",
      name: "stored-image",
      api: "litellm-images",
      provider: "litellm",
      baseUrl: "https://proxy.example.com/v1",
      input: ["text"],
      output: ["image"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      litellmDiscoveryVersion: LITELLM_DISCOVERY_VERSION,
    });
    await writeFile(join(agentDir, "models-store.json"), JSON.stringify(catalog), "utf8");
    const elsewhere = join(agentDir, "elsewhere", "models-store.json");
    await mkdir(dirname(elsewhere), { recursive: true });
    const requested: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      requested.push(String(input));
      throw new Error("no discovery should be needed when a stored catalog is readable");
    });

    const runtime = await startup(agentDir, elsewhere);

    expect(runtime.getAllModels("litellm").map((model) => model.id)).toEqual(["stored-model", "stored-image"]);
    expect(requested.filter((url) => url.includes("/model/info") || url.includes("/v1/models"))).toEqual([]);
  });

  it("discovers when Pi cannot restore a stored model", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-short-model-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    const catalog = storedCatalog();
    // enrichCachedModel throws on it, so Pi's restore publishes nothing and the seed is the whole catalog.
    delete (catalog.litellm.models[0] as Record<string, unknown>).name;
    await writeFile(join(agentDir, "models-store.json"), JSON.stringify(catalog), "utf8");
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      if (String(input).endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
      }
      throw new Error(`unexpected URL: ${String(input)}`);
    });

    const runtime = await startup(agentDir);

    expect(runtime.getModels("litellm").map((model) => model.id)).toEqual(["fresh-model"]);
  });

  it("discovers when the stored catalog is older than four hours", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-stale-store-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    const catalog = storedCatalog({ checkedAt: Date.now() - 4 * 60 * 60 * 1000 });
    await writeFile(join(agentDir, "models-store.json"), JSON.stringify(catalog), "utf8");
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      if (String(input).endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
      }
      throw new Error(`unexpected URL: ${String(input)}`);
    });

    const runtime = await startup(agentDir);

    expect(runtime.getModels("litellm").map((model) => model.id)).toContain("fresh-model");
  });

  it("discovers when the stored catalog is for another proxy", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-stored-other-proxy-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    await writeFile(
      join(agentDir, "models-store.json"),
      JSON.stringify(storedCatalog({ baseUrl: "https://elsewhere.example.com/v1" })),
      "utf8",
    );
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      if (String(input).endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
      }
      throw new Error(`unexpected URL: ${String(input)}`);
    });

    const runtime = await startup(agentDir);

    expect(runtime.getModels("litellm").map((model) => model.id)).toContain("fresh-model");
  });

  it("discovers when the stored catalog is from an older discovery version", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-stored-old-version-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    // Pi keeps a same-version stored entry over what startup discovery proves, so an older one must not be trusted:
    // skipping here would hide the metadata the new version adds.
    await writeFile(
      join(agentDir, "models-store.json"),
      JSON.stringify(storedCatalog({ version: LITELLM_DISCOVERY_VERSION - 1 })),
      "utf8",
    );
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      if (String(input).endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }] });
      }
      throw new Error(`unexpected URL: ${String(input)}`);
    });

    const runtime = await startup(agentDir);

    expect(runtime.getModels("litellm").map((model) => model.id)).toContain("fresh-model");
  });

  it("discovers when the store is empty, unreadable, or for another provider", async () => {
    for (const body of ["{}", "not json", JSON.stringify({ litellm: { models: [] } }), JSON.stringify({ other: {} })]) {
      const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-stored-unusable-"));
      process.env.LITELLM_BASE_URL = "https://proxy.example.com";
      process.env.LITELLM_API_KEY = "env-key";
      await writeFile(join(agentDir, "models-store.json"), body, "utf8");
      const spy = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
        if (String(input).endsWith("/model/info")) {
          return jsonResponse(200, { data: [{ model_name: "gpt-4o", model_info: { mode: "chat" } }] });
        }
        throw new Error(`unexpected URL: ${String(input)}`);
      });

      const runtime = await startup(agentDir);

      expect(
        runtime.getModels("litellm").map((model) => model.id),
        body,
      ).toEqual(["gpt-4o"]);
      spy.mockRestore();
      vi.resetModules();
    }
  });

  it("stays offline when PI_OFFLINE is set", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-cold-offline-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    process.env.PI_OFFLINE = "1";
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      throw new Error("network must not be used while offline");
    });

    const runtime = await startup(agentDir);

    expect(fetchSpy).not.toHaveBeenCalled();
    expect(runtime.getModels("litellm")).toEqual([]);
    expect(stderr).not.toHaveBeenCalled();
  });

  it.each([
    ["PI_OFFLINE", "1", "PI_OFFLINE"],
    ["PI_OFFLINE", "0", "PI_OFFLINE"],
    ["PI_OFFLINE", "", "PI_OFFLINE"],
    ["LITELLM_OFFLINE", "1", "LITELLM_OFFLINE=1"],
    ["LITELLM_DISCOVERY_TIMEOUT_MS", "0", "LITELLM_DISCOVERY_TIMEOUT_MS=0"],
  ])("reports the %s=%j startup skip when verbose", async (variable, value, reason) => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-cold-verbose-offline-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    process.env.LITELLM_VERBOSE_DISCOVERY = "1";
    process.env[variable] = value;
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("network must not be used"));

    const runtime = await startup(agentDir);

    expect(fetchSpy).not.toHaveBeenCalled();
    expect(runtime.getModels("litellm")).toEqual([]);
    expect(stderr).toHaveBeenCalledExactlyOnceWith(`LiteLLM (litellm): startup discovery skipped (${reason}).\n`);
  });

  it("stays offline when LITELLM_OFFLINE is set", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-cold-lloffline-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
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
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("connect ECONNREFUSED"));

    const runtime = await startup(agentDir);

    expect(runtime.getProvider("litellm")).toBeDefined();
    expect(runtime.getModels("litellm")).toEqual([]);
  });

  it("gives up quickly when the proxy hangs, whatever the discovery timeout is", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-cold-hang-"));
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
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
    delete process.env.LITELLM_BASE_URL;
    delete process.env.LITELLM_API_KEY;
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      throw new Error("must not reach the network without credentials");
    });

    await startup(agentDir);

    expect(fetchSpy).not.toHaveBeenCalled();
  });
});
