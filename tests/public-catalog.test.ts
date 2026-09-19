import { mkdtemp, rename, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { getModels } from "@earendil-works/pi-ai/compat";
import { afterEach, describe, expect, it, vi } from "vitest";
import { loadPublicCatalog } from "../src/public-catalog.js";

vi.mock("@earendil-works/pi-ai/compat", { spy: true });
vi.mock("node:fs/promises", async (importOriginal) => {
  const actual = await importOriginal<typeof import("node:fs/promises")>();
  return {
    ...actual,
    rename: vi.fn(actual.rename),
    writeFile: vi.fn(actual.writeFile),
  };
});

const modelsDev = {
  "amazon-bedrock": {
    models: {
      "anthropic.claude-sonnet-4-6-v1:0": {
        modalities: { input: ["text", "image"] },
        limit: { context: 200_000, output: 64_000 },
        cost: { input: 3, output: 15 },
        reasoning_options: { type: "effort", values: ["low", "medium", "high"] },
      },
    },
  },
  "fireworks-ai": {
    models: {
      "accounts/fireworks/models/kimi-k3": { limit: { context: 262_144, output: 32_768 } },
    },
  },
};

function response(body: unknown): Response {
  return new Response(JSON.stringify(body), { status: 200, headers: { "content-type": "application/json" } });
}

async function loadWithFreshCache() {
  const dir = await mkdtemp(join(tmpdir(), "public-catalog-test-"));
  return loadPublicCatalog({ cachePath: join(dir, "models-dev.json") });
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.mocked(getModels).mockReset();
});

describe("loadPublicCatalog", () => {
  it.each(["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"])(
    "enriches the ChatGPT subscription model %s from Pi's Codex catalog",
    async (modelId) => {
      vi.stubGlobal(
        "fetch",
        vi.fn(async () => response({})),
      );
      const catalog = await loadWithFreshCache();
      expect(catalog.lookup("chatgpt", modelId)).toMatchObject({
        provider: "openai-codex",
        modelId,
        limits: { context: 272_000, output: 128_000 },
      });
    },
  );

  it.each(["azure", "azure_ai"])(
    "prefers the %s-specific Pi catalog over the generic OpenAI catalog",
    async (provider) => {
      vi.stubGlobal(
        "fetch",
        vi.fn(async () => response({})),
      );
      const catalog = await loadWithFreshCache();

      expect(catalog.lookup(provider, "gpt-5.6-sol")).toMatchObject({
        source: "pi-adapter",
        provider: "azure-openai-responses",
        limits: { context: 1_050_000, output: 128_000 },
      });
    },
  );

  it.each(["azure", "azure_ai"])(
    "keeps %s as the Pi fallback for a sparse OpenAI models.dev record",
    async (provider) => {
      vi.stubGlobal(
        "fetch",
        vi.fn(async () =>
          response({
            openai: { models: { "gpt-5.6-sol": { reasoning_options: [{ type: "effort", values: ["low", "high"] }] } } },
          }),
        ),
      );
      const catalog = await loadWithFreshCache();

      expect(catalog.lookup(provider, "gpt-5.6-sol")).toMatchObject({
        source: "models.dev",
        provider: "openai",
        piProvider: "azure-openai-responses",
        effortLevels: ["low", "high"],
      });
    },
  );

  it("falls back to OpenAI when the Azure catalog does not know the model", async () => {
    vi.mocked(getModels).mockReturnValueOnce([]);
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => response({})),
    );
    const catalog = await loadWithFreshCache();

    expect(catalog.lookup("azure", "gpt-4")).toMatchObject({
      source: "pi-vendor",
      provider: "openai",
      limits: { context: 8192 },
    });
  });

  it("keeps the vendor fallback for sparse enrichment when the adapter has no catalog entry", async () => {
    vi.mocked(getModels).mockReturnValueOnce([]);
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        response({
          openai: { models: { "gpt-4": { cost: { input: 30 } } } },
        }),
      ),
    );
    const catalog = await loadWithFreshCache();

    expect(catalog.lookup("azure", "gpt-4")).toMatchObject({
      source: "models.dev",
      piProvider: "openai",
    });
  });

  // models.dev has no `chatgpt` provider; ChatGPT subscription routes serve OpenAI models.
  it("looks up ChatGPT routes under OpenAI on models.dev", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        response({
          openai: {
            models: {
              "gpt-5.6-sol": {
                reasoning_options: [{ type: "effort", values: ["none", "low", "medium", "high", "xhigh", "max"] }],
              },
            },
          },
        }),
      ),
    );
    const catalog = await loadWithFreshCache();
    expect(catalog.lookup("chatgpt", "gpt-5.6-sol")).toMatchObject({
      source: "models.dev",
      provider: "openai",
      piProvider: "openai-codex",
      effortLevels: ["none", "low", "medium", "high", "xhigh", "max"],
    });
  });

  // A Pi map omits standard levels it leaves at Pi's defaults, so it is not a complete list.
  it("carries a Pi thinking level map without flattening it into effort levels", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => response({})),
    );
    const catalog = await loadWithFreshCache();
    const record = catalog.lookup("chatgpt", "gpt-5.6-sol");
    expect(record?.thinkingLevelMap).toEqual({ xhigh: "xhigh", max: "max", minimal: "low" });
    expect(record).not.toHaveProperty("effortLevels");
  });

  it("reads Bedrock Claude evidence from models.dev", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => response(modelsDev)),
    );
    const catalog = await loadWithFreshCache();
    expect(catalog.lookup("bedrock", "anthropic.claude-sonnet-4-6-v1:0")).toEqual({
      source: "models.dev",
      provider: "amazon-bedrock",
      modelId: "anthropic.claude-sonnet-4-6-v1:0",
      limits: { context: 200_000, output: 64_000 },
      cost: { input: 3, output: 15 },
      modalities: ["text", "image"],
      effortLevels: ["low", "medium", "high"],
    });
  });

  it("uses a Fireworks base_model hint regardless of the transport adapter", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => response(modelsDev)),
    );
    const catalog = await loadWithFreshCache();
    expect(catalog.lookup("fireworks", "accounts/fireworks/models/kimi-k3")).toMatchObject({
      source: "models.dev",
      provider: "fireworks-ai",
      limits: { context: 262_144, output: 32_768 },
    });
  });

  it("withholds modalities when a models.dev record has a non-array input list", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        response({
          moonshotai: {
            models: { "kimi-k3-preview": { modalities: { input: 7 }, limit: { context: 262_144, output: 32_768 } } },
          },
        }),
      ),
    );
    const catalog = await loadWithFreshCache();
    const record = catalog.lookup("moonshot", "kimi-k3-preview");
    expect(record).toMatchObject({ source: "models.dev", limits: { context: 262_144, output: 32_768 } });
    expect(record).not.toHaveProperty("modalities");
  });

  it("omits malformed negative models.dev prices while preserving zero", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        response({
          private: {
            models: {
              priced: { cost: { input: -1, output: 2, cache_read: 0, cache_write: 0 } },
            },
          },
        }),
      ),
    );
    const catalog = await loadWithFreshCache();
    expect(catalog.lookup("private", "priced")?.cost).toEqual({
      output: 2,
      cacheRead: 0,
      cacheWrite: 0,
    });
  });

  it("returns no opinion for an unknown model", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => response({})),
    );
    expect((await loadWithFreshCache()).lookup("unknown", "not-real")).toBeUndefined();
  });

  it("writes through a process-unique temporary cache path", async () => {
    const dir = await mkdtemp(join(tmpdir(), "public-catalog-"));
    const cachePath = join(dir, "models-dev.json");
    vi.mocked(writeFile).mockClear();
    vi.mocked(rename).mockClear();
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => response(modelsDev)),
    );

    await loadPublicCatalog({ cachePath });

    const temporaryPath = vi.mocked(writeFile).mock.calls[0]?.[0];
    expect(temporaryPath).toEqual(expect.stringMatching(/\.\d+\.\d+\.tmp$/));
    expect(temporaryPath).not.toBe(`${cachePath}.tmp`);
    expect(rename).toHaveBeenCalledWith(temporaryPath, cachePath);
  });

  it("uses a stale disk cache without fetching while offline", async () => {
    const dir = await mkdtemp(join(tmpdir(), "public-catalog-"));
    const cachePath = join(dir, "models-dev.json");
    await writeFile(cachePath, JSON.stringify({ fetchedAt: 1, catalog: modelsDev }));
    const fetchSpy = vi.fn();
    vi.stubGlobal("fetch", fetchSpy);

    const catalog = await loadPublicCatalog({ cachePath, offline: true });

    expect(catalog.lookup("bedrock", "anthropic.claude-sonnet-4-6-v1:0")).toBeDefined();
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("refreshes instead of accepting a future-dated disk cache", async () => {
    const dir = await mkdtemp(join(tmpdir(), "public-catalog-"));
    const cachePath = join(dir, "models-dev.json");
    await writeFile(cachePath, JSON.stringify({ fetchedAt: Date.now() + 60_000, catalog: {} }));
    const fetchSpy = vi.fn(async () => response(modelsDev));
    vi.stubGlobal("fetch", fetchSpy);

    const catalog = await loadPublicCatalog({ cachePath });

    expect(fetchSpy).toHaveBeenCalledOnce();
    expect(catalog.lookup("bedrock", "anthropic.claude-sonnet-4-6-v1:0")).toBeDefined();
  });
});
