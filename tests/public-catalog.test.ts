import { mkdtemp, readFile, rename, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { loadPublicCatalog } from "../src/public-catalog.js";

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

afterEach(() => vi.restoreAllMocks());

describe("loadPublicCatalog", () => {
  it("keeps the shared module portable across upstream PR worktrees", async () => {
    const source = await readFile(join(process.cwd(), "src/public-catalog.ts"), "utf8");
    const relativeImports = [...source.matchAll(/from\s+["'](\.\.?\/[^"']+)["']/g)].map((match) => match[1]);
    expect(relativeImports).toEqual(["./backend-identity.js"]);
  });

  it("falls back from the Azure adapter to the OpenAI vendor catalog", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => response({})),
    );
    const catalog = await loadWithFreshCache();
    expect(catalog.lookup("azure", "gpt-4")).toMatchObject({
      source: "pi-vendor",
      provider: "openai",
      limits: { context: 8192, output: 8192 },
    });
  });

  it("derives effort levels from non-null Pi thinking levels", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => response({})),
    );
    const catalog = await loadWithFreshCache();
    expect(catalog.lookup("azure", "gpt-5")?.effortLevels).toEqual(expect.arrayContaining(["low", "medium", "high"]));
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
