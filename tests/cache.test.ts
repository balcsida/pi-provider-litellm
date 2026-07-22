import { createHash } from "node:crypto";
import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { fingerprint, isCacheValid, readCache } from "../src/cache.js";
import type { CacheFile } from "../src/types.js";

describe("fingerprint", () => {
  it("produces a stable keyed hex digest", () => {
    expect(fingerprint("secret")).toBe(fingerprint("secret"));
    expect(fingerprint("secret")).toHaveLength(64);
    expect(fingerprint("secret")).toMatch(/^[a-f0-9]+$/);
  });

  it("differs across inputs", () => {
    expect(fingerprint("a")).not.toBe(fingerprint("b"));
  });

  it("does not store the raw sha256 digest of the API key", () => {
    const rawDigest = createHash("sha256").update("secret").digest("hex");

    expect(fingerprint("secret")).not.toBe(rawDigest);
  });
});

describe("isCacheValid", () => {
  const cache: CacheFile = {
    baseUrl: "https://litellm.example.com",
    apiKeyFingerprint: fingerprint("k1"),
    fetchedAt: Date.now(),
    source: "model_info",
    models: [],
  };

  it("returns true when baseUrl and fingerprint match", () => {
    expect(isCacheValid(cache, "https://litellm.example.com", "k1", undefined)).toBe(true);
  });

  it("returns false when baseUrl differs", () => {
    expect(isCacheValid(cache, "https://other.example.com", "k1", undefined)).toBe(false);
  });

  it("returns false when api key differs", () => {
    expect(isCacheValid(cache, "https://litellm.example.com", "k2", undefined)).toBe(false);
  });

  it("returns false when headers differ", () => {
    expect(isCacheValid({ ...cache, headersFingerprint: "old" }, "https://litellm.example.com", "k1", "new")).toBe(
      false,
    );
  });

  it("returns false for null cache", () => {
    expect(isCacheValid(null, "https://litellm.example.com", "k1", undefined)).toBe(false);
  });
});

describe("readCache", () => {
  it("returns null for missing file", async () => {
    const dir = await mkdtemp(join(tmpdir(), "pi-litellm-"));
    expect(await readCache(join(dir, "missing.json"))).toBeNull();
  });

  it("returns null for malformed JSON", async () => {
    const dir = await mkdtemp(join(tmpdir(), "pi-litellm-"));
    const path = join(dir, "bad.json");
    await writeFile(path, "{not json", "utf8");
    expect(await readCache(path)).toBeNull();
  });

  it("returns null when shape doesn't match CacheFile", async () => {
    const dir = await mkdtemp(join(tmpdir(), "pi-litellm-"));
    const path = join(dir, "wrong.json");
    await writeFile(path, JSON.stringify({ foo: "bar" }), "utf8");
    expect(await readCache(path)).toBeNull();
  });

  it("returns parsed CacheFile when shape is correct", async () => {
    const dir = await mkdtemp(join(tmpdir(), "pi-litellm-"));
    const path = join(dir, "good.json");
    const cache = {
      baseUrl: "https://x.example.com",
      apiKeyFingerprint: fingerprint("k"),
      fetchedAt: 12345,
      source: "model_info" as const,
      models: [],
    };
    await writeFile(path, JSON.stringify(cache), "utf8");
    expect(await readCache(path)).toEqual(cache);
  });

  it("accepts cache files populated from the health discovery fallback", async () => {
    const dir = await mkdtemp(join(tmpdir(), "pi-litellm-"));
    const path = join(dir, "health.json");
    const cache = {
      baseUrl: "https://x.example.com",
      apiKeyFingerprint: fingerprint("k"),
      fetchedAt: 12345,
      source: "health" as const,
      models: [],
    };
    await writeFile(path, JSON.stringify(cache), "utf8");
    expect(await readCache(path)).toEqual(cache);
  });
});
