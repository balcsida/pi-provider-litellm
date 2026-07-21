# Models.dev Cache Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make models.dev enrichment optional and persist its catalog for 28 days with stale-while-revalidate refreshes.

**Architecture:** Extend discovery options with an enable flag and one injected agent-directory cache path. Keep the catalog mapping in `src/discover.ts`, add a small validated `{ fetchedAt, catalog }` cache around the existing fetch, and reuse the atomic JSON writer in `src/cache.ts`. Extension call sites share one public catalog cache across provider aliases; direct discovery callers without a path retain process-local caching.

**Tech Stack:** TypeScript ESM, Node.js `fs/promises`, Vitest, Biome, tsgo

## Global Constraints

- Node.js support remains `>=22.19.0`; add no dependency.
- Models.dev enrichment remains enabled unless `LITELLM_MODELS_DEV=0`.
- Disabling models.dev skips its memory cache, disk cache, and network request, but does not disable `/v1/models`.
- The cache lifetime is exactly `28 * 24 * 60 * 60 * 1000` milliseconds and is not configurable.
- Fresh cache data returns without network access; stale data returns immediately while one background refresh runs per cache path.
- Initial cache misses may await one deduplicated models.dev fetch.
- Failed reads, writes, initial fetches, and background refreshes remain non-fatal; stale data is never deleted on failure.
- Models.dev remains limited to the successful `/v1/models` fallback path.
- Use test-first development and signed, conventional, single-line commits no longer than 72 characters.
- Keep the branch local unless the user explicitly authorizes a push.

---

### Task 1: Make Models.dev Enrichment Optional

**Files:**
- Modify: `src/types.ts:19-23`
- Modify: `src/discover.ts:236-243,427-430`
- Test: `tests/discover.test.ts:342-440`

**Interfaces:**
- Consumes: existing `discoverModels(baseUrl, apiKey, options)`.
- Produces: `DiscoveryOptions.modelsDev?: boolean`; `false` skips models.dev and unset preserves enabled behavior.

- [ ] **Step 1: Write the failing opt-out test**

Add this case inside `describe("discoverModels fallback to /v1/models", ...)`:

```ts
it("skips models.dev enrichment when disabled", async () => {
  const urls: string[] = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = input instanceof URL ? input.toString() : String(input);
    urls.push(url);
    if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
    if (url.endsWith("/v1/models")) {
      return jsonResponse(200, {
        data: [{ id: "gpt-5.5", object: "model", owned_by: "openai" }],
      });
    }
    throw new Error(`unexpected URL: ${url}`);
  });

  const result = await discoverModels("https://litellm.example.com", "sk-test", {
    modelsDev: false,
  });

  expect(urls).not.toContain("https://models.dev/api.json");
  expect(result.models[0]).toMatchObject({
    id: "gpt-5.5",
    name: "GPT-5.5",
    contextWindow: 272000,
  });
});
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
rtk npm test -- tests/discover.test.ts
```

Expected: FAIL because `DiscoveryOptions` has no `modelsDev` property or because discovery requests the unexpected models.dev URL.

- [ ] **Step 3: Add the minimal option and guard**

Extend `DiscoveryOptions` in `src/types.ts`:

```ts
export interface DiscoveryOptions {
  timeoutMs?: number;
  signal?: AbortSignal;
  headers?: Record<string, string>;
  modelsDev?: boolean;
}
```

Guard the catalog lookup in `src/discover.ts`, make the progress text accurate for
cache hits, and leave mapping priority unchanged:

```ts
let modelsDev: ModelsDevResponse | undefined;
if (options.modelsDev !== false) {
  options.onProgress?.("Loading models.dev catalog for metadata enrichment...");
  modelsDev = await getModelsDevCatalog(options);
}
```

- [ ] **Step 4: Run the focused test to verify GREEN**

Run:

```bash
rtk npm test -- tests/discover.test.ts
```

Expected: PASS, including the existing unavailable and successful models.dev enrichment cases.

- [ ] **Step 5: Commit the optional enrichment behavior**

```bash
rtk git status --short
rtk git add src/types.ts src/discover.ts tests/discover.test.ts
rtk git commit -m "feat: make models.dev enrichment optional"
```

Expected: one signed commit containing only the option, guard, and regression test.

---

### Task 2: Persist Fresh Models.dev Metadata

**Files:**
- Modify: `src/cache.ts:1-51`
- Modify: `src/types.ts:19-24`
- Modify: `src/discover.ts:1-43,221-265`
- Test: `tests/discover.test.ts:1-13,342-440`

**Interfaces:**
- Consumes: `DiscoveryOptions.modelsDev?: boolean` from Task 1 and the existing models.dev response mapping.
- Produces: `DiscoveryOptions.modelsDevCachePath?: string`; `writeJsonAtomic(path: string, value: unknown): Promise<void>`; a private cache file shaped as `{ fetchedAt: number; catalog: ModelsDevResponse }`.

- [ ] **Step 1: Write failing cache-miss, fresh-cache, and malformed-cache tests**

Add imports to `tests/discover.test.ts`:

```ts
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
```

Add a local fixture:

```ts
const MODELS_DEV_CATALOG = {
  openai: {
    models: {
      "gpt-5.5": {
        name: "Models.dev GPT-5.5",
        reasoning: true,
        limit: { context: 1_050_000, output: 128_000 },
        cost: { input: 5, output: 30, cache_read: 0.5 },
      },
    },
  },
};
```

Add three cases to the `/v1/models` describe block:

```ts
it("persists models.dev metadata after an initial cache miss", async () => {
  const dir = await mkdtemp(join(tmpdir(), "pi-litellm-models-dev-"));
  const cachePath = join(dir, "litellm-models-dev.json");
  vi.spyOn(Date, "now").mockReturnValue(1_000);
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input);
    if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
    if (url.endsWith("/v1/models")) {
      return jsonResponse(200, { data: [{ id: "gpt-5.5", owned_by: "openai" }] });
    }
    if (url === "https://models.dev/api.json") return jsonResponse(200, MODELS_DEV_CATALOG);
    throw new Error(`unexpected URL: ${url}`);
  });

  const result = await discoverModels("https://litellm.example.com", "sk-test", {
    modelsDevCachePath: cachePath,
  });

  expect(result.models[0]?.contextWindow).toBe(1_050_000);
  expect(JSON.parse(await readFile(cachePath, "utf8"))).toEqual({
    fetchedAt: 1_000,
    catalog: MODELS_DEV_CATALOG,
  });
});

it("uses a fresh persistent models.dev cache without fetching", async () => {
  const dir = await mkdtemp(join(tmpdir(), "pi-litellm-models-dev-"));
  const cachePath = join(dir, "litellm-models-dev.json");
  await writeFile(cachePath, JSON.stringify({ fetchedAt: 1_000, catalog: MODELS_DEV_CATALOG }), "utf8");
  vi.spyOn(Date, "now").mockReturnValue(2_000);
  const urls: string[] = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input);
    urls.push(url);
    if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
    if (url.endsWith("/v1/models")) {
      return jsonResponse(200, { data: [{ id: "gpt-5.5", owned_by: "openai" }] });
    }
    throw new Error(`unexpected URL: ${url}`);
  });

  const result = await discoverModels("https://litellm.example.com", "sk-test", {
    modelsDevCachePath: cachePath,
  });

  expect(urls).not.toContain("https://models.dev/api.json");
  expect(result.models[0]?.contextWindow).toBe(1_050_000);
});

it("replaces a malformed models.dev cache from the network", async () => {
  const dir = await mkdtemp(join(tmpdir(), "pi-litellm-models-dev-"));
  const cachePath = join(dir, "litellm-models-dev.json");
  await writeFile(cachePath, JSON.stringify({ fetchedAt: "invalid", catalog: [] }), "utf8");
  const urls: string[] = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input);
    urls.push(url);
    if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
    if (url.endsWith("/v1/models")) {
      return jsonResponse(200, { data: [{ id: "gpt-5.5", owned_by: "openai" }] });
    }
    if (url === "https://models.dev/api.json") return jsonResponse(200, MODELS_DEV_CATALOG);
    throw new Error(`unexpected URL: ${url}`);
  });

  await discoverModels("https://litellm.example.com", "sk-test", { modelsDevCachePath: cachePath });

  expect(urls).toContain("https://models.dev/api.json");
  expect(JSON.parse(await readFile(cachePath, "utf8")).catalog).toEqual(MODELS_DEV_CATALOG);
});
```

- [ ] **Step 2: Run the tests to verify RED**

Run:

```bash
rtk npm test -- tests/discover.test.ts
```

Expected: FAIL because `modelsDevCachePath` and persistent cache behavior do not exist.

- [ ] **Step 3: Reuse the atomic JSON writer**

In `src/cache.ts`, extract the existing write body without changing `writeCache` callers:

```ts
export async function writeJsonAtomic(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const tmp = `${path}.${process.pid}.${Date.now()}.tmp`;
  await writeFile(tmp, JSON.stringify(value, null, 2), "utf8");
  await rename(tmp, path);
}

export async function writeCache(path: string, cache: CacheFile): Promise<void> {
  await writeJsonAtomic(path, cache);
}
```

- [ ] **Step 4: Add the cache path and validated persistent cache**

Extend `DiscoveryOptions`:

```ts
modelsDevCachePath?: string;
```

In `src/discover.ts`, import `readFile` and the shared writer, then replace the singleton catalog with path-keyed state:

```ts
import { readFile } from "node:fs/promises";
import { writeJsonAtomic } from "./cache.js";

interface ModelsDevCacheFile {
  fetchedAt: number;
  catalog: ModelsDevResponse;
}

const modelsDevCaches = new Map<string, ModelsDevCacheFile>();
```

Add narrow cache validation and IO:

```ts
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

async function readModelsDevCache(path: string): Promise<ModelsDevCacheFile | undefined> {
  try {
    const parsed: unknown = JSON.parse(await readFile(path, "utf8"));
    if (
      !isRecord(parsed) ||
      typeof parsed.fetchedAt !== "number" ||
      !Number.isFinite(parsed.fetchedAt) ||
      parsed.fetchedAt < 0 ||
      !isRecord(parsed.catalog)
    ) {
      return undefined;
    }
    return { fetchedAt: parsed.fetchedAt, catalog: parsed.catalog as ModelsDevResponse };
  } catch {
    return undefined;
  }
}

async function fetchAndStoreModelsDevCatalog(
  key: string,
  options: DiscoveryOptions,
): Promise<ModelsDevResponse | undefined> {
  try {
    const catalog = await fetchPublicJson<ModelsDevResponse>(MODELS_DEV_URL, options);
    const cache = { fetchedAt: Date.now(), catalog };
    modelsDevCaches.set(key, cache);
    if (options.modelsDevCachePath) {
      await writeJsonAtomic(options.modelsDevCachePath, cache).catch(() => undefined);
    }
    return catalog;
  } catch {
    return undefined;
  }
}

async function getModelsDevCatalog(options: DiscoveryOptions): Promise<ModelsDevResponse | undefined> {
  const key = options.modelsDevCachePath ?? MODELS_DEV_URL;
  let cache = modelsDevCaches.get(key);
  if (!cache && options.modelsDevCachePath) {
    cache = await readModelsDevCache(options.modelsDevCachePath);
    if (cache) modelsDevCaches.set(key, cache);
  }
  if (cache) return cache.catalog;
  return fetchAndStoreModelsDevCatalog(key, options);
}
```

The `modelsDev === false` guard from Task 1 must run before this function, so disabled discovery reads no cache.

- [ ] **Step 5: Run focused discovery and existing cache tests**

Run:

```bash
rtk npm test -- tests/discover.test.ts tests/cache.test.ts
```

Expected: PASS; existing LiteLLM model-cache round trips still use atomic writes.

- [ ] **Step 6: Commit persistent caching**

```bash
rtk git status --short
rtk git add src/cache.ts src/types.ts src/discover.ts tests/discover.test.ts
rtk git commit -m "feat: persist models.dev metadata cache"
```

Expected: one signed commit with fresh-cache persistence and validation.

---

### Task 3: Refresh Stale Metadata in the Background

**Files:**
- Modify: `src/discover.ts:15-43,236-265`
- Test: `tests/discover.test.ts:342-440`

**Interfaces:**
- Consumes: path-keyed `ModelsDevCacheFile` state and `fetchAndStoreModelsDevCatalog` from Task 2.
- Produces: fixed `MODELS_DEV_CACHE_TTL_MS`; one `Promise` per cache key for initial-fetch and stale-refresh deduplication.

- [ ] **Step 1: Write failing stale-while-revalidate tests**

Add a small deferred helper to `tests/discover.test.ts`:

```ts
function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}
```

Add these cases to the `/v1/models` describe block. Use a stale fixture whose `fetchedAt` is more than 28 days behind mocked `Date.now()`:

```ts
it("returns stale metadata while refreshing it in the background", async () => {
  const dir = await mkdtemp(join(tmpdir(), "pi-litellm-models-dev-"));
  const cachePath = join(dir, "litellm-models-dev.json");
  const staleCatalog = {
    openai: { models: { "gpt-5.5": { name: "Stale GPT", limit: { context: 200_000 } } } },
  };
  await writeFile(cachePath, JSON.stringify({ fetchedAt: 1, catalog: staleCatalog }), "utf8");
  vi.spyOn(Date, "now").mockReturnValue(28 * 24 * 60 * 60 * 1000 + 2);
  const refresh = deferred<Response>();
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input);
    if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
    if (url.endsWith("/v1/models")) {
      return jsonResponse(200, { data: [{ id: "gpt-5.5", owned_by: "openai" }] });
    }
    if (url === "https://models.dev/api.json") return refresh.promise;
    throw new Error(`unexpected URL: ${url}`);
  });

  const result = await discoverModels("https://litellm.example.com", "sk-test", {
    modelsDevCachePath: cachePath,
  });

  expect(result.models[0]).toMatchObject({ name: "Stale GPT", contextWindow: 200_000 });
  refresh.resolve(jsonResponse(200, MODELS_DEV_CATALOG));
  await vi.waitFor(async () => {
    const cache = JSON.parse(await readFile(cachePath, "utf8"));
    expect(cache.catalog).toEqual(MODELS_DEV_CATALOG);
  });
});

it("keeps stale metadata when background refresh fails", async () => {
  const dir = await mkdtemp(join(tmpdir(), "pi-litellm-models-dev-"));
  const cachePath = join(dir, "litellm-models-dev.json");
  const stale = { fetchedAt: 1, catalog: MODELS_DEV_CATALOG };
  await writeFile(cachePath, JSON.stringify(stale), "utf8");
  vi.spyOn(Date, "now").mockReturnValue(28 * 24 * 60 * 60 * 1000 + 2);
  const modelsDevRequests: string[] = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input);
    if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
    if (url.endsWith("/v1/models")) {
      return jsonResponse(200, { data: [{ id: "gpt-5.5", owned_by: "openai" }] });
    }
    if (url === "https://models.dev/api.json") {
      modelsDevRequests.push(url);
      return new Response(null, { status: 503 });
    }
    throw new Error(`unexpected URL: ${url}`);
  });

  const result = await discoverModels("https://litellm.example.com", "sk-test", {
    modelsDevCachePath: cachePath,
  });

  expect(result.models[0]?.contextWindow).toBe(1_050_000);
  await vi.waitFor(() => expect(modelsDevRequests).toHaveLength(1));
  expect(JSON.parse(await readFile(cachePath, "utf8"))).toEqual(stale);
});

it("deduplicates concurrent stale cache refreshes", async () => {
  const dir = await mkdtemp(join(tmpdir(), "pi-litellm-models-dev-"));
  const cachePath = join(dir, "litellm-models-dev.json");
  await writeFile(cachePath, JSON.stringify({ fetchedAt: 1, catalog: MODELS_DEV_CATALOG }), "utf8");
  vi.spyOn(Date, "now").mockReturnValue(28 * 24 * 60 * 60 * 1000 + 2);
  const refresh = deferred<Response>();
  let refreshes = 0;
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input);
    if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
    if (url.endsWith("/v1/models")) {
      return jsonResponse(200, { data: [{ id: "gpt-5.5", owned_by: "openai" }] });
    }
    if (url === "https://models.dev/api.json") {
      refreshes++;
      return refresh.promise;
    }
    throw new Error(`unexpected URL: ${url}`);
  });

  await Promise.all([
    discoverModels("https://one.example.com", "sk-test", { modelsDevCachePath: cachePath }),
    discoverModels("https://two.example.com", "sk-test", { modelsDevCachePath: cachePath }),
  ]);

  expect(refreshes).toBe(1);
  refresh.resolve(jsonResponse(200, MODELS_DEV_CATALOG));
  await vi.waitFor(async () => {
    expect(JSON.parse(await readFile(cachePath, "utf8")).fetchedAt).toBe(Date.now());
  });
});
```

- [ ] **Step 2: Run the tests to verify RED**

Run:

```bash
rtk npm test -- tests/discover.test.ts
```

Expected: FAIL because stale cache currently returns without starting a refresh and concurrent refresh state is not tracked.

- [ ] **Step 3: Add the fixed TTL and deduplicated refresh promise**

In `src/discover.ts`, add:

```ts
const MODELS_DEV_CACHE_TTL_MS = 28 * 24 * 60 * 60 * 1000;
const modelsDevRefreshes = new Map<string, Promise<ModelsDevResponse | undefined>>();
```

Replace `fetchAndStoreModelsDevCatalog` with a deduplicating version:

```ts
function refreshModelsDevCatalog(
  key: string,
  options: DiscoveryOptions,
): Promise<ModelsDevResponse | undefined> {
  const active = modelsDevRefreshes.get(key);
  if (active) return active;
  const refresh = (async () => {
    try {
      const catalog = await fetchPublicJson<ModelsDevResponse>(MODELS_DEV_URL, options);
      const cache = { fetchedAt: Date.now(), catalog };
      modelsDevCaches.set(key, cache);
      if (options.modelsDevCachePath) {
        await writeJsonAtomic(options.modelsDevCachePath, cache).catch(() => undefined);
      }
      return catalog;
    } catch {
      return undefined;
    } finally {
      modelsDevRefreshes.delete(key);
    }
  })();
  modelsDevRefreshes.set(key, refresh);
  return refresh;
}
```

Update `getModelsDevCatalog`:

```ts
async function getModelsDevCatalog(options: DiscoveryOptions): Promise<ModelsDevResponse | undefined> {
  const key = options.modelsDevCachePath ?? MODELS_DEV_URL;
  let cache = modelsDevCaches.get(key);
  if (!cache && options.modelsDevCachePath) {
    cache = await readModelsDevCache(options.modelsDevCachePath);
    if (cache) modelsDevCaches.set(key, cache);
  }
  if (!cache) return refreshModelsDevCatalog(key, options);
  if (Date.now() - cache.fetchedAt < MODELS_DEV_CACHE_TTL_MS) return cache.catalog;
  void refreshModelsDevCatalog(key, { ...options, signal: undefined });
  return cache.catalog;
}
```

Detaching only the stale background refresh from the caller signal lets revalidation finish after discovery returns; initial misses still honor cancellation and the existing timeout.

- [ ] **Step 4: Run the focused tests to verify GREEN**

Run:

```bash
rtk npm test -- tests/discover.test.ts tests/cache.test.ts
```

Expected: PASS with one models.dev request for concurrent stale callers.

- [ ] **Step 5: Commit stale-while-revalidate**

```bash
rtk git status --short
rtk git add src/discover.ts tests/discover.test.ts
rtk git commit -m "perf: refresh stale models.dev cache in background"
```

Expected: one signed commit containing only TTL, background refresh, deduplication, and tests.

---

### Task 4: Wire the Shared Cache and Environment Control

**Files:**
- Modify: `src/index.ts:34-45,111-127,716-721,953-959,1044-1060,1141-1146`
- Modify: `tests/index.test.ts:9-23,217-267`
- Modify: `README.md:147-158`

**Interfaces:**
- Consumes: `DiscoveryOptions.modelsDev` and `DiscoveryOptions.modelsDevCachePath` from Tasks 1 and 2.
- Produces: `getModelsDevDiscoveryOptions(): Pick<DiscoveryOptions, "modelsDev" | "modelsDevCachePath">`; shared `litellm-models-dev.json` path; documented `LITELLM_MODELS_DEV=0` behavior.

- [ ] **Step 1: Write the failing extension-level environment test**

Add `"LITELLM_MODELS_DEV"` to `ENV_KEYS`, then add this startup test:

```ts
it("disables models.dev enrichment with LITELLM_MODELS_DEV=0", async () => {
  const agentDir = await makeAgentDir();
  process.env.LITELLM_BASE_URL = "https://litellm.example.com";
  process.env.LITELLM_API_KEY = "sk-test";
  process.env.LITELLM_MODELS_DEV = "0";
  const urls: string[] = [];
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input);
    urls.push(url);
    if (url.endsWith("/model/info")) return new Response(null, { status: 403 });
    if (url.endsWith("/v1/models")) {
      return jsonResponse(200, { data: [{ id: "gpt-5.5", owned_by: "openai" }] });
    }
    if (url.endsWith("/mcp-rest/tools/list")) return jsonResponse(200, { tools: [] });
    throw new Error(`unexpected URL: ${url}`);
  });
  const extension = await loadExtension(agentDir);
  const pi = createPi();

  await extension(pi);
  await startSession(pi);

  expect(urls).not.toContain("https://models.dev/api.json");
  expect((pi.providers.at(-1)?.config.models as Array<{ id: string }>)[0]?.id).toBe("gpt-5.5");
});
```

- [ ] **Step 2: Run the extension test to verify RED**

Run:

```bash
rtk npm test -- tests/index.test.ts
```

Expected: FAIL with an unexpected models.dev URL because the environment flag is not wired into extension discovery.

- [ ] **Step 3: Add one shared extension option helper**

In `src/index.ts`, add constants beside the existing discovery constants:

```ts
const ENV_MODELS_DEV = "LITELLM_MODELS_DEV";
const MODELS_DEV_CACHE_FILENAME = "litellm-models-dev.json";
```

Add the shared options helper beside `getCachePath`:

```ts
function getModelsDevDiscoveryOptions(): Pick<DiscoveryOptions, "modelsDev" | "modelsDevCachePath"> {
  return {
    modelsDev: process.env[ENV_MODELS_DEV] !== "0",
    modelsDevCachePath: join(getAgentDir(), MODELS_DEV_CACHE_FILENAME),
  };
}
```

Spread `...getModelsDevDiscoveryOptions()` into all three direct extension discovery flows:

```ts
await discoverModels(baseUrl, apiKey, {
  ...getModelsDevDiscoveryOptions(),
  timeoutMs: LOGIN_TIMEOUT_MS,
  signal: callbacks.signal,
  headers: options.headers,
  onProgress: (message) => callbacks.onProgress?.(`LiteLLM: ${message}`),
});
```

```ts
await discoverWithFallback(creds.baseUrl, creds.apiKey, {
  ...getModelsDevDiscoveryOptions(),
  timeoutMs,
  headers,
  onProgress: (message) => process.stderr.write(`LiteLLM: ${message}\n`),
});
```

```ts
await discoverModels(fresh.baseUrl, fresh.apiKey, {
  ...getModelsDevDiscoveryOptions(),
  timeoutMs: getDiscoveryTimeoutMs(),
  signal,
  headers: state.headers,
  onProgress: progress,
});
```

- [ ] **Step 4: Document the cache contract**

Add this row to the optional environment-variable table in `README.md`:

```md
| `LITELLM_MODELS_DEV` | enabled | Set to `0` to disable models.dev metadata enrichment, including its cache and network request; `/v1/models` still uses Pi catalog metadata and defaults |
```

Below the table, document the fixed cache behavior:

```md
Models.dev metadata is cached in `litellm-models-dev.json` under the Pi agent directory for 28 days. Fresh data avoids the public request; stale data is used immediately while one background refresh updates the cache. Set `LITELLM_MODELS_DEV=0` when your LiteLLM metadata is authoritative and no external enrichment is wanted.
```

- [ ] **Step 5: Run focused tests and typecheck**

Run:

```bash
rtk npm test -- tests/index.test.ts tests/discover.test.ts tests/cache.test.ts
rtk npm run typecheck
```

Expected: PASS; the env flag reaches session refresh and all three call sites typecheck with the shared options.

- [ ] **Step 6: Commit extension wiring and documentation**

```bash
rtk git status --short
rtk git add src/index.ts tests/index.test.ts README.md
rtk git commit -m "feat: expose models.dev cache controls"
```

Expected: one signed commit containing only extension configuration, shared path wiring, README text, and its integration test.

---

### Task 5: Verify the Complete Branch

**Files:**
- Verify only; no planned modifications.

**Interfaces:**
- Consumes: all behavior from Tasks 1-4.
- Produces: evidence that lint, types, tests, runtime build, diff hygiene, and signed history pass.

- [ ] **Step 1: Run the repository gate**

```bash
rtk npm run check
```

Expected: Biome, typecheck, and the full Vitest suite PASS. If nested `.worktrees/**` contaminate root discovery, verify from a clean archive instead of changing unrelated files.

- [ ] **Step 2: Rebuild runtime output**

```bash
rtk npm run clean
rtk npm run build
```

Expected: both commands exit 0; generated `dist/` output is not edited by hand.

- [ ] **Step 3: Check the final diff and signatures**

```bash
rtk git diff --check main...HEAD
rtk git status --short --branch
rtk git log --show-signature --format=fuller main..HEAD
```

Expected: no whitespace errors, no uncommitted source changes, and every feature-branch commit has a good signature.
