# Non-blocking Cache Miss Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep Pi activation responsive by moving cache-miss model and MCP discovery to the existing background refresh path.

**Architecture:** Provider activation always registers available cached models without network access, while recording whether the cache needs immediate refresh. The existing `session_start` refresh path handles invalid or missing caches, and explicit discovery commands keep their synchronous behavior.

**Tech Stack:** TypeScript ESM, Vitest, Pi extension API, Node.js `>=22.19.0`

## Global Constraints

- Do not add dependencies, environment variables, or cache-format fields.
- Keep `--list-models`, `/login litellm`, and `/litellm-refresh` synchronous.
- Keep activation-time MCP discovery non-blocking.
- Do not edit generated `dist/` output by hand.
- Commit locally only; never push.

---

### Task 1: Defer cache-miss discovery

**Files:**
- Modify: `src/index.ts:75-83,880-972,1085-1086,1088-1115,1225-1233`
- Test: `tests/index.test.ts:221-330`

**Interfaces:**
- Consumes: existing `runRefresh(state: ProviderState): Promise<ProviderRefreshResult>` and `session_start` handler
- Produces: `ProviderState.refreshOnStart: boolean`, cleared after a successful refresh

- [ ] **Step 1: Write the failing startup test**

Add this case under `describe("extension startup")`:

```ts
it("uses mismatched cached models until session-start refresh", async () => {
  const agentDir = await makeAgentDir();
  process.env.LITELLM_BASE_URL = "https://litellm.example.com";
  process.env.LITELLM_API_KEY = "new-key";
  await writeFile(
    join(agentDir, "litellm-models.json"),
    JSON.stringify({
      baseUrl: "https://litellm.example.com",
      apiKeyFingerprint: fingerprint("old-key"),
      fetchedAt: Date.now(),
      source: "model_info",
      models: cachedModels,
    }),
    "utf8",
  );
  const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input);
    if (url.endsWith("/model/info")) {
      return jsonResponse(200, {
        data: [{ model_name: "fresh-model", model_info: { mode: "chat" } }],
      });
    }
    if (url.endsWith("/mcp-rest/tools/list")) return jsonResponse(200, { tools: [] });
    throw new Error(`unexpected URL: ${url}`);
  });
  const extension = await loadExtension(agentDir);
  const pi = createPi();

  await extension(pi);

  expect(fetchMock).not.toHaveBeenCalled();
  expect(pi.providers[0]?.config.models).toEqual(cachedModels);

  for (const handler of pi.handlers.get("session_start") ?? []) {
    await handler({ reason: "start" }, { sessionManager: { getSessionFile: () => undefined } });
  }
  await vi.waitFor(() => {
    expect((pi.providers.at(-1)?.config.models as Array<{ id: string }>)[0]?.id).toBe("fresh-model");
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
npm test -- tests/index.test.ts -t "uses mismatched cached models"
```

Expected: FAIL because activation calls `fetch` and registers freshly discovered models.

- [ ] **Step 3: Implement stale-while-revalidate startup**

Add the refresh flag to `ProviderState`:

```ts
type ProviderState = {
  definition: ProviderDefinition;
  creds: ResolvedCredentials;
  headers?: Record<string, string>;
  models: ProviderModelConfig[];
  cacheFetchedAt: number;
  refreshOnStart: boolean;
  liveDiscoveryApiKey?: string;
  refreshInProgress: Promise<ProviderRefreshResult> | null;
};
```

In `loadProviderState`, retain any cached models and only perform synchronous discovery for `--list-models`:

```ts
let models: ProviderModelConfig[] = cache?.models ?? [];
const canDiscover =
  creds.baseUrl !== undefined && fp !== undefined && !isOffline() && getDiscoveryTimeoutMs() > 0;
const shouldFetch = canDiscover && isListModelsMode();
const refreshOnStart = canDiscover && !cacheValid && !shouldFetch;
```

Return `refreshOnStart` with the provider state:

```ts
return {
  definition,
  creds,
  headers,
  models,
  cacheFetchedAt,
  refreshOnStart,
  liveDiscoveryApiKey,
  refreshInProgress: null,
};
```

Remove activation-time MCP discovery; background refresh, login, and explicit
refresh already register MCP tools after successful model discovery:

```ts
registerSkillTools(defaultState);
```

Clear the immediate-refresh flag after successful discovery:

```ts
state.cacheFetchedAt = now;
state.refreshOnStart = false;
registerProvider(state, overridden, fresh.apiKeyConfig);
```

Let `session_start` refresh invalid caches as well as stale valid caches:

```ts
for (const state of providerStates) {
  const cacheIsStale = state.cacheFetchedAt > 0 && Date.now() - state.cacheFetchedAt > CACHE_STALE_MS;
  if (!state.refreshOnStart && !cacheIsStale) continue;
  void runRefresh(state).catch(() => undefined);
}
```

- [ ] **Step 4: Run the focused test**

Run:

```bash
npm test -- tests/index.test.ts -t "uses mismatched cached models"
```

Expected: PASS with one test passing.

- [ ] **Step 5: Run the complete startup test file**

Run:

```bash
npm test -- tests/index.test.ts
```

Expected: PASS. Update assertions in existing startup-discovery cases only where they must trigger `session_start` before expecting background discovery.

- [ ] **Step 6: Commit the behavior**

```bash
git status --short
git add src/index.ts tests/index.test.ts
git commit -S -m "perf: defer cache-miss discovery"
```

### Task 2: Document background cache-miss refresh

**Files:**
- Modify: `README.md:132-139,153-160,210-212`

**Interfaces:**
- Consumes: Task 1 startup behavior
- Produces: user-facing cache and MCP discovery documentation

- [ ] **Step 1: Update the cache behavior documentation**

Replace the cache refresh paragraph with:

```md
When the cache is missing, invalid, or older than 24 hours, the extension starts without waiting for discovery and refreshes models in the background on session start. Existing cached models remain available while an invalid or stale cache refreshes. Failures are silent; run `/litellm-refresh` to force an immediate refresh and see its result.
```

Replace the two environment-variable rows and the following note with:

```md
| `LITELLM_OFFLINE` | unset | If `1`, disable automatic model discovery and use cached models only |
| `LITELLM_DISCOVERY_TIMEOUT_MS` | `5000` | Background and explicit discovery fetch timeout in ms; `0` disables automatic discovery |

`LITELLM_DISCOVERY_TIMEOUT_MS=0` disables automatic and explicit refresh model discovery. It does not replace the base URL or API key settings required to send requests when you are not using `/login litellm`.
```

Replace the final sentence of the MCP discovery paragraph with:

```md
MCP discovery runs after background model discovery, `/login litellm`, or `/litellm-refresh`; extension activation never waits for it. MCP tools run in Pi's parallel tool mode and retry transient failures once.
```

- [ ] **Step 2: Verify documentation and full checks**

Run:

```bash
git diff --check
npm run check
npm run clean && npm run build
```

Expected: all commands exit zero; Biome, typecheck, and all Vitest tests pass.

- [ ] **Step 3: Benchmark the built extension**

Run:

```bash
for i in {1..5}; do
  /usr/bin/time -p /Users/hu901131/.bun/bin/pi --help --no-extensions >/dev/null
done
for i in {1..5}; do
  /usr/bin/time -p env \
    LITELLM_BASE_URL=https://10.255.255.1 \
    LITELLM_API_KEY=benchmark-key \
    /Users/hu901131/.bun/bin/pi --help --no-extensions -e "$PWD/dist/index.js" >/dev/null
done
```

Expected: each isolated extension run remains near the fresh-cache timing and does not wait for the five-second discovery timeout.

- [ ] **Step 4: Commit the documentation**

```bash
git status --short
git add README.md
git commit -S -m "docs: explain background model discovery"
```
