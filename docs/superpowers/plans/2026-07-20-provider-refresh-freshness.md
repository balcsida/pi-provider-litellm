# Provider Refresh Freshness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent Pi startup from performing duplicate LiteLLM model and MCP discovery.

**Architecture:** Preserve the background `session_start` refresh and make the registered provider callback honor its existing cache timestamp plus Pi's `force` flag. The callback returns current models for an ordinary refresh while the cache is fresh and retains existing behavior for missing, stale, invalid, or forced refreshes.

**Tech Stack:** TypeScript ESM, Pi provider API, Vitest

## Global Constraints

- Keep startup cache-miss discovery non-blocking.
- Do not add settings, environment variables, dependencies, or logging abstractions.
- Preserve explicit `/litellm-refresh` behavior.

---

### Task 1: Honor Provider Refresh Freshness

**Files:**
- Modify: `src/index.ts`
- Test: `tests/index.test.ts`

**Interfaces:**
- Consumes: `ProviderState.cacheFetchedAt`, `ProviderState.refreshOnStart`, `CACHE_STALE_MS`, and Pi's `RefreshModelsContext.force`.
- Produces: a `refreshModels` callback that returns `state.models` for an ordinary fresh-cache refresh and calls `runRefresh` otherwise.

- [ ] **Step 1: Write the failing regression test**

Add a startup test that completes the existing `session_start` refresh, clears
the fetch mock, invokes `config.refreshModels` without `force`, and expects the
current models with no network request. Then invoke it with `force: true` and
expect discovery to run.

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `npm test -- tests/index.test.ts -t "does not repeat a fresh session refresh unless forced"`

Expected: FAIL because the ordinary provider callback performs another
`/model/info` request.

- [ ] **Step 3: Implement the minimal freshness guard**

In `registerProvider`, accept `force` from the refresh context. Return
`state.models` when networking is disabled, discovery is disabled, or the
cache is fresh and no refresh is pending or forced. Otherwise call the
existing `runRefresh` function with Pi's credential and signal.

- [ ] **Step 4: Verify focused and repository checks**

Run: `npm test -- tests/index.test.ts -t "does not repeat a fresh session refresh unless forced"`

Expected: PASS.

Run: `npm run check`

Expected: Biome, typecheck, and all tests pass.

Run: `npm run clean && npm run build`

Expected: the package builds successfully from clean generated output.

- [ ] **Step 5: Commit**

Stage `src/index.ts` and `tests/index.test.ts`, then create the signed commit:
`fix: avoid duplicate startup discovery`.

### Task 2: Synchronize Terminal Smoke Commands

**Files:**
- Modify: `tests/terminal-smoke.test.ts`

**Interfaces:**
- Consumes: `Session.keyboard.type`, `Session.screen.waitForText`, and `Session.keyboard.press`.
- Produces: `submit(session, text)` with editor-echo synchronization for slash commands only.

- [ ] **Step 1: Write the failing regression test**

Add one enabled test with a structural fake `Session`. Record calls while
submitting `/login litellm` and a masked value. Expect the command order to be
`type`, `waitForText`, `Enter`, and expect the masked value to skip the screen
check.

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `npm test -- tests/terminal-smoke.test.ts -t "waits for command echo without inspecting form values"`

Expected: FAIL because `submit` presses Enter without waiting for the command
to appear.

- [ ] **Step 3: Implement the minimal synchronization**

After typing, call `session.screen.waitForText(text, { timeoutMs: waitTimeoutMs })`
only when `text.startsWith("/")`, then press Enter as before.

- [ ] **Step 4: Verify focused and repository checks**

Run the focused test again, then `npm run check` and
`npm run clean && npm run build`.

Expected: the focused test, repository checks, and clean build all pass.

- [ ] **Step 5: Publish and verify CI**

Create the signed commit `test: synchronize terminal smoke commands`, push the
branch, and wait for every PR check, including `LiteLLM Smoke`, to finish.
