# Null Model Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep LiteLLM models whose `model_info.mode` is `null` available to Pi.

**Architecture:** Extend the existing nullable response boundary and reuse the shared chat-style mode predicate. No new discovery path or configuration is needed.

**Tech Stack:** TypeScript ESM, Vitest, Biome

## Global Constraints

- Treat `null` and `undefined` as an unset mode.
- Keep explicit non-chat modes filtered.
- Add no dependencies or new abstractions.

---

### Task 1: Accept nullable model modes

**Files:**
- Modify: `src/types.ts:28`
- Modify: `src/discover.ts:48-53`
- Test: `tests/discover.test.ts`

**Interfaces:**
- Consumes: LiteLLM `ModelInfoEntry.model_info.mode`
- Produces: `mode?: string | null` and unchanged `discoverModels()` results containing unset-mode models

- [ ] **Step 1: Write the failing test**

Add a `/model/info` discovery test that returns `{ model_name: "local/model", model_info: { mode: null } }` and expects `result.models.map((model) => model.id)` to equal `["local/model"]`.

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `npm test -- tests/discover.test.ts -t "keeps models with a null mode"`

Expected: FAIL because the returned model list is empty.

- [ ] **Step 3: Implement the minimal fix**

Change the response type to:

```ts
mode?: string | null;
```

Accept nullish values in the existing predicates:

```ts
function isResponsesMode(mode: string | null | undefined): boolean {
  return mode != null && RESPONSES_MODE_PATTERN.test(mode);
}

function isChatStyleMode(mode: string | null | undefined): boolean {
  return mode == null || mode === "chat" || isResponsesMode(mode);
}
```

- [ ] **Step 4: Run verification**

Run: `npm test -- tests/discover.test.ts`

Expected: PASS.

Run: `npm run check`

Expected: PASS with no warnings.

Run: `npm run clean && npm run build`

Expected: PASS and a fresh `dist/` build.

- [ ] **Step 5: Commit**

```bash
git add src/types.ts src/discover.ts tests/discover.test.ts
git commit -S -m "fix: accept null LiteLLM model modes"
```
