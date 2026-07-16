# Smoke Tests on Pull Requests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the interactive terminal smoke and run the secret-free smoke workflow on relevant pull requests.

**Architecture:** Preserve the workflow environment at the Terminal Control process boundary instead of copying individual variables. Prime each fresh terminal agent directory before launching Pi and report startup logs on failure. Add the existing path-filtered workflow to pull requests while withholding the optional LiteLLM license on PR events.

**Tech Stack:** GitHub Actions YAML, TypeScript ESM, Vitest, Terminal Control

## Global Constraints

- Keep pull-request runs free of repository secrets.
- Keep enterprise auth coverage on trusted non-PR events.
- Add no dependencies or production runtime changes.
- Keep the existing smoke path filters and bounded readiness probes.

---

### Task 1: Preserve the terminal smoke environment

**Files:**
- Modify: `tests/litellm-smoke-workflow.test.ts`
- Modify: `tests/terminal-smoke.test.ts`

**Interfaces:**
- Consumes: GitHub Actions environment inherited by the Vitest process
- Produces: Terminal Control child environment containing `PATH` and `LITELLM_*` variables

- [ ] **Step 1: Write the failing regression assertion**

Read `tests/terminal-smoke.test.ts` from the workflow test and assert that the launch options contain `inheritEnv: true`.

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `npm test -- tests/litellm-smoke-workflow.test.ts`

Expected: FAIL because the terminal smoke does not enable environment inheritance.

- [ ] **Step 3: Implement the minimal fix**

Add `inheritEnv: true` to the existing `terminal.launch()` options beside the isolated `PI_CODING_AGENT_DIR` environment.

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `npm test -- tests/litellm-smoke-workflow.test.ts`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/litellm-smoke-workflow.test.ts tests/terminal-smoke.test.ts
git commit -S -m "fix: inherit environment in terminal smoke"
```

### Task 2: Run the smoke workflow on pull requests

**Files:**
- Modify: `tests/litellm-smoke-workflow.test.ts`
- Modify: `.github/workflows/litellm-smoke.yml`

**Interfaces:**
- Consumes: GitHub `pull_request`, `push`, `schedule`, and `workflow_dispatch` events
- Produces: path-filtered PR smoke runs using the unlicensed LiteLLM image

- [ ] **Step 1: Write the failing regression assertions**

Assert that the workflow contains a path-filtered `pull_request` trigger and that `LITELLM_LICENSE` resolves to an empty string for pull-request events.

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `npm test -- tests/litellm-smoke-workflow.test.ts`

Expected: FAIL because the workflow has no PR trigger and exposes the license expression unconditionally.

- [ ] **Step 3: Implement the minimal workflow change**

Add `pull_request` with the existing path filters and gate `secrets.LITELLM_LICENSE` behind `github.event_name != 'pull_request'`.

- [ ] **Step 4: Run verification**

Run: `npm test -- tests/litellm-smoke-workflow.test.ts`

Expected: PASS.

Run: `npm run check`

Expected: PASS with no warnings.

Run: `npm run clean && npm run build`

Expected: PASS with fresh runtime output.

- [ ] **Step 5: Commit**

```bash
git add tests/litellm-smoke-workflow.test.ts .github/workflows/litellm-smoke.yml
git commit -S -m "ci: run LiteLLM smoke on pull requests"
```

### Task 3: Prime fresh terminal smoke directories

**Files:**
- Modify: `tests/litellm-smoke-workflow.test.ts`
- Modify: `tests/terminal-smoke.test.ts`

**Interfaces:**
- Consumes: the Pi binary, extension, environment, working directory, and fresh agent directory used by the TUI smoke
- Produces: a populated LiteLLM model cache before each terminal launch and startup-only diagnostics on failure

- [ ] **Step 1: Write failing regression assertions**

Assert that the terminal smoke primes each fresh agent directory before launch and reports terminal logs when the initial model does not render.

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `npm test -- tests/litellm-smoke-workflow.test.ts`

Expected: FAIL because fresh terminal directories are launched without a model cache or startup diagnostics.

- [ ] **Step 3: Implement the minimal fix**

Run `pi --list-models litellm` with the same binary, extension, working directory, environment, and agent directory immediately before the TUI launch. Wrap only the initial model wait to attach terminal logs on failure.

- [ ] **Step 4: Run verification**

Run the focused workflow test, `npm run check`, a clean build, and `actionlint`.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-07-15-smoke-tests-on-prs-design.md docs/superpowers/plans/2026-07-15-smoke-tests-on-prs.md tests/litellm-smoke-workflow.test.ts tests/terminal-smoke.test.ts
git commit -S -m "fix: prime terminal smoke model cache"
```

### Task 4: Separate community and Enterprise auth smoke

**Files:**
- Modify: `tests/litellm-smoke-workflow.test.ts`
- Modify: `.github/workflows/litellm-smoke.yml`

**Interfaces:**
- Consumes: the shared LiteLLM proxy, `LITELLM_LICENSE`, and `scripts/smoke-auth.ts`
- Produces: an always-running community auth step and a separately named licensed Enterprise auth step

- [ ] **Step 1: Write the failing regression assertion**

Replace the existing auth-step assertion in `tests/litellm-smoke-workflow.test.ts` with:

```ts
it("separates community and Enterprise auth smoke", () => {
  const workflow = readWorkflow();

  expect(workflow).toContain(`- name: Run community auth smoke
        env:
          LITELLM_LICENSE: ''
          LITELLM_SMOKE_TIMEOUT_MS: '60000'
        run: npx tsx scripts/smoke-auth.ts`);
  expect(workflow).toContain(`- name: Run Enterprise auth smoke
        if: \${{ env.LITELLM_LICENSE != '' }}
        env:
          LITELLM_SMOKE_TIMEOUT_MS: '60000'
        run: npx tsx scripts/smoke-auth.ts`);
});
```

Update the broad workflow contract to expect both `Run community auth smoke` and `Run Enterprise auth smoke` instead of `Run auth smoke`.

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `mise exec node@24.16.0 -- npm test -- tests/litellm-smoke-workflow.test.ts`

Expected: FAIL because the workflow still has one licensed-only `Run auth smoke` step.

- [ ] **Step 3: Split the workflow steps**

Replace the licensed-only auth step in `.github/workflows/litellm-smoke.yml` with:

```yaml
- name: Run community auth smoke
  env:
    LITELLM_LICENSE: ''
    LITELLM_SMOKE_TIMEOUT_MS: '60000'
  run: npx tsx scripts/smoke-auth.ts

- name: Run Enterprise auth smoke
  if: ${{ env.LITELLM_LICENSE != '' }}
  env:
    LITELLM_SMOKE_TIMEOUT_MS: '60000'
  run: npx tsx scripts/smoke-auth.ts
```

- [ ] **Step 4: Run focused verification**

Run: `mise exec node@24.16.0 -- npm test -- tests/litellm-smoke-workflow.test.ts`

Expected: PASS.

Run: `actionlint .github/workflows/litellm-smoke.yml`

Expected: PASS with no output.

- [ ] **Step 5: Run the repository gate**

Run: `mise exec node@24.16.0 -- npm run check`

Expected: PASS with no warnings or test failures.

- [ ] **Step 6: Commit**

```bash
git add tests/litellm-smoke-workflow.test.ts .github/workflows/litellm-smoke.yml
git commit -S -m "test: separate community and enterprise auth smoke"
```
