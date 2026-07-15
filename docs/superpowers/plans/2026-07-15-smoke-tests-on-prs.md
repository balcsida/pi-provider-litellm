# Smoke Tests on Pull Requests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the interactive terminal smoke and run the secret-free smoke workflow on relevant pull requests.

**Architecture:** Preserve the workflow environment at the Terminal Control process boundary instead of copying individual variables. Add the existing path-filtered workflow to pull requests while withholding the optional LiteLLM license on PR events.

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
