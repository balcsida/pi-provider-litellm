# Run LiteLLM smoke workflow on pull requests

**Date:** 2026-07-05
**Status:** Approved

## Problem

The `LiteLLM Smoke` workflow (`.github/workflows/litellm-smoke.yml`) only runs
on pushes to `main`, a weekly cron, and manual dispatch. Regressions in the
smoke-covered integration path (LiteLLM proxy discovery, OpenAI-compatible and
Anthropic routes, Pi CLI loading) are only caught after merge.

## Goal

Run the smoke suite on pull requests as well, so PR updates get the same
integration coverage before merge.

## Design

1. **Trigger** — add a `pull_request:` trigger to `litellm-smoke.yml` with the
   same `paths:` filter as the existing `push:` trigger
   (`.github/workflows/litellm-smoke.yml`, `package-lock.json`, `package.json`,
   `scripts/smoke*.ts`, `src/**`, `tests/**`). Docs-only PRs skip the run.
   Fork PRs work: the workflow uses no repository secrets (VidaiMock upstream,
   dummy master key) and already has `permissions: contents: read`.

2. **Concurrency** — add a workflow-level `concurrency` block:

   ```yaml
   concurrency:
     group: ${{ github.workflow }}-${{ github.ref }}
     cancel-in-progress: ${{ github.event_name == 'pull_request' }}
   ```

   Rapid pushes to a PR cancel the superseded smoke run instead of stacking
   20-minute jobs. Runs for `main`, cron, and dispatch are never cancelled.

3. **Guard test** — extend `tests/litellm-smoke-workflow.test.ts` to assert the
   workflow declares a `pull_request` trigger with a `paths` filter and the
   concurrency block. Written before the workflow change (TDD).

4. **README** — add one sentence to the "Mocked LiteLLM smoke workflow"
   section stating the workflow also runs on pull requests that touch the
   relevant paths.

## Out of scope

- Running the smoke suite against every individual commit of a PR.
- Restructuring the workflow as a reusable `workflow_call` from `ci.yml`.
- Any change to the smoke runner or its assertions.

## Success criteria

- `npm test` passes, including the new guard-test assertions.
- After push, the `LiteLLM Smoke` workflow appears as a check on the PR.
