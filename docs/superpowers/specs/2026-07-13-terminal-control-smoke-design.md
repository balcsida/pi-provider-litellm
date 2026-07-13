# Test interactive Pi flows with terminal-control

**Date:** 2026-07-13
**Status:** Approved

## Problem

The smoke workflow launches Pi only with non-interactive flags. Unit tests cover
the extension callbacks, but no automated test crosses the terminal boundary for
interactive login, model refresh, or model selection.

## Goal

Use `@kitlangton/terminal-control` to exercise `/login litellm`,
`/litellm-refresh`, and `/model` through the real Pi TUI against the existing
LiteLLM and VidaiMock smoke services.

## Design

1. Add `@kitlangton/terminal-control` as a pinned development dependency. It is
   compatible with the package's ESM, Node, and Vitest versions and supplies
   native binaries for the workflow's Linux runner and local macOS development.
2. Add one focused Vitest smoke file. It launches the built Pi extension in an
   isolated agent directory, uses bounded screen waits, and drives all three
   commands through a pseudo-terminal.
3. Verify the login prompts and success notification, the model refresh result,
   and both mocked model names in the model picker. Each scenario gets a fresh
   Pi process so failures remain independent.
4. Skip the live-backend smoke during the default unit suite. Run it explicitly
   in the existing smoke job after the extension is built and the LiteLLM proxy
   is ready.
5. Extend the workflow guard test and README description to keep the new
   coverage visible and prevent accidental removal.

## Failure handling

- Every terminal wait has a fixed timeout so CI cannot hang indefinitely.
- The test uses a temporary Pi agent directory and the workflow's fake API key.
- Recordings and automatic artifacts stay disabled because terminal captures
  could include credentials.

## Out of scope

- Real provider API calls.
- SSO login automation.
- Snapshotting full terminal frames.
- A reusable terminal-test abstraction before a second test file needs it.

## Success criteria

- The unchanged unit suite still passes under the repository Node runtime.
- The smoke job proves interactive login, refresh, and model-picker behavior.
- `npm run check`, build, and package verification pass.
