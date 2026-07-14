# Terminal-Control Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exercise LiteLLM login, model refresh, and model selection through the real Pi terminal UI in the existing mocked smoke workflow.

**Architecture:** Add one opt-in Vitest smoke file backed by `@kitlangton/terminal-control`. The existing GitHub Actions smoke job supplies the built extension, real LiteLLM proxy, and VidaiMock models; the default unit suite skips the live-backend file.

**Tech Stack:** TypeScript ESM, Vitest 4, `@kitlangton/terminal-control` 0.3.1, Pi CLI, GitHub Actions

## Global Constraints

- Keep Node support at `>=22.19.0` and CI on Node `24.16.0`.
- Pin `@kitlangton/terminal-control` to exactly `0.3.1` as a development dependency.
- Use the existing LiteLLM and VidaiMock smoke services; do not call real providers.
- Use bounded terminal waits and an isolated Pi agent directory for every scenario.
- Do not enable recordings, transcripts, or automatic terminal artifacts.
- Keep SSO automation, full-frame snapshots, and reusable cross-file terminal abstractions out of scope.
- Do not change production files under `src/`.

---

### Task 1: Add interactive terminal smoke coverage

**Files:**
- Create: `tests/terminal-smoke.test.ts`
- Modify: `tests/litellm-smoke-workflow.test.ts`
- Modify: `.github/workflows/litellm-smoke.yml`
- Modify: `README.md`
- Modify: `package.json`
- Modify: `package-lock.json`

**Interfaces:**
- Consumes: the built `./dist/index.js`, `./node_modules/.bin/pi`, `LITELLM_BASE_URL`, `LITELLM_API_KEY`, `LITELLM_CLI_SMOKE_MODEL`, and the existing two-model LiteLLM smoke proxy.
- Produces: an opt-in Vitest suite enabled by `LITELLM_TERMINAL_SMOKE=1`; no production API.

- [ ] **Step 1: Add failing workflow and documentation assertions**

Add these assertions to the first workflow test in `tests/litellm-smoke-workflow.test.ts`:

```ts
expect(workflow).toContain("Run interactive Pi terminal smoke");
expect(workflow).toContain("LITELLM_TERMINAL_SMOKE: '1'");
expect(workflow).toContain("npm test -- tests/terminal-smoke.test.ts");
```

Add this assertion to the README test in the same file:

```ts
expect(readme).toContain("interactive Pi TUI smoke");
```

- [ ] **Step 2: Run the focused test to verify RED**

Run:

```bash
mise exec node@24.10.0 -- npm test -- tests/litellm-smoke-workflow.test.ts
```

Expected: FAIL because the workflow does not contain `Run interactive Pi terminal smoke`.

- [ ] **Step 3: Install the pinned development dependency**

Run:

```bash
mise exec node@24.10.0 -- npm install --save-dev --save-exact @kitlangton/terminal-control@0.3.1
```

Expected: `package.json` and `package-lock.json` contain version `0.3.1`; no runtime dependency is added.

- [ ] **Step 4: Add the opt-in terminal smoke file**

Create `tests/terminal-smoke.test.ts` with:

```ts
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { TerminalControl, type Session } from "@kitlangton/terminal-control";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const enabled = process.env.LITELLM_TERMINAL_SMOKE === "1";
const timeoutMs = 30_000;
let terminal: TerminalControl;

async function launchPi(): Promise<Session> {
  const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-terminal-"));
  return terminal.launch({
    command: [
      resolve(repoRoot, "node_modules/.bin/pi"),
      "-e",
      resolve(repoRoot, "dist/index.js"),
      "--provider",
      "litellm",
      "--model",
      process.env.LITELLM_CLI_SMOKE_MODEL ?? "vidaimock-openai",
      "--no-tools",
      "--no-session",
    ],
    cwd: repoRoot,
    env: { PI_CODING_AGENT_DIR: agentDir },
    viewport: { cols: 100, rows: 30 },
  });
}

async function submit(session: Session, text: string): Promise<void> {
  await session.keyboard.type(text);
  await session.keyboard.press("Enter");
}

describe.skipIf(!enabled)("interactive Pi terminal smoke", () => {
  beforeAll(async () => {
    terminal = await TerminalControl.make();
  });

  afterAll(async () => {
    await terminal.close();
  });

  it(
    "logs in to LiteLLM",
    async () => {
      await using session = await launchPi();
      await session.screen.waitForText("vidaimock-openai", { timeoutMs });

      await submit(session, "/login litellm");
      await session.screen.waitForText("Enter LiteLLM proxy URL", { timeoutMs });
      await submit(session, process.env.LITELLM_BASE_URL ?? "http://127.0.0.1:4000");
      await session.screen.waitForText("Select login method", { timeoutMs });
      await submit(session, "1");
      await session.screen.waitForText("Enter API key", { timeoutMs });
      await submit(session, process.env.LITELLM_API_KEY ?? "sk-ci-litellm-smoke");

      await session.screen.waitForText("Logged in to LiteLLM", { timeoutMs });
    },
    timeoutMs,
  );

  it(
    "refreshes LiteLLM models",
    async () => {
      await using session = await launchPi();
      await session.screen.waitForText("vidaimock-openai", { timeoutMs });

      await submit(session, "/litellm-refresh");

      await session.screen.waitForText("LiteLLM: 2 models refreshed (source: model_info)", { timeoutMs });
    },
    timeoutMs,
  );

  it(
    "shows LiteLLM models in the model picker",
    async () => {
      await using session = await launchPi();
      await session.screen.waitForText("vidaimock-openai", { timeoutMs });

      await submit(session, "/model");
      await session.screen.waitForText("anthropic/vidaimock-claude", { timeoutMs });
      const screen = await session.screen.text({ settleMs: 50, deadlineMs: 2_000 });

      expect(screen).toContain("vidaimock-openai");
      expect(screen).toContain("anthropic/vidaimock-claude");
    },
    timeoutMs,
  );
});
```

- [ ] **Step 5: Add the smoke workflow step**

After `Run Pi CLI smoke` in `.github/workflows/litellm-smoke.yml`, add:

```yaml
      - name: Run interactive Pi terminal smoke
        env:
          LITELLM_DISCOVERY_TIMEOUT_MS: '60000'
          LITELLM_TERMINAL_SMOKE: '1'
        run: npm test -- tests/terminal-smoke.test.ts
```

- [ ] **Step 6: Document the interactive coverage**

Extend the final sentence of the mocked smoke workflow section in `README.md` so it states that the workflow also runs an `interactive Pi TUI smoke` covering `/login litellm`, `/litellm-refresh`, and `/model`.

- [ ] **Step 7: Run focused tests to verify GREEN**

Run:

```bash
mise exec node@24.10.0 -- npm test -- tests/litellm-smoke-workflow.test.ts tests/terminal-smoke.test.ts
```

Expected: the workflow tests pass and the terminal smoke suite is skipped without `LITELLM_TERMINAL_SMOKE=1`.

- [ ] **Step 8: Run static checks**

Run:

```bash
mise exec node@24.10.0 -- npm run lint
mise exec node@24.10.0 -- npm run typecheck
```

Expected: both commands pass without warnings.

- [ ] **Step 9: Commit the test implementation**

Run `git status --short`, then commit the six implementation files with:

```bash
git add package.json package-lock.json tests/terminal-smoke.test.ts tests/litellm-smoke-workflow.test.ts .github/workflows/litellm-smoke.yml README.md
git commit -S -m "test: cover interactive Pi terminal flows"
```

Expected: one signed, independently reversible commit.

### Task 2: Verify the complete branch

**Files:**
- Verify only; no planned file changes.

**Interfaces:**
- Consumes: the terminal smoke coverage from Task 1.
- Produces: verification evidence and the pushed `tests/terminal-control` branch.

- [ ] **Step 1: Run the repository gate**

Run:

```bash
mise exec node@24.10.0 -- npm run check
```

Expected: Biome, type checking, and all unit tests pass; the opt-in terminal suite is skipped.

- [ ] **Step 2: Build and verify package policy**

Run:

```bash
mise exec node@24.10.0 -- npm run clean
mise exec node@24.10.0 -- npm run build
mise exec node@24.10.0 -- npm run supply-chain:guard
mise exec node@24.10.0 -- npm pack --dry-run
```

Expected: the build succeeds, the supply-chain guard reports no errors, and the package contains only `dist`, `README.md`, and `LICENSE`.

- [ ] **Step 3: Review the final branch**

Run:

```bash
git status --short --branch
git log --show-signature --oneline origin/main..HEAD
git diff --check origin/main...HEAD
```

Expected: the worktree is clean, all commits have good signatures, and the diff has no whitespace errors.

- [ ] **Step 4: Push the authorized feature branch**

Run:

```bash
git push -u origin tests/terminal-control
```

Expected: the remote feature branch points at the verified local HEAD. Do not create or merge a pull request without separate authorization.
