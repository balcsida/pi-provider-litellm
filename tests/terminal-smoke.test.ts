import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { type Session, TerminalControl } from "@kitlangton/terminal-control";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const enabled = process.env.LITELLM_TERMINAL_SMOKE === "1";
const waitTimeoutMs = 90_000;
const testTimeoutMs = 6 * waitTimeoutMs;
let terminal: TerminalControl | undefined;

async function withPi(run: (session: Session) => Promise<void>): Promise<void> {
  const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-terminal-"));
  try {
    await using session = await terminal?.launch({
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
      inheritEnv: true,
      viewport: { cols: 100, rows: 30 },
    });
    if (!session) throw new Error("Terminal control is not initialized");
    await run(session);
  } finally {
    await rm(agentDir, { force: true, recursive: true });
  }
}

async function submit(session: Session, text: string): Promise<void> {
  await session.keyboard.type(text);
  await session.keyboard.press("Enter");
}

async function waitForInitialModel(session: Session): Promise<void> {
  try {
    await session.screen.waitForText("vidaimock-openai", { timeoutMs: waitTimeoutMs });
  } catch (error) {
    const logs = await session.logs.text();
    throw new Error(`Pi exited before showing the smoke model:\n${logs}`, { cause: error });
  }
}

describe.skipIf(!enabled)("interactive Pi terminal smoke", () => {
  beforeAll(async () => {
    terminal = await TerminalControl.make();
  });

  afterAll(async () => {
    await terminal?.close();
  });

  it(
    "logs in to LiteLLM",
    async () => {
      await withPi(async (session) => {
        await waitForInitialModel(session);

        await submit(session, "/login litellm");
        await session.screen.waitForText("Enter LiteLLM proxy URL", { timeoutMs: waitTimeoutMs });
        await submit(session, process.env.LITELLM_BASE_URL ?? "http://127.0.0.1:4000");
        await session.screen.waitForText("Select login method", { timeoutMs: waitTimeoutMs });
        await submit(session, "1");
        await session.screen.waitForText("Enter API key", { timeoutMs: waitTimeoutMs });
        await submit(session, process.env.LITELLM_API_KEY ?? "sk-ci-litellm-smoke");

        await session.screen.waitForText("Logged in to LiteLLM", { timeoutMs: waitTimeoutMs });
      });
    },
    testTimeoutMs,
  );

  it(
    "refreshes LiteLLM models",
    async () => {
      await withPi(async (session) => {
        await waitForInitialModel(session);

        await submit(session, "/litellm-refresh");

        await session.screen.waitForText("LiteLLM: 2 models refreshed (source: model_info)", {
          timeoutMs: waitTimeoutMs,
        });
      });
    },
    testTimeoutMs,
  );

  it(
    "shows LiteLLM models in the model picker",
    async () => {
      await withPi(async (session) => {
        await waitForInitialModel(session);

        await submit(session, "/model");
        await session.screen.waitForText("anthropic/vidaimock-claude", { timeoutMs: waitTimeoutMs });
        const screen = await session.screen.text({ settleMs: 50, deadlineMs: 2_000 });

        expect(screen).toContain("vidaimock-openai");
        expect(screen).toContain("anthropic/vidaimock-claude");
      });
    },
    testTimeoutMs,
  );
});
