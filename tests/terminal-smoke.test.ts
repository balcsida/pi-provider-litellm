import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { type Session, TerminalControl } from "@kitlangton/terminal-control";
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
