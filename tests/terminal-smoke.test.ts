import { execFile } from "node:child_process";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { type Session, TerminalControl } from "@kitlangton/terminal-control";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const piPath = resolve(repoRoot, "node_modules/.bin/pi");
const extensionPath = resolve(repoRoot, "dist/index.js");
const execFileAsync = promisify(execFile);
const enabled = process.env.LITELLM_TERMINAL_SMOKE === "1";
const waitTimeoutMs = 90_000;
const testTimeoutMs = 6 * waitTimeoutMs;
let terminal: TerminalControl | undefined;

async function withPi(run: (session: Session) => Promise<void>): Promise<void> {
  const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-terminal-"));
  try {
    const env = { ...process.env, PI_CODING_AGENT_DIR: agentDir };
    await execFileAsync(piPath, ["-e", extensionPath, "--list-models", "litellm"], {
      cwd: repoRoot,
      env,
    });
    await using session = await terminal?.launch({
      command: [
        piPath,
        "-e",
        extensionPath,
        "--provider",
        "litellm",
        "--model",
        process.env.LITELLM_CLI_SMOKE_MODEL ?? "vidaimock-openai",
        "--no-tools",
        "--no-session",
      ],
      cwd: repoRoot,
      env,
      inheritEnv: true,
      viewport: { cols: 100, rows: 30 },
    });
    if (!session) throw new Error("Terminal control is not initialized");
    await run(session);
  } finally {
    await rm(agentDir, { force: true, recursive: true });
  }
}

async function submit(session: Session, text: string, autocompleteText?: string): Promise<void> {
  await session.keyboard.type(text);
  if (text.startsWith("/")) {
    await session.screen.waitForText(text, { timeoutMs: waitTimeoutMs });
  }
  if (autocompleteText) {
    await session.screen.waitForText(autocompleteText, { timeoutMs: waitTimeoutMs });
    await session.keyboard.press("Escape");
    await session.screen.waitForIdle({ timeoutMs: waitTimeoutMs });
  }
  await session.keyboard.press("Enter");
}

it("dismisses command autocomplete without inspecting form values", async () => {
  const calls: unknown[] = [];
  const session = {
    keyboard: {
      press: async (key: string) => void calls.push(key),
      type: async () => void calls.push("type"),
    },
    screen: {
      waitForIdle: async (options?: { timeoutMs?: number }) => void calls.push(["waitForIdle", options]),
      waitForText: async (text: string | RegExp, options?: { timeoutMs?: number }) =>
        void calls.push(["waitForText", text, options]),
    },
  } as unknown as Session;

  await submit(session, "/login litellm", "LiteLLM API key");
  await submit(session, "sk-ci-litellm-smoke");

  expect(calls).toEqual([
    "type",
    ["waitForText", "/login litellm", { timeoutMs: 90_000 }],
    ["waitForText", "LiteLLM API key", { timeoutMs: 90_000 }],
    "Escape",
    ["waitForIdle", { timeoutMs: 90_000 }],
    "Enter",
    "type",
    "Enter",
  ]);
});

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

        await submit(session, "/login litellm", "LiteLLM API key");
        await session.screen.waitForText("Select authentication method for LiteLLM", { timeoutMs: waitTimeoutMs });
        await session.screen.waitForText("LiteLLM API key", { timeoutMs: waitTimeoutMs });
        await session.keyboard.press("Enter");
        await session.screen.waitForText("Enter LiteLLM proxy URL", { timeoutMs: waitTimeoutMs });
        await submit(session, process.env.LITELLM_BASE_URL ?? "http://127.0.0.1:4000");
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
        await session.screen.waitForText("Only showing models from configured providers", { timeoutMs: waitTimeoutMs });
        await session.screen.waitForText("anthropic/vidaimock-claude", { timeoutMs: waitTimeoutMs });
      });
    },
    testTimeoutMs,
  );
});
