// Drives PiG's interactive `/login litellm` through a real terminal against the mock proxy's SSO modes, then
// proves the stored credential works outside the TUI. Run from the repository root with the dev dependencies
// installed:
//
//   LITELLM_BASE_URL=http://127.0.0.1:48417 LOGIN_SMOKE_MODE=cli PIG_BIN=pig \
//     npx tsx pig/tools/login-smoke/login-smoke.mts
//
// LOGIN_SMOKE_MODE is cli (device code, auto-approved by the mock), pkce (the driver follows the authorize URL
// the way a browser would) or paste (a pasted SSO token exchanged for a virtual key). LOGIN_SMOKE_EXPLORE=1
// prints the screen after every step.
import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { type Session, TerminalControl } from "@kitlangton/terminal-control";

const here = dirname(fileURLToPath(import.meta.url));
const pigBin = process.env.PIG_BIN ?? "pig";
const extensionPath = process.env.LITELLM_EXT ?? resolve(here, "../../extensions/litellm");
const baseUrl = process.env.LITELLM_BASE_URL;
const mode = process.env.LOGIN_SMOKE_MODE ?? "cli";
const explore = process.env.LOGIN_SMOKE_EXPLORE === "1";
// LOGIN_SMOKE_COMMAND overrides the slash command; with LOGIN_SMOKE_EXPLORE_ONLY=1 the driver prints the screen
// after the command runs and exits, which is how PiG's own menus are inspected.
const command = process.env.LOGIN_SMOKE_COMMAND ?? "/login litellm";
const exploreOnly = process.env.LOGIN_SMOKE_EXPLORE_ONLY === "1";
const waitTimeoutMs = Number(process.env.LOGIN_SMOKE_TIMEOUT_MS ?? 90_000);
if (!baseUrl) throw new Error("LITELLM_BASE_URL is required");
if (!["cli", "pkce", "paste", "apikey"].includes(mode)) throw new Error(`unknown LOGIN_SMOKE_MODE ${mode}`);

const agentDir = await mkdtemp(join(tmpdir(), "pig-litellm-login-smoke-"));
const env: Record<string, string> = {};
for (const [key, value] of Object.entries(process.env)) {
  if (value !== undefined && !key.startsWith("LITELLM_")) env[key] = value;
}
env.PIG_CODING_AGENT_DIR = agentDir;
env.LITELLM_BASE_URL = baseUrl; // offered as the known proxy URL, so login confirms it with one keypress

function log(step: string): void {
  process.stderr.write(`[login-smoke] ${step}\n`);
}

// screen returns the current text even while an animation (a spinner) keeps the frame from settling.
async function screen(session: Session): Promise<string> {
  const snapshot = await session.screen.capture({ allowIncomplete: true });
  return snapshot.text;
}

// settle waits briefly for output to stop; a spinner never stops, so a timeout is not a failure.
async function settle(session: Session): Promise<void> {
  try {
    await session.screen.waitForIdle({ quietForMs: 150, timeoutMs: 3000 });
  } catch {
    // animated screen
  }
}

async function show(session: Session, step: string): Promise<void> {
  if (!explore) return;
  const text = await screen(session);
  process.stderr.write(`\n===== ${step} =====\n${text.replace(/[ \t]+$/gm, "").replace(/\n{3,}/g, "\n\n")}\n`);
}

async function waitForAny(session: Session, needles: string[], step: string): Promise<string> {
  let found = "";
  try {
    await session.screen.waitUntil(
      (snapshot) => {
        found = needles.find((needle) => snapshot.text.includes(needle)) ?? "";
        return found !== "";
      },
      { timeoutMs: waitTimeoutMs },
    );
  } catch (error) {
    process.stderr.write(`\n===== screen when waiting for ${needles.join(" | ")} (${step}) =====\n${await screen(session)}\n`);
    throw error;
  }
  return found;
}

async function submit(session: Session, text: string, visible = true): Promise<void> {
  await session.keyboard.type(text);
  if (visible) await session.screen.waitForText(text, { timeoutMs: waitTimeoutMs });
  await settle(session);
  await session.keyboard.press("Enter");
}

type StoredCredential = Record<string, unknown> & { type?: string; access?: string; expires?: number };

const authPath = join(agentDir, "auth.json");

// readAuth parses auth.json; a missing or half-written file reads as empty.
async function readAuth(): Promise<Record<string, StoredCredential>> {
  try {
    return JSON.parse(await readFile(authPath, "utf8")) as Record<string, StoredCredential>;
  } catch {
    return {};
  }
}

async function readCredential(): Promise<StoredCredential | undefined> {
  return (await readAuth()).litellm;
}

async function waitForCredential(session: Session): Promise<StoredCredential> {
  const deadline = Date.now() + waitTimeoutMs;
  while (Date.now() < deadline) {
    const credential = await readCredential();
    if (credential?.type === "oauth" && credential.access) return credential;
    await new Promise((done) => setTimeout(done, 500));
  }
  process.stderr.write(`\n===== screen while waiting for the stored credential =====\n${await screen(session)}\n`);
  throw new Error("no oauth credential was stored in auth.json");
}

function run(args: string[], extra: Record<string, string> = {}): string {
  const result = spawnSync(pigBin, args, { env: { ...env, ...extra }, encoding: "utf8", timeout: 300_000 });
  if (result.error) throw result.error;
  return `${result.stdout}\n${result.stderr}`;
}

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

const terminal = await TerminalControl.make();
try {
  await using session = await terminal.launch({
    command: [pigBin, "-e", extensionPath, "--no-tools", "--no-session"],
    cwd: agentDir,
    env,
    inheritEnv: false,
    // Wide enough that a PKCE authorize URL (two 43-character PKCE values plus encoded URLs) stays on one line.
    viewport: { cols: 900, rows: 40 },
  });

  await session.screen.waitForIdle({ timeoutMs: waitTimeoutMs });
  await show(session, "startup");

  await session.keyboard.type(command);
  await session.screen.waitForText(command, { timeoutMs: waitTimeoutMs });
  await session.screen.waitForIdle({ timeoutMs: waitTimeoutMs });
  await show(session, `typed ${command}`);
  // PiG's command popup (its rows start with "→ ") takes the first Enter to accept the completion; the command
  // runs on the next one. Wait for the popup so the two keystrokes land in order, as Pi's terminal smoke does.
  let popup = true;
  try {
    await session.screen.waitForText("→ ", { timeoutMs: 5000 });
  } catch {
    popup = false;
  }
  if (popup) {
    await session.keyboard.press("Enter");
    await session.screen.waitUntil((snapshot) => !snapshot.text.includes("→ "), { timeoutMs: waitTimeoutMs });
    await show(session, "completion accepted");
  }
  await session.keyboard.press("Enter");
  await settle(session);
  await show(session, "command submitted");
  if (exploreOnly) {
    await new Promise((done) => setTimeout(done, 3000));
    process.stderr.write(`\n===== explore only: screen after ${command} =====\n${await screen(session)}\n`);
    process.exit(0);
  }

  const first = await waitForAny(
    session,
    ["Select authentication method", "LiteLLM proxy URL", "Enter API key"],
    "after /login litellm",
  );
  await show(session, `after Enter: ${first}`);
  if (first === "Select authentication method") {
    const text = await screen(session);
    const label = ["Sign in with LiteLLM SSO", "Sign in with an account"].find((candidate) => text.includes(candidate));
    assert(label, `the method selector does not offer an account sign-in:\n${text}`);
    log(`method selector offers "${label}"`);
    if (mode === "apikey") {
      log("method selector offers LiteLLM SSO; choosing the API key instead");
      await session.keyboard.press("ArrowDown");
    } else {
      log("method selector offers LiteLLM SSO; choosing it");
    }
    await session.keyboard.press("Enter");
  } else if ((await screen(session)).includes("Login to LiteLLM SSO")) {
    // PiG opens the account flow directly for this provider, where Pi first asks for the method.
    assert(mode !== "apikey", "PiG offered no method selector, so the API-key login cannot be chosen");
    log("PiG opened the SSO flow directly, without a method selector");
  } else {
    throw new Error(`PiG went straight to the API-key login (${first}); it offers no account sign-in for this provider`);
  }

  await waitForAny(session, ["LiteLLM proxy URL"], "base URL prompt");
  await show(session, "base URL prompt");
  await session.keyboard.press("Enter"); // the known URL is the first option

  if (mode === "apikey") {
    await waitForAny(session, ["Enter API key"], "API key prompt");
    await show(session, "API key prompt");
    await submit(session, "test-key", false); // a secret prompt never echoes the value
    const deadline = Date.now() + waitTimeoutMs;
    let stored: StoredCredential | undefined;
    while (Date.now() < deadline && !(stored = await readCredential())) await new Promise((done) => setTimeout(done, 500));
    assert(stored?.type === "api_key" && stored.key === "test-key", `no api_key credential stored: ${JSON.stringify(stored)}`);
    const env = stored.env as Record<string, string> | undefined;
    assert(env?.LITELLM_BASE_URL === baseUrl, `the stored key does not carry the proxy URL: ${JSON.stringify(stored)}`);
    log("stored api_key credential carries the proxy URL");
  } else if (mode === "pkce") {
    const authorize = await session.screen.waitUntil(
      (snapshot) => /https?:\/\/127\.0\.0\.1:\d+\/oauth\/authorize\?\S+/.test(snapshot.text),
      { timeoutMs: waitTimeoutMs },
    );
    const url = authorize.text.match(/https?:\/\/127\.0\.0\.1:\d+\/oauth\/authorize\?\S+/)?.[0];
    assert(url, "no authorize URL on screen");
    await show(session, "authorize URL shown");
    log(`opening ${url.slice(0, 60)}... as the browser would`);
    const response = await fetch(url, { redirect: "follow" });
    assert(response.ok, `callback answered ${response.status}`);
  } else if (mode === "paste") {
    await waitForAny(session, ["Paste your SSO token"], "paste prompt");
    await show(session, "paste prompt");
    await submit(session, "sk-mock-pasted-sso-token", false); // a secret prompt never echoes the value
    await waitForAny(session, ["Generate a LiteLLM virtual key"], "virtual key prompt");
    await submit(session, "y");
  } else {
    await waitForAny(session, ["code", "Code"], "device code shown");
    await show(session, "device code shown");
  }

  const credential = mode === "apikey" ? undefined : await waitForCredential(session);
  if (credential) {
    await show(session, "credential stored");
    log(`stored oauth credential with keys ${Object.keys(credential).sort().join(", ")}`);
    assert(credential.baseUrl === baseUrl, `stored baseUrl ${String(credential.baseUrl)} differs from ${baseUrl}`);
  }
  if (credential && mode === "pkce") {
    assert(credential.flow === "litellm_cli_pkce", "pkce credential lacks flow=litellm_cli_pkce");
    assert(credential.refresh, "pkce credential lacks a refresh token");
    assert(credential.clientId && credential.tokenEndpoint && credential.resource, "pkce credential lacks registration fields");
  }
  if (credential && mode === "cli") assert(credential.refresh === "", "cli credential should have an empty refresh token");
  if (credential && mode === "paste") assert(String(credential.access).startsWith("sk-mock-"), "paste did not store the generated virtual key");

  await session.keyboard.press("Control+C");
  await session.keyboard.press("Control+C");
} finally {
  await terminal.close();
}

log("checking the stored credential outside the TUI (no LITELLM_API_KEY in the environment)");
const list = run(["-e", extensionPath, "--list-models"]);
const listed = (list.match(/^litellm\s+\S+/gm) ?? []).length;
assert(listed >= 3, `--list-models listed ${listed} litellm models:\n${list}`);
const chat = run(["-e", extensionPath, "--no-tools", "--model", "litellm/mock-chat", "-p", "say hi"]);
assert(chat.includes("mock reply to: say hi"), `chat through the stored credential failed:\n${chat}`);

if (mode === "pkce") {
  log("expiring the stored credential to force a refresh through PiG");
  const stored = await readAuth();
  const before = stored.litellm;
  assert(before, "auth.json lost the litellm credential");
  stored.litellm = { ...before, expires: Date.now() - 60_000 };
  await writeFile(authPath, JSON.stringify(stored, null, 2));
  const refreshed = run(["-e", extensionPath, "--no-tools", "--model", "litellm/mock-chat", "-p", "say hi"]);
  assert(refreshed.includes("mock reply to: say hi"), `chat after expiry failed:\n${refreshed}`);
  const after = (await readAuth()).litellm;
  assert(after, "auth.json lost the litellm credential after the refresh");
  assert(after.access !== before.access, "refresh did not rotate the access token");
  assert(after.refresh !== before.refresh, "refresh did not rotate the refresh token");
  assert((after.expires ?? 0) > Date.now(), "refreshed credential is not in the future");
  log("refresh rotated both tokens and extended the expiry");
}

await rm(agentDir, { recursive: true, force: true });
log(`${mode} login smoke passed`);
