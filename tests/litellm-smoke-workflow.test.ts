import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function readWorkflow(): string {
  return readFileSync(resolve(repoRoot, ".github/workflows/litellm-smoke.yml"), "utf8");
}

function readReadme(): string {
  return readFileSync(resolve(repoRoot, "README.md"), "utf8");
}

function readTerminalSmoke(): string {
  return readFileSync(resolve(repoRoot, "tests/terminal-smoke.test.ts"), "utf8");
}

describe("LiteLLM smoke workflow", () => {
  it("routes smoke completions through VidaiMock instead of real LLM APIs", () => {
    const workflow = readWorkflow();

    expect(workflow).toContain("Start VidaiMock");
    expect(workflow).toContain("Wait for VidaiMock");
    expect(workflow).toContain("VIDAIMOCK_BASE_URL: http://127.0.0.1:8100");
    expect(workflow).toContain("LITELLM_DATABASE_URL: postgresql://litellm:litellm@host.docker.internal:5432/litellm");
    expect(workflow).toContain("docker.litellm.ai/berriai/litellm-database:main-latest");
    expect(workflow).toContain("docker.litellm.ai/berriai/litellm:main-latest");
    expect(workflow).toContain("LITELLM_SMOKE_MODELS: vidaimock-openai anthropic/vidaimock-claude");
    expect(workflow).toContain("LITELLM_SMOKE_EXPECT_SOURCE: model_info");
    expect(workflow).toContain("LITELLM_CLI_SMOKE_MODEL: vidaimock-openai");
    expect(workflow).toContain("model_name: vidaimock-openai");
    expect(workflow).toContain(`- model_name: anthropic/vidaimock-claude
              model_info:
                mode: chat
              litellm_params:`);
    expect(workflow).toContain("model: openai/gpt-4o-mini");
    expect(workflow).toContain("model: anthropic/claude-3-5-sonnet");
    expect(workflow).toContain("api_base: http://host.docker.internal:8100/v1");
    expect(workflow).toContain("api_base: http://host.docker.internal:8100");
    expect(workflow).toContain("--add-host=host.docker.internal:host-gateway");
    expect(workflow).toContain("-e LITELLM_LICENSE");
    expect(workflow).toContain("Start LiteLLM smoke database");
    expect(workflow).toContain("postgres:16-alpine");
    expect(workflow).toContain('admin_only_routes: ["/key/generate"]');
    expect(workflow).toContain("Run auth smoke");
    expect(workflow).toContain("npx tsx scripts/smoke-auth.ts");
    expect(workflow.match(/curl -fsS --connect-timeout 1 --max-time 3/g)).toHaveLength(2);
    expect(workflow).toContain("Run Pi CLI smoke");
    expect(workflow).toContain("Run interactive Pi terminal smoke");
    expect(workflow).toContain("LITELLM_TERMINAL_SMOKE: '1'");
    expect(workflow).toContain("npm test -- tests/terminal-smoke.test.ts");
    expect(workflow).toContain("./node_modules/.bin/pi -e ./dist/index.js --list-models litellm");
    expect(workflow).toContain("--provider litellm");
    expect(workflow).toContain('--model "$LITELLM_CLI_SMOKE_MODEL"');
    expect(workflow).toContain("LITELLM_CLI_SMOKE_MODEL_ANTHROPIC: anthropic/vidaimock-claude");
    expect(workflow).toContain('--model "$LITELLM_CLI_SMOKE_MODEL_ANTHROPIC"');
    expect(workflow).toContain('grep -F "Anthropic mock response"');

    expect(workflow).not.toContain("models: read");
    expect(workflow).not.toContain("GH_MODELS_SMOKE_MODEL");
    expect(workflow).not.toContain("OPENAI_API_KEY");
    expect(workflow).not.toContain("ANTHROPIC_API_KEY");
    expect(workflow).not.toContain("GEMINI_API_KEY");
    expect(workflow).not.toContain("require_vendors");
    expect(workflow).not.toContain("model_name: kimi-vidaimock");
  });

  it("runs for path-filtered pull requests", () => {
    expect(readWorkflow()).toContain(`pull_request:
    paths:
      - '.github/workflows/litellm-smoke.yml'
      - 'package-lock.json'
      - 'package.json'
      - 'scripts/smoke*.ts'
      - 'src/**'
      - 'tests/**'`);
  });

  it("does not expose the optional license secret to pull requests", () => {
    expect(readWorkflow()).toContain(
      "LITELLM_LICENSE: $" + "{{ github.event_name != 'pull_request' && secrets.LITELLM_LICENSE || '' }}",
    );
  });

  it("selects the unlicensed image for pull requests", () => {
    expect(readWorkflow()).toContain(
      "LITELLM_IMAGE: $" +
        "{{ github.event_name != 'pull_request' && secrets.LITELLM_LICENSE != '' && 'docker.litellm.ai/berriai/litellm-database:main-latest' || 'docker.litellm.ai/berriai/litellm:main-latest' }}",
    );
  });

  it("gates auth smoke on the optional LiteLLM license", () => {
    expect(readWorkflow()).toMatch(/- name: Run auth smoke\n {8}if: \$\{\{ env\.LITELLM_LICENSE != '' \}\}/);
  });

  it("uses minimal permissions and a pinned, checksum-verified VidaiMock build", () => {
    const workflow = readWorkflow();

    expect(workflow).toMatch(/permissions:\n {2}contents: read/);
    expect(workflow).toMatch(/VIDAIMOCK_VERSION: v\d+\.\d+\.\d+$/m);
    expect(workflow).toMatch(/sha256sum -c "\$\{asset%\.tar\.gz\}\.sha256"/);
  });

  it("preserves the workflow environment in the terminal smoke", () => {
    expect(readTerminalSmoke()).toContain("inheritEnv: true");
  });

  it("primes each fresh terminal smoke agent directory", () => {
    const terminalSmoke = readTerminalSmoke();
    const primeAndLaunch = `const env = { ...process.env, PI_CODING_AGENT_DIR: agentDir };
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
      inheritEnv: true,`;

    expect(terminalSmoke).toContain(primeAndLaunch);
  });

  it("documents the mocked smoke workflow without provider secrets", () => {
    const readme = readReadme();

    expect(readme).toContain("## Mocked LiteLLM smoke workflow");
    expect(readme).toContain("VidaiMock");
    expect(readme).toContain("does not call real LLM APIs");
    expect(readme).toContain("No provider API keys or GitHub Models permission are required");
    expect(readme).toContain("OpenAI-compatible and Anthropic routes");
    expect(readme).toContain("optional Postgres-backed auth checks when `LITELLM_LICENSE` is configured");
    expect(readme).toContain("non-interactive Pi CLI smoke");
    expect(readme).toContain("interactive Pi TUI smoke");
    expect(readme).not.toContain("Kimi-shaped routes");

    expect(readme).not.toContain("## Real LiteLLM smoke workflow");
    expect(readme).not.toContain("OPENAI_API_KEY");
    expect(readme).not.toContain("ANTHROPIC_API_KEY");
    expect(readme).not.toContain("GEMINI_API_KEY");
    expect(readme).not.toContain("require_vendors");
  });
});
