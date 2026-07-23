import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function readWorkflow(): string {
  return readFileSync(resolve(repoRoot, ".github/workflows/litellm-smoke.yml"), "utf8");
}

function readCiWorkflow(): string {
  return readFileSync(resolve(repoRoot, ".github/workflows/ci.yml"), "utf8");
}

function readReleaseWorkflow(): string {
  return readFileSync(resolve(repoRoot, ".github/workflows/release.yml"), "utf8");
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
    expect(workflow).toContain("Run community auth smoke");
    expect(workflow).toContain("Run Enterprise auth smoke");
    expect(workflow).toContain("npx tsx scripts/smoke-runner.ts");
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

  it("reuses the package publish gate in CI and release", () => {
    expect(readCiWorkflow()).toContain("run: npm run prepublishOnly");
    expect(readCiWorkflow()).not.toContain("run: npm pack --dry-run");
    expect(readReleaseWorkflow()).not.toContain("run: npm run check");
    expect(readReleaseWorkflow()).not.toContain("run: npm pack --dry-run");
    expect(readReleaseWorkflow()).toContain("run: npm publish --access public --provenance");
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

  it("separates community and Enterprise auth smoke", () => {
    const workflow = readWorkflow();
    const communityStart = workflow.indexOf("- name: Run community auth smoke");
    const enterpriseStart = workflow.indexOf("- name: Run Enterprise auth smoke");
    const cliStart = workflow.indexOf("- name: Run Pi CLI smoke");

    expect(communityStart).toBeGreaterThan(-1);
    expect(enterpriseStart).toBeGreaterThan(communityStart);
    expect(cliStart).toBeGreaterThan(enterpriseStart);

    const communityStep = workflow.slice(communityStart, enterpriseStart);
    const enterpriseStep = workflow.slice(enterpriseStart, cliStart);
    expect(communityStep).toContain("LITELLM_LICENSE: ''");
    expect(communityStep).toContain("run: npx tsx scripts/smoke-auth.ts");
    expect(enterpriseStep).toContain("if: $" + "{{ env.LITELLM_LICENSE != '' }}");
    expect(enterpriseStep).toContain("run: npx tsx scripts/smoke-auth.ts");
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

  it("runs one cold terminal lifecycle without preselecting an unavailable model", () => {
    const terminalSmoke = readTerminalSmoke();

    expect(terminalSmoke).toContain('it(\n    "logs in and selects LiteLLM models"');
    expect(terminalSmoke).toContain("process.env.PI_CODING_AGENT_DIR?.trim()");
    expect(terminalSmoke).toContain("if (!configuredAgentDir) await rm(agentDir, { force: true, recursive: true });");
    expect(terminalSmoke).toContain('waitForText("Warning: No models available"');
    expect(terminalSmoke).toContain('waitUntil((snapshot) => !snapshot.text.includes("Enter API key")');
    expect(terminalSmoke).not.toContain('execFileAsync(piPath, ["-e", extensionPath, "--list-models", "litellm"]');
    expect(terminalSmoke).not.toContain('"--provider",\n        "litellm"');
    expect(terminalSmoke).not.toContain('"--model",');
  });

  it("shares terminal login state with the later Pi CLI smoke", () => {
    const workflow = readWorkflow();
    const agentDir = "PI_CODING_AGENT_DIR: $" + "{{ runner.temp }}/pi-cli-smoke";
    const stepsStart = workflow.indexOf("    steps:");
    const initializeStart = workflow.indexOf("- name: Initialize shared Pi agent directory");
    const terminalStart = workflow.indexOf("- name: Run interactive Pi terminal smoke");
    const cliStart = workflow.indexOf("- name: Run Pi CLI smoke");
    const dumpLogsStart = workflow.indexOf("- name: Dump LiteLLM logs");

    expect(workflow.slice(0, stepsStart)).not.toContain(agentDir);
    expect(initializeStart).toBeGreaterThan(-1);
    expect(terminalStart).toBeGreaterThan(initializeStart);
    expect(cliStart).toBeGreaterThan(terminalStart);
    expect(dumpLogsStart).toBeGreaterThan(cliStart);
    expect(workflow.slice(initializeStart, terminalStart)).toContain(agentDir);
    expect(workflow.slice(initializeStart, terminalStart)).toContain('mkdir -p "$PI_CODING_AGENT_DIR"');
    expect(workflow.slice(terminalStart, cliStart)).toContain(agentDir);
    expect(workflow.slice(cliStart, dumpLogsStart)).toContain(agentDir);
    expect(workflow.slice(cliStart)).not.toContain('rm -rf "$PI_CODING_AGENT_DIR"');
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
