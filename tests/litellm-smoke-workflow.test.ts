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

function readAgentNotes(): string {
  return readFileSync(resolve(repoRoot, "AGENTS.md"), "utf8");
}

describe("LiteLLM smoke workflow", () => {
  it("routes smoke completions through VidaiMock instead of real LLM APIs", () => {
    const workflow = readWorkflow();

    expect(workflow).toContain("Start VidaiMock");
    expect(workflow).toContain("Wait for VidaiMock");
    expect(workflow).toContain("VIDAIMOCK_BASE_URL: http://127.0.0.1:8100");
    expect(workflow).toContain("LITELLM_SMOKE_MODELS: vidaimock-openai anthropic/vidaimock-claude");
    expect(workflow).toContain("LITELLM_CLI_SMOKE_MODEL: vidaimock-openai");
    expect(workflow).toContain("model_name: vidaimock-openai");
    expect(workflow).toContain("model_name: anthropic/vidaimock-claude");
    expect(workflow).toContain("model: openai/gpt-4o-mini");
    expect(workflow).toContain("model: anthropic/claude-3-5-sonnet");
    expect(workflow).toContain("api_base: http://host.docker.internal:8100/v1");
    expect(workflow).toContain("api_base: http://host.docker.internal:8100");
    expect(workflow).toContain("--add-host=host.docker.internal:host-gateway");
    expect(workflow.match(/curl -fsS --connect-timeout 1 --max-time 3/g)).toHaveLength(2);
    expect(workflow).toContain("Run Pi CLI smoke");
    expect(workflow).toContain("./node_modules/.bin/pi -e ./dist/index.js --list-models litellm");
    expect(workflow).toContain("--provider litellm");
    expect(workflow).toContain('--model "$LITELLM_CLI_SMOKE_MODEL"');

    expect(workflow).not.toContain("models: read");
    expect(workflow).not.toContain("GH_MODELS_SMOKE_MODEL");
    expect(workflow).not.toContain("OPENAI_API_KEY");
    expect(workflow).not.toContain("ANTHROPIC_API_KEY");
    expect(workflow).not.toContain("GEMINI_API_KEY");
    expect(workflow).not.toContain("require_vendors");
    expect(workflow).not.toContain("model_name: kimi-vidaimock");
  });

  it("uses minimal permissions and a pinned, checksum-verified VidaiMock build", () => {
    const workflow = readWorkflow();

    expect(workflow).toMatch(/permissions:\n {2}contents: read/);
    expect(workflow).toMatch(/VIDAIMOCK_VERSION: v\d+\.\d+\.\d+$/m);
    expect(workflow).toMatch(/sha256sum -c "\$\{asset%\.tar\.gz\}\.sha256"/);
  });

  it("documents the mocked smoke workflow for agents without provider secrets", () => {
    const agents = readAgentNotes();
    const readme = readReadme();

    expect(agents).toContain("## Smoke And CI");
    expect(agents).toContain("VidaiMock");
    expect(agents).toContain("without real LLM API calls");
    expect(agents).toContain("do not add real provider secrets or GitHub Models permission");
    expect(agents).toContain("OpenAI-compatible and Anthropic routes");
    expect(agents).toContain("non-interactive Pi CLI smoke checks");
    expect(agents).not.toContain("Kimi-shaped routes");

    expect(readme).not.toContain("## Mocked LiteLLM smoke workflow");
    expect(readme).not.toContain("## Real LiteLLM smoke workflow");
    expect(readme).not.toContain("OPENAI_API_KEY");
    expect(readme).not.toContain("ANTHROPIC_API_KEY");
    expect(readme).not.toContain("GEMINI_API_KEY");
    expect(readme).not.toContain("require_vendors");
  });
});
