import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";
import { discoverModels } from "../src/discover.js";
import { parsePushTriggers } from "./workflow-triggers.js";

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

// Parse the `model_list:` entries out of the workflow's LiteLLM config heredoc, so the
// deployments CI actually runs are the ones asserted here.
function workflowDeployments(): Array<Record<string, unknown>> {
  const workflow = readWorkflow();
  const block = workflow.slice(workflow.indexOf("model_list:"), workflow.indexOf("general_settings:"));
  return block
    .split(/^\s*- model_name:/m)
    .slice(1)
    .map((entry, index) => {
      const value = (key: string) => entry.match(new RegExp(`^\\s*${key}:\\s*(\\S.*?)\\s*$`, "m"))?.[1];
      const list = (key: string) =>
        entry
          .match(new RegExp(`^\\s*${key}:\\s*\\[(.*?)\\]\\s*$`, "m"))?.[1]
          .split(",")
          .map((item) => item.trim())
          .filter(Boolean);
      return {
        model_name: entry.split("\n")[0]?.trim(),
        model_info: {
          id: `deployment-${index}`,
          ...(value("mode") ? { mode: value("mode") } : {}),
          ...(value("litellm_provider") ? { litellm_provider: value("litellm_provider") } : {}),
          ...(list("supported_openai_params") ? { supported_openai_params: list("supported_openai_params") } : {}),
        },
        litellm_params: {
          model: value("model"),
          ...(list("allowed_openai_params") ? { allowed_openai_params: list("allowed_openai_params") } : {}),
        },
      };
    });
}

function workflowExpectedApis(): Map<string, string> {
  const raw = readWorkflow().match(/^\s*LITELLM_SMOKE_EXPECT_APIS:\s*(.+)$/m)?.[1] ?? "";
  return new Map(
    raw
      .trim()
      .split(/\s+/)
      .map((entry) => {
        const separator = entry.lastIndexOf("=");
        return [entry.slice(0, separator), entry.slice(separator + 1)] as const;
      }),
  );
}

function readTerminalSmoke(): string {
  return readFileSync(resolve(repoRoot, "tests/terminal-smoke.test.ts"), "utf8");
}

describe("LiteLLM smoke workflow config", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  // The workflow declares the deployments AND the expected transports. Those two
  // declarations drifted apart once selection required positive compatibility
  // evidence: the Anthropic route pointed at an uncatalogued Claude, so it reduced to
  // Chat while the workflow still expected Messages, and only CI could notice.
  it("resolves each configured deployment to the transport the workflow expects", async () => {
    const deployments = workflowDeployments();
    const expected = workflowExpectedApis();
    expect(deployments.length).toBeGreaterThan(0);
    expect(expected.size).toBeGreaterThan(0);
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return Response.json({ data: deployments });
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await discoverModels("https://litellm.example.com", "sk-test", {});
    const resolved = new Map(result.models.map((model) => [model.id, model.api]));

    for (const [modelId, expectedApi] of expected) {
      expect(resolved.get(modelId), `workflow model ${modelId}`).toBe(expectedApi);
    }
    // The Messages route is the reason this workflow exists; prove one is present.
    expect([...resolved.values()]).toContain("anthropic-messages");
  });
});

describe("LiteLLM smoke workflow", () => {
  it("routes smoke completions through VidaiMock instead of real LLM APIs", () => {
    const workflow = readWorkflow();

    expect(workflow).toContain("Start VidaiMock");
    expect(workflow).toContain("Wait for VidaiMock");
    expect(workflow).toContain("run: npm ci --ignore-scripts");
    expect(workflow).not.toMatch(/run: npm ci\s*$/m);
    expect(workflow).toContain("VIDAIMOCK_BASE_URL: http://127.0.0.1:8100");
    expect(workflow).toContain("LITELLM_DATABASE_URL: postgresql://litellm:litellm@host.docker.internal:5432/litellm");
    expect(workflow).toContain("docker.litellm.ai/berriai/litellm-database@sha256:");
    expect(workflow).toContain("docker.litellm.ai/berriai/litellm@sha256:");
    expect(workflow).toContain(
      "LITELLM_SMOKE_MODELS: vidaimock-openai anthropic/vidaimock-claude vidaimock-responses grouped-vidaimock",
    );
    expect(workflow).toContain("LITELLM_SMOKE_EXPECT_SOURCE: model_info");
    expect(workflow).toContain(
      "LITELLM_SMOKE_EXPECT_APIS: vidaimock-openai=openai-completions" +
        " anthropic/vidaimock-claude=anthropic-messages vidaimock-responses=openai-responses" +
        " grouped-vidaimock=openai-completions",
    );
    expect(workflow).toContain("LITELLM_SMOKE_EXPECT_RESPONSE_COST:");
    expect(workflow).toContain("LITELLM_SMOKE_REQUIRE_ALL_PROTOCOLS: '1'");
    expect(workflow).toContain("LITELLM_CLI_SMOKE_MODEL: vidaimock-openai");
    expect(workflow).toContain("model_name: vidaimock-openai");
    expect(workflow).toContain(`- model_name: anthropic/vidaimock-claude
              model_info:
                mode: chat
                litellm_provider: anthropic
              litellm_params:`);
    expect(workflow).toContain("model: openai/gpt-4o-mini");
    expect(workflow).toContain("model: anthropic/claude-sonnet-4-6");
    expect(workflow).toContain("model_name: vidaimock-responses");
    expect(workflow).toContain("mode: responses");
    expect(workflow.match(/model_name: grouped-vidaimock/g)).toHaveLength(2);
    expect(workflow).toContain("Capture deployment-group model info");
    expect(workflow).toContain("run: npx tsx scripts/smoke-deployment-group.ts");
    expect(workflow).toContain("api_base: http://host.docker.internal:8100/v1");
    expect(workflow).toContain("api_base: http://host.docker.internal:8100");
    expect(workflow).toContain("--add-host=host.docker.internal:host-gateway");
    expect(workflow).toContain("-e LITELLM_LICENSE");
    expect(workflow).toContain("Start LiteLLM smoke database");
    expect(workflow).toMatch(/postgres:16-alpine@sha256:[a-f0-9]{64}/);
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
    expect(workflow).toContain("./node_modules/.bin/pi -e . --list-models litellm");
    expect(workflow).not.toContain("-e ./dist/index.js");
    expect(workflow).toContain("--provider litellm");
    expect(workflow).toContain('--model "$LITELLM_CLI_SMOKE_MODEL"');
    expect(workflow).toContain("LITELLM_CLI_SMOKE_MODEL_ANTHROPIC: anthropic/vidaimock-claude");
    expect(workflow).toContain('--model "$LITELLM_CLI_SMOKE_MODEL_ANTHROPIC"');
    expect(workflow).toContain('grep -F "Anthropic mock response"');
    // Endpoint coverage is proven from captured request logs, not response text.
    expect(workflow).toContain("Assert captured LiteLLM endpoint logs");
    expect(workflow).toContain('POST /v1/chat/completions HTTP/1.1" 200');
    expect(workflow).toContain('POST /v1/responses HTTP/1.1" 200');
    // The smoke runner posts to /v1/messages itself, so coverage alone cannot prove the
    // extension's own path: it is pinned by a scoped grep plus an occurrence count.
    expect(workflow).toContain('messages_log_since="$(date -u +%Y-%m-%dT%H:%M:%SZ)"');
    expect(workflow).toContain('docker logs --since "$messages_log_since" litellm-smoke');
    expect(workflow).toContain('POST /v1/messages HTTP/1.1" 200');
    expect(workflow).toContain(".tmp/litellm.log || true)");
    expect(workflow).toContain(`messages_requests="\${messages_requests:-0}"`);
    expect(workflow).toContain('test "$messages_requests" -ge 2');

    expect(workflow).not.toContain("models: read");
    expect(workflow).not.toContain("GH_MODELS_SMOKE_MODEL");
    expect(workflow).not.toContain("OPENAI_API_KEY");
    expect(workflow).not.toContain("ANTHROPIC_API_KEY");
    expect(workflow).not.toContain("GEMINI_API_KEY");
    expect(workflow).not.toContain("require_vendors");
    expect(workflow).not.toContain("model_name: kimi-vidaimock");
  });

  it("reuses the package publish gate in CI and release", () => {
    const manifest = JSON.parse(readFileSync(join(repoRoot, "package.json"), "utf8")) as {
      scripts: Record<string, string>;
    };
    expect(readCiWorkflow()).toContain("run: npm run prepublishOnly");
    expect(manifest.scripts.check).toContain("npm run supply-chain:guard");
    // `npm pack` selects by the `files` manifest, not by what is on disk, so the
    // post-build run is not there to observe `dist/` existing. It is there to catch a
    // manifest that re-authorizes `dist`: with `files` listing it, the pre-build run
    // passes on a fresh checkout that has no `dist/` yet, and only a run after `build`
    // fails. Pinned as full equality so the ordering cannot be relaxed and
    // `npm run check` -- CI's only quality gate -- cannot be dropped.
    expect(manifest.scripts.prepublishOnly).toBe(
      "npm run check && npm run clean && npm run build && npm run supply-chain:guard",
    );
    expect(readCiWorkflow()).not.toContain("run: npm pack --dry-run");
    expect(readReleaseWorkflow()).not.toContain("run: npm pack --dry-run");
    expect(readReleaseWorkflow()).not.toContain("run: npm run prepublishOnly");
    expect(readReleaseWorkflow()).toContain("run: npm publish --access public --provenance");
  });

  it("keeps the publish flow tag-only", () => {
    expect(parsePushTriggers(readReleaseWorkflow())).toEqual({
      branches: [],
      tags: ["v*.*.*"],
    });
  });

  // `--list-models` also reports models from Pi's own store, so this leg must run against an
  // agent dir no earlier step has populated, and must prove it discriminates.
  it("isolates and negative-controls the list-models smoke leg", () => {
    const workflow = readWorkflow();

    expect(workflow).toContain('list_models_dir="$' + '{{ runner.temp }}/pi-list-models"');
    expect(workflow).toContain(
      'PI_CODING_AGENT_DIR="$list_models_dir" ./node_modules/.bin/pi -e . --list-models litellm',
    );
    expect(workflow).toContain('p.pi.extensions=["./src/does-not-exist.ts"]');
    expect(workflow).toContain("negative control failed: models listed without a loadable entrypoint");
    // The shared agent dir stays available to the completion legs, which need the login the
    // terminal smoke performed, but must not be what --list-models reads.
    expect(workflow).toContain('export PI_CODING_AGENT_DIR="$' + '{{ runner.temp }}/pi-cli-smoke"');
  });

  it("does not leave a dist build step in the smoke job", () => {
    expect(readWorkflow()).not.toContain("run: npm run build");
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
        "{{ github.event_name != 'pull_request' && secrets.LITELLM_LICENSE != '' && 'docker.litellm.ai/berriai/litellm-database@sha256:8b229a4b48fbe62d7f994b502106c3c1dbab958c07934fb446ac0e048a62745e' || 'docker.litellm.ai/berriai/litellm@sha256:f2dc9ba8a62cf2c51e3ed00e6975f4c70bb577b8ef0c2d7040e3228dc7d42b09' }}",
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

  it("pins downloaded and container smoke dependencies by repository-owned digests", () => {
    const workflow = readWorkflow();

    expect(workflow).toMatch(/permissions:\n {2}contents: read/);
    expect(workflow).toMatch(/VIDAIMOCK_VERSION: v\d+\.\d+\.\d+$/m);
    expect(workflow).toMatch(/VIDAIMOCK_SHA256: [a-f0-9]{64}$/m);
    expect(workflow).toMatch(/\(cd \.tmp && printf '%s {2}%s\\n' "\$VIDAIMOCK_SHA256" "\$asset" \| sha256sum -c -\)/);
    expect(workflow).not.toMatch(/\$\{asset%\.tar\.gz\}\.sha256/);
    expect(workflow).toMatch(/postgres:16-alpine@sha256:[a-f0-9]{64}/);
    expect(workflow).not.toMatch(/\s+postgres:16-alpine\s*$/m);
  });

  it("preserves the workflow environment in the terminal smoke", () => {
    expect(readTerminalSmoke()).toContain("inheritEnv: true");
  });

  it("runs one cold terminal lifecycle without preselecting an unavailable model", () => {
    const terminalSmoke = readTerminalSmoke();

    expect(terminalSmoke).toContain('it(\n    "logs in and selects LiteLLM models"');
    expect(terminalSmoke).toContain("process.env.PI_CODING_AGENT_DIR?.trim()");
    expect(terminalSmoke).toContain("if (!configuredAgentDir) await rm(agentDir, { force: true, recursive: true });");
    expect(terminalSmoke).toContain('waitForText("[Extensions]"');
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
    // The CLI step exports the shared dir inside its script rather than declaring it as step
    // env, because its --list-models leg deliberately runs against a separate empty dir.
    expect(workflow.slice(cliStart, dumpLogsStart)).toContain(
      'export PI_CODING_AGENT_DIR="$' + '{{ runner.temp }}/pi-cli-smoke"',
    );
    expect(workflow.slice(cliStart)).not.toContain('rm -rf "$PI_CODING_AGENT_DIR"');
  });

  it("documents the mocked smoke workflow without provider secrets", () => {
    const readme = readReadme();

    expect(readme).toContain("## Mocked LiteLLM smoke workflow");
    expect(readme).toContain("VidaiMock");
    expect(readme).toContain("does not call real LLM APIs");
    expect(readme).toContain("No provider API keys or GitHub Models permission are required");
    expect(readme).toContain("route-distinct Chat, Responses, native Messages, and mixed-deployment models");
    expect(readme).toContain("x-litellm-response-cost");
    expect(readme).toContain("asserts each model's expected API");
    expect(readme).toContain("proves endpoint coverage from captured LiteLLM request logs rather than response text");
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
