import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createPi, loadExtension } from "./test-helpers.js";

const ENV_KEYS = ["LITELLM_BASE_URL", "LITELLM_API_KEY"];
const ORIGINAL_ENV = new Map(ENV_KEYS.map((key) => [key, process.env[key]]));

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

async function writeModelsConfig(agentDir: string, modelId: string, name: string): Promise<void> {
  await writeFile(
    join(agentDir, "models.json"),
    JSON.stringify({ providers: { litellm: { modelOverrides: { [modelId]: { name, contextWindow: 321 } } } } }),
    "utf8",
  );
}

afterEach(() => {
  for (const key of ENV_KEYS) {
    const original = ORIGINAL_ENV.get(key);
    if (original === undefined) delete process.env[key];
    else process.env[key] = original;
  }
  vi.restoreAllMocks();
  vi.resetModules();
});

describe("Pi core model overrides", () => {
  it("applies reloaded overrides to cached and refreshed LiteLLM models", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-overrides-"));
    process.env.LITELLM_BASE_URL = "https://litellm.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    await writeModelsConfig(agentDir, "cached-model", "Cached override");
    await writeFile(
      join(agentDir, "models-store.json"),
      JSON.stringify({
        litellm: {
          checkedAt: Date.now(),
          models: [
            {
              id: "cached-model",
              name: "cached-model",
              provider: "litellm",
              api: "openai-completions",
              baseUrl: "https://litellm.example.com/v1",
              reasoning: false,
              input: ["text"],
              cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
              contextWindow: 128_000,
              maxTokens: 4096,
            },
          ],
        },
      }),
      "utf8",
    );
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [{ model_name: "refreshed-model", model_info: { mode: "chat" } }],
        });
      }
      if (url.endsWith("/mcp-rest/tools/list")) return jsonResponse(200, { tools: [] });
      throw new Error(`unexpected URL: ${url}`);
    });

    const runtime = await ModelRuntime.create({
      authPath: join(agentDir, "auth.json"),
      modelsPath: join(agentDir, "models.json"),
      allowModelNetwork: false,
    });
    const pi = createPi();
    pi.registerProvider = (provider) => runtime.registerNativeProvider(provider);
    await (await loadExtension(agentDir))(pi);
    await runtime.refresh({ allowNetwork: false });

    expect(runtime.getModel("litellm", "cached-model")).toMatchObject({ name: "Cached override", contextWindow: 321 });

    await writeModelsConfig(agentDir, "refreshed-model", "Refreshed override");
    await runtime.reloadConfig();
    await runtime.refresh({ allowNetwork: true, force: true });

    expect(runtime.getModel("litellm", "refreshed-model")).toMatchObject({
      name: "Refreshed override",
      contextWindow: 321,
    });
  });
});
