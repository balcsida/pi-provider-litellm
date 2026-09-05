import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import { afterAll, afterEach, describe, expect, it, vi } from "vitest";

const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-probe-"));
vi.mock("@earendil-works/pi-coding-agent", () => ({
  getAgentDir: () => agentDir,
}));

import {
  acceptanceOracle,
  compareLive,
  errorClass,
  normalizeBaseUrl,
  type ProbeSnapshot,
  parseProbeArgs,
  probeDiscovery,
  protocolPrediction,
  publicCatalogOptions,
  readSnapshot,
  reasoningPrediction,
} from "../scripts/probe-proxy.js";

const snapshot: ProbeSnapshot = {
  modelInfo: {
    data: [
      {
        model_name: "route-gpt",
        litellm_params: { model: "azure/gpt-5", allowed_openai_params: ["reasoning_effort"] },
        model_info: {
          base_model: "openai/gpt-5",
          litellm_provider: "azure",
          supports_reasoning: true,
          supports_low_reasoning_effort: true,
          max_input_tokens: 1000,
          max_output_tokens: 100,
        },
      },
    ],
  },
  modelGroupInfo: { data: [] },
  models: { data: [{ id: "route-gpt", owned_by: "openai" }] },
};

afterAll(async () => {
  await rm(agentDir, { recursive: true, force: true });
});

afterEach(() => vi.unstubAllGlobals());

const probeModel = {
  id: "route-gpt",
  publicSources: [],
  liteLLMFlags: {},
  api: "openai-completions",
  reasoning: true,
  thinkingLevelMap: { low: "low", high: null, xhigh: null },
  limits: { context: 1000, output: 100 },
  cost: {},
  predictions: { protocol: "openai-responses", reasoning: { low: true, high: false } },
};

describe("parseProbeArgs", () => {
  it("parses snapshot and bounded live options", () => {
    expect(
      parseProbeArgs([
        "--base-url",
        "https://proxy.example/v1/",
        "--api-key",
        "secret",
        "--src",
        "/tmp/tree/src",
        "--live",
        "--max-requests",
        "12",
      ]),
    ).toMatchObject({ baseUrl: "https://proxy.example", src: "/tmp/tree/src", live: true, maxRequests: 12 });
  });

  it("normalizes proxy roots", () => {
    expect(normalizeBaseUrl("https://proxy.example/v1/")).toBe("https://proxy.example");
  });
});

describe("readSnapshot", () => {
  it("does not infer companions from a filename without model-info", async () => {
    const dir = await mkdtemp(join(tmpdir(), "probe-snapshot-"));
    const file = join(dir, "snapshot.json");
    const value = { data: [{ model_name: "route-gpt" }] };
    await writeFile(file, JSON.stringify(value));

    await expect(readSnapshot(file)).resolves.toEqual({ modelInfo: value });
    expect(JSON.parse(await readFile(file, "utf8"))).toEqual(value);
  });
});

describe("predictions", () => {
  it("treats the Azure GA boundary as Responses-compatible", () => {
    expect(
      protocolPrediction({
        model_name: "route-gpt",
        litellm_params: { model: "azure/gpt-5", api_version: "2025-03-01" } as never,
        model_info: { base_model: "openai/gpt-5", litellm_provider: "azure" },
      }),
    ).toBe("openai-responses");
  });

  it("predicts rejection when reasoning_effort is unavailable", () => {
    expect(reasoningPrediction({ model_name: "route-gpt" }, ["low", "medium", "high"])).toMatchObject({
      low: false,
      medium: false,
      high: false,
    });
  });

  it("seeds off and minimal as selectable when reasoning_effort is available", () => {
    expect(
      reasoningPrediction(
        {
          model_name: "route-gpt",
          litellm_params: { model: "openai/route-gpt", allowed_openai_params: ["reasoning_effort"] } as never,
        },
        undefined,
      ),
    ).toMatchObject({ off: true, minimal: true, low: true, medium: true, high: true });
  });
});

describe("probeDiscovery", () => {
  it("uses a probe-specific models.dev cache only for live probes", () => {
    const probeCachePath = join(getAgentDir(), "litellm-probe-models-dev.json");
    expect(publicCatalogOptions(undefined)).toEqual({ cachePath: probeCachePath });
    expect(probeCachePath).not.toBe(join(getAgentDir(), "litellm-models-dev.json"));
    expect(publicCatalogOptions(snapshot)).toEqual({ offline: true });
  });

  it("fails when model info is empty for a non-empty discovery", async () => {
    const dir = await mkdtemp(join(tmpdir(), "probe-empty-info-"));
    const sourceDir = join(dir, "src");
    await mkdir(sourceDir);
    const discoverFile = join(sourceDir, "discover.ts");
    await writeFile(
      discoverFile,
      'export async function discoverModels() { return { source: "test", models: [{ id: "route-gpt", api: "openai-completions", contextWindow: 1000, maxTokens: 100, cost: {} }] }; }\n',
    );
    vi.stubGlobal("fetch", async () => new Response(JSON.stringify({ data: [] }), { status: 200 }));

    await expect(probeDiscovery({ baseUrl: "https://proxy.example/v1", apiKey: "secret", src: dir })).rejects.toThrow(
      "/model/info returned zero rows for 1 discovered models",
    );
  });

  it("injects snapshot fetch and reports identity, flags, selected metadata, and predictions", async () => {
    const dir = await mkdtemp(join(tmpdir(), "probe-proxy-"));
    const file = join(dir, "snapshot.json");
    await writeFile(file, JSON.stringify(snapshot));

    const report = await probeDiscovery({ snapshot: file, src: join(process.cwd(), "src") });

    expect(report.source).toBe("model_info");
    expect(report.models).toHaveLength(1);
    expect(report.models[0]).toMatchObject({
      id: "route-gpt",
      identity: { provider: "openai", modelId: "gpt-5", family: "openai" },
      liteLLMFlags: { supports_low_reasoning_effort: true },
      api: "openai-completions",
      reasoning: true,
      limits: { context: 1000, output: 100 },
      predictions: { protocol: "openai-responses", reasoning: { low: true, medium: true, high: true } },
    });
  });
});

describe("live outcomes", () => {
  it("classifies LiteLLM errors from message objects and raw strings", () => {
    expect(errorClass({ error: { code: "400", message: "litellm.UnsupportedParamsError: unsupported" } })).toBe(
      "UnsupportedParamsError",
    );
    expect(errorClass('{"message":"litellm.UnsupportedParamsError: unsupported"}')).toBe("UnsupportedParamsError");
  });

  it("derives classified acceptance without classifying unknown outcomes", () => {
    expect(acceptanceOracle({ status: 200 })).toBe(true);
    expect(acceptanceOracle({ status: 400, errorClass: "UnsupportedParamsError" })).toBe(false);
    expect(acceptanceOracle({ status: 500, errorClass: "InternalServerError" })).toBeNull();
  });

  it("records unclassified outcomes and applies soundness and completeness", () => {
    expect(
      compareLive(
        [
          {
            path: "chat",
            model: "route-gpt",
            level: "medium",
            status: 500,
            errorClass: "InternalServerError",
            accepted: null,
          },
          { path: "chat", model: "route-gpt", level: "low", status: 400, accepted: false },
          { path: "chat", model: "route-gpt", level: "high", status: 200, accepted: true },
          { path: "chat", model: "route-gpt", level: "xhigh", status: 200, accepted: true },
        ],
        [probeModel],
      ),
    ).toEqual({
      mismatches: [
        "route-gpt chat medium: unclassified HTTP 500 InternalServerError",
        "route-gpt chat low: predicted selectable, actual rejected",
        "route-gpt chat high: accepted but not predicted selectable",
      ],
      informational: ["route-gpt chat xhigh: accepted but not offered"],
    });
  });

  it("reports minimal soundness mismatches and informational completeness gaps", () => {
    const minimalModel = {
      ...probeModel,
      thinkingLevelMap: { minimal: "minimal" },
      predictions: { ...probeModel.predictions, reasoning: { minimal: true } },
    };
    const minimalNotOfferedModel = {
      ...probeModel,
      thinkingLevelMap: { minimal: null },
      predictions: { ...probeModel.predictions, reasoning: { minimal: false } },
    };

    expect(
      compareLive(
        [{ path: "chat", model: "route-gpt", level: "minimal", status: 400, accepted: false }],
        [minimalModel],
      ).mismatches,
    ).toEqual(["route-gpt chat minimal: predicted selectable, actual rejected"]);
    expect(
      compareLive(
        [{ path: "chat", model: "route-gpt", level: "minimal", status: 200, accepted: true }],
        [minimalNotOfferedModel],
      ),
    ).toEqual({
      mismatches: [],
      informational: ["route-gpt chat minimal: accepted but not offered"],
    });
  });

  it("treats accepted but unoffered off as generation-contract information", () => {
    expect(
      compareLive(
        [{ path: "chat", model: "route-gpt", level: "off", status: 200, accepted: true }],
        [{ ...probeModel, thinkingLevelMap: { off: null } }],
      ),
    ).toEqual({
      mismatches: [],
      informational: ["route-gpt chat off: accepted but not offered"],
    });
  });

  it("reports oracle rows whose model is missing from discovery", () => {
    expect(
      compareLive([{ path: "chat", model: "missing-model", level: "low", status: 400, accepted: false }], [probeModel]),
    ).toEqual({
      mismatches: ["missing-model chat low: model missing from discovery"],
      informational: [],
    });
  });

  it("matches the committed production acceptance oracle with four informational xhigh rows", async () => {
    const report = await probeDiscovery({
      snapshot: join(process.cwd(), "tests/fixtures/proxy/prod-model-info-2026-09-04.json"),
    });
    const oracle = JSON.parse(
      await readFile(join(process.cwd(), "tests/fixtures/proxy/expected-acceptance-2026-09-04.json"), "utf8"),
    );

    expect(compareLive(oracle, report.models)).toEqual({
      mismatches: [],
      informational: [
        "o3 chat xhigh: accepted but not offered",
        "deepseek-v4-flash chat xhigh: accepted but not offered",
        "kimi-k3 chat xhigh: accepted but not offered",
        "glm-5 chat xhigh: accepted but not offered",
      ],
    });
  });
});
