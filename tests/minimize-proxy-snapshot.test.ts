import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
  assertSafeFixture,
  deriveAcceptanceOracle,
  minimizeModelGroups,
  minimizeModelInfo,
} from "../scripts/minimize-proxy-snapshot.js";

describe("minimizeModelInfo", () => {
  it("keeps only allowlisted fields and replaces api_base", () => {
    const minimized = minimizeModelInfo({
      data: [
        {
          model_name: "route-gpt",
          api_key: "secret",
          litellm_params: {
            model: "azure/gpt-5",
            custom_llm_provider: "azure",
            api_base: "https://private.example/v1",
            api_version: "2025-03-01",
            allowed_openai_params: ["reasoning_effort", "extra_headers"],
            timeout: 10,
          },
          model_info: {
            base_model: "openai/gpt-5",
            supported_openai_params: ["reasoning_effort", "extra_headers"],
            access_groups: ["private"],
          },
        },
      ],
    });

    expect(minimized).toEqual({
      data: [
        {
          model_name: "route-gpt",
          litellm_params: {
            model: "azure/gpt-5",
            custom_llm_provider: "azure",
            api_base: "https://example.invalid",
            api_version: "2025-03-01",
            allowed_openai_params: ["reasoning_effort"],
          },
          model_info: {
            base_model: "openai/gpt-5",
            supported_openai_params: ["reasoning_effort"],
          },
        },
      ],
    });
  });
});

describe("minimizeModelGroups", () => {
  it("keeps only allowlisted group fields", () => {
    expect(
      minimizeModelGroups({
        data: [
          {
            model_group: "gpt-5",
            max_input_tokens: 100,
            supported_openai_params: ["reasoning_effort", "extra_headers"],
            api_key: "secret",
          },
        ],
      }),
    ).toEqual({
      data: [
        {
          model_group: "gpt-5",
          max_input_tokens: 100,
          supported_openai_params: ["reasoning_effort"],
        },
      ],
    });
  });
});

describe("assertSafeFixture", () => {
  it("rejects credential fields and non-placeholder api_base values", () => {
    expect(() => assertSafeFixture({ api_key: "secret" })).toThrow("Unsafe fixture field: api_key");
    expect(() => assertSafeFixture({ api_base: "https://private.example" })).toThrow("Unsafe fixture api_base");
    expect(() => assertSafeFixture({ api_base: "https://example.invalid" })).not.toThrow();
  });

  it("guards every committed proxy fixture and reproduces the acceptance oracle", async () => {
    const fixtureDir = join(process.cwd(), "tests/fixtures/proxy");
    const names = [
      "dev-model-info-2026-09-04.json",
      "expected-acceptance-2026-09-04.json",
      "probe-results-2026-09-04.json",
      "prod-model-group-info-2026-09-04.json",
      "prod-model-info-2026-09-04.json",
    ];
    const fixtures = new Map(
      await Promise.all(
        names.map(async (name) => [name, JSON.parse(await readFile(join(fixtureDir, name), "utf8"))] as const),
      ),
    );
    for (const fixture of fixtures.values()) expect(() => assertSafeFixture(fixture)).not.toThrow();
    expect(deriveAcceptanceOracle(fixtures.get("probe-results-2026-09-04.json"))).toEqual(
      fixtures.get("expected-acceptance-2026-09-04.json"),
    );
  });
});

describe("deriveAcceptanceOracle", () => {
  it("classifies successes and recorded UnsupportedParamsError rejections", () => {
    expect(
      deriveAcceptanceOracle([
        { path: "chat", model: "gpt-5", level: "low", status: 200 },
        {
          path: "chat",
          model: "gpt-5",
          level: "xhigh",
          status: 400,
          error: '{"message":"litellm.UnsupportedParamsError: unsupported"}',
        },
        { path: "chat", model: "gpt-5", level: "high", status: 500, errorClass: "InternalServerError" },
      ]),
    ).toEqual([
      { path: "chat", model: "gpt-5", level: "low", status: 200, accepted: true },
      {
        path: "chat",
        model: "gpt-5",
        level: "xhigh",
        status: 400,
        errorClass: "UnsupportedParamsError",
        accepted: false,
      },
      {
        path: "chat",
        model: "gpt-5",
        level: "high",
        status: 500,
        errorClass: "InternalServerError",
        accepted: null,
      },
    ]);
  });
});
