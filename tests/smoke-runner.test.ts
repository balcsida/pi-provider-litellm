import { afterEach, describe, expect, it, vi } from "vitest";
import {
  parseExpectedApis,
  parseExpectedResponseCost,
  parseSmokeModels,
  runSmoke,
  runSmokeFromEnv,
  smokeChatCompletion,
} from "../scripts/smoke-runner.js";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe("parseSmokeModels", () => {
  it("parses comma and whitespace separated model ids", () => {
    expect(parseSmokeModels(" github-gpt-4.1-mini,openai-gpt-5.4-nano\nanthropic-claude-haiku ")).toEqual([
      "github-gpt-4.1-mini",
      "openai-gpt-5.4-nano",
      "anthropic-claude-haiku",
    ]);
  });

  it("returns an empty list for undefined or separator-only input", () => {
    expect(parseSmokeModels(undefined)).toEqual([]);
    expect(parseSmokeModels(" \n ,, \t ")).toEqual([]);
  });
});

describe("smoke expectations", () => {
  it("parses per-model API and response-cost expectations", () => {
    expect(parseExpectedApis("claude=anthropic-messages chat=openai-completions")).toEqual(
      new Map([
        ["claude", "anthropic-messages"],
        ["chat", "openai-completions"],
      ]),
    );
    expect(parseExpectedResponseCost("claude=present chat=absent")).toEqual(
      new Map([
        ["claude", true],
        ["chat", false],
      ]),
    );
  });
});

describe("smokeChatCompletion", () => {
  it("is reusable by the auth smoke", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { choices: [{ message: { content: "pong" } }] }));

    await expect(smokeChatCompletion("http://127.0.0.1:4000", "sk-smoke", "vidaimock-openai", 1000)).resolves.toEqual({
      modelId: "vidaimock-openai",
      api: "openai-completions",
      endpoint: "/v1/chat/completions",
      content: "pong",
      hasResponseCost: false,
    });
  });
});

describe("runSmoke", () => {
  it("uses each discovered model's route-distinct endpoint and records response-cost availability", async () => {
    const requests: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      requests.push(url);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            {
              model_name: "claude-route",
              model_info: { id: "claude", mode: "chat", litellm_provider: "anthropic" },
              litellm_params: { model: "anthropic/claude-sonnet-4-6" },
            },
            { model_name: "chat-route", model_info: { id: "chat", mode: "chat" } },
            { model_name: "responses-route", model_info: { id: "responses", mode: "responses" } },
          ],
        });
      }
      if (url.endsWith("/v1/messages")) {
        return new Response(JSON.stringify({ content: [{ type: "text", text: "messages" }] }), {
          headers: { "content-type": "application/json", "x-litellm-response-cost": "0.01" },
        });
      }
      if (url.endsWith("/v1/chat/completions")) {
        return jsonResponse(200, { choices: [{ message: { content: "chat" } }] });
      }
      if (url.endsWith("/v1/responses")) return jsonResponse(200, { output_text: "responses" });
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await runSmoke({
      baseUrl: "http://127.0.0.1:4000",
      apiKey: "sk-smoke",
      modelIds: ["claude-route", "chat-route", "responses-route"],
      timeoutMs: 1000,
      expectedApis: new Map([
        ["claude-route", "anthropic-messages"],
        ["chat-route", "openai-completions"],
        ["responses-route", "openai-responses"],
      ]),
      expectedResponseCost: new Map([
        ["claude-route", true],
        ["chat-route", false],
        ["responses-route", false],
      ]),
    });

    expect(result.completions).toEqual([
      {
        modelId: "claude-route",
        api: "anthropic-messages",
        endpoint: "/v1/messages",
        content: "messages",
        hasResponseCost: true,
      },
      {
        modelId: "chat-route",
        api: "openai-completions",
        endpoint: "/v1/chat/completions",
        content: "chat",
        hasResponseCost: false,
      },
      {
        modelId: "responses-route",
        api: "openai-responses",
        endpoint: "/v1/responses",
        content: "responses",
        hasResponseCost: false,
      },
    ]);
    expect(requests.slice(1)).toEqual([
      "http://127.0.0.1:4000/v1/messages",
      "http://127.0.0.1:4000/v1/chat/completions",
      "http://127.0.0.1:4000/v1/responses",
    ]);
  });

  it("reports missing endpoint coverage when the explicit completeness check is enabled", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "chat-route", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/v1/chat/completions")) {
        return jsonResponse(200, { choices: [{ message: { content: "chat" } }] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    await expect(
      runSmoke({
        baseUrl: "http://127.0.0.1:4000",
        apiKey: "sk-smoke",
        modelIds: ["chat-route"],
        timeoutMs: 1000,
        expectedApis: new Map([["chat-route", "openai-completions"]]),
        requireAllProtocols: true,
      }),
    ).rejects.toThrow(
      /LITELLM_SMOKE_REQUIRE_ALL_PROTOCOLS is enabled, but smoke endpoint coverage is missing: \/v1\/messages, \/v1\/responses/,
    );
  });

  it("allows expected APIs without implicitly requiring every endpoint", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "chat-route", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/v1/chat/completions")) {
        return jsonResponse(200, { choices: [{ message: { content: "chat" } }] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    await expect(
      runSmoke({
        baseUrl: "http://127.0.0.1:4000",
        apiKey: "sk-smoke",
        modelIds: ["chat-route"],
        timeoutMs: 1000,
        expectedApis: new Map([["chat-route", "openai-completions"]]),
      }),
    ).resolves.toMatchObject({ completions: [{ endpoint: "/v1/chat/completions" }] });
  });

  it("rejects an expected API mismatch before calling completion endpoints", async () => {
    const requests: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      requests.push(url);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "claude-route", model_info: { mode: "chat" } }] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    await expect(
      runSmoke({
        baseUrl: "http://127.0.0.1:4000",
        apiKey: "sk-smoke",
        modelIds: ["claude-route"],
        timeoutMs: 1000,
        expectedApis: new Map([["claude-route", "anthropic-messages"]]),
      }),
    ).rejects.toThrow(/API mismatch for claude-route/);
    expect(requests).toEqual(["http://127.0.0.1:4000/model/info"]);
  });

  it("rejects a response-cost expectation mismatch", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "chat-route", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/v1/chat/completions")) {
        return jsonResponse(200, { choices: [{ message: { content: "ok" } }] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    await expect(
      runSmoke({
        baseUrl: "http://127.0.0.1:4000",
        apiKey: "sk-smoke",
        modelIds: ["chat-route"],
        timeoutMs: 1000,
        expectedResponseCost: new Map([["chat-route", true]]),
      }),
    ).rejects.toThrow(/Response-cost header mismatch for chat-route: expected present, got absent/);
  });

  it("rejects a configured expectation map that omits a requested model", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "chat-route", model_info: { mode: "chat" } }] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    await expect(
      runSmoke({
        baseUrl: "http://127.0.0.1:4000",
        apiKey: "sk-smoke",
        modelIds: ["chat-route"],
        timeoutMs: 1000,
        expectedApis: new Map([["other-route", "openai-completions"]]),
      }),
    ).rejects.toThrow(/No expected API configured for smoke model chat-route/);
  });

  it.each([
    ["missing separator", "vidaimock-openai"],
    ["leading separator", "=openai-completions"],
    ["unknown value", "vidaimock-openai=openai-chat"],
    ["empty value", "vidaimock-openai="],
  ])("rejects a malformed expected-API entry (%s)", (_label, raw) => {
    expect(() => parseExpectedApis(raw)).toThrow(/LITELLM_SMOKE_EXPECT_APIS entries must use model=value/);
  });

  it("rejects a malformed response-cost entry", () => {
    expect(() => parseExpectedResponseCost("vidaimock-openai=maybe")).toThrow(
      /LITELLM_SMOKE_EXPECT_RESPONSE_COST entries must use model=value/,
    );
  });

  it("discovers models and sends a chat completion request to each requested model", async () => {
    const requests: Array<{ url: string; body?: unknown; headers?: Record<string, string> }> = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      requests.push({
        url,
        body: init?.body ? JSON.parse(String(init.body)) : undefined,
        headers: init?.headers as Record<string, string>,
      });
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            { model_name: "github-gpt-4.1-mini", model_info: { mode: "chat" } },
            { model_name: "gemini-flash", model_info: { mode: "chat" } },
          ],
        });
      }
      if (url.endsWith("/v1/chat/completions")) {
        return jsonResponse(200, {
          choices: [{ message: { content: "pong" } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await runSmoke({
      baseUrl: "http://127.0.0.1:4000/v1",
      apiKey: "sk-smoke",
      modelIds: ["github-gpt-4.1-mini", "gemini-flash"],
      timeoutMs: 1000,
    });

    expect(result).toEqual({
      source: "model_info",
      discoveredCount: 2,
      completions: [
        {
          modelId: "github-gpt-4.1-mini",
          api: "openai-completions",
          endpoint: "/v1/chat/completions",
          content: "pong",
          hasResponseCost: false,
        },
        {
          modelId: "gemini-flash",
          api: "openai-completions",
          endpoint: "/v1/chat/completions",
          content: "pong",
          hasResponseCost: false,
        },
      ],
    });
    expect(requests.filter((request) => request.url.endsWith("/v1/chat/completions"))).toMatchObject([
      {
        url: "http://127.0.0.1:4000/v1/chat/completions",
        body: {
          model: "github-gpt-4.1-mini",
          messages: [{ role: "user", content: "Reply with one short word." }],
          max_tokens: 16,
          temperature: 0,
        },
        headers: { Authorization: "Bearer sk-smoke" },
      },
      {
        url: "http://127.0.0.1:4000/v1/chat/completions",
        body: {
          model: "gemini-flash",
          messages: [{ role: "user", content: "Reply with one short word." }],
          max_tokens: 16,
          temperature: 0,
        },
      },
    ]);
  });

  it("fails before completion calls when a requested model is not discovered", async () => {
    const requestedUrls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      requestedUrls.push(url);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [{ model_name: "github-gpt-4.1-mini", model_info: { mode: "chat" } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    await expect(
      runSmoke({
        baseUrl: "http://127.0.0.1:4000",
        apiKey: "sk-smoke",
        modelIds: ["anthropic-claude-haiku"],
        timeoutMs: 1000,
      }),
    ).rejects.toThrow(/Requested smoke models were not discovered: anthropic-claude-haiku/);
    expect(requestedUrls).toEqual(["http://127.0.0.1:4000/model/info"]);
  });

  it("fails without any network calls when no smoke models are configured", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("unexpected fetch"));

    await expect(
      runSmoke({
        baseUrl: "http://127.0.0.1:4000",
        apiKey: "sk-smoke",
        modelIds: [],
        timeoutMs: 1000,
      }),
    ).rejects.toThrow(/At least one smoke model must be configured in LITELLM_SMOKE_MODELS/);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("fails when a completion returns no assistant text", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [{ model_name: "github-models-openai", model_info: { mode: "chat" } }],
        });
      }
      if (url.endsWith("/v1/chat/completions")) {
        return jsonResponse(200, { choices: [{ message: { content: "   " } }] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    await expect(
      runSmoke({
        baseUrl: "http://127.0.0.1:4000",
        apiKey: "sk-smoke",
        modelIds: ["github-models-openai"],
        timeoutMs: 1000,
      }),
    ).rejects.toThrow(/\/v1\/chat\/completions for github-models-openai returned no assistant text/);
  });

  it("aborts a completion that exceeds the configured timeout", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return Promise.resolve(
          jsonResponse(200, {
            data: [{ model_name: "github-models-openai", model_info: { mode: "chat" } }],
          }),
        );
      }
      return new Promise<Response>((_resolve, reject) => {
        const signal = init?.signal;
        signal?.addEventListener("abort", () => reject(signal.reason), { once: true });
      });
    });

    await expect(
      runSmoke({
        baseUrl: "http://127.0.0.1:4000",
        apiKey: "sk-smoke",
        modelIds: ["github-models-openai"],
        timeoutMs: 25,
      }),
    ).rejects.toMatchObject({ name: "TimeoutError" });
  });

  it("fails before completion calls when discovery uses an unexpected source", async () => {
    const requestedUrls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      requestedUrls.push(url);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [{ model_name: "github-models-openai", model_info: { mode: "chat" } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    await expect(
      runSmoke({
        baseUrl: "http://127.0.0.1:4000",
        apiKey: "sk-smoke",
        modelIds: ["github-models-openai"],
        timeoutMs: 1000,
        expectedSource: "models_list",
      }),
    ).rejects.toThrow(/Discovery source mismatch: expected models_list, got model_info/);
    expect(requestedUrls).toEqual(["http://127.0.0.1:4000/model/info"]);
  });

  it("does not expose provider response bodies in completion failures", async () => {
    const echoedSecret = "sk-provider-secret-echo";
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [{ model_name: "github-models-openai", model_info: { mode: "chat" } }],
        });
      }
      if (url.endsWith("/v1/chat/completions")) {
        return jsonResponse(429, { error: `request rejected for ${echoedSecret}` });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    await expect(
      runSmoke({
        baseUrl: "http://127.0.0.1:4000",
        apiKey: "sk-smoke",
        modelIds: ["github-models-openai"],
        timeoutMs: 1000,
      }),
    ).rejects.toSatisfy((error: Error) => {
      expect(error.message).toBe("/v1/chat/completions for github-models-openai returned 429");
      expect(error.message).not.toContain(echoedSecret);
      return true;
    });
  });
});

describe("runSmokeFromEnv", () => {
  it("loads LiteLLM smoke settings from the environment", async () => {
    const requests: Array<{ url: string; headers?: Record<string, string> }> = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      requests.push({
        url,
        headers: init?.headers as Record<string, string>,
      });
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [{ model_name: "github-models-openai", model_info: { mode: "chat" } }],
        });
      }
      if (url.endsWith("/v1/chat/completions")) {
        return jsonResponse(200, {
          choices: [{ message: { content: "pong" } }],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await runSmokeFromEnv({
      LITELLM_BASE_URL: " http://127.0.0.1:4000/v1 ",
      LITELLM_API_KEY: " sk-env ",
      LITELLM_SMOKE_MODELS: "github-models-openai",
      LITELLM_SMOKE_TIMEOUT_MS: "1000",
    });

    expect(result.completions).toEqual([
      {
        modelId: "github-models-openai",
        api: "openai-completions",
        endpoint: "/v1/chat/completions",
        content: "pong",
        hasResponseCost: false,
      },
    ]);
    expect(requests[0]).toMatchObject({
      url: "http://127.0.0.1:4000/model/info",
      headers: { Authorization: "Bearer sk-env" },
    });
  });

  it("accepts a matching LITELLM_SMOKE_EXPECT_SOURCE", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [{ model_name: "github-models-openai", model_info: { mode: "chat" } }],
        });
      }
      if (url.endsWith("/v1/chat/completions")) {
        return jsonResponse(200, { choices: [{ message: { content: "pong" } }] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await runSmokeFromEnv({
      LITELLM_BASE_URL: "http://127.0.0.1:4000",
      LITELLM_API_KEY: "sk-env",
      LITELLM_SMOKE_MODELS: "github-models-openai",
      LITELLM_SMOKE_TIMEOUT_MS: "1000",
      LITELLM_SMOKE_EXPECT_SOURCE: "model_info",
    });

    expect(result.source).toBe("model_info");
  });

  it("rejects an invalid LITELLM_SMOKE_EXPECT_SOURCE without any network calls", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("unexpected fetch"));

    await expect(
      runSmokeFromEnv({
        LITELLM_BASE_URL: "http://127.0.0.1:4000",
        LITELLM_API_KEY: "sk-env",
        LITELLM_SMOKE_MODELS: "github-models-openai",
        LITELLM_SMOKE_EXPECT_SOURCE: "bogus",
      }),
    ).rejects.toThrow(/LITELLM_SMOKE_EXPECT_SOURCE must be one of model_info, models_list, health/);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("requires LiteLLM base URL and API key settings", async () => {
    await expect(runSmokeFromEnv({ LITELLM_BASE_URL: "http://127.0.0.1:4000" })).rejects.toThrow(
      /LITELLM_BASE_URL and LITELLM_API_KEY must be set/,
    );
  });

  it("requires at least one model in LITELLM_SMOKE_MODELS", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("unexpected fetch"));

    await expect(
      runSmokeFromEnv({
        LITELLM_BASE_URL: "http://127.0.0.1:4000",
        LITELLM_API_KEY: "sk-env",
      }),
    ).rejects.toThrow(/At least one smoke model must be configured in LITELLM_SMOKE_MODELS/);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("falls back to the default timeout when LITELLM_SMOKE_TIMEOUT_MS is invalid", async () => {
    const timeoutSpy = vi.spyOn(AbortSignal, "timeout");
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [{ model_name: "github-models-openai", model_info: { mode: "chat" } }],
        });
      }
      return jsonResponse(200, { choices: [{ message: { content: "pong" } }] });
    });

    await runSmokeFromEnv({
      LITELLM_BASE_URL: "http://127.0.0.1:4000",
      LITELLM_API_KEY: "sk-env",
      LITELLM_SMOKE_MODELS: "github-models-openai",
      LITELLM_SMOKE_TIMEOUT_MS: "not-a-number",
    });
    expect(timeoutSpy).toHaveBeenCalledWith(30_000);
  });
});
