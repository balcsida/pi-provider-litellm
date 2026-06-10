import { afterEach, describe, expect, it, vi } from "vitest";
import { parseSmokeModels, runSmoke, runSmokeFromEnv } from "../scripts/smoke-runner.js";

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

describe("runSmoke", () => {
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
        { modelId: "github-gpt-4.1-mini", content: "pong" },
        { modelId: "gemini-flash", content: "pong" },
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
    ).rejects.toThrow(/Timed out after 25ms/);
  });

  it("truncates oversized provider error bodies in failures", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [{ model_name: "github-models-openai", model_info: { mode: "chat" } }],
        });
      }
      if (url.endsWith("/v1/chat/completions")) {
        return new Response("x".repeat(600), { status: 500 });
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
    ).rejects.toThrow(/returned 500: x{500}$/);
  });

  it("includes provider response bodies in chat completion failures", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [{ model_name: "github-models-openai", model_info: { mode: "chat" } }],
        });
      }
      if (url.endsWith("/v1/chat/completions")) {
        return jsonResponse(429, { error: "rate limited" });
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
    ).rejects.toThrow(/\/v1\/chat\/completions for github-models-openai returned 429.*rate limited/);
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

    expect(result.completions).toEqual([{ modelId: "github-models-openai", content: "pong" }]);
    expect(requests[0]).toMatchObject({
      url: "http://127.0.0.1:4000/model/info",
      headers: { Authorization: "Bearer sk-env" },
    });
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
    vi.useFakeTimers();
    vi.spyOn(globalThis, "fetch").mockImplementation(
      (_input, init) =>
        new Promise<Response>((_resolve, reject) => {
          const signal = init?.signal;
          signal?.addEventListener("abort", () => reject(signal.reason), { once: true });
        }),
    );

    const smoke = runSmokeFromEnv({
      LITELLM_BASE_URL: "http://127.0.0.1:4000",
      LITELLM_API_KEY: "sk-env",
      LITELLM_SMOKE_MODELS: "github-models-openai",
      LITELLM_SMOKE_TIMEOUT_MS: "not-a-number",
    });
    const rejection = expect(smoke).rejects.toThrow(/Timed out after 30000ms/);
    await vi.advanceTimersByTimeAsync(30_000);
    await rejection;
  });
});
