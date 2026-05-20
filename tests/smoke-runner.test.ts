import { describe, expect, it, vi, afterEach } from "vitest";
import { parseSmokeModels, runSmoke } from "../scripts/smoke-runner.js";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("parseSmokeModels", () => {
  it("parses comma and whitespace separated model ids", () => {
    expect(parseSmokeModels(" github-gpt-4.1-mini,openai-gpt-5.4-nano\nanthropic-claude-haiku ")).toEqual([
      "github-gpt-4.1-mini",
      "openai-gpt-5.4-nano",
      "anthropic-claude-haiku",
    ]);
  });
});

describe("runSmoke", () => {
  it("discovers models and sends a chat completion request to each requested model", async () => {
    const requests: Array<{ url: string; body?: unknown }> = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      requests.push({
        url,
        body: init?.body ? JSON.parse(String(init.body)) : undefined,
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
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
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
  });
});
