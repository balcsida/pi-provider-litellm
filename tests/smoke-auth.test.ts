import { afterEach, describe, expect, it, vi } from "vitest";
import { runAuthSmoke, runAuthSmokeFromEnv } from "../scripts/smoke-auth.js";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("runAuthSmoke", () => {
  it("checks missing, bad, and master-key auth without enterprise checks", async () => {
    const requests: Array<{ url: string; body?: unknown; auth?: string }> = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      const headers = init?.headers as Record<string, string> | undefined;
      const auth = headers?.Authorization;
      requests.push({
        url,
        auth,
        body: init?.body ? JSON.parse(String(init.body)) : undefined,
      });

      if (url.endsWith("/v1/models")) {
        if (!auth) return jsonResponse(401, { error: "missing token" });
        if (auth === "Bearer bad-smoke-key") return jsonResponse(403, { error: "bad token" });
        return jsonResponse(200, { data: [{ id: "vidaimock-openai" }] });
      }
      if (url.endsWith("/v1/chat/completions")) {
        return jsonResponse(200, { choices: [{ message: { content: "pong" } }] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await runAuthSmoke({
      baseUrl: "http://127.0.0.1:4000/v1",
      masterKey: "sk-master",
      modelId: "vidaimock-openai",
      timeoutMs: 1000,
      enterprise: false,
    });

    expect(result).toEqual({
      enterprise: false,
      checks: ["missing-token", "bad-token", "master-key-models", "master-key-chat"],
    });
    expect(requests.map((request) => request.url)).toEqual([
      "http://127.0.0.1:4000/v1/models",
      "http://127.0.0.1:4000/v1/models",
      "http://127.0.0.1:4000/v1/models",
      "http://127.0.0.1:4000/v1/chat/completions",
    ]);
  });

  it("checks virtual-key auth and enterprise admin-route enforcement", async () => {
    const requests: Array<{ url: string; body?: unknown; auth?: string }> = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      const headers = init?.headers as Record<string, string> | undefined;
      const auth = headers?.Authorization;
      requests.push({
        url,
        auth,
        body: init?.body ? JSON.parse(String(init.body)) : undefined,
      });

      if (url.endsWith("/v1/models")) {
        if (!auth) return jsonResponse(401, { error: "missing token" });
        if (auth === "Bearer bad-smoke-key") return jsonResponse(403, { error: "bad token" });
        return jsonResponse(200, { data: [{ id: "vidaimock-openai" }] });
      }
      if (url.endsWith("/key/generate") && auth === "Bearer sk-master") {
        return jsonResponse(200, { key: "sk-virtual" });
      }
      if (url.endsWith("/key/generate") && auth === "Bearer sk-virtual") {
        return jsonResponse(403, { error: "admin only" });
      }
      if (url.endsWith("/v1/chat/completions")) {
        return jsonResponse(200, { choices: [{ message: { content: "pong" } }] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await runAuthSmoke({
      baseUrl: "http://127.0.0.1:4000",
      masterKey: "sk-master",
      modelId: "vidaimock-openai",
      timeoutMs: 1000,
      enterprise: true,
    });

    expect(result).toEqual({
      enterprise: true,
      checks: [
        "missing-token",
        "bad-token",
        "master-key-models",
        "master-key-chat",
        "virtual-key-chat",
        "enterprise-admin-route",
      ],
    });
    expect(requests.filter((request) => request.url.endsWith("/key/generate"))).toMatchObject([
      {
        auth: "Bearer sk-master",
        body: { models: ["vidaimock-openai"], duration: "1h" },
      },
      {
        auth: "Bearer sk-virtual",
        body: { models: ["vidaimock-openai"], duration: "1h" },
      },
    ]);
  });
});

describe("runAuthSmokeFromEnv", () => {
  it("loads auth smoke settings from the environment", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      const headers = init?.headers as Record<string, string> | undefined;
      const auth = headers?.Authorization;
      if (url.endsWith("/v1/models")) {
        if (!auth) return jsonResponse(401, { error: "missing token" });
        if (auth === "Bearer bad-smoke-key") return jsonResponse(403, { error: "bad token" });
        return jsonResponse(200, { data: [{ id: "vidaimock-openai" }] });
      }
      if (url.endsWith("/v1/chat/completions")) {
        return jsonResponse(200, { choices: [{ message: { content: "pong" } }] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const result = await runAuthSmokeFromEnv({
      LITELLM_BASE_URL: " http://127.0.0.1:4000/v1 ",
      LITELLM_API_KEY: " sk-master ",
      LITELLM_CLI_SMOKE_MODEL: " vidaimock-openai ",
      LITELLM_SMOKE_TIMEOUT_MS: "1000",
    });

    expect(result.enterprise).toBe(false);
    expect(result.checks).toContain("master-key-chat");
  });
});
