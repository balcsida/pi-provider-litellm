import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { runAuthSmoke, runAuthSmokeFromEnv } from "../scripts/smoke-auth.js";
import { fingerprint } from "../src/cache.js";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.resetModules();
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

  it("checks virtual-key auth, enterprise admin-route enforcement, and SSO login", async () => {
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
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "vidaimock-openai", model_info: { mode: "chat" } }] });
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
        "sso-login",
        "sso-virtual-key-chat",
      ],
    });
    expect(requests.filter((request) => request.url.endsWith("/key/generate"))).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          auth: "Bearer sk-master",
          body: { models: ["vidaimock-openai"], duration: "1h" },
        }),
        expect.objectContaining({
          auth: "Bearer sk-virtual",
          body: { models: ["vidaimock-openai"], duration: "1h" },
        }),
      ]),
    );
  });
});

describe("runSsoLoginSmoke", () => {
  it("drives the extension SSO login callback path and validates the generated key", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-smoke-auth-"));
    const requests: Array<{ url: string; body?: unknown; auth?: string }> = [];

    vi.doMock("@earendil-works/pi-coding-agent", () => ({
      AuthStorage: {
        create: () => ({ getApiKey: async () => undefined }),
      },
      defineTool: (tool: unknown) => tool,
      getAgentDir: () => agentDir,
    }));

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      const headers = init?.headers as Record<string, string> | undefined;
      const auth = headers?.Authorization;
      requests.push({
        url,
        auth,
        body: init?.body ? JSON.parse(String(init.body)) : undefined,
      });

      if (url.endsWith("/model/info")) {
        return jsonResponse(200, { data: [{ model_name: "vidaimock-openai", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/key/generate")) {
        expect(auth).toBe("Bearer sk-master");
        return jsonResponse(200, { key: "sk-sso-virtual" });
      }
      if (url.endsWith("/v1/chat/completions")) {
        expect(auth).toBe("Bearer sk-sso-virtual");
        return jsonResponse(200, { choices: [{ message: { content: "pong" } }] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const { runSsoLoginSmoke } = await import("../scripts/smoke-auth.js");
    await runSsoLoginSmoke({
      baseUrl: "http://127.0.0.1:4000",
      masterKey: "sk-master",
      modelId: "vidaimock-openai",
      timeoutMs: 1000,
    });

    expect(requests).toContainEqual(
      expect.objectContaining({
        url: "http://127.0.0.1:4000/key/generate",
        auth: "Bearer sk-master",
        body: {},
      }),
    );
    expect(requests).toContainEqual(
      expect.objectContaining({
        url: "http://127.0.0.1:4000/v1/chat/completions",
        auth: "Bearer sk-sso-virtual",
      }),
    );
  });

  it("uses a valid model cache without waiting for a background refresh", async () => {
    const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-smoke-auth-"));
    await writeFile(
      join(agentDir, "litellm-models.json"),
      JSON.stringify({
        baseUrl: "http://127.0.0.1:4000",
        apiKeyFingerprint: fingerprint("sk-master"),
        fetchedAt: Date.now(),
        source: "model_info",
        models: [{ id: "vidaimock-openai", name: "vidaimock-openai", provider: "litellm" }],
      }),
      "utf8",
    );

    vi.doMock("@earendil-works/pi-coding-agent", () => ({
      AuthStorage: {
        create: () => ({ getApiKey: async () => undefined }),
      },
      defineTool: (tool: unknown) => tool,
      getAgentDir: () => agentDir,
    }));

    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const url = String(input);
      const auth = (init?.headers as Record<string, string> | undefined)?.Authorization;
      if (url.endsWith("/key/generate")) {
        expect(auth).toBe("Bearer sk-master");
        return jsonResponse(200, { key: "sk-sso-virtual" });
      }
      if (url.endsWith("/model/info")) {
        expect(auth).toBe("Bearer sk-sso-virtual");
        return jsonResponse(200, { data: [{ model_name: "vidaimock-openai", model_info: { mode: "chat" } }] });
      }
      if (url.endsWith("/v1/chat/completions")) {
        expect(auth).toBe("Bearer sk-sso-virtual");
        return jsonResponse(200, { choices: [{ message: { content: "pong" } }] });
      }
      throw new Error(`unexpected URL: ${url}`);
    });

    const { runSsoLoginSmoke } = await import("../scripts/smoke-auth.js");
    await runSsoLoginSmoke({
      baseUrl: "http://127.0.0.1:4000",
      masterKey: "sk-master",
      modelId: "vidaimock-openai",
      timeoutMs: 50,
    });
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
