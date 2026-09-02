import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createPi, loadExtension } from "./test-helpers.js";

vi.unmock("@earendil-works/pi-coding-agent");

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.resetModules();
  delete process.env.LITELLM_BASE_URL;
  delete process.env.LITELLM_API_KEY;
  delete process.env.LITELLM_VERBOSE_DISCOVERY;
});

async function startHook() {
  const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-skills-"));
  process.env.LITELLM_BASE_URL = "https://proxy.example.com";
  process.env.LITELLM_API_KEY = "sk-test";
  const pi = createPi();
  await (await loadExtension(agentDir))(pi);
  return pi.handlers.get("before_agent_start")?.[0];
}

describe("before_agent_start skills hook", () => {
  // An expired LiteLLM SSO credential makes Pi's getProviderAuth throw. The hook must not
  // turn that into a per-turn `Extension "..." error:` report.
  it("survives an unrefreshable OAuth credential", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, []));
    const beforeAgentStart = await startHook();

    await expect(
      beforeAgentStart?.(
        { systemPrompt: "Base prompt" },
        {
          modelRegistry: {
            getProviderAuth: async () => {
              throw new Error("OAuth refresh failed for litellm: LiteLLM credential cannot be refreshed");
            },
            getProvider: () => undefined,
          },
        },
      ),
    ).resolves.toBeUndefined();
  });

  it("survives a failing skills catalog request", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("connect ECONNREFUSED"));
    const beforeAgentStart = await startHook();

    await expect(
      beforeAgentStart?.(
        { systemPrompt: "Base prompt" },
        {
          modelRegistry: {
            getProviderAuth: async () => ({
              auth: { apiKey: "sk-test" },
              env: { LITELLM_BASE_URL: "https://skills.example.com/v1" },
            }),
            getProvider: () => undefined,
          },
        },
      ),
    ).resolves.toBeUndefined();
  });

  it("rejects placeholder runtime hosts before sending credentials", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, []));
    const beforeAgentStart = await startHook();
    vi.mocked(globalThis.fetch).mockClear();

    await expect(
      beforeAgentStart?.(
        { systemPrompt: "Base prompt" },
        {
          modelRegistry: {
            getProviderAuth: async () => ({
              auth: { apiKey: "sk-test" },
              env: { LITELLM_BASE_URL: "https://litellm.example.com/v1" },
            }),
            getProvider: () => undefined,
          },
        },
      ),
    ).resolves.toBeUndefined();
    expect(globalThis.fetch).not.toHaveBeenCalled();
  });

  it("reports the reason on stderr under LITELLM_VERBOSE_DISCOVERY", async () => {
    process.env.LITELLM_VERBOSE_DISCOVERY = "1";
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, []));
    const stderr = vi.spyOn(process.stderr, "write").mockReturnValue(true);
    const beforeAgentStart = await startHook();

    await beforeAgentStart?.(
      { systemPrompt: "Base prompt" },
      {
        modelRegistry: {
          getProviderAuth: async () => {
            throw new Error("LiteLLM credential cannot be refreshed");
          },
          getProvider: () => undefined,
        },
      },
    );

    expect(stderr.mock.calls.map(([line]) => String(line)).join("")).toContain("cannot be refreshed");
  });
});
