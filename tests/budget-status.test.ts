import { mkdtemp, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ModelRegistry, ModelRuntime } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { type BudgetOptions, formatBudgetDetails, setupLiteLLMBudget } from "../src/budget.js";
import { AUTH, fixture, mockProxy, ok, status } from "./budget-helpers.js";
import { createPi, loadExtension, type TestPi, useHermeticEnv } from "./test-helpers.js";

vi.unmock("@earendil-works/pi-coding-agent");

useHermeticEnv();

const PLAIN = { fg: (_color: string, text: string) => text };
const KEY = "litellm-budget";

const KEY_INFO = () => ok(fixture("key-info-personal"));
const USER_INFO = () => ok(fixture("user-info-v2"));
const PERSONAL = { "/key/info": KEY_INFO, "/v2/user/info": USER_INFO };
const TEAM_KEY = () => ok(fixture("key-info-team-user"));
const TEAM_INFO = () => ok(fixture("team-info"));

function makeCtx({ provider = "litellm", hasUI = true }: { provider?: string; hasUI?: boolean } = {}) {
  return {
    hasUI,
    model: provider && { provider, id: "m" },
    ui: { setStatus: vi.fn(), notify: vi.fn(), theme: PLAIN },
  } as any;
}

async function emit(pi: TestPi, event: string, payload: unknown, ctx: unknown): Promise<void> {
  for (const handler of pi.handlers.get(event) ?? []) await handler(payload, ctx);
}

const SECOND = { baseUrl: "https://team.example.com", apiKey: "sk-team" };

let pi: TestPi;
let auths: Record<string, typeof AUTH | typeof SECOND | undefined>;
let providers: BudgetOptions["providers"];
let overrides: Partial<BudgetOptions>;
let resolveAuth: ReturnType<typeof vi.fn>;

function setup(): void {
  resolveAuth = vi.fn(async (_ctx: unknown, name: string) => auths[name]);
  setupLiteLLMBudget(
    pi as never,
    {
      providers,
      display: "all",
      resolveAuth,
      timeoutMs: () => 5000,
      disabledReason: () => null,
      hostOffline: () => false,
      missingCredentials: (name) => `no credentials for ${name}.`,
      ...overrides,
    } as BudgetOptions,
  );
}

function command(args: string, ctx: unknown): Promise<void> | void {
  return pi.commands.get("litellm-budget")!.handler(args, ctx as never);
}

const texts = (ctx: any) => ctx.ui.setStatus.mock.calls.filter(([, text]: [string, unknown]) => text !== undefined);

beforeEach(() => {
  pi = createPi();
  auths = { litellm: AUTH };
  providers = [{ name: "litellm", displayName: "LiteLLM" }];
  overrides = {};
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("budget footer", () => {
  it("shows the active provider's budgets after session start", async () => {
    mockProxy(PERSONAL);
    setup();
    const ctx = makeCtx();
    await emit(pi, "session_start", { type: "session_start" }, ctx);
    await vi.waitFor(() =>
      expect(ctx.ui.setStatus).toHaveBeenLastCalledWith(KEY, "LiteLLM key $0.15/$10 · user $0.34/$100"),
    );
  });

  it("does nothing for another provider's model", async () => {
    const seen = mockProxy(PERSONAL);
    setup();
    const ctx = makeCtx({ provider: "openai" });
    await emit(pi, "session_start", {}, ctx);
    await emit(pi, "turn_end", {}, ctx);
    await emit(
      pi,
      "after_provider_response",
      { headers: { "x-litellm-key-spend": "1", "x-litellm-key-max-budget": "2" } },
      ctx,
    );
    expect(seen).toEqual([]);
    expect(ctx.ui.setStatus).not.toHaveBeenCalled();
  });

  it("does nothing without UI", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"] });
    const seen = mockProxy(PERSONAL);
    setup();
    const ctx = makeCtx({ hasUI: false });
    await emit(pi, "session_start", {}, ctx);
    await emit(pi, "turn_end", {}, ctx);
    await vi.advanceTimersByTimeAsync(120_000);
    expect(seen).toEqual([]);
    expect(ctx.ui.setStatus).not.toHaveBeenCalled();
  });

  it.each([
    ["LITELLM_OFFLINE=1", false],
    ["LITELLM_DISCOVERY_TIMEOUT_MS=0", false],
    [null, true],
  ])("does not poll automatically under %s (host offline %s), but applies headers", async (reason, host) => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"] });
    const seen = mockProxy(PERSONAL);
    overrides = { disabledReason: () => reason, hostOffline: () => host };
    setup();
    const ctx = makeCtx();
    await emit(pi, "session_start", {}, ctx);
    await emit(pi, "turn_end", {}, ctx);
    await vi.advanceTimersByTimeAsync(120_000);
    expect(seen).toEqual([]);
    await emit(
      pi,
      "after_provider_response",
      { headers: { "x-litellm-key-spend": "0.144", "x-litellm-key-max-budget": "10.0" } },
      ctx,
    );
    expect(ctx.ui.setStatus).toHaveBeenLastCalledWith(KEY, "LiteLLM key $0.14/$10");
  });

  it("polls 15 s after a turn and at most once a minute", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"] });
    const seen = mockProxy(PERSONAL);
    setup();
    const ctx = makeCtx();
    const keyPolls = () => seen.filter((path) => path === "/key/info").length;
    await emit(pi, "session_start", {}, ctx);
    await vi.advanceTimersByTimeAsync(0);
    expect(keyPolls()).toBe(1);

    await vi.advanceTimersByTimeAsync(1000);
    await emit(pi, "turn_end", {}, ctx);
    await vi.advanceTimersByTimeAsync(58_000);
    expect(keyPolls()).toBe(1);
    await vi.advanceTimersByTimeAsync(1000);
    expect(keyPolls()).toBe(2);

    await vi.advanceTimersByTimeAsync(140_000);
    await emit(pi, "turn_end", {}, ctx);
    await vi.advanceTimersByTimeAsync(14_000);
    expect(keyPolls()).toBe(2);
    await vi.advanceTimersByTimeAsync(1000);
    expect(keyPolls()).toBe(3);
  });

  it("polls again after the last turn of a run that ends within the settle delay of a poll", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"] });
    const seen = mockProxy(PERSONAL);
    setup();
    const ctx = makeCtx();
    const keyPolls = () => seen.filter((path) => path === "/key/info").length;
    await emit(pi, "session_start", {}, ctx);
    await vi.advanceTimersByTimeAsync(0);
    expect(keyPolls()).toBe(1);

    await vi.advanceTimersByTimeAsync(50_000);
    await emit(pi, "turn_end", {}, ctx);
    await vi.advanceTimersByTimeAsync(5000);
    await emit(pi, "turn_end", {}, ctx);
    await vi.advanceTimersByTimeAsync(9999);
    expect(keyPolls()).toBe(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(keyPolls()).toBe(2);

    await vi.advanceTimersByTimeAsync(59_999);
    expect(keyPolls()).toBe(2);
    await vi.advanceTimersByTimeAsync(1);
    expect(keyPolls()).toBe(3);

    await vi.advanceTimersByTimeAsync(175_000);
    expect(keyPolls()).toBe(3);
  });

  it("does not poll again when the timer fires a few milliseconds early", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"] });
    const seen = mockProxy(PERSONAL);
    setup();
    const ctx = makeCtx();
    const keyPolls = () => seen.filter((path) => path === "/key/info").length;
    await emit(pi, "session_start", {}, ctx);
    await vi.advanceTimersByTimeAsync(100_000);
    await emit(pi, "turn_end", {}, ctx);
    // Timers keep their own schedule: lag the wall clock so the timer fires before Date.now() reaches its target.
    vi.setSystemTime(Date.now() - 5);
    await vi.advanceTimersByTimeAsync(15_000);
    expect(keyPolls()).toBe(2);
    await vi.advanceTimersByTimeAsync(120_000);
    expect(keyPolls()).toBe(2);
  });

  it("shows a provider's last result on model_select and polls only when stale", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"] });
    providers = [
      { name: "litellm", displayName: "LiteLLM" },
      { name: "team", displayName: "Team GW" },
    ];
    auths = { litellm: AUTH, team: { ...AUTH, baseUrl: "https://team.example.com" } };
    const hosts: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = new URL(String(input));
      hosts.push(url.host + url.pathname);
      if (url.pathname === "/key/info") return ok(fixture("key-info-personal"));
      return new Response("{}", { status: 404 });
    });
    setup();
    const ctx = makeCtx();
    await emit(pi, "session_start", {}, ctx);
    await vi.advanceTimersByTimeAsync(0);
    expect(hosts.filter((h) => h === "proxy.example.com/key/info")).toHaveLength(1);

    await emit(pi, "model_select", { model: { provider: "team", id: "m" } }, ctx);
    await vi.advanceTimersByTimeAsync(0);
    expect(hosts.filter((h) => h === "team.example.com/key/info")).toHaveLength(1);

    const before = hosts.length;
    await emit(pi, "model_select", { model: { provider: "litellm", id: "m" } }, ctx);
    await vi.advanceTimersByTimeAsync(0);
    expect(hosts).toHaveLength(before);
    expect(ctx.ui.setStatus).toHaveBeenLastCalledWith(KEY, "LiteLLM key $0.15/$10");

    await emit(pi, "model_select", { model: { provider: "openai", id: "m" } }, ctx);
    expect(ctx.ui.setStatus).toHaveBeenLastCalledWith(KEY, undefined);
  });

  it("drops a pending turn poll when another provider is selected", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"] });
    providers = [
      { name: "litellm", displayName: "LiteLLM" },
      { name: "team", displayName: "Team GW" },
    ];
    auths = { litellm: AUTH, team: { ...AUTH, baseUrl: "https://team.example.com" } };
    const keyPolls: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = new URL(String(input));
      if (url.pathname !== "/key/info") return new Response("{}", { status: 404 });
      keyPolls.push(url.host);
      return ok(fixture("key-info-personal"));
    });
    setup();
    const ctx = makeCtx();
    await emit(pi, "session_start", {}, ctx);
    await vi.advanceTimersByTimeAsync(1000);
    await emit(pi, "turn_end", {}, ctx);
    await vi.advanceTimersByTimeAsync(54_000);
    // The turn's poll was due at +60 s for litellm; selecting team polls team now instead.
    await emit(pi, "model_select", { model: { provider: "team", id: "m" } }, ctx);
    await vi.advanceTimersByTimeAsync(120_000);
    expect(keyPolls).toEqual(["proxy.example.com", "team.example.com"]);
  });

  it("keeps each provider's budgets under its own name", async () => {
    providers = [
      { name: "litellm", displayName: "LiteLLM" },
      { name: "team", displayName: "Team GW" },
    ];
    auths = { litellm: AUTH, team: { ...AUTH, baseUrl: "https://team.example.com" } };
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = new URL(String(input));
      if (url.pathname !== "/key/info") return new Response("{}", { status: 404 });
      const body = fixture("key-info-personal");
      if (url.host === "team.example.com") Object.assign(body.info, { spend: 7, max_budget: 20 });
      return ok(body);
    });
    setup();
    const ctx = makeCtx();
    await emit(pi, "session_start", {}, ctx);
    await vi.waitFor(() => expect(ctx.ui.setStatus).toHaveBeenLastCalledWith(KEY, "LiteLLM key $0.15/$10"));
    await emit(pi, "model_select", { model: { provider: "team", id: "m" } }, ctx);
    await vi.waitFor(() => expect(ctx.ui.setStatus).toHaveBeenLastCalledWith(KEY, "Team GW key $7/$20"));
    for (const [, text] of texts(ctx)) {
      expect(text.startsWith("LiteLLM") ? text : "").not.toContain("$20");
      expect(text.startsWith("Team GW") ? text : "").not.toContain("$10");
    }
  });

  it("updates the key figure from response headers without a request", async () => {
    const seen = mockProxy(PERSONAL);
    setup();
    const ctx = makeCtx();
    await emit(pi, "session_start", {}, ctx);
    await vi.waitFor(() => expect(texts(ctx)).toHaveLength(1));
    const count = seen.length;
    await emit(pi, "after_provider_response", { headers: { "x-litellm-key-spend": "0.5" } }, ctx);
    expect(ctx.ui.setStatus).toHaveBeenLastCalledWith(KEY, "LiteLLM key $0.50/$10 · user $0.34/$100");
    expect(seen).toHaveLength(count);
  });

  it("drops levels and 4xx memory when the credential changes", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    const routes = { "/key/info": TEAM_KEY, "/v2/user/info": USER_INFO };
    mockProxy({ ...routes, "/team/info?team_id=T": () => status(403) }, { token: "sk-a", gateway: "g1" });
    auths = { litellm: { ...AUTH, apiKey: "sk-a" } };
    setup();
    const ctx = makeCtx();
    const select = { model: { provider: "litellm", id: "m" } };
    await emit(pi, "session_start", {}, ctx);
    await vi.waitFor(() => expect(ctx.ui.setStatus).toHaveBeenLastCalledWith(KEY, "LiteLLM user $0.34/$100"));

    // Key B can read the team, which key A's remembered 403 would have hidden.
    const seen = mockProxy({ ...routes, "/team/info?team_id=T": TEAM_INFO }, { token: "sk-b", gateway: "g1" });
    auths.litellm = { ...AUTH, apiKey: "sk-b" };
    vi.setSystemTime(Date.now() + 61_000);
    await emit(pi, "model_select", select, ctx);
    await vi.waitFor(() => expect(seen).toContain("/team/info?team_id=T"));
    await vi.waitFor(() => expect(texts(ctx)).toHaveLength(2));
    expect(ctx.ui.setStatus.mock.calls.map(([, text]: [string, unknown]) => text)).toEqual([
      "LiteLLM user $0.34/$100",
      undefined,
      "LiteLLM user $0.34/$100 · team $0.29/$1k · member $0.15/$50",
    ]);

    auths.litellm = undefined;
    vi.setSystemTime(Date.now() + 61_000);
    await emit(pi, "model_select", select, ctx);
    await vi.waitFor(() => expect(ctx.ui.setStatus).toHaveBeenLastCalledWith(KEY, undefined));
  });

  it("treats changed custom headers as a new credential", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    const routes = { "/key/info": TEAM_KEY, "/v2/user/info": USER_INFO };
    mockProxy({ ...routes, "/team/info?team_id=T": () => status(403) });
    setup();
    const ctx = makeCtx();
    await emit(pi, "session_start", {}, ctx);
    await vi.waitFor(() => expect(ctx.ui.setStatus).toHaveBeenLastCalledWith(KEY, "LiteLLM user $0.34/$100"));

    // Same key, new gateway token in the custom headers: the old token's 403 must not hide the team.
    const seen = mockProxy({ ...routes, "/team/info?team_id=T": TEAM_INFO }, { token: "sk-test", gateway: "g2" });
    auths.litellm = { ...AUTH, headers: { "x-gateway": "g2" } };
    vi.setSystemTime(Date.now() + 61_000);
    await emit(pi, "model_select", { model: { provider: "litellm", id: "m" } }, ctx);
    await vi.waitFor(() => expect(seen).toContain("/team/info?team_id=T"));
  });

  it("never shows proxy-supplied text", async () => {
    const keyBody = fixture("key-info-team-user");
    keyBody.info.organization_id = "O";
    mockProxy({
      "/key/info": () => ok(keyBody),
      "/v2/user/info": USER_INFO,
      "/team/info?team_id=T": TEAM_INFO,
      "/organization/list": () => status(403),
    });
    setup();
    const ctx = makeCtx();
    await emit(pi, "session_start", {}, ctx);
    await vi.waitFor(() => expect(texts(ctx)).not.toHaveLength(0));
    await command("", ctx);
    const output = [
      ...texts(ctx).map(([, text]: [string, string]) => text),
      ...ctx.ui.notify.mock.calls.map(([m]: [string]) => m),
    ].join("\n");
    expect(output).toContain("org (403)");
    for (const bad of ["\u001b", "proxy text", "T\u001b", "secret detail", "@example.com"])
      expect(output).not.toContain(bad);
  });

  it("clears the timer and status on session_shutdown and ignores late results", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout", "Date"] });
    let release: (response: Response) => void = () => {};
    const held = new Promise<Response>((resolve) => {
      release = resolve;
    });
    const seen: string[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = new URL(String(input));
      seen.push(url.pathname);
      return url.pathname === "/key/info" ? held : new Response("{}", { status: 404 });
    });
    setup();
    const ctx = makeCtx();
    await emit(pi, "session_start", {}, ctx);
    await vi.advanceTimersByTimeAsync(0);
    await emit(pi, "turn_end", {}, ctx);
    await emit(pi, "session_shutdown", {}, ctx);
    release(ok(fixture("key-info-personal")));
    await vi.advanceTimersByTimeAsync(120_000);
    expect(texts(ctx)).toEqual([]);
    expect(seen.filter((p) => p === "/key/info")).toHaveLength(1);
  });

  it("clears a shown footer on session_shutdown", async () => {
    mockProxy(PERSONAL);
    setup();
    const ctx = makeCtx();
    await emit(pi, "session_start", {}, ctx);
    await vi.waitFor(() => expect(texts(ctx)).toHaveLength(1));
    await emit(pi, "session_shutdown", {}, ctx);
    expect(ctx.ui.setStatus).toHaveBeenLastCalledWith(KEY, undefined);
  });

  it("hides the footer when no level has a limit", async () => {
    const team = fixture("team-info");
    Object.assign(team.team_info, { max_budget: null, team_member_budget_table: null });
    team.team_memberships = [];
    mockProxy({
      "/key/info": TEAM_KEY,
      "/v2/user/info": () => ok({ ...fixture("user-info-v2"), max_budget: null }),
      "/team/info?team_id=T": () => ok(team),
    });
    setup();
    const ctx = makeCtx();
    await emit(pi, "session_start", {}, ctx);
    await vi.waitFor(() => expect(resolveAuth).toHaveBeenCalled());
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(texts(ctx)).toEqual([]);
  });

  it("warns once about an unknown display setting", async () => {
    overrides = { displayWarning: 'LiteLLM budget: unknown display "x"; using "all".' };
    setup();
    const ctx = makeCtx({ provider: "openai" });
    await emit(pi, "session_start", {}, ctx);
    await emit(pi, "session_start", {}, ctx);
    expect(ctx.ui.notify.mock.calls).toEqual([['LiteLLM budget: unknown display "x"; using "all".', "warning"]]);

    const write = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    pi = createPi();
    setup();
    await emit(pi, "session_start", {}, makeCtx({ hasUI: false }));
    expect(write).toHaveBeenCalledWith('LiteLLM budget: unknown display "x"; using "all".\n');
  });
});

describe("/litellm-budget", () => {
  it("registers /litellm-budget, which polls now, forgets 4xx endpoints, and shows the breakdown", async () => {
    let teamStatus = 403;
    const seen = mockProxy({
      "/key/info": TEAM_KEY,
      "/v2/user/info": USER_INFO,
      "/team/info?team_id=T": () => (teamStatus === 403 ? status(403) : TEAM_INFO()),
    });
    setup();
    const ctx = makeCtx();
    await command("", ctx);
    expect(ctx.ui.notify.mock.calls[0][0]).toContain("Not readable with this credential: team (403)");

    teamStatus = 200;
    const count = seen.filter((p) => p.startsWith("/team/info")).length;
    await command("litellm", ctx);
    expect(seen.filter((p) => p.startsWith("/team/info"))).toHaveLength(count + 1);
    const [message, level] = ctx.ui.notify.mock.calls[1];
    expect(level).toBe("info");
    expect(message).toBe(
      formatBudgetDetails(
        "litellm",
        {
          user: { spend: 0.343, maxBudget: 100, resetAt: Date.parse("2026-11-01T00:00:00Z") },
          team: { spend: 0.294, maxBudget: 1000, resetAt: Date.parse("2026-11-01T00:00:00Z") },
          member: { spend: 0.147, maxBudget: 50 },
        },
        new Map(),
        Date.now(),
      ),
    );
  });

  it("forgets 4xx endpoints and polls again instead of joining an in-flight poll", async () => {
    const seen = mockProxy({
      "/key/info": TEAM_KEY,
      "/v2/user/info": USER_INFO,
      "/team/info?team_id=T": () => status(403),
    });
    setup();
    const ctx = makeCtx();
    await command("", ctx);
    const inner = (globalThis.fetch as any).getMockImplementation();
    let release!: () => void;
    let reached = false;
    const held = new Promise<void>((resolve) => {
      release = resolve;
    });
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      if (new URL(String(input)).pathname === "/key/info" && !reached) {
        reached = true;
        await held;
      }
      return inner(input, init);
    });
    const before = seen.length;
    await emit(pi, "session_start", {}, ctx);
    await vi.waitFor(() => expect(reached).toBe(true));
    const pending = command("", ctx);
    release();
    await pending;
    expect(seen.slice(before).filter((p) => p.startsWith("/key/info") || p.startsWith("/team/info"))).toEqual([
      "/key/info",
      "/key/info",
      "/team/info?team_id=T",
    ]);
  });

  it("reports unknown providers, missing credentials, and failed polls", async () => {
    setup();
    const ctx = makeCtx();
    await command(" nope ", ctx);
    auths = {};
    await command("", ctx);
    auths = { litellm: AUTH };
    mockProxy({ "/key/info": () => status(503) });
    await command("", ctx);
    expect(ctx.ui.notify.mock.calls).toEqual([
      ['LiteLLM: unknown provider "nope"; configured: "litellm".', "warning"],
      ['LiteLLM ("litellm"): budget check skipped; no credentials for litellm.', "warning"],
      ['LiteLLM ("litellm"): budget check failed (HTTP 503).', "warning"],
    ]);
  });

  it("keeps a header-only key level through polls that cannot read /key/info", async () => {
    mockProxy({ "/key/info": () => status(403), "/v2/user/info": USER_INFO });
    setup();
    const ctx = makeCtx();
    await emit(pi, "session_start", {}, ctx);
    await vi.waitFor(() => expect(ctx.ui.setStatus).toHaveBeenLastCalledWith(KEY, "LiteLLM user $0.34/$100"));
    await emit(
      pi,
      "after_provider_response",
      { headers: { "x-litellm-key-spend": "0.5", "x-litellm-key-max-budget": "10" } },
      ctx,
    );
    const both = "LiteLLM key $0.50/$10 · user $0.34/$100";
    expect(ctx.ui.setStatus).toHaveBeenLastCalledWith(KEY, both);
    await command("", ctx);
    expect(ctx.ui.setStatus).toHaveBeenLastCalledWith(KEY, both);
    const [message] = ctx.ui.notify.mock.calls[0];
    expect(message).toContain("  key     $0.50 of $10 (5%)");
    expect(message).toContain("  Not readable with this credential: key (403)");
  });

  it("treats a throwing credential lookup as missing credentials", async () => {
    setup();
    resolveAuth.mockRejectedValue(new Error("placeholder root"));
    const ctx = makeCtx();
    await command("", ctx);
    expect(ctx.ui.notify).toHaveBeenCalledWith(
      'LiteLLM ("litellm"): budget check skipped; no credentials for litellm.',
      "warning",
    );
  });

  it("skips the command under LITELLM_OFFLINE or a zero timeout but not under PI_OFFLINE", async () => {
    const seen = mockProxy(PERSONAL);
    overrides = { disabledReason: () => "LITELLM_OFFLINE=1" };
    setup();
    const ctx = makeCtx();
    await command("", ctx);
    expect(ctx.ui.notify.mock.calls).toEqual([["LiteLLM: budget check skipped (LITELLM_OFFLINE=1).", "warning"]]);
    expect(seen).toEqual([]);

    pi = createPi();
    overrides = { hostOffline: () => true };
    setup();
    await command("", ctx);
    expect(seen).toContain("/key/info");
  });

  it("writes the breakdown to stderr without UI", async () => {
    mockProxy(PERSONAL);
    setup();
    const write = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const ctx = makeCtx({ hasUI: false });
    await command("", ctx);
    const message = String(write.mock.calls[0]?.[0]);
    expect(message.startsWith('LiteLLM ("litellm") budget\n')).toBe(true);
    expect(message.endsWith("\n")).toBe(true);
    expect(ctx.ui.notify).not.toHaveBeenCalled();
  });

  it("completes provider names", () => {
    setup();
    const complete = pi.commands.get("litellm-budget") as any;
    expect(complete.getArgumentCompletions("li")).toEqual([{ value: "litellm", label: "litellm" }]);
    expect(complete.getArgumentCompletions("x")).toBeNull();
  });
});

describe("through the extension", () => {
  async function agentDir(settings?: unknown): Promise<string> {
    const dir = await mkdtemp(join(tmpdir(), "pi-litellm-budget-"));
    if (settings) await writeFile(join(dir, "settings.json"), JSON.stringify(settings));
    return dir;
  }

  it("registers budget hooks after cost tracking, and nothing when budget.enabled is false", async () => {
    const loaded = createPi();
    await (await loadExtension(await agentDir()))(loaded);
    expect(loaded.commands.has("litellm-budget")).toBe(true);
    expect(loaded.handlers.get("turn_end")).toHaveLength(1);
    expect(loaded.handlers.get("model_select")).toHaveLength(1);
    expect(loaded.handlers.get("after_provider_response")).toHaveLength(3);
    const costCtx = {
      model: { provider: "litellm" },
      hasUI: true,
      ui: { setStatus: vi.fn(), notify: vi.fn(), theme: PLAIN },
    };
    await loaded.handlers.get("after_provider_response")![0]!(
      { headers: { "x-litellm-response-cost": "0.5" } },
      costCtx,
    );
    const message = { role: "assistant", provider: "litellm", usage: { input: 1, output: 1, cost: { total: 0 } } };
    const result = await loaded.handlers.get("message_end")![0]!({ message });
    expect(result.message.usage.cost.total).toBe(0.5);

    const disabled = createPi();
    await (await loadExtension(await agentDir({ litellm: { budget: { enabled: false } } })))(disabled);
    expect(disabled.commands.has("litellm-budget")).toBe(false);
    expect(disabled.handlers.has("turn_end")).toBe(false);
    expect(disabled.handlers.has("model_select")).toBe(false);
  });

  it("polls through Pi's provider auth with the configured headers", async () => {
    process.env.LITELLM_BASE_URL = "https://proxy.example.com";
    process.env.LITELLM_API_KEY = "env-key";
    process.env.LITELLM_HEADERS = JSON.stringify({ "x-gateway": "g1" });
    const dir = await agentDir();
    const runtime = await ModelRuntime.create({
      authPath: join(dir, "auth.json"),
      modelsPath: join(dir, "models.json"),
    });
    const loaded = createPi();
    loaded.registerProvider = (provider) => runtime.registerNativeProvider(provider);
    await (await loadExtension(dir))(loaded);
    await runtime.refresh({ allowNetwork: false });
    const seen = mockProxy({ "/key/info": KEY_INFO, "/v2/user/info": USER_INFO }, { token: "env-key", gateway: "g1" });
    const notify = vi.fn();
    await loaded.commands.get("litellm-budget")!.handler("", {
      hasUI: true,
      ui: { notify },
      modelRegistry: new ModelRegistry(runtime),
    } as never);
    expect(seen).toContain("/key/info");
    expect(notify.mock.calls[0]?.[1]).toBe("info");
  });
});
