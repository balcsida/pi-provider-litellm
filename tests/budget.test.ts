import { afterEach, describe, expect, it, vi } from "vitest";
import {
  type BudgetEndpoint,
  type BudgetLevels,
  type BudgetTheme,
  budgetDisplaySetting,
  formatBudgetDetails,
  formatBudgetStatus,
  formatCompactAmount,
  formatRelativeReset,
  mergeKeyHeaders,
  pollBudget,
} from "../src/budget.js";
import { AUTH, fixture, mockProxy, ok, RESET, status } from "./budget-helpers.js";

afterEach(() => vi.restoreAllMocks());

const keyWith = (name: string, info: Record<string, unknown>) => {
  const base = fixture(name);
  return { ...base, info: { ...base.info, ...info } };
};
const poll = (denied = new Map<BudgetEndpoint, number>(), previous: BudgetLevels = {}) =>
  pollBudget(AUTH, denied, previous, 5000);

describe("pollBudget", () => {
  it("reads key, user, team and member budgets for a key with a user and a team", async () => {
    const seen = mockProxy({
      "/key/info": () => ok(fixture("key-info-team-user")),
      "/v2/user/info": () => ok(fixture("user-info-v2")),
      "/team/info?team_id=T": () => ok(fixture("team-info")),
    });
    const denied = new Map<BudgetEndpoint, number>();
    expect(await poll(denied)).toEqual({
      ok: true,
      keyPolled: true,
      levels: {
        user: { spend: 0.343, maxBudget: 100, resetAt: RESET },
        team: { spend: 0.294, maxBudget: 1000, resetAt: RESET },
        member: { spend: 0.147, maxBudget: 50 },
      },
    });
    expect(seen.sort()).toEqual(["/key/info", "/team/info?team_id=T", "/v2/user/info"]);
    expect(denied.size).toBe(0);
  });

  it("reads the key budget and its reset time", async () => {
    const seen = mockProxy({
      "/key/info": () => ok(fixture("key-info-personal")),
      "/v2/user/info": () => ok(fixture("user-info-v2")),
    });
    const result = await poll();
    expect(result).toEqual({
      ok: true,
      keyPolled: true,
      levels: {
        key: { spend: 0.147, maxBudget: 10, resetAt: Date.parse("2026-10-08T00:00:00+00:00") },
        user: { spend: 0.343, maxBudget: 100, resetAt: RESET },
      },
    });
    expect(seen.sort()).toEqual(["/key/info", "/v2/user/info"]);
  });

  it("uses the team default member budget when the caller has no own entry", async () => {
    for (const userId of ["U9", "U2"]) {
      mockProxy({
        "/key/info": () => ok(keyWith("key-info-team-user", { user_id: userId })),
        "/v2/user/info": () => ok(fixture("user-info-v2")),
        "/team/info?team_id=T": () => ok(fixture("team-info")),
      });
      const result = await poll();
      expect(result.ok && result.levels.member).toEqual({ spend: 0, maxBudget: 25, resetAt: RESET });
      vi.restoreAllMocks();
    }
  });

  it("skips user and member for a team key without a user", async () => {
    const seen = mockProxy({
      "/key/info": () => ok(fixture("key-info-team-only")),
      "/team/info?team_id=T": () => ok(fixture("team-info")),
    });
    const result = await poll();
    expect(seen.sort()).toEqual(["/key/info", "/team/info?team_id=T"]);
    expect(result).toEqual({
      ok: true,
      keyPolled: true,
      levels: { team: { spend: 0.294, maxBudget: 1000, resetAt: RESET } },
    });
  });

  it("falls back to /user/info when /v2/user/info answers 404 and remembers it", async () => {
    const seen = mockProxy({
      "/key/info": () => ok(keyWith("key-info-team-user", { team_id: null })),
      "/v2/user/info": () => status(404),
      "/user/info": () => ok(fixture("user-info-v1")),
    });
    const denied = new Map<BudgetEndpoint, number>();
    const expected = { user: { spend: 0.343, maxBudget: 100, resetAt: RESET } };
    expect(await poll(denied)).toEqual({ ok: true, keyPolled: true, levels: expected });
    expect(denied.get("userV2")).toBe(404);
    seen.length = 0;
    expect(await poll(denied)).toEqual({ ok: true, keyPolled: true, levels: expected });
    expect(seen).toContain("/user/info");
    expect(seen).not.toContain("/v2/user/info");
  });

  it("does not fall back to /user/info on other 4xx", async () => {
    const seen = mockProxy({
      "/key/info": () => ok(keyWith("key-info-team-user", { team_id: null })),
      "/v2/user/info": () => status(403),
    });
    const denied = new Map<BudgetEndpoint, number>();
    await poll(denied);
    expect(seen).not.toContain("/user/info");
    expect(denied.get("userV2")).toBe(403);
  });

  it("remembers 4xx endpoints and keeps 429 and 5xx retryable", async () => {
    const team = { spend: 0.294, maxBudget: 1000, resetAt: RESET };
    const member = { spend: 0.147, maxBudget: 50 };
    const routes = (reply: () => Response) => ({
      "/key/info": () => ok(fixture("key-info-team-user")),
      "/v2/user/info": () => ok(fixture("user-info-v2")),
      "/team/info?team_id=T": reply,
    });
    const seen = mockProxy(routes(() => status(403)));
    const denied = new Map<BudgetEndpoint, number>();
    await poll(denied);
    expect(denied.get("team")).toBe(403);
    seen.length = 0;
    await poll(denied);
    expect(seen).not.toContain("/team/info?team_id=T");
    vi.restoreAllMocks();

    for (const code of [503, 429]) {
      mockProxy(routes(() => status(code)));
      const retry = new Map<BudgetEndpoint, number>();
      const result = await poll(retry, { team, member });
      expect(result.ok && result.levels).toMatchObject({ team, member });
      expect(retry.size).toBe(0);
      vi.restoreAllMocks();
    }
  });

  it("still tries the user when /key/info answers 4xx, and skips team and org", async () => {
    const seen = mockProxy({
      "/key/info": () => status(403),
      "/v2/user/info": () => ok(fixture("user-info-v2")),
    });
    const denied = new Map<BudgetEndpoint, number>();
    const result = await poll(denied);
    expect(seen.sort()).toEqual(["/key/info", "/v2/user/info"]);
    expect(result).toMatchObject({ ok: true, keyPolled: false });
    expect(denied.get("key")).toBe(403);
  });

  it("ends the poll on a transient /key/info failure", async () => {
    const cases: Array<[() => Response | Error, string]> = [
      [() => status(503), "HTTP 503"],
      [() => new Error("offline"), "request failed"],
      [
        () => new Response("<html>login</html>", { status: 200, headers: { "content-type": "text/html" } }),
        "request failed",
      ],
      [() => ok({}), "malformed response"],
    ];
    for (const [reply, reason] of cases) {
      const seen = mockProxy({ "/key/info": reply });
      expect(await poll()).toEqual({ ok: false, reason });
      expect(seen).toEqual(["/key/info"]);
      vi.restoreAllMocks();
    }
  });

  it("reads the org budget for the key's org, else the team's org", async () => {
    const org = { spend: 9210, maxBudget: 50000 };
    const team = fixture("team-info");
    const withOrg = { ...team, team_info: { ...team.team_info, organization_id: "O" } };
    mockProxy({
      "/key/info": () => ok(keyWith("key-info-team-user", { organization_id: "O", team_id: null })),
      "/v2/user/info": () => ok(fixture("user-info-v2")),
      "/organization/list": () => ok(fixture("organization-list")),
    });
    let result = await poll();
    expect(result.ok && result.levels.org).toEqual(org);
    vi.restoreAllMocks();

    mockProxy({
      "/key/info": () => ok(fixture("key-info-team-user")),
      "/v2/user/info": () => ok(fixture("user-info-v2")),
      "/team/info?team_id=T": () => ok(withOrg),
      "/organization/list": () => ok(fixture("organization-list")),
    });
    result = await poll();
    expect(result.ok && result.levels.org).toEqual(org);
    vi.restoreAllMocks();

    mockProxy({
      "/key/info": () => ok(keyWith("key-info-team-user", { organization_id: "O", team_id: null })),
      "/v2/user/info": () => ok(fixture("user-info-v2")),
      "/organization/list": () => ok(fixture("organization-list").slice(0, 1)),
    });
    result = await poll();
    expect(result.ok && result.levels.org).toBeUndefined();
    vi.restoreAllMocks();

    const seen = mockProxy({
      "/key/info": () => ok(fixture("key-info-team-only")),
      "/team/info?team_id=T": () => ok(withOrg),
    });
    await poll();
    expect(seen).not.toContain("/organization/list");
  });

  it("treats missing, null, zero and non-numeric limits as unset", async () => {
    for (const max_budget of [0, "10", null]) {
      mockProxy({ "/key/info": () => ok(keyWith("key-info-personal", { max_budget, user_id: null })) });
      const result = await poll();
      expect(result.ok && result.levels.key).toBeUndefined();
      vi.restoreAllMocks();
    }
    mockProxy({ "/key/info": () => ok(keyWith("key-info-personal", { spend: undefined, user_id: null })) });
    const result = await poll();
    expect(result.ok && result.levels.key).toMatchObject({ spend: 0, maxBudget: 10 });
  });

  it("encodes the team id", async () => {
    const seen = mockProxy({ "/key/info": () => ok(keyWith("key-info-team-only", { team_id: "a b/c" })) });
    await poll();
    expect(seen).toContain("/team/info?team_id=a%20b%2Fc");
  });

  it("sends nothing to a non-loopback http root unless insecure http is allowed", async () => {
    const seen = mockProxy({ "/key/info": () => ok(fixture("key-info-team-only")) });
    const http = { ...AUTH, baseUrl: "http://proxy.example.com" };
    expect(await pollBudget(http, new Map(), {}, 5000)).toEqual({ ok: false, reason: "request failed" });
    expect(seen).toEqual([]);
    expect(await pollBudget({ ...http, allowInsecureHttp: true }, new Map(), {}, 5000)).toMatchObject({ ok: true });
    expect(seen).toContain("/key/info");
  });

  it("drops a /v1 suffix from the proxy root", async () => {
    const seen = mockProxy({ "/key/info": () => ok(fixture("key-info-personal")) });
    await pollBudget({ ...AUTH, baseUrl: "https://proxy.example.com/v1/" }, new Map(), {}, 5000);
    expect(seen).toContain("/key/info");
    expect(seen).not.toContain("/v1/key/info");
  });
});

const PLAIN: BudgetTheme = { fg: (_c, t) => t };
const MARK: BudgetTheme = { fg: (c, t) => `<${c}>${t}</${c}>` };
const NOW = Date.parse("2026-10-07T12:00:00Z");
const DAY = 86_400_000;

describe("mergeKeyHeaders", () => {
  const key = { spend: 5, maxBudget: 10, resetAt: 1 };
  it("raises a polled key spend but never lowers it", () => {
    expect(mergeKeyHeaders(key, true, { "x-litellm-key-spend": "6", "x-litellm-key-max-budget": "10.0" })).toEqual({
      spend: 6,
      maxBudget: 10,
      resetAt: 1,
    });
    expect(mergeKeyHeaders(key, true, { "x-litellm-key-spend": "0.0" })).toEqual(key);
  });
  it("takes a numeric header limit and ignores a missing, empty, or non-numeric one", () => {
    expect(
      mergeKeyHeaders(key, true, { "x-litellm-key-spend": "5", "x-litellm-key-max-budget": "12" })?.maxBudget,
    ).toBe(12);
    for (const limit of [undefined, "", "None"]) {
      const headers: Record<string, string> = { "x-litellm-key-spend": "5" };
      if (limit !== undefined) headers["x-litellm-key-max-budget"] = limit;
      expect(mergeKeyHeaders(key, true, headers)?.maxBudget).toBe(10);
    }
  });
  it("lets headers alone set the key until a poll succeeds", () => {
    expect(
      mergeKeyHeaders(undefined, false, { "x-litellm-key-spend": "0.144", "x-litellm-key-max-budget": "10.0" }),
    ).toEqual({ spend: 0.144, maxBudget: 10 });
    expect(mergeKeyHeaders(undefined, false, { "x-litellm-key-spend": "0.144" })).toBeUndefined();
  });
  it("creates a key level when a polled key gains a limit", () => {
    expect(mergeKeyHeaders(undefined, true, { "x-litellm-key-spend": "1", "x-litellm-key-max-budget": "10" })).toEqual({
      spend: 1,
      maxBudget: 10,
    });
  });
  it("ignores responses without a key-spend header", () => {
    expect(mergeKeyHeaders(key, true, { "x-litellm-key-max-budget": "99" })).toBe(key);
  });
});

describe("budget formatting", () => {
  it("formats compact amounts", () => {
    expect([3.1, 40, 0.042, 412.3, 1000, 9210, 50000, 1_200_000].map(formatCompactAmount)).toEqual([
      "$3.10",
      "$40",
      "$0.04",
      "$412",
      "$1k",
      "$9.2k",
      "$50k",
      "$1.2M",
    ]);
  });
  it("formats relative reset times", () => {
    expect(formatRelativeReset(NOW + 12.5 * DAY, NOW)).toBe("12d");
    expect(formatRelativeReset(NOW + 5.5 * 3_600_000, NOW)).toBe("5h");
    expect(formatRelativeReset(NOW + 40.5 * 60_000, NOW)).toBe("40m");
    expect(formatRelativeReset(NOW + 30_000, NOW)).toBe("1m");
    expect(formatRelativeReset(NOW - 1, NOW)).toBeUndefined();
    expect(formatRelativeReset(undefined, NOW)).toBeUndefined();
  });
  it("shows every set level in order", () => {
    const levels = {
      org: { spend: 9210, maxBudget: 50000 },
      key: { spend: 3.1, maxBudget: 10 },
      member: { spend: 20, maxBudget: 50 },
      user: { spend: 40, maxBudget: 100 },
      team: { spend: 412.3, maxBudget: 1000 },
    };
    expect(formatBudgetStatus(levels, "all", "LiteLLM", PLAIN, NOW)).toBe(
      "LiteLLM key $3.10/$10 · user $40/$100 · team $412/$1k · member $20/$50 · org $9.2k/$50k",
    );
  });
  it("shows only the level with the least money left in tightest mode", () => {
    const levels = {
      user: { spend: 40, maxBudget: 1000 },
      team: { spend: 412.3, maxBudget: 1000, resetAt: NOW + 12.5 * DAY },
    };
    expect(formatBudgetStatus(levels, "tightest", "LiteLLM", PLAIN, NOW)).toBe(
      "LiteLLM team $412/$1k (41%) · resets 12d",
    );
  });
  it("colours segments by use and picks an exceeded level as tightest", () => {
    const one = (spend: number) => formatBudgetStatus({ key: { spend, maxBudget: 10 } }, "all", "P", MARK, NOW);
    expect(one(7.9)).toBe("<dim>P</dim> <dim>key $7.90/$10</dim>");
    expect(one(8)).toContain("<warning>key $8/$10</warning>");
    expect(one(10.2)).toContain("<error>key $10.20/$10</error>");
    const levels = { key: { spend: 10.2, maxBudget: 10 }, team: { spend: 1, maxBudget: 1000 } };
    expect(formatBudgetStatus(levels, "tightest", "P", MARK, NOW)).toBe(
      "<dim>P</dim> <error>key $10.20/$10 (102%)</error>",
    );
    expect(formatBudgetStatus({ ...levels, user: { spend: 0, maxBudget: 5 } }, "all", "P", MARK, NOW)).toContain(
      "</error><dim> · </dim><dim>user",
    );
  });
  it("breaks tightest ties in level order", () => {
    const levels = { key: { spend: 5, maxBudget: 10 }, team: { spend: 995, maxBudget: 1000 } };
    expect(formatBudgetStatus(levels, "tightest", "P", PLAIN, NOW)).toMatch(/^P key /);
  });
  it("returns undefined when no level is set", () => {
    expect(formatBudgetStatus({}, "all", "P", PLAIN, NOW)).toBeUndefined();
  });
});

describe("formatBudgetDetails", () => {
  it("formats the command breakdown", () => {
    const levels = {
      key: { spend: 3.1, maxBudget: 10, resetAt: NOW + 12.5 * 3_600_000 },
      user: { spend: 40, maxBudget: 100, resetAt: NOW + 24.5 * DAY },
      team: { spend: 412.3, maxBudget: 1000, resetAt: NOW + 24.5 * DAY },
      member: { spend: 20, maxBudget: 50 },
    };
    expect(formatBudgetDetails("litellm", levels, new Map([["org", 401]]), NOW)).toBe(
      [
        'LiteLLM ("litellm") budget',
        "  key     $3.10 of $10 (31%), resets in 12h",
        "  user    $40.00 of $100 (40%), resets in 24d",
        "  team    $412.30 of $1,000 (41%), resets in 24d",
        "  member  $20.00 of $50 (40%)",
        "  Not readable with this credential: org (401)",
      ].join("\n"),
    );
  });
  it("lists unreadable user endpoints once, and not a successful fallback", () => {
    const text = (denied: [BudgetEndpoint, number][]) => formatBudgetDetails("p", {}, new Map(denied), NOW);
    expect(text([["userV2", 403]])).toContain("user (403)");
    expect(
      text([
        ["userV2", 404],
        ["user", 404],
      ]),
    ).toContain("user (404)");
    expect(text([["userV2", 404]])).not.toContain("Not readable");
  });
  it("says when no budgets are set", () => {
    expect(formatBudgetDetails("litellm", {}, new Map(), NOW)).toBe('LiteLLM ("litellm") budget\n  no budgets set');
  });
});

describe("budgetDisplaySetting", () => {
  it("reads the display setting", () => {
    expect(budgetDisplaySetting(undefined)).toEqual({ display: "all" });
    expect(budgetDisplaySetting({})).toEqual({ display: "all" });
    expect(budgetDisplaySetting({ display: "tightest" })).toEqual({ display: "tightest" });
    expect(budgetDisplaySetting({ display: "compact" })).toEqual({
      display: "all",
      warning: 'LiteLLM budget: unknown display "compact"; using "all".',
    });
    expect(budgetDisplaySetting({ display: 5 }).warning).toBe('LiteLLM budget: unknown display 5; using "all".');
  });
});
