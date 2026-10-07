import { afterEach, describe, expect, it, vi } from "vitest";
import { type BudgetEndpoint, type BudgetLevels, pollBudget } from "../src/budget.js";
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
