import { fetchJson } from "./discover.js";
import type { LiteLLMRuntimeAuth } from "./types.js";

export type BudgetLevelName = "key" | "user" | "team" | "member" | "org";
export const BUDGET_LEVELS: readonly BudgetLevelName[] = ["key", "user", "team", "member", "org"];

export type BudgetEndpoint = "key" | "userV2" | "user" | "team" | "org";

export interface BudgetLevel {
  spend: number;
  maxBudget: number;
  resetAt?: number;
}
export type BudgetLevels = Partial<Record<BudgetLevelName, BudgetLevel>>;

export type BudgetPoll = { ok: true; levels: BudgetLevels; keyPolled: boolean } | { ok: false; reason: string };

type Obj = Record<string, unknown>;
type Fetched = { status: "ok"; data: unknown } | { status: "denied" } | { status: "transient"; reason: string };

function obj(value: unknown): Obj | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value) ? (value as Obj) : undefined;
}

function str(value: unknown): string | undefined {
  return typeof value === "string" && value !== "" ? value : undefined;
}

// A level exists only for a positive, finite limit; anything else means "no budget".
function level(spend: unknown, maxBudget: unknown, resetAt: unknown): BudgetLevel | undefined {
  if (typeof maxBudget !== "number" || !Number.isFinite(maxBudget) || maxBudget <= 0) return undefined;
  const parsedReset = typeof resetAt === "string" ? Date.parse(resetAt) : Number.NaN;
  return {
    spend: typeof spend === "number" && Number.isFinite(spend) && spend >= 0 ? spend : 0,
    maxBudget,
    ...(Number.isFinite(parsedReset) ? { resetAt: parsedReset } : {}),
  };
}

// Levels carried over from the last good poll when their endpoint failed transiently.
function keep(previous: BudgetLevels, names: readonly BudgetLevelName[]): BudgetLevels {
  const kept: BudgetLevels = {};
  for (const name of names) if (previous[name]) kept[name] = previous[name];
  return kept;
}

async function get(
  auth: LiteLLMRuntimeAuth,
  path: string,
  endpoint: BudgetEndpoint,
  denied: Map<BudgetEndpoint, number>,
  timeoutMs: number,
): Promise<Fetched> {
  try {
    const url = `${auth.baseUrl.replace(/\/+$/, "")}${path}`;
    const result = await fetchJson<unknown>(url, auth.apiKey, { timeoutMs, headers: auth.headers });
    if (result.ok) return { status: "ok", data: result.data };
    if (result.status >= 400 && result.status < 500 && result.status !== 429) {
      denied.set(endpoint, result.status);
      return { status: "denied" };
    }
    return { status: "transient", reason: `HTTP ${result.status}` };
  } catch {
    return { status: "transient", reason: "request failed" };
  }
}

async function pollUser(
  auth: LiteLLMRuntimeAuth,
  denied: Map<BudgetEndpoint, number>,
  previous: BudgetLevels,
  timeoutMs: number,
): Promise<BudgetLevels> {
  const keepUser = (): BudgetLevels => keep(previous, ["user"]);
  let fetched: Fetched | undefined;
  let nested = false;
  if (!denied.has("userV2")) {
    fetched = await get(auth, "/v2/user/info", "userV2", denied, timeoutMs);
    if (fetched.status === "denied" && denied.get("userV2") === 404) fetched = undefined;
  } else if (denied.get("userV2") !== 404) {
    return {};
  }
  if (!fetched) {
    if (denied.has("user")) return {};
    nested = true;
    fetched = await get(auth, "/user/info", "user", denied, timeoutMs);
  }
  if (fetched.status === "denied") return {};
  if (fetched.status === "transient") return keepUser();
  const body = nested ? obj(obj(fetched.data)?.user_info) : obj(fetched.data);
  if (!body) return keepUser();
  const user = level(body.spend, body.max_budget, body.budget_reset_at);
  return user ? { user } : {};
}

async function pollTeam(
  auth: LiteLLMRuntimeAuth,
  teamId: string,
  userId: string | undefined,
  denied: Map<BudgetEndpoint, number>,
  previous: BudgetLevels,
  timeoutMs: number,
): Promise<{ levels: BudgetLevels; organizationId?: string }> {
  const path = `/team/info?team_id=${encodeURIComponent(teamId)}`;
  const fetched = await get(auth, path, "team", denied, timeoutMs);
  if (fetched.status === "denied") return { levels: {} };
  const body = fetched.status === "ok" ? obj(fetched.data) : undefined;
  const info = obj(body?.team_info);
  if (!info) return { levels: keep(previous, ["team", "member"]) };

  const levels: BudgetLevels = {};
  const team = level(info.spend, info.max_budget, info.budget_reset_at);
  if (team) levels.team = team;
  if (userId) {
    const memberships = body?.team_memberships;
    const own = Array.isArray(memberships) ? memberships.map(obj).find((m) => m?.user_id === userId) : undefined;
    const table = obj(own?.litellm_budget_table) ?? obj(info.team_member_budget_table);
    const member = level(own?.spend, table?.max_budget, table?.budget_reset_at);
    if (member) levels.member = member;
  }
  return { levels, organizationId: str(info.organization_id) };
}

async function pollOrg(
  auth: LiteLLMRuntimeAuth,
  organizationId: string,
  denied: Map<BudgetEndpoint, number>,
  previous: BudgetLevels,
  timeoutMs: number,
): Promise<BudgetLevels> {
  const fetched = await get(auth, "/organization/list", "org", denied, timeoutMs);
  if (fetched.status === "denied") return {};
  if (fetched.status === "transient" || !Array.isArray(fetched.data)) return keep(previous, ["org"]);
  const entry = fetched.data.map(obj).find((o) => o?.organization_id === organizationId);
  const table = obj(entry?.litellm_budget_table);
  const org = level(entry?.spend, table?.max_budget, table?.budget_reset_at);
  return org ? { org } : {};
}

export async function pollBudget(
  auth: LiteLLMRuntimeAuth,
  denied: Map<BudgetEndpoint, number>,
  previous: BudgetLevels,
  timeoutMs: number,
): Promise<BudgetPoll> {
  const levels: BudgetLevels = {};
  let userId: string | undefined;
  let teamId: string | undefined;
  let organizationId: string | undefined;
  const keyPolled = !denied.has("key");

  if (keyPolled) {
    const fetched = await get(auth, "/key/info", "key", denied, timeoutMs);
    if (fetched.status === "transient") return { ok: false, reason: fetched.reason };
    if (fetched.status === "ok") {
      const info = obj(obj(fetched.data)?.info);
      if (!info) return { ok: false, reason: "malformed response" };
      const key = level(info.spend, info.max_budget, info.budget_reset_at);
      if (key) levels.key = key;
      userId = str(info.user_id);
      teamId = str(info.team_id);
      organizationId = str(info.organization_id);
    }
  }
  const keyKnown = keyPolled && !denied.has("key");

  const [user, team] = await Promise.all([
    userId || !keyKnown ? pollUser(auth, denied, previous, timeoutMs) : {},
    teamId && !denied.has("team")
      ? pollTeam(auth, teamId, userId, denied, previous, timeoutMs)
      : { levels: {} as BudgetLevels, organizationId: undefined },
  ]);
  Object.assign(levels, user, team.levels);

  const org = organizationId ?? team.organizationId;
  if (org && userId && !denied.has("org")) Object.assign(levels, await pollOrg(auth, org, denied, previous, timeoutMs));
  return { ok: true, levels, keyPolled: keyKnown };
}
