import { fetchJson, normalizeBaseUrl } from "./discover.js";
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
    const url = `${normalizeBaseUrl(auth.baseUrl, auth.allowInsecureHttp)}${path}`;
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

export type BudgetDisplay = "all" | "tightest";
export interface BudgetTheme {
  fg(color: "dim" | "warning" | "error", text: string): string;
}

// A header counts when it is a non-blank string holding a finite number >= 0.
function headerNumber(value: string | undefined): number | undefined {
  if (value === undefined || value.trim() === "") return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : undefined;
}

export function mergeKeyHeaders(
  key: BudgetLevel | undefined,
  keyPolled: boolean,
  headers: Record<string, string>,
): BudgetLevel | undefined {
  const spend = headerNumber(headers["x-litellm-key-spend"]);
  if (spend === undefined) return key;
  const limit = headerNumber(headers["x-litellm-key-max-budget"]);
  const maxBudget = limit !== undefined && limit > 0 ? limit : undefined;
  if (!keyPolled) return maxBudget === undefined ? undefined : { spend, maxBudget };
  if (key) return { ...key, spend: Math.max(key.spend, spend), maxBudget: maxBudget ?? key.maxBudget };
  return maxBudget === undefined ? undefined : { spend, maxBudget };
}

export function formatCompactAmount(value: number): string {
  if (value < 100) return `$${Number.isInteger(value) ? value : value.toFixed(2)}`;
  if (value < 1000) return `$${Math.round(value)}`;
  const [scaled, suffix] = value >= 1_000_000 ? [value / 1_000_000, "M"] : [value / 1000, "k"];
  return `$${scaled < 10 ? scaled.toFixed(1).replace(/\.0$/, "") : Math.round(scaled)}${suffix}`;
}

export function formatRelativeReset(resetAt: number | undefined, now: number): string | undefined {
  if (resetAt === undefined || resetAt <= now) return undefined;
  const ms = resetAt - now;
  if (ms >= 86_400_000) return `${Math.floor(ms / 86_400_000)}d`;
  if (ms >= 3_600_000) return `${Math.floor(ms / 3_600_000)}h`;
  return `${Math.max(1, Math.floor(ms / 60_000))}m`;
}

const percent = (l: BudgetLevel): number => Math.floor((l.spend / l.maxBudget) * 100);

function colour(l: BudgetLevel): "dim" | "warning" | "error" {
  const ratio = l.spend / l.maxBudget;
  return ratio >= 1 ? "error" : ratio >= 0.8 ? "warning" : "dim";
}

export function formatBudgetStatus(
  levels: BudgetLevels,
  display: BudgetDisplay,
  prefix: string,
  theme: BudgetTheme,
  now: number,
): string | undefined {
  const set = BUDGET_LEVELS.flatMap((name) => (levels[name] ? [{ name, level: levels[name] }] : []));
  if (set.length === 0) return undefined;
  const head = `${theme.fg("dim", prefix)} `;
  const text = (name: string, l: BudgetLevel) =>
    `${name} ${formatCompactAmount(l.spend)}/${formatCompactAmount(l.maxBudget)}`;
  if (display === "all") {
    return head + set.map(({ name, level: l }) => theme.fg(colour(l), text(name, l))).join(theme.fg("dim", " · "));
  }
  // Strict "<" keeps the first level in order on ties.
  const { name, level: l } = set.reduce((a, b) =>
    b.level.maxBudget - b.level.spend < a.level.maxBudget - a.level.spend ? b : a,
  );
  const reset = formatRelativeReset(l.resetAt, now);
  return (
    head +
    theme.fg(colour(l), `${text(name, l)} (${percent(l)}%)`) +
    (reset ? theme.fg("dim", ` · resets ${reset}`) : "")
  );
}

const money = (value: number, decimals: number): string =>
  `$${value.toLocaleString("en-US", { minimumFractionDigits: decimals, maximumFractionDigits: 2 })}`;

export function formatBudgetDetails(
  providerName: string,
  levels: BudgetLevels,
  denied: ReadonlyMap<BudgetEndpoint, number>,
  now: number,
): string {
  const lines = [`LiteLLM ("${providerName}") budget`];
  for (const name of BUDGET_LEVELS) {
    const l = levels[name];
    if (!l) continue;
    const reset = formatRelativeReset(l.resetAt, now);
    const limit = money(l.maxBudget, Number.isInteger(l.maxBudget) ? 0 : 2);
    lines.push(
      `  ${name.padEnd(8)}${money(l.spend, 2)} of ${limit} (${percent(l)}%)${reset ? `, resets in ${reset}` : ""}`,
    );
  }
  if (lines.length === 1) lines.push("  no budgets set");
  const userV2 = denied.get("userV2");
  const unreadable = [
    ["key", denied.get("key")],
    ["user", userV2 !== undefined && userV2 !== 404 ? userV2 : denied.get("user")],
    ["team", denied.get("team")],
    ["org", denied.get("org")],
  ].flatMap(([name, code]) => (code === undefined ? [] : [`${name} (${code})`]));
  if (unreadable.length > 0) lines.push(`  Not readable with this credential: ${unreadable.join(", ")}`);
  return lines.join("\n");
}

export function budgetDisplaySetting(budgetSettings: unknown): { display: BudgetDisplay; warning?: string } {
  const display = obj(budgetSettings)?.display;
  if (display === undefined || display === "all" || display === "tightest") return { display: display ?? "all" };
  return { display: "all", warning: `LiteLLM budget: unknown display ${JSON.stringify(display)}; using "all".` };
}
