import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { dirname, extname, isAbsolute, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import { type BackendIdentityRow, isRecord, resolveBackendIdentity } from "../src/backend-identity.js";
import { loadPublicCatalog } from "../src/public-catalog.js";

const LEVELS = ["off", "minimal", "low", "medium", "high", "xhigh", "max"] as const;
type ReasoningLevel = (typeof LEVELS)[number];

type JsonObject = Record<string, unknown>;
type Discover = (
  baseUrl: string,
  apiKey: string,
  options?: JsonObject,
) => Promise<{ source: string; models: ProbeModel[] }>;
type ProbeModel = {
  id: string;
  api: string;
  reasoning: boolean;
  thinkingLevelMap?: Record<string, unknown>;
  contextWindow: number;
  maxTokens: number;
  cost: JsonObject;
};

export interface ProbeSnapshot {
  modelInfo: unknown;
  modelGroupInfo?: unknown;
  models?: unknown;
}

export interface ProbeOptions {
  snapshot?: string;
  baseUrl?: string;
  apiKey?: string;
  src?: string;
  live?: boolean;
  models?: string[];
  levels?: string[];
  maxRequests?: number;
}

export interface ProbeReport {
  source: string;
  models: Array<{
    id: string;
    identity?: ReturnType<typeof resolveBackendIdentity>;
    publicSources: string[];
    liteLLMFlags: Record<string, boolean>;
    api: string;
    reasoning: boolean;
    thinkingLevelMap?: Record<string, unknown>;
    limits: { context: number; output: number };
    cost: JsonObject;
    predictions: { protocol: string; reasoning: Partial<Record<ReasoningLevel, boolean>> };
  }>;
  live?: LiveResult[];
  mismatches?: string[];
  informational?: string[];
}

export interface LiveComparison {
  mismatches: string[];
  informational: string[];
}

export interface LiveResult {
  path: "chat" | "messages";
  model: string;
  level: string;
  status: number;
  errorClass?: string;
  reasoningTokens?: number;
  accepted: boolean | null;
}

function flagValue(args: string[], index: number, name: string): string {
  const value = args[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`${name} requires a value`);
  return value;
}

export function parseProbeArgs(args: string[]): ProbeOptions {
  const options: ProbeOptions = {};
  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--snapshot") options.snapshot = flagValue(args, index++, arg);
    else if (arg === "--base-url") options.baseUrl = flagValue(args, index++, arg);
    else if (arg === "--api-key") options.apiKey = flagValue(args, index++, arg);
    else if (arg === "--src") options.src = flagValue(args, index++, arg);
    else if (arg === "--models") options.models = flagValue(args, index++, arg).split(",").filter(Boolean);
    else if (arg === "--levels") options.levels = flagValue(args, index++, arg).split(",").filter(Boolean);
    else if (arg === "--max-requests") {
      options.maxRequests = Number.parseInt(flagValue(args, index++, arg), 10);
      if (!Number.isSafeInteger(options.maxRequests) || options.maxRequests < 1) {
        throw new Error("--max-requests must be a positive integer");
      }
    } else if (arg === "--live") options.live = true;
    else throw new Error(`Unknown argument: ${arg}`);
  }
  options.baseUrl ??= process.env.LITELLM_BASE_URL;
  if (options.baseUrl) options.baseUrl = normalizeBaseUrl(options.baseUrl);
  options.apiKey ??= process.env.LITELLM_API_KEY;
  if (!options.snapshot && (!options.baseUrl || !options.apiKey)) {
    throw new Error("Use --snapshot, or provide --base-url and --api-key (or LITELLM_BASE_URL/LITELLM_API_KEY)");
  }
  if (options.snapshot && options.live) throw new Error("--live requires a live --base-url, not --snapshot");
  return options;
}

export function normalizeBaseUrl(input: string): string {
  return input.replace(/\/+$/, "").replace(/\/v1\/?$/i, "");
}

function rowsFrom(value: unknown): BackendIdentityRow[] {
  if (!isRecord(value) || !Array.isArray(value.data)) return [];
  return value.data.filter(isRecord) as BackendIdentityRow[];
}

async function optionalJson(path: string): Promise<unknown | undefined> {
  try {
    return JSON.parse(await readFile(path, "utf8"));
  } catch {
    return undefined;
  }
}

export async function readSnapshot(path: string): Promise<ProbeSnapshot> {
  const parsed: unknown = JSON.parse(await readFile(path, "utf8"));
  if (isRecord(parsed) && Object.hasOwn(parsed, "modelInfo")) return parsed as unknown as ProbeSnapshot;
  const filename = path.slice(path.lastIndexOf("/") + 1);
  if (!filename.includes("model-info")) return { modelInfo: parsed };
  const snapshotPath = resolve(path);
  const directory = dirname(snapshotPath);
  const companion = async (name: string): Promise<unknown | undefined> => {
    const candidate = resolve(directory, name);
    return candidate === snapshotPath ? undefined : optionalJson(candidate);
  };
  return {
    modelInfo: parsed,
    modelGroupInfo: await companion(filename.replace("model-info", "model-group-info")),
    models: await companion(filename.replace("model-info", "models")),
  };
}

function jsonResponse(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), { status, headers: { "content-type": "application/json" } });
}

function snapshotFetch(snapshot: ProbeSnapshot): typeof fetch {
  return (async (input: string | URL | Request) => {
    const url = new URL(typeof input === "string" || input instanceof URL ? input : input.url);
    if (url.hostname === "models.dev") return jsonResponse({});
    if (url.pathname.endsWith("/model/info")) return jsonResponse(snapshot.modelInfo);
    if (url.pathname.endsWith("/model_group/info")) return jsonResponse(snapshot.modelGroupInfo ?? { data: [] });
    if (url.pathname.endsWith("/v1/models")) return jsonResponse(snapshot.models ?? { data: [] });
    if (url.pathname.endsWith("/health")) return jsonResponse({}, 404);
    throw new Error(`Snapshot has no response for ${url.pathname}`);
  }) as typeof fetch;
}

async function loadDiscover(src = join(process.cwd(), "src")): Promise<Discover> {
  let path = isAbsolute(src) ? src : resolve(src);
  if (!extname(path)) {
    const worktreeDiscover = join(path, "src", "discover.ts");
    path = existsSync(worktreeDiscover) ? worktreeDiscover : join(path, "discover.ts");
  }
  const module = (await import(pathToFileURL(path).href)) as { discoverModels?: unknown };
  if (typeof module.discoverModels !== "function") throw new Error(`${path} does not export discoverModels`);
  return module.discoverModels as Discover;
}

function reasoningFlags(row: BackendIdentityRow): Record<string, boolean> {
  const result: Record<string, boolean> = {};
  const modelInfo = row.model_info as JsonObject | undefined;
  if (!modelInfo) return result;
  for (const [key, value] of Object.entries(modelInfo)) {
    if (/^supports_(?:minimal|low|xhigh|max)_reasoning_effort$/.test(key) && typeof value === "boolean") {
      result[key] = value;
    }
  }
  return result;
}

function allowedReasoning(row: BackendIdentityRow): boolean {
  const info = row.model_info as JsonObject | undefined;
  const supported = Array.isArray(info?.supported_openai_params) ? info.supported_openai_params : [];
  const allowed = Array.isArray((row.litellm_params as JsonObject | undefined)?.allowed_openai_params)
    ? ((row.litellm_params as JsonObject).allowed_openai_params as unknown[])
    : [];
  return [...supported, ...allowed].includes("reasoning_effort");
}

export function protocolPrediction(row: BackendIdentityRow): string {
  const identity = resolveBackendIdentity(row);
  const info = row.model_info as JsonObject | undefined;
  if (info?.mode === "responses") return "openai-responses";
  if (identity?.family !== "openai") return "openai-completions";
  const params = row.litellm_params as JsonObject | undefined;
  const version = typeof params?.api_version === "string" ? params.api_version : undefined;
  const adapter = typeof info?.litellm_provider === "string" ? info.litellm_provider : undefined;
  if ((adapter === "azure" || adapter === "azure_ai") && version && version.slice(0, 10) < "2025-03-01") {
    return "openai-completions";
  }
  return "openai-responses";
}

export function reasoningPrediction(
  row: BackendIdentityRow,
  publicEfforts: readonly string[] | undefined,
): Partial<Record<ReasoningLevel, boolean>> {
  if (!allowedReasoning(row)) return Object.fromEntries(LEVELS.map((level) => [level, false]));
  // pi-ai getSupportedThinkingLevels treats absent standard levels as selectable for reasoning models.
  const predictions: Partial<Record<ReasoningLevel, boolean>> = {
    off: true,
    minimal: true,
    low: true,
    medium: true,
    high: true,
  };
  for (const level of publicEfforts ?? []) {
    const normalized = level === "none" ? "off" : level;
    if ((LEVELS as readonly string[]).includes(normalized)) predictions[normalized as ReasoningLevel] = true;
  }
  for (const [flag, value] of Object.entries(reasoningFlags(row))) {
    const level = flag.slice("supports_".length, -"_reasoning_effort".length) as ReasoningLevel;
    predictions[level] = value;
  }
  for (const level of ["xhigh", "max"] as const) {
    if (reasoningFlags(row)[`supports_${level}_reasoning_effort`] !== true) predictions[level] = false;
  }
  return predictions;
}

export function publicCatalogOptions(snapshot: ProbeSnapshot | undefined): { offline: true } | { cachePath: string } {
  return snapshot ? { offline: true } : { cachePath: join(getAgentDir(), "litellm-probe-models-dev.json") };
}

export async function probeDiscovery(options: ProbeOptions): Promise<ProbeReport> {
  const snapshot = options.snapshot ? await readSnapshot(options.snapshot) : undefined;
  const baseUrl = snapshot ? "https://snapshot.invalid" : normalizeBaseUrl(options.baseUrl as string);
  const apiKey = snapshot ? "snapshot" : (options.apiKey as string);
  const originalFetch = globalThis.fetch;
  if (snapshot) globalThis.fetch = snapshotFetch(snapshot);
  try {
    const discoverModels = await loadDiscover(options.src);
    const discovery = await discoverModels(baseUrl, apiKey, { silent: true, modelsDev: false });
    const rawInfo = snapshot
      ? snapshot.modelInfo
      : await originalFetch(`${baseUrl.replace(/\/+$/, "")}/model/info`, {
          headers: { Authorization: `Bearer ${apiKey}`, Accept: "application/json" },
        }).then((response) => response.json());
    const rows = rowsFrom(rawInfo);
    if (discovery.models.length > 0 && rows.length === 0) {
      throw new Error(`/model/info returned zero rows for ${discovery.models.length} discovered models`);
    }
    const rowsByName = new Map(rows.map((row) => [row.model_name, row]));
    const catalog = await loadPublicCatalog(publicCatalogOptions(snapshot));
    const models = discovery.models.map((model) => {
      const row = rowsByName.get(model.id) ?? { model_name: model.id };
      const identity = resolveBackendIdentity(row);
      const provider = identity?.provider ?? (row.model_info as JsonObject | undefined)?.litellm_provider;
      const publicRecord = identity
        ? catalog.lookup(typeof provider === "string" ? provider : undefined, identity.modelId)
        : undefined;
      return {
        id: model.id,
        identity,
        publicSources: publicRecord ? [publicRecord.source] : [],
        liteLLMFlags: reasoningFlags(row),
        api: model.api,
        reasoning: model.reasoning,
        ...(model.thinkingLevelMap ? { thinkingLevelMap: model.thinkingLevelMap } : {}),
        limits: { context: model.contextWindow, output: model.maxTokens },
        cost: model.cost,
        predictions: {
          protocol: protocolPrediction(row),
          reasoning: reasoningPrediction(row, publicRecord?.effortLevels),
        },
      };
    });
    const report: ProbeReport = { source: discovery.source, models };
    if (options.live) {
      report.live = await runLiveMatrix(baseUrl, apiKey, models, options);
      const comparison = compareLive(report.live, models);
      report.mismatches = comparison.mismatches;
      report.informational = comparison.informational;
      if (comparison.mismatches.length > 0) process.exitCode = 1;
    }
    return report;
  } finally {
    globalThis.fetch = originalFetch;
  }
}

export function acceptanceOracle(result: { status: number; errorClass?: string }): boolean | null {
  if (result.status >= 200 && result.status < 300) return true;
  if (result.status === 400 && result.errorClass === "UnsupportedParamsError") return false;
  return null;
}

export function errorClass(body: unknown): string | undefined {
  const error = isRecord(body) && isRecord(body.error) ? body.error : body;
  if (isRecord(error)) {
    for (const key of ["type", "error_class"]) {
      if (typeof error[key] === "string" && /Error$/.test(error[key])) return error[key];
    }
    if (typeof error.message === "string") {
      const match = /litellm\.(\w+Error)/.exec(error.message);
      if (match) return match[1];
    }
    if (typeof error.code === "string" && /Error$/.test(error.code)) return error.code;
  }
  const raw = typeof error === "string" ? error : typeof body === "string" ? body : undefined;
  return raw ? /litellm\.(\w+Error)/.exec(raw)?.[1] : undefined;
}

function reasoningTokens(body: unknown): number | undefined {
  if (!isRecord(body)) return undefined;
  const usage = isRecord(body.usage) ? body.usage : undefined;
  const details = usage && isRecord(usage.completion_tokens_details) ? usage.completion_tokens_details : undefined;
  const value = details?.reasoning_tokens ?? usage?.reasoning_tokens;
  return typeof value === "number" ? value : undefined;
}

async function liveRequest(
  baseUrl: string,
  apiKey: string,
  model: string,
  level: string,
  path: "chat" | "messages",
): Promise<LiveResult> {
  const endpoint = path === "messages" ? "/v1/messages" : "/v1/chat/completions";
  const body =
    path === "messages"
      ? {
          model,
          max_tokens: 16,
          messages: [{ role: "user", content: "Reply with one word." }],
          thinking: { type: "adaptive" },
          output_config: { effort: level },
        }
      : {
          model,
          max_tokens: 16,
          messages: [{ role: "user", content: "Reply with one word." }],
          reasoning_effort: level === "off" ? "none" : level,
        };
  try {
    const response = await fetch(`${baseUrl}${endpoint}`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        Accept: "application/json",
        "Content-Type": "application/json",
        "x-litellm-session-id": `pi-probe-${new Date().toISOString().slice(0, 10)}`,
      },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(30_000),
    });
    let parsed: unknown;
    try {
      parsed = await response.json();
    } catch {
      parsed = undefined;
    }
    const result = { status: response.status, errorClass: errorClass(parsed) };
    const accepted = acceptanceOracle(result);
    return {
      path,
      model,
      level,
      ...result,
      reasoningTokens: reasoningTokens(parsed),
      accepted,
    };
  } catch (error) {
    return {
      path,
      model,
      level,
      status: 0,
      errorClass: error instanceof Error ? error.name : "UnknownError",
      accepted: null,
    };
  }
}

export async function runLiveMatrix(
  baseUrl: string,
  apiKey: string,
  models: ProbeReport["models"],
  options: ProbeOptions,
): Promise<LiveResult[]> {
  const selected = options.models ? new Set(options.models) : undefined;
  const levels = options.levels ?? ["low", "medium", "high", "xhigh"];
  const requests: Array<{ model: string; level: string; path: "chat" | "messages" }> = [];
  for (const model of models) {
    if (selected && !selected.has(model.id)) continue;
    const path = model.api === "anthropic-messages" ? "messages" : "chat";
    for (const level of levels) requests.push({ model: model.id, level, path });
  }
  const bounded = requests.slice(0, options.maxRequests ?? 100);
  const results: LiveResult[] = [];
  for (const request of bounded)
    results.push(await liveRequest(baseUrl, apiKey, request.model, request.level, request.path));
  return results;
}

function emittedSelectable(model: ProbeReport["models"][number], level: ReasoningLevel): boolean {
  if (!model.reasoning) return false;
  const mapped = model.thinkingLevelMap?.[level];
  if (mapped === null) return false;
  return level === "xhigh" || level === "max" ? mapped !== undefined : true;
}

export function compareLive(results: LiveResult[], models: ProbeReport["models"]): LiveComparison {
  const byId = new Map(models.map((model) => [model.id, model]));
  const mismatches: string[] = [];
  const informational: string[] = [];
  for (const result of results) {
    const model = byId.get(result.model);
    if (!model) {
      mismatches.push(`${result.model} ${result.path} ${result.level}: model missing from discovery`);
      continue;
    }
    const predicted = result.path === "messages" ? true : emittedSelectable(model, result.level as ReasoningLevel);
    if (result.accepted === null) {
      mismatches.push(
        `${result.model} ${result.path} ${result.level}: unclassified HTTP ${result.status}${result.errorClass ? ` ${result.errorClass}` : ""}`,
      );
      continue;
    }
    if (predicted === true && !result.accepted) {
      mismatches.push(`${result.model} ${result.path} ${result.level}: predicted selectable, actual rejected`);
      continue;
    }
    if (result.accepted && predicted !== true) {
      // D4 requires completeness only for low/medium/high. Whether `off` can
      // disable reasoning belongs to the generation contract; `minimal`,
      // xhigh, and max also remain explicit-evidence-only extensions.
      if (
        result.path === "chat" &&
        (result.level === "off" || result.level === "minimal" || result.level === "xhigh" || result.level === "max")
      ) {
        informational.push(`${result.model} ${result.path} ${result.level}: accepted but not offered`);
      } else if (
        result.path === "messages" ||
        result.level === "low" ||
        result.level === "medium" ||
        result.level === "high"
      ) {
        mismatches.push(`${result.model} ${result.path} ${result.level}: accepted but not predicted selectable`);
      }
    }
  }
  return { mismatches, informational };
}

if (import.meta.main) {
  probeDiscovery(parseProbeArgs(process.argv.slice(2)))
    .then((report) => console.log(JSON.stringify(report, null, 2)))
    .catch((error) => {
      console.error(error instanceof Error ? error.message : String(error));
      process.exitCode = 1;
    });
}
