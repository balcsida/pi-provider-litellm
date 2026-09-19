import { existsSync } from "node:fs";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, extname, isAbsolute, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import {
  type BackendIdentityRow,
  isRecord,
  resolveBackendIdentity,
  resolveCatalogProvider,
} from "../src/backend-identity.js";
import { wildcardMatches } from "../src/discover.js";
import { prepareLiteLLMRequestPayload } from "../src/index.js";
import { isResponsesMode } from "../src/model-groups.js";
import { loadPublicCatalog } from "../src/public-catalog.js";
import type { LiteLLMModel, LiteLLMModelPolicy } from "../src/types.js";

const LEVELS = ["off", "minimal", "low", "medium", "high", "xhigh", "max"] as const;
type ReasoningLevel = (typeof LEVELS)[number];

type JsonObject = Record<string, unknown>;
type Discover = (
  baseUrl: string,
  apiKey: string,
  options?: JsonObject,
) => Promise<{ source: string; models: ProbeModel[] }>;
type PreparePayload = typeof prepareLiteLLMRequestPayload;
type ProbeModel = {
  id: string;
  api: string;
  compat?: JsonObject;
  litellmPolicy?: LiteLLMModelPolicy;
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
    deployments: number;
    identity?: ReturnType<typeof resolveBackendIdentity>;
    publicSources: string[];
    liteLLMFlags: Record<string, boolean>;
    api: string;
    compat?: JsonObject;
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
  path: "chat" | "messages" | "responses";
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

function sourceFile(src: string | undefined, name: "discover.ts" | "index.ts"): string {
  let path = isAbsolute(src ?? "") ? (src as string) : resolve(src ?? join(process.cwd(), "src"));
  if (!extname(path)) {
    const worktreeFile = join(path, "src", name);
    path = existsSync(worktreeFile) ? worktreeFile : join(path, name);
  } else if (name !== "discover.ts") {
    path = join(dirname(path), name);
  }
  return path;
}

async function loadDiscover(src?: string): Promise<Discover> {
  const path = sourceFile(src, "discover.ts");
  const module = (await import(pathToFileURL(path).href)) as { discoverModels?: unknown };
  if (typeof module.discoverModels !== "function") throw new Error(`${path} does not export discoverModels`);
  return module.discoverModels as Discover;
}

async function loadPreparePayload(src?: string): Promise<PreparePayload> {
  const path = sourceFile(src, "index.ts");
  const module = (await import(pathToFileURL(path).href)) as { prepareLiteLLMRequestPayload?: unknown };
  if (typeof module.prepareLiteLLMRequestPayload !== "function") {
    throw new Error(`${path} does not export prepareLiteLLMRequestPayload`);
  }
  return module.prepareLiteLLMRequestPayload as PreparePayload;
}

function reasoningFlags(row: BackendIdentityRow): Record<string, boolean> {
  const result: Record<string, boolean> = {};
  const modelInfo = row.model_info as JsonObject | undefined;
  if (!modelInfo) return result;
  for (const [key, value] of Object.entries(modelInfo)) {
    if (/^supports_(?:none|minimal|low|xhigh|max)_reasoning_effort$/.test(key) && typeof value === "boolean") {
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
  if ([...supported, ...allowed].includes("reasoning_effort")) return true;
  // Discovery's off-map opt-in: LiteLLM omits the list for a deployment outside its model
  // map, and Kimi/DeepSeek generation contracts name their own carriers.
  const family = resolveBackendIdentity(row)?.family;
  return (
    (info?.supported_openai_params === undefined || info?.supported_openai_params === null) &&
    info?.supports_reasoning === true &&
    family !== "kimi" &&
    family !== "deepseek"
  );
}

export function protocolPrediction(row: BackendIdentityRow): string {
  const identity = resolveBackendIdentity(row);
  const info = row.model_info as JsonObject | undefined;
  if (Array.isArray(info?.supported_endpoints)) {
    return info.supported_endpoints.includes("/v1/responses") ? "openai-responses" : "openai-completions";
  }
  if (isResponsesMode(info?.mode)) return "openai-responses";
  if (identity?.family !== "openai") return "openai-completions";
  const params = row.litellm_params as JsonObject | undefined;
  const providers = [
    params?.custom_llm_provider,
    info?.litellm_provider,
    typeof params?.model === "string" ? params.model.trim().split("/")[0] : undefined,
  ];
  const azure = providers.some(
    (provider) => typeof provider === "string" && ["azure", "azure_ai"].includes(provider.trim().toLowerCase()),
  );
  return azure ? "openai-completions" : "openai-responses";
}

export function reasoningPrediction(
  row: BackendIdentityRow,
  publicEfforts: readonly string[] | undefined,
  catalogMap?: Record<string, unknown>,
): Partial<Record<ReasoningLevel, boolean>> {
  if (!allowedReasoning(row)) return Object.fromEntries(LEVELS.map((level) => [level, false]));
  const declared = (row.model_info as JsonObject | undefined)?.reasoning_effort_levels;
  if (Array.isArray(declared)) {
    // Discovery reads a declared list whole, ahead of public effort lists and flags.
    const levels = new Set(declared.map((level) => (level === "none" ? "off" : level)));
    return Object.fromEntries(LEVELS.map((level) => [level, levels.has(level)]));
  }
  const normalizedPublicEfforts = (publicEfforts ?? [])
    .map((level) => (level === "none" ? "off" : level))
    .filter((level): level is ReasoningLevel => (LEVELS as readonly string[]).includes(level));
  const publicSet = new Set(normalizedPublicEfforts);
  // A usable public effort list is complete. Without one, pi-ai treats absent
  // standard levels as selectable for reasoning models.
  const predictions: Partial<Record<ReasoningLevel, boolean>> = Object.fromEntries(
    (["off", "minimal", "low", "medium", "high"] as const).map((level) => [
      level,
      normalizedPublicEfforts.length > 0 ? publicSet.has(level) : true,
    ]),
  );
  // A catalog map is tristate, as in discovery: a null denies, a value keeps a level
  // the public list did not deny, and an omitted level keeps the default.
  for (const level of LEVELS) {
    const value = catalogMap?.[level];
    if (value === null) predictions[level] = false;
    else if (value !== undefined && predictions[level] !== false) predictions[level] = true;
  }
  for (const [flag, value] of Object.entries(reasoningFlags(row))) {
    const effort = flag.slice("supports_".length, -"_reasoning_effort".length);
    const level = effort === "none" ? "off" : (effort as ReasoningLevel);
    predictions[level] = value;
  }
  for (const level of ["xhigh", "max"] as const) {
    if (reasoningFlags(row)[`supports_${level}_reasoning_effort`] !== true) predictions[level] = false;
  }
  return predictions;
}

export function publicCatalogOptions(
  snapshot: ProbeSnapshot | undefined,
): { offline: true; cachePath: string } | { cachePath: string } {
  const cachePath = join(getAgentDir(), "litellm-models-dev.json");
  return snapshot ? { offline: true, cachePath } : { cachePath };
}

export async function probeDiscovery(options: ProbeOptions): Promise<ProbeReport> {
  const originalFetch = globalThis.fetch;
  const originalAgentDir = process.env.PI_CODING_AGENT_DIR;
  const probeAgentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-probe-"));
  process.env.PI_CODING_AGENT_DIR = probeAgentDir;
  try {
    const snapshot = options.snapshot ? await readSnapshot(options.snapshot) : undefined;
    const baseUrl = snapshot ? "https://snapshot.invalid" : normalizeBaseUrl(options.baseUrl as string);
    const apiKey = snapshot ? "snapshot" : (options.apiKey as string);
    if (snapshot) globalThis.fetch = snapshotFetch(snapshot);
    const discoverModels = await loadDiscover(options.src);
    const preparePayload = options.live ? await loadPreparePayload(options.src) : prepareLiteLLMRequestPayload;
    const discovery = await discoverModels(baseUrl, apiKey, { silent: true, modelsDev: snapshot ? false : undefined });
    const rawInfo = snapshot
      ? snapshot.modelInfo
      : discovery.source === "model_info"
        ? await originalFetch(`${baseUrl.replace(/\/+$/, "")}/model/info`, {
            headers: { Authorization: `Bearer ${apiKey}`, Accept: "application/json" },
            signal: AbortSignal.timeout(30_000),
          }).then((response) => response.json())
        : undefined;
    const rows = rowsFrom(rawInfo);
    // Discovery that fell back to /v1/models or /health is reported from route-only rows; only a
    // /model/info discovery is contradicted by an empty second read of the same endpoint.
    if (discovery.source === "model_info" && discovery.models.length > 0 && rows.length === 0) {
      throw new Error(`/model/info returned zero rows for ${discovery.models.length} discovered models`);
    }
    const rowsByName = new Map<string, BackendIdentityRow[]>();
    for (const row of rows) {
      if (typeof row.model_name !== "string") continue;
      const group = rowsByName.get(row.model_name) ?? [];
      group.push(row);
      rowsByName.set(row.model_name, group);
    }
    const wildcardRows = rows.filter((row) => typeof row.model_name === "string" && row.model_name.includes("*"));
    const catalog = await loadPublicCatalog(publicCatalogOptions(snapshot));
    const models = discovery.models.map((model) => {
      const matchingWildcardRows = wildcardRows.filter((row) => wildcardMatches(row.model_name as string, model.id));
      const group =
        rowsByName.get(model.id) ??
        (matchingWildcardRows.length > 0 ? matchingWildcardRows : [{ model_name: model.id }]);
      const identities = group.map(resolveBackendIdentity);
      const identityKey = identities[0] ? JSON.stringify(identities[0]) : undefined;
      const identity =
        identityKey && identities.every((candidate) => JSON.stringify(candidate) === identityKey)
          ? identities[0]
          : undefined;
      const publicRecords = group.map((row, index) => {
        const rowIdentity = identities[index];
        const provider = resolveCatalogProvider(row, rowIdentity);
        return rowIdentity
          ? catalog.lookup(typeof provider === "string" ? provider : undefined, rowIdentity.modelId)
          : undefined;
      });
      const flagNames = new Set(group.flatMap((row) => Object.keys(reasoningFlags(row))));
      const liteLLMFlags = Object.fromEntries(
        [...flagNames].map((flag) => [flag, group.every((row) => reasoningFlags(row)[flag] === true)]),
      );
      const reasoningPredictions = group.map((row, index) =>
        reasoningPrediction(row, publicRecords[index]?.effortLevels, publicRecords[index]?.thinkingLevelMap),
      );
      return {
        id: model.id,
        deployments: group.length,
        ...(identity ? { identity } : {}),
        publicSources:
          identity && publicRecords.every((record) => record !== undefined)
            ? [...new Set(publicRecords.map((record) => record?.source).filter((source) => source !== undefined))]
            : [],
        liteLLMFlags,
        api: model.api,
        ...(model.compat ? { compat: model.compat } : {}),
        reasoning: model.reasoning,
        ...(model.thinkingLevelMap ? { thinkingLevelMap: model.thinkingLevelMap } : {}),
        limits: { context: model.contextWindow, output: model.maxTokens },
        cost: model.cost,
        predictions: {
          protocol: group.every((row) => protocolPrediction(row) === "openai-responses")
            ? "openai-responses"
            : "openai-completions",
          reasoning: Object.fromEntries(
            LEVELS.map((level) => [level, reasoningPredictions.every((prediction) => prediction[level] === true)]),
          ),
        },
      };
    });
    const report: ProbeReport = { source: discovery.source, models };
    if (options.live) {
      const modelPolicies = new Map(discovery.models.map((model) => [model.id, model.litellmPolicy]));
      report.live = await runLiveMatrix(baseUrl, apiKey, models, options, modelPolicies, preparePayload);
      const comparison = compareLive(report.live, models);
      report.mismatches = comparison.mismatches;
      report.informational = comparison.informational;
      if (comparison.mismatches.length > 0) process.exitCode = 1;
    }
    return report;
  } finally {
    globalThis.fetch = originalFetch;
    if (originalAgentDir === undefined) delete process.env.PI_CODING_AGENT_DIR;
    else process.env.PI_CODING_AGENT_DIR = originalAgentDir;
    await rm(probeAgentDir, { recursive: true, force: true });
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

function wireReasoningLevel(level: string, thinkingLevelMap?: Record<string, unknown>): string {
  const mapped = thinkingLevelMap?.[level];
  return typeof mapped === "string" ? mapped : level;
}

async function liveRequest(
  baseUrl: string,
  apiKey: string,
  model: string,
  level: string,
  path: "chat" | "messages" | "responses",
  compat?: JsonObject,
  thinkingLevelMap?: Record<string, unknown>,
  modelPolicy?: LiteLLMModelPolicy,
  preparePayload: PreparePayload = prepareLiteLLMRequestPayload,
): Promise<LiveResult> {
  const wireLevel = wireReasoningLevel(level, thinkingLevelMap);
  const endpoint =
    path === "messages" ? "/v1/messages" : path === "responses" ? "/v1/responses" : "/v1/chat/completions";
  const body =
    path === "messages"
      ? {
          model,
          max_tokens: 16,
          messages: [{ role: "user", content: "Reply with one word." }],
          ...(level === "off"
            ? { thinking: { type: "disabled" } }
            : { thinking: { type: "adaptive" }, output_config: { effort: level } }),
        }
      : path === "responses"
        ? {
            model,
            input: "Reply with one word.",
            max_output_tokens: 16,
            reasoning: {
              effort: wireLevel === "off" ? "none" : wireLevel,
              ...(level === "off" ? {} : { summary: "auto" }),
            },
            ...(level === "off" ? {} : { include: ["reasoning.encrypted_content"] }),
          }
        : {
            model,
            [compat?.maxTokensField === "max_tokens" ? "max_tokens" : "max_completion_tokens"]: 16,
            messages: [{ role: "user", content: "Reply with one word." }],
            ...chatReasoningCarrier(level, compat, thinkingLevelMap),
          };
  const preparedBody =
    preparePayload(
      body,
      { id: model, api: path === "responses" ? "openai-responses" : "openai-completions" } as LiteLLMModel,
      modelPolicy,
    ) ?? body;
  try {
    const response = await fetch(`${baseUrl}${endpoint}`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${apiKey}`,
        Accept: "application/json",
        "Content-Type": "application/json",
        "x-litellm-session-id": `pi-probe-${new Date().toISOString().slice(0, 10)}`,
      },
      body: JSON.stringify(preparedBody),
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

// Mirrors pi-ai's Chat Completions serializer for the compat this provider discovers: a
// `deepseek` thinking format rides `thinking`, and `reasoning_effort` is added only when the
// compat allows it. A model with neither carrier has nothing to probe for the level.
export function chatReasoningCarrier(
  level: string,
  compat?: JsonObject,
  thinkingLevelMap?: Record<string, unknown>,
): JsonObject | undefined {
  const wireLevel = wireReasoningLevel(level, thinkingLevelMap);
  const effort = wireLevel === "off" ? "none" : wireLevel;
  const supportsEffort = compat?.supportsReasoningEffort !== false;
  if (compat?.thinkingFormat === "deepseek") {
    return {
      thinking: { type: level === "off" ? "disabled" : "enabled" },
      ...(supportsEffort && level !== "off" ? { reasoning_effort: effort } : {}),
    };
  }
  return supportsEffort ? { reasoning_effort: effort } : undefined;
}

export async function runLiveMatrix(
  baseUrl: string,
  apiKey: string,
  models: ProbeReport["models"],
  options: ProbeOptions,
  modelPolicies: ReadonlyMap<string, LiteLLMModelPolicy | undefined> = new Map(),
  preparePayload: PreparePayload = prepareLiteLLMRequestPayload,
): Promise<LiveResult[]> {
  const selected = options.models ? new Set(options.models) : undefined;
  const levels = options.levels ?? ["low", "medium", "high", "xhigh"];
  const requests: Array<{
    model: string;
    level: string;
    path: "chat" | "messages" | "responses";
    compat?: JsonObject;
    thinkingLevelMap?: Record<string, unknown>;
    modelPolicy?: LiteLLMModelPolicy;
  }> = [];
  for (const model of models) {
    if (selected && !selected.has(model.id)) continue;
    // Production offers no reasoning selector for a non-reasoning model, so no level is ever sent.
    if (!model.reasoning) continue;
    const path =
      model.api === "anthropic-messages" ? "messages" : model.api === "openai-responses" ? "responses" : "chat";
    for (const level of levels) {
      // A Chat model without a reasoning carrier sends no level; a request that omits the
      // level would be trivially accepted and prove nothing about the map.
      if (path === "chat" && chatReasoningCarrier(level, model.compat, model.thinkingLevelMap) === undefined) continue;
      requests.push({
        model: model.id,
        level,
        path,
        compat: model.compat,
        thinkingLevelMap: model.thinkingLevelMap,
        modelPolicy: modelPolicies.get(model.id),
      });
    }
  }
  const bounded = requests.slice(0, options.maxRequests ?? 100);
  const results: LiveResult[] = [];
  for (const request of bounded) {
    results.push(
      await liveRequest(
        baseUrl,
        apiKey,
        request.model,
        request.level,
        request.path,
        request.compat,
        request.thinkingLevelMap,
        request.modelPolicy,
        preparePayload,
      ),
    );
  }
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
      // The level map must be complete only for low/medium/high. Whether `off` can
      // disable reasoning belongs to the generation contract; `minimal`,
      // xhigh, and max also remain explicit-evidence-only extensions.
      if (
        (result.path === "chat" || result.path === "responses") &&
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
