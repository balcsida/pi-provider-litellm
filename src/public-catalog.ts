import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import type { Api, Model } from "@earendil-works/pi-ai";
import { getModels, getProviders } from "@earendil-works/pi-ai/compat";
import type { BuiltinProvider } from "@earendil-works/pi-ai/providers/all";
import { isRecord } from "./backend-identity.js";

export const MODELS_DEV_URL = "https://models.dev/api.json";
const MODELS_DEV_CACHE_TTL_MS = 28 * 24 * 60 * 60 * 1000;
const DEFAULT_TIMEOUT_MS = 5000;
const KNOWN_PROVIDERS = new Set<string>(getProviders());

export interface PublicCatalogRecord {
  source: "models.dev" | "pi-vendor" | "pi-adapter";
  provider: string;
  modelId: string;
  limits?: { context?: number; output?: number };
  cost?: { input?: number; output?: number; cacheRead?: number; cacheWrite?: number };
  modalities?: Array<"text" | "image">;
  effortLevels?: string[];
  thinkingLevelMap?: Record<string, unknown>;
}

export interface PublicCatalog {
  lookup(provider: string | undefined, id: string): PublicCatalogRecord | undefined;
}

interface ModelsDevModel {
  modalities?: { input?: string[] };
  limit?: { context?: number; input?: number; output?: number };
  cost?: { input?: number; output?: number; cache_read?: number; cache_write?: number };
  reasoning_options?: { type?: string; values?: string[] } | Array<{ type?: string; values?: string[] }>;
}

type ModelsDevCatalog = Record<string, { models?: Record<string, ModelsDevModel> }>;
interface CacheFile {
  fetchedAt: number;
  catalog: ModelsDevCatalog;
}

export interface LoadPublicCatalogOptions {
  cachePath?: string;
  offline?: boolean;
  timeoutMs?: number;
  signal?: AbortSignal;
  fetch?: typeof fetch;
}

const memoryCaches = new Map<string, CacheFile>();
const refreshes = new Map<string, Promise<ModelsDevCatalog | undefined>>();

const PROVIDER_ALIASES: Readonly<Record<string, readonly string[]>> = {
  anthropic: ["anthropic"],
  azure: ["azure", "openai"],
  azure_ai: ["azure", "openai"],
  bedrock: ["amazon-bedrock"],
  bedrock_converse: ["amazon-bedrock"],
  deepseek: ["deepseek"],
  fireworks: ["fireworks-ai"],
  fireworks_ai: ["fireworks-ai"],
  gemini: ["google"],
  moonshot: ["moonshotai"],
  nvidia_nim: ["nvidia"],
  openai: ["openai"],
  together_ai: ["together"],
  vertex_ai: ["google-vertex"],
};

const PI_PROVIDER_ALIASES: Readonly<Record<string, readonly string[]>> = {
  "amazon-bedrock": ["amazon-bedrock"],
  azure: ["azure-openai-responses"],
  "fireworks-ai": ["fireworks"],
  openai: ["openai"],
};

function finite(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function normalizeCatalog(value: unknown): ModelsDevCatalog | undefined {
  if (!isRecord(value)) return undefined;
  const result: ModelsDevCatalog = {};
  for (const [provider, rawProvider] of Object.entries(value)) {
    if (!isRecord(rawProvider) || !isRecord(rawProvider.models)) continue;
    const models: Record<string, ModelsDevModel> = {};
    for (const [id, rawModel] of Object.entries(rawProvider.models)) {
      if (!isRecord(rawModel)) continue;
      models[id] = rawModel as ModelsDevModel;
    }
    result[provider] = { models };
  }
  return result;
}

async function writeJsonAtomic(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const temporaryPath = `${path}.${process.pid}.${Date.now()}.tmp`;
  await writeFile(temporaryPath, JSON.stringify(value, null, 2), "utf8");
  await rename(temporaryPath, path);
}

async function readCache(path: string): Promise<CacheFile | undefined> {
  try {
    const parsed: unknown = JSON.parse(await readFile(path, "utf8"));
    if (!isRecord(parsed)) return undefined;
    const fetchedAt = finite(parsed.fetchedAt);
    if (fetchedAt === undefined || fetchedAt < 0 || fetchedAt > Date.now()) return undefined;
    const catalog = normalizeCatalog(parsed.catalog);
    return catalog ? { fetchedAt, catalog } : undefined;
  } catch {
    return undefined;
  }
}

async function refreshCatalog(key: string, options: LoadPublicCatalogOptions): Promise<ModelsDevCatalog | undefined> {
  const active = refreshes.get(key);
  if (active) return active;
  const refresh = (async () => {
    try {
      const fetchImpl = options.fetch ?? globalThis.fetch;
      const timeout = AbortSignal.timeout(options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
      const response = await fetchImpl(MODELS_DEV_URL, {
        headers: { Accept: "application/json" },
        signal: options.signal ? AbortSignal.any([options.signal, timeout]) : timeout,
      });
      if (!response.ok) return undefined;
      const catalog = normalizeCatalog(await response.json());
      if (!catalog) return undefined;
      const cache = { fetchedAt: Date.now(), catalog };
      memoryCaches.set(key, cache);
      if (options.cachePath) await writeJsonAtomic(options.cachePath, cache).catch(() => undefined);
      return catalog;
    } catch {
      return undefined;
    } finally {
      refreshes.delete(key);
    }
  })();
  refreshes.set(key, refresh);
  return refresh;
}

function providerCandidates(provider: string | undefined): string[] {
  if (!provider) return [];
  const normalized = provider.trim().toLowerCase();
  return [...new Set(PROVIDER_ALIASES[normalized] ?? [normalized])];
}

function lookupIds(id: string): string[] {
  const result = new Set([id]);
  if (id.includes("/")) result.add(id.slice(id.indexOf("/") + 1));
  return [...result];
}

function mapModelsDev(provider: string, modelId: string, model: ModelsDevModel): PublicCatalogRecord {
  const context = finite(model.limit?.context) ?? finite(model.limit?.input);
  const output = finite(model.limit?.output);
  const input = finite(model.cost?.input);
  const outputCost = finite(model.cost?.output);
  const cacheRead = finite(model.cost?.cache_read);
  const cacheWrite = finite(model.cost?.cache_write);
  const options = Array.isArray(model.reasoning_options) ? model.reasoning_options : [model.reasoning_options];
  const effortLevels = options
    .filter((option) => option?.type === "effort")
    .flatMap((option) => option?.values ?? [])
    .filter((value): value is string => typeof value === "string");
  return {
    source: "models.dev",
    provider,
    modelId,
    ...(context !== undefined || output !== undefined ? { limits: { context, output } } : {}),
    ...(input !== undefined || outputCost !== undefined || cacheRead !== undefined || cacheWrite !== undefined
      ? { cost: { input, output: outputCost, cacheRead, cacheWrite } }
      : {}),
    ...(model.modalities?.input
      ? { modalities: model.modalities.input.includes("image") ? ["text", "image"] : ["text"] }
      : {}),
    ...(effortLevels.length > 0 ? { effortLevels } : {}),
  };
}

function toPiRecord(source: "pi-vendor" | "pi-adapter", provider: string, model: Model<Api>): PublicCatalogRecord {
  const thinkingLevelMap = model.thinkingLevelMap as Record<string, unknown> | undefined;
  const effortLevels = thinkingLevelMap
    ? Object.entries(thinkingLevelMap)
        .filter(([, value]) => value !== null)
        .map(([level]) => level)
    : [];
  return {
    source,
    provider,
    modelId: model.id,
    limits: { context: model.contextWindow, output: model.maxTokens },
    cost: {
      input: model.cost.input,
      output: model.cost.output,
      cacheRead: model.cost.cacheRead,
      cacheWrite: model.cost.cacheWrite,
    },
    modalities: model.input.includes("image") ? ["text", "image"] : ["text"],
    ...(effortLevels.length > 0 ? { effortLevels } : {}),
    ...(thinkingLevelMap ? { thinkingLevelMap } : {}),
  };
}

function findPiModel(provider: string, ids: readonly string[]): Model<Api> | undefined {
  if (!KNOWN_PROVIDERS.has(provider)) return undefined;
  const models = getModels(provider as BuiltinProvider);
  return models.find((model) => ids.includes(model.id) || ids.includes(`${provider}/${model.id}`));
}

export async function loadPublicCatalog(options: LoadPublicCatalogOptions = {}): Promise<PublicCatalog> {
  const key = options.cachePath ?? MODELS_DEV_URL;
  let cache = memoryCaches.get(key);
  if (!cache && options.cachePath) {
    cache = await readCache(options.cachePath);
    if (cache) memoryCaches.set(key, cache);
  }
  const offline = options.offline ?? process.env.LITELLM_OFFLINE === "1";
  let catalog = cache?.catalog;
  if (!offline) {
    if (!cache) catalog = await refreshCatalog(key, options);
    else if (Date.now() - cache.fetchedAt >= MODELS_DEV_CACHE_TTL_MS) void refreshCatalog(key, options);
  }

  return {
    lookup(provider, id) {
      const providers = providerCandidates(provider);
      const ids = lookupIds(id);
      for (const candidate of providers) {
        const models = catalog?.[candidate]?.models;
        for (const modelId of ids) {
          const model = models?.[modelId];
          if (model) return mapModelsDev(candidate, modelId, model);
        }
      }
      const vendor = providers.find((candidate) => candidate !== "azure") ?? providers[0];
      if (vendor) {
        for (const piProvider of PI_PROVIDER_ALIASES[vendor] ?? [vendor]) {
          const model = findPiModel(piProvider, ids);
          if (model) return toPiRecord("pi-vendor", piProvider, model);
        }
      }
      const adapter = provider?.trim().toLowerCase();
      if (adapter) {
        for (const piProvider of PI_PROVIDER_ALIASES[adapter] ?? [adapter]) {
          const model = findPiModel(piProvider, ids);
          if (model) return toPiRecord("pi-adapter", piProvider, model);
        }
      }
      return undefined;
    },
  };
}
