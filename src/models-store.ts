/**
 * Reads Pi's persisted catalog, so that a start does not re-prove what Pi already kept.
 *
 * The factory discovers before it returns because Pi's startup refresh runs with `allowNetwork:false` and returns before the
 * network phase, so a provider that waits for Pi has no models when the session picks one (issue #137). That holds on a first
 * run. It does not hold once Pi has persisted the catalog: with discovery switched off and a populated `models-store.json`,
 * Pi offers the stored models and a request works, measured at a 5 ms factory against 1.8 s to discover.
 *
 * So the seed reads the stored catalog and hands those models over, instead of discovering them again. It only reads, and
 * only `models-store.json`, which this repository's AGENTS.md assigns to Pi. Nothing here writes a model cache; the removed
 * `litellm-models.json` is what that rule is about.
 *
 * The models are returned rather than an empty list on purpose. Pi restores its own catalog, so an empty list would do in
 * the ordinary case. It would not do when Pi's active store is somewhere else: `ModelRuntime` takes a `modelsStorePath`, and
 * resolves the default beside whatever `modelsPath` it was given. Every caller inside Pi passes an agent-directory path, but
 * an SDK embedder need not. A catalog left in the default file would then satisfy this check while Pi's active store was
 * empty, and the provider would have no models at all -- the very fault the seed exists to prevent. Returning the models
 * costs nothing and cannot fail that way.
 *
 * Four things make a stored entry unusable, and all are checked here rather than left to Pi:
 *
 *   - Age. Only a network refresh writes the store, and `pi -p` never runs one, so without a limit print mode would keep
 *     whatever catalog an interactive session last left. Pi's own remote catalogs skip the network for 4 hours after
 *     `checkedAt`, and so does this.
 *   - A different proxy. Stored models carry the `baseUrl` they were discovered from, so a changed root must still discover.
 *   - An older `litellmDiscoveryVersion`. Pi keeps a same-version stored entry over what startup discovery just proved
 *     (AGENTS.md), so skipping discovery for an older entry would hide the metadata a new version adds.
 *   - An entry Pi cannot restore. Pi's no-network restore publishes the stored models over the seed, so the seed cannot keep
 *     a short entry out of a session; it decides the catalog only when that restore fails. The restore runs
 *     `enrichCachedModel` (provider.ts), and when that throws Pi restores nothing, so the seed makes the same call and
 *     discovers instead.
 *
 * What is deliberately not checked is the credential. Nothing in Pi's store records which key discovered it, and a key scopes
 * which models a LiteLLM proxy shows. Within those 4 hours, a key swapped on the same proxy, or a model added to it, therefore
 * shows the stored catalog until Pi's own network refresh runs: interactive and RPC sessions run one in the background,
 * `pi -p` does not. Recording a key fingerprint would mean writing a file of our own, which is the thing the rule forbids.
 */

import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { LITELLM_DISCOVERY_VERSION } from "./backend-identity.js";
import { enrichCachedModel } from "./discover.js";
import { LITELLM_IMAGES_API, type LiteLLMImageModel } from "./images.js";
import { isLiteLLMApi, resolveModelBaseUrl } from "./protocols.js";
import { imageBaseUrl } from "./provider.js";
import type { LiteLLMModel } from "./types.js";

/**
 * Where Pi keeps it. `FileModelsStore` defaults to `<agentDir>/models-store.json`, and `ModelRuntime` resolves it beside
 * whatever `modelsPath` it was given -- `modelsPath` itself is the model *config* (`models.json`), a different file with a
 * different shape, which is deliberately not read here.
 */
const STORE_FILENAME = "models-store.json";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Pi's own remote catalogs skip the network for 4 hours after `checkedAt` (`REMOTE_CATALOG_REFRESH_INTERVAL_MS`). */
const STORE_FRESH_MS = 4 * 60 * 60 * 1000;

function isImageModel(value: Record<string, unknown>): boolean {
  return value.type === "image" && value.api === LITELLM_IMAGES_API;
}

function isUsableModel(value: unknown, provider: string, baseUrl: string, allowInsecureHttp?: boolean): boolean {
  if (!isRecord(value) || value.provider !== provider) return false;
  if (value.litellmDiscoveryVersion !== LITELLM_DISCOVERY_VERSION) return false;
  try {
    // A stored baseUrl is per protocol -- `/v1` for the OpenAI APIs and images, the bare root for Messages -- so the root
    // goes through the same resolution rather than being compared to it.
    if (isImageModel(value)) return value.baseUrl === imageBaseUrl(baseUrl, allowInsecureHttp);
    // `api` indexes LITELLM_PROTOCOLS in resolveModelBaseUrl, so an unknown one is a TypeError, not a bad model.
    return isLiteLLMApi(value.api) && value.baseUrl === resolveModelBaseUrl(baseUrl, value.api, allowInsecureHttp);
  } catch {
    return false;
  }
}

async function readStore(agentDir: string): Promise<Record<string, unknown> | undefined> {
  try {
    const parsed: unknown = JSON.parse(await readFile(join(agentDir, STORE_FILENAME), "utf8"));
    return isRecord(parsed) ? parsed : undefined;
  } catch {
    // Absent, unreadable, or not JSON: discover instead.
    return undefined;
  }
}

/**
 * Pi's stored models for `provider`, when every one of them can be used as it is, and `undefined` otherwise. Never throws;
 * anything it cannot read means "discover".
 */
export async function readStoredModels(options: {
  agentDir: string;
  provider: string;
  baseUrl: string;
  allowInsecureHttp?: boolean;
}): Promise<(LiteLLMModel | LiteLLMImageModel)[] | undefined> {
  const entry = (await readStore(options.agentDir))?.[options.provider];
  if (!isRecord(entry) || !Array.isArray(entry.models) || entry.models.length === 0) return undefined;
  if (typeof entry.checkedAt !== "number" || Date.now() - entry.checkedAt >= STORE_FRESH_MS) return undefined;
  const usable = entry.models.every((model) =>
    isUsableModel(model, options.provider, options.baseUrl, options.allowInsecureHttp),
  );
  if (!usable) return undefined;
  try {
    // Pi's restore path (provider.ts) runs the same call; when it throws, nothing is restored and the seed is the catalog.
    // Image models carry no request policy, so that path restores them as stored.
    return entry.models.map((model) => (isImageModel(model) ? model : enrichCachedModel(model))) as (
      | LiteLLMModel
      | LiteLLMImageModel
    )[];
  } catch {
    return undefined;
  }
}
