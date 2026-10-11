/**
 * Reads Pi's persisted catalog, so that a start does not re-prove what Pi already kept.
 *
 * The factory discovers before it returns because Pi's startup refresh runs with `allowNetwork:false` and returns before the
 * network phase, so a provider that waits for Pi has no models when the session picks one (issue #137). That holds on a first
 * run. It does not hold once Pi has persisted the catalog: with discovery switched off and a populated `models-store.json`,
 * Pi offers the stored models and a request works, measured at a 5 ms factory against 1.8 s to discover.
 *
 * So the seed reads the stored catalog and hands those models over, instead of discovering them again. It only reads, and
 * only the file Pi names in AGENTS.md. Nothing here writes a model cache; the removed `litellm-models.json` is what that
 * rule is about.
 *
 * The models are returned rather than an empty list on purpose. Pi restores its own catalog, so an empty list would do in
 * the ordinary case. It would not do when Pi's active store is somewhere else: `ModelRuntime` takes a `modelsStorePath`, and
 * resolves the default beside whatever `modelsPath` it was given. Every caller inside Pi passes an agent-directory path, but
 * an SDK embedder need not. A catalog left in the default file would then satisfy this check while Pi's active store was
 * empty, and the provider would have no models at all -- the very fault the seed exists to prevent. Returning the models
 * costs nothing and cannot fail that way.
 *
 * Three things make a stored entry unusable, and all are checked here rather than left to Pi:
 *
 *   - A different proxy. Stored models carry the `baseUrl` they were discovered from, so a changed root must still discover.
 *   - An older `litellmDiscoveryVersion`. Pi keeps a same-version stored entry over what startup discovery just proved
 *     (AGENTS.md), so skipping discovery for an older entry would hide the metadata a new version adds.
 *   - A field a model needs to be used. These models go straight to Pi, so a short entry must discover, not reach a session.
 *
 * What is deliberately not checked is the credential. Nothing in Pi's store records which key discovered it, and a key scopes
 * which models a LiteLLM proxy shows. A key swapped on the same proxy therefore shows the stored catalog until Pi's own
 * network refresh runs. That is the behaviour Pi already gives every provider, and recording a key fingerprint would mean
 * writing a file of our own, which is the thing the rule forbids.
 */

import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { LITELLM_DISCOVERY_VERSION } from "./backend-identity.js";
import { isLiteLLMApi, resolveModelBaseUrl } from "./protocols.js";
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

const isNumber = (value: unknown): boolean => typeof value === "number" && Number.isFinite(value);

/** The cost rates Pi expects on every model. */
function hasCost(value: unknown): boolean {
  return (
    isRecord(value) &&
    isNumber(value.input) &&
    isNumber(value.output) &&
    isNumber(value.cacheRead) &&
    isNumber(value.cacheWrite)
  );
}

/**
 * Whether a stored entry can be handed to Pi as it is. The fields are the required ones of `Model`: anything short of them
 * would reach a session as a model that cannot be used, so it discovers instead.
 */
function isUsableModel(value: unknown, provider: string, baseUrl: string, allowInsecureHttp?: boolean): boolean {
  if (!isRecord(value)) return false;
  if (typeof value.id !== "string" || value.id.length === 0) return false;
  if (typeof value.name !== "string" || value.name.length === 0) return false;
  if (value.provider !== provider) return false;
  if (typeof value.reasoning !== "boolean") return false;
  if (!isNumber(value.contextWindow) || !isNumber(value.maxTokens)) return false;
  if (!Array.isArray(value.input) || value.input.length === 0) return false;
  if (!hasCost(value.cost)) return false;
  if (value.litellmDiscoveryVersion !== LITELLM_DISCOVERY_VERSION) return false;
  // `api` indexes LITELLM_PROTOCOLS in resolveModelBaseUrl, so an unknown one is a TypeError, not a bad model.
  if (!isLiteLLMApi(value.api)) return false;
  try {
    // A stored baseUrl is per protocol -- `/v1` for the OpenAI APIs, the bare root for Messages -- so the root goes through
    // the same resolution rather than being compared to it.
    return value.baseUrl === resolveModelBaseUrl(baseUrl, value.api, allowInsecureHttp);
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
}): Promise<LiteLLMModel[] | undefined> {
  const entry = (await readStore(options.agentDir))?.[options.provider];
  if (!isRecord(entry) || !Array.isArray(entry.models) || entry.models.length === 0) return undefined;
  const usable = entry.models.every((model) =>
    isUsableModel(model, options.provider, options.baseUrl, options.allowInsecureHttp),
  );
  return usable ? (entry.models as LiteLLMModel[]) : undefined;
}
