import type { ClassifierAnswer, ClassifierModel, ClassifierResult, ProviderClassifier } from "@earendil-works/pi-ai";
import { LITELLM_DISCOVERY_VERSION } from "./backend-identity.js";
import { errorDetail, parseUsage, perMillion, requestHeaders, requestSignal, responseCost } from "./images.js";
import { defaultContextWindow, wireString } from "./model-groups.js";
import type { ModelInfoEntry } from "./types.js";

/** Decision models through the proxy's System One `/v1/systemone` endpoint. */
export const LITELLM_DECISIONS_API = "litellm-decisions" as const;
export type LiteLLMDecisionsApi = typeof LITELLM_DECISIONS_API;

export type DiscoveredClassifierModel = Omit<ClassifierModel<LiteLLMDecisionsApi>, "provider" | "baseUrl"> & {
  litellmDiscoveryVersion?: typeof LITELLM_DISCOVERY_VERSION;
};

export type LiteLLMClassifierModel = ClassifierModel<LiteLLMDecisionsApi> & {
  litellmDiscoveryVersion?: typeof LITELLM_DISCOVERY_VERSION;
};

const DECISION_MODES = new Set(["evaluation", "decisions"]);
const DECISION_ENDPOINTS = new Set(["/v1/systemone", "/v1/decisions"]);

/**
 * LiteLLM's own rule: a declared `supported_endpoints` list decides alone, so a list without a
 * decision endpoint denies even with an evaluation mode; without a list, the mode decides.
 */
export function isDecisionDeployment(entry: ModelInfoEntry): boolean {
  const info = entry.model_info;
  const endpoints: unknown = info?.supported_endpoints;
  if (Array.isArray(endpoints)) {
    return endpoints.some((endpoint) => typeof endpoint === "string" && DECISION_ENDPOINTS.has(endpoint.trim()));
  }
  const mode = wireString(info?.mode);
  return mode !== undefined && DECISION_MODES.has(mode.trim().toLowerCase());
}

/**
 * Reduce one `/model/info` route group to a classifier model. Every deployment behind the route
 * must be a decision deployment: LiteLLM load-balances across them, so a mixed group could answer
 * from a non-decision deployment.
 */
export function mapFromClassifierModelInfoGroup(
  route: string,
  entries: readonly ModelInfoEntry[],
): DiscoveredClassifierModel | undefined {
  // A wildcard row is not addressable, and `/v1/models` expansion carries no mode evidence.
  if (route.includes("*")) return undefined;
  const deployments = entries.filter((entry) => wireString(entry.model_name) === route);
  if (deployments.length === 0) return undefined;
  if (!deployments.every(isDecisionDeployment)) return undefined;
  // Prices are estimates for Pi's usage display; the `x-litellm-response-cost` header replaces them.
  const rate = (read: (entry: ModelInfoEntry) => number | undefined): number => {
    const values = deployments.map(read);
    return values.every((value) => value !== undefined) ? Math.max(...(values as number[])) : 0;
  };
  // The smallest declared window; an undeclared one counts as the default so it never raises the result.
  const windows = deployments.map((entry) => entry.model_info?.max_input_tokens);
  const declared = windows.filter(
    (value): value is number => typeof value === "number" && Number.isFinite(value) && value > 0,
  );
  const contextWindow = Math.min(...declared, ...(declared.length < windows.length ? [defaultContextWindow()] : []));
  return {
    type: "classifier",
    id: route,
    name: route,
    api: LITELLM_DECISIONS_API,
    input: ["text"],
    cost: {
      input: rate((entry) => perMillion(entry.model_info?.input_cost_per_token)),
      output: rate((entry) => perMillion(entry.model_info?.output_cost_per_token)),
      cacheRead: 0,
      cacheWrite: 0,
    },
    contextWindow,
    litellmDiscoveryVersion: LITELLM_DISCOVERY_VERSION,
  };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

const isFiniteNumber = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);

function parseAnswer(model: string, id: string, type: "choice" | "score" | "bool", raw: unknown): ClassifierAnswer {
  const wireType = type === "bool" ? "noul" : type;
  if (!isRecord(raw) || raw.type !== wireType) {
    throw new Error(`LiteLLM classifier ${model} did not return a ${type} answer for ${id}`);
  }
  const invalid = (field: string): Error =>
    new Error(`LiteLLM classifier ${model} returned an invalid ${field} for ${id}`);
  if (type === "bool") {
    if (!isFiniteNumber(raw.noul)) throw invalid("noul");
    return { type: "bool", probability: raw.noul };
  }
  if (!isFiniteNumber(raw.confidence)) throw invalid("confidence");
  if (type === "score") {
    if (!isFiniteNumber(raw.score)) throw invalid("score");
    return { type: "score", score: raw.score, confidence: raw.confidence };
  }
  if (typeof raw.choice !== "string") throw invalid("choice");
  if (!isRecord(raw.probabilities) || !Object.values(raw.probabilities).every(isFiniteNumber)) {
    throw invalid("probabilities");
  }
  return {
    type: "choice",
    choice: raw.choice,
    probabilities: raw.probabilities as Record<string, number>,
    confidence: raw.confidence,
  };
}

/** Classify through LiteLLM's `/v1/systemone`. Never rejects. */
export const classify: ProviderClassifier["classify"] = async (model, context, options) => {
  const result: ClassifierResult = {
    api: model.api,
    provider: model.provider,
    model: model.id,
    answers: {},
    stopReason: "stop",
    timestamp: Date.now(),
  };
  try {
    const apiKey = options?.apiKey;
    if (!apiKey) throw new Error(`No API key for provider: ${model.provider}`);
    if (context.images?.length) {
      throw new Error(`LiteLLM classifier ${model.id} takes text state only; images are not sent to /v1/systemone`);
    }
    // System One calls Pi's `bool` question `noul`; the fields are otherwise the same.
    const questions = Object.fromEntries(
      Object.entries(context.questions).map(([id, question]) => [
        id,
        question.type === "bool" ? { ...question, type: "noul" } : question,
      ]),
    );
    let params: Record<string, unknown> = { model: model.id, state: context.state, questions };
    const nextParams = await options?.onPayload?.(params, model);
    if (nextParams !== undefined) params = nextParams as Record<string, unknown>;
    const response = await (options?.fetch ?? fetch)(`${model.baseUrl}/systemone`, {
      method: "POST",
      headers: requestHeaders(apiKey, model.headers, options?.headers),
      body: JSON.stringify(params),
      signal: requestSignal(options),
    });
    const responseHeaders = Object.fromEntries(response.headers.entries());
    await options?.onResponse?.({ status: response.status, headers: responseHeaders }, model);
    const body = await response.text();
    if (!response.ok) throw new Error(`LiteLLM decisions request failed (${response.status}): ${errorDetail(body)}`);
    // The proxy billed the request whatever the body holds; token counts are added once it parses.
    const cost = responseCost(responseHeaders);
    const billed = parseUsage(undefined, model.cost, cost);
    if (billed) result.usage = billed;
    const parsed: unknown = JSON.parse(body);
    if (!isRecord(parsed)) throw new Error(`LiteLLM classifier ${model.id} returned an unexpected response`);
    // A malformed answer set was still billed.
    const usage = parseUsage(isRecord(parsed.usage) ? parsed.usage : undefined, model.cost, cost);
    if (usage) result.usage = usage;
    const answers = isRecord(parsed.answers) ? parsed.answers : {};
    const parsedAnswers: Record<string, ClassifierAnswer> = {};
    for (const [id, question] of Object.entries(context.questions)) {
      parsedAnswers[id] = parseAnswer(model.id, id, question.type, answers[id]);
    }
    result.answers = parsedAnswers;
    return result;
  } catch (error) {
    result.answers = {};
    result.stopReason = options?.signal?.aborted ? "aborted" : "error";
    result.errorMessage = error instanceof Error ? error.message : String(error);
    return result;
  }
};

export function createLiteLLMClassifierApis(): Record<LiteLLMDecisionsApi, ProviderClassifier> {
  return { [LITELLM_DECISIONS_API]: { classify } };
}
