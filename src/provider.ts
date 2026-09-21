import {
  type ApiStreamOptions,
  type Context,
  type Credential,
  createProvider,
  type Model,
  type Provider,
  type ProviderAuth,
  type SimpleStreamOptions,
} from "@earendil-works/pi-ai";
import { LITELLM_DISCOVERY_VERSION } from "./backend-identity.js";
import { enrichCachedModel, normalizeBaseUrl, restoreCachedModelPolicy } from "./discover.js";
import { createLiteLLMProtocolApis, isLiteLLMApi, LITELLM_API_NAMES, resolveModelBaseUrl } from "./protocols.js";
import type { DiscoveredModel, DiscoveryResult, LiteLLMApi, LiteLLMModel } from "./types.js";

export type LiteLLMProviderOptions = {
  id: string;
  name: string;
  baseUrl: string;
  auth: ProviderAuth;
  /** Catalog discovered during activation. Pi's startup refresh never allows network access, so
   * without a seed the provider stays empty until the user opens /model. */
  models?: readonly LiteLLMModel[];
  allowInsecureHttp?: boolean;
  resolveCredentialRoot: (credential?: Credential, requestBaseUrl?: string, apiKey?: string) => string | undefined;
  discover(credential: Credential, signal?: AbortSignal): Promise<DiscoveryResult & { baseUrl?: string }>;
};

export function toNativeModels(
  provider: string,
  baseUrl: string,
  models: readonly DiscoveredModel[],
  allowInsecureHttp = false,
): LiteLLMModel[] {
  return models.map((model) => ({
    ...model,
    provider,
    baseUrl: resolveModelBaseUrl(baseUrl, model.api, allowInsecureHttp),
  }));
}

export const DEFAULT_LITELLM_BASE_URL = "https://litellm.example.com";
const CHAT_TOOL_CAP = 128;
const PLACEHOLDER_HOSTNAMES = new Set([canonicalHostname(new URL(DEFAULT_LITELLM_BASE_URL).hostname)]);

function canonicalHostname(hostname: string): string {
  return hostname.replace(/\.$/, "");
}

function refreshRequired(message: string): Error {
  return new Error(`${message}; a network refresh with a valid LiteLLM base URL is required`);
}

function proxyRoot(
  baseUrl: string,
  subject: string,
  allowInsecureHttp = false,
  verb = "has",
): { root: string; canonical: string; hostname: string } {
  try {
    const root = normalizeBaseUrl(baseUrl, allowInsecureHttp);
    const url = new URL(root);
    if (url.protocol !== "http:" && url.protocol !== "https:") throw new Error("unsupported protocol");
    return { root, canonical: url.href, hostname: url.hostname };
  } catch {
    throw refreshRequired(`${subject} ${verb} an invalid LiteLLM model URL`);
  }
}

function activeCredentialRoot(
  root: string,
  allowInsecureHttp = false,
): { root: string; canonical: string; hostname: string } {
  const active = proxyRoot(root, "Active credentials", allowInsecureHttp, "have");
  if (isPlaceholderHost(active.hostname)) {
    throw refreshRequired("Active credentials use a placeholder LiteLLM model host");
  }
  return active;
}

export function isPlaceholderHost(hostname: string): boolean {
  return PLACEHOLDER_HOSTNAMES.has(canonicalHostname(hostname));
}

function modelRootError(
  model: Model<LiteLLMApi>,
  active: { root: string; canonical: string; hostname: string },
  allowInsecureHttp = false,
): Error | undefined {
  if (!isLiteLLMApi(model.api)) {
    return new Error(
      `LiteLLM model ${model.id} declares unsupported protocol "${String(model.api)}"; ` +
        `set "api" to one of ${LITELLM_API_NAMES.join(", ")} in models.json`,
    );
  }
  let stored: { root: string; canonical: string; hostname: string };
  try {
    stored = proxyRoot(model.baseUrl, "Cached model", allowInsecureHttp);
  } catch (error) {
    return error instanceof Error ? error : refreshRequired("Cached model has an invalid LiteLLM model URL");
  }
  if (isPlaceholderHost(stored.hostname)) {
    return refreshRequired("Cached model uses a placeholder LiteLLM model host");
  }
  if (stored.canonical !== active.canonical) {
    return refreshRequired(
      `Cached model has stale LiteLLM model root ${stored.root}; active credentials use ${active.root}`,
    );
  }
}

function requestModel(
  provider: string,
  model: Model<LiteLLMApi>,
  context: Context,
  credentialRoot: string | undefined,
  allowInsecureHttp = false,
): Model<LiteLLMApi> {
  if (!credentialRoot) throw refreshRequired("Active credentials do not identify a LiteLLM model host");
  const active = activeCredentialRoot(credentialRoot, allowInsecureHttp);
  const error = modelRootError(model, active, allowInsecureHttp);
  if (error) throw error;
  if (
    model.api === "openai-completions" &&
    (model as LiteLLMModel).litellmBackendFamily === "openai" &&
    context.tools &&
    context.tools.length > CHAT_TOOL_CAP
  ) {
    throw new Error(
      `LiteLLM model ${model.id} uses Chat Completions with ${context.tools.length} tools, exceeding the ` +
        `${CHAT_TOOL_CAP}-tool cap; route this model via Responses or reduce enabled extensions`,
    );
  }
  return { ...model, provider, baseUrl: resolveModelBaseUrl(active.root, model.api, allowInsecureHttp) };
}

export function createLiteLLMProvider(options: LiteLLMProviderOptions): Provider<LiteLLMApi> {
  const reportedAvailabilityDiagnostics = new Set<string>();
  const reportUnavailable = (message: string): void => {
    if (reportedAvailabilityDiagnostics.has(message)) return;
    reportedAvailabilityDiagnostics.add(message);
    process.stderr.write(`LiteLLM (${options.id}): ${message}\n`);
  };
  const provider = createProvider<LiteLLMApi>({
    id: options.id,
    name: options.name,
    baseUrl: options.baseUrl,
    auth: options.auth,
    models: options.models ?? [],
    async fetchModels(context) {
      if (!context.credential) throw new Error("LiteLLM model discovery requires a credential");
      const result = await options.discover(context.credential, context.signal);
      return toNativeModels(options.id, result.baseUrl ?? options.baseUrl, result.models, options.allowInsecureHttp);
    },
    filterModels(models, credential) {
      let active: { root: string; canonical: string; hostname: string };
      try {
        const root = options.resolveCredentialRoot(credential);
        if (!root) {
          if (models.length > 0) {
            reportUnavailable(
              `${models.length} model(s) hidden because no LiteLLM base URL is configured; ` +
                "set LITELLM_BASE_URL or run /login litellm",
            );
          }
          return [];
        }
        active = activeCredentialRoot(root, options.allowInsecureHttp);
      } catch (error) {
        reportUnavailable(error instanceof Error ? error.message : String(error));
        return [];
      }
      const available: Model<LiteLLMApi>[] = [];
      for (const model of models) {
        const error = modelRootError(model, active, options.allowInsecureHttp);
        if (error) {
          reportUnavailable(error.message);
          continue;
        }
        available.push({
          ...model,
          baseUrl: resolveModelBaseUrl(active.root, model.api, options.allowInsecureHttp),
        });
      }
      return available;
    },
    api: createLiteLLMProtocolApis(),
  });
  const refreshModels = provider.refreshModels;
  const guardedProvider: Provider<LiteLLMApi> = {
    ...provider,
    stream: <T extends LiteLLMApi>(model: Model<T>, context: Context, requestOptions?: ApiStreamOptions<T>) =>
      provider.stream(
        requestModel(
          options.id,
          model,
          context,
          options.resolveCredentialRoot(undefined, requestOptions?.env?.LITELLM_BASE_URL, requestOptions?.apiKey),
          options.allowInsecureHttp,
        ),
        context,
        requestOptions,
      ),
    streamSimple: (model: Model<LiteLLMApi>, context: Context, requestOptions?: SimpleStreamOptions) =>
      provider.streamSimple(
        requestModel(
          options.id,
          model,
          context,
          options.resolveCredentialRoot(undefined, requestOptions?.env?.LITELLM_BASE_URL, requestOptions?.apiKey),
          options.allowInsecureHttp,
        ),
        context,
        requestOptions,
      ),
  };
  if (!refreshModels) return guardedProvider;
  const seedKey = (model: { id: string; baseUrl: string }) => `${model.id}\0${model.baseUrl}`;
  const seeded = new Map((options.models ?? []).map((model) => [seedKey(model), model]));
  return {
    ...guardedProvider,
    refreshModels: async (context) => {
      const storedModels = context.stored?.models ?? [];
      let legacyCount = 0;
      const models = storedModels.map((model) => {
        if ((model as LiteLLMModel).litellmDiscoveryVersion !== LITELLM_DISCOVERY_VERSION) {
          // Pi applies stored models over the seed, so a stale entry would otherwise mask what
          // startup discovery just proved for the same route.
          const fresh = seeded.get(seedKey(model));
          if (fresh) return fresh;
          legacyCount++;
          return restoreCachedModelPolicy(model);
        }
        return enrichCachedModel(model);
      });
      try {
        await refreshModels({
          ...context,
          // The current pi-ai provider has no freshness gate, but force is the documented refresh contract field.
          force: context.force || (context.allowNetwork && legacyCount > 0),
          stored: context.stored && { ...context.stored, models },
        });
      } catch (error) {
        if (context.allowNetwork && legacyCount > 0) {
          reportUnavailable(
            `keeping cached models whose discovery metadata version does not match ` +
              `${LITELLM_DISCOVERY_VERSION} because the required network refresh failed`,
          );
        }
        throw error;
      }
    },
  };
}
