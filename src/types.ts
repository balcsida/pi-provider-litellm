import type { Model } from "@earendil-works/pi-ai";

export type DiscoverySource = "model_info" | "models_list" | "health";

export type LiteLLMApi = "openai-completions" | "openai-responses";

export type LiteLLMRuntimeAuth = {
  baseUrl: string;
  apiKey: string;
  headers?: Record<string, string>;
  allowInsecureHttp?: boolean;
};

export interface LiteLLMModelPolicy {
  // Moonshot/Kimi's strict schema requires assistant tool calls to carry string
  // content. Discovery enables this outbound rewrite only from deployment evidence.
  normalizeStrictToolMessages: boolean;
  // Moonshot routes can inline reasoning as `<think>` text in the visible
  // answer. Whether to unwrap it is a per-model conclusion discovery reaches
  // from deployment evidence, carried here so the `message_end` hook does not
  // re-derive it from the route name.
  normalizeThinkTags: boolean;
  // Hide duplicate visible reasoning only for deployment-evidenced Kimi routes
  // that do not use an always-thinking generation.
  suppressReasoningVisibility: boolean;
  // Gemini-backed deployments require lowercase effort values. Discovery carries
  // this evidence so the request hook does not infer a backend from the route name.
  normalizeGeminiReasoningEffort?: boolean;
}

export type DiscoveredModel = Omit<Model<"openai-completions">, "provider" | "api" | "baseUrl"> & {
  api?: LiteLLMApi;
  litellmPolicy?: LiteLLMModelPolicy;
  // Persist the deployment's accepted Responses reasoning carrier so cached
  // models cannot regain catalog-derived selectors without the same evidence.
  litellmResponsesReasoningControl?: true;
};

export type LiteLLMModel = Model<LiteLLMApi> & {
  litellmPolicy?: LiteLLMModelPolicy;
  litellmResponsesReasoningControl?: true;
};

export interface DiscoveryResult {
  models: DiscoveredModel[];
  source: DiscoverySource;
}

export interface DiscoveryOptions {
  timeoutMs?: number;
  signal?: AbortSignal;
  headers?: Record<string, string>;
  allowInsecureHttp?: boolean;
}

export interface ModelInfoEntry {
  model_name?: string;
  litellm_params?: {
    model?: string;
    custom_llm_provider?: string;
    allowed_openai_params?: string[];
  };
  model_info?: {
    id?: string;
    mode?: string | null;
    litellm_provider?: string;
    base_model?: string;
    supported_openai_params?: string[];
    input_cost_per_token?: number;
    output_cost_per_token?: number;
    cache_read_input_token_cost?: number;
    cache_creation_input_token_cost?: number;
    max_input_tokens?: number;
    max_output_tokens?: number;
    supports_reasoning?: boolean;
    supports_none_reasoning_effort?: boolean;
    supports_minimal_reasoning_effort?: boolean;
    supports_low_reasoning_effort?: boolean;
    supports_medium_reasoning_effort?: boolean;
    supports_high_reasoning_effort?: boolean;
    supports_xhigh_reasoning_effort?: boolean;
    supports_max_reasoning_effort?: boolean;
    supports_vision?: boolean;
  };
}

export interface ModelInfoResponse {
  data?: ModelInfoEntry[];
}

export interface HealthModelEntry {
  model?: string;
  model_id?: string;
  api_base?: string;
}

export interface HealthResponse {
  healthy_endpoints?: HealthModelEntry[];
}

export interface ModelsListEntry {
  id?: string;
  owned_by?: string;
}

export interface ModelsListResponse {
  data?: ModelsListEntry[];
}

export type AuthFileEntry =
  | { type: "oauth"; access: string; refresh: string; expires: number; baseUrl?: string }
  | { type: "api_key"; key: string };

export interface ResolvedCredentials {
  baseUrl?: string;
  apiKey?: string;
  apiKeyConfig?: string;
  // `apiKey` was minted from Google ADC rather than config, helper, or env.
  apiKeyFromGcloudAdc?: boolean;
}

export interface LiteLLMMcpTool {
  name: string;
  server_name: string;
  server_id?: string;
  description: string;
  input_schema: Record<string, unknown>;
}

export interface LiteLLMSkill {
  id?: string;
  name: string;
  description?: string;
  enabled?: boolean;
  source?: Record<string, unknown>;
  version?: string;
  keywords?: string[];
  domain?: string;
  namespace?: string;
  category?: string;
  author?: string;
  homepage?: string;
  input_schema?: Record<string, unknown>;
  code?: string;
}
