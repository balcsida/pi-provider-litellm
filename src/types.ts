import type { Model } from "@earendil-works/pi-ai";

export type DiscoverySource = "model_info" | "models_list" | "health";

export type LiteLLMApi = "openai-completions" | "openai-responses";

export type LiteLLMRuntimeAuth = {
  baseUrl: string;
  apiKey: string;
  headers?: Record<string, string>;
  allowInsecureHttp?: boolean;
};

export type DiscoveredModel = Omit<Model<"openai-completions">, "provider" | "api" | "baseUrl"> & {
  api?: LiteLLMApi;
  suppressReasoningContent?: boolean;
};

export type LiteLLMModel = Model<LiteLLMApi> & { suppressReasoningContent?: boolean };

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
  };
  model_info?: {
    id?: string;
    mode?: string | null;
    litellm_provider?: string;
    base_model?: string;
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
