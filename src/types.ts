import type { ProviderModelConfig } from "@earendil-works/pi-coding-agent";

export type DiscoverySource = "model_info" | "models_list";

export interface CacheFile {
  baseUrl: string;
  apiKeyFingerprint: string;
  fetchedAt: number;
  source: DiscoverySource;
  models: ProviderModelConfig[];
}

export interface DiscoveryResult {
  models: ProviderModelConfig[];
  source: DiscoverySource;
}

export interface DiscoveryOptions {
  timeoutMs?: number;
  signal?: AbortSignal;
}

export interface ModelInfoEntry {
  model_name?: string;
  model_info?: {
    mode?: string;
    input_cost_per_token?: number;
    output_cost_per_token?: number;
    cache_read_input_token_cost?: number;
    cache_creation_input_token_cost?: number;
    max_input_tokens?: number;
    max_output_tokens?: number;
    supports_reasoning?: boolean;
    supports_vision?: boolean;
  };
}

export interface ModelInfoResponse {
  data?: ModelInfoEntry[];
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
  apiKeyFingerprint?: string;
}
