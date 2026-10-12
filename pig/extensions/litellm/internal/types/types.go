// Ports src/types.ts
package types

import (
	"time"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/proxyversion"
)

// DiscoverySource is where a discovery result came from.
type DiscoverySource string

const (
	SourceModelInfo  DiscoverySource = "model_info"
	SourceModelsList DiscoverySource = "models_list"
	SourceHealth     DiscoverySource = "health"
)

// LiteLLMApi is one of the three protocols a LiteLLM route can speak.
type LiteLLMApi string

const (
	APIAnthropicMessages LiteLLMApi = "anthropic-messages"
	APIOpenAICompletions LiteLLMApi = "openai-completions"
	APIOpenAIResponses   LiteLLMApi = "openai-responses"
)

// DiscoveryVersion is LITELLM_DISCOVERY_VERSION.
const DiscoveryVersion = 6

// BackendFamily is a backend vendor family; the empty string means no family.
type BackendFamily string

const (
	FamilyClaude   BackendFamily = "claude"
	FamilyDeepSeek BackendFamily = "deepseek"
	FamilyGemini   BackendFamily = "gemini"
	FamilyKimi     BackendFamily = "kimi"
	FamilyOpenAI   BackendFamily = "openai"
)

type LiteLLMRuntimeAuth struct {
	BaseURL           string            `json:"baseUrl"`
	APIKey            string            `json:"-"`
	Headers           map[string]string `json:"headers,omitempty"`
	AllowInsecureHTTP bool              `json:"allowInsecureHttp,omitempty"`
}

// LiteLLMModelPolicy scopes request and response behavior to backend evidence. Optional TypeScript
// booleans are plain omitempty bools: absent and false mean the same to every reader.
type LiteLLMModelPolicy struct {
	// NormalizeStrictToolMessages: Moonshot/Kimi requires assistant tool calls to carry string content.
	NormalizeStrictToolMessages bool `json:"normalizeStrictToolMessages"`
	// NormalizeThinkTags unwraps `<think>` text Moonshot routes inline in the visible answer.
	NormalizeThinkTags bool `json:"normalizeThinkTags"`
	// SuppressReasoningVisibility hides duplicate visible reasoning on evidenced Kimi routes.
	SuppressReasoningVisibility bool `json:"suppressReasoningVisibility"`
	// NormalizeGeminiReasoningEffort: Gemini-backed deployments require lowercase effort values.
	NormalizeGeminiReasoningEffort bool `json:"normalizeGeminiReasoningEffort,omitempty"`
	// DropToolReasoning: GPT-5.5+ rejects reasoning_effort alongside function tools on Chat Completions.
	DropToolReasoning bool `json:"dropToolReasoning,omitempty"`
	// ExplicitToolReasoningOff: GPT-6 falls back to its default effort when the field is omitted.
	ExplicitToolReasoningOff bool `json:"explicitToolReasoningOff,omitempty"`
}

// DiscoveredModel mirrors PiG's modelsCatalogRecord (ai/models_catalog_codec.go) plus the
// extension fields, which do not survive the host's closed-struct round trip.
type DiscoveredModel struct {
	ID                            string                           `json:"id"`
	Name                          string                           `json:"name"`
	API                           ai.API                           `json:"api"`
	Provider                      string                           `json:"provider,omitempty"`
	BaseURL                       string                           `json:"baseUrl,omitempty"`
	Reasoning                     bool                             `json:"reasoning"`
	ThinkingLevelMap              ai.ThinkingLevelMap              `json:"thinkingLevelMap,omitempty"`
	Input                         []string                         `json:"input"`
	InputLimits                   *ai.ModelInputLimits             `json:"inputLimits,omitempty"`
	Cost                          ai.ModelCost                     `json:"cost"`
	PromptCache                   ai.ModelPromptCache              `json:"promptCache,omitempty"`
	ContextWindow                 int                              `json:"contextWindow"`
	MaxTokens                     int                              `json:"maxTokens"`
	SamplingParams                map[string]any                   `json:"samplingParams,omitempty"`
	SamplingParamsByThinkingLevel ai.SamplingParamsByThinkingLevel `json:"samplingParamsByThinkingLevel,omitempty"`
	Headers                       map[string]string                `json:"headers,omitempty"`
	Compat                        *ai.ModelCompat                  `json:"compat,omitempty"`

	SuppressReasoningContent         *bool               `json:"suppressReasoningContent,omitempty"`
	LiteLLMPolicy                    *LiteLLMModelPolicy `json:"litellmPolicy,omitempty"`
	LiteLLMResponsesReasoningControl bool                `json:"litellmResponsesReasoningControl,omitempty"`
	LiteLLMBackendFamily             BackendFamily       `json:"litellmBackendFamily,omitempty"`
	// LiteLLMDiscoveryVersion is DiscoveryVersion when set.
	LiteLLMDiscoveryVersion int `json:"litellmDiscoveryVersion,omitempty"`
}

// ModelProtocol is the protocol-bearing part of a discovered model.
type ModelProtocol struct {
	API    ai.API          `json:"api"`
	Compat *ai.ModelCompat `json:"compat,omitempty"`
}

type DiscoveryResult struct {
	Models []DiscoveredModel `json:"models"`
	Source DiscoverySource   `json:"source"`
	// ProxyVersion is set only when a /model/info discovery had to read it and the proxy
	// answered. Nil means unknown, not old.
	ProxyVersion *proxyversion.Version `json:"proxyVersion,omitempty"`
}

// DiscoveryOptions: nil pointers are TypeScript `undefined`. It carries no context: AbortSignal
// is the ctx first parameter of every discovery function.
type DiscoveryOptions struct {
	// Timeout nil means the default; a pointer to zero disables network discovery.
	Timeout           *time.Duration
	Headers           map[string]string
	AllowInsecureHTTP bool
	// ModelsDev false reads only an existing cache at ModelsDevCachePath; with no cache path,
	// models.dev is not consulted at all.
	ModelsDev          *bool
	ModelsDevCachePath string
}

type ModelInfoParams struct {
	Model               string   `json:"model,omitempty"`
	CustomLLMProvider   string   `json:"custom_llm_provider,omitempty"`
	APIVersion          string   `json:"api_version,omitempty"`
	AllowedOpenAIParams []string `json:"allowed_openai_params,omitempty"`
}

type ModelInfoDetails struct {
	ID                             string   `json:"id,omitempty"`
	Mode                           *string  `json:"mode,omitempty"`
	BaseModel                      string   `json:"base_model,omitempty"`
	LiteLLMProvider                string   `json:"litellm_provider,omitempty"`
	SupportedEndpoints             []string `json:"supported_endpoints,omitempty"`
	SupportedOpenAIParams          []string `json:"supported_openai_params,omitempty"`
	InputCostPerToken              *float64 `json:"input_cost_per_token,omitempty"`
	OutputCostPerToken             *float64 `json:"output_cost_per_token,omitempty"`
	CacheReadInputTokenCost        *float64 `json:"cache_read_input_token_cost,omitempty"`
	CacheCreationInputTokenCost    *float64 `json:"cache_creation_input_token_cost,omitempty"`
	MaxInputTokens                 *float64 `json:"max_input_tokens,omitempty"`
	MaxOutputTokens                *float64 `json:"max_output_tokens,omitempty"`
	SupportsReasoning              *bool    `json:"supports_reasoning,omitempty"`
	ReasoningEffortLevels          []string `json:"reasoning_effort_levels,omitempty"`
	SupportsNoneReasoningEffort    *bool    `json:"supports_none_reasoning_effort,omitempty"`
	SupportsMinimalReasoningEffort *bool    `json:"supports_minimal_reasoning_effort,omitempty"`
	SupportsLowReasoningEffort     *bool    `json:"supports_low_reasoning_effort,omitempty"`
	SupportsMediumReasoningEffort  *bool    `json:"supports_medium_reasoning_effort,omitempty"`
	SupportsHighReasoningEffort    *bool    `json:"supports_high_reasoning_effort,omitempty"`
	SupportsXHighReasoningEffort   *bool    `json:"supports_xhigh_reasoning_effort,omitempty"`
	SupportsMaxReasoningEffort     *bool    `json:"supports_max_reasoning_effort,omitempty"`
	SupportsVision                 *bool    `json:"supports_vision,omitempty"`
}

type ModelInfoEntry struct {
	ModelName     string            `json:"model_name,omitempty"`
	LiteLLMParams *ModelInfoParams  `json:"litellm_params,omitempty"`
	ModelInfo     *ModelInfoDetails `json:"model_info,omitempty"`
}

type ModelInfoResponse struct {
	Data []ModelInfoEntry `json:"data,omitempty"`
}

type HealthModelEntry struct {
	Model   string `json:"model,omitempty"`
	ModelID string `json:"model_id,omitempty"`
	APIBase string `json:"api_base,omitempty"`
}

type HealthResponse struct {
	HealthyEndpoints []HealthModelEntry `json:"healthy_endpoints,omitempty"`
}

type ModelsListEntry struct {
	ID      string `json:"id,omitempty"`
	OwnedBy string `json:"owned_by,omitempty"`
}

type ModelsListResponse struct {
	Data []ModelsListEntry `json:"data,omitempty"`
}

const (
	AuthTypeOAuth  = "oauth"
	AuthTypeAPIKey = "api_key"
)

// AuthFileEntry is an auth.json entry: Type "oauth" uses Access/Refresh/Expires/BaseURL, "api_key" uses Key.
type AuthFileEntry struct {
	Type    string `json:"type"`
	Access  string `json:"access,omitempty"`
	Refresh string `json:"refresh,omitempty"`
	Expires *int64 `json:"expires,omitempty"`
	BaseURL string `json:"baseUrl,omitempty"`
	Key     string `json:"key,omitempty"`
}

type ResolvedCredentials struct {
	BaseURL      string `json:"baseUrl,omitempty"`
	APIKey       string `json:"-"`
	APIKeyConfig string `json:"apiKeyConfig,omitempty"`
	// APIKeyFromGcloudADC: APIKey was minted from Google ADC rather than config, helper, or env.
	APIKeyFromGcloudADC bool `json:"apiKeyFromGcloudAdc,omitempty"`
}

type LiteLLMSkill struct {
	ID          string         `json:"id,omitempty"`
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Enabled     *bool          `json:"enabled,omitempty"`
	Source      map[string]any `json:"source,omitempty"`
	Version     string         `json:"version,omitempty"`
	Keywords    []string       `json:"keywords,omitempty"`
	Domain      string         `json:"domain,omitempty"`
	Namespace   string         `json:"namespace,omitempty"`
	Category    string         `json:"category,omitempty"`
	Author      string         `json:"author,omitempty"`
	Homepage    string         `json:"homepage,omitempty"`
	InputSchema map[string]any `json:"input_schema,omitempty"`
	Code        string         `json:"code,omitempty"`
}
