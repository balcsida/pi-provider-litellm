// Ports src/backend-identity.ts
package backend

import (
	"regexp"
	"strings"

	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

// DiscoveryVersion is LITELLM_DISCOVERY_VERSION.
const DiscoveryVersion = types.DiscoveryVersion

// Family is BackendFamily; the empty string means no family.
type Family = types.BackendFamily

const (
	FamilyClaude   = types.FamilyClaude
	FamilyDeepSeek = types.FamilyDeepSeek
	FamilyGemini   = types.FamilyGemini
	FamilyKimi     = types.FamilyKimi
	FamilyOpenAI   = types.FamilyOpenAI
)

// fields flattens a /model/info entry into the strings backend identity reads; absent is "".
type fields struct{ modelName, model, customProvider, baseModel, litellmProvider string }

func fieldsOf(entry types.ModelInfoEntry) fields {
	f := fields{modelName: entry.ModelName}
	if entry.LiteLLMParams != nil {
		f.model, f.customProvider = entry.LiteLLMParams.Model, entry.LiteLLMParams.CustomLLMProvider
	}
	if entry.ModelInfo != nil {
		f.baseModel, f.litellmProvider = entry.ModelInfo.BaseModel, entry.ModelInfo.LiteLLMProvider
	}
	return f
}

// Identity is BackendIdentity; Provider and Family are empty when absent.
type Identity struct {
	Provider    string
	ModelID     string
	QualifiedID string
	Family      Family
}

var (
	genericAdapters         = set("azure", "azure_ai", "custom_openai", "openai_like")
	genericCatalogProviders = set("azure", "azure_ai", "custom_openai", "openai_like", "openai", "text-completion-openai")
	// LiteLLM's openai provider is an OpenAI-compatible transport, so a model or base_model id
	// there may namespace itself by vendor without contradicting the routing provider.
	openAITransportProviders = set("openai", "text-completion-openai")
	// openai is deliberately absent: LiteLLM uses that provider for any OpenAI-compatible server.
	providerFamilies = map[string]Family{
		"anthropic":  FamilyClaude,
		"deepseek":   FamilyDeepSeek,
		"gemini":     FamilyGemini,
		"moonshot":   FamilyKimi,
		"moonshotai": FamilyKimi,
	}
	// Fireworks' account-scoped ids ("accounts/fireworks/models/x") are model path, not a provider.
	// Only base_model and the route name need it, as LiteLLM never routes them.
	modelPathSegments = set("accounts")

	claudePattern   = regexp.MustCompile(`(?:^|[./_-])(?:anthropic|claude|opus|sonnet|haiku|fable)(?:$|[./_:-])`)
	kimiPattern     = regexp.MustCompile(`(?:^|[./_-])(?:moonshotai|moonshot|kimi)(?:$|[./_:-])`)
	deepseekPattern = regexp.MustCompile(`(?:^|[./_-])deepseek(?:$|[./_:-])`)
	geminiPattern   = regexp.MustCompile(`(?:^|[./_-])gemini(?:$|[./_:-])`)
	openAIPattern   = regexp.MustCompile(`(?i)(?:^(?:ft:)?|[./_-])(?:openai|gpt|codex)(?:$|[./_:-])|(?:^(?:ft:)?|/)o\d(?:$|[./_:-])`)
)

func set(values ...string) map[string]bool {
	out := make(map[string]bool, len(values))
	for _, v := range values {
		out[v] = true
	}
	return out
}

// wireString trims and treats "" and "undefined" as absent.
func wireString(value string) string {
	trimmed := strings.TrimSpace(value)
	if trimmed == "undefined" {
		return ""
	}
	return trimmed
}

func semanticFamily(id string) Family {
	value := strings.ToLower(id)
	switch {
	case claudePattern.MatchString(value):
		return FamilyClaude
	case kimiPattern.MatchString(value):
		return FamilyKimi
	case deepseekPattern.MatchString(value):
		return FamilyDeepSeek
	case geminiPattern.MatchString(value):
		return FamilyGemini
	case IsOpenAIBackend(value):
		return FamilyOpenAI
	}
	return ""
}

// IsOpenAIBackend reports whether the id spells an OpenAI-family model.
func IsOpenAIBackend(id string) bool {
	return openAIPattern.MatchString(id)
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" {
			return v
		}
	}
	return ""
}

// ResolveIdentity is resolveBackendIdentity; nil means the identity is withheld.
func ResolveIdentity(entry types.ModelInfoEntry) *Identity {
	row := fieldsOf(entry)
	baseModel, configuredModel := wireString(row.baseModel), wireString(row.model)
	raw := firstNonEmpty(baseModel, configuredModel, wireString(row.modelName))
	if raw == "" {
		return nil
	}

	slash := strings.Index(raw, "/")
	prefixProvider := ""
	if slash > 0 {
		prefix := strings.ToLower(strings.TrimSpace(raw[:slash]))
		if !genericAdapters[prefix] {
			prefixProvider = prefix
		}
	}
	// custom_llm_provider is how LiteLLM routes a model, and it prepends itself to a configured
	// model whose first segment differs (_get_llm_provider). That segment is model namespace,
	// so resolve that prefixed form: OpenRouter's "openai/gpt-4o", vLLM's "Qwen/Qwen3-32B".
	// base_model and the route name are not routed, so there a differing prefix is a conflict.
	// A generic adapter prefix ("azure_ai/") is not a namespace and stays stripped.
	customProvider := strings.ToLower(wireString(row.customProvider))
	if genericAdapters[customProvider] {
		customProvider = ""
	}
	if prefixProvider != "" && customProvider != "" && prefixProvider != customProvider {
		// An OpenAI-compatible transport says nothing about the backend, so a differing first
		// segment on a base_model is a vendor namespace there too.
		namespacesOverTransport := openAITransportProviders[customProvider]
		if (baseModel != "" || configuredModel == "") && !modelPathSegments[prefixProvider] && !namespacesOverTransport {
			return nil
		}
		raw = customProvider + "/" + raw
		slash = len(customProvider)
		prefixProvider = customProvider
	}
	provider := firstNonEmpty(prefixProvider, customProvider)
	modelID := raw
	if slash > 0 {
		modelID = raw[slash+1:]
	}
	// Scan the full string only when the prefix is itself a known vendor name.
	scanTarget := modelID
	if _, known := providerFamilies[provider]; provider != "" && known {
		scanTarget = raw
	}
	family := semanticFamily(scanTarget)
	if family == "" && provider != "" {
		family = providerFamilies[provider]
	}
	qualified := modelID
	if provider != "" {
		qualified = provider + "/" + modelID
	}
	return &Identity{Provider: provider, ModelID: modelID, QualifiedID: qualified, Family: family}
}

// ResolveCatalogProvider is resolveCatalogProvider; "" means undefined.
func ResolveCatalogProvider(entry types.ModelInfoEntry) string {
	row := fieldsOf(entry)
	identity := ResolveIdentity(entry)
	if identity != nil && identity.Provider != "" && !genericCatalogProviders[identity.Provider] {
		return identity.Provider
	}
	reported := strings.ToLower(wireString(row.litellmProvider))
	custom := strings.ToLower(wireString(row.customProvider))
	if custom != "" && reported != "" && genericCatalogProviders[reported] {
		return custom
	}
	if identity != nil && identity.Provider != "" {
		return identity.Provider
	}
	return firstNonEmpty(reported, custom)
}

// RoutesOnlyThrough is routesOnlyThrough: every declared routing signal must name an allowed provider.
func RoutesOnlyThrough(entry types.ModelInfoEntry, providers map[string]bool) bool {
	row := fieldsOf(entry)
	model := wireString(row.model)
	var routing []string
	if custom := strings.ToLower(strings.TrimSpace(wireString(row.customProvider))); custom != "" {
		routing = append(routing, custom)
	}
	if slash := strings.Index(model, "/"); slash > 0 {
		if prefix := strings.ToLower(strings.TrimSpace(model[:slash])); prefix != "" {
			routing = append(routing, prefix)
		}
	}
	if len(routing) == 0 {
		return false
	}
	for _, p := range routing {
		if !providers[p] {
			return false
		}
	}
	return true
}
