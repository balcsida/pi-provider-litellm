// Ports src/backend-identity.ts
package backend

import (
	"regexp"
	"strings"
)

// DiscoveryVersion is LITELLM_DISCOVERY_VERSION.
const DiscoveryVersion = 6

// Family is BackendFamily; the empty string means no family.
type Family string

const (
	FamilyClaude   Family = "claude"
	FamilyDeepSeek Family = "deepseek"
	FamilyGemini   Family = "gemini"
	FamilyKimi     Family = "kimi"
	FamilyOpenAI   Family = "openai"
)

// Row is BackendIdentityRow. Empty strings stand for absent fields.
type Row struct {
	ModelName     string
	LiteLLMParams RowParams
	ModelInfo     RowInfo
}

type RowParams struct {
	Model             string
	CustomLLMProvider string
}

type RowInfo struct {
	BaseModel       string
	LiteLLMProvider string
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
	// openai is deliberately absent: LiteLLM uses that provider for any OpenAI-compatible server.
	providerFamilies = map[string]Family{
		"anthropic":  FamilyClaude,
		"deepseek":   FamilyDeepSeek,
		"gemini":     FamilyGemini,
		"moonshot":   FamilyKimi,
		"moonshotai": FamilyKimi,
	}
	// Fireworks' account-scoped ids ("accounts/fireworks/models/x") are model path, not a provider.
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
func ResolveIdentity(row Row) *Identity {
	raw := firstNonEmpty(wireString(row.ModelInfo.BaseModel), wireString(row.LiteLLMParams.Model), wireString(row.ModelName))
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
	// custom_llm_provider is provider evidence in its own right: a differing prefix is a
	// conflict unless it is a known model-path segment.
	customProvider := strings.ToLower(wireString(row.LiteLLMParams.CustomLLMProvider))
	if genericAdapters[customProvider] {
		customProvider = ""
	}
	if prefixProvider != "" && customProvider != "" && prefixProvider != customProvider {
		if !modelPathSegments[prefixProvider] {
			return nil
		}
		prefixProvider = ""
		slash = -1
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
func ResolveCatalogProvider(row Row) string {
	identity := ResolveIdentity(row)
	if identity != nil && identity.Provider != "" && !genericCatalogProviders[identity.Provider] {
		return identity.Provider
	}
	reported := strings.ToLower(wireString(row.ModelInfo.LiteLLMProvider))
	custom := strings.ToLower(wireString(row.LiteLLMParams.CustomLLMProvider))
	if custom != "" && reported != "" && genericCatalogProviders[reported] {
		return custom
	}
	if identity != nil && identity.Provider != "" {
		return identity.Provider
	}
	return firstNonEmpty(reported, custom)
}

// RoutesOnlyThrough is routesOnlyThrough: every declared routing signal must name an allowed provider.
func RoutesOnlyThrough(row Row, providers map[string]bool) bool {
	model := wireString(row.LiteLLMParams.Model)
	var routing []string
	if custom := strings.ToLower(strings.TrimSpace(wireString(row.LiteLLMParams.CustomLLMProvider))); custom != "" {
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
