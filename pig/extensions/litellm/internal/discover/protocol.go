// Ports src/discover.ts (protocol selection, compat, request policy and family evidence)
package discover

import (
	"math"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/backend"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/modelgroups"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/proxyversion"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

// Go's RE2 has no lookahead; where the TypeScript patterns end in `(?=$|[...])`, the groups below consume
// the character instead. Every use is a plain match test or reads only earlier groups, so the match
// outcome is identical.
var (
	// Matches both the conventional `anthropic/...` prefix and aliases that LiteLLM deployments
	// commonly assign to Anthropic-backed routes (e.g. `google/claude-sonnet-4-6`, `opus-4.7`,
	// `sonnet-4.6`, `haiku-4.5`). Without the `cacheControlFormat: "anthropic"` flag, pi never relays
	// cache_control markers through the proxy, so prompt caching silently no-ops on Claude models.
	anthropicModelPattern      = regexp.MustCompile(`(?i)(?:^|[-_/.:])(?:anthropic/|(?:claude|fable|opus|sonnet|haiku)(?:$|[-_/.:]))`)
	moonshotModelPattern       = regexp.MustCompile(`(?i)^(moonshotai/|moonshot/|kimi[-/])`)
	forcedThinkingModelPattern = regexp.MustCompile(`(?i)(?:^|[-/])thinking(?:[-/]|$)`)
	// GPT ids appear bare, prefixed, or dated (`gpt-6-sol`, `llm-gateway/gpt-5.5`,
	// `gpt-5.5-20260504143601`); match them all so the tool+reasoning workaround survives renames.
	// `gpt-4o` and `gpt-oss` carry no generation number here.
	gptGenerationPattern = regexp.MustCompile(`(?i)(?:^|/)gpt-(\d+)(?:\.(\d+))?(?:$|[-.])`)
)

// genericTransportAdapters are LiteLLM adapters that describe transport, not the backend vendor.
var genericTransportAdapters = map[string]bool{
	"azure":                  true,
	"azure_ai":               true,
	"custom_openai":          true,
	"openai":                 true,
	"openai_like":            true,
	"text-completion-openai": true,
}

// azureAIResponsesFloor: before v1.103.0 LiteLLM has no native Responses config for the `azure_ai`
// provider, so every azure_ai deployment reaches `/v1/responses` through the Chat Completions bridge.
// That bridge crashes on Azure's empty-`choices` stream chunks with "list index out of range"
// (BerriAI/litellm#34455), while the same deployment streams fine on `/v1/chat/completions`. v1.103.0
// guards the bridge and serves azure_ai natively where the host allows it.
var azureAIResponsesFloor = proxyversion.Floor{1, 103, 0}

var azureAIModelPrefix = regexp.MustCompile(`^azure_ai/`)
var azureModelPrefix = regexp.MustCompile(`^azure(?:_ai)?/`)

func boolPtr(value bool) *bool { return &value }

func lowerTrim(value string) string { return strings.ToLower(strings.TrimSpace(value)) }

// IsMoonshotModel is isMoonshotModel.
func IsMoonshotModel(modelID string) bool {
	return moonshotModelPattern.MatchString(modelID)
}

// gptGeneration returns the major and minor generation of a GPT id, ok false when it has none.
func gptGeneration(modelID string) (major, minor int, ok bool) {
	match := gptGenerationPattern.FindStringSubmatch(modelID)
	if match == nil {
		return 0, 0, false
	}
	number := func(digits string) int {
		if digits == "" {
			return 0
		}
		parsed, err := strconv.Atoi(digits)
		if err != nil {
			return math.MaxInt32
		}
		return parsed
	}
	return number(match[1]), number(match[2]), true
}

// IsGpt55OrNewerModel: GPT-5.5 and later reject `reasoning_effort` alongside function tools on
// /v1/chat/completions ("use /v1/responses").
func IsGpt55OrNewerModel(modelID string) bool {
	major, minor, ok := gptGeneration(modelID)
	return ok && (major > 5 || (major == 5 && minor >= 5))
}

// IsGpt6OrNewerModel: GPT-6 also rejects an omitted effort there: it falls back to its default effort.
func IsGpt6OrNewerModel(modelID string) bool {
	major, _, ok := gptGeneration(modelID)
	return ok && major >= 6
}

func shouldSuppressReasoningContent(modelID string) bool {
	return IsMoonshotModel(modelID) && !forcedThinkingModelPattern.MatchString(modelID)
}

// EmitsThinkTags is emitsThinkTags.
func EmitsThinkTags(modelID string) bool {
	return shouldSuppressReasoningContent(modelID)
}

// ResponsesCompat is responsesCompat. Pi's Responses transport has no cacheControlFormat setting and
// uses Responses-native prompt-cache fields instead of Anthropic cache_control markers.
func ResponsesCompat(modelID string) *ai.ModelCompat {
	if IsMoonshotModel(modelID) {
		return &ai.ModelCompat{SupportsDeveloperRole: boolPtr(false)}
	}
	return nil
}

// CompletionsCompat is completionsCompat; an empty semanticFamily is `undefined`.
func CompletionsCompat(modelID string, semanticFamily modelgroups.FamilyEvidence) *ai.ModelCompat {
	// Conflicting deployment evidence must not re-enable route-name inference.
	if semanticFamily == modelgroups.FamilyConflicting {
		return &ai.ModelCompat{SupportsStore: boolPtr(false)}
	}
	if semanticFamily == modelgroups.SemanticFamilyKimi || (semanticFamily == "" && IsMoonshotModel(modelID)) {
		return &ai.ModelCompat{
			SupportsStore:           boolPtr(false),
			SupportsDeveloperRole:   boolPtr(false),
			SupportsReasoningEffort: boolPtr(false),
			SupportsStrictMode:      boolPtr(false),
			MaxTokensField:          "max_tokens",
		}
	}
	if semanticFamily == modelgroups.SemanticFamilyClaude || (semanticFamily == "" && anthropicModelPattern.MatchString(modelID)) {
		return &ai.ModelCompat{SupportsStore: boolPtr(false), CacheControlFormat: "anthropic"}
	}
	return &ai.ModelCompat{SupportsStore: boolPtr(false)}
}

// BuildCompat is buildCompat, retained as the completions alias.
func BuildCompat(modelID string) *ai.ModelCompat {
	return CompletionsCompat(modelID, "")
}

func isAzureAIDeployment(entry types.ModelInfoEntry) bool {
	adapter, configuredModel := "", ""
	if entry.LiteLLMParams != nil {
		adapter = lowerTrim(entry.LiteLLMParams.CustomLLMProvider)
		configuredModel = lowerTrim(entry.LiteLLMParams.Model)
	}
	return adapter == "azure_ai" || azureAIModelPrefix.MatchString(configuredModel)
}

// listsResponsesEndpoint: the second result is false when the entry carries no endpoint list.
func listsResponsesEndpoint(entry types.ModelInfoEntry) (lists bool, known bool) {
	if entry.ModelInfo == nil || entry.ModelInfo.SupportedEndpoints == nil {
		return false, false
	}
	return slices.Contains(entry.ModelInfo.SupportedEndpoints, "/v1/responses"), true
}

func entryMode(entry types.ModelInfoEntry) modelgroups.Mode {
	if entry.ModelInfo == nil {
		return modelgroups.NormalizedMode(nil)
	}
	return modelgroups.NormalizedMode(entry.ModelInfo.Mode)
}

// transportAwaitsProxyVersion: an azure_ai `supported_endpoints` list is copied from the model cost
// map and describes the model, not the transport LiteLLM uses, so on its own it cannot authorize
// Responses. These are the only deployments whose transport the proxy version decides; discovery asks
// for the version only when one is published.
func transportAwaitsProxyVersion(entry types.ModelInfoEntry) bool {
	lists, known := listsResponsesEndpoint(entry)
	return isAzureAIDeployment(entry) && known && lists && entryMode(entry) != modelgroups.ModeResponses
}

func supportsResponses(entry types.ModelInfoEntry, proxyVersion *proxyversion.Version) bool {
	lists, known := listsResponsesEndpoint(entry)
	if known && !lists {
		return false
	}
	if entryMode(entry) == modelgroups.ModeResponses {
		return true
	}
	if transportAwaitsProxyVersion(entry) && !proxyversion.AtLeast(proxyVersion, azureAIResponsesFloor) {
		return false
	}
	if known && lists {
		return true
	}

	identity := backend.ResolveIdentity(entry)
	if identity == nil || identity.Family != backend.FamilyOpenAI {
		return false
	}
	adapter, configuredModel, reportedProvider := "", "", ""
	if entry.LiteLLMParams != nil {
		adapter = lowerTrim(entry.LiteLLMParams.CustomLLMProvider)
		configuredModel = lowerTrim(entry.LiteLLMParams.Model)
	}
	if entry.ModelInfo != nil {
		reportedProvider = lowerTrim(entry.ModelInfo.LiteLLMProvider)
	}
	azureAdapter := adapter == "azure" ||
		adapter == "azure_ai" ||
		azureModelPrefix.MatchString(configuredModel) ||
		reportedProvider == "azure" ||
		reportedProvider == "azure_ai"
	// An Azure API version describes the API surface, not whether this deployment serves Responses.
	// Require the explicit mode/endpoint evidence above instead of promoting a working Chat route to an
	// unavailable endpoint. Generic adapters remain eligible for LiteLLM's Responses-to-Chat bridge.
	return !azureAdapter
}

// ModelProtocol is modelProtocol. modeOrEntry mirrors the TypeScript union: nil, a mode string or
// *string, or a types.ModelInfoEntry (or pointer to one), whose model_name is ignored.
func ModelProtocol(modelID string, modeOrEntry any, proxyVersion *proxyversion.Version) types.ModelProtocol {
	var selected types.ModelInfoEntry
	switch value := modeOrEntry.(type) {
	case types.ModelInfoEntry:
		selected = value
		selected.ModelName = ""
	case *types.ModelInfoEntry:
		selected = *value
		selected.ModelName = ""
	case string:
		selected = types.ModelInfoEntry{ModelName: modelID, ModelInfo: &types.ModelInfoDetails{Mode: &value}}
	case *string:
		selected = types.ModelInfoEntry{ModelName: modelID, ModelInfo: &types.ModelInfoDetails{Mode: value}}
	default:
		selected = types.ModelInfoEntry{ModelName: modelID, ModelInfo: &types.ModelInfoDetails{}}
	}
	if supportsResponses(selected, proxyVersion) {
		return types.ModelProtocol{API: ai.APIOpenAIResponses, Compat: ResponsesCompat(modelID)}
	}
	return types.ModelProtocol{API: ai.APIOpenAICompletions, Compat: CompletionsCompat(modelID, "")}
}

// MoonshotPolicy is moonshotPolicy.
func MoonshotPolicy(modelID string, strictToolRepair bool) types.LiteLLMModelPolicy {
	return types.LiteLLMModelPolicy{
		NormalizeStrictToolMessages: strictToolRepair,
		NormalizeThinkTags:          !forcedThinkingModelPattern.MatchString(modelID),
		// Route-only fallback cannot authorize request-side reasoning suppression.
		SuppressReasoningVisibility: false,
	}
}

func requestPolicy(
	modelID string,
	deploymentFamilies []modelgroups.FamilyEvidence,
	normalizeThinkTags, suppressReasoningVisibility, strictToolRepair bool,
) *types.LiteLLMModelPolicy {
	// A contradiction within one deployment is evidence against applying any family-specific request
	// policy; do not discard it like a missing label.
	if slices.Contains(deploymentFamilies, modelgroups.FamilyConflicting) {
		return nil
	}
	var evidenced []modelgroups.FamilyEvidence
	for _, family := range deploymentFamilies {
		if family != "" {
			evidenced = append(evidenced, family)
		}
	}
	allGemini := true
	for _, family := range evidenced {
		allGemini = allGemini && family == modelgroups.SemanticFamilyGemini
	}
	if len(evidenced) == len(deploymentFamilies) && allGemini {
		return &types.LiteLLMModelPolicy{NormalizeGeminiReasoningEffort: true}
	}
	if slices.Contains(evidenced, modelgroups.SemanticFamilyKimi) {
		return &types.LiteLLMModelPolicy{
			NormalizeStrictToolMessages: strictToolRepair,
			NormalizeThinkTags:          normalizeThinkTags,
			SuppressReasoningVisibility: suppressReasoningVisibility,
		}
	}
	if len(evidenced) == 0 && IsMoonshotModel(modelID) {
		policy := MoonshotPolicy(modelID, false)
		return &policy
	}
	return nil
}

func semanticFamilyOf(id string) modelgroups.SemanticFamily {
	identity := backend.ResolveIdentity(types.ModelInfoEntry{LiteLLMParams: &types.ModelInfoParams{Model: id}})
	if identity == nil {
		return ""
	}
	return modelgroups.SemanticFamily(identity.Family)
}

var (
	kimiK25K26Pattern  = regexp.MustCompile(`(?:^|[./_-])kimi[-_/]?k?2[._-]?[56](?:$|[./_:-])`)
	kimiK27CodePattern = regexp.MustCompile(`(?:^|[./_-])kimi[-_/]?k?2[._-]?7[./_-]?(?:code|highspeed)(?:$|[./_:-])`)
	kimiK3Pattern      = regexp.MustCompile(`(?:^|[./_-])kimi[-_/]?k?3(?:$|[./_:-])`)
	deepseekV4Pattern  = regexp.MustCompile(`(?:^|[./_-])deepseek[-_/]?v?4(?:$|[./_:-])`)
)

func semanticModelOf(id string) modelgroups.SemanticModel {
	identity := backend.ResolveIdentity(types.ModelInfoEntry{LiteLLMParams: &types.ModelInfoParams{Model: id}})
	value := ""
	if identity != nil {
		value = strings.ToLower(identity.ModelID)
	}
	switch {
	case kimiK25K26Pattern.MatchString(value):
		return modelgroups.SemanticModelKimiK25K26
	case kimiK27CodePattern.MatchString(value):
		return modelgroups.SemanticModelKimiK27Code
	case kimiK3Pattern.MatchString(value):
		return modelgroups.SemanticModelKimiK3
	case deepseekV4Pattern.MatchString(value):
		return modelgroups.SemanticModelDeepSeekV4
	}
	return ""
}

// deploymentFamily is deploymentFamily; the empty result is `undefined`.
func deploymentFamily(entry types.ModelInfoEntry) modelgroups.FamilyEvidence {
	model, customProvider, baseModel, litellmProvider := "", "", "", ""
	if entry.LiteLLMParams != nil {
		model, customProvider = strings.TrimSpace(entry.LiteLLMParams.Model), strings.TrimSpace(entry.LiteLLMParams.CustomLLMProvider)
	}
	if entry.ModelInfo != nil {
		baseModel, litellmProvider = strings.TrimSpace(entry.ModelInfo.BaseModel), lowerTrim(entry.ModelInfo.LiteLLMProvider)
	}
	withoutName := entry
	withoutName.ModelName = ""
	if (model != "" || baseModel != "") && backend.ResolveIdentity(withoutName) == nil {
		return modelgroups.FamilyConflicting
	}
	var identityFamilies []modelgroups.SemanticFamily
	for _, candidate := range []string{model, customProvider, baseModel} {
		if candidate == "" {
			continue
		}
		if family := semanticFamilyOf(candidate); family != "" {
			identityFamilies = append(identityFamilies, family)
		}
	}
	distinct := map[modelgroups.SemanticFamily]bool{}
	for _, family := range identityFamilies {
		distinct[family] = true
	}
	if len(distinct) > 1 {
		return modelgroups.FamilyConflicting
	}

	var identityFamily modelgroups.SemanticFamily
	if len(identityFamilies) > 0 {
		identityFamily = identityFamilies[0]
	}
	adapter := litellmProvider
	var adapterFamily modelgroups.SemanticFamily
	if adapter != "" {
		if genericTransportAdapters[adapter] {
			adapterFamily = modelgroups.SemanticFamilyOpenAI
		} else {
			adapterFamily = semanticFamilyOf(adapter)
		}
	}
	if identityFamily == "" {
		return adapterFamily
	}
	// OpenAI-compatible and Azure adapters describe transport, not the backend vendor. They remain
	// useful fallback evidence for opaque identities, but must not contradict a model/base identity
	// such as `openai/kimi-k2.5`.
	if adapterFamily == "" || (adapter != "" && genericTransportAdapters[adapter]) {
		return identityFamily
	}
	if adapterFamily == identityFamily {
		return identityFamily
	}
	return modelgroups.FamilyConflicting
}

func hasReadableBackendEvidence(entry types.ModelInfoEntry) bool {
	if entry.LiteLLMParams != nil &&
		(strings.TrimSpace(entry.LiteLLMParams.Model) != "" || strings.TrimSpace(entry.LiteLLMParams.CustomLLMProvider) != "") {
		return true
	}
	return entry.ModelInfo != nil &&
		(strings.TrimSpace(entry.ModelInfo.BaseModel) != "" || strings.TrimSpace(entry.ModelInfo.LiteLLMProvider) != "")
}

func hasMoonshotCompatEvidence(compat *ai.ModelCompat) bool {
	return compat != nil && compat.MaxTokensField == "max_tokens" &&
		compat.SupportsStrictMode != nil && !*compat.SupportsStrictMode
}
