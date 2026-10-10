// Ports src/discover.ts (Pi catalog lookup and public-catalog adaptation)
package discover

import (
	"context"
	"regexp"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/backend"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/catalog"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/modelgroups"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

const (
	defaultTimeout = 5 * time.Second
	// publicCatalogBudget bounds how long discovery waits for the models.dev catalog.
	publicCatalogBudget = 1000 * time.Millisecond
)

var publicCatalogProviderByFamily = map[types.BackendFamily]string{
	types.FamilyClaude:   "anthropic",
	types.FamilyDeepSeek: "deepseek",
	types.FamilyGemini:   "gemini",
	types.FamilyKimi:     "moonshot",
	types.FamilyOpenAI:   "openai",
}

var knownProviderSet = func() map[string]bool {
	known := map[string]bool{}
	for _, provider := range ai.ListProviders() {
		known[provider] = true
	}
	return known
}()

// adapterCatalogProviders maps LiteLLM adapters onto Pi catalog providers. An empty value is
// `undefined`: the PiG catalog does not list that provider.
var adapterCatalogProviders = map[string]string{
	"anthropic":                  "anthropic",
	"claude":                     "anthropic",
	"azure":                      toKnownProvider(catalog.PiAzureProvider),
	"azure_ai":                   toKnownProvider(catalog.PiAzureProvider),
	"bedrock":                    "amazon-bedrock",
	"bedrock_converse":           "amazon-bedrock",
	"chatgpt":                    "openai-codex",
	"deepseek":                   "deepseek",
	"fireworks-ai":               "fireworks",
	"fireworks_ai":               "fireworks",
	"gemini":                     "google",
	"kimi":                       "moonshotai",
	"moonshot":                   "moonshotai",
	"nvidia_nim":                 "nvidia",
	"openai":                     "openai",
	"together_ai":                "together",
	"vertex_ai":                  "google-vertex",
	"vertex_ai-anthropic_models": "google-vertex",
}

// toKnownProvider returns the lower-cased provider when Pi's catalog lists it, else "".
func toKnownProvider(provider string) string {
	normalized := lowerTrim(provider)
	if normalized != "" && knownProviderSet[normalized] {
		return normalized
	}
	return ""
}

func adapterCatalogProvider(adapter string) string {
	normalized := lowerTrim(adapter)
	if normalized == "" {
		return ""
	}
	if provider, ok := adapterCatalogProviders[normalized]; ok && provider != "" {
		return provider
	}
	return toKnownProvider(normalized)
}

var (
	providerModelsMu sync.Mutex
	providerModels   = map[string][]ai.GeneratedModel{}
)

// piProviderModels is getModels(provider): the generated catalog never changes, so each provider's
// slice is copied once.
func piProviderModels(provider string) []ai.GeneratedModel {
	providerModelsMu.Lock()
	defer providerModelsMu.Unlock()
	models, ok := providerModels[provider]
	if !ok {
		models = ai.ListModels(provider)
		providerModels[provider] = models
	}
	return models
}

// Anthropic recognition is derived from the single `catalogLookupIds` rule so a second alias pattern
// cannot drift away from it. Every Anthropic catalog id and every alias that maps onto one is
// canonicalized to a `claude-` lookup id, including single-number names and dated snapshots.
func catalogProviderCandidates(lookupIds []string, id, ownedBy string) []string {
	candidates := []string{toKnownProvider(ownedBy), toKnownProvider(strings.SplitN(id, "/", 2)[0])}
	if slices.ContainsFunc(lookupIds, func(lookupID string) bool { return strings.HasPrefix(lookupID, "claude-") }) {
		candidates = append(candidates, "anthropic")
	}
	var unique []string
	for _, provider := range candidates {
		if provider != "" && !slices.Contains(unique, provider) {
			unique = append(unique, provider)
		}
	}
	return unique
}

type catalogHit struct {
	provider string
	model    *ai.GeneratedModel
}

func resolveCatalogModel(id, ownedBy string) *catalogHit {
	lookupIds := catalogLookupIds(id)
	for _, provider := range catalogProviderCandidates(lookupIds, id, ownedBy) {
		if model := findCatalogModelInProvider(provider, lookupIds); model != nil {
			return &catalogHit{provider: provider, model: model}
		}
	}
	return nil
}

func findCatalogModel(id, ownedBy string) *ai.GeneratedModel {
	if hit := resolveCatalogModel(id, ownedBy); hit != nil {
		return hit.model
	}
	return nil
}

var (
	variantSuffixV     = regexp.MustCompile(`-v\d+(?::\d+)?$`)
	variantSuffixAt    = regexp.MustCompile(`@[a-z0-9-]+$`)
	variantSuffixColon = regexp.MustCompile(`:\d+$`)
	variantSuffixDate  = regexp.MustCompile(`-\d{8}$`)
	anthropicAliasForm = regexp.MustCompile(`^(?:claude-)?(opus|sonnet|haiku)-(\d+)-(\d+)$`)
	anthropicRoutedPre = regexp.MustCompile(`^(?:[a-z0-9-]+\.)*anthropic[./]`)
	claudeGeneration   = regexp.MustCompile(`^(claude-[a-z]+-\d+)-\d+$`)
)

func unprefixedID(normalized string) string {
	if index := strings.Index(normalized, "/"); index >= 0 {
		return normalized[index+1:]
	}
	return normalized
}

func lastSegment(normalized string) string {
	return normalized[strings.LastIndex(normalized, "/")+1:]
}

func uniqueNonEmpty(values []string) []string {
	var unique []string
	for _, value := range values {
		if value != "" && !slices.Contains(unique, value) {
			unique = append(unique, value)
		}
	}
	return unique
}

func undecoratedBackendIds(id string) []string {
	normalized := strings.ToLower(id)
	variants := func(candidate string) []string {
		undecorated := variantSuffixAt.ReplaceAllString(variantSuffixV.ReplaceAllString(candidate, ""), "")
		return []string{
			candidate,
			variantSuffixColon.ReplaceAllString(candidate, ""),
			undecorated,
			variantSuffixDate.ReplaceAllString(undecorated, ""),
		}
	}
	return uniqueNonEmpty(append(variants(unprefixedID(normalized)), variants(lastSegment(normalized))...))
}

func catalogLookupIds(id string) []string {
	normalized := strings.ToLower(id)
	lookupIds := []string{normalized}
	add := func(candidate string) {
		if !slices.Contains(lookupIds, candidate) {
			lookupIds = append(lookupIds, candidate)
		}
	}
	unprefixed := unprefixedID(normalized)
	add(unprefixed)
	for _, candidate := range undecoratedBackendIds(id) {
		add(candidate)
	}

	anthropicAlias := strings.ReplaceAll(unprefixed, ".", "-")
	if match := anthropicAliasForm.FindStringSubmatch(anthropicAlias); match != nil {
		add("claude-" + match[1] + "-" + match[2] + "-" + match[3])
	}
	if anthropicAlias == "fable-5" || anthropicAlias == "opus-5" {
		add("claude-" + anthropicAlias)
	}
	return lookupIds
}

func findCatalogModelInProvider(provider string, lookupIds []string) *ai.GeneratedModel {
	if provider == "" {
		return nil
	}
	models := piProviderModels(provider)
	for _, lookupID := range lookupIds {
		normalized := strings.ToLower(lookupID)
		for i := range models {
			if strings.ToLower(models[i].ID) == normalized {
				return &models[i]
			}
		}
		qualified := provider + "/" + normalized
		for i := range models {
			if strings.ToLower(models[i].ID) == qualified {
				return &models[i]
			}
		}
	}
	return nil
}

// messagesCompatOf: native Messages is safe only when Pi's Anthropic catalog supplies the serializer
// policy for that generation. Lookup tries the exact id first, then that generation's entry; provider
// adapter metadata is not a wire contract.
func messagesCompatOf(model *ai.GeneratedModel) *modelgroups.MessagesBackendCompat {
	if model.API != ai.APIAnthropicMessages {
		return nil
	}
	carried := &modelgroups.MessagesBackendCompat{}
	if model.Compat != nil {
		carried.ForceAdaptiveThinking = model.Compat.ForceAdaptiveThinking
		carried.SupportsTemperature = model.Compat.SupportsTemperature
	}
	return carried
}

var anthropicRoutes = map[string]bool{"anthropic": true}

// messagesStrictToolEvidence: the Anthropic catalog describes the Anthropic API, so only deployments
// whose declared routing names Anthropic keep its grant. A hosted route (Bedrock, Vertex) can reject
// tools[].strict for the same model, and LiteLLM reports no per-route strict-tool capability, so it
// stays unknown.
func messagesStrictToolEvidence(entry types.ModelInfoEntry, model *ai.GeneratedModel) bool {
	if !backend.RoutesOnlyThrough(entry, anthropicRoutes) || model.API != ai.APIAnthropicMessages || model.Compat == nil {
		return false
	}
	return model.Compat.SupportsStrictTools != nil && *model.Compat.SupportsStrictTools
}

func anthropicBackendLookupIds(id string) []string {
	routed := strings.ToLower(lastSegment(id))
	base := anthropicRoutedPre.ReplaceAllString(routed, "")
	lookupIds := undecoratedBackendIds(base)
	for _, candidate := range slices.Clone(lookupIds) {
		if match := claudeGeneration.FindStringSubmatch(candidate); match != nil && !slices.Contains(lookupIds, match[1]) {
			lookupIds = append(lookupIds, match[1])
		}
	}
	return lookupIds
}

// claudeCapableAdapters are the LiteLLM adapters whose native request path can terminate at a Claude
// backend. Other adapters may expose Claude-like public aliases, but an alias alone is not evidence
// that LiteLLM accepts the Anthropic Messages schema.
var claudeCapableAdapters = map[string]bool{
	"anthropic":                  true,
	"bedrock":                    true,
	"bedrock_converse":           true,
	"vertex_ai":                  true,
	"vertex_ai-anthropic_models": true,
}

var claudeModelPattern = regexp.MustCompile(`(?i)(?:^|[./_-])(?:claude|opus|sonnet|haiku|fable)(?:$|[./_:-])`)

// nativeMessagesCatalog returns the Messages fields of a CatalogResolution; ok false is the empty
// result.
func nativeMessagesCatalog(entry types.ModelInfoEntry) (resolution modelgroups.CatalogResolution, ok bool) {
	adapter := ""
	if entry.ModelInfo != nil {
		adapter = lowerTrim(entry.ModelInfo.LiteLLMProvider)
	}
	if adapter == "" || !claudeCapableAdapters[adapter] || deploymentFamily(entry) != modelgroups.SemanticFamilyClaude {
		return resolution, false
	}
	var candidates []string
	if entry.LiteLLMParams != nil {
		candidates = append(candidates, strings.TrimSpace(entry.LiteLLMParams.Model))
	}
	if entry.ModelInfo != nil {
		candidates = append(candidates, strings.TrimSpace(entry.ModelInfo.BaseModel))
	}
	candidates = slices.DeleteFunc(candidates, func(id string) bool { return id == "" })
	if len(candidates) == 0 || !slices.ContainsFunc(candidates, claudeModelPattern.MatchString) {
		return resolution, false
	}
	models := make([]*ai.GeneratedModel, len(candidates))
	for i, id := range candidates {
		models[i] = findCatalogModelInProvider("anthropic", anthropicBackendLookupIds(id))
		// Every declared backend must resolve to the same native serializer policy. An unresolved
		// identity cannot be discarded to make the remainder unanimous.
		if models[i] == nil || models[i].ID != models[0].ID {
			return resolution, false
		}
	}
	model := models[0]
	compat := messagesCompatOf(model)
	if compat == nil {
		return resolution, false
	}
	resolution.MessagesCompat = compat
	resolution.MessagesStrictTools = messagesStrictToolEvidence(entry, model)
	resolution.MessagesThinkingLevelMap = cloneLevels(model.ThinkingLevelMap)
	return resolution, true
}

func cloneLevels(levels ai.ThinkingLevelMap) ai.ThinkingLevelMap {
	if levels == nil {
		return nil
	}
	cloned := make(ai.ThinkingLevelMap, len(levels))
	for level, mapped := range levels {
		if mapped == nil {
			cloned[level] = nil
			continue
		}
		value := *mapped
		cloned[level] = &value
	}
	return cloned
}

func firstFloat(values ...*float64) *float64 {
	for _, value := range values {
		if value != nil {
			return value
		}
	}
	return nil
}

func adaptPublicCatalogRecord(provider, catalogModelID string, record *catalog.PublicCatalogRecord) modelgroups.CatalogResolution {
	var piProvider string
	if record != nil {
		piProvider = adapterCatalogProvider(firstNonEmpty(record.PiProvider, record.Provider, provider))
	} else {
		piProvider = adapterCatalogProvider(provider)
	}
	var resolved *catalogHit
	if piProvider != "" {
		recordID := catalogModelID
		if record != nil && record.ModelID != "" {
			recordID = record.ModelID
		}
		resolved = resolveCatalogModel(recordID, piProvider)
		if resolved == nil {
			resolved = resolveCatalogModel(catalogModelID, piProvider)
		}
	}
	var piCatalog *modelgroups.CatalogResolution
	if resolved != nil {
		fromPi := modelgroups.CatalogResolutionFromModel(resolved.provider, resolved.model.ToModel())
		piCatalog = &fromPi
	}
	var piCost *modelgroups.CatalogCost
	if piCatalog != nil {
		piCost = piCatalog.Cost
	}

	resolution := modelgroups.CatalogResolution{Provider: provider}
	// A models.dev hit names the model canonically even when Pi's catalog does not know it yet, so two
	// spellings of one backend model reduce to one identity instead of a conflict.
	switch {
	case resolved != nil:
		resolution.CatalogModelID = resolved.model.ID
	case record != nil && record.ModelID != "":
		resolution.CatalogModelID = record.ModelID
	default:
		resolution.CatalogModelID = catalogModelID
	}
	if piCatalog != nil && piCatalog.Reasoning != nil {
		resolution.Reasoning = piCatalog.Reasoning
	}
	if record != nil && record.EffortLevels != nil {
		reasoning := true
		resolution.Reasoning = &reasoning
		resolution.EffortLevels = record.EffortLevels
	}
	switch {
	case record != nil && record.ThinkingLevelMap != nil:
		levels := make(ai.ThinkingLevelMap, len(record.ThinkingLevelMap))
		for level, mapped := range record.ThinkingLevelMap {
			levels[ai.ThinkingLevel(level)] = mapped
		}
		resolution.ThinkingLevelMap = levels
	case piCatalog != nil && piCatalog.ThinkingLevelMap != nil:
		resolution.ThinkingLevelMap = piCatalog.ThinkingLevelMap
	}
	switch {
	case record != nil && record.Modalities != nil:
		vision := slices.Contains(record.Modalities, "image")
		resolution.Vision = &vision
	case piCatalog != nil && piCatalog.Vision != nil:
		resolution.Vision = piCatalog.Vision
	}
	var recordLimits catalog.Limits
	if record != nil && record.Limits != nil {
		recordLimits = *record.Limits
	}
	if limit := firstFloat(recordLimits.Context); limit != nil {
		resolution.ContextWindow = limit
	} else if piCatalog != nil && piCatalog.ContextWindow != nil {
		resolution.ContextWindow = piCatalog.ContextWindow
	}
	if limit := firstFloat(recordLimits.Output); limit != nil {
		resolution.MaxTokens = limit
	} else if piCatalog != nil && piCatalog.MaxTokens != nil {
		resolution.MaxTokens = piCatalog.MaxTokens
	}
	if (record != nil && record.Cost != nil) || piCost != nil {
		var recordCost catalog.Cost
		if record != nil && record.Cost != nil {
			recordCost = *record.Cost
		}
		var pi modelgroups.CatalogCost
		if piCost != nil {
			pi = *piCost
		}
		cost := &modelgroups.CatalogCost{
			Input:      firstFloat(recordCost.Input, pi.Input),
			Output:     firstFloat(recordCost.Output, pi.Output),
			CacheRead:  firstFloat(recordCost.CacheRead, pi.CacheRead),
			CacheWrite: firstFloat(recordCost.CacheWrite, pi.CacheWrite),
		}
		if pi.Tiers != nil {
			cost.Tiers = pi.Tiers
		}
		resolution.Cost = cost
	}
	return resolution
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

// ResolveModelInfoCatalog is resolveModelInfoCatalog; nil means no catalog evidence. publicCatalog may
// be nil.
func ResolveModelInfoCatalog(entry types.ModelInfoEntry, publicCatalog catalog.PublicCatalog) *modelgroups.CatalogResolution {
	hasBackendIdentity := false
	if entry.ModelInfo != nil && strings.TrimSpace(entry.ModelInfo.BaseModel) != "" {
		hasBackendIdentity = true
	}
	if entry.LiteLLMParams != nil && strings.TrimSpace(entry.LiteLLMParams.Model) != "" {
		hasBackendIdentity = true
	}
	if !hasBackendIdentity {
		return nil
	}
	identity := backend.ResolveIdentity(entry)
	if identity == nil {
		return nil
	}
	lookup := func(provider string) *modelgroups.CatalogResolution {
		var record *catalog.PublicCatalogRecord
		if publicCatalog != nil {
			record = publicCatalog.Lookup(provider, identity.ModelID)
		}
		resolution := adaptPublicCatalogRecord(provider, identity.QualifiedID, record)
		if native, ok := nativeMessagesCatalog(entry); ok {
			resolution.MessagesCompat = native.MessagesCompat
			resolution.MessagesStrictTools = native.MessagesStrictTools
			resolution.MessagesThinkingLevelMap = native.MessagesThinkingLevelMap
		}
		return &resolution
	}
	if provider := backend.ResolveCatalogProvider(entry); provider != "" {
		return lookup(provider)
	}
	if identity.Family == "" {
		return nil
	}
	return lookup(publicCatalogProviderByFamily[identity.Family])
}

// awaitEnrichmentWithinBudget is awaitEnrichmentWithinBudget: it returns the zero value when the budget
// elapses first, and ctx's error when ctx ends first.
func awaitEnrichmentWithinBudget[T any](ctx context.Context, result <-chan T, budget time.Duration) (T, error) {
	var zero T
	if ctx.Err() != nil {
		return zero, context.Cause(ctx)
	}
	timer := time.NewTimer(budget)
	defer timer.Stop()
	select {
	case value := <-result:
		return value, nil
	case <-timer.C:
		return zero, nil
	case <-ctx.Done():
		return zero, context.Cause(ctx)
	}
}

// loadDiscoveryPublicCatalog loads the public catalog within its budget. The load itself is not
// cancelled when the budget elapses or ctx ends, as the TypeScript promise keeps running: it is bounded
// by the catalog's own timeout and refreshes the cache for the next discovery.
func loadDiscoveryPublicCatalog(ctx context.Context, options Options) (catalog.PublicCatalog, error) {
	loadOptions := catalog.LoadPublicCatalogOptions{
		CachePath: options.ModelsDevCachePath,
		Timeout:   defaultTimeout,
		Client:    options.HTTPClient,
	}
	if options.ModelsDev != nil && !*options.ModelsDev {
		loadOptions.Offline = boolPtr(true)
	}
	result := make(chan catalog.PublicCatalog, 1)
	detached := context.WithoutCancel(ctx)
	go func() { result <- catalog.LoadPublicCatalog(detached, loadOptions) }()
	return awaitEnrichmentWithinBudget(ctx, result, min(options.timeout(), publicCatalogBudget))
}

// catalogProtocol is catalogProtocol.
func catalogProtocol(modelID string, catalogModel *ai.GeneratedModel) types.ModelProtocol {
	if catalogModel != nil && catalogModel.API == ai.APIOpenAIResponses {
		return types.ModelProtocol{API: ai.APIOpenAIResponses, Compat: ResponsesCompat(modelID)}
	}
	return types.ModelProtocol{API: ai.APIOpenAICompletions, Compat: CompletionsCompat(modelID, "")}
}

// catalogCost is a Pi catalog model's price as a discovered-model cost.
func catalogCost(model *ai.GeneratedModel) ai.ModelCost {
	return ai.ModelCost{
		Input:      model.InputCostPerMTokens,
		Output:     model.OutputCostPerMTokens,
		CacheRead:  model.CacheReadCost,
		CacheWrite: model.CacheWriteCost,
		Tiers:      slices.Clone(model.Tiers),
	}
}
