// Ports src/discover.ts (wildcard routes expanded through /v1/models)
package discover

import (
	"cmp"
	"regexp"
	"slices"
	"strings"
	"unicode"
	"unicode/utf16"

	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/catalog"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/modelgroups"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/proxyversion"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/thinking"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"

	"github.com/MichaelKinsy/PiG/ai"
)

// wildcardSource is a reduced wildcard route with its per-deployment family evidence.
type wildcardSource struct {
	model              types.DiscoveredModel
	deploymentFamilies []modelgroups.FamilyEvidence
}

// WildcardMatches is wildcardMatches.
func WildcardMatches(route, modelID string) bool {
	segments := strings.Split(route, "*")
	offset := 0
	for index, segment := range segments {
		if segment == "" {
			continue
		}
		found := strings.Index(modelID[offset:], segment)
		if found < 0 || (index == 0 && found != 0) {
			return false
		}
		offset += found + len(segment)
	}
	suffix := segments[len(segments)-1]
	return suffix == "" || strings.HasSuffix(modelID, suffix)
}

// resolveWildcardRow: LiteLLM serves a wildcard route by substituting the requested id's wildcard
// portion into the deployment's own `model` (`azure/*` + `azure/gpt-5.5` → `azure/gpt-5.5`), so the row
// a child actually runs on names that concrete backend model.
func resolveWildcardRow(row types.ModelInfoEntry, modelID string) types.ModelInfoEntry {
	route := row.ModelName
	backendModel := ""
	if row.LiteLLMParams != nil {
		backendModel = row.LiteLLMParams.Model
	}
	resolved := row
	resolved.ModelName = modelID
	if !strings.Contains(route, "*") || !strings.Contains(backendModel, "*") {
		return resolved
	}
	// Capture each `*` segment of the route against modelID, then substitute those captures
	// positionally into the backend's own wildcards, so a route with 2+ stars (e.g. "team/*-*")
	// resolves every star instead of only the first.
	segments := strings.Split(route, "*")
	for i, segment := range segments {
		segments[i] = regexp.QuoteMeta(segment)
	}
	match := regexp.MustCompile("^" + strings.Join(segments, "(.*)") + "$").FindStringSubmatch(modelID)
	if match == nil {
		return resolved
	}
	captures := match[1:]
	index := 0
	var substituted strings.Builder
	for _, character := range backendModel {
		if character != '*' {
			substituted.WriteRune(character)
			continue
		}
		if index < len(captures) {
			substituted.WriteString(captures[index])
		}
		index++
	}
	params := *row.LiteLLMParams
	params.Model = substituted.String()
	resolved.LiteLLMParams = &params
	return resolved
}

// wildcardPatternSpecificity returns the escaped pattern length (in UTF-16 units) and its complexity.
func wildcardPatternSpecificity(pattern string) (length, complexity int) {
	var escaped strings.Builder
	for _, character := range pattern {
		switch {
		case character == '*':
			escaped.WriteString("(.*)")
		case strings.ContainsRune(`()[]{}?+-|^$\.&~#`, character) || unicode.IsSpace(character):
			escaped.WriteRune('\\')
			escaped.WriteRune(character)
		default:
			escaped.WriteRune(character)
		}
	}
	text := escaped.String()
	for _, character := range text {
		if strings.ContainsRune("*+?\\^$|()", character) {
			complexity++
		}
	}
	return len(utf16.Encode([]rune(text))), complexity
}

// selectedWildcardRows returns the rows of the most-specific published wildcard route LiteLLM would
// select for a concrete id, using its pattern order, with the id substituted into each row. nil means
// none.
func selectedWildcardRows(modelID string, wildcardRows []types.ModelInfoEntry, publishedWildcardIDs map[string]bool) []types.ModelInfoEntry {
	var matchingRows []types.ModelInfoEntry
	for _, row := range wildcardRows {
		if row.ModelName != "" && WildcardMatches(row.ModelName, modelID) {
			matchingRows = append(matchingRows, row)
		}
	}
	if len(matchingRows) == 0 {
		return nil
	}
	patterns := make([]string, len(matchingRows))
	for i, row := range matchingRows {
		patterns[i] = row.ModelName
	}
	slices.SortStableFunc(patterns, func(left, right string) int {
		leftLength, leftComplexity := wildcardPatternSpecificity(left)
		rightLength, rightComplexity := wildcardPatternSpecificity(right)
		// Fewer wildcards (lower complexity) is more specific when escaped length ties.
		return cmp.Or(rightLength-leftLength, leftComplexity-rightComplexity, strings.Compare(left, right))
	})
	selectedPattern := patterns[0]
	if !publishedWildcardIDs[selectedPattern] {
		return nil
	}
	var selected []types.ModelInfoEntry
	for _, row := range matchingRows {
		if row.ModelName == selectedPattern {
			selected = append(selected, resolveWildcardRow(row, modelID))
		}
	}
	return selected
}

// applyWildcardEvidence: a concrete id expanded from a wildcard route inherits the deployment evidence
// from the most-specific matching route, using LiteLLM's pattern order. Rows sharing that route still
// vote together, so any deployment needing Chat keeps the child on Chat. nil means the child is not
// published.
func applyWildcardEvidence(
	model types.DiscoveredModel,
	wildcardRows []types.ModelInfoEntry,
	publishedWildcardIDs map[string]bool,
	publicCatalog catalog.PublicCatalog,
	proxyVersion *proxyversion.Version,
) *types.DiscoveredModel {
	parents := selectedWildcardRows(model.ID, wildcardRows, publishedWildcardIDs)
	if parents == nil {
		return nil
	}
	selected := mapFromModelInfoGroup(parents, publicCatalog, &groupOptions{proxyVersion: proxyVersion})
	if selected == nil {
		return nil
	}
	var policies []types.LiteLLMModelPolicy
	for _, policy := range []*types.LiteLLMModelPolicy{model.LiteLLMPolicy, selected.LiteLLMPolicy} {
		if policy != nil {
			policies = append(policies, *policy)
		}
	}
	every := func(flag func(types.LiteLLMModelPolicy) bool) bool {
		return !slices.ContainsFunc(policies, func(policy types.LiteLLMModelPolicy) bool { return !flag(policy) })
	}
	some := func(flag func(types.LiteLLMModelPolicy) bool) bool { return slices.ContainsFunc(policies, flag) }

	result := model
	result.LiteLLMBackendFamily, result.LiteLLMResponsesReasoningControl, result.LiteLLMPolicy, result.ThinkingLevelMap = "", false, nil, nil
	result.API = selected.API
	result.Reasoning = selected.Reasoning
	result.Compat = selected.Compat
	if selected.ThinkingLevelMap != nil {
		result.ThinkingLevelMap = selected.ThinkingLevelMap
	}
	if selected.LiteLLMBackendFamily != "" {
		result.LiteLLMBackendFamily = selected.LiteLLMBackendFamily
	}
	if len(policies) > 0 {
		result.LiteLLMPolicy = &types.LiteLLMModelPolicy{
			NormalizeStrictToolMessages:    every(func(p types.LiteLLMModelPolicy) bool { return p.NormalizeStrictToolMessages }),
			NormalizeThinkTags:             every(func(p types.LiteLLMModelPolicy) bool { return p.NormalizeThinkTags }),
			SuppressReasoningVisibility:    every(func(p types.LiteLLMModelPolicy) bool { return p.SuppressReasoningVisibility }),
			NormalizeGeminiReasoningEffort: every(func(p types.LiteLLMModelPolicy) bool { return p.NormalizeGeminiReasoningEffort }),
			DropToolReasoning:              some(func(p types.LiteLLMModelPolicy) bool { return p.DropToolReasoning }),
			ExplicitToolReasoningOff:       some(func(p types.LiteLLMModelPolicy) bool { return p.ExplicitToolReasoningOff }),
		}
	}
	if selected.LiteLLMResponsesReasoningControl {
		result.LiteLLMResponsesReasoningControl = true
	}
	return &result
}

func mapFromWildcardExpansion(
	entry types.ModelsListEntry,
	wildcards []wildcardSource,
	withheldRepairRoutes *routeCollector,
) *types.DiscoveredModel {
	id := entry.ID
	if id == "" || strings.Contains(id, "*") {
		return nil
	}
	var matchingSources []wildcardSource
	for _, source := range wildcards {
		if WildcardMatches(source.model.ID, id) {
			matchingSources = append(matchingSources, source)
		}
	}
	if len(matchingSources) == 0 {
		return nil
	}
	matches := make([]types.DiscoveredModel, len(matchingSources))
	for i, source := range matchingSources {
		matches[i] = source.model
	}
	all := func(predicate func(types.DiscoveredModel) bool) bool {
		return !slices.ContainsFunc(matches, func(model types.DiscoveredModel) bool { return !predicate(model) })
	}
	some := func(predicate func(types.DiscoveredModel) bool) bool { return slices.ContainsFunc(matches, predicate) }

	api := ai.APIOpenAICompletions
	if all(func(model types.DiscoveredModel) bool { return model.API == ai.APIOpenAIResponses }) {
		api = ai.APIOpenAIResponses
	}
	reasoning := all(func(model types.DiscoveredModel) bool { return model.Reasoning })
	var thinkingLevelMap ai.ThinkingLevelMap
	if reasoning {
		levels := make([]ai.ThinkingLevelMap, len(matches))
		for i, model := range matches {
			levels[i] = model.ThinkingLevelMap
		}
		thinkingLevelMap = thinking.Intersect(levels)
	}
	hasFamilyEvidence := slices.ContainsFunc(matchingSources, func(source wildcardSource) bool {
		return slices.ContainsFunc(source.deploymentFamilies, func(family modelgroups.FamilyEvidence) bool { return family != "" })
	})
	var vendorCompat *ai.ModelCompat
	if hasFamilyEvidence {
		compats := make([]*ai.ModelCompat, len(matches))
		for i, model := range matches {
			compats[i] = model.Compat
		}
		vendorCompat = modelgroups.MeetVendorCompat(compats)
	} else {
		vendorCompat = BuildCompat(id)
	}
	policyFlag := func(flag func(types.LiteLLMModelPolicy) bool) func(types.DiscoveredModel) bool {
		return func(model types.DiscoveredModel) bool {
			return model.LiteLLMPolicy != nil && flag(*model.LiteLLMPolicy)
		}
	}
	forcedThinking := forcedThinkingModelPattern.MatchString(id)
	var modelPolicy *types.LiteLLMModelPolicy
	if some(func(model types.DiscoveredModel) bool { return model.LiteLLMPolicy != nil }) {
		modelPolicy = &types.LiteLLMModelPolicy{
			NormalizeStrictToolMessages: all(policyFlag(func(p types.LiteLLMModelPolicy) bool { return p.NormalizeStrictToolMessages })),
			NormalizeThinkTags: !forcedThinking &&
				all(policyFlag(func(p types.LiteLLMModelPolicy) bool { return p.NormalizeThinkTags })),
			SuppressReasoningVisibility: !forcedThinking &&
				all(policyFlag(func(p types.LiteLLMModelPolicy) bool { return p.SuppressReasoningVisibility })),
			NormalizeGeminiReasoningEffort: all(policyFlag(func(p types.LiteLLMModelPolicy) bool { return p.NormalizeGeminiReasoningEffort })),
			DropToolReasoning: api == ai.APIOpenAICompletions &&
				some(policyFlag(func(p types.LiteLLMModelPolicy) bool { return p.DropToolReasoning })),
			ExplicitToolReasoningOff: api == ai.APIOpenAICompletions &&
				some(policyFlag(func(p types.LiteLLMModelPolicy) bool { return p.ExplicitToolReasoningOff })),
		}
	} else if !hasFamilyEvidence && IsMoonshotModel(id) {
		moonshot := MoonshotPolicy(id, false)
		modelPolicy = &moonshot
	}
	responsesControl := api == ai.APIOpenAIResponses &&
		all(func(model types.DiscoveredModel) bool { return model.LiteLLMResponsesReasoningControl })
	policy := modelgroups.CloseSerializerPolicy(modelgroups.CloseSerializerPolicyInput{
		API:                              types.LiteLLMApi(api),
		Reasoning:                        reasoning,
		VendorCompat:                     vendorCompat,
		CatalogLevels:                    thinkingLevelMap,
		RequireChatCarrier:               true,
		AllowInferredChatCarrier:         boolPtr(false),
		AcceptsResponsesReasoningControl: responsesControl,
	})
	hasKimiEvidence := slices.ContainsFunc(matchingSources, func(source wildcardSource) bool {
		return slices.Contains(source.deploymentFamilies, modelgroups.SemanticFamilyKimi)
	})
	routeOnlyMoonshot := !hasFamilyEvidence && IsMoonshotModel(id)
	if modelPolicy != nil && !modelPolicy.NormalizeStrictToolMessages && (hasKimiEvidence || routeOnlyMoonshot) {
		withheldRepairRoutes.add(id)
	}
	// A concrete /v1/models id may enrich presentation only. It never contributes reasoning levels,
	// compatibility carriers, or request policy to the child.
	catalogModel := findCatalogModel(id, entry.OwnedBy)
	parentIncomplete := some(func(model types.DiscoveredModel) bool { return strings.HasSuffix(model.Name, " (incomplete metadata)") })
	base := id
	if catalogModel != nil {
		base = catalogModel.DisplayName
	}
	name := base
	if parentIncomplete {
		name = base + " (incomplete metadata)"
	}
	// Preserve every known tier even when a sibling has incomplete metadata. Omitting a complete
	// sibling's higher tier would understate the known worst-case rate; the incomplete marker
	// continues to signal that the resulting envelope is partial.
	costs := make([]ai.ModelCost, len(matches))
	for i, model := range matches {
		costs[i] = model.Cost
	}
	cost := ai.ModelCost{Tiers: modelgroups.ConservativeCostTiers(costs)}
	contextWindow, maxTokens := matches[0].ContextWindow, matches[0].MaxTokens
	for i, model := range matches {
		if i == 0 {
			cost.Input, cost.Output, cost.CacheRead, cost.CacheWrite = model.Cost.Input, model.Cost.Output, model.Cost.CacheRead, model.Cost.CacheWrite
			continue
		}
		cost.Input = max(cost.Input, model.Cost.Input)
		cost.Output = max(cost.Output, model.Cost.Output)
		cost.CacheRead = max(cost.CacheRead, model.Cost.CacheRead)
		cost.CacheWrite = max(cost.CacheWrite, model.Cost.CacheWrite)
		contextWindow = min(contextWindow, model.ContextWindow)
		maxTokens = min(maxTokens, model.MaxTokens)
	}
	input := []string{"text"}
	if all(func(model types.DiscoveredModel) bool { return slices.Contains(model.Input, "image") }) {
		input = []string{"text", "image"}
	}
	return &types.DiscoveredModel{
		ID:                               id,
		Name:                             name,
		Reasoning:                        policy.Reasoning,
		ThinkingLevelMap:                 policy.ThinkingLevelMap,
		Compat:                           policy.Compat,
		Input:                            input,
		Cost:                             cost,
		ContextWindow:                    contextWindow,
		MaxTokens:                        maxTokens,
		API:                              api,
		LiteLLMDiscoveryVersion:          types.DiscoveryVersion,
		LiteLLMResponsesReasoningControl: responsesControl,
		LiteLLMPolicy:                    modelPolicy,
	}
}
