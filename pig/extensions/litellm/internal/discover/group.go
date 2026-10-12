// Ports src/discover.ts (group reduction into models, /v1/models mapping, cache enrichment)
package discover

import (
	"reflect"
	"slices"
	"strings"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/backend"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/catalog"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/modelgroups"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/proxyversion"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

// groupOptions are the collectors and switches of mapFromModelInfoGroup. A nil *groupOptions leaves no
// diagnostic behind.
type groupOptions struct {
	ambiguousRoutes         *routeCollector
	conflictingFamilyRoutes *routeCollector
	withheldRepairRoutes    *routeCollector
	defaultedContextRoutes  map[string]bool
	denyLevels              bool
	// withholdMessages is `allowMessages === false`.
	withholdMessages bool
	proxyVersion     *proxyversion.Version
}

func mapFromModelInfoGroup(
	entries []types.ModelInfoEntry,
	publicCatalog catalog.PublicCatalog,
	options *groupOptions,
) *types.DiscoveredModel {
	if options == nil {
		options = &groupOptions{}
	}
	reduced := modelgroups.ReduceModelGroup(entries, func(entry types.ModelInfoEntry) *modelgroups.CatalogResolution {
		resolution := modelgroups.CatalogResolution{}
		if resolved := ResolveModelInfoCatalog(entry, publicCatalog); resolved != nil {
			resolution = *resolved
		}
		family := deploymentFamily(entry)
		generations := map[modelgroups.SemanticModel]bool{}
		var generation modelgroups.SemanticModel
		var ids []string
		if entry.LiteLLMParams != nil {
			ids = append(ids, strings.TrimSpace(entry.LiteLLMParams.Model))
		}
		if entry.ModelInfo != nil {
			ids = append(ids, strings.TrimSpace(entry.ModelInfo.BaseModel))
		}
		for _, id := range ids {
			if id == "" {
				continue
			}
			if semantic := semanticModelOf(id); semantic != "" {
				generations[semantic] = true
				generation = semantic
			}
		}
		if options.withholdMessages {
			resolution.MessagesCompat, resolution.MessagesStrictTools = nil, false
		}
		if family != "" {
			resolution.SemanticFamily = family
		}
		if len(generations) == 1 && family != modelgroups.FamilyConflicting {
			resolution.SemanticModel = generation
		}
		return &resolution
	})
	if reduced == nil {
		return nil
	}
	if reduced.ContextWindowDefaulted && options.defaultedContextRoutes != nil {
		options.defaultedContextRoutes[reduced.ID] = true
	}
	if reduced.CatalogAuthorityAmbiguous {
		options.ambiguousRoutes.add(reduced.ID)
	}
	if slices.Contains(reduced.DeploymentFamilies, modelgroups.FamilyConflicting) {
		options.conflictingFamilyRoutes.add(reduced.ID)
	}
	protocol := ModelProtocol(reduced.ID, entries[0], options.proxyVersion)
	for _, entry := range entries {
		candidate := ModelProtocol(reduced.ID, entry, options.proxyVersion)
		if candidate.API == ai.APIOpenAICompletions {
			protocol = candidate
			break
		}
	}
	api := types.LiteLLMApi(protocol.API)
	if reduced.API == types.APIAnthropicMessages {
		api = reduced.API
	}
	families := map[types.BackendFamily]bool{}
	var family types.BackendFamily
	for i, entry := range entries {
		withoutName := entry
		withoutName.ModelName = ""
		var entryFamily types.BackendFamily
		if identity := backend.ResolveIdentity(withoutName); identity != nil {
			entryFamily = identity.Family
		}
		if i == 0 {
			family = entryFamily
		}
		families[entryFamily] = true
	}
	hasBackendEvidence := slices.ContainsFunc(entries, hasReadableBackendEvidence)
	var reasoningPolicy *modelgroups.ReasoningPolicy
	if api != types.APIAnthropicMessages && reduced.SemanticModel != "" && hasBackendEvidence {
		reasoningPolicy = &reduced.ReasoningPolicy
	}
	reasoning := reduced.Reasoning
	if reasoningPolicy != nil {
		reasoning = reasoningPolicy.Reasoning
	}
	// The semantic policy's compat only applies on Chat, so its level map must be gated the same way.
	// Vendor compatibility is reduced from deployment evidence; route text is consulted only when no
	// deployment identifies a family. This gives vanity Kimi routes the complete strict-schema block
	// while withholding shape-changing fields from mixed or unidentified groups.
	unlabeled := !slices.ContainsFunc(reduced.DeploymentFamilies, func(family modelgroups.FamilyEvidence) bool { return family != "" })
	var vendorCompat *ai.ModelCompat
	if unlabeled {
		vendorCompat = BuildCompat(reduced.ID)
	} else {
		perDeployment := make([]*ai.ModelCompat, len(reduced.DeploymentFamilies))
		for i, family := range reduced.DeploymentFamilies {
			if family != "" {
				perDeployment[i] = CompletionsCompat(reduced.ID, family)
			}
		}
		vendorCompat = modelgroups.MeetVendorCompat(perDeployment)
	}
	var messagesCompat *ai.ModelCompat
	if api == types.APIAnthropicMessages {
		messagesCompat = &ai.ModelCompat{}
		if reduced.MessagesCompat != nil {
			messagesCompat.ForceAdaptiveThinking = reduced.MessagesCompat.ForceAdaptiveThinking
			messagesCompat.SupportsTemperature = reduced.MessagesCompat.SupportsTemperature
		}
		if reduced.MessagesStrictTools {
			messagesCompat.SupportsStrictTools = boolPtr(true)
		}
	}
	input := modelgroups.CloseSerializerPolicyInput{
		API:                              api,
		Reasoning:                        reasoning,
		VendorCompat:                     vendorCompat,
		CatalogLevels:                    reduced.ThinkingLevelMap,
		RequireChatCarrier:               true,
		AllowInferredChatCarrier:         boolPtr(false),
		AcceptsResponsesReasoningControl: reduced.AcceptsResponsesReasoningControl,
		DenyLevels:                       options.denyLevels,
	}
	if api == types.APIAnthropicMessages {
		input.VendorCompat = messagesCompat
	}
	if reasoningPolicy != nil {
		input.SemanticCompat = reasoningPolicy.Compat
		input.SemanticLevels = reasoningPolicy.ThinkingLevelMap
	}
	if reduced.AcceptsResponsesReasoningControl {
		semantic := modelgroups.ReasoningCompat{}
		if reasoningPolicy != nil && reasoningPolicy.Compat != nil {
			semantic = *reasoningPolicy.Compat
		}
		semantic.SupportsReasoningEffort = boolPtr(true)
		input.SemanticCompat = &semantic
	}
	policy := modelgroups.CloseSerializerPolicy(input)

	hasConflictingFamily := slices.Contains(reduced.DeploymentFamilies, modelgroups.FamilyConflicting)
	unanimousMoonshot := len(reduced.DeploymentFamilies) > 0 &&
		!slices.ContainsFunc(reduced.DeploymentFamilies, func(family modelgroups.FamilyEvidence) bool {
			return family != modelgroups.SemanticFamilyKimi
		})
	moonshotEvidence := !hasConflictingFamily &&
		(slices.Contains(reduced.DeploymentFamilies, modelgroups.SemanticFamilyKimi) || (unlabeled && IsMoonshotModel(reduced.ID)))
	if moonshotEvidence && !unanimousMoonshot {
		options.withheldRepairRoutes.add(reduced.ID)
	}
	familyPolicy := requestPolicy(
		reduced.ID,
		reduced.DeploymentFamilies,
		reduced.NormalizeThinkTags,
		reduced.SuppressReasoningVisibility,
		unanimousMoonshot,
	)
	// One GPT-5.5+ deployment is enough: a tool request routed to it fails with reasoning attached,
	// while dropping reasoning elsewhere only loses effort.
	var chatBackendIDs []string
	if api == types.APIOpenAICompletions {
		for _, entry := range entries {
			withoutName := entry
			withoutName.ModelName = ""
			if identity := backend.ResolveIdentity(withoutName); identity != nil {
				chatBackendIDs = append(chatBackendIDs, identity.ModelID)
			}
		}
	}
	modelPolicy := familyPolicy
	if slices.ContainsFunc(chatBackendIDs, IsGpt55OrNewerModel) {
		combined := types.LiteLLMModelPolicy{}
		if familyPolicy != nil {
			combined = *familyPolicy
		}
		combined.DropToolReasoning = true
		if slices.ContainsFunc(chatBackendIDs, IsGpt6OrNewerModel) {
			combined.ExplicitToolReasoningOff = true
		}
		modelPolicy = &combined
	}

	// Reduced groups never borrow the ` (no metadata)` sentinel, which authorizes catalog re-derivation
	// from the model id during offline cache reads.
	name := reduced.ID + " (incomplete metadata)"
	if reduced.HasCompleteMetadata {
		name = reduced.ID
	}
	inputModalities := []string{"text"}
	if reduced.Vision {
		inputModalities = []string{"text", "image"}
	}
	model := &types.DiscoveredModel{
		ID:                      reduced.ID,
		Name:                    name,
		Reasoning:               policy.Reasoning,
		ThinkingLevelMap:        policy.ThinkingLevelMap,
		Compat:                  policy.Compat,
		Input:                   inputModalities,
		Cost:                    reduced.Cost,
		ContextWindow:           int(reduced.ContextWindow),
		MaxTokens:               int(reduced.MaxTokens),
		API:                     ai.API(api),
		LiteLLMDiscoveryVersion: types.DiscoveryVersion,
		LiteLLMPolicy:           modelPolicy,
	}
	if api == types.APIOpenAIResponses && reduced.AcceptsResponsesReasoningControl {
		model.LiteLLMResponsesReasoningControl = true
	}
	if len(families) == 1 && family != "" {
		model.LiteLLMBackendFamily = family
	}
	return model
}

// publishesVersionGatedTransport: whether a group that will be published has its transport decided by
// the proxy version. Mapping without collectors leaves no diagnostic behind, and whether a group is
// published does not depend on the version.
func publishesVersionGatedTransport(group []types.ModelInfoEntry, publicCatalog catalog.PublicCatalog) bool {
	return slices.ContainsFunc(group, transportAwaitsProxyVersion) && mapFromModelInfoGroup(group, publicCatalog, nil) != nil
}

// mapFromModelsList: an evidence-free fallback entry has no deployment or adapter evidence. Its route
// name authorizes nothing; the Pi catalog entry for its id supplies its protocol and presentation
// metadata, while an unknown id stays on Chat Completions until /model/info can be read.
func mapFromModelsList(entry types.ModelsListEntry) *types.DiscoveredModel {
	id := entry.ID
	if id == "" {
		return nil
	}
	catalogModel := findCatalogModel(id, entry.OwnedBy)
	api := ai.APIOpenAICompletions
	if catalogModel != nil && catalogModel.API == ai.APIOpenAIResponses {
		api = ai.APIOpenAIResponses
	}
	model := &types.DiscoveredModel{
		ID:                      id,
		Name:                    id + " (no metadata)",
		Input:                   []string{"text"},
		ContextWindow:           modelgroups.DefaultContextWindow(),
		MaxTokens:               modelgroups.DefaultMaxTokens,
		API:                     api,
		LiteLLMDiscoveryVersion: types.DiscoveryVersion,
	}
	input := modelgroups.CloseSerializerPolicyInput{
		API:                      types.LiteLLMApi(api),
		VendorCompat:             catalogProtocol(id, catalogModel).Compat,
		RequireChatCarrier:       true,
		AllowInferredChatCarrier: boolPtr(false),
	}
	if catalogModel != nil {
		model.Name = catalogModel.DisplayName
		model.Input = cloneStrings(catalogModel.Capabilities)
		model.Cost = catalogCost(catalogModel)
		model.ContextWindow = catalogModel.ContextWindow
		model.MaxTokens = catalogModel.MaxOutputTokens
		input.Reasoning = catalogModel.Reasoning
		input.CatalogLevels = cloneLevels(catalogModel.ThinkingLevelMap)
	}
	policy := modelgroups.CloseSerializerPolicy(input)
	model.Reasoning, model.ThinkingLevelMap, model.Compat = policy.Reasoning, policy.ThinkingLevelMap, policy.Compat
	if IsMoonshotModel(id) {
		moonshot := MoonshotPolicy(id, false)
		model.LiteLLMPolicy = &moonshot
	}
	return model
}

func cloneStrings(values []string) []string { return slices.Clone(values) }

// deduplicateModels keeps the first model of each id; reasoning-content suppression survives only when
// every duplicate carried it.
func deduplicateModels(models []types.DiscoveredModel) []types.DiscoveredModel {
	type entry struct {
		model     types.DiscoveredModel
		suppress  bool
		hasSource bool
	}
	order := []string{}
	entries := map[string]*entry{}
	for _, model := range models {
		suppress := model.SuppressReasoningContent != nil && *model.SuppressReasoningContent
		if existing, ok := entries[model.ID]; ok {
			existing.suppress = existing.suppress && suppress
			continue
		}
		order = append(order, model.ID)
		entries[model.ID] = &entry{model: model, suppress: suppress, hasSource: true}
	}
	deduplicated := make([]types.DiscoveredModel, 0, len(order))
	for _, id := range order {
		model := entries[id].model
		if entries[id].suppress {
			model.SuppressReasoningContent = boolPtr(true)
		} else {
			model.SuppressReasoningContent = nil
		}
		deduplicated = append(deduplicated, model)
	}
	return deduplicated
}

// EnrichCachedModel is enrichCachedModel: it re-derives the serializer policy of a cached model and
// re-enriches an evidence-free fallback from Pi's catalog.
func EnrichCachedModel(input types.DiscoveredModel) types.DiscoveredModel {
	restored := RestoreCachedModelPolicy(input)
	// A model stored by a release that predates the transmissibility gate carries whatever level map
	// that release published, so the gate applies to the cached map on the way in — not only to
	// catalog metadata on the way out, which every reasoning model skips via the guard below.
	cachedThinkingLevelMap := restored.ThinkingLevelMap
	model := restored
	api := types.APIOpenAICompletions
	switch restored.API {
	case ai.APIAnthropicMessages:
		api = types.APIAnthropicMessages
	case ai.APIOpenAIResponses:
		api = types.APIOpenAIResponses
	}
	policy := modelgroups.CloseSerializerPolicy(modelgroups.CloseSerializerPolicyInput{
		API:                              api,
		Reasoning:                        restored.Reasoning,
		VendorCompat:                     restored.Compat,
		CatalogLevels:                    cachedThinkingLevelMap,
		RequireChatCarrier:               true,
		AllowInferredChatCarrier:         boolPtr(false),
		AcceptsResponsesReasoningControl: restored.LiteLLMResponsesReasoningControl,
	})
	model.Reasoning, model.ThinkingLevelMap, model.Compat = policy.Reasoning, policy.ThinkingLevelMap, policy.Compat
	// Reduced deployment groups use a distinct marker; this sentinel remains exclusive to
	// evidence-free fallback models that may be enriched safely. A window is fallback evidence when it
	// matches either default, so a cached entry still qualifies after an operator configures
	// LITELLM_DEFAULT_CONTEXT_WINDOW.
	if !strings.HasSuffix(model.Name, " (no metadata)") ||
		model.Reasoning ||
		model.ThinkingLevelMap != nil ||
		len(model.Input) != 1 ||
		model.Input[0] != "text" ||
		model.Cost.Input != 0 ||
		model.Cost.Output != 0 ||
		model.Cost.CacheRead != 0 ||
		model.Cost.CacheWrite != 0 ||
		model.Cost.Tiers != nil ||
		!modelgroups.IsFallbackContextWindow(float64(model.ContextWindow)) ||
		model.MaxTokens != modelgroups.DefaultMaxTokens {
		return model
	}
	catalogModel := findCatalogModel(model.ID, "")
	// The gate above proved this window is an assumption, so it is re-derived rather than restored:
	// the cache is seeded offline, where discovery never re-reads the setting.
	if catalogModel == nil {
		model.ContextWindow = modelgroups.DefaultContextWindow()
		return model
	}
	// Match evidence-free discovery: only an explicit Responses catalog transport changes the protocol;
	// all other catalog APIs continue through Chat.
	catalogAPI := ai.APIOpenAICompletions
	if catalogModel.API == ai.APIOpenAIResponses {
		catalogAPI = ai.APIOpenAIResponses
	}
	// The cached compat stays as stored, so catalog levels are closed against it rather than trusted
	// because the catalog offered them.
	policy = modelgroups.CloseSerializerPolicy(modelgroups.CloseSerializerPolicyInput{
		API:                              types.LiteLLMApi(catalogAPI),
		Reasoning:                        catalogModel.Reasoning,
		VendorCompat:                     catalogProtocol(model.ID, catalogModel).Compat,
		CatalogLevels:                    cloneLevels(catalogModel.ThinkingLevelMap),
		RequireChatCarrier:               true,
		AllowInferredChatCarrier:         boolPtr(false),
		AcceptsResponsesReasoningControl: model.LiteLLMResponsesReasoningControl,
	})
	model.Name = catalogModel.DisplayName
	model.Reasoning, model.ThinkingLevelMap, model.Compat = policy.Reasoning, policy.ThinkingLevelMap, policy.Compat
	model.Input = cloneStrings(catalogModel.Capabilities)
	model.Cost = catalogCost(catalogModel)
	model.ContextWindow = catalogModel.ContextWindow
	model.MaxTokens = catalogModel.MaxOutputTokens
	model.API = catalogAPI
	return model
}

// RestoreCachedModelPolicy is restoreCachedModelPolicy.
func RestoreCachedModelPolicy(cached types.DiscoveredModel) types.DiscoveredModel {
	restored := cached
	if cached.API == ai.APIAnthropicMessages &&
		cached.LiteLLMDiscoveryVersion != types.DiscoveryVersion &&
		cached.Compat != nil && cached.Compat.SupportsStrictTools != nil {
		// Older discovery copied this value from the direct Anthropic catalog even for hosted routes.
		// Without the original deployment rows it cannot be re-proven offline, so fail closed until the
		// forced network refresh.
		compat := *cached.Compat
		compat.SupportsStrictTools = nil
		restored.Compat = &compat
		if reflect.DeepEqual(compat, ai.ModelCompat{}) {
			restored.Compat = nil
		}
	}
	if restored.LiteLLMPolicy != nil {
		return restored
	}
	// The cache no longer holds the deployment rows that prove a GPT-5.5+ backend, so an OpenAI-family
	// Chat entry fails toward the tool-reasoning workaround until refreshed.
	if restored.API == ai.APIOpenAICompletions &&
		restored.LiteLLMBackendFamily == types.FamilyOpenAI &&
		IsGpt55OrNewerModel(restored.ID) {
		restored.LiteLLMPolicy = &types.LiteLLMModelPolicy{
			DropToolReasoning:        true,
			ExplicitToolReasoningOff: IsGpt6OrNewerModel(restored.ID),
		}
		return restored
	}
	if !hasMoonshotCompatEvidence(restored.Compat) {
		return restored
	}
	moonshot := MoonshotPolicy(restored.ID, false)
	restored.LiteLLMPolicy = &moonshot
	return restored
}
