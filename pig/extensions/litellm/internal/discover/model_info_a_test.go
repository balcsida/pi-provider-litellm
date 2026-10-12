package discover

// Ports "discoverModels via /model/info" of tests/discover.test.ts, from "withholds backend family when the
// model prefix conflicts with custom_llm_provider" up to "never emits the fallback-only sentinel for a
// reduced deployment group".

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/catalog"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

type aMap = map[string]any

// aBody marshals rows into a /model/info envelope.
func aBody(rows ...any) json.RawMessage {
	body, _ := json.Marshal(aMap{"data": rows})
	return body
}

// aServe serves rows on /model/info only.
func aServe(t *testing.T, rows ...any) string {
	t.Helper()
	return mockEndpoints(t, map[string]http.HandlerFunc{"/model/info": jsonResponse(200, aBody(rows...))})
}

func aRates(cost ai.ModelCost) [4]float64 {
	return [4]float64{cost.Input, cost.Output, cost.CacheRead, cost.CacheWrite}
}

// aExpectIncomplete is the repeated toMatchObject block of a model withheld from catalog authority.
func aExpectIncomplete(t *testing.T, model types.DiscoveredModel, id string, window, maxTokens int) {
	t.Helper()
	if model.ID != id || model.Name != id+" (incomplete metadata)" || model.Reasoning ||
		!reflect.DeepEqual(model.Input, []string{"text"}) || model.ContextWindow != window || model.MaxTokens != maxTokens ||
		aRates(model.Cost) != [4]float64{} {
		t.Errorf("model = %+v", model)
	}
}

// aLevels builds a level map; the value "null" is an explicit denial.
func aLevels(pairs map[string]string) ai.ThinkingLevelMap {
	out := ai.ThinkingLevelMap{}
	for level, value := range pairs {
		if value == "null" {
			out[ai.ThinkingLevel(level)] = nil
		} else {
			wire := value
			out[ai.ThinkingLevel(level)] = &wire
		}
	}
	return out
}

// aLevelsMatch is toMatchObject on a level map: every wanted level must be present with its value.
func aLevelsMatch(t *testing.T, got ai.ThinkingLevelMap, want map[string]string) {
	t.Helper()
	for level, value := range want {
		have, ok := got[ai.ThinkingLevel(level)]
		switch {
		case !ok:
			t.Errorf("level %s missing from %v", level, got)
		case value == "null" && have != nil:
			t.Errorf("level %s = %q, want null", level, *have)
		case value != "null" && (have == nil || *have != value):
			t.Errorf("level %s = %v, want %q", level, have, value)
		}
	}
}

func aSupportedLevels(model types.DiscoveredModel) []ai.ThinkingLevel {
	return ai.GetSupportedThinkingLevels(&ai.Model{
		ProviderMeta:     ai.ProviderMetadata{Reasoning: model.Reasoning},
		ThinkingLevelMap: model.ThinkingLevelMap,
	})
}

func aOffline(t *testing.T) Options {
	options, _ := testOptions(t)
	return options
}

// aCatalog answers lookups whose id ends with suffix.
type aCatalog struct {
	suffix string
	record catalog.PublicCatalogRecord
}

func (c aCatalog) Lookup(_, id string) *catalog.PublicCatalogRecord {
	if strings.HasSuffix(id, c.suffix) {
		record := c.record
		return &record
	}
	return nil
}

func aFloat(v float64) *float64 { return &v }

func TestModelInfoA_WithholdsBackendFamilyWhenTheModelPrefixConflictsWithCustomLlmProvider(t *testing.T) {
	base := aServe(t, aMap{
		"model_name":     "gpt-prod",
		"litellm_params": aMap{"model": "openai/gpt-5", "custom_llm_provider": "fireworks_ai"},
		"model_info":     aMap{"mode": "chat"},
	})
	options, _ := testOptions(t)
	options.ModelsDev = nil

	model := firstModel(t, base, options)

	if model.ID != "gpt-prod" || model.API != ai.APIOpenAICompletions || model.LiteLLMBackendFamily != "" {
		t.Errorf("model = %+v", model)
	}
}

func TestModelInfoA_KeepsAnAzureGroupOnChatWhenASiblingLacksResponsesEvidence(t *testing.T) {
	for _, explicitSibling := range []bool{false, true} {
		t.Run(fmt.Sprint(explicitSibling), func(t *testing.T) {
			rows := []any{}
			for index, responses := range []bool{false, explicitSibling} {
				mode := "chat"
				if responses {
					mode = "responses"
				}
				rows = append(rows, aMap{
					"model_name":     "azure-chat-route",
					"litellm_params": aMap{"model": "azure/gpt-5", "api_version": "2025-04-01-preview"},
					"model_info": aMap{
						"id": fmt.Sprintf("deployment-%d", index), "mode": mode,
						"supports_reasoning": true, "supported_openai_params": []string{"reasoning_effort"},
					},
				})
			}
			base := aServe(t, rows...)

			model := firstModel(t, base, aOffline(t))

			if model.API != ai.APIOpenAICompletions || !model.Reasoning || model.Compat == nil ||
				model.Compat.SupportsReasoningEffort == nil || !*model.Compat.SupportsReasoningEffort {
				t.Errorf("model = %+v", model)
			}
			if !slices.Contains(aSupportedLevels(model), ai.ThinkingMedium) {
				t.Errorf("levels = %v, want medium", aSupportedLevels(model))
			}
		})
	}
}

func TestModelInfoA_PreservesProviderSpecificContextLimitsThroughDiscovery(t *testing.T) {
	for _, provider := range []string{"azure", "azure_ai"} {
		t.Run(provider, func(t *testing.T) {
			base := aServe(t, aMap{
				"model_name":     "large-context-route",
				"litellm_params": aMap{"model": provider + "/gpt-5.6-sol"},
				"model_info": aMap{
					"mode": "chat", "litellm_provider": provider,
					"max_input_tokens": 1_050_000, "max_output_tokens": 128_000,
				},
			})

			model := firstModel(t, base, aOffline(t))

			if model.ContextWindow != 1_050_000 || model.MaxTokens != 128_000 {
				t.Errorf("model = %+v", model)
			}
		})
	}
}

func TestModelInfoA_ReducesMixedAzureDeploymentVersionsToChatCompletions(t *testing.T) {
	row := func(version string) aMap {
		return aMap{
			"model_name":     "gpt-production",
			"litellm_params": aMap{"model": "azure/gpt-5", "api_version": version},
			"model_info":     aMap{"mode": "chat"},
		}
	}
	base := aServe(t, row("2025-04-01-preview"), row("2024-12-01-preview"))
	options, _ := testOptions(t)
	options.ModelsDev = nil

	model := firstModel(t, base, options)

	if model.ID != "gpt-production" || model.API != ai.APIOpenAICompletions || model.LiteLLMBackendFamily != types.FamilyOpenAI ||
		model.LiteLLMDiscoveryVersion != 6 {
		t.Errorf("model = %+v", model)
	}
}

func TestModelInfoA_ParsesAModelInfoSuccessResponseWithCostMapping(t *testing.T) {
	base := aServe(t,
		aMap{
			"model_name":     "anthropic/claude-3-5-sonnet",
			"litellm_params": aMap{"model": "anthropic/claude-3-5-sonnet"},
			"model_info": aMap{
				"mode": "chat", "max_input_tokens": 200000, "max_output_tokens": 8192,
				"supports_vision": true, "supports_reasoning": false,
				"input_cost_per_token": 0.000003, "output_cost_per_token": 0.000015,
				"cache_read_input_token_cost": 0.0000003, "cache_creation_input_token_cost": 0.00000375,
			},
		},
		aMap{
			"model_name":     "openai/gpt-4o",
			"litellm_params": aMap{"model": "openai/gpt-4o"},
			"model_info":     aMap{"mode": "chat", "max_input_tokens": 128000, "max_output_tokens": 16384},
		},
		aMap{"model_name": "openai/text-embedding-3-large", "model_info": aMap{"mode": "embedding"}},
	)

	result := discover(t, base, aOffline(t))

	if result.Source != types.SourceModelInfo {
		t.Errorf("source = %q", result.Source)
	}
	// The embedding model is filtered out by mode != "chat".
	if len(result.Models) != 2 {
		t.Fatalf("models = %+v", result.Models)
	}
	var anthropic, openai *types.DiscoveredModel
	for i := range result.Models {
		switch result.Models[i].ID {
		case "anthropic/claude-3-5-sonnet":
			anthropic = &result.Models[i]
		case "openai/gpt-4o":
			openai = &result.Models[i]
		}
	}
	if anthropic == nil || openai == nil {
		t.Fatalf("models = %+v", result.Models)
	}
	if anthropic.Name != "anthropic/claude-3-5-sonnet" || anthropic.ContextWindow != 200000 || anthropic.MaxTokens != 8192 ||
		!reflect.DeepEqual(anthropic.Input, []string{"text", "image"}) || anthropic.Compat == nil ||
		anthropic.Compat.SupportsStore == nil || *anthropic.Compat.SupportsStore || anthropic.Compat.CacheControlFormat != "anthropic" {
		t.Errorf("anthropic = %+v", anthropic)
	}
	// Cost is per-token in LiteLLM, per-million-tokens in pi-ai.
	if got, want := aRates(anthropic.Cost), [4]float64{3, 15, 0.3, 3.75}; got != want || len(anthropic.Cost.Tiers) != 0 {
		t.Errorf("anthropic cost = %+v", anthropic.Cost)
	}
	if openai.Name != "openai/gpt-4o" || !reflect.DeepEqual(openai.Input, []string{"text", "image"}) ||
		openai.API != ai.APIOpenAIResponses || openai.Compat != nil {
		t.Errorf("openai = %+v", openai)
	}
}

func TestModelInfoA_MapsLiteLLMReasoningEffortCapabilitiesForASingleton(t *testing.T) {
	base := aServe(t, aMap{
		"model_name":     "custom/reasoner",
		"litellm_params": aMap{"allowed_openai_params": []string{"reasoning_effort"}},
		"model_info": aMap{
			"mode": "chat", "supports_reasoning": true, "supports_none_reasoning_effort": true,
			"supports_minimal_reasoning_effort": false, "supports_low_reasoning_effort": false,
			"supports_xhigh_reasoning_effort": false, "supports_max_reasoning_effort": true,
		},
	})

	model := firstModel(t, base, aOffline(t))

	want := aLevels(map[string]string{"off": "none", "minimal": "null", "low": "null", "xhigh": "null", "max": "max"})
	if !reflect.DeepEqual(model.ThinkingLevelMap, want) {
		t.Errorf("levels = %v, want %v", model.ThinkingLevelMap, want)
	}
}

func TestModelInfoA_MapsRouterMediumAndHighEffortFlagsForASingleton(t *testing.T) {
	base := aServe(t, aMap{
		"model_name":     "custom/reasoner",
		"litellm_params": aMap{"allowed_openai_params": []string{"reasoning_effort"}},
		"model_info": aMap{
			"mode": "chat", "supports_reasoning": true, "supports_none_reasoning_effort": true,
			"supports_low_reasoning_effort": true, "supports_medium_reasoning_effort": true,
			"supports_high_reasoning_effort": true, "supports_xhigh_reasoning_effort": true,
			"supports_max_reasoning_effort": true,
		},
	})

	model := firstModel(t, base, aOffline(t))

	aLevelsMatch(t, model.ThinkingLevelMap, map[string]string{
		"off": "none", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "max",
	})
}

func TestModelInfoA_DeniesMediumAndHighWhenTheRouterReportsThemUnsupported(t *testing.T) {
	base := aServe(t, aMap{
		"model_name":     "custom/reasoner",
		"litellm_params": aMap{"allowed_openai_params": []string{"reasoning_effort"}},
		"model_info": aMap{
			"mode": "chat", "supports_reasoning": true, "supports_low_reasoning_effort": true,
			"supports_medium_reasoning_effort": false, "supports_high_reasoning_effort": false,
		},
	})

	model := firstModel(t, base, aOffline(t))

	aLevelsMatch(t, model.ThinkingLevelMap, map[string]string{"low": "low", "medium": "null", "high": "null"})
}

func TestModelInfoA_KeepsMaxSelectableOnAResponsesRouteThatOptsIntoIt(t *testing.T) {
	base := aServe(t, aMap{
		"model_name":     "high",
		"litellm_params": aMap{"model": "chatgpt/gpt-5.6-sol"},
		"model_info": aMap{
			"mode": "responses", "litellm_provider": "chatgpt",
			"supported_openai_params": []string{"reasoning_effort"}, "supports_reasoning": true,
			"supports_none_reasoning_effort": true, "supports_low_reasoning_effort": true,
			"supports_medium_reasoning_effort": true, "supports_high_reasoning_effort": true,
			"supports_xhigh_reasoning_effort": true, "supports_max_reasoning_effort": true,
		},
	})

	model := firstModel(t, base, aOffline(t))

	if model.API != ai.APIOpenAIResponses {
		t.Errorf("api = %q", model.API)
	}
	aLevelsMatch(t, model.ThinkingLevelMap, map[string]string{
		"off": "none", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "max",
	})
}

func TestModelInfoA_KeepsStandardLevelsACodexCatalogMapLeavesImplicit(t *testing.T) {
	base := aServe(t, aMap{
		"model_name":     "gpt-5.6-sol",
		"litellm_params": aMap{"model": "chatgpt/gpt-5.6-sol"},
		"model_info": aMap{
			"mode": "responses", "litellm_provider": "chatgpt",
			"supported_openai_params":           []string{"reasoning_effort"},
			"supports_reasoning":                nil,
			"supports_minimal_reasoning_effort": false,
			"supports_xhigh_reasoning_effort":   true,
			"supports_max_reasoning_effort":     true,
		},
	})

	model := firstModel(t, base, aOffline(t))

	want := []ai.ThinkingLevel{ai.ThinkingOff, ai.ThinkingLow, ai.ThinkingMedium, ai.ThinkingHigh, ai.ThinkingXHigh, ai.ThinkingMax}
	if got := aSupportedLevels(model); !reflect.DeepEqual(got, want) {
		t.Errorf("levels = %v, want %v", got, want)
	}
}

func TestModelInfoA_ReadsADeclaredReasoningEffortLevelsListAsTheCompleteLevelSet(t *testing.T) {
	base := aServe(t, aMap{
		"model_name": "zai-org/GLM-5.3",
		"litellm_params": aMap{
			"model": "hosted_vllm/zai-org/GLM-5.3", "allowed_openai_params": []string{"reasoning_effort"},
		},
		"model_info": aMap{
			"mode": "chat", "supports_reasoning": true,
			"reasoning_effort_levels":           []string{"none", "low", "high", "max"},
			"supports_none_reasoning_effort":    nil,
			"supports_minimal_reasoning_effort": nil,
			"supports_low_reasoning_effort":     nil,
			"supports_xhigh_reasoning_effort":   nil,
			"supports_max_reasoning_effort":     nil,
		},
	})

	model := firstModel(t, base, aOffline(t))

	want := []ai.ThinkingLevel{ai.ThinkingOff, ai.ThinkingLow, ai.ThinkingHigh, ai.ThinkingMax}
	if got := aSupportedLevels(model); !reflect.DeepEqual(got, want) {
		t.Errorf("levels = %v, want %v", got, want)
	}
	aLevelsMatch(t, model.ThinkingLevelMap, map[string]string{"off": "none", "max": "max"})
}

func TestModelInfoA_MergesSingletonRouterEffortFlagsIntoSupportedResponsesLevels(t *testing.T) {
	base := aServe(t, aMap{
		"model_name":     "openai/gpt-5.6-luna",
		"litellm_params": aMap{"model": "openai/gpt-5.6-luna", "allowed_openai_params": []string{"reasoning_effort"}},
		"model_info": aMap{
			"mode": "chat", "supports_reasoning": true,
			"supports_xhigh_reasoning_effort": false, "supports_max_reasoning_effort": true,
		},
	})

	model := firstModel(t, base, aOffline(t))

	aLevelsMatch(t, model.ThinkingLevelMap, map[string]string{"off": "none", "xhigh": "null", "max": "max"})
}

func TestModelInfoA_UsesFamilyOnlyIdentityToLookUpModelsDevUnderItsPublicProvider(t *testing.T) {
	cachePath := writeModelsDevCache(t, `{"anthropic":{"models":{"opus-4.8":{"limit":{"context":750000,"output":96000}}}}}`)
	base := aServe(t, aMap{
		"model_name": "opus-4.8", "litellm_params": aMap{"model": "opus-4.8"}, "model_info": aMap{"mode": "chat"},
	})
	options := aOffline(t)
	options.ModelsDevCachePath = cachePath

	result := discover(t, base, options)

	model := result.Models[0]
	if result.Source != types.SourceModelInfo || model.ID != "opus-4.8" || model.Name != "opus-4.8" || !model.Reasoning ||
		!reflect.DeepEqual(model.Input, []string{"text", "image"}) || model.ContextWindow != 750_000 || model.MaxTokens != 96_000 ||
		model.Cost.Input <= 0 {
		t.Errorf("source = %q, model = %+v", result.Source, model)
	}
}

func TestModelInfoA_WithholdsCatalogMetadataWhenModelInfoSuppliesOnlyThePublicRouteName(t *testing.T) {
	base := aServe(t, aMap{"model_name": "opus-4.8", "model_info": aMap{"mode": "chat"}})

	result := discover(t, base, aOffline(t))

	if result.Source != types.SourceModelInfo {
		t.Errorf("source = %q", result.Source)
	}
	aExpectIncomplete(t, result.Models[0], "opus-4.8", 128_000, 16_384)
}

func TestModelInfoA_WithholdsCatalogMetadataWhenBackendEvidenceIsOnlyWhitespace(t *testing.T) {
	base := aServe(t, aMap{
		"model_name": "openai/gpt-5", "litellm_params": aMap{"model": "   "}, "model_info": aMap{"mode": "chat"},
	})

	result := discover(t, base, aOffline(t))

	if result.Source != types.SourceModelInfo {
		t.Errorf("source = %q", result.Source)
	}
	aExpectIncomplete(t, result.Models[0], "openai/gpt-5", 128_000, 16_384)
}

func TestModelInfoA_PreservesCatalogPricingTiersForModelInfoModels(t *testing.T) {
	base := aServe(t, aMap{
		"model_name": "openai/gpt-5.5", "litellm_params": aMap{"model": "openai/gpt-5.5"}, "model_info": aMap{"mode": "chat"},
	})

	model := firstModel(t, base, aOffline(t))

	want := []ai.CostTier{{InputTokensAbove: 272000, InputCostPer1M: 10, OutputCostPer1M: 45, CacheReadCostPer1M: 1, CacheWriteCostPer1M: 0}}
	if !reflect.DeepEqual(model.Cost.Tiers, want) {
		t.Errorf("tiers = %+v, want %+v", model.Cost.Tiers, want)
	}
}

func TestModelInfoA_PreservesCatalogXhighAndMaxOnResponses(t *testing.T) {
	base := aServe(t, aMap{
		"model_name":     "openai/gpt-5.6-luna",
		"litellm_params": aMap{"model": "openai/gpt-5.6-luna", "allowed_openai_params": []string{"reasoning_effort"}},
		"model_info":     aMap{"mode": "chat", "supports_xhigh_reasoning_effort": true, "supports_max_reasoning_effort": true},
	})

	model := firstModel(t, base, aOffline(t))

	aLevelsMatch(t, model.ThinkingLevelMap, map[string]string{"off": "none", "xhigh": "xhigh", "max": "max"})
}

func TestModelInfoA_ReducesDuplicateModelIdsConservativelyInsteadOfMergingRicherFields(t *testing.T) {
	base := aServe(t,
		aMap{"model_name": "custom-model", "model_info": aMap{"mode": "chat"}},
		aMap{
			"model_name": "custom-model",
			"model_info": aMap{
				"mode": "chat", "max_input_tokens": 200000, "max_output_tokens": 8192,
				"input_cost_per_token": 0.000003, "output_cost_per_token": 0.000015,
			},
		},
	)

	result := discover(t, base, aOffline(t))

	if result.Source != types.SourceModelInfo || len(result.Models) != 1 {
		t.Fatalf("source = %q, models = %+v", result.Source, result.Models)
	}
	aExpectIncomplete(t, result.Models[0], "custom-model", 128000, 8192)
}

func TestModelInfoA_FallsBackToChatAndGroupGuaranteesForMixedDeploymentsRegardlessOfRowOrder(t *testing.T) {
	route := "shared-route"
	deployments := []any{
		aMap{
			"model_name": route, "litellm_params": aMap{"model": "openai/gpt-4o"},
			"model_info": aMap{
				"id": "deployment-a", "mode": "responses", "supports_reasoning": true, "supports_vision": true,
				"max_input_tokens": 128000, "max_output_tokens": 16384,
				"input_cost_per_token": 0.000005, "output_cost_per_token": 0.000015,
				"cache_read_input_token_cost": 0.0000025, "cache_creation_input_token_cost": 0,
			},
		},
		aMap{
			"model_name": route, "litellm_params": aMap{"model": "internal/unknown"},
			"model_info": aMap{
				"id": "deployment-b", "mode": "chat", "supports_reasoning": false, "supports_vision": false,
				"max_input_tokens": 64000, "max_output_tokens": 8192,
			},
		},
	}
	options, stderr := testOptions(t)
	for _, rows := range [][]any{deployments, {deployments[1], deployments[0]}} {
		base := aServe(t, rows...)

		result := discover(t, base, options)

		if len(result.Models) != 1 {
			t.Fatalf("models = %+v", result.Models)
		}
		model := result.Models[0]
		aExpectIncomplete(t, model, route, 64000, 8192)
		if model.API != ai.APIOpenAICompletions || model.ThinkingLevelMap != nil {
			t.Errorf("api = %q, levels = %v", model.API, model.ThinkingLevelMap)
		}
	}
	if lines := stderr.Lines(); len(lines) != 1 {
		t.Errorf("diagnostics = %q, want one", lines)
	}
}

func TestModelInfoA_MapsTheAdapterConservativelyToItsCatalog(t *testing.T) {
	for _, tc := range []struct{ adapter, backend string }{
		{"moonshot", "moonshot/kimi-k2.6"},
		{"gemini", "gemini/gemini-3.1-pro-preview"},
		{"xai", "xai/grok-4.5"},
	} {
		t.Run(tc.adapter, func(t *testing.T) {
			base := aServe(t, aMap{
				"model_name":     tc.adapter + "-route",
				"litellm_params": aMap{"model": tc.backend},
				"model_info":     aMap{"mode": "chat", "litellm_provider": tc.adapter},
			})
			options, _ := testOptions(t)
			options.ModelsDev = nil

			model := firstModel(t, base, options)

			if model.Name != tc.adapter+"-route" || strings.Contains(model.Name, "no metadata") ||
				model.ContextWindow <= 128_000 || model.Cost.Input <= 0 || model.ID != tc.adapter+"-route" {
				t.Errorf("model = %+v", model)
			}
			if tc.adapter == "gemini" && (model.LiteLLMPolicy == nil || !model.LiteLLMPolicy.NormalizeGeminiReasoningEffort) {
				t.Errorf("policy = %+v, want normalizeGeminiReasoningEffort", model.LiteLLMPolicy)
			}
		})
	}
}

func TestModelInfoA_TrimsBackendCandidatesBeforeCatalogResolution(t *testing.T) {
	base := aServe(t, aMap{
		"model_name": "spacey", "model_info": aMap{"mode": "chat", "base_model": " openai/gpt-4o "},
	})

	model := firstModel(t, base, aOffline(t))

	if model.Name != "spacey" || !reflect.DeepEqual(model.Input, []string{"text", "image"}) || model.ContextWindow != 128_000 ||
		model.Cost.Input <= 0 {
		t.Errorf("model = %+v", model)
	}
}

func TestModelInfoA_DoesNotUseAQualifiedPublicRouteAsEvidenceForAMultiDeploymentGroup(t *testing.T) {
	route := "openai/gpt-5.5-group"
	base := aServe(t,
		aMap{"model_name": route, "litellm_params": aMap{"model": "openai/gpt-5.5"}, "model_info": aMap{"id": "known", "mode": "chat"}},
		aMap{"model_name": route, "litellm_params": aMap{"model": "internal/mystery"}, "model_info": aMap{"id": "unknown", "mode": "chat"}},
	)
	options, stderr := testOptions(t)

	model := firstModel(t, base, options)

	aExpectIncomplete(t, model, route, 128_000, 16_384)
	if model.ThinkingLevelMap != nil {
		t.Errorf("levels = %v, want none", model.ThinkingLevelMap)
	}
	if lines := stderr.Lines(); len(lines) != 1 || !strings.Contains(lines[0], route) {
		t.Errorf("diagnostics = %q", lines)
	}
}

func TestModelInfoA_WithholdsCatalogMetadataForASingletonWithOpaqueEvidence(t *testing.T) {
	for _, tc := range []struct {
		name     string
		evidence aMap
		info     aMap
	}{
		{"opaque backend model", aMap{"litellm_params": aMap{"model": "internal/mystery"}}, aMap{}},
		{"opaque base model", nil, aMap{"base_model": "internal/mystery"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			info := aMap{"id": "only", "mode": "chat"}
			for key, value := range tc.info {
				info[key] = value
			}
			row := aMap{"model_name": "openai/gpt-5.5", "model_info": info}
			for key, value := range tc.evidence {
				row[key] = value
			}
			base := aServe(t, row)

			model := firstModel(t, base, aOffline(t))

			aExpectIncomplete(t, model, "openai/gpt-5.5", 128_000, 16_384)
			if model.ThinkingLevelMap != nil {
				t.Errorf("levels = %v, want none", model.ThinkingLevelMap)
			}
		})
	}
}

func TestModelInfoA_TreatsRepeatedIdentifiedDeploymentRowsAsOneEffectiveSingleton(t *testing.T) {
	deployment := aMap{
		"model_name": "openai/gpt-5.5", "litellm_params": aMap{"model": "openai/gpt-5.5"},
		"model_info": aMap{"id": "deployment-a", "mode": "chat"},
	}
	for _, rows := range [][]any{{deployment}, {deployment, deployment}} {
		base := aServe(t, rows...)

		model := firstModel(t, base, aOffline(t))

		if model.ID != "openai/gpt-5.5" || model.Name != "openai/gpt-5.5" || !model.Reasoning ||
			model.ContextWindow != 272_000 || model.MaxTokens != 128_000 {
			t.Errorf("%d rows: model = %+v", len(rows), model)
		}
	}
}

func TestModelInfoA_KeepsUnresolvedAzureFoundryBackendEvidenceIncomplete(t *testing.T) {
	base := aServe(t, aMap{
		"model_name":     "foundry-route",
		"litellm_params": aMap{"model": " azure_ai/DeepSeek-V4 "},
		"model_info":     aMap{"mode": "chat", "litellm_provider": "azure_ai"},
	})

	model := firstModel(t, base, aOffline(t))

	if model.ID != "foundry-route" || model.Name != "foundry-route (incomplete metadata)" || model.Reasoning ||
		model.ContextWindow != 128_000 {
		t.Errorf("model = %+v", model)
	}
}

func TestModelInfoA_UsesTheBackendCandidateThatSuppliesCatalogAuthority(t *testing.T) {
	mixed := ResolveModelInfoCatalog(entryFrom(t, `{"model_name":"mixed-evidence","litellm_params":{"model":"internal/claude-magic"},`+
		`"model_info":{"mode":"chat","base_model":"openai/gpt-4o"}}`), nil)
	if mixed == nil || mixed.Provider != "openai" {
		t.Errorf("mixed = %+v", mixed)
	}
	aliased := ResolveModelInfoCatalog(entryFrom(t, `{"model_name":"aliased","litellm_params":{"model":"anthropic/opus-4-7"},`+
		`"model_info":{"mode":"chat"}}`), nil)
	if aliased == nil || aliased.Provider != "anthropic" || aliased.CatalogModelID != "claude-opus-4-7" {
		t.Errorf("aliased = %+v", aliased)
	}
}

func TestModelInfoA_UsesBaseModelAsAuthorityOverConfiguredModelAndTransportAdapter(t *testing.T) {
	resolved := ResolveModelInfoCatalog(entryFrom(t, `{"model_name":"kimi-k3",`+
		`"litellm_params":{"model":"azure_ai/FW-Kimi-K3","custom_llm_provider":"azure"},`+
		`"model_info":{"mode":"chat","litellm_provider":"azure","base_model":"fireworks_ai/accounts/fireworks/models/kimi-k3"}}`), nil)

	if resolved == nil || resolved.Provider != "fireworks_ai" || resolved.CatalogModelID != "accounts/fireworks/models/kimi-k3" {
		t.Errorf("resolved = %+v", resolved)
	}
}

func TestModelInfoA_UsesCustomLlmProviderAsCatalogAuthorityForAnUnprefixedBackendModel(t *testing.T) {
	resolved := ResolveModelInfoCatalog(entryFrom(t, `{"model_name":"kimi-route",`+
		`"litellm_params":{"model":"kimi-k3","custom_llm_provider":"moonshot"},"model_info":{"mode":"chat"}}`), nil)

	if resolved == nil || resolved.Provider != "moonshot" || resolved.CatalogModelID != "kimi-k3" {
		t.Errorf("resolved = %+v", resolved)
	}
}

func TestModelInfoA_WithholdsCatalogAuthorityWhenCustomLlmProviderConflictsWithTheModelPrefix(t *testing.T) {
	row := aMap{
		"model_name":     "openai/gpt-4o",
		"litellm_params": aMap{"model": "openai/gpt-4o", "custom_llm_provider": "anthropic"},
		"model_info":     aMap{"mode": "chat"},
	}
	encoded, _ := json.Marshal(row)
	if resolved := ResolveModelInfoCatalog(entryFrom(t, string(encoded)), nil); resolved != nil {
		t.Errorf("resolved = %+v, want none", resolved)
	}
	base := aServe(t, row)

	model := firstModel(t, base, aOffline(t))

	aExpectIncomplete(t, model, "openai/gpt-4o", 128_000, 16_384)
}

func TestModelInfoA_KeepsTheCatalogProviderStableAcrossAsymmetricPublicCatalogHits(t *testing.T) {
	public := aCatalog{suffix: "kimi-k3", record: catalog.PublicCatalogRecord{
		Source: "models.dev", Provider: "fireworks-ai", ModelID: "accounts/fireworks/models/kimi-k3",
		Limits: &catalog.Limits{Context: aFloat(131_072)},
	}}
	entry := func(model string) string {
		return fmt.Sprintf(`{"model_name":"kimi-route","litellm_params":{"model":%q},"model_info":{"mode":"chat"}}`, model)
	}

	for _, model := range []string{
		"fireworks_ai/accounts/fireworks/models/kimi-k3",
		"fireworks_ai/accounts/fireworks/models/other",
	} {
		resolved := ResolveModelInfoCatalog(entryFrom(t, entry(model)), public)
		if resolved == nil || resolved.Provider != "fireworks_ai" {
			t.Errorf("%s: resolved = %+v", model, resolved)
		}
	}
}

func TestModelInfoA_CanonicalizesAModelsDevOnlyModelAcrossQualifiedAndBareSpellings(t *testing.T) {
	public := aCatalog{suffix: "claude-future", record: catalog.PublicCatalogRecord{
		Source: "models.dev", Provider: "anthropic", ModelID: "claude-future",
		Limits: &catalog.Limits{Context: aFloat(200_000)},
	}}
	entry := func(model, custom string) string {
		params := fmt.Sprintf(`{"model":%q`, model)
		if custom != "" {
			params += fmt.Sprintf(`,"custom_llm_provider":%q`, custom)
		}
		return `{"model_name":"future-route","litellm_params":` + params + `},"model_info":{"mode":"chat"}}`
	}

	qualified := ResolveModelInfoCatalog(entryFrom(t, entry("anthropic/claude-future", "")), public)
	bare := ResolveModelInfoCatalog(entryFrom(t, entry("claude-future", "anthropic")), public)
	suffixed := ResolveModelInfoCatalog(entryFrom(t, entry("anthropic/us.claude-future", "")), public)

	if qualified == nil || qualified.Provider != "anthropic" || qualified.CatalogModelID != "claude-future" ||
		qualified.ContextWindow == nil || *qualified.ContextWindow != 200_000 {
		t.Errorf("qualified = %+v", qualified)
	}
	if bare == nil || bare.CatalogModelID != "claude-future" {
		t.Errorf("bare = %+v", bare)
	}
	if suffixed == nil || suffixed.CatalogModelID != "claude-future" {
		t.Errorf("suffixed = %+v", suffixed)
	}
}

func TestModelInfoA_UsesBaseModelBeforeAConflictingConfiguredModel(t *testing.T) {
	resolved := ResolveModelInfoCatalog(entryFrom(t, `{"model_name":"conflicting-evidence","litellm_params":{"model":"openai/gpt-4o"},`+
		`"model_info":{"mode":"chat","litellm_provider":"openai","base_model":"anthropic/opus-4-7"}}`), nil)

	if resolved == nil || resolved.Provider != "anthropic" || resolved.CatalogModelID != "claude-opus-4-7" {
		t.Errorf("resolved = %+v", resolved)
	}
}

func TestModelInfoA_GrantsCatalogAuthorityWhenPrefixedAndFamilyOnlyDeploymentsIdentifyOneModel(t *testing.T) {
	route := "anthropic-alias-group"
	base := aServe(t,
		aMap{"model_name": route, "litellm_params": aMap{"model": "anthropic/claude-opus-4-7"}, "model_info": aMap{"id": "canonical", "mode": "chat"}},
		aMap{"model_name": route, "litellm_params": aMap{"model": "opus-4.7"}, "model_info": aMap{"id": "family-only", "mode": "chat"}},
	)
	options, stderr := testOptions(t)

	model := firstModel(t, base, options)

	if model.ID != route || model.Name != route || !model.Reasoning || !reflect.DeepEqual(model.Input, []string{"text", "image"}) ||
		model.ContextWindow != 1_000_000 || model.MaxTokens != 128_000 ||
		aRates(model.Cost) != [4]float64{5, 25, 0.5, 6.25} {
		t.Errorf("model = %+v", model)
	}
	if lines := stderr.Lines(); len(lines) != 0 {
		t.Errorf("diagnostics = %q, want none", lines)
	}
}

// aConflictingGroup serves two single-model deployments of one route and checks the withheld result.
func aConflictingGroup(t *testing.T, route string, rows ...any) (types.DiscoveredModel, []string) {
	t.Helper()
	base := aServe(t, rows...)
	options, stderr := testOptions(t)
	return firstModel(t, base, options), stderr.Lines()
}

func TestModelInfoA_WithholdsCatalogAuthorityForGenuinelyDifferentAnthropicModels(t *testing.T) {
	route := "anthropic-model-conflict"
	model, lines := aConflictingGroup(t, route,
		aMap{"model_name": route, "litellm_params": aMap{"model": "anthropic/claude-opus-4-7"}, "model_info": aMap{"id": "opus", "mode": "chat"}},
		aMap{"model_name": route, "litellm_params": aMap{"model": "anthropic/claude-sonnet-4-6"}, "model_info": aMap{"id": "sonnet", "mode": "chat"}},
	)

	aExpectIncomplete(t, model, route, 128_000, 16_384)
	if len(lines) != 1 || !strings.Contains(lines[0], "missing or conflicting deployment provider evidence") ||
		!strings.Contains(lines[0], route) {
		t.Errorf("diagnostics = %q", lines)
	}
}

func TestModelInfoA_WithholdsDifferentConcreteCatalogIdentitiesFromOneProviderAcrossARouteGroup(t *testing.T) {
	route := "same-provider-group-conflict"
	model, lines := aConflictingGroup(t, route,
		aMap{"model_name": route, "litellm_params": aMap{"model": "openai/gpt-4o"}, "model_info": aMap{"id": "gpt-4o", "mode": "chat", "litellm_provider": "openai"}},
		aMap{"model_name": route, "litellm_params": aMap{"model": "openai/gpt-4.1"}, "model_info": aMap{"id": "gpt-4.1", "mode": "chat", "litellm_provider": "openai"}},
	)

	aExpectIncomplete(t, model, route, 128_000, 16_384)
	if model.ThinkingLevelMap != nil {
		t.Errorf("levels = %v, want none", model.ThinkingLevelMap)
	}
	if len(lines) != 1 || !strings.Contains(lines[0], "missing or conflicting deployment provider evidence") ||
		!strings.Contains(lines[0], route) {
		t.Errorf("diagnostics = %q", lines)
	}
}

func TestModelInfoA_PreservesBedrockCatalogAuthority(t *testing.T) {
	resolved := ResolveModelInfoCatalog(entryFrom(t, `{"model_name":"bedrock-claude-route",`+
		`"litellm_params":{"model":"bedrock/anthropic.claude-sonnet-4-6"},"model_info":{"mode":"chat","litellm_provider":"bedrock"}}`), nil)

	if resolved == nil || resolved.Provider != "bedrock" {
		t.Errorf("resolved = %+v", resolved)
	}
}

func TestModelInfoA_WithholdsACrossHostClaudeRouteSpanningVertexAndBedrock(t *testing.T) {
	route := "cross-host-claude"
	model, lines := aConflictingGroup(t, route,
		aMap{
			"model_name": route, "litellm_params": aMap{"model": "vertex_ai/claude-sonnet-4@20250514"},
			"model_info": aMap{
				"id": "vertex", "mode": "chat", "litellm_provider": "vertex_ai", "supports_reasoning": true,
				"supports_vision": true, "max_input_tokens": 96_000, "max_output_tokens": 8_000,
			},
		},
		aMap{
			"model_name": route, "litellm_params": aMap{"model": "bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0"},
			"model_info": aMap{
				"id": "bedrock", "mode": "chat", "litellm_provider": "bedrock", "supports_reasoning": false,
				"supports_vision": false, "max_input_tokens": 64_000, "max_output_tokens": 4_000,
			},
		},
	)

	aExpectIncomplete(t, model, route, 64_000, 4_000)
	if model.ThinkingLevelMap != nil {
		t.Errorf("levels = %v, want none", model.ThinkingLevelMap)
	}
	if len(lines) != 1 || !strings.Contains(lines[0], "missing or conflicting deployment provider evidence") ||
		!strings.Contains(lines[0], route) {
		t.Errorf("diagnostics = %q", lines)
	}
}

func TestModelInfoA_DoesNotEnrichAnUnqualifiedOpenAIPublicRouteWithoutBackendIdentity(t *testing.T) {
	base := aServe(t, aMap{"model_name": "gpt-4o", "model_info": aMap{"mode": "chat"}})

	model := firstModel(t, base, aOffline(t))

	aExpectIncomplete(t, model, "gpt-4o", 128_000, 16_384)
}

func TestModelInfoA_KeepsProvenDisplayPricesWhileMarkingAnyUnresolvedCostFieldIncomplete(t *testing.T) {
	// Display cost reduces per field: proven input/output survive, unresolved fields fall to zero, and the
	// model stays marked incomplete.
	priced := func(id string, input float64) aMap {
		return aMap{
			"model_name":     "priced-without-cache-rates",
			"litellm_params": aMap{"model": "internal/unknown"},
			"model_info": aMap{
				"id": id, "mode": "chat", "max_input_tokens": 32_000, "max_output_tokens": 4_000,
				"input_cost_per_token": input, "output_cost_per_token": 0.000015,
			},
		}
	}
	for _, tc := range []struct {
		rows          []any
		expectedInput float64
	}{
		{[]any{priced("only", 0.000003)}, 3},
		{[]any{priced("a", 0.000003), priced("b", 0.000004)}, 4},
	} {
		base := aServe(t, tc.rows...)

		result := discover(t, base, aOffline(t))

		if len(result.Models) != 1 {
			t.Fatalf("models = %+v", result.Models)
		}
		model := result.Models[0]
		if model.ID != "priced-without-cache-rates" || model.Name != "priced-without-cache-rates (incomplete metadata)" ||
			aRates(model.Cost) != [4]float64{tc.expectedInput, 15, 0, 0} || model.ContextWindow != 32_000 || model.MaxTokens != 4_000 {
			t.Errorf("%d rows: model = %+v", len(tc.rows), model)
		}
	}
}

func TestModelInfoA_MarksAnAmbiguousGroupIncompleteDespiteCompleteRouterPricingAndReportsWithheldAuthority(t *testing.T) {
	route := "priced-ambiguous"
	priced := func(id, model string) aMap {
		return aMap{
			"model_name": route, "litellm_params": aMap{"model": model},
			"model_info": aMap{
				"id": id, "mode": "chat", "input_cost_per_token": 0.000003, "output_cost_per_token": 0.000015,
				"cache_read_input_token_cost": 0.0000003, "cache_creation_input_token_cost": 0.00000375,
			},
		}
	}
	base := aServe(t, priced("openai", "openai/gpt-4o"), priced("anthropic", "anthropic/claude-sonnet-4-6"))
	options, stderr := testOptions(t)

	model := firstModel(t, base, options)

	if model.ID != route || model.Name != route+" (incomplete metadata)" || aRates(model.Cost) != [4]float64{3, 15, 0.3, 3.75} ||
		model.ContextWindow != 128_000 || model.MaxTokens != 16_384 {
		t.Errorf("model = %+v", model)
	}
	lines := stderr.Lines()
	if len(lines) != 1 || !strings.Contains(lines[0], route) ||
		!strings.Contains(lines[0], "catalog limits, pricing, and reasoning metadata are withheld") {
		t.Errorf("diagnostics = %q", lines)
	}
}

func TestModelInfoA_UsesResponsesCompatibilityMetadataForResponsesModeRoutes(t *testing.T) {
	// Compat is derived from the model id independently of transport, so a Responses-mode Moonshot route
	// retains the complete Kimi repair object.
	base := aServe(t, aMap{"model_name": "moonshot/kimi-k2.6", "model_info": aMap{"id": "one", "mode": "responses"}})

	model := firstModel(t, base, aOffline(t))

	denied := false
	want := &ai.ModelCompat{SupportsDeveloperRole: &denied, SupportsStrictMode: &denied}
	if model.API != ai.APIOpenAIResponses || !reflect.DeepEqual(model.Compat, want) {
		t.Errorf("api = %q, compat = %+v", model.API, model.Compat)
	}
}

func TestModelInfoA_UsesNativePromptCachingForAResponsesModeAlias(t *testing.T) {
	base := aServe(t, aMap{"model_name": "anthropic/claude-sonnet-4-6", "model_info": aMap{"id": "one", "mode": "responses"}})

	model := firstModel(t, base, aOffline(t))

	if model.API != ai.APIOpenAIResponses || model.Compat != nil {
		t.Errorf("api = %q, compat = %+v", model.API, model.Compat)
	}
}

func TestModelInfoA_WithholdsARowWithMalformedFieldsInsteadOfFailingTheWholeDiscovery(t *testing.T) {
	// One operator typo in proxy config must not cost every other model.
	for _, tc := range []struct {
		name string
		bad  aMap
	}{
		{"a numeric mode", aMap{"model_name": "bad-mode", "model_info": aMap{"id": "a", "mode": 7}}},
		{"a numeric deployment id", aMap{"model_name": "bad-id", "model_info": aMap{"id": 7, "mode": "chat"}}},
		{"a numeric route name", aMap{"model_name": 4.1, "model_info": aMap{"id": "a", "mode": "chat"}}},
		{"a numeric adapter", aMap{"model_name": "bad-adapter", "model_info": aMap{"id": "a", "mode": "chat", "litellm_provider": 7}}},
		{"a numeric backend model", aMap{"model_name": "bad-backend", "litellm_params": aMap{"model": 7}, "model_info": aMap{"id": "a", "mode": "chat"}}},
		{"a numeric base model", aMap{"model_name": "bad-base", "model_info": aMap{"id": "a", "mode": "chat", "base_model": 7}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			base := aServe(t, tc.bad, aMap{
				"model_name": "healthy-route", "litellm_params": aMap{"model": "openai/gpt-4o"},
				"model_info": aMap{"id": "b", "mode": "chat"},
			})

			result := discover(t, base, aOffline(t))

			found := false
			for _, model := range result.Models {
				if model.ID == "healthy-route" {
					found = true
					if model.Cost.Input <= 0 {
						t.Errorf("healthy-route cost = %+v", model.Cost)
					}
				}
			}
			if !found {
				t.Errorf("models = %+v, want healthy-route", result.Models)
			}
		})
	}
}

func aModelIDs(models []types.DiscoveredModel) []string {
	ids := []string{}
	for _, model := range models {
		ids = append(ids, model.ID)
	}
	return ids
}

func TestModelInfoA_WithholdsAFallbackOrHealthEntryWhoseIdIsNotAString(t *testing.T) {
	// The same invariant on the two paths that build a model straight from an id.
	base := mockEndpoints(t, map[string]http.HandlerFunc{
		"/model/info": jsonResponse(403, aMap{}),
		"/v1/models":  jsonResponse(200, aMap{"data": []any{aMap{"id": 7}, aMap{"id": "gpt-4o", "owned_by": 7}}}),
	})
	if ids := aModelIDs(discover(t, base, aOffline(t)).Models); !reflect.DeepEqual(ids, []string{"gpt-4o"}) {
		t.Errorf("fallback ids = %v", ids)
	}

	base = mockEndpoints(t, map[string]http.HandlerFunc{
		"/model/info": jsonResponse(403, aMap{}),
		"/v1/models":  jsonResponse(404, aMap{}),
		"/health": jsonResponse(200, aMap{"healthy_endpoints": []any{
			aMap{"model": 7}, aMap{"model": "anthropic/claude-opus-4-7"},
		}}),
	})
	if ids := aModelIDs(discover(t, base, aOffline(t)).Models); !reflect.DeepEqual(ids, []string{"anthropic/claude-opus-4-7"}) {
		t.Errorf("health ids = %v", ids)
	}
}

func TestModelInfoA_FallsBackToTheHealthRouteNameWithoutGrantingDeploymentThinkingLevels(t *testing.T) {
	// The detail row's model_name is unusable, but /health named the route, so the model must survive under
	// that name without treating the route as level authority.
	base := mockEndpoints(t, map[string]http.HandlerFunc{
		"/model/info?litellm_model_id=uuid-1": jsonResponse(200, aMap{"data": []any{aMap{
			"model_name": 7,
			"model_info": aMap{"mode": "chat", "supports_reasoning": true, "supports_low_reasoning_effort": true},
		}}}),
		"/model/info": jsonResponse(403, aMap{}),
		"/v1/models":  jsonResponse(404, aMap{}),
		"/health": jsonResponse(200, aMap{"healthy_endpoints": []any{
			aMap{"model": "named-route", "model_id": "uuid-1"},
		}}),
	})

	result := discover(t, base, aOffline(t))

	if result.Source != types.SourceHealth || !reflect.DeepEqual(aModelIDs(result.Models), []string{"named-route"}) {
		t.Fatalf("source = %q, ids = %v", result.Source, aModelIDs(result.Models))
	}
	want := aLevels(map[string]string{
		"off": "null", "minimal": "null", "low": "null", "medium": "null", "high": "null", "xhigh": "null", "max": "null",
	})
	if !reflect.DeepEqual(result.Models[0].ThinkingLevelMap, want) {
		t.Errorf("levels = %v, want every level denied", result.Models[0].ThinkingLevelMap)
	}
}

func TestModelInfoA_WithholdsAHealthDeploymentWhoseRouteNameIsNotAString(t *testing.T) {
	// This path bypasses the grouping loop: /health supplies the route name directly to the reducer.
	base := mockEndpoints(t, map[string]http.HandlerFunc{
		"/model/info?litellm_model_id=uuid-1": jsonResponse(200, aMap{"data": []any{aMap{"model_info": aMap{"mode": "chat"}}}}),
		"/model/info":                         jsonResponse(403, aMap{}),
		"/v1/models":                          jsonResponse(404, aMap{}),
		"/health": jsonResponse(200, aMap{"healthy_endpoints": []any{
			aMap{"model": 7, "model_id": "uuid-1"}, aMap{"model": "anthropic/claude-opus-4-7"},
		}}),
	})

	result := discover(t, base, aOffline(t))

	if ids := aModelIDs(result.Models); !reflect.DeepEqual(ids, []string{"anthropic/claude-opus-4-7"}) {
		t.Errorf("ids = %v", ids)
	}
}

func TestModelInfoA_SurvivesADeeplyNestedDeploymentRow(t *testing.T) {
	// Canonicalization is depth-bounded, so a pathological payload cannot exhaust the stack and take every
	// model down with it.
	const depth = 20_000
	nested := strings.Repeat(`{"nested":`, depth) + `"leaf"` + strings.Repeat("}", depth)
	payload := `{"data":[{"model_name":"deep-route","litellm_params":{"model":"openai/gpt-4o"},` +
		`"model_info":{"id":"a","mode":"chat","extra":` + nested + `}}]}`
	base := mockEndpoints(t, map[string]http.HandlerFunc{"/model/info": jsonResponse(200, json.RawMessage(payload))})

	result := discover(t, base, aOffline(t))

	if len(result.Models) == 0 || result.Models[0].ID != "deep-route" {
		t.Errorf("models = %+v", result.Models)
	}
}

func TestModelInfoA_NeverEmitsTheFallbackOnlySentinelForAReducedDeploymentGroup(t *testing.T) {
	// The " (no metadata)" suffix authorizes catalog re-derivation from the model id during offline cache
	// reads, so no /model/info group may carry it.
	groups := [][]any{
		{aMap{"model_name": "singleton-route", "model_info": aMap{"id": "one", "mode": "chat"}}},
		{
			aMap{"model_name": "plural-route", "model_info": aMap{"id": "a", "mode": "chat"}},
			aMap{"model_name": "plural-route", "model_info": aMap{"id": "b", "mode": "chat"}, "litellm_params": aMap{"model": "x/y"}},
		},
		{
			aMap{"model_name": "conflicting-route", "model_info": aMap{"id": "same", "mode": "chat"}},
			aMap{"model_name": "conflicting-route", "model_info": aMap{"id": "same", "mode": "chat", "max_input_tokens": 8_000}},
		},
		{
			aMap{"model_name": "embedding-sibling-route", "model_info": aMap{"id": "chat", "mode": "chat"}},
			aMap{"model_name": "embedding-sibling-route", "model_info": aMap{"id": "embed", "mode": "embedding"}},
		},
	}
	for _, data := range groups {
		route := data[0].(aMap)["model_name"].(string)
		t.Run(route, func(t *testing.T) {
			base := aServe(t, data...)
			options, stderr := testOptions(t)

			result := discover(t, base, options)

			if route == "embedding-sibling-route" {
				if len(result.Models) != 0 || len(stderr.Lines()) != 1 {
					t.Errorf("models = %+v, diagnostics = %q", result.Models, stderr.Lines())
				}
				return
			}
			if len(result.Models) != 1 {
				t.Fatalf("models = %+v", result.Models)
			}
			model := result.Models[0]
			if strings.Contains(model.Name, " (no metadata)") || model.Name != model.ID+" (incomplete metadata)" {
				t.Errorf("name = %q, id = %q", model.Name, model.ID)
			}
		})
	}
}
