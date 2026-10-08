package discover

// Ports the modelProtocol, buildCompat, Kimi reasoning compatibility, moonshotPolicy and enrichCachedModel
// describe blocks of tests/discover.test.ts.

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/modelgroups"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/proxyversion"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

func apiOf(t *testing.T, modelID string, modeOrEntry any, version *proxyversion.Version) ai.API {
	t.Helper()
	return ModelProtocol(modelID, modeOrEntry, version).API
}

func expectAPI(t *testing.T, got, want ai.API) {
	t.Helper()
	if got != want {
		t.Errorf("api = %s, want %s", got, want)
	}
}

func TestModelProtocol(t *testing.T) {
	t.Run("keeps an Azure deployment with a non-string API version on Chat", func(t *testing.T) {
		entry := entryFrom(t, `{"litellm_params":{"model":"azure/gpt-5","api_version":20240101}}`)
		expectAPI(t, apiOf(t, "opaque-route", entry, nil), ai.APIOpenAICompletions)
	})

	t.Run("does not infer a deployment protocol from its public route name", func(t *testing.T) {
		got := ModelProtocol("openai/gpt-5", entryFrom(t, `{"model_name":"openai/gpt-5"}`), nil)
		want := types.ModelProtocol{API: ai.APIOpenAICompletions, Compat: &ai.ModelCompat{SupportsStore: boolPtr(false)}}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("protocol = %+v, want %+v", got, want)
		}
	})

	t.Run("keeps OpenAI-compatible adapter routes of non-OpenAI models on Chat unless endpoints say otherwise", func(t *testing.T) {
		row := func(model, extra string) types.ModelInfoEntry {
			info := `"mode":"chat","supported_endpoints":null,"litellm_provider":"openai","supports_reasoning":true,` +
				`"supports_function_calling":true,"supported_openai_params":["tools","tool_choice","reasoning_effort"]`
			if extra != "" {
				// A later duplicate key overrides, like the object spread in the TypeScript row().
				info += "," + extra
			}
			return entryFrom(t, fmt.Sprintf(`{"litellm_params":{"model":%q},"model_info":{%s}}`, model, info))
		}
		for _, model := range []string{"openai/qwen3.8-27B-a", "openai/LongCat-2.0", "openai_like/qwen3", "custom_openai/llama-4"} {
			expectAPI(t, apiOf(t, "route", row(model, ""), nil), ai.APIOpenAICompletions)
		}
		expectAPI(t, apiOf(t, "route", row("openai/gpt-5.5", ""), nil), ai.APIOpenAIResponses)
		expectAPI(t, apiOf(t, "route", row("openai/qwen3.8-27B-a", `"supported_endpoints":["/v1/responses"]`), nil), ai.APIOpenAIResponses)
		expectAPI(t, apiOf(t, "route", row("openai/qwen3.8-27B-a", `"mode":"responses"`), nil), ai.APIOpenAIResponses)
	})

	t.Run("pairs each upstream-selected mode with protocol-specific compatibility", func(t *testing.T) {
		equal := func(got types.ModelProtocol, api ai.API, compat *ai.ModelCompat) {
			t.Helper()
			if want := (types.ModelProtocol{API: api, Compat: compat}); !reflect.DeepEqual(got, want) {
				t.Errorf("protocol = %+v (compat %+v), want %+v (compat %+v)", got, got.Compat, want, want.Compat)
			}
		}
		equal(ModelProtocol("openai/gpt-4o", nil, nil), ai.APIOpenAIResponses, nil)
		equal(ModelProtocol("openai/gpt-4o", "responses", nil), ai.APIOpenAIResponses, nil)
		for _, id := range []string{"anthropic/claude-sonnet-4-6", "fable-5", "sonnet-4.6"} {
			equal(ModelProtocol(id, "chat", nil), ai.APIOpenAICompletions,
				&ai.ModelCompat{SupportsStore: boolPtr(false), CacheControlFormat: "anthropic"})
			equal(ModelProtocol(id, "responses", nil), ai.APIOpenAIResponses, nil)
		}
		equal(ModelProtocol("moonshotai/kimi-k2", "responses", nil), ai.APIOpenAIResponses,
			&ai.ModelCompat{SupportsDeveloperRole: boolPtr(false)})
	})

	for _, provider := range []string{"azure", "azure_ai"} {
		t.Run("requires explicit Responses evidence for "+provider+" deployments", func(t *testing.T) {
			for _, apiVersion := range []string{"", "2024-12-01-preview", "2025-03-01", "2025-04-01-preview", "v1"} {
				version := ""
				if apiVersion != "" {
					version = fmt.Sprintf(`,"api_version":%q`, apiVersion)
				}
				for _, evidence := range []string{
					fmt.Sprintf(`{"litellm_params":{"model":"%s/gpt-5"%s}}`, provider, version),
					fmt.Sprintf(`{"litellm_params":{"model":"gpt-5","custom_llm_provider":%q%s}}`, provider, version),
					fmt.Sprintf(`{"litellm_params":{"model":"gpt-5"%s},"model_info":{"mode":"chat","litellm_provider":%q}}`, version, provider),
				} {
					expectAPI(t, apiOf(t, "opaque-route", entryFrom(t, evidence), nil), ai.APIOpenAICompletions)
				}
			}
		})
	}

	// Before v1.103.0 LiteLLM bridges every azure_ai Responses call through Chat Completions and that
	// bridge fails streaming with "list index out of range". The cost-map `supported_endpoints` list
	// describes the model, not that bridge, so without a known proxy version it must not promote the
	// route; only an explicit mode may.
	for _, params := range []string{
		`{"model":"azure_ai/gpt-6-astra"}`,
		`{"model":"gpt-6-astra","custom_llm_provider":"azure_ai"}`,
		`{"model":"azure_ai/gpt-5.5","api_version":"2025-04-01-preview"}`,
	} {
		t.Run("keeps azure_ai deployments on Chat despite a /v1/responses endpoint list: "+params, func(t *testing.T) {
			for _, provider := range []string{"azure", "azure_ai", ""} {
				adapter := ""
				if provider != "" {
					adapter = fmt.Sprintf(`,"litellm_provider":%q`, provider)
				}
				entry := entryFrom(t, fmt.Sprintf(
					`{"litellm_params":%s,"model_info":{"mode":"chat","supported_endpoints":["/v1/chat/completions","/v1/responses"]%s}}`, params, adapter))
				expectAPI(t, apiOf(t, "gpt-6-astra", entry, nil), ai.APIOpenAICompletions)
			}
			entry := entryFrom(t, fmt.Sprintf(
				`{"litellm_params":%s,"model_info":{"mode":"responses","supported_endpoints":["/v1/chat/completions","/v1/responses"]}}`, params))
			expectAPI(t, apiOf(t, "gpt-6-astra", entry, nil), ai.APIOpenAIResponses)
		})
	}

	t.Run("still lets a native azure deployment pin Responses through supported_endpoints", func(t *testing.T) {
		entry := entryFrom(t, `{"litellm_params":{"model":"azure/gpt-6-astra"},`+
			`"model_info":{"mode":"chat","litellm_provider":"azure_ai","supported_endpoints":["/v1/responses"]}}`)
		expectAPI(t, apiOf(t, "gpt-6-astra", entry, nil), ai.APIOpenAIResponses)
	})

	t.Run("on a proxy of a known version", func(t *testing.T) {
		azureAIRows := []string{
			`"litellm_params":{"model":"azure_ai/gpt-6-astra"}`,
			`"litellm_params":{"model":"gpt-6-astra","custom_llm_provider":"azure_ai"}`,
		}
		listsResponses := `{"mode":"chat","supported_endpoints":["/v1/chat/completions","/v1/responses"]}`
		entryOf := func(row, info string) types.ModelInfoEntry {
			return entryFrom(t, fmt.Sprintf(`{%s,"model_info":%s}`, row, info))
		}

		for _, version := range []string{"1.103.0", "1.103.1", "1.104.0-rc.1", "2.0.0"} {
			t.Run("lets an azure_ai endpoint list select Responses on "+version, func(t *testing.T) {
				for _, row := range azureAIRows {
					expectAPI(t, apiOf(t, "gpt-6-astra", entryOf(row, listsResponses), proxyversion.Parse(version)), ai.APIOpenAIResponses)
				}
			})
		}
		for _, version := range []string{"1.102.0", "1.102.1", "1.103.0-rc.1", "1.103.0.dev2"} {
			t.Run("keeps azure_ai deployments on Chat on "+version, func(t *testing.T) {
				for _, row := range azureAIRows {
					expectAPI(t, apiOf(t, "gpt-6-astra", entryOf(row, listsResponses), proxyversion.Parse(version)), ai.APIOpenAICompletions)
				}
			})
		}
		t.Run("does not promote an azure_ai deployment that lists no endpoints", func(t *testing.T) {
			for _, row := range azureAIRows {
				for _, provider := range []string{"azure", "azure_ai", ""} {
					info := `{"mode":"chat"}`
					if provider != "" {
						info = fmt.Sprintf(`{"mode":"chat","litellm_provider":%q}`, provider)
					}
					expectAPI(t, apiOf(t, "gpt-6-astra", entryOf(row, info), proxyversion.Parse("1.103.0")), ai.APIOpenAICompletions)
				}
			}
		})
		t.Run("still denies an azure_ai endpoint list that omits /v1/responses", func(t *testing.T) {
			for _, row := range azureAIRows {
				info := `{"mode":"responses","supported_endpoints":["/v1/chat/completions"]}`
				expectAPI(t, apiOf(t, "gpt-6-astra", entryOf(row, info), proxyversion.Parse("1.103.0")), ai.APIOpenAICompletions)
			}
		})
	})

	for _, tc := range []struct {
		info string
		api  ai.API
	}{
		{`{"mode":"responses"}`, ai.APIOpenAIResponses},
		{`{"mode":"chat","supported_endpoints":["/v1/responses"]}`, ai.APIOpenAIResponses},
		{`{"mode":"responses","supported_endpoints":["/v1/chat/completions"]}`, ai.APIOpenAICompletions},
		{`{"mode":"responses","supported_endpoints":[]}`, ai.APIOpenAICompletions},
	} {
		t.Run("honors explicit Azure transport evidence: "+tc.info, func(t *testing.T) {
			entry := entryFrom(t, fmt.Sprintf(
				`{"litellm_params":{"model":"azure/gpt-5","api_version":"2025-04-01-preview"},"model_info":%s}`, tc.info))
			expectAPI(t, apiOf(t, "opaque-route", entry, nil), tc.api)
		})
	}

	t.Run("uses Chat Completions when the model prefix conflicts with custom_llm_provider", func(t *testing.T) {
		entry := entryFrom(t, `{"model_name":"gpt-prod","litellm_params":{"model":"openai/gpt-5","custom_llm_provider":"fireworks_ai"},`+
			`"model_info":{"mode":"chat"}}`)
		expectAPI(t, apiOf(t, "gpt-prod", entry, nil), ai.APIOpenAICompletions)
	})
}

func TestBuildCompat(t *testing.T) {
	storeOff := &ai.ModelCompat{SupportsStore: boolPtr(false)}
	kimi := &ai.ModelCompat{
		SupportsStore:           boolPtr(false),
		SupportsDeveloperRole:   boolPtr(false),
		SupportsReasoningEffort: boolPtr(false),
		SupportsStrictMode:      boolPtr(false),
		MaxTokensField:          "max_tokens",
	}
	anthropic := &ai.ModelCompat{SupportsStore: boolPtr(false), CacheControlFormat: "anthropic"}
	check := func(t *testing.T, id string, want *ai.ModelCompat) {
		t.Helper()
		if got := BuildCompat(id); !reflect.DeepEqual(got, want) {
			t.Errorf("BuildCompat(%q) = %+v, want %+v", id, got, want)
		}
	}

	t.Run("returns supportsStore: false for non-anthropic models", func(t *testing.T) {
		for _, id := range []string{"openai/gpt-4o", "gemini/gemini-2.0-flash", "gpt-5.5"} {
			check(t, id, storeOff)
		}
	})
	t.Run("adds Moonshot-compatible tool calling flags for Kimi models", func(t *testing.T) {
		check(t, "kimi-k2.6", kimi)
		check(t, "moonshotai/kimi-k2", kimi)
	})
	t.Run("adds cacheControlFormat for anthropic-prefixed models", func(t *testing.T) {
		check(t, "anthropic/claude-3-5-sonnet", anthropic)
	})
	t.Run("adds cacheControlFormat for bare Claude aliases", func(t *testing.T) {
		for _, id := range []string{"claude-3-5-sonnet", "opus-4.7", "sonnet-4.6", "haiku-4.5"} {
			check(t, id, anthropic)
		}
	})
	t.Run("adds cacheControlFormat for routed Anthropic aliases", func(t *testing.T) {
		check(t, "google/claude-sonnet-4-6", anthropic)
	})
	t.Run("does not match non-Anthropic tokens that start with Anthropic family names", func(t *testing.T) {
		check(t, "openai/sonnetic-gpt", storeOff)
		check(t, "vendor/opusflow", storeOff)
	})
	t.Run("matches case-insensitively", func(t *testing.T) {
		check(t, "Opus-4.7", anthropic)
		check(t, "CLAUDE-3-5-SONNET", anthropic)
	})
}

func TestKimiReasoningCompatibility(t *testing.T) {
	t.Run("still parses think tags for unknown routes that look like Kimi", func(t *testing.T) {
		if !EmitsThinkTags("kimi-k3") {
			t.Error("kimi-k3 should emit think tags")
		}
		if EmitsThinkTags("openai/gpt-4o") {
			t.Error("openai/gpt-4o should not emit think tags")
		}
	})
}

func TestMoonshotPolicy(t *testing.T) {
	t.Run("keeps request suppression disabled for route-name-only fallback evidence", func(t *testing.T) {
		want := types.LiteLLMModelPolicy{NormalizeThinkTags: true}
		if got := MoonshotPolicy("kimi-k2.6", false); got != want {
			t.Errorf("policy = %+v, want %+v", got, want)
		}
	})
	t.Run("preserves always-thinking output and visibility behavior", func(t *testing.T) {
		if got := MoonshotPolicy("kimi-k2-thinking", false); got != (types.LiteLLMModelPolicy{}) {
			t.Errorf("policy = %+v, want all false", got)
		}
	})
}

func cachedReasoningModel(api ai.API, overrides ...func(*types.DiscoveredModel)) types.DiscoveredModel {
	model := types.DiscoveredModel{
		ID:            "cached-reasoning",
		Name:          "Cached reasoning",
		Provider:      "litellm",
		API:           api,
		BaseURL:       "https://litellm.example.com/v1",
		Reasoning:     true,
		Input:         []string{"text"},
		ContextWindow: 128_000,
		MaxTokens:     16_384,
	}
	for _, override := range overrides {
		override(&model)
	}
	return model
}

func levels(pairs ...string) ai.ThinkingLevelMap {
	out := ai.ThinkingLevelMap{}
	for i := 0; i < len(pairs); i += 2 {
		if pairs[i+1] == "" {
			out[ai.ThinkingLevel(pairs[i])] = nil
			continue
		}
		out[ai.ThinkingLevel(pairs[i])] = &pairs[i+1]
	}
	return out
}

func TestEnrichCachedModelFallbackContextWindow(t *testing.T) {
	cachedFallback := func(contextWindow int, id string) types.DiscoveredModel {
		return cachedReasoningModel(ai.APIOpenAICompletions, func(m *types.DiscoveredModel) {
			m.ID, m.Name, m.Reasoning, m.ContextWindow = id, id+" (no metadata)", false, contextWindow
		})
	}

	t.Run("re-applies the configured default to a model cached under the old one", func(t *testing.T) {
		t.Setenv("LITELLM_DEFAULT_CONTEXT_WINDOW", "922000")

		if got := EnrichCachedModel(cachedFallback(128_000, "private-route")).ContextWindow; got != 922_000 {
			t.Errorf("contextWindow = %d, want 922000", got)
		}
		// A window already stored under the setting is left where it is.
		if got := EnrichCachedModel(cachedFallback(922_000, "private-route")).ContextWindow; got != 922_000 {
			t.Errorf("contextWindow = %d, want 922000", got)
		}
	})

	t.Run("still enriches a cached fallback from the catalog after the setting changes", func(t *testing.T) {
		t.Setenv("LITELLM_DEFAULT_CONTEXT_WINDOW", "922000")

		enriched := EnrichCachedModel(cachedFallback(128_000, "claude-haiku-4-5"))

		// Catalog evidence outranks the fallback; the setting only fills a gap.
		if strings.Contains(enriched.Name, "(no metadata)") {
			t.Errorf("name = %q still carries the fallback marker", enriched.Name)
		}
		if enriched.ContextWindow == 922_000 {
			t.Errorf("contextWindow = %d, want the catalog window", enriched.ContextWindow)
		}
	})

	t.Run("leaves a model that carries real metadata alone", func(t *testing.T) {
		t.Setenv("LITELLM_DEFAULT_CONTEXT_WINDOW", "922000")

		// A window matching neither default is partial enrichment, not an assumption.
		if got := EnrichCachedModel(cachedFallback(128_001, "private-route")).ContextWindow; got != 128_001 {
			t.Errorf("contextWindow = %d, want 128001", got)
		}
		measured := cachedReasoningModel(ai.APIOpenAICompletions, func(m *types.DiscoveredModel) {
			m.Reasoning, m.ContextWindow = false, 64_000
		})
		if got := EnrichCachedModel(measured).ContextWindow; got != 64_000 {
			t.Errorf("contextWindow = %d, want 64000", got)
		}
	})
}

func TestEnrichCachedModelReasoningPolicy(t *testing.T) {
	noReasoningLevels := modelgroups.NoTransmissibleLevels()

	t.Run("removes a stale thinking level map from a cached non-reasoning model", func(t *testing.T) {
		enriched := EnrichCachedModel(cachedReasoningModel(ai.APIOpenAICompletions, func(m *types.DiscoveredModel) {
			m.Reasoning, m.ThinkingLevelMap = false, levels("low", "low", "high", "high")
		}))

		if enriched.Reasoning || enriched.ThinkingLevelMap != nil {
			t.Errorf("reasoning = %v, thinkingLevelMap = %v; want false, none", enriched.Reasoning, enriched.ThinkingLevelMap)
		}
	})

	t.Run("denies Pi default Chat levels when a legacy cache has no level or carrier evidence", func(t *testing.T) {
		enriched := EnrichCachedModel(cachedReasoningModel(ai.APIOpenAICompletions))

		if !enriched.Reasoning || !reflect.DeepEqual(enriched.ThinkingLevelMap, noReasoningLevels) {
			t.Errorf("reasoning = %v, levels = %v", enriched.Reasoning, enriched.ThinkingLevelMap)
		}
	})

	t.Run("denies Pi default Chat levels after catalog enrichment has no level or carrier evidence", func(t *testing.T) {
		enriched := EnrichCachedModel(cachedReasoningModel(ai.APIOpenAICompletions, func(m *types.DiscoveredModel) {
			m.ID, m.Name, m.Reasoning = "claude-haiku-4-5", "claude-haiku-4-5 (no metadata)", false
		}))

		if enriched.Name != "Claude Haiku 4.5 (latest)" || !enriched.Reasoning ||
			!reflect.DeepEqual(enriched.ThinkingLevelMap, noReasoningLevels) {
			t.Errorf("name = %q, reasoning = %v, levels = %v", enriched.Name, enriched.Reasoning, enriched.ThinkingLevelMap)
		}
	})

	t.Run("updates cached fallback transport from the resolved Pi catalog entry", func(t *testing.T) {
		enriched := EnrichCachedModel(cachedReasoningModel(ai.APIOpenAICompletions, func(m *types.DiscoveredModel) {
			m.ID, m.Name, m.Reasoning = "openai/gpt-4o", "openai/gpt-4o (no metadata)", false
			m.Compat = &ai.ModelCompat{SupportsStore: boolPtr(false)}
		}))

		if enriched.Name != "GPT-4o" || enriched.API != ai.APIOpenAIResponses || enriched.Compat != nil {
			t.Errorf("name = %q, api = %s, compat = %+v", enriched.Name, enriched.API, enriched.Compat)
		}
	})

	t.Run("denies cached Chat levels when no compatibility metadata proves a carrier", func(t *testing.T) {
		enriched := EnrichCachedModel(cachedReasoningModel(ai.APIOpenAICompletions, func(m *types.DiscoveredModel) {
			m.ThinkingLevelMap = levels("low", "low", "high", "high")
		}))

		if !enriched.Reasoning || !reflect.DeepEqual(enriched.ThinkingLevelMap, noReasoningLevels) ||
			enriched.Compat == nil || enriched.Compat.SupportsReasoningEffort == nil || *enriched.Compat.SupportsReasoningEffort {
			t.Errorf("reasoning = %v, levels = %v, compat = %+v", enriched.Reasoning, enriched.ThinkingLevelMap, enriched.Compat)
		}
	})

	t.Run("closes cached Chat levels when compatibility metadata proves a carrier", func(t *testing.T) {
		enriched := EnrichCachedModel(cachedReasoningModel(ai.APIOpenAICompletions, func(m *types.DiscoveredModel) {
			m.ThinkingLevelMap = levels("low", "low", "high", "high")
			m.Compat = &ai.ModelCompat{SupportsReasoningEffort: boolPtr(true)}
		}))

		if want := levels("low", "low", "high", "high"); !reflect.DeepEqual(enriched.ThinkingLevelMap, want) {
			t.Errorf("levels = %v, want %v", enriched.ThinkingLevelMap, want)
		}
	})

	t.Run("keeps Responses cache behavior unchanged with and without explicit control evidence", func(t *testing.T) {
		plain := EnrichCachedModel(cachedReasoningModel(ai.APIOpenAIResponses))
		if !plain.Reasoning || !reflect.DeepEqual(plain.ThinkingLevelMap, noReasoningLevels) {
			t.Errorf("reasoning = %v, levels = %v", plain.Reasoning, plain.ThinkingLevelMap)
		}

		controlled := EnrichCachedModel(cachedReasoningModel(ai.APIOpenAIResponses, func(m *types.DiscoveredModel) {
			m.ThinkingLevelMap = levels("low", "low", "high", "high")
			m.LiteLLMResponsesReasoningControl = true
		}))
		want := levels("off", "none", "minimal", "minimal", "low", "low", "medium", "medium", "high", "high", "xhigh", "", "max", "")
		if !reflect.DeepEqual(controlled.ThinkingLevelMap, want) {
			t.Errorf("levels = %v, want %v", controlled.ThinkingLevelMap, want)
		}
	})
}
