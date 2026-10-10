// Ports tests/model-groups.test.ts: upstream reduction regressions, stableJson, native Messages route selection
package modelgroups

import (
	"encoding/json"
	"testing"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

func TestUpstreamReductionRegressions(t *testing.T) {
	t.Run("takes the per-field maximum from duplicate thresholds in one ladder", func(t *testing.T) {
		cost := modelCost(1, 2, 3, 4, tier(100, 10, 2, 30, 4), tier(100, 1, 20, 3, 40))

		eq(t, ConservativeCostTiers([]ai.ModelCost{cost}), []ai.CostTier{tier(100, 10, 20, 30, 40)})
	})

	t.Run("floors every matched tier field at its deployment base rate", func(t *testing.T) {
		cost := modelCost(10, 20, 3, 4, tier(100, 1, 2, 0.3, 0.4))

		eq(t, ConservativeCostTiers([]ai.ModelCost{cost}), []ai.CostTier{tier(100, 10, 20, 3, 4)})
	})

	// The Vitest rows for NaN and infinite thresholds cannot exist: ai.CostTier.InputTokensAbove is an int.
	t.Run("ignores a negative tier threshold", func(t *testing.T) {
		cost := modelCost(1, 2, 3, 4, tier(-1, 100, 200, 300, 400), tier(100, 10, 20, 30, 40))

		eq(t, ConservativeCostTiers([]ai.ModelCost{cost}), []ai.CostTier{tier(100, 10, 20, 30, 40)})
	})

	// TestMistypedWireFields ports the same case with the Vitest `model_name: 42`.
	t.Run("filters rows without a readable route name before reducing limits and capabilities", func(t *testing.T) {
		roomy := row(idMode("roomy", "chat"), info(func(i *types.ModelInfoDetails) { i.MaxInputTokens = f(200_000) }))
		badName := row(name(""), idMode("bad-name", "chat"), info(func(i *types.ModelInfoDetails) {
			i.SupportsReasoning, i.MaxInputTokens = b(false), f(8_000)
		}))

		got := mustReduce(t, []types.ModelInfoEntry{roomy, badName}, resolveCatalog)
		if got.ContextWindow != 200_000 || !got.Reasoning {
			t.Fatalf("%s", dump(got))
		}
	})

	t.Run("tracks metadata completeness independently from complete router pricing", func(t *testing.T) {
		explicit := row(backendModel("internal/unknown"), info(func(i *types.ModelInfoDetails) {
			i.ID, i.Mode, i.SupportsReasoning, i.SupportsVision = "explicit", s("chat"), b(false), b(false)
			i.MaxInputTokens, i.MaxOutputTokens = f(32_000), f(4_000)
		}))
		defaults := row(backendModel("internal/unknown"), info(func(i *types.ModelInfoDetails) {
			i.ID, i.SupportsReasoning, i.SupportsVision, i.MaxInputTokens, i.MaxOutputTokens = "defaults", nil, nil, nil, nil
		}))

		got := mustReduce(t, []types.ModelInfoEntry{explicit}, resolveCatalog)
		if !got.HasCompleteCost || !got.HasCompleteMetadata {
			t.Fatalf("%s", dump(got))
		}
		got = mustReduce(t, []types.ModelInfoEntry{defaults}, resolveCatalog)
		if !got.HasCompleteCost || got.HasCompleteMetadata {
			t.Fatalf("%s", dump(got))
		}
	})

	t.Run("resolves catalog metadata once per routable deployment", func(t *testing.T) {
		var seen []types.ModelInfoEntry
		record := func(entry types.ModelInfoEntry) *CatalogResolution {
			seen = append(seen, entry)
			return nil
		}
		chat := row(idMode("chat", "chat"))
		embedding := row(idMode("embed", "embedding"))
		other := row(idMode("other", "chat"))
		conflicting := row(idMode("chat", "chat"), info(func(i *types.ModelInfoDetails) { i.MaxInputTokens = f(8_000) }))

		ReduceModelGroup([]types.ModelInfoEntry{chat}, record)
		eq(t, seen, []types.ModelInfoEntry{chat})

		seen = nil
		ReduceModelGroup([]types.ModelInfoEntry{chat, chat}, record)
		eq(t, seen, []types.ModelInfoEntry{chat})

		// An incompatible sibling withholds the route before catalog resolution.
		seen = nil
		if ReduceModelGroup([]types.ModelInfoEntry{chat, embedding}, record) != nil {
			t.Fatal("expected nil")
		}
		if len(seen) != 0 {
			t.Fatalf("seen %d", len(seen))
		}

		seen = nil
		ReduceModelGroup([]types.ModelInfoEntry{chat, other}, record)
		eq(t, seen, []types.ModelInfoEntry{chat, other})

		seen = nil
		ReduceModelGroup([]types.ModelInfoEntry{chat, conflicting}, record)
		if len(seen) != 2 {
			t.Fatalf("seen %d", len(seen))
		}
	})

	t.Run("withholds groups containing an explicitly incompatible deployment mode", func(t *testing.T) {
		responses := row(idMode("responses", "responses"))
		chat := row(idMode("chat", "chat"))
		unsupported := row(idMode("embed", "embedding"), info(func(i *types.ModelInfoDetails) {
			i.MaxInputTokens, i.MaxOutputTokens = f(1), f(1)
		}), backendModel("internal/embedding"))

		if ReduceModelGroup([]types.ModelInfoEntry{responses, unsupported}, resolveCatalog) != nil ||
			ReduceModelGroup([]types.ModelInfoEntry{chat, unsupported}, resolveCatalog) != nil {
			t.Fatal("expected nil")
		}
		if !HasMixedIncompatibleDeploymentModes([]types.ModelInfoEntry{chat, unsupported}) {
			t.Fatal("expected a mixed incompatible group")
		}
	})

	t.Run("takes the smaller valid limit when explicit and public catalog values disagree", func(t *testing.T) {
		limits := func(id string, in, out float64) types.ModelInfoEntry {
			return row(idMode(id, "chat"), info(func(i *types.ModelInfoDetails) { i.MaxInputTokens, i.MaxOutputTokens = f(in), f(out) }))
		}

		got := mustReduce(t, []types.ModelInfoEntry{limits("larger-explicit", 300_000, 80_000)}, resolveCatalog)
		if got.ContextWindow != 200_000 || got.MaxTokens != 64_000 {
			t.Fatalf("%v %v", got.ContextWindow, got.MaxTokens)
		}
		got = mustReduce(t, []types.ModelInfoEntry{limits("smaller-explicit", 100_000, 8_000)}, resolveCatalog)
		if got.ContextWindow != 100_000 || got.MaxTokens != 8_000 {
			t.Fatalf("%v %v", got.ContextWindow, got.MaxTokens)
		}
	})

	t.Run("keeps explicit prices and fills modalities from public catalog metadata", func(t *testing.T) {
		explicit := row(idMode("explicit", "chat"), info(func(i *types.ModelInfoDetails) {
			i.SupportsVision, i.InputCostPerToken, i.OutputCostPerToken = nil, f(0.000009), f(0.000019)
			i.CacheReadInputTokenCost, i.CacheCreationInputTokenCost = nil, nil
		}))

		got := mustReduce(t, []types.ModelInfoEntry{explicit}, resolveCatalog)
		if !got.Vision || got.Cost.Input != 9 || got.Cost.Output != 19 || got.Cost.CacheRead != 0.3 || got.Cost.CacheWrite != 3.75 {
			t.Fatalf("%s", dump(got))
		}
	})

	t.Run("withholds catalog authority for different concrete models from one provider", func(t *testing.T) {
		bare := func(id, model string) types.ModelInfoEntry {
			return row(backendModel(model), info(func(i *types.ModelInfoDetails) {
				i.ID, i.Mode, i.SupportsReasoning, i.SupportsVision, i.MaxInputTokens, i.MaxOutputTokens = id, s("chat"), nil, nil, nil, nil
				i.InputCostPerToken, i.OutputCostPerToken, i.CacheReadInputTokenCost, i.CacheCreationInputTokenCost = nil, nil, nil, nil
			}))
		}
		result := mustReduce(t, []types.ModelInfoEntry{bare("a", "route-a"), bare("b", "route-b")}, func(entry types.ModelInfoEntry) *CatalogResolution {
			modelID, window := "claude-opus-4-5", 180_000.0
			if entry.LiteLLMParams.Model == "route-a" {
				modelID, window = "claude-sonnet-4-6", 200_000
			}
			return &CatalogResolution{
				Provider: "anthropic", CatalogModelID: modelID, Reasoning: b(true), Vision: b(true),
				ContextWindow: f(window), MaxTokens: f(64_000), Cost: cost4(3, 15, 0.3, 3.75),
			}
		})

		if !result.CatalogAuthorityAmbiguous || result.ContextWindow != 128_000 || result.MaxTokens != 16_384 ||
			result.CatalogProvider != "" || result.ThinkingLevelMap != nil {
			t.Fatalf("%s", dump(result))
		}
		eq(t, result.Cost, modelCost(0, 0, 0, 0))
	})

	fullCatalog := func(levels ai.ThinkingLevelMap) CatalogResolution {
		return CatalogResolution{
			Provider: "openai", Reasoning: b(true), ThinkingLevelMap: levels, Vision: b(false),
			ContextWindow: f(128_000), MaxTokens: f(16_384), Cost: cost4(1, 2, 0, 0),
		}
	}
	effort := info(func(i *types.ModelInfoDetails) { i.SupportedOpenAIParams = []string{"reasoning_effort"} })

	t.Run("keeps standard levels a catalog thinking map omits at Pi's defaults", func(t *testing.T) {
		resolution := fullCatalog(lv("low", "low", "high", "high"))
		resolution.Provider = "xai"
		result := mustReduce(t, []types.ModelInfoEntry{row(effort)}, fixed(resolution))

		eq(t, result.ThinkingLevelMap, lv("low", "low", "high", "high", "xhigh", "null", "max", "null"))
	})

	t.Run("intersects differing catalog thinking maps per level regardless of deployment order", func(t *testing.T) {
		entries := []types.ModelInfoEntry{row(effort, idMode("a", "chat")), row(effort, idMode("b", "chat"))}
		maps := map[string]ai.ThinkingLevelMap{
			"a": lv("off", "none", "low", "low", "high", "high", "max", "null"),
			"b": lv("off", "none", "low", "null", "high", "high", "xhigh", "xhigh"),
		}
		for _, order := range permutations(entries) {
			result := mustReduce(t, order, func(entry types.ModelInfoEntry) *CatalogResolution {
				resolution := fullCatalog(maps[entry.ModelInfo.ID])
				return &resolution
			})

			eq(t, result.ThinkingLevelMap, lv("off", "none", "low", "null", "high", "high", "xhigh", "null", "max", "null"))
		}
	})

	t.Run("keeps catalog levels a deployment without a thinking map leaves at Pi's defaults", func(t *testing.T) {
		entries := []types.ModelInfoEntry{row(effort, idMode("mapped", "chat")), row(effort, idMode("absent", "chat"))}
		for _, order := range permutations(entries) {
			result := mustReduce(t, order, func(entry types.ModelInfoEntry) *CatalogResolution {
				var levels ai.ThinkingLevelMap
				if entry.ModelInfo.ID == "mapped" {
					levels = lv("low", "low", "high", "high")
				}
				resolution := fullCatalog(levels)
				return &resolution
			})

			eq(t, result.ThinkingLevelMap, lv("low", "low", "high", "high", "xhigh", "null", "max", "null"))
		}
	})

	t.Run("suppresses reasoning controls when the router explicitly disables reasoning", func(t *testing.T) {
		result := mustReduce(t, []types.ModelInfoEntry{
			row(idMode("non-reasoner", "chat"), info(func(i *types.ModelInfoDetails) {
				i.SupportsReasoning, i.SupportsLowReasoningEffort = b(false), b(true)
			})),
		}, fixed(func() CatalogResolution {
			resolution := fullCatalog(lv("low", "low", "high", "high"))
			resolution.Vision = b(true)
			return resolution
		}()))

		if result.Reasoning || result.ThinkingLevelMap != nil {
			t.Fatalf("%s", dump(result))
		}
	})

	t.Run("preserves explicit router reasoning efforts for a singleton", func(t *testing.T) {
		result := mustReduce(t, []types.ModelInfoEntry{
			row(effort, idMode("reasoner", "chat"), info(func(i *types.ModelInfoDetails) {
				i.SupportsNoneReasoningEffort, i.SupportsMinimalReasoningEffort, i.SupportsXHighReasoningEffort = b(true), b(false), b(true)
			}), backendModel("internal/reasoner")),
		}, resolveCatalog)

		eq(t, result.ThinkingLevelMap, lv("off", "none", "minimal", "null", "xhigh", "xhigh", "max", "null"))
	})

	t.Run("preserves defaults for missing effort flags and honors explicit opt-outs", func(t *testing.T) {
		supportsLow := func(id string, low *bool) types.ModelInfoEntry {
			return row(effort, idMode(id, "chat"), info(func(i *types.ModelInfoDetails) {
				i.SupportsLowReasoningEffort, i.SupportsXHighReasoningEffort = low, b(true)
			}), backendModel("internal/"+id))
		}

		eq(t, mustReduce(t, []types.ModelInfoEntry{supportsLow("a", b(true)), supportsLow("b", b(true))}, resolveCatalog).ThinkingLevelMap,
			lv("low", "low", "xhigh", "xhigh", "max", "null"))
		// Both an absent and a null flag read as a nil pointer.
		eq(t, mustReduce(t, []types.ModelInfoEntry{supportsLow("a", b(true)), supportsLow("b", nil)}, resolveCatalog).ThinkingLevelMap,
			lv("xhigh", "xhigh", "max", "null"))
		eq(t, mustReduce(t, []types.ModelInfoEntry{supportsLow("a", b(true)), supportsLow("b", b(false))}, resolveCatalog).ThinkingLevelMap,
			lv("low", "null", "xhigh", "xhigh", "max", "null"))
	})

	t.Run("overlays conservative router reasoning evidence on catalog metadata", func(t *testing.T) {
		catalogMap := lv("off", "none", "low", "low", "high", "high", "max", "max")
		low := func(id string, supported bool) types.ModelInfoEntry {
			return row(effort, idMode(id, "chat"), info(func(i *types.ModelInfoDetails) { i.SupportsLowReasoningEffort = b(supported) }))
		}
		resolution := fullCatalog(catalogMap)
		resolution.Vision = b(true)

		result := mustReduce(t, []types.ModelInfoEntry{low("a", true), low("b", false)}, fixed(resolution))

		eq(t, result.ThinkingLevelMap, lv("off", "none", "low", "null", "high", "high", "xhigh", "null", "max", "null"))
	})

	t.Run("merges different rates at identical tier thresholds conservatively regardless of order", func(t *testing.T) {
		rows := []types.ModelInfoEntry{row(idMode("a", "chat"), unpriced), row(idMode("b", "chat"), unpriced)}
		costs := []ai.ModelCost{
			modelCost(3, 15, 0.3, 3.75, tier(200_000, 6, 20, 0.6, 7.5)),
			modelCost(4, 12, 0.4, 4, tier(200_000, 5, 22.5, 0.5, 8)),
		}
		for _, order := range permutations(costs) {
			call := 0
			result := mustReduce(t, rows, func(types.ModelInfoEntry) *CatalogResolution {
				resolution := &CatalogResolution{
					Provider: "anthropic", Reasoning: b(true), Vision: b(true), ContextWindow: f(200_000), MaxTokens: f(64_000),
					Cost: catalogCostOf(order[call]),
				}
				call++
				return resolution
			})

			eq(t, result.Cost, modelCost(4, 15, 0.4, 4, tier(200_000, 6, 22.5, 0.6, 8)))
			if !result.HasCompleteCost || !result.HasCompleteMetadata {
				t.Fatalf("%s", dump(result))
			}
		}
	})

	t.Run("constructs a safe piecewise envelope for different tier thresholds regardless of order", func(t *testing.T) {
		rateAt := func(cost ai.ModelCost, inputTokens int) (input, output float64) {
			input, output = cost.Input, cost.Output
			best := -1
			for _, candidate := range cost.Tiers {
				if candidate.InputTokensAbove < inputTokens && candidate.InputTokensAbove > best {
					best, input, output = candidate.InputTokensAbove, candidate.InputCostPer1M, candidate.OutputCostPer1M
				}
			}
			return input, output
		}
		sampledCost := func(cost ai.ModelCost, inputTokens, outputTokens int) float64 {
			input, output := rateAt(cost, inputTokens)
			return input*float64(inputTokens) + output*float64(outputTokens)
		}
		rows := []types.ModelInfoEntry{row(idMode("a", "chat"), unpriced), row(idMode("b", "chat"), unpriced), row(idMode("c", "chat"), unpriced)}
		costs := []ai.ModelCost{
			modelCost(3, 15, 0.3, 3.75, tier(200_000, 6, 18, 0.6, 7.5), tier(500_000, 12, 36, 1.2, 15)),
			modelCost(4, 16, 0.4, 4, tier(400_000, 8, 32, 0.8, 8)),
			modelCost(5, 14, 0.5, 5),
		}
		expected := modelCost(5, 16, 0.5, 5,
			tier(200_000, 6, 18, 0.6, 7.5), tier(400_000, 8, 32, 0.8, 8), tier(500_000, 12, 36, 1.2, 15))
		for _, order := range permutations(costs) {
			call := 0
			result := mustReduce(t, rows, func(types.ModelInfoEntry) *CatalogResolution {
				resolution := &CatalogResolution{
					Provider: "anthropic", Reasoning: b(true), Vision: b(true), ContextWindow: f(200_000), MaxTokens: f(64_000),
					Cost: catalogCostOf(order[call]),
				}
				call++
				return resolution
			})

			eq(t, result.Cost, expected)
			if !result.HasCompleteCost || !result.HasCompleteMetadata {
				t.Fatalf("%s", dump(result))
			}
			for _, inputTokens := range []int{0, 200_000, 200_001, 400_000, 400_001, 500_000, 500_001} {
				envelope := sampledCost(result.Cost, inputTokens, 1_000)
				for _, source := range costs {
					if envelope < sampledCost(source, inputTokens, 1_000) {
						t.Fatalf("envelope %v below source at %d", envelope, inputTokens)
					}
				}
			}
		}
	})

	t.Run("drops the catalog tier ladder when every base-price field is explicit", func(t *testing.T) {
		result := mustReduce(t, []types.ModelInfoEntry{
			row(idMode("explicit-prices", "chat"), info(func(i *types.ModelInfoDetails) {
				i.InputCostPerToken, i.OutputCostPerToken = f(0.000004), f(0.00002)
				i.CacheReadInputTokenCost, i.CacheCreationInputTokenCost = f(0.0000004), f(0.000005)
			})),
		}, fixed(CatalogResolution{
			Provider: "anthropic", Reasoning: b(true), Vision: b(true), ContextWindow: f(200_000), MaxTokens: f(64_000),
			Cost: cost4(3, 15, 0.3, 3.75, tier(200_000, 6, 30, 0.6, 7.5)),
		}))

		if !result.HasCompleteCost || result.Cost.Input != 4 || result.Cost.Output != 20 || result.Cost.CacheWrite != 5 || result.Cost.Tiers != nil {
			t.Fatalf("%s", dump(result))
		}
		near(t, result.Cost.CacheRead, 0.4)
	})

	t.Run("retains catalog tiers for fields without an explicit router base price", func(t *testing.T) {
		result := mustReduce(t, []types.ModelInfoEntry{
			row(idMode("partial-explicit-prices", "chat"), unpriced, info(func(i *types.ModelInfoDetails) { i.InputCostPerToken = f(0.000004) })),
		}, fixed(CatalogResolution{
			Provider: "anthropic", Reasoning: b(true), Vision: b(true), ContextWindow: f(200_000), MaxTokens: f(64_000),
			Cost: cost4(3, 15, 0.3, 3.75, tier(200_000, 6, 30, 0.6, 7.5)),
		}))

		if !result.HasCompleteCost {
			t.Fatal("hasCompleteCost")
		}
		eq(t, result.Cost, modelCost(4, 15, 0.3, 3.75, tier(200_000, 4, 30, 0.6, 7.5)))
	})

	t.Run("does not attach catalog tiers when base price evidence is incomplete", func(t *testing.T) {
		rows := []types.ModelInfoEntry{row(idMode("a", "chat"), unpriced), row(idMode("b", "chat"), unpriced)}
		call := 0
		result := mustReduce(t, rows, func(types.ModelInfoEntry) *CatalogResolution {
			var cost *CatalogCost
			if call == 0 {
				cost = cost4(3, 15, 0.3, 3.75, tier(200_000, 6, 30, 0.6, 7.5))
			}
			call++
			return &CatalogResolution{
				Provider: "anthropic", Reasoning: b(true), Vision: b(true), ContextWindow: f(200_000), MaxTokens: f(64_000), Cost: cost,
			}
		})

		if result.HasCompleteCost || result.HasCompleteMetadata || result.Cost.Tiers != nil {
			t.Fatalf("%s", dump(result))
		}
		eq(t, result.Cost, modelCost(0, 0, 0, 0))
	})

	t.Run("flags ambiguous catalog authority only when resolved identities disagree", func(t *testing.T) {
		anthropic := row(idMode("anthropic", "chat"))
		openai := row(idMode("openai", "chat"), backendModel("openai/gpt-4o"))
		unresolved := row(idMode("unknown", "chat"), backendModel("internal/unknown"))
		ambiguous := func(entries ...types.ModelInfoEntry) bool {
			return mustReduce(t, entries, resolveCatalog).CatalogAuthorityAmbiguous
		}

		// Unanimous identity, and wholly unknown identity, are not ambiguity.
		if ambiguous(anthropic) || ambiguous(anthropic, anthropic) || ambiguous(unresolved) || ambiguous(unresolved, unresolved) {
			t.Fatal("unexpected ambiguity")
		}
		// Conflicting identities, and partial evidence, both withhold authority.
		if !ambiguous(anthropic, openai) || !ambiguous(anthropic, unresolved) {
			t.Fatal("expected ambiguity")
		}
	})
}

func TestStableJson(t *testing.T) {
	t.Run("preserves array order in canonical identity", func(t *testing.T) {
		type threshold struct {
			Threshold int `json:"threshold"`
		}
		first, _ := StableJSON([]threshold{{1}, {2}})
		reversed, _ := StableJSON([]threshold{{2}, {1}})
		if first == reversed {
			t.Fatal("array order must matter")
		}
		a, _ := StableJSON([]map[string]int{{"b": 2, "a": 1}})
		c, _ := StableJSON([]map[string]int{{"a": 1, "b": 2}})
		if a != c || a != `[{"a":1,"b":2}]` {
			t.Fatalf("%s %s", a, c)
		}
		if _, ok := StableJSON(nil); ok {
			t.Fatal("nil is undefined")
		}
	})

	t.Run("bounds canonicalization depth", func(t *testing.T) {
		var deep any = "leaf"
		for range 20 {
			deep = map[string]any{"k": deep}
		}
		encoded, _ := StableJSON(deep)
		var generic any
		if err := json.Unmarshal([]byte(encoded), &generic); err != nil {
			t.Fatal(err)
		}
		if want := `{"k":{"k":{"k":{"k":{"k":{"k":{"k":{"k":{"k":{"k":{"k":{"k":"[depth-limit]"}}}}}}}}}}}}`; encoded != want {
			t.Fatalf("%s", encoded)
		}
	})
}

func TestCatalogResolutionFromModel(t *testing.T) {
	raw := json.RawMessage(`{"id":"gpt-5.5","name":"GPT","api":"openai-responses","provider":"openai","baseUrl":"https://example.test",` +
		`"reasoning":true,"input":["text","image"],"cost":{"input":1,"output":2,"cacheRead":0.1,"cacheWrite":0},` +
		`"contextWindow":400000,"maxTokens":128000}`)
	models, err := ai.DecodeModelsCatalog([]json.RawMessage{raw}, "openai")
	if err != nil || len(models) != 1 {
		t.Fatalf("decode: %v %d", err, len(models))
	}
	model, ok := models[0].(*ai.Model)
	if !ok {
		t.Fatalf("model type %T", models[0])
	}

	got := CatalogResolutionFromModel("openai", model)

	if got.Provider != "openai" || got.CatalogModelID != "gpt-5.5" || !*got.Reasoning || !*got.Vision ||
		*got.ContextWindow != 400_000 || *got.MaxTokens != 128_000 || *got.Cost.Input != 1 || *got.Cost.CacheRead != 0.1 {
		t.Fatalf("%s", dump(got))
	}
}

func TestNativeMessagesRouteSelection(t *testing.T) {
	t.Run("closes native reasoning without inheriting OpenAI controls", func(t *testing.T) {
		got := CloseSerializerPolicy(CloseSerializerPolicyInput{
			API:            types.APIAnthropicMessages,
			Reasoning:      true,
			VendorCompat:   &ai.ModelCompat{ForceAdaptiveThinking: b(true), SupportsReasoningEffort: b(true)},
			SemanticCompat: &ReasoningCompat{ThinkingFormat: "openai"},
			SemanticLevels: lv("off", "none", "max", "xhigh"),
			CatalogLevels:  lv("off", "null", "max", "max"),
		})

		eq(t, got, SerializerPolicy{
			Reasoning:        true,
			Compat:           &ai.ModelCompat{ForceAdaptiveThinking: b(true)},
			ThinkingLevelMap: lv("off", "null", "max", "max"),
		})
	})

	claude := func(compat MessagesBackendCompat) CatalogResolution {
		return CatalogResolution{Provider: "amazon-bedrock", SemanticFamily: SemanticFamilyClaude, MessagesCompat: &compat}
	}
	adaptive := MessagesBackendCompat{ForceAdaptiveThinking: b(true)}
	pair := func() []types.ModelInfoEntry { return []types.ModelInfoEntry{bare("a", "chat"), bare("b", "chat")} }

	t.Run("selects Messages for a homogeneous strongly evidenced Claude group", func(t *testing.T) {
		result := mustReduce(t, pair(), fixed(claude(adaptive)))

		if result.API != types.APIAnthropicMessages || result.CatalogProvider != "amazon-bedrock" ||
			result.SemanticFamily != SemanticFamilyClaude {
			t.Fatalf("%s", dump(result))
		}
		eq(t, result.MessagesCompat, &adaptive)
	})

	for _, c := range []struct {
		name     string
		evidence []bool
		expected bool
	}{
		{"all affirmative", []bool{true, true}, true},
		{"conflicting evidence", []bool{true, false}, false},
		{"denial plus unknown", []bool{false, false}, false},
		{"partial evidence", []bool{true, false}, false},
		{"no evidence", []bool{false, false}, false},
	} {
		t.Run("reduces strict-tool evidence independently for "+c.name+" deployments", func(t *testing.T) {
			// A false stands for both a denial and an unknown: only an affirmative counts.
			remaining := append([]bool(nil), c.evidence...)
			result := mustReduce(t, pair(), func(types.ModelInfoEntry) *CatalogResolution {
				resolution := claude(adaptive)
				resolution.MessagesStrictTools = remaining[0]
				remaining = remaining[1:]
				return &resolution
			})

			if result.API != types.APIAnthropicMessages || result.MessagesStrictTools != c.expected {
				t.Fatalf("%s", dump(result))
			}
			eq(t, result.MessagesCompat, &adaptive)
		})
	}

	t.Run("respects explicit Messages endpoint capability", func(t *testing.T) {
		withEndpoints := func(endpoints ...string) types.ModelInfoEntry {
			entry := bare("a", "chat")
			entry.ModelInfo.SupportedEndpoints = endpoints
			return entry
		}
		supported := mustReduce(t, []types.ModelInfoEntry{withEndpoints("/v1/chat/completions", "/v1/messages")}, fixed(claude(MessagesBackendCompat{})))
		excluded := mustReduce(t, []types.ModelInfoEntry{withEndpoints("/v1/chat/completions")}, fixed(claude(MessagesBackendCompat{})))

		if supported.API != types.APIAnthropicMessages || excluded.API != types.APIOpenAICompletions {
			t.Fatalf("%s %s", supported.API, excluded.API)
		}
	})

	// LiteLLM v1.100 sends `supported_endpoints: null` for models outside its model map, as it does for
	// Bedrock-served Claude; that is no evidence either way, like an absent field.
	t.Run("treats null endpoint metadata as unknown rather than a denial", func(t *testing.T) {
		entry := bare("a", "chat")
		entry.ModelInfo.SupportedEndpoints = nil

		if got := mustReduce(t, []types.ModelInfoEntry{entry}, fixed(claude(MessagesBackendCompat{}))).API; got != types.APIAnthropicMessages {
			t.Fatalf("api %s", got)
		}
	})

	mixed := []struct {
		name     string
		evidence []*CatalogResolution
	}{
		{"mixed family", []*CatalogResolution{ptr(claude(MessagesBackendCompat{})), {Provider: "openai", SemanticFamily: SemanticFamilyOpenAI}}},
		{"unknown sibling", []*CatalogResolution{ptr(claude(MessagesBackendCompat{})), nil}},
		{"different Messages compatibility", []*CatalogResolution{ptr(claude(adaptive)), ptr(claude(MessagesBackendCompat{}))}},
	}
	for _, c := range mixed {
		t.Run("keeps "+c.name+" groups on Chat Completions", func(t *testing.T) {
			remaining := append([]*CatalogResolution(nil), c.evidence...)
			entries := []types.ModelInfoEntry{bare("a", "chat"), bare("b", "chat")}
			entries[0].ModelName, entries[1].ModelName = "mixed-route", "mixed-route"
			result := mustReduce(t, entries, func(types.ModelInfoEntry) *CatalogResolution {
				next := remaining[0]
				remaining = remaining[1:]
				return next
			})

			if result.API != types.APIOpenAICompletions {
				t.Fatalf("api %s", result.API)
			}
		})
	}

	t.Run("keeps explicit Responses mode authoritative for Claude", func(t *testing.T) {
		result := mustReduce(t, []types.ModelInfoEntry{bare("a", "responses")}, fixed(claude(MessagesBackendCompat{})))

		if result.API != types.APIOpenAIResponses {
			t.Fatalf("api %s", result.API)
		}
	})

	t.Run("does not inject router OpenAI effort values into a Messages model", func(t *testing.T) {
		entry := bare("a", "chat")
		entry.ModelInfo.SupportsMinimalReasoningEffort, entry.ModelInfo.SupportsXHighReasoningEffort = b(true), b(true)
		resolution := claude(adaptive)
		resolution.MessagesThinkingLevelMap = lv("max", "max")

		result := mustReduce(t, []types.ModelInfoEntry{entry}, fixed(resolution))

		if result.API != types.APIAnthropicMessages {
			t.Fatalf("api %s", result.API)
		}
		eq(t, result.ThinkingLevelMap, lv("max", "max"))
	})

	t.Run("lets router evidence disable but not rename a catalogued Messages effort", func(t *testing.T) {
		entry := bare("a", "chat")
		entry.ModelInfo.SupportsXHighReasoningEffort = b(false)
		resolution := claude(adaptive)
		resolution.MessagesThinkingLevelMap = lv("xhigh", "xhigh", "max", "max")

		result := mustReduce(t, []types.ModelInfoEntry{entry}, fixed(resolution))

		eq(t, result.ThinkingLevelMap, lv("xhigh", "null", "max", "max"))
	})
}

func ptr[T any](v T) *T { return &v }
