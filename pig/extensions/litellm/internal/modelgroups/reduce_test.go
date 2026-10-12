// Ports tests/model-groups.test.ts: reduceModelGroup
package modelgroups

import (
	"strconv"
	"testing"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/thinking"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

func TestReduceModelGroup(t *testing.T) {
	t.Run("is permutation invariant for heterogeneous deployment evidence", func(t *testing.T) {
		deployments := []types.ModelInfoEntry{
			row(info(func(i *types.ModelInfoDetails) {
				i.ID, i.Mode, i.MaxInputTokens = "deployment-a", s("responses"), f(150_000)
			})),
			row(idMode("deployment-b", "chat"), info(func(i *types.ModelInfoDetails) { i.MaxOutputTokens = f(16_000) }),
				backendModel("openai/gpt-4o")),
			row(idMode("deployment-c", ""), backendModel("internal/unknown")),
			row(idMode("", "chat"), info(func(i *types.ModelInfoDetails) { i.OutputCostPerToken = f(0.00002) }),
				backendModel("internal/unknown")),
		}
		expected := &ReducedModelGroup{
			ID:                        "route",
			API:                       types.APIOpenAICompletions,
			Reasoning:                 true,
			Vision:                    true,
			ContextWindow:             150_000,
			MaxTokens:                 16_000,
			Cost:                      modelCost(3, 20, 0.3, 3.75),
			HasCompleteCost:           true,
			HasCompleteMetadata:       true,
			CatalogAuthorityAmbiguous: true,
			DeploymentFamilies:        make([]FamilyEvidence, 4),
			AcceptedOpenAIParams:      []string{},
			ReasoningPolicy:           ReasoningPolicy{Reasoning: false},
		}
		for _, order := range permutations(deployments) {
			eq(t, ReduceModelGroup(order, resolveCatalog), expected)
		}
	})

	t.Run("deduplicates exact rows and reduces conflicting duplicate ids conservatively", func(t *testing.T) {
		repeated := row()
		conflicting := row(idMode("deployment-a", "chat"), info(func(i *types.ModelInfoDetails) { i.MaxInputTokens = f(8_000) }))
		anonymous := row(idMode("", "chat"))

		eq(t, ReduceModelGroup([]types.ModelInfoEntry{repeated, repeated}, resolveCatalog),
			ReduceModelGroup([]types.ModelInfoEntry{repeated}, resolveCatalog))
		// Conflicting variants of one deployment id both stay in the reduction.
		expected := mustReduce(t, []types.ModelInfoEntry{repeated, conflicting}, resolveCatalog)
		if expected.ContextWindow != 8_000 {
			t.Fatalf("contextWindow %v", expected.ContextWindow)
		}
		eq(t, ReduceModelGroup([]types.ModelInfoEntry{conflicting, repeated}, resolveCatalog), expected)

		// Exact id-less repeats remain plural: equal content is not enough evidence that two rows
		// describe the same deployment.
		calls := 0
		ReduceModelGroup([]types.ModelInfoEntry{anonymous, anonymous}, func(types.ModelInfoEntry) *CatalogResolution {
			calls++
			return nil
		})
		if calls != 2 {
			t.Fatalf("calls %d", calls)
		}
	})

	t.Run("selects Responses only when every deployment explicitly reports it", func(t *testing.T) {
		responses := row(idMode("responses", "responses"))
		response := row(idMode("response", "response"))
		chat := row(idMode("chat", "chat"))
		unknown := row(idMode("unknown", ""))

		for _, c := range []struct {
			entries []types.ModelInfoEntry
			api     types.LiteLLMApi
		}{
			{[]types.ModelInfoEntry{responses, response}, types.APIOpenAIResponses},
			{[]types.ModelInfoEntry{responses, chat}, types.APIOpenAICompletions},
			{[]types.ModelInfoEntry{responses, unknown}, types.APIOpenAICompletions},
		} {
			if got := mustReduce(t, c.entries, resolveCatalog).API; got != c.api {
				t.Fatalf("api %s want %s", got, c.api)
			}
		}
	})

	t.Run("requires every Responses deployment to accept reasoning_effort", func(t *testing.T) {
		accepted := func(id string, p []string) types.ModelInfoEntry {
			return row(idMode(id, "responses"), info(func(i *types.ModelInfoDetails) { i.SupportedOpenAIParams = p }),
				backendModel("internal/"+id))
		}
		for _, order := range permutations([]types.ModelInfoEntry{accepted("effort", []string{"reasoning_effort"}), accepted("thinking", []string{"thinking"})}) {
			got := mustReduce(t, order, resolveCatalog)
			if got.API != types.APIOpenAIResponses || got.AcceptsResponsesReasoningControl {
				t.Fatalf("got %s %v", got.API, got.AcceptsResponsesReasoningControl)
			}
		}
		got := mustReduce(t, []types.ModelInfoEntry{
			accepted("a", []string{"reasoning_effort", "thinking"}), accepted("b", []string{"reasoning_effort"}),
		}, resolveCatalog)
		if got.API != types.APIOpenAIResponses || !got.AcceptsResponsesReasoningControl {
			t.Fatalf("got %s %v", got.API, got.AcceptsResponsesReasoningControl)
		}
	})

	for _, c := range []struct{ name, first, second string }{
		{"chat first", "chat", "embedding"},
		{"embedding first", "embedding", "chat"},
		{"Responses first", "responses", "embedding"},
		{"embedding before Responses", "embedding", "responses"},
	} {
		t.Run("rejects a mixed chat-style and unsupported group with "+c.name, func(t *testing.T) {
			deployment := func(id, mode string) types.ModelInfoEntry {
				return row(idMode(id, mode), backendModel("internal/"+id))
			}
			if got := ReduceModelGroup([]types.ModelInfoEntry{deployment("first", c.first), deployment("second", c.second)}, resolveCatalog); got != nil {
				t.Fatalf("expected nil, got %s", dump(got))
			}
		})
	}

	for _, mode := range []string{"chat", "response", "responses"} {
		t.Run("retains a pure "+mode+" group", func(t *testing.T) {
			var deployments []types.ModelInfoEntry
			for _, id := range []string{"first", "second"} {
				deployments = append(deployments, row(idMode(id, mode), backendModel("internal/"+id)))
			}
			want := types.APIOpenAIResponses
			if mode == "chat" {
				want = types.APIOpenAICompletions
			}
			if got := mustReduce(t, deployments, resolveCatalog).API; got != want {
				t.Fatalf("api %s want %s", got, want)
			}
		})
	}

	t.Run("ignores limits that are not finite positive token counts", func(t *testing.T) {
		good := row(idMode("good", "chat"), info(func(i *types.ModelInfoDetails) {
			i.MaxInputTokens, i.MaxOutputTokens = f(64_000), f(8_000)
		}))
		for _, invalid := range []float64{0, -1, nan, inf} {
			broken := row(idMode("broken", "chat"), info(func(i *types.ModelInfoDetails) {
				i.MaxInputTokens, i.MaxOutputTokens = f(invalid), f(invalid)
			}))
			// The catalog resolves for both rows, so an unusable router limit falls back to catalog
			// evidence instead of clamping the group to zero.
			got := mustReduce(t, []types.ModelInfoEntry{good, broken}, resolveCatalog)
			if got.ContextWindow != 64_000 || got.MaxTokens != 8_000 {
				t.Fatalf("invalid %v: %v %v", invalid, got.ContextWindow, got.MaxTokens)
			}
		}
		unknownBackend := row(idMode("broken", "chat"), info(func(i *types.ModelInfoDetails) {
			i.MaxInputTokens, i.MaxOutputTokens = f(0), f(0)
		}), backendModel("internal/unknown"))
		got := mustReduce(t, []types.ModelInfoEntry{unknownBackend}, resolveCatalog)
		if got.ContextWindow != 128_000 || got.MaxTokens != 16_384 {
			t.Fatalf("%v %v", got.ContextWindow, got.MaxTokens)
		}
	})

	for _, c := range []struct {
		configured string
		expected   float64
	}{
		{"922000", 922_000}, {"", 128_000}, {"not-a-number", 128_000}, {"0", 128_000},
		{"-1", 128_000}, {"1.5", 128_000}, {"922000junk", 128_000},
	} {
		t.Run("uses LITELLM_DEFAULT_CONTEXT_WINDOW="+strconv.Quote(c.configured)+" as the fallback window", func(t *testing.T) {
			t.Setenv("LITELLM_DEFAULT_CONTEXT_WINDOW", c.configured)
			noLimits := row(idMode("only", "chat"), info(func(i *types.ModelInfoDetails) { i.MaxInputTokens = nil }))

			// Only the missing limit is filled; a reported window still wins.
			if got := mustReduce(t, []types.ModelInfoEntry{noLimits}, none).ContextWindow; got != c.expected {
				t.Fatalf("contextWindow %v want %v", got, c.expected)
			}
			reported := row(idMode("only", "chat"), info(func(i *types.ModelInfoDetails) { i.MaxInputTokens = f(8_000) }))
			if got := mustReduce(t, []types.ModelInfoEntry{reported}, none).ContextWindow; got != 8_000 {
				t.Fatalf("contextWindow %v", got)
			}
		})
	}

	for _, c := range []struct {
		name    string
		invalid float64
	}{{"zero", 0}, {"negative", -1}, {"not a number", nan}, {"infinite", inf}} {
		t.Run("ignores a "+c.name+" catalog limit and falls back to the conservative default", func(t *testing.T) {
			// The same validation must apply to catalog-supplied limits, not only to the
			// router-reported ones, or a bad catalog value would clamp the whole group.
			noRouterLimits := row(idMode("only", "chat"), info(func(i *types.ModelInfoDetails) {
				i.MaxInputTokens, i.MaxOutputTokens = nil, nil
			}))
			brokenCatalog := fixed(CatalogResolution{
				Provider: "anthropic", Reasoning: b(true), Vision: b(true),
				ContextWindow: f(c.invalid), MaxTokens: f(c.invalid), Cost: cost4(3, 15, 0.3, 3.75),
			})
			got := mustReduce(t, []types.ModelInfoEntry{noRouterLimits}, brokenCatalog)
			if got.ContextWindow != 128_000 || got.MaxTokens != 16_384 {
				t.Fatalf("%v %v", got.ContextWindow, got.MaxTokens)
			}
			// Authority is not discarded wholesale; only the unusable limits are.
			if got.CatalogProvider != "anthropic" || !got.Reasoning {
				t.Fatalf("%s", dump(got))
			}
		})
	}

	// TestMistypedWireFields ports the same case with the Vitest `mode: 7`; this one uses a null mode.
	t.Run("treats an unreadable mode as unknown rather than as evidence of a non-chat deployment", func(t *testing.T) {
		roomy := row(idMode("roomy", "chat"), info(func(i *types.ModelInfoDetails) { i.MaxInputTokens = f(200_000) }))
		cramped := func(mode string) types.ModelInfoEntry {
			return row(idMode("cramped", mode), info(func(i *types.ModelInfoDetails) { i.MaxInputTokens = f(8_000) }))
		}
		unreadable, embedding := cramped(""), cramped("embedding")

		// Unreadable: still a deployment, so its tighter limit clamps the group.
		got := mustReduce(t, []types.ModelInfoEntry{roomy, unreadable}, resolveCatalog)
		if got.ContextWindow != 8_000 || got.API != types.APIOpenAICompletions {
			t.Fatalf("%v %s", got.ContextWindow, got.API)
		}
		// Genuinely non-chat: rejects the entire mixed route.
		if ReduceModelGroup([]types.ModelInfoEntry{roomy, embedding}, resolveCatalog) != nil {
			t.Fatal("expected nil")
		}
		// A lone unreadable row is surfaced conservatively rather than silently hidden.
		got = mustReduce(t, []types.ModelInfoEntry{unreadable}, resolveCatalog)
		if got.ContextWindow != 8_000 || got.API != types.APIOpenAICompletions {
			t.Fatalf("%v %s", got.ContextWindow, got.API)
		}
	})

	t.Run("drops a group when every deployment is non-chat", func(t *testing.T) {
		got := ReduceModelGroup([]types.ModelInfoEntry{row(idMode("embed-a", "embedding")), row(idMode("embed-b", "embedding"))}, resolveCatalog)
		if got != nil {
			t.Fatal("expected nil")
		}
	})

	for _, c := range []struct {
		values   []*bool
		expected bool
	}{
		{[]*bool{b(true), b(true)}, true},
		{[]*bool{b(true), b(false)}, false},
		{[]*bool{b(true), nil}, false},
	} {
		t.Run("reduces capability guarantees "+dump(c.values)+" to "+strconv.FormatBool(c.expected), func(t *testing.T) {
			var deployments []types.ModelInfoEntry
			for index, value := range c.values {
				mods := []mod{idMode("deployment-"+strconv.Itoa(index), "chat"), info(func(i *types.ModelInfoDetails) { i.SupportsVision = value })}
				if value == nil {
					mods = append(mods, backendModel("internal/unknown"))
				}
				deployments = append(deployments, row(mods...))
			}
			if got := mustReduce(t, deployments, resolveCatalog).Vision; got != c.expected {
				t.Fatalf("vision %v", got)
			}
		})
	}

	t.Run("uses the smaller valid router and catalog limit", func(t *testing.T) {
		router := row(idMode("router", "chat"), info(func(i *types.ModelInfoDetails) {
			i.MaxInputTokens, i.MaxOutputTokens = f(300_000), f(100_000)
		}))
		got := mustReduce(t, []types.ModelInfoEntry{router}, fixed(CatalogResolution{
			Provider: "openai", ContextWindow: f(200_000), MaxTokens: f(64_000),
		}))
		if got.ContextWindow != 200_000 || got.MaxTokens != 64_000 {
			t.Fatalf("%v %v", got.ContextWindow, got.MaxTokens)
		}
	})

	t.Run("resolves deployment limits before taking the safe group minimum", func(t *testing.T) {
		explicit := row(idMode("explicit", "chat"), info(func(i *types.ModelInfoDetails) {
			i.MaxInputTokens, i.MaxOutputTokens = f(100_000), f(8_000)
		}))
		noLimits := info(func(i *types.ModelInfoDetails) { i.MaxInputTokens, i.MaxOutputTokens = nil, nil })
		fromCatalog := row(idMode("catalog", "chat"), noLimits)
		unknown := row(idMode("unknown", "chat"), noLimits, backendModel("internal/unknown"))

		for _, c := range []struct {
			entries        []types.ModelInfoEntry
			context, limit float64
		}{
			{[]types.ModelInfoEntry{explicit, fromCatalog}, 100_000, 8_000},
			{[]types.ModelInfoEntry{explicit, unknown}, 100_000, 8_000},
			{[]types.ModelInfoEntry{unknown}, 128_000, 16_384},
		} {
			got := mustReduce(t, c.entries, resolveCatalog)
			if got.ContextWindow != c.context || got.MaxTokens != c.limit {
				t.Fatalf("%v %v", got.ContextWindow, got.MaxTokens)
			}
		}
	})

	t.Run("uses the maximum complete display price and marks incomplete price evidence unknown", func(t *testing.T) {
		priced := func(id string, in, out, cr, cw float64) types.ModelInfoEntry {
			return row(idMode(id, "chat"), info(func(i *types.ModelInfoDetails) {
				i.InputCostPerToken, i.OutputCostPerToken = f(in), f(out)
				i.CacheReadInputTokenCost, i.CacheCreationInputTokenCost = f(cr), f(cw)
			}))
		}
		cheaper := priced("cheap", 0, 0.00001, 0.0000002, 0.000003)
		pricier := priced("pricey", 0.000004, 0.00002, 0.0000004, 0.000004)
		complete := mustReduce(t, []types.ModelInfoEntry{cheaper, pricier}, resolveCatalog)
		if !complete.HasCompleteCost || complete.Cost.Input != 4 || complete.Cost.Output != 20 || complete.Cost.CacheWrite != 4 {
			t.Fatalf("%s", dump(complete))
		}
		near(t, complete.Cost.CacheRead, 0.4)

		incomplete := row(idMode("incomplete", "chat"), info(func(i *types.ModelInfoDetails) {
			i.InputCostPerToken, i.OutputCostPerToken = f(0.000004), nil
		}), backendModel("internal/unknown"))
		got := mustReduce(t, []types.ModelInfoEntry{cheaper, incomplete}, resolveCatalog)
		if got.HasCompleteCost || got.Cost.Input != 4 || got.Cost.Output != 0 || got.Cost.CacheRead != 0.3 || got.Cost.CacheWrite != 3.75 {
			t.Fatalf("%s", dump(got))
		}
	})

	t.Run("rejects negative explicit prices as unresolved", func(t *testing.T) {
		negative := row(idMode("negative", "chat"), info(func(i *types.ModelInfoDetails) {
			i.InputCostPerToken, i.OutputCostPerToken = f(-0.000001), f(0.000002)
		}), backendModel("internal/unknown"))
		got := mustReduce(t, []types.ModelInfoEntry{negative}, resolveCatalog)
		if got.HasCompleteCost || got.Cost.Input != 0 || got.Cost.Output != 2 {
			t.Fatalf("%s", dump(got))
		}
	})

	t.Run("retains proven display prices and zeroes only unresolved fields without catalog authority", func(t *testing.T) {
		// Characterization of the existing per-field cost block. No backend resolves, so cache pricing
		// is genuinely unknown rather than free: input and output survive at their proven values, the
		// unresolved cache fields read zero, and `hasCompleteCost` stays false.
		priced := func(id string, in, out float64) types.ModelInfoEntry {
			return row(idMode(id, "chat"), info(func(i *types.ModelInfoDetails) {
				i.InputCostPerToken, i.OutputCostPerToken = f(in), f(out)
				i.CacheReadInputTokenCost, i.CacheCreationInputTokenCost = nil, nil
			}), backendModel("internal/unknown"))
		}
		singleton := mustReduce(t, []types.ModelInfoEntry{priced("only", 0.000003, 0.000015)}, resolveCatalog)
		eq(t, singleton.Cost, modelCost(3, 15, 0, 0))
		if singleton.HasCompleteCost || singleton.CatalogProvider != "" {
			t.Fatalf("%s", dump(singleton))
		}

		// Proven fields still reduce to the maximum across a group.
		got := mustReduce(t, []types.ModelInfoEntry{priced("a", 0.000003, 0.000015), priced("b", 0.000004, 0.000015)}, resolveCatalog)
		eq(t, got.Cost, modelCost(4, 15, 0, 0))
		if got.HasCompleteCost {
			t.Fatal("hasCompleteCost")
		}
	})

	t.Run("applies public effort levels and LiteLLM overrides", func(t *testing.T) {
		result := mustReduce(t, []types.ModelInfoEntry{
			row(idMode("reasoner", "chat"), info(func(i *types.ModelInfoDetails) {
				i.SupportedOpenAIParams = []string{"reasoning_effort"}
				i.SupportsMinimalReasoningEffort, i.SupportsXHighReasoningEffort = b(false), b(true)
			})),
		}, fixed(CatalogResolution{Provider: "openai", Reasoning: b(true), EffortLevels: []string{"minimal", "low", "medium", "high"}}))

		eq(t, result.ThinkingLevelMap,
			lv("off", "null", "minimal", "null", "low", "low", "medium", "medium", "high", "high", "xhigh", "xhigh", "max", "null"))
	})

	t.Run("leaves standard levels absent without a public opinion", func(t *testing.T) {
		result := mustReduce(t, []types.ModelInfoEntry{
			row(info(func(i *types.ModelInfoDetails) {
				i.SupportsReasoning, i.SupportedOpenAIParams = b(true), []string{"reasoning_effort"}
			}), backendModel("internal/reasoner")),
		}, fixed(CatalogResolution{Reasoning: b(true)}))

		eq(t, result.ThinkingLevelMap, lv("xhigh", "null", "max", "null"))
	})

	t.Run("keeps off denied for an always-thinking Kimi generation when the catalog supplies effort levels", func(t *testing.T) {
		result := mustReduce(t, []types.ModelInfoEntry{
			row(info(func(i *types.ModelInfoDetails) {
				i.SupportsReasoning, i.SupportedOpenAIParams = b(true), []string{"reasoning_effort"}
			}), backendModel("moonshot/kimi-k2.7-code")),
		}, fixed(CatalogResolution{
			Provider: "moonshotai", Reasoning: b(true), SemanticModel: SemanticModelKimiK27Code,
			EffortLevels: []string{"low", "medium", "high"},
		}))

		eq(t, result.ReasoningPolicy.ThinkingLevelMap,
			lv("off", "null", "minimal", "null", "low", "low", "medium", "medium", "high", "high", "xhigh", "null", "max", "null"))
	})

	catalogTiers := []ai.CostTier{tier(200_000, 6, 22.5, 0.6, 7.5)}
	t.Run("keeps catalog tiers off a deployment whose prices the operator configured", func(t *testing.T) {
		catalog := fixed(CatalogResolution{Provider: "openai", Reasoning: b(true), Cost: cost4(3, 15, 0.3, 3.75, catalogTiers...)})
		priced := row(idMode("custom", "chat"), info(func(i *types.ModelInfoDetails) {
			i.InputCostPerToken, i.OutputCostPerToken = f(0.000001), f(0.000002)
			i.CacheReadInputTokenCost, i.CacheCreationInputTokenCost = f(0.0000001), f(0.0000002)
		}))

		result := mustReduce(t, []types.ModelInfoEntry{priced}, catalog)

		if result.Cost.Tiers != nil || result.Cost.Input != 1 || result.Cost.Output != 2 {
			t.Fatalf("%s", dump(result.Cost))
		}
	})

	t.Run("substitutes an operator price into every catalog tier for that field only", func(t *testing.T) {
		catalog := fixed(CatalogResolution{Provider: "openai", Reasoning: b(true), Cost: cost4(3, 15, 0.3, 3.75, catalogTiers...)})
		partial := row(idMode("partial", "chat"), unpriced, info(func(i *types.ModelInfoDetails) { i.InputCostPerToken = f(0.000001) }))

		eq(t, mustReduce(t, []types.ModelInfoEntry{partial}, catalog).Cost.Tiers, []ai.CostTier{tier(200_000, 1, 22.5, 0.6, 7.5)})
	})

	t.Run("adopts tiered pricing when identical tiers are declared in any property order", func(t *testing.T) {
		// Go values have no property order; the case keeps its alternating-resolver shape.
		withTiers := fixed(CatalogResolution{
			Provider: "anthropic", Reasoning: b(true), Vision: b(true), ContextWindow: f(200_000), MaxTokens: f(64_000),
			Cost: cost4(3, 15, 0.3, 3.75, catalogTiers...),
		})
		rows := []types.ModelInfoEntry{row(idMode("a", "chat"), unpriced), row(idMode("b", "chat"), unpriced)}

		eq(t, mustReduce(t, rows, withTiers).Cost.Tiers, catalogTiers)
		call := 0
		alternating := func(entry types.ModelInfoEntry) *CatalogResolution {
			call++
			return withTiers(entry)
		}
		eq(t, mustReduce(t, rows, alternating).Cost.Tiers, catalogTiers)
	})

	t.Run("builds the union-threshold envelope for deployments with different ladders", func(t *testing.T) {
		rows := []types.ModelInfoEntry{row(idMode("a", "chat"), unpriced), row(idMode("b", "chat"), unpriced)}
		call := 0
		differing := func(types.ModelInfoEntry) *CatalogResolution {
			ladder := tier(400_000, 5, 22.5, 0.5, 7.5)
			if call == 0 {
				ladder = tier(200_000, 6, 18, 0.6, 4)
			}
			call++
			return &CatalogResolution{
				Provider: "anthropic", Reasoning: b(true), Vision: b(true), ContextWindow: f(200_000), MaxTokens: f(64_000),
				Cost: cost4(3, 15, 0.3, 3.75, ladder),
			}
		}

		eq(t, mustReduce(t, rows, differing).Cost.Tiers, []ai.CostTier{
			tier(200_000, 6, 18, 0.6, 4),
			tier(400_000, 6, 22.5, 0.6, 7.5),
		})
	})

	t.Run("omits tiered pricing and thinking maps entirely when no catalog declares them", func(t *testing.T) {
		result := mustReduce(t, []types.ModelInfoEntry{row()}, resolveCatalog)

		if result.Cost.Tiers != nil || result.ThinkingLevelMap != nil {
			t.Fatalf("%s", dump(result))
		}
	})

	t.Run("disables catalog authority for conflicting provider identities", func(t *testing.T) {
		result := mustReduce(t, []types.ModelInfoEntry{
			row(idMode("anthropic", "chat")),
			row(idMode("openai", "chat"), backendModel("openai/gpt-4o")),
		}, resolveCatalog)

		if result.CatalogProvider != "" || result.ThinkingLevelMap != nil {
			t.Fatalf("%s", dump(result))
		}
	})

	// LiteLLM omits the list (or returns null) for an off-map deployment; only that absence is an
	// operator opt-in. The Vitest rows for a string or object list cannot be decoded into the typed
	// wire model, so only the absent and null forms (both a nil slice) are ported.
	for _, label := range []string{"undefined", "null"} {
		t.Run("treats supported_openai_params="+label+" with supports_reasoning as carrier=true", func(t *testing.T) {
			result := mustReduce(t, []types.ModelInfoEntry{
				row(info(func(i *types.ModelInfoDetails) { i.SupportedOpenAIParams, i.SupportsReasoning = nil, b(true) }),
					backendModel("internal/reasoner")),
			}, none)

			if !result.AcceptsResponsesReasoningControl {
				t.Fatal("expected carrier")
			}
		})
	}

	t.Run("intersects accepted parameters across deployments", func(t *testing.T) {
		result := mustReduce(t, []types.ModelInfoEntry{
			row(idMode("a", "chat"), info(func(i *types.ModelInfoDetails) { i.SupportedOpenAIParams = []string{"temperature", "reasoning_effort"} }),
				backendModel("internal/a")),
			row(idMode("b", "chat"), info(func(i *types.ModelInfoDetails) { i.SupportedOpenAIParams = []string{"reasoning_effort", "thinking"} }),
				params(func(p *types.ModelInfoParams) {
					p.Model, p.AllowedOpenAIParams = "internal/b", []string{"reasoning_effort"}
				})),
		}, resolveCatalog)

		eq(t, result.AcceptedOpenAIParams, []string{"reasoning_effort"})
	})

	policyCases := []struct {
		name     string
		semantic SemanticModel
		params   []string
		expected ReasoningPolicy
	}{
		{
			"Kimi K2.6 with binary thinking", SemanticModelKimiK25K26, []string{"thinking"},
			ReasoningPolicy{
				Reasoning:        true,
				ThinkingLevelMap: lv("off", "off", "minimal", "null", "low", "null", "medium", "null", "high", "high", "xhigh", "null", "max", "null"),
				Compat:           &ReasoningCompat{ThinkingFormat: "deepseek", SupportsReasoningEffort: b(false)},
			},
		},
		{
			// K2.7 Code cannot be switched off, so `off` stays denied while `high` rides the accepted
			// `thinking` param.
			"Kimi K2.7 Code with accepted thinking", SemanticModelKimiK27Code, []string{"thinking"},
			ReasoningPolicy{
				Reasoning:        true,
				ThinkingLevelMap: lv("off", "null", "minimal", "null", "low", "null", "medium", "null", "high", "high", "xhigh", "null", "max", "null"),
				Compat: &ReasoningCompat{
					SupportsReasoningEffort: b(false), RequiresReasoningContentOnAssistantMessages: b(true), ThinkingFormat: "deepseek",
				},
			},
		},
		{
			"Kimi K2.7 Code without accepted controls", SemanticModelKimiK27Code, nil,
			ReasoningPolicy{
				Reasoning:        true,
				ThinkingLevelMap: noLevels(),
				Compat:           &ReasoningCompat{SupportsReasoningEffort: b(false), RequiresReasoningContentOnAssistantMessages: b(true)},
			},
		},
		{
			"Kimi K2.6 without accepted controls", SemanticModelKimiK25K26, nil,
			ReasoningPolicy{
				Reasoning:        true,
				ThinkingLevelMap: noLevels(),
				Compat:           &ReasoningCompat{SupportsReasoningEffort: b(false)},
			},
		},
		{
			"DeepSeek V4 through a thinking-only route", SemanticModelDeepSeekV4, []string{"thinking"},
			ReasoningPolicy{
				Reasoning:        true,
				ThinkingLevelMap: lv("off", "off", "minimal", "null", "low", "null", "medium", "null", "high", "high", "xhigh", "null", "max", "null"),
				Compat: &ReasoningCompat{
					ThinkingFormat: "deepseek", SupportsReasoningEffort: b(false), RequiresReasoningContentOnAssistantMessages: b(true),
				},
			},
		},
	}
	for _, c := range policyCases {
		t.Run("derives "+c.name+" policy from semantic and accepted-control evidence", func(t *testing.T) {
			result := mustReduce(t, []types.ModelInfoEntry{
				row(info(func(i *types.ModelInfoDetails) { i.SupportedOpenAIParams = c.params }), backendModel("internal/model")),
			}, fixed(CatalogResolution{SemanticModel: c.semantic}))

			eq(t, result.ReasoningPolicy, c.expected)
		})
	}

	literalCases := []struct {
		name     string
		semantic SemanticModel
		params   []string
		expected ReasoningPolicy
	}{
		{
			// Without `thinking` there is no carrier for K2.7 Code's binary control, and an accepted
			// `reasoning_effort` with no public level opinion cannot reopen the levels the generation denies.
			"Kimi K2.7 Code with effort", SemanticModelKimiK27Code, []string{"reasoning_effort"},
			ReasoningPolicy{
				Reasoning:        true,
				ThinkingLevelMap: noLevels(),
				Compat:           &ReasoningCompat{SupportsReasoningEffort: b(false), RequiresReasoningContentOnAssistantMessages: b(true)},
			},
		},
		{
			"Kimi K3 with effort", SemanticModelKimiK3, []string{"reasoning_effort"},
			ReasoningPolicy{
				Reasoning:        true,
				ThinkingLevelMap: lv("off", "null", "xhigh", "null", "max", "null"),
				Compat: &ReasoningCompat{
					ThinkingFormat: "openai", SupportsReasoningEffort: b(true), RequiresReasoningContentOnAssistantMessages: b(true),
				},
			},
		},
		{
			"DeepSeek V4 with native controls", SemanticModelDeepSeekV4, []string{"thinking", "reasoning_effort"},
			ReasoningPolicy{
				Reasoning:        true,
				ThinkingLevelMap: lv("off", "off", "xhigh", "null", "max", "null"),
				Compat: &ReasoningCompat{
					ThinkingFormat: "deepseek", SupportsReasoningEffort: b(true), RequiresReasoningContentOnAssistantMessages: b(true),
				},
			},
		},
		{
			"DeepSeek V4 through an effort-only route", SemanticModelDeepSeekV4, []string{"reasoning_effort"},
			ReasoningPolicy{
				Reasoning:        true,
				ThinkingLevelMap: lv("off", "null", "xhigh", "null", "max", "null"),
				Compat: &ReasoningCompat{
					ThinkingFormat: "openai", SupportsReasoningEffort: b(true), RequiresReasoningContentOnAssistantMessages: b(true),
				},
			},
		},
	}
	for _, c := range literalCases {
		t.Run("derives "+c.name+" policy from literal level-map expectations", func(t *testing.T) {
			result := mustReduce(t, []types.ModelInfoEntry{
				row(info(func(i *types.ModelInfoDetails) { i.SupportedOpenAIParams = c.params }), backendModel("internal/model")),
			}, fixed(CatalogResolution{SemanticModel: c.semantic}))

			eq(t, result.ReasoningPolicy, c.expected)
		})
	}

	kimiRow := func(extra ...func(*types.ModelInfoDetails)) types.ModelInfoEntry {
		return row(info(func(i *types.ModelInfoDetails) {
			i.SupportsReasoning, i.SupportedOpenAIParams = b(true), []string{"thinking", "reasoning_effort"}
			for _, fn := range extra {
				fn(i)
			}
		}), backendModel("openrouter/moonshotai/kimi-k2.6"))
	}
	kimiCatalog := func(effort ...string) CatalogResolver {
		return fixed(CatalogResolution{Provider: "moonshotai", Reasoning: b(true), SemanticModel: SemanticModelKimiK25K26, EffortLevels: effort})
	}
	binaryKimi := lv("off", "off", "minimal", "null", "low", "null", "medium", "null", "high", "high", "xhigh", "null", "max", "null")

	// LiteLLM reports both carriers for K2.5/K2.6 on OpenRouter and Databricks. The generation still
	// has one on/off switch, so the deployment map's absent standard levels must not become five
	// selectable efforts.
	t.Run("keeps the binary Kimi map when a host also accepts reasoning_effort", func(t *testing.T) {
		result := mustReduce(t, []types.ModelInfoEntry{kimiRow()}, kimiCatalog())

		eq(t, result.ReasoningPolicy.ThinkingLevelMap, binaryKimi)
	})

	t.Run("lets an explicit LiteLLM denial close a level the binary Kimi map keeps", func(t *testing.T) {
		result := mustReduce(t, []types.ModelInfoEntry{
			kimiRow(func(i *types.ModelInfoDetails) { i.SupportsNoneReasoningEffort = b(false) }),
		}, kimiCatalog())

		got := result.ReasoningPolicy.ThinkingLevelMap
		eq(t, []*string{got["off"], got["high"], got["low"]}, []*string{nil, s("high"), nil})
	})

	t.Run("lets a public effort list govern Kimi levels while preserving semantic off", func(t *testing.T) {
		result := mustReduce(t, []types.ModelInfoEntry{kimiRow()}, kimiCatalog("low", "high"))

		eq(t, result.ReasoningPolicy.ThinkingLevelMap,
			lv("off", "off", "minimal", "null", "low", "low", "medium", "null", "high", "high", "xhigh", "null", "max", "null"))
	})

	t.Run("keeps the binary Kimi map when a public effort list has no recognized values", func(t *testing.T) {
		result := mustReduce(t, []types.ModelInfoEntry{kimiRow()}, kimiCatalog("adaptive"))

		eq(t, result.ReasoningPolicy.ThinkingLevelMap, binaryKimi)
	})

	t.Run("preserves catalog map denials when its public effort list is empty", func(t *testing.T) {
		result := mustReduce(t, []types.ModelInfoEntry{
			row(info(func(i *types.ModelInfoDetails) {
				i.SupportsReasoning, i.SupportedOpenAIParams = b(true), []string{"reasoning_effort"}
			}), backendModel("openai/private-reasoning")),
		}, fixed(CatalogResolution{Provider: "openai", Reasoning: b(true), EffortLevels: []string{}, ThinkingLevelMap: lv("off", "null")}))

		eq(t, result.ThinkingLevelMap, lv("off", "null", "xhigh", "null", "max", "null"))
	})

	t.Run("lets catalog map denials override a models.dev effort list", func(t *testing.T) {
		result := mustReduce(t, []types.ModelInfoEntry{
			row(info(func(i *types.ModelInfoDetails) {
				i.SupportsReasoning, i.SupportedOpenAIParams = b(true), []string{"reasoning_effort"}
			}), backendModel("openai/private-reasoning")),
		}, fixed(CatalogResolution{
			Provider: "openai", Reasoning: b(true), EffortLevels: []string{"low", "high"}, ThinkingLevelMap: lv("off", "null", "low", "null"),
		}))

		eq(t, result.ThinkingLevelMap,
			lv("off", "null", "minimal", "null", "low", "null", "medium", "null", "high", "high", "xhigh", "null", "max", "null"))
	})

	for _, c := range []struct {
		name         string
		supportsNone *bool
		expectedOff  string
	}{
		{"lets semantic off through without an explicit denial", nil, "off"},
		{"lets an explicit LiteLLM denial override semantic off", b(false), "null"},
	} {
		t.Run(c.name+" for DeepSeek V4", func(t *testing.T) {
			result := mustReduce(t, []types.ModelInfoEntry{
				row(info(func(i *types.ModelInfoDetails) {
					i.SupportsReasoning, i.SupportsNoneReasoningEffort = b(true), c.supportsNone
					i.SupportedOpenAIParams = []string{"thinking", "reasoning_effort"}
				}), backendModel("deepseek/deepseek-v4-pro")),
			}, fixed(CatalogResolution{
				Provider: "deepseek", Reasoning: b(true), SemanticModel: SemanticModelDeepSeekV4, EffortLevels: []string{"low", "high"},
			}))

			eq(t, result.ReasoningPolicy.ThinkingLevelMap,
				lv("off", c.expectedOff, "minimal", "null", "low", "low", "medium", "null", "high", "high", "xhigh", "null", "max", "null"))
		})
	}

	t.Run("keeps a standard level a wildcard parent leaves at Pi's default but closes extended ones", func(t *testing.T) {
		eq(t, thinking.Intersect([]ai.ThinkingLevelMap{nil, lv("high", "high")}), lv("high", "high"))
		eq(t, thinking.Intersect([]ai.ThinkingLevelMap{nil, lv("max", "max")}), lv("max", "null"))
	})

	t.Run("closes a level when wildcard parents disagree on its wire value", func(t *testing.T) {
		eq(t, thinking.Intersect([]ai.ThinkingLevelMap{lv("high", "high"), lv("high", "max")}), lv("high", "null"))
	})

	for _, semantic := range []SemanticModel{SemanticModelKimiK3, SemanticModelDeepSeekV4} {
		t.Run("preserves "+string(semantic)+" capability and replay without accepted-control evidence", func(t *testing.T) {
			result := mustReduce(t, []types.ModelInfoEntry{
				row(info(func(i *types.ModelInfoDetails) { i.SupportsReasoning = b(true) }), backendModel("internal/"+string(semantic))),
			}, fixed(CatalogResolution{SemanticModel: semantic}))

			eq(t, result.ReasoningPolicy, ReasoningPolicy{
				Reasoning:        true,
				ThinkingLevelMap: noLevels(),
				Compat:           &ReasoningCompat{RequiresReasoningContentOnAssistantMessages: b(true), SupportsReasoningEffort: b(false)},
			})
		})
	}

	t.Run("lets any explicit reasoning denial override accepted-control promotion", func(t *testing.T) {
		withReasoning := func(id string, supported bool) types.ModelInfoEntry {
			return row(info(func(i *types.ModelInfoDetails) {
				i.ID, i.SupportsReasoning, i.SupportedOpenAIParams = id, b(supported), []string{"thinking"}
			}), backendModel("moonshot/kimi-k2.6"))
		}
		result := mustReduce(t, []types.ModelInfoEntry{withReasoning("denied", false), withReasoning("accepted", true)},
			fixed(CatalogResolution{SemanticModel: SemanticModelKimiK25K26, Reasoning: b(true)}))

		if result.Reasoning {
			t.Fatal("reasoning")
		}
		eq(t, result.ReasoningPolicy, ReasoningPolicy{Reasoning: false, Compat: &ReasoningCompat{SupportsReasoningEffort: b(false)}})
	})

	for _, model := range []string{"moonshot/kimi-k2-thinking", "moonshot/kimi_k2_thinking", "moonshot/kimi.k2.thinking"} {
		t.Run("preserves always-thinking Kimi display behavior for "+model, func(t *testing.T) {
			result := mustReduce(t, []types.ModelInfoEntry{
				row(name("misleading-public-route"), backendModel(model), info(func(i *types.ModelInfoDetails) { i.SupportsReasoning = b(true) })),
			}, none)

			if result.NormalizeThinkTags || result.SuppressReasoningVisibility {
				t.Fatalf("%s", dump(result))
			}
		})
	}

	for _, c := range []struct {
		name   string
		model  string
		detail func(*types.ModelInfoDetails)
	}{
		{"Azure-hosted Kimi", "azure/FW-Kimi-K3", func(i *types.ModelInfoDetails) {
			i.BaseModel, i.LiteLLMProvider = "fireworks/accounts/fireworks/models/kimi-k3", "azure"
		}},
		{"Bedrock-hosted Kimi", "bedrock/moonshotai.kimi-k2.5", func(i *types.ModelInfoDetails) {
			i.BaseModel, i.LiteLLMProvider = "moonshotai.kimi-k2.5", "bedrock_converse"
		}},
	} {
		t.Run("normalizes "+c.name+" responses without sending Moonshot visibility parameters", func(t *testing.T) {
			result := mustReduce(t, []types.ModelInfoEntry{
				row(name("hosted-kimi"), backendModel(c.model), info(c.detail), info(func(i *types.ModelInfoDetails) {
					i.Mode, i.SupportsReasoning = s("chat"), b(true)
				})),
			}, fixed(CatalogResolution{SemanticFamily: SemanticFamilyKimi}))

			if !result.NormalizeThinkTags || result.SuppressReasoningVisibility {
				t.Fatalf("%s", dump(result))
			}
		})
	}

	t.Run("suppresses reasoning visibility on Moonshot transport", func(t *testing.T) {
		result := mustReduce(t, []types.ModelInfoEntry{
			row(name("k3-prod"), backendModel("moonshot/kimi-k2.5"), info(func(i *types.ModelInfoDetails) {
				i.Mode, i.SupportsReasoning = s("chat"), b(true)
			})),
		}, fixed(CatalogResolution{SemanticFamily: SemanticFamilyKimi}))

		if !result.NormalizeThinkTags || !result.SuppressReasoningVisibility {
			t.Fatalf("%s", dump(result))
		}
	})

	t.Run("keeps conflicting routing signals from enabling Moonshot visibility parameters", func(t *testing.T) {
		result := mustReduce(t, []types.ModelInfoEntry{
			row(name("conflicting-route"), params(func(p *types.ModelInfoParams) {
				p.Model, p.CustomLLMProvider = "azure_ai/FW-Kimi-K3", "moonshot"
			}), info(func(i *types.ModelInfoDetails) { i.Mode, i.SupportsReasoning = s("chat"), b(true) })),
		}, fixed(CatalogResolution{SemanticFamily: SemanticFamilyKimi}))

		if !result.NormalizeThinkTags || result.SuppressReasoningVisibility {
			t.Fatalf("%s", dump(result))
		}
	})

	t.Run("does not suppress visibility when any Kimi deployment is always-thinking", func(t *testing.T) {
		result := mustReduce(t, []types.ModelInfoEntry{
			row(name("mixed-kimi-route"), backendModel("moonshot/kimi-k2.6"), info(func(i *types.ModelInfoDetails) { i.ID = "normal" })),
			row(name("mixed-kimi-route"), backendModel("moonshot/kimi-k2-thinking"), info(func(i *types.ModelInfoDetails) { i.ID = "thinking" })),
		}, fixed(CatalogResolution{SemanticFamily: SemanticFamilyKimi}))

		if result.NormalizeThinkTags || result.SuppressReasoningVisibility {
			t.Fatalf("%s", dump(result))
		}
	})

	for _, c := range []struct{ name, model string }{
		{"Claude", "anthropic/claude-sonnet-4-6"}, {"OpenAI", "openai/gpt-4o"},
	} {
		t.Run("does not normalize think tags for a mixed Kimi/"+c.name+" route", func(t *testing.T) {
			result := mustReduce(t, []types.ModelInfoEntry{
				row(name("mixed-family-route"), backendModel("moonshot/kimi-k2.6"), info(func(i *types.ModelInfoDetails) { i.ID = "kimi" })),
				row(name("mixed-family-route"), backendModel(c.model), info(func(i *types.ModelInfoDetails) { i.ID = "other" })),
			}, resolveCatalog)

			if result.NormalizeThinkTags || result.SuppressReasoningVisibility {
				t.Fatalf("%s", dump(result))
			}
		})
	}

	t.Run("lets explicit unanimous reasoning denial override the K2.7 Code contract", func(t *testing.T) {
		result := mustReduce(t, []types.ModelInfoEntry{
			row(info(func(i *types.ModelInfoDetails) {
				i.SupportsReasoning, i.SupportedOpenAIParams = b(false), []string{"thinking"}
			}), backendModel("moonshot/kimi-k2.7-code")),
		}, fixed(CatalogResolution{SemanticModel: SemanticModelKimiK27Code, Reasoning: b(true)}))

		if result.Reasoning {
			t.Fatal("reasoning")
		}
		eq(t, result.ReasoningPolicy, ReasoningPolicy{
			Reasoning: false,
			Compat:    &ReasoningCompat{SupportsReasoningEffort: b(false), RequiresReasoningContentOnAssistantMessages: b(true)},
		})
	})

	t.Run("fails closed for mixed semantic generations and accepted controls", func(t *testing.T) {
		deployments := []types.ModelInfoEntry{
			row(info(func(i *types.ModelInfoDetails) { i.ID, i.SupportedOpenAIParams = "k2", []string{"thinking"} }), backendModel("moonshot/kimi-k2.6")),
			row(info(func(i *types.ModelInfoDetails) { i.ID, i.SupportedOpenAIParams = "k3", []string{"reasoning_effort"} }), backendModel("moonshot/kimi-k3")),
		}
		result := mustReduce(t, deployments, func(entry types.ModelInfoEntry) *CatalogResolution {
			semantic := SemanticModelKimiK3
			if entry.ModelInfo.ID == "k2" {
				semantic = SemanticModelKimiK25K26
			}
			return &CatalogResolution{SemanticModel: semantic}
		})

		eq(t, result.AcceptedOpenAIParams, []string{})
		eq(t, result.ReasoningPolicy, ReasoningPolicy{Reasoning: false})
	})
}
