package modelgroups

import (
	"encoding/json"
	"math"
	"reflect"
	"testing"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

func s(v string) *string   { return &v }
func b(v bool) *bool       { return &v }
func f(v float64) *float64 { return &v }

var nan = math.NaN()
var inf = math.Inf(1)

type mod func(*types.ModelInfoEntry)

func info(fn func(*types.ModelInfoDetails)) mod {
	return func(e *types.ModelInfoEntry) { fn(e.ModelInfo) }
}

func params(fn func(*types.ModelInfoParams)) mod {
	return func(e *types.ModelInfoEntry) { fn(e.LiteLLMParams) }
}

func name(n string) mod { return func(e *types.ModelInfoEntry) { e.ModelName = n } }

func backendModel(m string) mod { return params(func(p *types.ModelInfoParams) { p.Model = m }) }

// idMode sets the deployment id and mode ("" mode leaves it null).
func idMode(id, mode string) mod {
	return info(func(i *types.ModelInfoDetails) {
		i.ID = id
		i.Mode = nil
		if mode != "" {
			i.Mode = s(mode)
		}
	})
}

// unpriced is a deployment LiteLLM did not price, so the catalog supplies the schedule.
var unpriced = info(func(i *types.ModelInfoDetails) {
	i.InputCostPerToken, i.OutputCostPerToken, i.CacheReadInputTokenCost, i.CacheCreationInputTokenCost = nil, nil, nil, nil
})

// row is the Vitest `row()` helper.
func row(mods ...mod) types.ModelInfoEntry {
	e := types.ModelInfoEntry{
		ModelName:     "route",
		LiteLLMParams: &types.ModelInfoParams{Model: "anthropic/claude-sonnet-4-6"},
		ModelInfo: &types.ModelInfoDetails{
			ID:   "deployment-a",
			Mode: s("chat"),
			// Described by LiteLLM's model map without a carrier; an absent list is an operator opt-in.
			SupportedOpenAIParams:       []string{},
			SupportsReasoning:           b(true),
			SupportsVision:              b(true),
			MaxInputTokens:              f(200_000),
			MaxOutputTokens:             f(32_000),
			InputCostPerToken:           f(0.000003),
			OutputCostPerToken:          f(0.000015),
			CacheReadInputTokenCost:     f(0.0000003),
			CacheCreationInputTokenCost: f(0.00000375),
		},
	}
	for _, m := range mods {
		m(&e)
	}
	return e
}

// bare is a Vitest literal `{ model_name, model_info: { id, mode } }` with no litellm_params.
func bare(id, mode string) types.ModelInfoEntry {
	e := types.ModelInfoEntry{ModelName: "claude-route", ModelInfo: &types.ModelInfoDetails{ID: id}}
	if mode != "" {
		e.ModelInfo.Mode = s(mode)
	}
	return e
}

func cost4(input, output, cacheRead, cacheWrite float64, tiers ...ai.CostTier) *CatalogCost {
	return &CatalogCost{Input: f(input), Output: f(output), CacheRead: f(cacheRead), CacheWrite: f(cacheWrite), Tiers: tiers}
}

func modelCost(input, output, cacheRead, cacheWrite float64, tiers ...ai.CostTier) ai.ModelCost {
	return ai.ModelCost{Input: input, Output: output, CacheRead: cacheRead, CacheWrite: cacheWrite, Tiers: tiers}
}

func tier(above int, input, output, cacheRead, cacheWrite float64) ai.CostTier {
	return ai.CostTier{InputTokensAbove: above, InputCostPer1M: input, OutputCostPer1M: output, CacheReadCostPer1M: cacheRead, CacheWriteCostPer1M: cacheWrite}
}

func catalogCostOf(c ai.ModelCost) *CatalogCost {
	return &CatalogCost{Input: f(c.Input), Output: f(c.Output), CacheRead: f(c.CacheRead), CacheWrite: f(c.CacheWrite), Tiers: c.Tiers}
}

// lv builds a level map from level, value pairs; the value "null" denies the level.
func lv(pairs ...string) ai.ThinkingLevelMap {
	out := ai.ThinkingLevelMap{}
	for i := 0; i < len(pairs); i += 2 {
		if pairs[i+1] == "null" {
			out[ai.ThinkingLevel(pairs[i])] = nil
		} else {
			out[ai.ThinkingLevel(pairs[i])] = s(pairs[i+1])
		}
	}
	return out
}

func noLevels() ai.ThinkingLevelMap {
	return lv("off", "null", "minimal", "null", "low", "null", "medium", "null", "high", "null", "xhigh", "null", "max", "null")
}

func fixed(c CatalogResolution) CatalogResolver {
	return func(types.ModelInfoEntry) *CatalogResolution {
		copied := c
		return &copied
	}
}

func none(types.ModelInfoEntry) *CatalogResolution { return nil }

var testCatalog = map[string]CatalogResolution{
	"openai/gpt-4o": {
		Provider: "openai", Reasoning: b(false), Vision: b(true),
		ContextWindow: f(128_000), MaxTokens: f(16_384), Cost: cost4(5, 15, 2.5, 0),
	},
	"anthropic/claude-sonnet-4-6": {
		Provider: "anthropic", Reasoning: b(true), Vision: b(true),
		ContextWindow: f(200_000), MaxTokens: f(64_000), Cost: cost4(3, 15, 0.3, 3.75),
	},
	"bedrock/anthropic.claude-sonnet-4-6": {
		Provider: "amazon-bedrock", Reasoning: b(true), Vision: b(true),
		ContextWindow: f(200_000), MaxTokens: f(64_000), Cost: cost4(3, 15, 0.3, 3.75),
	},
}

func resolveCatalog(entry types.ModelInfoEntry) *CatalogResolution {
	backendID := ""
	if entry.LiteLLMParams != nil {
		backendID = entry.LiteLLMParams.Model
	}
	if backendID == "" && entry.ModelInfo != nil {
		backendID = entry.ModelInfo.BaseModel
	}
	if found, ok := testCatalog[backendID]; ok {
		return &found
	}
	return nil
}

func permutations[T any](values []T) [][]T {
	if len(values) < 2 {
		return [][]T{append([]T(nil), values...)}
	}
	var out [][]T
	for i, value := range values {
		rest := append(append([]T(nil), values[:i]...), values[i+1:]...)
		for _, tail := range permutations(rest) {
			out = append(out, append([]T{value}, tail...))
		}
	}
	return out
}

func dump(v any) string {
	encoded, err := json.Marshal(v)
	if err != nil {
		return err.Error()
	}
	return string(encoded)
}

func eq(t *testing.T, got, want any) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got  %s\nwant %s", dump(got), dump(want))
	}
}

func near(t *testing.T, got, want float64) {
	t.Helper()
	if math.Abs(got-want) > 1e-9 {
		t.Fatalf("got %v want %v", got, want)
	}
}

func mustReduce(t *testing.T, entries []types.ModelInfoEntry, resolver CatalogResolver) *ReducedModelGroup {
	t.Helper()
	group := ReduceModelGroup(entries, resolver)
	if group == nil {
		t.Fatal("expected a reduced model group, got nil")
	}
	return group
}
