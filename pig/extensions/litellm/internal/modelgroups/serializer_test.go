// Ports tests/model-groups.test.ts: toResponsesLevels, closeSerializerPolicy, meetVendorCompat
package modelgroups

import (
	"testing"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

func TestToResponsesLevels(t *testing.T) {
	cases := []struct {
		name     string
		levels   ai.ThinkingLevelMap
		expected ai.ThinkingLevelMap
	}{
		{
			"an absent map", nil,
			lv("off", "none", "minimal", "minimal", "low", "low", "medium", "medium", "high", "high", "xhigh", "null", "max", "null"),
		},
		{
			"a partial Chat map", lv("low", "high"),
			lv("off", "none", "minimal", "minimal", "low", "high", "medium", "medium", "high", "high", "xhigh", "null", "max", "null"),
		},
		{
			"explicit extended levels", lv("off", "null", "xhigh", "xhigh", "max", "max"),
			lv("off", "null", "minimal", "minimal", "low", "low", "medium", "medium", "high", "high", "xhigh", "xhigh", "max", "max"),
		},
		{
			"a Chat-only value with no Responses spelling", lv("high", "off"),
			lv("off", "none", "minimal", "minimal", "low", "low", "medium", "medium", "high", "null", "xhigh", "null", "max", "null"),
		},
	}
	for _, c := range cases {
		t.Run("never widens Responses beyond Chat for "+c.name, func(t *testing.T) {
			eq(t, ToResponsesLevels(c.levels), c.expected)
		})
	}
}

func TestCloseSerializerPolicy(t *testing.T) {
	t.Run("retains only Responses compatibility when closing a Responses policy", func(t *testing.T) {
		policy := CloseSerializerPolicy(CloseSerializerPolicyInput{
			API:       types.APIOpenAIResponses,
			Reasoning: true,
			VendorCompat: &ai.ModelCompat{
				SupportsDeveloperRole:           b(false),
				SupportsStrictMode:              b(true),
				SessionAffinityFormat:           ai.SessionAffinityOpenAI,
				SupportsLongCacheRetention:      b(false),
				SupportsOpenAIGrammarTools:      b(true),
				SupportsAdditionalTools:         b(true),
				SupportsToolSearch:              b(true),
				SupportsExplicitPromptCacheMode: b(true),
				ThinkingFormat:                  "deepseek",
				SupportsReasoningEffort:         b(false),
			},
			DenyLevels: true,
		})
		eq(t, policy.Compat, &ai.ModelCompat{
			SupportsDeveloperRole:           b(false),
			SupportsStrictMode:              b(true),
			SessionAffinityFormat:           ai.SessionAffinityOpenAI,
			SupportsLongCacheRetention:      b(false),
			SupportsOpenAIGrammarTools:      b(true),
			SupportsAdditionalTools:         b(true),
			SupportsToolSearch:              b(true),
			SupportsExplicitPromptCacheMode: b(true),
		})
		eq(t, policy.ThinkingLevelMap, noLevels())
		closed := CloseSerializerPolicy(CloseSerializerPolicyInput{
			API:          types.APIOpenAIResponses,
			Reasoning:    false,
			VendorCompat: &ai.ModelCompat{SupportsReasoningEffort: b(false)},
			DenyLevels:   true,
		})
		if closed.Compat != nil {
			t.Fatalf("expected nil compat, got %s", dump(closed.Compat))
		}
	})

	t.Run("keeps Chat and Responses closed when vendor compatibility denies reasoning effort", func(t *testing.T) {
		for _, api := range []types.LiteLLMApi{types.APIOpenAICompletions, types.APIOpenAIResponses} {
			policy := CloseSerializerPolicy(CloseSerializerPolicyInput{
				API:           api,
				Reasoning:     true,
				VendorCompat:  &ai.ModelCompat{SupportsReasoningEffort: b(false)},
				CatalogLevels: lv("off", "off", "low", "low", "high", "high"),
			})
			eq(t, policy.ThinkingLevelMap, noLevels())
		}
	})

	for _, api := range []types.LiteLLMApi{types.APIOpenAICompletions, types.APIOpenAIResponses} {
		t.Run("makes denyLevels explicitly deny selectable levels for "+string(api), func(t *testing.T) {
			var compat *ai.ModelCompat
			if api != types.APIOpenAIResponses {
				compat = &ai.ModelCompat{SupportsStore: b(false), SupportsReasoningEffort: b(false)}
			}
			got := CloseSerializerPolicy(CloseSerializerPolicyInput{
				API:                              api,
				Reasoning:                        true,
				VendorCompat:                     &ai.ModelCompat{SupportsStore: b(false), SupportsReasoningEffort: b(true)},
				CatalogLevels:                    lv("low", "low", "high", "high"),
				AcceptsResponsesReasoningControl: true,
				DenyLevels:                       true,
			})
			eq(t, got, SerializerPolicy{Reasoning: true, ThinkingLevelMap: noLevels(), Compat: compat})
		})
	}

	t.Run("denies Responses levels until reasoning_effort acceptance is evidenced", func(t *testing.T) {
		input := CloseSerializerPolicyInput{
			API:            types.APIOpenAIResponses,
			Reasoning:      true,
			VendorCompat:   &ai.ModelCompat{SupportsStore: b(false)},
			SemanticLevels: lv("off", "off", "high", "high", "max", "max"),
		}
		eq(t, CloseSerializerPolicy(input).ThinkingLevelMap, noLevels())
		input.AcceptsResponsesReasoningControl = true
		eq(t, CloseSerializerPolicy(input).ThinkingLevelMap,
			lv("off", "none", "minimal", "minimal", "low", "low", "medium", "medium", "high", "high", "xhigh", "null", "max", "max"))
	})

	t.Run("denies implicit Chat levels until a carrier is evidenced", func(t *testing.T) {
		input := CloseSerializerPolicyInput{
			API:          types.APIOpenAICompletions,
			Reasoning:    true,
			VendorCompat: &ai.ModelCompat{SupportsStore: b(false)},
		}
		withCarrier := input
		withCarrier.RequireChatCarrier = true
		eq(t, CloseSerializerPolicy(withCarrier).ThinkingLevelMap, noLevels())

		input.VendorCompat = &ai.ModelCompat{SupportsStore: b(false), SupportsReasoningEffort: b(true)}
		eq(t, CloseSerializerPolicy(input), SerializerPolicy{
			Reasoning: true,
			Compat:    &ai.ModelCompat{SupportsStore: b(false), SupportsReasoningEffort: b(true)},
		})
	})
}

func TestMeetVendorCompat(t *testing.T) {
	t.Run("keeps Moonshot restrictions but withholds shape changes from an unidentified sibling", func(t *testing.T) {
		got := MeetVendorCompat([]*ai.ModelCompat{
			{
				SupportsStore:           b(false),
				SupportsDeveloperRole:   b(false),
				SupportsReasoningEffort: b(false),
				SupportsStrictMode:      b(false),
				MaxTokensField:          "max_tokens",
			},
			nil,
		})
		eq(t, got, &ai.ModelCompat{
			SupportsStore:           b(false),
			SupportsDeveloperRole:   b(false),
			SupportsReasoningEffort: b(false),
			SupportsStrictMode:      b(false),
		})
	})

	t.Run("retains the complete Moonshot block only when every deployment agrees", func(t *testing.T) {
		moonshot := func() *ai.ModelCompat {
			return &ai.ModelCompat{
				SupportsStore:           b(false),
				SupportsDeveloperRole:   b(false),
				SupportsReasoningEffort: b(false),
				SupportsStrictMode:      b(false),
				MaxTokensField:          "max_tokens",
			}
		}
		eq(t, MeetVendorCompat([]*ai.ModelCompat{moonshot(), moonshot()}), moonshot())
	})
}
