package modelgroups

import (
	"encoding/json"
	"testing"

	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

// wireRow is the Vitest `row()` with overrides given as raw JSON, decoded through the lenient
// /model/info decoder so that mistyped fields reach the reduction the way a proxy would send them.
func wireRow(t *testing.T, top, modelInfo, litellmParams string) types.ModelInfoEntry {
	t.Helper()
	merged := func(base, overrides string) map[string]any {
		out := map[string]any{}
		for _, doc := range []string{base, overrides} {
			var parsed map[string]any
			if doc == "" {
				continue
			}
			if err := json.Unmarshal([]byte(doc), &parsed); err != nil {
				t.Fatal(err)
			}
			for k, v := range parsed {
				out[k] = v
			}
		}
		return out
	}
	doc := merged(`{"model_name":"route"}`, top)
	doc["model_info"] = merged(`{"id":"deployment-a","mode":"chat","supported_openai_params":[],"supports_reasoning":true,
		"supports_vision":true,"max_input_tokens":200000,"max_output_tokens":32000,"input_cost_per_token":0.000003,
		"output_cost_per_token":0.000015,"cache_read_input_token_cost":0.0000003,"cache_creation_input_token_cost":0.00000375}`, modelInfo)
	doc["litellm_params"] = merged(`{"model":"anthropic/claude-sonnet-4-6"}`, litellmParams)
	encoded, err := json.Marshal(doc)
	if err != nil {
		t.Fatal(err)
	}
	var entry types.ModelInfoEntry
	if err := json.Unmarshal(encoded, &entry); err != nil {
		t.Fatalf("lenient decode failed: %v", err)
	}
	return entry
}

func TestMistypedWireFields(t *testing.T) {
	claude := func(compat MessagesBackendCompat) CatalogResolution {
		return CatalogResolution{Provider: "amazon-bedrock", SemanticFamily: SemanticFamilyClaude, MessagesCompat: &compat}
	}
	t.Run("treats an unreadable mode as unknown rather than as evidence of a non-chat deployment", func(t *testing.T) {
		roomy := wireRow(t, "", `{"id":"roomy","mode":"chat","max_input_tokens":200000}`, "")
		unreadable := wireRow(t, "", `{"id":"cramped","max_input_tokens":8000,"mode":7}`, "")
		embedding := wireRow(t, "", `{"id":"cramped","max_input_tokens":8000,"mode":"embedding"}`, "")

		got := mustReduce(t, []types.ModelInfoEntry{roomy, unreadable}, resolveCatalog)
		if got.ContextWindow != 8_000 || got.API != types.APIOpenAICompletions {
			t.Fatalf("%v %s", got.ContextWindow, got.API)
		}
		if ReduceModelGroup([]types.ModelInfoEntry{roomy, embedding}, resolveCatalog) != nil {
			t.Fatal("expected nil")
		}
		got = mustReduce(t, []types.ModelInfoEntry{unreadable}, resolveCatalog)
		if got.ContextWindow != 8_000 || got.API != types.APIOpenAICompletions {
			t.Fatalf("%v %s", got.ContextWindow, got.API)
		}
	})

	t.Run("does not read an unreadable capability flag as true", func(t *testing.T) {
		lying := wireRow(t, "", `{"id":"lying","mode":"chat","supports_vision":"no","supports_reasoning":"false"}`, `{"model":"internal/unknown"}`)
		got := mustReduce(t, []types.ModelInfoEntry{lying}, resolveCatalog)
		if got.Vision || got.Reasoning {
			t.Fatalf("vision %v reasoning %v", got.Vision, got.Reasoning)
		}
	})

	t.Run("filters rows without a readable route name before reducing limits and capabilities", func(t *testing.T) {
		roomy := wireRow(t, "", `{"id":"roomy","mode":"chat","max_input_tokens":200000}`, "")
		badName := wireRow(t, `{"model_name":42}`, `{"id":"bad-name","mode":"chat","supports_reasoning":false,"max_input_tokens":8000}`, "")
		got := mustReduce(t, []types.ModelInfoEntry{roomy, badName}, resolveCatalog)
		if got.ContextWindow != 200_000 || !got.Reasoning {
			t.Fatalf("%v %v", got.ContextWindow, got.Reasoning)
		}
	})

	// LiteLLM omits the list (or returns null) for an off-map deployment; only that absence is an
	// operator opt-in. Present but malformed data is no carrier evidence.
	for _, c := range []struct {
		params   string
		expected bool
	}{
		{`null`, true},
		{`"reasoning_effort"`, false},
		{`{"reasoning_effort":true}`, false},
	} {
		t.Run("treats supported_openai_params="+c.params+" with supports_reasoning as carrier", func(t *testing.T) {
			entry := wireRow(t, "", `{"supported_openai_params":`+c.params+`,"supports_reasoning":true}`, `{"model":"internal/reasoner"}`)
			got := mustReduce(t, []types.ModelInfoEntry{entry}, none)
			if got.AcceptsResponsesReasoningControl != c.expected {
				t.Fatalf("carrier %v want %v", got.AcceptsResponsesReasoningControl, c.expected)
			}
		})
	}

	t.Run("treats partially malformed endpoint lists as lists", func(t *testing.T) {
		reduce := func(endpoints string) types.LiteLLMApi {
			entry := wireRow(t, `{"model_name":"claude-route"}`, `{"supported_endpoints":`+endpoints+`}`, "")
			return mustReduce(t, []types.ModelInfoEntry{entry}, fixed(claude(MessagesBackendCompat{}))).API
		}
		if got := reduce(`["/v1/messages",42]`); got != types.APIAnthropicMessages {
			t.Fatalf("including Messages: %s", got)
		}
		if got := reduce(`["/v1/chat/completions",42]`); got != types.APIOpenAICompletions {
			t.Fatalf("omitting Messages: %s", got)
		}
	})

	for _, endpoints := range []string{`"/v1/chat/completions"`, `"/v1/messages"`, `42`, `{"endpoint":"/v1/messages"}`} {
		t.Run("withholds Messages for malformed endpoint metadata "+endpoints, func(t *testing.T) {
			entry := wireRow(t, `{"model_name":"claude-route"}`, `{"supported_endpoints":`+endpoints+`}`, "")
			got := mustReduce(t, []types.ModelInfoEntry{entry}, fixed(claude(MessagesBackendCompat{})))
			if got.API != types.APIOpenAICompletions {
				t.Fatalf("api %s", got.API)
			}
		})
	}
}
