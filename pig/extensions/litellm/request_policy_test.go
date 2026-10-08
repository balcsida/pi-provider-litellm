package litellm

// Ports tests/features.test.ts "feature parity" (request payload, reasoning effort, tool reasoning,
// Kimi/Moonshot and think-tag cases) and two tests/index.test.ts alias cases as unit tests of the Go
// functions. Cases that depend on discovery output are covered by the discover package tests; here the
// discovered policy is built by hand.

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

func obj(t *testing.T, text string) map[string]any {
	t.Helper()
	var value map[string]any
	if err := json.Unmarshal([]byte(text), &value); err != nil {
		t.Fatal(err)
	}
	return value
}

func sameJSON(t *testing.T, name string, got any, want string) {
	t.Helper()
	var expected any
	if err := json.Unmarshal([]byte(want), &expected); err != nil {
		t.Fatal(err)
	}
	raw, _ := json.Marshal(got)
	var actual any
	_ = json.Unmarshal(raw, &actual)
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("%s:\n got %s\nwant %s", name, raw, want)
	}
}

const toolsJSON = `[{"type":"function","function":{"name":"noop","parameters":{"type":"object"}}}]`

var kimiPolicy = &types.LiteLLMModelPolicy{NormalizeStrictToolMessages: true, NormalizeThinkTags: true, SuppressReasoningVisibility: true}

const visibilityJSON = `"include_reasoning":false,"reasoning_content":false,"merge_reasoning_content_in_choices":true`

func TestPrepareRequestPayload(t *testing.T) {
	prepare := func(payload string, model hookModel, policy *types.LiteLLMModelPolicy) map[string]any {
		return prepareRequestPayload(obj(t, payload), model, policy)
	}
	expectUnchanged := func(t *testing.T, name string, got map[string]any) {
		t.Helper()
		if got != nil {
			t.Fatalf("%s: want unchanged, got %v", name, got)
		}
	}

	t.Run("applies LiteLLM request compatibility hooks to configured provider aliases", func(t *testing.T) {
		// The hook is scoped to configured providers by extensionState.inScope; the alias's own policy applies.
		state := newExtensionState(&litellmSettings{}, []providerDefinition{{Name: "litellm"}, {Name: "litellm-anthropic"}}, newPolicyStore(""))
		if !state.providerNames["litellm-anthropic"] || state.providerNames["other"] {
			t.Fatal("provider scope wrong")
		}
		result := prepare(`{"model":"kimi-k2.6","messages":[{"role":"tool","tool_call_id":"call_1","content":[{"type":"text","text":"tool output"}]}]}`,
			hookModel{ID: "kimi-k2.6", API: "openai-completions"},
			&types.LiteLLMModelPolicy{NormalizeStrictToolMessages: true, NormalizeThinkTags: true})
		sameJSON(t, "strict", result["messages"], `[{"role":"tool","tool_call_id":"call_1","content":"tool output"}]`)
		sameJSON(t, "moonshot route", prepare(`{"messages":[]}`, hookModel{ID: "k3-prod", API: "openai-completions"}, kimiPolicy),
			`{"messages":[],`+visibilityJSON+`}`)
		expectUnchanged(t, "other route", prepare(`{"messages":[]}`, hookModel{ID: "k3-prod", API: "openai-completions"}, nil))
	})

	t.Run("does not infer reasoning controls from configured provider alias route text", func(t *testing.T) {
		expectUnchanged(t, "alias", prepare(`{"model":"kimi-k2.6"}`, hookModel{ID: "kimi-k2.6", API: "openai-completions", Family: ""}, nil))
		expectUnchanged(t, "kimi-k3", prepare(`{"messages":[]}`, hookModel{ID: "kimi-k3", API: "openai-completions"}, nil))
		expectUnchanged(t, "empty policy", prepare(`{"messages":[]}`, hookModel{ID: "k3-prod"}, &types.LiteLLMModelPolicy{}))
	})

	t.Run("keeps Moonshot suppression off routes that only look like Kimi", func(t *testing.T) {
		expectUnchanged(t, "no policy", prepare(`{"messages":[{"role":"tool","tool_call_id":"call_1","content":[{"type":"text","text":"tool output"}]}]}`,
			hookModel{ID: "kimi-k3", API: "openai-completions"}, nil))
	})

	t.Run("does not suppress reasoning for always-thinking backend models", func(t *testing.T) {
		policy := &types.LiteLLMModelPolicy{}
		expectUnchanged(t, "thinking", prepare(`{"messages":[]}`, hookModel{ID: "kimi-k2-thinking", API: "openai-completions", Family: types.FamilyKimi}, policy))
	})

	t.Run("does not send thinking for Kimi OpenAI completions requests", func(t *testing.T) {
		sameJSON(t, "kimi-k3", prepare(`{"messages":[]}`, hookModel{ID: "kimi-k3", API: "openai-completions"}, &types.LiteLLMModelPolicy{SuppressReasoningVisibility: true}),
			`{"messages":[],`+visibilityJSON+`}`)
	})

	t.Run("leaves payloads for an unregistered API unchanged", func(t *testing.T) {
		payload := `{"messages":[{"role":"user","content":"hi"}],"reasoning_effort":"HIGH","thinking":{"type":"enabled","budget_tokens":1024}}`
		expectUnchanged(t, "anthropic", prepare(payload, hookModel{ID: "kimi-k2.6", API: "anthropic-messages"}, kimiPolicy))
	})

	t.Run("leaves Kimi Responses requests unchanged", func(t *testing.T) {
		expectUnchanged(t, "responses", prepare(`{"input":[{"type":"message","role":"user","content":"hi"}]}`,
			hookModel{ID: "kimi-k2.6", API: "openai-responses"}, kimiPolicy))
	})

	t.Run("fills empty assistant content on Kimi tool calls", func(t *testing.T) {
		call := `[{"id":"call_1","type":"function","function":{"name":"noop","arguments":"{}"}}]`
		got := prepare(`{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":null,"tool_calls":`+call+`},{"role":"tool","tool_call_id":"call_1","content":"tool output"}]}`,
			hookModel{ID: "kimi-k3"}, kimiPolicy)
		sameJSON(t, "fill", got, `{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"","tool_calls":`+call+`},{"role":"tool","tool_call_id":"call_1","content":"tool output"}],`+visibilityJSON+`}`)
	})

	t.Run("preserves image blocks after Kimi tool results", func(t *testing.T) {
		got := prepare(`{"messages":[
			{"role":"tool","tool_call_id":"call_1","content":[{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]},
			{"role":"tool","tool_call_id":"call_2","content":[{"type":"text","text":"second output"}]},
			{"role":"tool","tool_call_id":"call_3","content":[]}]}`, hookModel{ID: "kimi-k3"}, kimiPolicy)
		sameJSON(t, "images", got, `{"messages":[
			{"role":"tool","tool_call_id":"call_1","content":"(see attached image)"},
			{"role":"tool","tool_call_id":"call_2","content":"second output"},
			{"role":"tool","tool_call_id":"call_3","content":"(no tool output)"},
			{"role":"user","content":[{"type":"text","text":"Attached image(s) from tool result:"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}],`+visibilityJSON+`}`)
	})

	t.Run("leaves well-formed Kimi tool messages untouched", func(t *testing.T) {
		messages := `[{"role":"assistant","content":"calling the tool","tool_calls":[{"id":"call_1","type":"function","function":{"name":"noop","arguments":"{}"}}]},{"role":"tool","tool_call_id":"call_1","content":"tool output"}]`
		sameJSON(t, "well formed", prepare(`{"messages":`+messages+`}`, hookModel{ID: "kimi-k3"}, kimiPolicy),
			`{"messages":`+messages+`,`+visibilityJSON+`}`)
	})

	t.Run("leaves strict-schema tool messages untouched without repair policy", func(t *testing.T) {
		payload := `{"messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"noop","arguments":"{}"}}]}]}`
		expectUnchanged(t, "moonshot-looking", prepare(payload, hookModel{ID: "kimi-k2.6"}, nil))
		expectUnchanged(t, "non-moonshot", prepare(payload, hookModel{ID: "anthropic/claude-3-5-sonnet"}, nil))
	})

	t.Run("drops reasoning fields for llm-gateway/gpt-5.5 tool requests", func(t *testing.T) {
		payload := `{"input":[{"type":"reasoning","id":"rs_1","encrypted_content":"opaque"},{"type":"message","role":"user","content":"hi"}],
			"tools":` + toolsJSON + `,"reasoning":{"effort":"high","summary":"auto"},"reasoning_effort":"high",
			"include":["reasoning.encrypted_content","other"],"include_reasoning":true,"reasoning_content":true,
			"merge_reasoning_content_in_choices":false,"thinking":{"type":"enabled"}}`
		got := prepare(payload, hookModel{ID: "llm-gateway/gpt-5.5"}, nil)
		sameJSON(t, "drops", got, `{"input":[{"type":"message","role":"user","content":"hi"}],"tools":`+toolsJSON+`,"include":["other"]}`)
	})

	t.Run("include is removed when only the encrypted reasoning entry remains", func(t *testing.T) {
		got := prepare(`{"tools":`+toolsJSON+`,"include":["reasoning.encrypted_content"]}`, hookModel{ID: "gpt-5.5"}, nil)
		sameJSON(t, "include", got, `{"tools":`+toolsJSON+`}`)
	})

	t.Run("leaves gpt-5.5 tool requests without reasoning fields unchanged", func(t *testing.T) {
		expectUnchanged(t, "no reasoning", prepare(`{"input":[{"type":"message","role":"user","content":"hi"}],"tools":`+toolsJSON+`,"include":["other"]}`,
			hookModel{ID: "llm-gateway/gpt-5.5"}, nil))
	})

	t.Run("keeps reasoning fields for gpt-5.5 Responses tool requests", func(t *testing.T) {
		expectUnchanged(t, "responses", prepare(`{"input":[{"type":"reasoning","id":"rs_1","encrypted_content":"opaque"}],"tools":`+toolsJSON+`,"reasoning":{"effort":"high","summary":"auto"},"reasoning_effort":"high"}`,
			hookModel{ID: "llm-gateway/gpt-5.5", API: "openai-responses"}, nil))
	})

	t.Run("drops reasoning fields for gpt-5.5 route aliases", func(t *testing.T) {
		for _, id := range []string{"gpt-5.5", "openai/gpt-5.5", "gpt-5.5-20260504143601"} {
			got := prepare(`{"messages":[],"tools":`+toolsJSON+`,"reasoning":{"effort":"high"}}`, hookModel{ID: id}, nil)
			sameJSON(t, id, got, `{"messages":[],"tools":`+toolsJSON+`}`)
		}
	})

	t.Run("turns reasoning off for tool requests to discovered GPT-5.5+ Chat backends", func(t *testing.T) {
		payload := `{"messages":[],"tools":` + toolsJSON + `,"reasoning_effort":"high"}`
		// GPT-6 backend: explicit none; GPT-5.6 backend: omitted. The route names carry no GPT hint.
		sol := prepare(payload, hookModel{ID: "team-sol", API: "openai-completions", Family: types.FamilyOpenAI},
			&types.LiteLLMModelPolicy{DropToolReasoning: true, ExplicitToolReasoningOff: true})
		sameJSON(t, "sol", sol, `{"messages":[],"tools":`+toolsJSON+`,"reasoning_effort":"none"}`)
		terra := prepare(payload, hookModel{ID: "team-terra", API: "openai-completions", Family: types.FamilyOpenAI},
			&types.LiteLLMModelPolicy{DropToolReasoning: true})
		sameJSON(t, "terra", terra, `{"messages":[],"tools":`+toolsJSON+`}`)
	})

	t.Run("keeps reasoning fields when deployment evidence names an older backend behind a GPT-6 route name", func(t *testing.T) {
		expectUnchanged(t, "gpt-6-sol", prepare(`{"messages":[],"tools":`+toolsJSON+`,"reasoning_effort":"high"}`,
			hookModel{ID: "gpt-6-sol", API: "openai-completions", Family: types.FamilyOpenAI}, &types.LiteLLMModelPolicy{}))
	})

	t.Run("turns reasoning off explicitly for tool requests when the model declares an off effort", func(t *testing.T) {
		off := "none"
		got := prepare(`{"messages":[],"tools":`+toolsJSON+`,"reasoning_effort":"high"}`,
			hookModel{ID: "team-luna", API: "openai-completions", Family: types.FamilyOpenAI, OffEffort: &off},
			&types.LiteLLMModelPolicy{DropToolReasoning: true})
		sameJSON(t, "off", got, `{"messages":[],"tools":`+toolsJSON+`,"reasoning_effort":"none"}`)
		minimal := "minimal"
		got = prepare(`{"messages":[],"tools":`+toolsJSON+`}`,
			hookModel{ID: "team-luna", Family: types.FamilyOpenAI, OffEffort: &minimal}, &types.LiteLLMModelPolicy{DropToolReasoning: true, ExplicitToolReasoningOff: true})
		sameJSON(t, "declared off wins over none", got, `{"messages":[],"tools":`+toolsJSON+`,"reasoning_effort":"minimal"}`)
	})

	t.Run("drops reasoning fields for evidence-free GPT-5.5+ route names only", func(t *testing.T) {
		payload := `{"messages":[],"tools":` + toolsJSON + `,"reasoning_effort":"high"}`
		for _, id := range []string{"gpt-6-luna", "llm-gateway/gpt-6-sol"} {
			sameJSON(t, id, prepare(payload, hookModel{ID: id}, nil), `{"messages":[],"tools":`+toolsJSON+`,"reasoning_effort":"none"}`)
		}
		for _, id := range []string{"gpt-5.6-luna-eu-west", "gpt-5.10"} {
			sameJSON(t, id, prepare(payload, hookModel{ID: id}, nil), `{"messages":[],"tools":`+toolsJSON+`}`)
		}
		for _, id := range []string{"gpt-5", "gpt-5.1-codex", "gpt-4.1", "gpt-4o", "gpt-oss-120b", "my-gpt-6-sol"} {
			expectUnchanged(t, id, prepare(payload, hookModel{ID: id}, nil))
		}
	})

	t.Run("normalizes capitalized reasoning effort values for Gemini models only", func(t *testing.T) {
		payload := `{"messages":[{"role":"user","content":"hello"}],"reasoning_effort":"HIGH","reasoning":{"effort":"HIGH","summary":"auto"}}`
		got := prepare(payload, hookModel{ID: "opaque-gemini-route"}, &types.LiteLLMModelPolicy{NormalizeGeminiReasoningEffort: true})
		sameJSON(t, "gemini", got, `{"messages":[{"role":"user","content":"hello"}],"reasoning_effort":"high","reasoning":{"effort":"high","summary":"auto"}}`)
		expectUnchanged(t, "non-gemini", prepare(`{"messages":[],"reasoning_effort":"MAX_THINKING","reasoning":{"effort":"MAX_THINKING"}}`,
			hookModel{ID: "google/gemini-3.1-pro-preview"}, nil))
	})

	t.Run("does not mutate the input payload", func(t *testing.T) {
		payload := obj(t, `{"messages":[],"reasoning":{"effort":"HIGH"},"reasoning_effort":"HIGH"}`)
		prepareRequestPayload(payload, hookModel{ID: "g"}, &types.LiteLLMModelPolicy{NormalizeGeminiReasoningEffort: true})
		sameJSON(t, "input", payload, `{"messages":[],"reasoning":{"effort":"HIGH"},"reasoning_effort":"HIGH"}`)
	})
}

func TestNormalizeThinkTags(t *testing.T) {
	names := map[string]bool{"litellm": true}
	message := func(text string, extra ...string) map[string]any {
		blocks := []any{map[string]any{"type": "text", "text": text}}
		for _, kind := range extra {
			blocks = append(blocks, map[string]any{"type": kind})
		}
		return map[string]any{"role": "assistant", "provider": "litellm", "model": "k", "content": blocks}
	}
	policy := &types.LiteLLMModelPolicy{NormalizeThinkTags: true, SuppressReasoningVisibility: true}

	t.Run("normalizes route-only Kimi think tags from the discovered policy", func(t *testing.T) {
		got := normalizeThinkTags(message("<think>internal reasoning</think>DONE"), names, "openai-completions", policy)
		sameJSON(t, "content", got["content"], `[{"type":"thinking","thinking":"internal reasoning"},{"type":"text","text":"DONE"}]`)
		if got["role"] != "assistant" || got["model"] != "k" {
			t.Fatalf("message fields lost: %v", got)
		}
		// An unspecified api (model not in the registry) is treated like Chat Completions.
		if normalizeThinkTags(message("<think>x</think>y"), names, "", policy) == nil {
			t.Fatal("empty api should normalize")
		}
	})
	t.Run("leaves think tags alone without a normalizing conclusion", func(t *testing.T) {
		for name, p := range map[string]*types.LiteLLMModelPolicy{
			"no conclusion": nil, "declined": {NormalizeThinkTags: false},
		} {
			if got := normalizeThinkTags(message("<think>internal reasoning</think>DONE"), names, "openai-completions", p); got != nil {
				t.Fatalf("%s: rewrote %v", name, got)
			}
		}
	})
	t.Run("does not normalize opaque native Messages responses from request-side suppression metadata", func(t *testing.T) {
		if got := normalizeThinkTags(message("<think>x</think>DONE"), names, "anthropic-messages", kimiPolicy); got != nil {
			t.Fatalf("rewrote %v", got)
		}
		if got := normalizeThinkTags(message("<think>x</think>DONE"), names, "openai-responses", kimiPolicy); got != nil {
			t.Fatalf("rewrote responses %v", got)
		}
	})
	t.Run("ignores providers outside the configured set", func(t *testing.T) {
		other := message("<think>x</think>DONE")
		other["provider"] = "anthropic"
		if normalizeThinkTags(other, names, "", policy) != nil {
			t.Fatal("other provider rewritten")
		}
	})
	t.Run("keeps final Kimi text visible when a dangling think tag prefixes it", func(t *testing.T) {
		got := normalizeThinkTags(message("<think>DONE"), names, "openai-completions", kimiPolicy)
		sameJSON(t, "dangling", got["content"], `[{"type":"text","text":"DONE"}]`)
	})
	t.Run("a dangling think tag before non-text content is reasoning", func(t *testing.T) {
		got := normalizeThinkTags(message("hi <think>plan", "toolCall"), names, "openai-completions", kimiPolicy)
		sameJSON(t, "dangling before tool", got["content"], `[{"type":"text","text":"hi "},{"type":"thinking","thinking":"plan"},{"type":"toolCall"}]`)
	})
	t.Run("text without tags is not rewritten", func(t *testing.T) {
		if normalizeThinkTags(message("plain"), names, "openai-completions", kimiPolicy) != nil {
			t.Fatal("rewrote plain text")
		}
	})
}
