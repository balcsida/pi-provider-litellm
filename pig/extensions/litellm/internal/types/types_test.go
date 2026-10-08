package types

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/MichaelKinsy/PiG/ai"
)

func TestDiscoveredModelJSONKeepsTristateThinkingLevels(t *testing.T) {
	high := "high"
	m := DiscoveredModel{
		ID: "m", Name: "m", API: ai.API("openai-completions"), Input: []string{"text"},
		ThinkingLevelMap: ai.ThinkingLevelMap{"high": &high, "low": nil},
		LiteLLMPolicy:    &LiteLLMModelPolicy{},
	}
	b, err := json.Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"low":null`) || !strings.Contains(string(b), `"high":"high"`) {
		t.Fatalf("tristate lost: %s", b)
	}
	var back DiscoveredModel
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatal(err)
	}
	if v, ok := back.ThinkingLevelMap["low"]; !ok || v != nil {
		t.Fatalf("null denial lost: %+v", back.ThinkingLevelMap)
	}
	if _, ok := back.ThinkingLevelMap["off"]; ok {
		t.Fatal("absent level appeared")
	}
	if back.LiteLLMPolicy == nil {
		t.Fatal("policy lost")
	}
}

func TestModelInfoEntryDecodesSnakeCase(t *testing.T) {
	var r ModelInfoResponse
	in := `{"data":[{"model_name":"r","model_info":{"mode":null,"supports_reasoning":false,"max_input_tokens":8}}]}`
	if err := json.Unmarshal([]byte(in), &r); err != nil {
		t.Fatal(err)
	}
	mi := r.Data[0].ModelInfo
	if mi.Mode != nil || mi.SupportsReasoning == nil || *mi.SupportsReasoning || *mi.MaxInputTokens != 8 {
		t.Fatalf("%+v", mi)
	}
}
