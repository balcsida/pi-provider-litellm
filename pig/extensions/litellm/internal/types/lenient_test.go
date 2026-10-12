package types

import (
	"encoding/json"
	"testing"
)

func TestLenientModelInfoDecoding(t *testing.T) {
	var response ModelInfoResponse
	doc := `{"data":[
		{"model_name":42,"litellm_params":{"model":7,"custom_llm_provider":"x"},
		 "model_info":{"id":"keep","mode":7,"supports_vision":"no","supports_reasoning":true,"max_input_tokens":"big",
		   "supported_endpoints":42,"supported_openai_params":{"a":1},"reasoning_effort_levels":"high"}},
		{"model_name":"ok","model_info":{"supported_endpoints":["/v1/messages",42,null],"supported_openai_params":[],"reasoning_effort_levels":null}},
		42]}`
	if err := json.Unmarshal([]byte(doc), &response); err != nil {
		t.Fatalf("one mistyped field must not fail the decode: %v", err)
	}
	bad, ok := response.Data[0], response.Data[1]
	if bad.ModelName != "" || bad.LiteLLMParams.Model != "" || bad.LiteLLMParams.CustomLLMProvider != "x" {
		t.Fatalf("params %+v %+v", bad, bad.LiteLLMParams)
	}
	info := bad.ModelInfo
	if info.ID != "keep" || info.Mode != nil || info.SupportsVision != nil || info.SupportsReasoning == nil || !*info.SupportsReasoning || info.MaxInputTokens != nil {
		t.Fatalf("info %+v", info)
	}
	if info.SupportedEndpoints == nil || len(info.SupportedEndpoints) != 0 || info.SupportedOpenAIParams == nil || info.ReasoningEffortLevels != nil {
		t.Fatalf("lists %+v", info)
	}
	if got := ok.ModelInfo.SupportedEndpoints; len(got) != 1 || got[0] != "/v1/messages" {
		t.Fatalf("endpoints %v", got)
	}
	if ok.ModelInfo.SupportedOpenAIParams == nil || ok.ModelInfo.ReasoningEffortLevels != nil {
		t.Fatal("empty list must stay non-nil and null must stay nil")
	}
	if response.Data[2].ModelName != "" {
		t.Fatalf("non-object entry %+v", response.Data[2])
	}
}

func TestSecretsAreNotMarshalled(t *testing.T) {
	for _, v := range []any{LiteLLMRuntimeAuth{APIKey: "s3cret"}, ResolvedCredentials{APIKey: "s3cret"}} {
		encoded, _ := json.Marshal(v)
		if string(encoded) != "{\"baseUrl\":\"\"}" && string(encoded) != "{}" {
			t.Fatalf("%s", encoded)
		}
	}
}
