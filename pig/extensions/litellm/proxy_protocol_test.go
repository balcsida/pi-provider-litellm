package litellm

// Ports tests/provider-compat/proxy-protocol.test.ts, the managed-effort cases of
// tests/provider-compat/messages.test.ts and the Azure Responses case of tests/provider.test.ts.

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/discover"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

type proxyRequest struct {
	path   string
	header http.Header
	body   map[string]any
}

// recordingProxy answers /model/info with its rows and each protocol endpoint with a short streamed reply,
// recording every protocol request.
type recordingProxy struct {
	*httptest.Server
	mu       sync.Mutex
	requests []proxyRequest
}

var (
	responsesReply = []string{
		`{"type":"response.created","response":{"id":"resp_1","status":"in_progress","output":[]}}`,
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"in_progress","content":[]}}`,
		`{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"hello"}`,
		`{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"hello","annotations":[]}]}}`,
		`{"type":"response.completed","response":{"id":"resp_1","status":"completed","output":[],"usage":{"input_tokens":1,"output_tokens":1,"input_tokens_details":{"cached_tokens":0}}}}`,
	}
	messagesReply = []string{
		`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"m","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}`,
		`{"type":"message_stop"}`,
	}
	// Managed effort sends thinking.block_binding and mid-conversation effort markers under these betas.
	bindingBetas = []string{"mid-conversation-output-config-2026-07-01", "thinking-binding-controls-2026-08-01"}
)

func newRecordingProxy(t *testing.T, rows string) *recordingProxy {
	t.Helper()
	proxy := &recordingProxy{}
	proxy.Server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/model/info" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"data":[` + rows + `]}`))
			return
		}
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		proxy.mu.Lock()
		proxy.requests = append(proxy.requests, proxyRequest{r.URL.Path, r.Header.Clone(), body})
		proxy.mu.Unlock()
		reply := responsesReply
		if r.URL.Path == "/v1/messages" {
			reply = messagesReply
		}
		w.Header().Set("Content-Type", "text/event-stream")
		for _, data := range reply {
			var event struct{ Type string }
			json.Unmarshal([]byte(data), &event)
			fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event.Type, data)
		}
	}))
	t.Cleanup(proxy.Close)
	return proxy
}

func (p *recordingProxy) recorded() []proxyRequest {
	p.mu.Lock()
	defer p.mu.Unlock()
	return slices.Clone(p.requests)
}

func (p *recordingProxy) harness(t *testing.T) providerHarness {
	t.Helper()
	return newHarness(t, func(o *providerOptions) {
		o.ResolveCredentialRoot = func(*ai.Credential, string, string) (string, error) { return p.URL, nil }
	})
}

// streamResult runs one request through the provider's guarded stream and returns the final message.
func streamResult(t *testing.T, h providerHarness, model *ai.Model, options ai.StreamOptions) *ai.AssistantMessage {
	t.Helper()
	stream, err := h.provider.StreamSimple(context.Background(), model, ai.NormalizeContext(ai.Context{
		Messages: []ai.Message{ai.UserMessage{Content: ai.UserText("hello")}},
	}), options)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	message, err := stream.ResultContext(ctx)
	if err != nil {
		t.Fatal(err)
	}
	return message
}

func messageText(message *ai.AssistantMessage) string {
	var text strings.Builder
	for _, block := range message.Content {
		switch content := block.(type) {
		case ai.TextContent:
			text.WriteString(content.Text)
		case *ai.TextContent:
			text.WriteString(content.Text)
		}
	}
	return text.String()
}

func responsesModel(t *testing.T, root string) types.DiscoveredModel {
	t.Helper()
	model := discoveredModel("team-responses")
	model.API = ai.APIOpenAIResponses
	native, err := toNativeModels("litellm", root, []types.DiscoveredModel{model}, false)
	if err != nil {
		t.Fatal(err)
	}
	return native[0]
}

func TestAzureResponsesAPIOnALiteLLMModel(t *testing.T) {
	t.Run("dispatches through the proxy's Responses endpoint with the proxy credential", func(t *testing.T) {
		proxy := newRecordingProxy(t, "")
		model := firstModel(t, "litellm", responsesModel(t, proxy.URL))
		model.ProviderMeta.API = ai.APIAzureOpenAIResponses

		message := streamResult(t, proxy.harness(t), model, ai.StreamOptions{APIKey: "sk-test"})

		if message.API != ai.APIOpenAIResponses || message.StopReason != ai.StopReasonStop || messageText(message) != "hello" {
			t.Fatalf("message api = %s stop = %s text = %q error = %q", message.API, message.StopReason, messageText(message), message.ErrorMessage)
		}
		requests := proxy.recorded()
		if len(requests) != 1 || requests[0].path != "/v1/responses" ||
			requests[0].header.Get("Authorization") != "Bearer sk-test" || requests[0].header.Get("api-key") != "" {
			t.Fatalf("requests = %+v", requests)
		}
	})

	t.Run("still rejects a protocol the proxy does not serve before any request", func(t *testing.T) {
		proxy := newRecordingProxy(t, "")
		model := firstModel(t, "litellm", responsesModel(t, proxy.URL))
		model.ProviderMeta.API = ai.APIGoogleGenerativeAI

		_, err := proxy.harness(t).provider.StreamSimple(context.Background(), model, chatContext(0), ai.StreamOptions{APIKey: "sk-test"})

		if err == nil || !strings.Contains(err.Error(), `declares unsupported protocol "google-generative-ai"`) || len(proxy.recorded()) != 0 {
			t.Fatalf("err = %v requests = %d", err, len(proxy.recorded()))
		}
	})

	t.Run("projects the model onto the proxy's Responses protocol for selection", func(t *testing.T) {
		h := newHarness(t, nil)
		model := firstModel(t, "litellm", nativeModel(t, "team-responses"))
		model.ProviderMeta.API = ai.APIAzureOpenAIResponses

		kept := h.provider.FilterModels([]*ai.Model{model}, nil)

		if len(kept) != 1 || kept[0].ProviderMeta.API != ai.APIOpenAIResponses || kept[0].ProviderMeta.BaseURL != testRoot+"/v1" {
			t.Fatalf("kept = %+v", kept)
		}
	})

	t.Run("restores a cached entry onto the proxy's Responses protocol offline", func(t *testing.T) {
		h := newHarness(t, nil)
		online := responsesModel(t, testRoot)
		h.policies.replace("litellm", []types.DiscoveredModel{online})
		cached := online
		cached.API = ai.APIAzureOpenAIResponses

		if _, err := refresh(h.provider, storedEntry(t, cached), false, false); err != nil {
			t.Fatal(err)
		}

		if got, want := modelByID(t, h.provider, online.ID), firstModel(t, "litellm", online); !reflect.DeepEqual(got, want) {
			t.Fatalf("restored %+v\nwant %+v", got.ProviderMeta, want.ProviderMeta)
		}
	})
}

func TestManagedEffortFollowsClaudeRouting(t *testing.T) {
	// LiteLLM reports the cost-map provider of base_model, so the hosted deployment still reads as `anthropic`.
	direct := `{"model_name":"team-claude","litellm_params":{"model":"anthropic/claude-opus-5-5","custom_llm_provider":"anthropic"},` +
		`"model_info":{"mode":"chat","litellm_provider":"anthropic","supports_reasoning":true}}`
	hosted := `{"model_name":"team-claude","litellm_params":{"model":"azure_ai/claude-opus-5-5","custom_llm_provider":"azure_ai"},` +
		`"model_info":{"mode":"chat","litellm_provider":"anthropic","base_model":"claude-opus-5-5","supports_reasoning":true}}`
	for _, tc := range []struct {
		name string
		row  string
		// optIn simulates a models.json override that enables managed effort.
		optIn, wantDenied, wantBinding bool
	}{
		{"sends adaptive thinking without binding controls to a Claude route hosted outside Anthropic", hosted, false, true, false},
		{"leaves a route only through Anthropic unchanged", direct, false, false, false},
		{"keeps binding betas consistent with the body where a route only through Anthropic opts in", direct, true, false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			proxy := newRecordingProxy(t, tc.row)
			disabled := false
			result, err := discover.DiscoverModels(context.Background(), proxy.URL, "sk-test", discover.Options{
				DiscoveryOptions: types.DiscoveryOptions{ModelsDev: &disabled}, Silent: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			native, err := toNativeModels("litellm", proxy.URL, result.Models, false)
			if err != nil || len(native) != 1 {
				t.Fatalf("native = %+v err = %v", native, err)
			}
			model := firstModel(t, "litellm", native[0])
			compat := model.ProviderMeta.Compat
			if model.ProviderMeta.API != ai.APIAnthropicMessages || compat == nil {
				t.Fatalf("api = %s compat = %+v", model.ProviderMeta.API, compat)
			}
			// Hosted routes declare the denial; a route only through Anthropic leaves the field unset.
			if tc.wantDenied != (compat.SupportsMidConvoEffort != nil) || tc.wantDenied && *compat.SupportsMidConvoEffort {
				t.Fatalf("supportsMidConvoEffort = %v, want denied %v", compat.SupportsMidConvoEffort, tc.wantDenied)
			}
			if tc.optIn {
				optedIn := *compat
				optedIn.SupportsMidConvoEffort = yes()
				model.ProviderMeta.Compat = &optedIn
			}

			message := streamResult(t, proxy.harness(t), model, ai.StreamOptions{APIKey: "sk-test", Thinking: ai.ThinkingLevelHigh})

			if message.API != ai.APIAnthropicMessages || message.StopReason != ai.StopReasonStop || messageText(message) != "hello" {
				t.Fatalf("message api = %s stop = %s text = %q error = %q", message.API, message.StopReason, messageText(message), message.ErrorMessage)
			}
			requests := proxy.recorded()
			if len(requests) != 1 || requests[0].path != "/v1/messages" {
				t.Fatalf("requests = %+v", requests)
			}
			thinking, _ := requests[0].body["thinking"].(map[string]any)
			outputConfig, _ := requests[0].body["output_config"].(map[string]any)
			_, binding := thinking["block_binding"]
			if thinking["type"] != "adaptive" || outputConfig["effort"] != "high" || binding != tc.wantBinding {
				t.Fatalf("thinking = %v output_config = %v", thinking, outputConfig)
			}
			for _, beta := range bindingBetas {
				if got := strings.Contains(requests[0].header.Get("anthropic-beta"), beta); got != tc.wantBinding {
					t.Errorf("anthropic-beta %q includes %s = %v, want %v", requests[0].header.Get("anthropic-beta"), beta, got, tc.wantBinding)
				}
			}
		})
	}
}
