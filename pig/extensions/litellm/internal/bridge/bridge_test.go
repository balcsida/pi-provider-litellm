package bridge

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/MichaelKinsy/PiG/extensions/sdk"
)

const providerID = "mock"

func chatModel(id string, api ai.API, baseURL string) *ai.Model {
	return &ai.Model{
		ID: id, DisplayName: id, Input: []string{"text"},
		ProviderMeta: ai.ProviderMetadata{ProviderID: providerID, API: api, BaseURL: baseURL},
		Capabilities: ai.ModelCapabilities{ContextWindow: 8000, MaxOutputTokens: 1000},
	}
}

func newProvider(baseURL string, fetch func(ai.RefreshModelsContext) ([]ai.AnyModel, error), filter func([]*ai.Model, *ai.Credential) []*ai.Model) *ai.ModelsProvider {
	return ai.CreateProvider(ai.CreateProviderOptions{
		ID: providerID,
		Models: []ai.AnyModel{
			chatModel("claude", ai.APIAnthropicMessages, baseURL),
			chatModel("chat", ai.APIOpenAICompletions, baseURL),
			chatModel("resp", ai.APIOpenAIResponses, baseURL),
		},
		API:          &ai.ProviderStreams{Stream: ai.StreamSimple, StreamSimple: ai.StreamSimple},
		FetchModels:  fetch,
		FilterModels: filter,
		Auth: ai.ProviderAuth{APIKey: &ai.APIKeyAuth{Name: "key",
			Resolve: func(_ context.Context, in ai.APIKeyAuthInput) (*ai.AuthResult, error) {
				if v, ok := in.Ctx.Env("MOCK_KEY"); ok {
					return &ai.AuthResult{Auth: ai.ModelAuth{APIKey: v}, Source: "MOCK_KEY"}, nil
				}
				return nil, nil
			}}},
	})
}

func drain(t *testing.T, s *sdk.ModelEventStream) []map[string]any {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	var events []map[string]any
	for e := range s.Events(ctx) {
		events = append(events, e)
	}
	if ctx.Err() != nil {
		t.Fatal("stream did not finish")
	}
	return events
}

func userTranscript(text string) map[string]any {
	return map[string]any{"messages": []any{map[string]any{"role": "user", "content": text, "timestamp": 1}}}
}

func sse(w http.ResponseWriter, chunks ...string) {
	w.Header().Set("Content-Type", "text/event-stream")
	for _, c := range chunks {
		fmt.Fprintf(w, "data: %s\n\n", c)
	}
	w.(http.Flusher).Flush()
}

func chunk(delta, finish string) string {
	f := "null"
	if finish != "" {
		f = fmt.Sprintf("%q", finish)
	}
	return fmt.Sprintf(`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"chat","choices":[{"index":0,"delta":%s,"finish_reason":%s}]}`, delta, f)
}

func TestStreamSimpleChatCompletions(t *testing.T) {
	var mu sync.Mutex
	var auth, path string
	var body map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		auth, path = r.Header.Get("Authorization"), r.URL.Path
		_ = json.NewDecoder(r.Body).Decode(&body)
		mu.Unlock()
		sse(w,
			chunk(`{"role":"assistant","content":"Hel"}`, ""),
			chunk(`{"content":"lo"}`, ""),
			chunk(`{}`, "stop"),
			`{"id":"c1","object":"chat.completion.chunk","created":1,"model":"chat","choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`,
			"[DONE]")
	}))
	defer server.Close()
	p := Wrap(newProvider(server.URL, nil, nil))

	models, err := p.GetModels()
	if err != nil || len(models) != 3 {
		t.Fatalf("GetModels = %v, %v", models, err)
	}
	var chat map[string]any
	for _, m := range models {
		if m["id"] == "chat" {
			chat = m
		}
	}
	stream, err := p.StreamSimple(chat, userTranscript("hi"), sdk.ProviderStreamOptions{Values: map[string]any{"apiKey": "sk-test", "maxRetries": 0}})
	if err != nil {
		t.Fatal(err)
	}
	var text strings.Builder
	events := drain(t, stream)
	for _, e := range events {
		if e["type"] == "text_delta" {
			text.WriteString(e["delta"].(string))
		}
	}
	if text.String() != "Hello" {
		t.Fatalf("text deltas = %q, events %v", text.String(), events)
	}
	if last := events[len(events)-1]; last["type"] != "done" {
		t.Fatalf("last event = %v", last)
	}
	result := stream.Result()
	usage, _ := result["usage"].(map[string]any)
	content, _ := result["content"].([]any)
	if result["role"] != "assistant" || result["stopReason"] != "stop" || usage["input"] != 3.0 || usage["output"] != 2.0 || len(content) != 1 {
		t.Fatalf("result = %v", result)
	}
	if first, _ := content[0].(map[string]any); first["text"] != "Hello" {
		t.Fatalf("content = %v", content)
	}
	mu.Lock()
	defer mu.Unlock()
	if auth != "Bearer sk-test" || !strings.HasSuffix(path, "/chat/completions") || body["model"] != "chat" {
		t.Fatalf("mock saw auth=%q path=%q body=%v", auth, path, body)
	}
}

func TestStreamSimpleCancellationClosesConnection(t *testing.T) {
	closed := make(chan struct{})
	started := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sse(w, chunk(`{"role":"assistant","content":"Hel"}`, ""))
		close(started)
		<-r.Context().Done()
		close(closed)
	}))
	defer server.Close()
	p := Wrap(newProvider(server.URL, nil, nil))
	models, _ := p.GetModels()
	var chat map[string]any
	for _, m := range models {
		if m["id"] == "chat" {
			chat = m
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	stream, err := p.StreamSimple(chat, userTranscript("hi"), sdk.ProviderStreamOptions{Signal: ctx, Values: map[string]any{"apiKey": "k", "maxRetries": 0}})
	if err != nil {
		t.Fatal(err)
	}
	<-started
	cancel()
	events := drain(t, stream)
	last := events[len(events)-1]
	if last["type"] != "error" || last["reason"] != "aborted" {
		t.Fatalf("last event = %v", last)
	}
	if stream.Result()["stopReason"] != "aborted" {
		t.Fatalf("result = %v", stream.Result())
	}
	select {
	case <-closed:
	case <-time.After(5 * time.Second):
		t.Fatal("mock never saw the connection close")
	}
}

func TestEveryAPIIsRouted(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer server.Close()
	p := Wrap(newProvider(server.URL, nil, nil))
	models, _ := p.GetModels()
	apis := map[any]bool{}
	for _, m := range models {
		apis[m["api"]] = true
		stream, err := p.Stream(m, userTranscript("hi"), sdk.ProviderStreamOptions{Values: map[string]any{"apiKey": "k", "maxRetries": 0}})
		if err != nil {
			t.Fatalf("%v: %v", m["id"], err)
		}
		drain(t, stream)
		result := stream.Result()
		msg, _ := result["errorMessage"].(string)
		if result["stopReason"] != "error" || strings.Contains(msg, "no API implementation") {
			t.Fatalf("%v: result = %v", m["id"], result)
		}
	}
	if len(apis) != 3 {
		t.Fatalf("apis = %v", apis)
	}
}

// TestFilterModelsDropsUnknownFields pins what ai.DecodeModelsCatalog keeps: only the fields of the
// catalog record survive decode+encode; extension fields and ModelInfo extras are dropped, and a
// record of another provider is rejected.
func TestFilterModelsDropsUnknownFields(t *testing.T) {
	var seen []*ai.Model
	raw := newProvider("http://x", nil, func(models []*ai.Model, _ *ai.Credential) []*ai.Model {
		seen = models
		return models[:2]
	})
	p := Wrap(raw)
	models, _ := p.GetModels()
	for _, m := range models {
		m["litellmPolicy"] = map[string]any{"x": 1}
		m["displayName"] = "extra"
		m["modelId"] = m["id"]
	}

	// Decode + encode alone loses every unknown field.
	decoded, err := decodeModel(models[0], providerID)
	if err != nil {
		t.Fatal(err)
	}
	reencoded, _ := encodeModel(decoded)
	for _, key := range []string{"litellmPolicy", "displayName", "modelId"} {
		if _, ok := reencoded[key]; ok {
			t.Fatalf("%s survived decode+encode", key)
		}
	}

	// The filtered models are the caller's own maps, so the extras stay on the kept ones.
	filtered, err := p.FilterModels(models, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(filtered) != 2 || len(seen) != 3 {
		t.Fatalf("filtered %d of %d", len(filtered), len(seen))
	}
	for i, m := range filtered {
		if fmt.Sprintf("%p", m) != fmt.Sprintf("%p", models[i]) || m["litellmPolicy"] == nil {
			t.Fatalf("filtered[%d] is not the input map or lost its extras: %v", i, m)
		}
	}

	// A record of another provider is dropped by ai.DecodeModelsCatalog and fails the call.
	models[0]["provider"] = "other"
	if _, err := p.FilterModels(models, nil); err == nil {
		t.Fatal("expected an error for a foreign-provider record")
	}
}

func TestRefreshModelsPassThroughAndPublish(t *testing.T) {
	var got ai.RefreshModelsContext
	fetch := func(refresh ai.RefreshModelsContext) ([]ai.AnyModel, error) {
		got = refresh
		return []ai.AnyModel{chatModel("fetched", ai.APIOpenAICompletions, "http://x")}, nil
	}
	raw := newProvider("http://x", fetch, nil)
	p := Wrap(raw)

	stored := map[string]any{"models": []any{map[string]any{"id": "restored", "name": "restored", "api": "openai-completions", "provider": providerID,
		"baseUrl": "http://x", "reasoning": false, "input": []any{"text"}, "cost": map[string]any{}, "contextWindow": 1, "maxTokens": 1}}, "etag": "e1"}
	var publications []sdk.ModelsPublication
	force := true
	err := p.RefreshModels(sdk.RefreshModelsContext{
		Credential: map[string]any{"type": "api_key", "key": "k"}, Stored: stored, AllowNetwork: true, Force: &force, Signal: context.Background(),
		Publish: func(pub sdk.ModelsPublication) (bool, error) {
			publications = append(publications, pub)
			if pub.Update != nil {
				return true, pub.Update()
			}
			return true, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.Stored == nil || got.Stored.ETag != "e1" || !got.AllowNetwork || got.Force == nil || !*got.Force || got.Credential == nil || got.Credential.Key != "k" || got.Signal == nil || got.Publish == nil {
		t.Fatalf("fetch saw %+v", got)
	}
	if len(publications) != 2 || publications[0].Persist != nil || publications[0].Update == nil {
		t.Fatalf("publications = %+v", publications)
	}
	var persisted ai.ModelsStoreEntry
	if err := json.Unmarshal(publications[1].Persist, &persisted); err != nil || len(persisted.Models) != 1 || !strings.Contains(string(persisted.Models[0]), `"fetched"`) || persisted.CheckedAt == nil {
		t.Fatalf("persist = %s (%v)", publications[1].Persist, err)
	}
	models, _ := p.GetModels()
	ids := []any{}
	for _, m := range models {
		ids = append(ids, m["id"])
	}
	if len(models) != 4 || !contains(ids, "fetched") {
		t.Fatalf("models after refresh = %v", ids)
	}

	// Without network access only the stored catalog is published.
	publications = nil
	got = ai.RefreshModelsContext{}
	err = p.RefreshModels(sdk.RefreshModelsContext{Stored: stored, Signal: context.Background(),
		Publish: func(pub sdk.ModelsPublication) (bool, error) {
			publications = append(publications, pub)
			return true, nil
		}})
	if err != nil || got.Signal != nil || len(publications) != 1 {
		t.Fatalf("offline refresh: err=%v fetched=%v publications=%d", err, got.Signal != nil, len(publications))
	}
}

func contains(list []any, want string) bool {
	for _, v := range list {
		if v == want {
			return true
		}
	}
	return false
}

func TestAuthResolve(t *testing.T) {
	p := Wrap(newProvider("http://x", nil, nil))
	env := "sk-env"
	result, err := p.Auth.APIKey.Resolve(sdk.APIKeyAuthInput{Ctx: sdk.AuthContext{Env: func(name string) (*string, error) {
		if name == "MOCK_KEY" {
			return &env, nil
		}
		return nil, nil
	}}})
	if err != nil || result == nil || result.Auth["apiKey"] != "sk-env" || *result.Source != "MOCK_KEY" {
		t.Fatalf("resolve = %+v, %v", result, err)
	}
}
