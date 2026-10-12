package bridge

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/MichaelKinsy/PiG/extensions/sdk"
)

// The host's filterModels records are extension.ProviderModelConfig: no provider field.
func TestFilterModelsWithoutProviderField(t *testing.T) {
	p := Wrap(newProvider("http://x", nil, func(models []*ai.Model, _ *ai.Credential) []*ai.Model { return models[:1] }))
	models, _ := p.GetModels()
	for _, m := range models {
		delete(m, "provider")
		m["extra"] = true
	}
	filtered, err := p.FilterModels(models, nil)
	if err != nil || len(filtered) != 1 {
		t.Fatalf("filtered = %v, %v", filtered, err)
	}
	if _, ok := filtered[0]["provider"]; ok || filtered[0]["extra"] != true {
		t.Fatalf("kept map = %v", filtered[0])
	}
}

func TestFilterModelsDeletesStaleKeysKeepsExtras(t *testing.T) {
	p := Wrap(newProvider("http://x", nil, func(models []*ai.Model, _ *ai.Credential) []*ai.Model {
		models[0].ProviderMeta.Headers = nil // the filter drops the headers, so the re-encoding omits the key
		return models
	}))
	models, _ := p.GetModels()
	models[0]["headers"] = map[string]any{"X-Old": "1"}
	models[0]["litellmPolicy"] = map[string]any{"x": 1}
	filtered, err := p.FilterModels(models, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := filtered[0]["headers"]; ok || filtered[0]["litellmPolicy"] == nil {
		t.Fatalf("kept map = %v", filtered[0])
	}
}

func TestStreamAbortKeepsDriverPartialContent(t *testing.T) {
	started := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		sse(w, chunk(`{"role":"assistant","content":"Hel"}`, ""))
		close(started)
		<-r.Context().Done()
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
	// Let the partial text reach the driver's message before aborting.
	cancelAfterText(stream, cancel)
	result := stream.Result()
	content, _ := result["content"].([]any)
	if result["stopReason"] != "aborted" || len(content) == 0 {
		t.Fatalf("aborted result lost partial content: %v", result)
	}
}

// cancelAfterText cancels once a text_delta arrives and returns when the stream ends.
func cancelAfterText(s *sdk.ModelEventStream, cancel context.CancelFunc) {
	for e := range s.Events(context.Background()) {
		if e["type"] == "text_delta" {
			cancel()
		}
	}
}

func TestDecodeOptionsNullTemperature(t *testing.T) {
	for _, tc := range []struct {
		name   string
		values map[string]any
		set    bool
	}{
		{"absent", map[string]any{"maxTokens": 5}, false},
		{"null", map[string]any{"temperature": nil}, false},
		{"zero", map[string]any{"temperature": 0.0}, true},
		{"value", map[string]any{"temperature": 0.7}, true},
	} {
		options, err := decodeOptions(tc.values)
		if err != nil || options.TemperatureSet != tc.set {
			t.Fatalf("%s: TemperatureSet = %v, %v", tc.name, options.TemperatureSet, err)
		}
	}
	options, err := decodeOptions(map[string]any{"apiKey": "k", "maxTokens": 7, "temperature": 0.5, "onPayload": "ignored", "unknown": 1})
	if err != nil || options.APIKey != "k" || !options.TemperatureSet || options.Temperature != 0.5 {
		t.Fatalf("options = %+v, %v", options, err)
	}
}

func TestDecodeTranscript(t *testing.T) {
	tool := map[string]any{"name": "read", "description": "read a file", "parameters": map[string]any{"type": "object"}}
	user := map[string]any{"role": "user", "content": "hi", "timestamp": 1}

	hostForm := map[string]any{"messages": []any{
		map[string]any{"role": "system", "content": "be brief", "toolsAdded": []any{tool}},
		user,
	}}
	piForm := map[string]any{"systemPrompt": "be brief", "tools": []any{tool}, "messages": []any{user}}
	for name, in := range map[string]map[string]any{"host": hostForm, "pi": piForm} {
		got, err := decodeTranscript(in)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		messages := got.Messages()
		system, ok := messages[0].(ai.SystemMessage)
		if len(messages) != 2 || !ok || len(system.ToolsAdded) != 1 || system.ToolsAdded[0].Name != "read" {
			t.Fatalf("%s: messages = %#v", name, messages)
		}
		if _, ok := messages[1].(ai.UserMessage); !ok {
			t.Fatalf("%s: messages[1] = %#v", name, messages[1])
		}
	}

	for name, in := range map[string]map[string]any{
		"unknown role": {"messages": []any{map[string]any{"role": "bogus", "content": "x"}}},
		"bad content":  {"messages": []any{map[string]any{"role": "user", "content": 5}}},
		"not an array": {"messages": "x"},
	} {
		if _, err := decodeTranscript(in); err == nil {
			t.Fatalf("%s: expected an error", name)
		}
	}
}

func TestOAuthWrapperRoundTrip(t *testing.T) {
	var notified []string
	auth := wrapAuth(ai.ProviderAuth{OAuth: &ai.OAuthAuth{
		Name: "sso", IsSubscription: true, LoginLabel: "Sign in",
		Login: func(_ context.Context, in ai.AuthInteraction, _ ai.LoginOptions) (ai.Credential, error) {
			in.Notify(ai.AuthInfoEvent{Message: "go"})
			code, err := in.Prompt(context.Background(), ai.AuthTextPrompt{Message: "code"})
			return ai.Credential{Type: "oauth", Access: "a-" + code, Refresh: "r1", Expires: 100}, err
		},
		Refresh: func(_ context.Context, c ai.Credential) (ai.Credential, error) {
			c.Access = "a2"
			return c, nil
		},
		ToAuth: func(c ai.Credential) (ai.ModelAuth, error) {
			return ai.ModelAuth{APIKey: c.Access, BaseURL: "http://base"}, nil
		},
	}})
	o := auth.OAuth
	if o == nil || o.Name != "sso" || o.IsSubscription == nil || !*o.IsSubscription || o.LoginLabel == nil || *o.LoginLabel != "Sign in" {
		t.Fatalf("oauth = %+v", o)
	}
	wire, err := o.Login(sdk.AuthInteraction{
		Prompt: func(map[string]any) (string, error) { return "code", nil },
		Notify: func(e map[string]any) error { notified = append(notified, e["message"].(string)); return nil },
	})
	if err != nil || wire["access"] != "a-code" || wire["refresh"] != "r1" || len(notified) != 1 {
		t.Fatalf("login = %v, %v, %v", wire, err, notified)
	}
	refreshed, err := o.Refresh(wire, context.Background())
	if err != nil || refreshed["access"] != "a2" || refreshed["refresh"] != "r1" {
		t.Fatalf("refresh = %v, %v", refreshed, err)
	}
	modelAuth, err := o.ToAuth(refreshed)
	if err != nil || modelAuth["apiKey"] != "a2" || modelAuth["baseUrl"] != "http://base" {
		t.Fatalf("toAuth = %v, %v", modelAuth, err)
	}
}

func TestAuthHostCallbackErrorsSurface(t *testing.T) {
	boom := errors.New("host down")
	auth := wrapAuth(ai.ProviderAuth{APIKey: &ai.APIKeyAuth{Name: "key",
		Resolve: func(_ context.Context, in ai.APIKeyAuthInput) (*ai.AuthResult, error) {
			if _, ok := in.Ctx.Env("K"); !ok {
				return nil, nil
			}
			return &ai.AuthResult{}, nil
		},
		Check: func(_ context.Context, in ai.APIKeyAuthInput) (*ai.AuthCheck, error) {
			in.Ctx.FileExists("/adc")
			return nil, nil
		},
		Login: func(_ context.Context, in ai.AuthInteraction) (ai.Credential, error) {
			in.Notify(ai.AuthInfoEvent{Message: "m"})
			return ai.Credential{Type: ai.CredentialAPIKey, Key: "k"}, nil
		},
	}})
	ctx := sdk.AuthContext{
		Env:        func(string) (*string, error) { return nil, boom },
		FileExists: func(string) (bool, error) { return false, boom },
	}
	if _, err := auth.APIKey.Resolve(sdk.APIKeyAuthInput{Ctx: ctx}); !errors.Is(err, boom) {
		t.Fatalf("Resolve err = %v", err)
	}
	if _, err := auth.APIKey.Check(sdk.APIKeyAuthInput{Ctx: ctx}); !errors.Is(err, boom) {
		t.Fatalf("Check err = %v", err)
	}
	_, err := auth.APIKey.Login(sdk.AuthInteraction{Notify: func(map[string]any) error { return boom }})
	if !errors.Is(err, boom) || !strings.Contains(err.Error(), "host down") {
		t.Fatalf("Login err = %v", err)
	}
}
