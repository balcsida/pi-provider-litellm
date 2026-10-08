package litellm

// Ports tests/provider.test.ts.

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

const testRoot = "https://proxy.example"

func collectDiagnostics(t *testing.T) *[]string {
	t.Helper()
	var messages []string
	previous := reportDiagnostic
	reportDiagnostic = func(message string) { messages = append(messages, message) }
	t.Cleanup(func() { reportDiagnostic = previous })
	return &messages
}

func discoveredModel(id string) types.DiscoveredModel {
	return types.DiscoveredModel{
		ID: id, Name: id, API: ai.APIOpenAICompletions, Input: []string{"text"}, ContextWindow: 128_000, MaxTokens: 4096,
		LiteLLMDiscoveryVersion: types.DiscoveryVersion,
	}
}

func nativeModel(t *testing.T, id string) types.DiscoveredModel {
	t.Helper()
	models, err := toNativeModels("litellm", testRoot+"/v1", []types.DiscoveredModel{discoveredModel(id)}, false)
	if err != nil {
		t.Fatal(err)
	}
	return models[0]
}

type providerHarness struct {
	provider *ai.ModelsProvider
	policies *policyStore
	calls    *atomic.Int32
}

func newHarness(t *testing.T, modify func(*providerOptions)) providerHarness {
	t.Helper()
	calls := new(atomic.Int32)
	policies := newPolicyStore("")
	options := providerOptions{
		ID: "litellm", Name: "LiteLLM", BaseURL: testRoot + "/v1", Policies: policies,
		Auth:                  ai.ProviderAuth{APIKey: &ai.APIKeyAuth{Name: "API key"}},
		ResolveCredentialRoot: func(*ai.Credential, string, string) (string, error) { return testRoot, nil },
		Discover: func(context.Context, ai.Credential) (*discoveredCatalog, error) {
			calls.Add(1)
			return &discoveredCatalog{Models: []types.DiscoveredModel{discoveredModel("fresh")}, BaseURL: testRoot}, nil
		},
	}
	if modify != nil {
		modify(&options)
	}
	provider, err := createLiteLLMProvider(options)
	if err != nil {
		t.Fatal(err)
	}
	return providerHarness{provider, policies, calls}
}

func storedEntry(t *testing.T, models ...any) *ai.ModelsStoreEntry {
	t.Helper()
	entry := &ai.ModelsStoreEntry{CheckedAt: new(float64(1))}
	for _, model := range models {
		data, err := json.Marshal(model)
		if err != nil {
			t.Fatal(err)
		}
		entry.Models = append(entry.Models, data)
	}
	return entry
}

// refresh runs RefreshModels; the returned count is the number of publications.
func refresh(provider *ai.ModelsProvider, stored *ai.ModelsStoreEntry, allowNetwork bool, force bool) (int, error) {
	count := 0
	err := provider.RefreshModels(ai.RefreshModelsContext{
		Credential: &ai.Credential{Type: ai.CredentialAPIKey, Key: "secret"}, Stored: stored, AllowNetwork: allowNetwork,
		Force: &force, Signal: context.Background(),
		Publish: func(publication ai.ModelsPublication) (bool, error) {
			count++
			if publication.Update != nil {
				publication.Update()
			}
			return true, nil
		},
	})
	return count, err
}

func modelIDs(t *testing.T, provider *ai.ModelsProvider) []string {
	t.Helper()
	models, err := provider.GetModels()
	if err != nil {
		t.Fatal(err)
	}
	ids := []string{}
	for _, model := range models {
		ids = append(ids, model.ID)
	}
	return ids
}

func equalStrings(a, b []string) bool { return strings.Join(a, "\x01") == strings.Join(b, "\x01") }

func TestToNativeModels(t *testing.T) {
	t.Run("converts discovery models into complete native models", func(t *testing.T) {
		model := nativeModel(t, "model-a")
		if model.Provider != "litellm" || model.API != ai.APIOpenAICompletions || model.BaseURL != testRoot+"/v1" {
			t.Fatalf("model = %+v", model)
		}
	})
	t.Run("projects each supported protocol from one normalized proxy root", func(t *testing.T) {
		base := discoveredModel("model")
		var in []types.DiscoveredModel
		for _, api := range []ai.API{ai.APIAnthropicMessages, ai.APIOpenAICompletions, ai.APIOpenAIResponses} {
			model := base
			model.API = api
			in = append(in, model)
		}
		out, err := toNativeModels("litellm", testRoot+"/v1/", in, false)
		if err != nil {
			t.Fatal(err)
		}
		want := []string{testRoot, testRoot + "/v1", testRoot + "/v1"}
		for i := range out {
			if out[i].BaseURL != want[i] {
				t.Errorf("%s baseUrl = %q, want %q", out[i].API, out[i].BaseURL, want[i])
			}
		}
	})
}

func TestRefreshModelsStoredModels(t *testing.T) {
	t.Run("restores current-version stored models offline without discovery", func(t *testing.T) {
		h := newHarness(t, nil)
		stored := nativeModel(t, "stored")
		h.policies.replace("litellm", []types.DiscoveredModel{stored})
		if _, err := refresh(h.provider, storedEntry(t, stored), false, false); err != nil {
			t.Fatal(err)
		}
		if ids := modelIDs(t, h.provider); !equalStrings(ids, []string{"stored"}) || h.calls.Load() != 0 {
			t.Fatalf("ids=%v calls=%d", ids, h.calls.Load())
		}
	})
	t.Run("ignores non-chat entries in the model store", func(t *testing.T) {
		h := newHarness(t, nil)
		var all []types.DiscoveredModel
		current := func(id string) map[string]any {
			model := nativeModel(t, id)
			all = append(all, model)
			data, _ := json.Marshal(model)
			var record map[string]any
			json.Unmarshal(data, &record)
			return record
		}
		untyped, typed, image, classifier := current("untyped-chat"), current("typed-chat"), current("image"), current("classifier")
		h.policies.replace("litellm", all)
		typed["type"], image["type"], classifier["type"] = "chat", "image", "classifier"
		if _, err := refresh(h.provider, storedEntry(t, untyped, image, typed, classifier), false, false); err != nil {
			t.Fatal(err)
		}
		if ids := modelIDs(t, h.provider); !equalStrings(ids, []string{"untyped-chat", "typed-chat"}) || h.calls.Load() != 0 {
			t.Fatalf("ids=%v", ids)
		}
	})
	t.Run("restores mixed legacy and current-version entries per entry without a warning offline", func(t *testing.T) {
		diagnostics := collectDiagnostics(t)
		h := newHarness(t, nil)
		legacy, current := nativeModel(t, "legacy"), nativeModel(t, "current")
		h.policies.replace("litellm", []types.DiscoveredModel{current})
		if _, err := refresh(h.provider, storedEntry(t, legacy, current), false, false); err != nil {
			t.Fatal(err)
		}
		if ids := modelIDs(t, h.provider); !equalStrings(ids, []string{"legacy", "current"}) || h.calls.Load() != 0 || len(*diagnostics) != 0 {
			t.Fatalf("ids=%v diagnostics=%v", ids, *diagnostics)
		}
	})
	t.Run("replaces legacy stored models after a successful forced refresh", func(t *testing.T) {
		diagnostics := collectDiagnostics(t)
		h := newHarness(t, nil)
		publications, err := refresh(h.provider, storedEntry(t, nativeModel(t, "stored")), true, false)
		if err != nil {
			t.Fatal(err)
		}
		if ids := modelIDs(t, h.provider); !equalStrings(ids, []string{"fresh"}) || h.calls.Load() != 1 || publications != 2 || len(*diagnostics) != 0 {
			t.Fatalf("ids=%v calls=%d publications=%d diagnostics=%v", ids, h.calls.Load(), publications, *diagnostics)
		}
		if _, _, ok := h.policies.policyFor("litellm", "fresh"); !ok {
			t.Fatal("discovery did not fill the policy store")
		}
	})
	t.Run("keeps legacy stored models and reports one version diagnostic when refresh fails", func(t *testing.T) {
		diagnostics := collectDiagnostics(t)
		h := newHarness(t, func(o *providerOptions) {
			o.Discover = func(context.Context, ai.Credential) (*discoveredCatalog, error) { return nil, errors.New("offline") }
		})
		stored, other := nativeModel(t, "stored"), nativeModel(t, "other-stored")
		if _, err := refresh(h.provider, storedEntry(t, stored), false, false); err != nil || len(*diagnostics) != 0 {
			t.Fatalf("offline refresh err=%v diagnostics=%v", err, *diagnostics)
		}
		for _, entry := range []*ai.ModelsStoreEntry{storedEntry(t, stored), storedEntry(t, stored, other)} {
			if _, err := refresh(h.provider, entry, true, false); err == nil || err.Error() != "offline" {
				t.Fatalf("err = %v", err)
			}
		}
		if ids := modelIDs(t, h.provider); !equalStrings(ids, []string{"stored", "other-stored"}) {
			t.Fatalf("ids=%v", ids)
		}
		if len(*diagnostics) != 1 || !strings.Contains((*diagnostics)[0], "version does not match 6") ||
			!strings.Contains((*diagnostics)[0], "network refresh failed") {
			t.Fatalf("diagnostics=%v", *diagnostics)
		}
	})
	t.Run("keeps a startup-discovered seed over a legacy stored entry and still requires refresh for the rest", func(t *testing.T) {
		diagnostics := collectDiagnostics(t)
		seed := nativeModel(t, "claude-sonnet-5")
		h := newHarness(t, func(o *providerOptions) {
			o.Models = []types.DiscoveredModel{seed}
			o.Discover = func(context.Context, ai.Credential) (*discoveredCatalog, error) { return nil, errors.New("offline") }
		})
		legacySeed := seed
		legacySeed.LiteLLMDiscoveryVersion = 2
		if _, err := refresh(h.provider, storedEntry(t, legacySeed), true, false); err == nil || len(*diagnostics) != 0 {
			t.Fatalf("seeded entry: err=%v diagnostics=%v", err, *diagnostics)
		}
		if _, err := refresh(h.provider, storedEntry(t, legacySeed, nativeModel(t, "stored")), true, false); err == nil || len(*diagnostics) != 1 {
			t.Fatalf("unseeded entry: err=%v diagnostics=%v", err, *diagnostics)
		}
	})
}

func TestFilterModels(t *testing.T) {
	models := func(t *testing.T, h providerHarness, ids ...string) []*ai.Model {
		t.Helper()
		var list []types.DiscoveredModel
		for _, id := range ids {
			list = append(list, nativeModel(t, id))
		}
		any, err := toAIModels("litellm", list)
		if err != nil {
			t.Fatal(err)
		}
		out := []*ai.Model{}
		for _, model := range any {
			out = append(out, model.(*ai.Model))
		}
		return out
	}
	t.Run("reprojects matching cached hosts and rejects stale or placeholder hosts", func(t *testing.T) {
		diagnostics := collectDiagnostics(t)
		h := newHarness(t, nil)
		list := models(t, h, "a", "stale", "placeholder")
		list[1].ProviderMeta.BaseURL = "https://other.example/v1"
		list[2].ProviderMeta.BaseURL = "https://litellm.example.com/v1"
		kept := h.provider.FilterModels(list, nil)
		if len(kept) != 1 || kept[0].ID != "a" || kept[0] != list[0] {
			t.Fatalf("kept = %v", kept)
		}
		if len(*diagnostics) != 2 || !strings.Contains((*diagnostics)[0], "stale LiteLLM model root https://other.example") ||
			!strings.Contains((*diagnostics)[1], "placeholder LiteLLM model host") {
			t.Fatalf("diagnostics = %q", *diagnostics)
		}
	})
	t.Run("accepts equivalent cached roots after URL canonicalization", func(t *testing.T) {
		h := newHarness(t, nil)
		list := models(t, h, "a")
		list[0].ProviderMeta.BaseURL = "HTTPS://PROXY.example:443/v1"
		if kept := h.provider.FilterModels(list, nil); len(kept) != 1 || kept[0].ProviderMeta.BaseURL != testRoot+"/v1" {
			t.Fatalf("kept = %v", kept)
		}
	})
	t.Run("keeps proxy-root paths case-sensitive and rejects another same-origin prefix", func(t *testing.T) {
		collectDiagnostics(t)
		h := newHarness(t, func(o *providerOptions) {
			o.ResolveCredentialRoot = func(*ai.Credential, string, string) (string, error) { return testRoot + "/Team", nil }
		})
		list := models(t, h, "a", "b")
		list[0].ProviderMeta.BaseURL = testRoot + "/team/v1"
		list[1].ProviderMeta.BaseURL = testRoot + "/Team/v1"
		if kept := h.provider.FilterModels(list, nil); len(kept) != 1 || kept[0].ID != "b" {
			t.Fatalf("kept = %v", kept)
		}
	})
	t.Run("keeps valid models while filtering unsupported protocols and malformed URLs", func(t *testing.T) {
		diagnostics := collectDiagnostics(t)
		h := newHarness(t, nil)
		list := models(t, h, "ok", "foreign", "bad-url")
		list[1].ProviderMeta.API = "google-generative-ai"
		list[2].ProviderMeta.BaseURL = "not a url"
		kept := h.provider.FilterModels(list, nil)
		if len(kept) != 1 || kept[0].ID != "ok" || len(*diagnostics) != 2 ||
			!strings.Contains((*diagnostics)[0], `unsupported protocol "google-generative-ai"`) ||
			!strings.Contains((*diagnostics)[1], "invalid LiteLLM model URL") {
			t.Fatalf("kept=%v diagnostics=%q", kept, *diagnostics)
		}
	})
	t.Run("reports hidden cached models when no LiteLLM base URL is configured, once", func(t *testing.T) {
		diagnostics := collectDiagnostics(t)
		h := newHarness(t, func(o *providerOptions) {
			o.ResolveCredentialRoot = func(*ai.Credential, string, string) (string, error) { return "", nil }
		})
		list := models(t, h, "a", "b")
		for range 2 {
			if kept := h.provider.FilterModels(list, nil); len(kept) != 0 {
				t.Fatalf("kept = %v", kept)
			}
		}
		want := "LiteLLM (litellm): 2 model(s) hidden because no LiteLLM base URL is configured; set LITELLM_BASE_URL or run /login litellm"
		if len(*diagnostics) != 1 || (*diagnostics)[0] != want {
			t.Fatalf("diagnostics = %q", *diagnostics)
		}
	})
	t.Run("stays silent when an unconfigured provider has no models to hide", func(t *testing.T) {
		diagnostics := collectDiagnostics(t)
		h := newHarness(t, func(o *providerOptions) {
			o.ResolveCredentialRoot = func(*ai.Credential, string, string) (string, error) { return "", nil }
		})
		h.provider.FilterModels(nil, nil)
		if len(*diagnostics) != 0 {
			t.Fatalf("diagnostics = %q", *diagnostics)
		}
	})
	t.Run("rejects a malformed or placeholder credential root with guided refresh advice", func(t *testing.T) {
		for root, want := range map[string]string{
			"not a url":                   "Active credentials have an invalid LiteLLM model URL",
			"https://litellm.example.com": "Active credentials use a placeholder LiteLLM model host",
		} {
			diagnostics := collectDiagnostics(t)
			h := newHarness(t, func(o *providerOptions) {
				o.ResolveCredentialRoot = func(*ai.Credential, string, string) (string, error) { return root, nil }
			})
			h.provider.FilterModels(models(t, h, "a"), nil)
			if len(*diagnostics) != 1 || !strings.Contains((*diagnostics)[0], want) ||
				!strings.Contains((*diagnostics)[0], "a network refresh with a valid LiteLLM base URL is required") {
				t.Fatalf("%s: diagnostics = %q", root, *diagnostics)
			}
		}
	})
	t.Run("preserves explicitly allowed insecure HTTP and rejects it by default", func(t *testing.T) {
		collectDiagnostics(t)
		local := "http://host.docker.internal"
		resolve := func(*ai.Credential, string, string) (string, error) { return local, nil }
		allowed := newHarness(t, func(o *providerOptions) { o.AllowInsecureHTTP, o.ResolveCredentialRoot = true, resolve })
		list, err := toNativeModels("litellm", local, []types.DiscoveredModel{discoveredModel("local")}, true)
		if err != nil || list[0].BaseURL != local+"/v1" {
			t.Fatalf("native = %v, %v", list, err)
		}
		if _, err := toNativeModels("litellm", local, []types.DiscoveredModel{discoveredModel("local")}, false); err == nil {
			t.Fatal("insecure projection accepted by default")
		}
		aiModels, _ := toAIModels("litellm", list)
		if kept := allowed.provider.FilterModels([]*ai.Model{aiModels[0].(*ai.Model)}, nil); len(kept) != 1 {
			t.Fatalf("kept = %v", kept)
		}
		denied := newHarness(t, func(o *providerOptions) { o.ResolveCredentialRoot = resolve })
		if kept := denied.provider.FilterModels([]*ai.Model{aiModels[0].(*ai.Model)}, nil); len(kept) != 0 {
			t.Fatalf("kept = %v", kept)
		}
	})
}

func TestFetchModels(t *testing.T) {
	t.Run("retains previous models when discovery rejects or is aborted", func(t *testing.T) {
		for _, failure := range []error{errors.New("offline"), context.Canceled} {
			h := newHarness(t, func(o *providerOptions) {
				o.Models = []types.DiscoveredModel{nativeModel(t, "previous")}
				o.Discover = func(context.Context, ai.Credential) (*discoveredCatalog, error) { return nil, failure }
			})
			if _, err := refresh(h.provider, nil, true, false); !errors.Is(err, failure) {
				t.Fatalf("err = %v", err)
			}
			if ids := modelIDs(t, h.provider); !equalStrings(ids, []string{"previous"}) {
				t.Fatalf("ids = %v", ids)
			}
		}
	})
	t.Run("publishes and persists successful discovery with the credential URL", func(t *testing.T) {
		var persisted *ai.ModelsStoreEntry
		h := newHarness(t, func(o *providerOptions) {
			o.Discover = func(context.Context, ai.Credential) (*discoveredCatalog, error) {
				return &discoveredCatalog{Models: []types.DiscoveredModel{discoveredModel("fresh")}, BaseURL: "https://credential.example"}, nil
			}
		})
		err := h.provider.RefreshModels(ai.RefreshModelsContext{
			Credential: &ai.Credential{Type: ai.CredentialAPIKey, Key: "k"}, AllowNetwork: true, Signal: context.Background(),
			Publish: func(p ai.ModelsPublication) (bool, error) {
				if p.Persist != nil {
					persisted = p.Persist
				}
				if p.Update != nil {
					p.Update()
				}
				return true, nil
			},
		})
		if err != nil || persisted == nil || len(persisted.Models) != 1 {
			t.Fatalf("err=%v persisted=%v", err, persisted)
		}
		models, _ := h.provider.GetModels()
		if models[0].ProviderMeta.BaseURL != "https://credential.example/v1" {
			t.Fatalf("baseUrl = %q", models[0].ProviderMeta.BaseURL)
		}
	})
	t.Run("requires a credential", func(t *testing.T) {
		h := newHarness(t, nil)
		err := h.provider.RefreshModels(ai.RefreshModelsContext{
			AllowNetwork: true, Signal: context.Background(), Publish: func(ai.ModelsPublication) (bool, error) { return true, nil },
		})
		if err == nil || !strings.Contains(err.Error(), "requires a credential") {
			t.Fatalf("err = %v", err)
		}
	})
}

func chatContext(tools int) ai.TranscriptContext {
	schemas := make([]ai.ToolSchema, tools)
	for i := range schemas {
		schemas[i] = ai.ToolSchema{Name: "tool-" + string(rune('a'+i%26)) + strings.Repeat("x", i/26), Description: "test tool"}
	}
	return ai.NormalizeContext(ai.Context{Tools: schemas})
}

func firstModel(t *testing.T, provider string, model types.DiscoveredModel) *ai.Model {
	t.Helper()
	models, err := toAIModels(provider, []types.DiscoveredModel{model})
	if err != nil {
		t.Fatal(err)
	}
	return models[0].(*ai.Model)
}

func TestStreamGuard(t *testing.T) {
	for _, method := range []string{"stream", "streamSimple"} {
		call := func(h providerHarness, model *ai.Model, transcript ai.TranscriptContext, options ai.StreamOptions) error {
			fn := h.provider.Stream
			if method == "streamSimple" {
				fn = h.provider.StreamSimple
			}
			_, err := fn(context.Background(), model, transcript, options)
			return err
		}
		t.Run(method+" passes the AuthResult env root and API key to resolveCredentialRoot", func(t *testing.T) {
			var gotRoot, gotKey string
			h := newHarness(t, func(o *providerOptions) {
				o.ResolveCredentialRoot = func(credential *ai.Credential, root, key string) (string, error) {
					if credential != nil {
						t.Error("credential must be nil on the request path")
					}
					gotRoot, gotKey = root, key
					return "", nil
				}
			})
			call(h, firstModel(t, "litellm", nativeModel(t, "m")), chatContext(0),
				ai.StreamOptions{APIKey: "resolved-key", Env: ai.ProviderEnv{envBaseURL: "https://auth-result.example"}})
			if gotRoot != "https://auth-result.example" || gotKey != "resolved-key" {
				t.Fatalf("root=%q key=%q", gotRoot, gotKey)
			}
		})
		t.Run(method+" blocks requests without a model host, with a stale host, or with an unsupported protocol", func(t *testing.T) {
			root := ""
			h := newHarness(t, func(o *providerOptions) {
				o.ResolveCredentialRoot = func(*ai.Credential, string, string) (string, error) { return root, nil }
			})
			model := firstModel(t, "litellm", nativeModel(t, "m"))
			if err := call(h, model, chatContext(0), ai.StreamOptions{}); err == nil ||
				!strings.Contains(err.Error(), "Active credentials do not identify a LiteLLM model host") {
				t.Fatalf("no host: %v", err)
			}
			root = "https://other.example"
			if err := call(h, model, chatContext(0), ai.StreamOptions{}); err == nil || !strings.Contains(err.Error(), "stale LiteLLM model root") {
				t.Fatalf("stale: %v", err)
			}
			root = "https://litellm.example.com"
			if err := call(h, model, chatContext(0), ai.StreamOptions{}); err == nil || !strings.Contains(err.Error(), "placeholder LiteLLM model host") {
				t.Fatalf("placeholder: %v", err)
			}
			root = testRoot
			foreign := *model
			foreign.ProviderMeta.API = "google-generative-ai"
			if err := call(h, &foreign, chatContext(0), ai.StreamOptions{}); err == nil ||
				!strings.Contains(err.Error(), `declares unsupported protocol "google-generative-ai"; set "api" to one of anthropic-messages, openai-completions, openai-responses`) {
				t.Fatalf("protocol: %v", err)
			}
		})
	}

	t.Run("blocks oversized OpenAI-family Chat tool catalogs and spares other families", func(t *testing.T) {
		h := newHarness(t, nil)
		model := firstModel(t, "litellm", nativeModel(t, "opaque-gpt-route"))
		h.policies.replace("litellm", []types.DiscoveredModel{withFamily(nativeModel(t, "opaque-gpt-route"), types.FamilyOpenAI)})
		_, err := h.provider.Stream(context.Background(), model, chatContext(129), ai.StreamOptions{})
		want := "LiteLLM model opaque-gpt-route uses Chat Completions with 129 tools, exceeding the 128-tool cap; route this model via Responses or reduce enabled extensions"
		if err == nil || err.Error() != want {
			t.Fatalf("err = %v", err)
		}
		h.policies.replace("litellm", []types.DiscoveredModel{withFamily(nativeModel(t, "opaque-gpt-route"), types.FamilyKimi)})
		stream, err := h.provider.Stream(context.Background(), model, chatContext(129), ai.StreamOptions{})
		if err != nil && strings.Contains(err.Error(), "tool cap") {
			t.Fatalf("kimi family was capped: %v", err)
		}
		_ = stream
	})

	t.Run("dispatches to the projected base URL, including explicitly allowed insecure HTTP", func(t *testing.T) {
		paths := make(chan string, 4)
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			paths <- r.URL.Path
			http.Error(w, `{"error":{"message":"stop"}}`, http.StatusBadRequest)
		}))
		defer server.Close()
		h := newHarness(t, func(o *providerOptions) {
			o.AllowInsecureHTTP = true
			o.ResolveCredentialRoot = func(*ai.Credential, string, string) (string, error) { return server.URL, nil }
		})
		models, err := toNativeModels("litellm", server.URL, []types.DiscoveredModel{discoveredModel("local")}, true)
		if err != nil {
			t.Fatal(err)
		}
		stream, err := h.provider.StreamSimple(context.Background(), firstModel(t, "litellm", models[0]), chatContext(0), ai.StreamOptions{APIKey: "k"})
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		stream.ResultContext(ctx)
		select {
		case path := <-paths:
			if path != "/v1/chat/completions" {
				t.Fatalf("path = %q", path)
			}
		default:
			t.Fatal("request never reached the projected base URL")
		}
	})
}

func withFamily(model types.DiscoveredModel, family types.BackendFamily) types.DiscoveredModel {
	model.LiteLLMBackendFamily = family
	return model
}
