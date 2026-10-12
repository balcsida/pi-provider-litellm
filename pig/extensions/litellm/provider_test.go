package litellm

// Ports tests/provider.test.ts.

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/discover"
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

// modelByID returns the published model with id.
func modelByID(t *testing.T, provider *ai.ModelsProvider, id string) *ai.Model {
	t.Helper()
	models, err := provider.GetModels()
	if err != nil {
		t.Fatal(err)
	}
	for _, model := range models {
		if model.ID == id {
			return model
		}
	}
	t.Fatalf("model %q not published", id)
	return nil
}

func yes() *bool { v := true; return &v }

func messagesModel(t *testing.T, id string, compat *ai.ModelCompat, version int) types.DiscoveredModel {
	t.Helper()
	model := discoveredModel(id)
	model.API, model.Compat, model.LiteLLMDiscoveryVersion = ai.APIAnthropicMessages, compat, version
	models, err := toNativeModels("litellm", testRoot+"/v1", []types.DiscoveredModel{model}, false)
	if err != nil {
		t.Fatal(err)
	}
	return models[0]
}

func TestRefreshModelsSeedMasking(t *testing.T) {
	t.Run("a stale sidecar entry does not let a stale stored model override the startup seed", func(t *testing.T) {
		policies := newPolicyStore("")
		stale := nativeModel(t, "claude-sonnet-5")
		stale.Name, stale.LiteLLMDiscoveryVersion = "stale-name", 2
		policies.replace("litellm", []types.DiscoveredModel{stale})
		seed := nativeModel(t, "claude-sonnet-5")
		seed.Name = "seed-name"
		// seedModels writes the seed's policies over the stale ones; the version check must still see the old one.
		if err := policies.replaceSeed("litellm", []types.DiscoveredModel{seed}); err != nil {
			t.Fatal(err)
		}
		h := newHarness(t, func(o *providerOptions) {
			o.Policies, o.Models = policies, []types.DiscoveredModel{seed}
		})
		if _, err := refresh(h.provider, storedEntry(t, stale), false, false); err != nil {
			t.Fatal(err)
		}
		if got := modelByID(t, h.provider, "claude-sonnet-5").DisplayName; got != "seed-name" {
			t.Fatalf("name = %q, the stale stored model masked the seed", got)
		}
	})
	t.Run("a current stored model still wins over a seed written after a network refresh", func(t *testing.T) {
		policies := newPolicyStore("")
		seed := nativeModel(t, "m")
		seed.Name = "seed-name"
		policies.replace("litellm", []types.DiscoveredModel{seed})
		h := newHarness(t, func(o *providerOptions) { o.Policies, o.Models = policies, []types.DiscoveredModel{seed} })
		stored := nativeModel(t, "m")
		stored.Name = "stored-name"
		if _, err := refresh(h.provider, storedEntry(t, stored), false, false); err != nil {
			t.Fatal(err)
		}
		if got := modelByID(t, h.provider, "m").DisplayName; got != "stored-name" {
			t.Fatalf("name = %q", got)
		}
	})
}

func TestRefreshModelsUndecodableRecord(t *testing.T) {
	diagnostics := collectDiagnostics(t)
	h := newHarness(t, nil)
	good := nativeModel(t, "good")
	h.policies.replace("litellm", []types.DiscoveredModel{good})
	stored := storedEntry(t, good, json.RawMessage(`{"id":5}`))
	if _, err := refresh(h.provider, stored, false, false); err != nil {
		t.Fatalf("one undecodable record aborted the restore: %v", err)
	}
	if ids := modelIDs(t, h.provider); !equalStrings(ids, []string{"good"}) {
		t.Fatalf("ids = %v", ids)
	}
	if len(*diagnostics) != 1 || !strings.Contains((*diagnostics)[0], "undecodable stored model") {
		t.Fatalf("diagnostics = %v", *diagnostics)
	}
}

func TestRefreshModelsCacheVersionRestores(t *testing.T) {
	t.Run("removes an inherited strict-tool grant from a v2 Messages cache and refreshes it online", func(t *testing.T) {
		fresh := messagesModel(t, "claude-fable-5-1", &ai.ModelCompat{ForceAdaptiveThinking: yes()}, types.DiscoveryVersion)
		h := newHarness(t, func(o *providerOptions) {
			o.Discover = func(context.Context, ai.Credential) (*discoveredCatalog, error) {
				return &discoveredCatalog{Models: []types.DiscoveredModel{fresh}, BaseURL: testRoot}, nil
			}
		})
		stale := messagesModel(t, "claude-fable-5-1", &ai.ModelCompat{ForceAdaptiveThinking: yes(), SupportsStrictTools: yes()}, 2)
		h.policies.replace("litellm", []types.DiscoveredModel{stale})
		if _, err := refresh(h.provider, storedEntry(t, stale), false, false); err != nil {
			t.Fatal(err)
		}
		compat := modelByID(t, h.provider, "claude-fable-5-1").ProviderMeta.Compat
		if compat == nil || compat.SupportsStrictTools != nil || compat.ForceAdaptiveThinking == nil || !*compat.ForceAdaptiveThinking {
			t.Fatalf("offline compat = %+v", compat)
		}
		if _, err := refresh(h.provider, storedEntry(t, stale), true, false); err != nil {
			t.Fatal(err)
		}
		if entry, _ := h.policies.entry("litellm", "claude-fable-5-1"); entry.Version != types.DiscoveryVersion {
			t.Fatalf("version after the online refresh = %d", entry.Version)
		}
		compat = modelByID(t, h.provider, "claude-fable-5-1").ProviderMeta.Compat
		if compat == nil || compat.SupportsStrictTools != nil || compat.ForceAdaptiveThinking == nil {
			t.Fatalf("online compat = %+v", compat)
		}
	})
	t.Run("keeps a startup-discovered strict-tool grant over its v2 cache entry offline", func(t *testing.T) {
		seed := messagesModel(t, "claude-sonnet-5", &ai.ModelCompat{SupportsStrictTools: yes()}, types.DiscoveryVersion)
		stale := seed
		stale.LiteLLMDiscoveryVersion = 2
		policies := newPolicyStore("")
		policies.replace("litellm", []types.DiscoveredModel{stale})
		policies.replaceSeed("litellm", []types.DiscoveredModel{seed})
		h := newHarness(t, func(o *providerOptions) { o.Policies, o.Models = policies, []types.DiscoveredModel{seed} })
		if _, err := refresh(h.provider, storedEntry(t, stale), false, false); err != nil {
			t.Fatal(err)
		}
		compat := modelByID(t, h.provider, "claude-sonnet-5").ProviderMeta.Compat
		if compat == nil || compat.SupportsStrictTools == nil || !*compat.SupportsStrictTools || h.calls.Load() != 0 {
			t.Fatalf("compat = %+v calls = %d", compat, h.calls.Load())
		}
	})
	t.Run("replaces a v3 cache entry from before the GPT-5.5+ tool-reasoning policy with startup discovery", func(t *testing.T) {
		seed := nativeModel(t, "gpt-6-sol")
		seed.LiteLLMBackendFamily = types.FamilyOpenAI
		seed.LiteLLMPolicy = &types.LiteLLMModelPolicy{DropToolReasoning: true}
		stale := nativeModel(t, "gpt-6-sol")
		stale.LiteLLMBackendFamily, stale.LiteLLMDiscoveryVersion = types.FamilyOpenAI, 3
		policies := newPolicyStore("")
		policies.replace("litellm", []types.DiscoveredModel{stale})
		policies.replaceSeed("litellm", []types.DiscoveredModel{seed})
		h := newHarness(t, func(o *providerOptions) { o.Policies, o.Models = policies, []types.DiscoveredModel{seed} })
		if _, err := refresh(h.provider, storedEntry(t, stale), false, false); err != nil {
			t.Fatal(err)
		}
		policy, _, _ := policies.policyFor("litellm", "gpt-6-sol")
		if policy == nil || !policy.DropToolReasoning || policy.ExplicitToolReasoningOff {
			t.Fatalf("policy = %+v, want the seed's", policy)
		}
	})
	t.Run("restores the tool-reasoning drop for a v3 GPT-5.5+ Chat cache entry offline", func(t *testing.T) {
		h := newHarness(t, nil)
		stale := nativeModel(t, "gpt-6-sol")
		stale.LiteLLMBackendFamily, stale.LiteLLMDiscoveryVersion = types.FamilyOpenAI, 3
		h.policies.replace("litellm", []types.DiscoveredModel{stale})
		if _, err := refresh(h.provider, storedEntry(t, stale), false, false); err != nil {
			t.Fatal(err)
		}
		policy, _, _ := h.policies.policyFor("litellm", "gpt-6-sol")
		if policy == nil || !policy.DropToolReasoning || !policy.ExplicitToolReasoningOff {
			t.Fatalf("policy = %+v", policy)
		}
	})
}

func TestRefreshModelsReenrichesCachedCatalogAliases(t *testing.T) {
	fallback := func(t *testing.T, id string) types.DiscoveredModel {
		model := nativeModel(t, id)
		model.Name, model.MaxTokens, model.Compat = id+" (no metadata)", 16_384, nil
		return model
	}
	restore := func(t *testing.T, models ...types.DiscoveredModel) providerHarness {
		h := newHarness(t, nil)
		h.policies.replace("litellm", models)
		if _, err := refresh(h.provider, storedEntry(t, anySlice(models)...), false, false); err != nil {
			t.Fatal(err)
		}
		if h.calls.Load() != 0 {
			t.Fatal("offline restore ran discovery")
		}
		return h
	}
	allNull := ai.ThinkingLevelMap{"off": nil, "minimal": nil, "low": nil, "medium": nil, "high": nil, "xhigh": nil, "max": nil}

	t.Run("re-enriches stale cached catalog aliases offline without discovery", func(t *testing.T) {
		h := restore(t, fallback(t, "opus-5"))
		model := modelByID(t, h.provider, "opus-5")
		compat := model.ProviderMeta.Compat
		if model.DisplayName != "Claude Opus 5" || !model.ProviderMeta.Reasoning || model.ProviderMeta.API != ai.APIOpenAICompletions ||
			model.ProviderMeta.BaseURL != testRoot+"/v1" || !reflect.DeepEqual(model.ThinkingLevelMap, allNull) ||
			compat == nil || compat.CacheControlFormat != "anthropic" || compat.SupportsStore == nil || *compat.SupportsStore ||
			compat.SupportsReasoningEffort == nil || *compat.SupportsReasoningEffort {
			t.Fatalf("model = %+v compat = %+v", model, compat)
		}
	})
	t.Run("updates a cached catalog model to the Responses transport offline", func(t *testing.T) {
		h := restore(t, fallback(t, "openai/gpt-5.5"))
		model := modelByID(t, h.provider, "openai/gpt-5.5")
		if model.DisplayName != "GPT-5.5" || model.ProviderMeta.API != ai.APIOpenAIResponses || model.ProviderMeta.Compat != nil {
			t.Fatalf("model = %+v", model)
		}
	})
	t.Run("does not re-enrich partially enriched cached aliases offline", func(t *testing.T) {
		base := fallback(t, "opus-5")
		variants := map[string]func(*types.DiscoveredModel){
			"reasoning":     func(m *types.DiscoveredModel) { m.Reasoning = true },
			"image input":   func(m *types.DiscoveredModel) { m.Input = []string{"text", "image"} },
			"cost":          func(m *types.DiscoveredModel) { m.Cost.Input = 1 },
			"context":       func(m *types.DiscoveredModel) { m.ContextWindow = 128_001 },
			"output tokens": func(m *types.DiscoveredModel) { m.MaxTokens = 16_385 },
		}
		for name, mutate := range variants {
			t.Run(name, func(t *testing.T) {
				cached := base
				mutate(&cached)
				h := restore(t, cached)
				model := modelByID(t, h.provider, "opus-5")
				if model.DisplayName != "opus-5 (no metadata)" {
					t.Fatalf("name = %q was re-enriched", model.DisplayName)
				}
				if name == "reasoning" && !reflect.DeepEqual(model.ThinkingLevelMap, allNull) {
					t.Fatalf("levels = %v", model.ThinkingLevelMap)
				}
			})
		}
	})
	t.Run("keeps unknown stale cached models unchanged offline", func(t *testing.T) {
		cached := fallback(t, "unknown-model")
		h := restore(t, cached)
		model := modelByID(t, h.provider, "unknown-model")
		if model.DisplayName != "unknown-model (no metadata)" || model.ProviderMeta.API != ai.APIOpenAICompletions || model.ProviderMeta.Reasoning {
			t.Fatalf("model = %+v", model)
		}
	})
}

func anySlice(models []types.DiscoveredModel) []any {
	out := make([]any, len(models))
	for i, model := range models {
		out[i] = model
	}
	return out
}

func TestDiscoveryCacheVersionTransition(t *testing.T) {
	legacyModel := func(t *testing.T) types.DiscoveredModel {
		model := nativeModel(t, "legacy")
		model.Reasoning, model.ThinkingLevelMap, model.LiteLLMDiscoveryVersion = true, ai.ThinkingLevelMap{"low": nil}, 0
		return model
	}
	moonshotCompat := &ai.ModelCompat{SupportsStore: new(false), SupportsDeveloperRole: new(false),
		SupportsReasoningEffort: new(false), SupportsStrictMode: new(false), MaxTokensField: "max_tokens"}
	restorableV2 := func(t *testing.T) types.DiscoveredModel {
		model := nativeModel(t, "moonshot/kimi-k2.6")
		model.Compat, model.LiteLLMDiscoveryVersion = moonshotCompat, 2
		return model
	}
	wantMoonshotPolicy := func(t *testing.T, h providerHarness) {
		t.Helper()
		policy, _, _ := h.policies.policyFor("litellm", "moonshot/kimi-k2.6")
		if policy == nil || policy.NormalizeStrictToolMessages || !policy.NormalizeThinkTags || policy.SuppressReasoningVisibility {
			t.Fatalf("policy = %+v", policy)
		}
	}

	t.Run("refreshes and replaces a legacy store when the network phase runs", func(t *testing.T) {
		diagnostics := collectDiagnostics(t)
		h := newHarness(t, nil)
		legacy := legacyModel(t)
		if _, err := refresh(h.provider, storedEntry(t, legacy), false, false); err != nil {
			t.Fatal(err)
		}
		if _, err := refresh(h.provider, storedEntry(t, legacy), true, false); err != nil {
			t.Fatal(err)
		}
		if ids := modelIDs(t, h.provider); !equalStrings(ids, []string{"fresh"}) || h.calls.Load() != 1 || len(*diagnostics) != 0 {
			t.Fatalf("ids=%v calls=%d diagnostics=%v", ids, h.calls.Load(), *diagnostics)
		}
	})
	t.Run("restores response policy without rewriting legacy reasoning levels offline", func(t *testing.T) {
		h := newHarness(t, nil)
		legacy := legacyModel(t)
		legacy.ID, legacy.Compat = "moonshot/kimi-k2.6", moonshotCompat
		if _, err := refresh(h.provider, storedEntry(t, legacy), false, false); err != nil {
			t.Fatal(err)
		}
		model := modelByID(t, h.provider, "moonshot/kimi-k2.6")
		if _, ok := model.ThinkingLevelMap["low"]; !ok || len(model.ThinkingLevelMap) != 1 || h.calls.Load() != 0 {
			t.Fatalf("levels = %v calls = %d", model.ThinkingLevelMap, h.calls.Load())
		}
		wantMoonshotPolicy(t, h)
	})
	t.Run("handles mixed stores per entry without warning during offline restore", func(t *testing.T) {
		diagnostics := collectDiagnostics(t)
		h := newHarness(t, nil)
		v2 := restorableV2(t)
		h.policies.replace("litellm", []types.DiscoveredModel{v2})
		if _, err := refresh(h.provider, storedEntry(t, legacyModel(t), v2), false, false); err != nil {
			t.Fatal(err)
		}
		legacy := modelByID(t, h.provider, "legacy")
		if !legacy.ProviderMeta.Reasoning || len(legacy.ThinkingLevelMap) != 1 || h.calls.Load() != 0 || len(*diagnostics) != 0 {
			t.Fatalf("legacy = %+v calls=%d diagnostics=%v", legacy, h.calls.Load(), *diagnostics)
		}
		wantMoonshotPolicy(t, h)
	})
	t.Run("warns once when repeated refresh attempts leave a mixed legacy store in place", func(t *testing.T) {
		diagnostics := collectDiagnostics(t)
		h := newHarness(t, func(o *providerOptions) {
			o.Discover = func(context.Context, ai.Credential) (*discoveredCatalog, error) {
				return nil, errors.New("refresh failed")
			}
		})
		v2 := restorableV2(t)
		h.policies.replace("litellm", []types.DiscoveredModel{v2})
		stored := storedEntry(t, legacyModel(t), v2)
		refresh(h.provider, stored, false, false)
		for range 2 {
			if _, err := refresh(h.provider, stored, true, false); err == nil || err.Error() != "refresh failed" {
				t.Fatalf("err = %v", err)
			}
			refresh(h.provider, stored, false, false)
		}
		if ids := modelIDs(t, h.provider); !equalStrings(ids, []string{"legacy", "moonshot/kimi-k2.6"}) {
			t.Fatalf("ids = %v", ids)
		}
		if len(*diagnostics) != 1 || !strings.Contains((*diagnostics)[0], "required network refresh failed") {
			t.Fatalf("diagnostics = %v", *diagnostics)
		}
	})
}

func TestReducedDiscoveryMetadataSurvivesTheProviderCache(t *testing.T) {
	const chat = `"mode":"chat"`
	cases := []struct{ name, rows, want string }{
		{"matching backend", `{"model_name":"openai/gpt-5.5","model_info":{"id":"only",` + chat + `},"litellm_params":{"model":"openai/gpt-5.5"}}`, "openai/gpt-5.5"},
		{"conflicting backend", `{"model_name":"openai/gpt-5.5","model_info":{"id":"only",` + chat + `},"litellm_params":{"model":"openai/gpt-5.5-internal-preview"}}`, "openai/gpt-5.5 (incomplete metadata)"},
		{"unknown backend", `{"model_name":"openai/gpt-5.5","model_info":{"id":"only",` + chat + `},"litellm_params":{"model":"internal/mystery"}}`, "openai/gpt-5.5 (incomplete metadata)"},
		{"mixed deployments", `{"model_name":"openai/gpt-5.5","model_info":{"id":"a",` + chat + `}},` +
			`{"model_name":"openai/gpt-5.5","model_info":{"id":"b",` + chat + `},"litellm_params":{"model":"internal/mystery"}}`, "openai/gpt-5.5 (incomplete metadata)"},
		{"differing duplicate ids", `{"model_name":"openai/gpt-5.5","model_info":{"id":"same",` + chat + `}},` +
			`{"model_name":"openai/gpt-5.5","model_info":{"id":"same",` + chat + `,"max_input_tokens":64000}}`, "openai/gpt-5.5 (incomplete metadata)"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/model/info" {
					http.NotFound(w, r)
					return
				}
				w.Header().Set("Content-Type", "application/json")
				w.Write([]byte(`{"data":[` + tc.rows + `]}`))
			}))
			defer server.Close()
			disabled := false
			result, err := discover.DiscoverModels(context.Background(), server.URL, "sk-test", discover.Options{
				DiscoveryOptions: types.DiscoveryOptions{ModelsDev: &disabled, AllowInsecureHTTP: true}, Silent: true,
			})
			if err != nil {
				t.Fatal(err)
			}
			online, err := toNativeModels("litellm", testRoot+"/v1", result.Models, false)
			if err != nil || len(online) != 1 || online[0].Name != tc.want {
				t.Fatalf("online = %+v err = %v", online, err)
			}
			want, err := toAIModels("litellm", online)
			if err != nil {
				t.Fatal(err)
			}
			h := newHarness(t, nil)
			h.policies.replace("litellm", online)
			if _, err := refresh(h.provider, storedEntry(t, online[0]), false, false); err != nil {
				t.Fatal(err)
			}
			if got := modelByID(t, h.provider, online[0].ID); !reflect.DeepEqual(got, want[0].(*ai.Model)) {
				t.Fatalf("offline restore changed the model:\n got %+v\nwant %+v", got, want[0])
			}
		})
	}
}

func TestStreamGuardOrdering(t *testing.T) {
	oversized := withFamily(nativeModel(t, "stale"), types.FamilyOpenAI)
	for _, method := range []string{"stream", "streamSimple"} {
		t.Run(method+" blocks stale hosts before tool-cap validation and protocol dispatch", func(t *testing.T) {
			h := newHarness(t, func(o *providerOptions) {
				o.ResolveCredentialRoot = func(*ai.Credential, string, string) (string, error) { return "https://other.example", nil }
			})
			h.policies.replace("litellm", []types.DiscoveredModel{oversized})
			fn := h.provider.Stream
			if method == "streamSimple" {
				fn = h.provider.StreamSimple
			}
			_, err := fn(context.Background(), firstModel(t, "litellm", oversized), chatContext(129), ai.StreamOptions{})
			if err == nil || !strings.Contains(err.Error(), "stale LiteLLM model root") || strings.Contains(err.Error(), "tool cap") {
				t.Fatalf("err = %v", err)
			}
		})
	}
	t.Run("blocks placeholder cached hosts on non-default ports before protocol dispatch", func(t *testing.T) {
		for _, host := range []string{"https://litellm.example.com:8443/v1", "https://LiteLLM.Example.com:8443/v1"} {
			h := newHarness(t, nil)
			model := firstModel(t, "litellm", nativeModel(t, "placeholder"))
			model.ProviderMeta.BaseURL = host
			_, err := h.provider.Stream(context.Background(), model, chatContext(0), ai.StreamOptions{})
			if err == nil || !strings.Contains(err.Error(), "placeholder LiteLLM model host") || !strings.Contains(err.Error(), "network refresh") {
				t.Fatalf("%s: err = %v", host, err)
			}
		}
	})
	t.Run("blocks a mixed-case placeholder credential root", func(t *testing.T) {
		h := newHarness(t, func(o *providerOptions) {
			o.ResolveCredentialRoot = func(*ai.Credential, string, string) (string, error) { return "https://LiteLLM.Example.com", nil }
		})
		_, err := h.provider.Stream(context.Background(), firstModel(t, "litellm", nativeModel(t, "m")), chatContext(0), ai.StreamOptions{})
		if err == nil || !strings.Contains(err.Error(), "placeholder LiteLLM model host") {
			t.Fatalf("err = %v", err)
		}
	})
}

func TestStreamAPIRouting(t *testing.T) {
	for api, wantPath := range map[ai.API]string{
		ai.APIAnthropicMessages: "/v1/messages",
		ai.APIOpenAICompletions: "/v1/chat/completions",
		ai.APIOpenAIResponses:   "/v1/responses",
	} {
		t.Run(string(api), func(t *testing.T) {
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
			model := discoveredModel("routed")
			model.API = api
			if api == ai.APIAnthropicMessages {
				model.Compat = &ai.ModelCompat{}
			}
			native, err := toNativeModels("litellm", server.URL, []types.DiscoveredModel{model}, true)
			if err != nil {
				t.Fatal(err)
			}
			stream, err := h.provider.Stream(context.Background(), firstModel(t, "litellm", native[0]), chatContext(0), ai.StreamOptions{APIKey: "k"})
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			stream.ResultContext(ctx)
			select {
			case path := <-paths:
				if path != wantPath {
					t.Fatalf("path = %q, want %q", path, wantPath)
				}
			default:
				t.Fatal("the request never reached the proxy")
			}
			if api == ai.APIAnthropicMessages && native[0].BaseURL != server.URL {
				t.Fatalf("Messages baseUrl = %q, want the root", native[0].BaseURL)
			}
		})
	}
}
