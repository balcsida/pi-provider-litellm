package litellm

// Ports the startup cases of tests/index.test.ts as unit tests of the Go factory pieces.

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

func startup(t *testing.T) (*extensionState, providerDefinition) {
	t.Helper()
	dir := hermeticAgentDir(t)
	return startupIn(t, dir)
}

func startupIn(t *testing.T, dir string) (*extensionState, providerDefinition) {
	t.Helper()
	definitions := getProviderDefinitions(readGlobalLiteLLMSettings())
	return newExtensionState(nil, definitions, newPolicyStore(filepath.Join(dir, policiesFilename))), definitions[0]
}

// proxy serves /model/info with one chat model and counts those requests.
func proxy(t *testing.T, status int, hook func(call int32)) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	calls := new(atomic.Int32)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/model/info" {
			http.NotFound(w, r)
			return
		}
		call := calls.Add(1)
		if hook != nil {
			hook(call)
		}
		if status != http.StatusOK {
			http.Error(w, "boom", status)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"data":[{"model_name":"fresh-model","model_info":{"mode":"chat"}}]}`))
	}))
	t.Cleanup(server.Close)
	return server, calls
}

func TestStartup(t *testing.T) {
	t.Run("restores Pi-managed models offline without discovery", func(t *testing.T) {
		state, definition := startup(t)
		t.Setenv(envOffline, "1")
		server, calls := proxy(t, 200, nil)
		t.Setenv(envBaseURL, server.URL)
		provider, err := state.newProvider(definition, state.seedModels(definition))
		if err != nil {
			t.Fatal(err)
		}
		stored := nativeModel(t, "stored-model")
		stored.BaseURL = server.URL + "/v1"
		if _, err := refresh(provider, storedEntry(t, stored), false, false); err != nil {
			t.Fatal(err)
		}
		if ids := modelIDs(t, provider); !equalStrings(ids, []string{"stored-model"}) || calls.Load() != 0 {
			t.Fatalf("ids=%v calls=%d", ids, calls.Load())
		}
	})

	t.Run("keeps Pi-managed models when an online refresh fails", func(t *testing.T) {
		state, definition := startup(t)
		server, _ := proxy(t, http.StatusInternalServerError, nil)
		t.Setenv(envBaseURL, server.URL)
		provider, err := state.newProvider(definition, nil)
		if err != nil {
			t.Fatal(err)
		}
		stored := nativeModel(t, "stored-model")
		stored.BaseURL = server.URL + "/v1"
		stored.LiteLLMDiscoveryVersion = types.DiscoveryVersion
		state.policies.replace(definition.Name, []types.DiscoveredModel{stored})
		credential := ai.Credential{Type: ai.CredentialAPIKey, Key: "sk-test", Env: map[string]string{envBaseURL: server.URL}}
		entry := storedEntry(t, stored)
		run := func(allowNetwork, force bool) error {
			return provider.RefreshModels(ai.RefreshModelsContext{
				Credential: &credential, Stored: entry, AllowNetwork: allowNetwork, Force: &force, Signal: context.Background(),
				Publish: func(p ai.ModelsPublication) (bool, error) {
					if p.Update != nil {
						p.Update()
					}
					return true, nil
				},
			})
		}
		if err := run(false, false); err != nil {
			t.Fatal(err)
		}
		if err := run(true, true); err == nil || !strings.Contains(err.Error(), "500") {
			t.Fatalf("err = %v", err)
		}
		if ids := modelIDs(t, provider); !equalStrings(ids, []string{"stored-model"}) {
			t.Fatalf("ids = %v", ids)
		}
		if state.networkAttempts[definition.Name] != 1 {
			t.Fatalf("attempts = %d", state.networkAttempts[definition.Name])
		}
	})

	t.Run("does not cache the credential of a model refresh that predates login", func(t *testing.T) {
		state, definition := startup(t)
		release := make(chan struct{})
		reached := make(chan struct{})
		server, _ := proxy(t, 200, func(call int32) {
			if call == 1 {
				close(reached)
				<-release
			}
		})
		t.Setenv(envBaseURL, server.URL)
		previous := defaultLoginFlows
		t.Cleanup(func() { defaultLoginFlows = previous })
		newCredential := ai.Credential{Type: ai.CredentialAPIKey, Key: "sk-new", Env: map[string]string{envBaseURL: server.URL}}
		defaultLoginFlows.LoginAPIKey = func(context.Context, ai.AuthInteraction, providerDefinition) (ai.Credential, error) {
			return newCredential, nil
		}
		provider, err := state.newProvider(definition, nil)
		if err != nil {
			t.Fatal(err)
		}
		publish := func(p ai.ModelsPublication) (bool, error) {
			if p.Update != nil {
				p.Update()
			}
			return true, nil
		}
		oldRefresh := make(chan error, 1)
		go func() {
			oldRefresh <- provider.RefreshModels(ai.RefreshModelsContext{
				Credential:   &ai.Credential{Type: ai.CredentialAPIKey, Key: "sk-old", Env: map[string]string{envBaseURL: server.URL}},
				AllowNetwork: true, Signal: context.Background(), Publish: publish,
			})
		}()
		<-reached
		credential, err := provider.Auth.APIKey.Login(context.Background(), ai.AuthInteraction{})
		if err != nil {
			t.Fatal(err)
		}
		close(release)
		if err := <-oldRefresh; err != nil {
			t.Fatal(err)
		}
		if _, err := state.resolveDefaultRuntimeAuth(nil); err == nil || !strings.Contains(err.Error(), "no credentials for litellm") {
			t.Fatalf("old refresh cached its credential: %v", err)
		}
		err = provider.RefreshModels(ai.RefreshModelsContext{Credential: &credential, AllowNetwork: true, Signal: context.Background(), Publish: publish})
		if err != nil {
			t.Fatal(err)
		}
		if auth, err := state.resolveDefaultRuntimeAuth(nil); err != nil || auth.APIKey != "sk-new" {
			t.Fatalf("auth=%+v err=%v", auth, err)
		}
	})

	t.Run("ignores legacy cache files without deleting them", func(t *testing.T) {
		state, definition := startup(t)
		t.Setenv(envOffline, "1")
		cache := filepath.Join(agentDir(), "litellm-models.json")
		legacy := `{"models":[{"id":"legacy-model"}]}`
		if err := os.WriteFile(cache, []byte(legacy), 0o600); err != nil {
			t.Fatal(err)
		}
		provider, err := state.newProvider(definition, state.seedModels(definition))
		if err != nil {
			t.Fatal(err)
		}
		if _, err := refresh(provider, storedEntry(t), false, false); err != nil {
			t.Fatal(err)
		}
		if data, _ := os.ReadFile(cache); string(data) != legacy {
			t.Fatalf("cache = %s", data)
		}
		if ids := modelIDs(t, provider); len(ids) != 0 {
			t.Fatalf("ids = %v", ids)
		}
	})

	t.Run("uses explicitly allowed insecure HTTP for a provider", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		writeFileIn(t, dir, "settings.json", `{"litellm":{"providers":{"litellm":{"baseUrl":"http://host.docker.internal","apiKey":"sk-local","allowInsecureHttp":true}}}}`)
		t.Setenv(envTimeout, "0")
		state, definition := startupIn(t, dir)
		provider, err := state.newProvider(definition, nil)
		if err != nil {
			t.Fatal(err)
		}
		result, err := provider.Auth.APIKey.Resolve(context.Background(), ai.APIKeyAuthInput{Ctx: ai.AuthContext{Env: processEnv}})
		if err != nil || result == nil {
			t.Fatalf("result=%v err=%v", result, err)
		}
		if result.Auth.APIKey != "sk-local" || result.Auth.BaseURL != "http://host.docker.internal" ||
			result.Env[envBaseURL] != "http://host.docker.internal" {
			t.Fatalf("result = %+v", result)
		}
	})
}

func TestSeedModels(t *testing.T) {
	t.Run("discovers at activation and fills the policy store", func(t *testing.T) {
		state, definition := startup(t)
		server, _ := proxy(t, 200, nil)
		t.Setenv(envBaseURL, server.URL)
		t.Setenv(envAPIKey, "sk-test")
		seed := state.seedModels(definition)
		if len(seed) != 1 || seed[0].ID != "fresh-model" || seed[0].Provider != "litellm" || seed[0].BaseURL != server.URL+"/v1" {
			t.Fatalf("seed = %+v", seed)
		}
		if _, _, ok := state.policies.policyFor("litellm", "fresh-model"); !ok {
			t.Fatal("policy store not filled")
		}
		if _, err := os.Stat(filepath.Join(agentDir(), policiesFilename)); err != nil {
			t.Fatalf("sidecar not persisted: %v", err)
		}
	})
	t.Run("is skipped without a request when discovery is disabled or credentials are missing", func(t *testing.T) {
		for name, setup := range map[string]func(*testing.T){
			"offline":     func(t *testing.T) { t.Setenv(envOffline, "1") },
			"zero":        func(t *testing.T) { t.Setenv(envTimeout, "0") },
			"PI_OFFLINE":  func(t *testing.T) { t.Setenv(envPIOffline, "1") },
			"PIG_OFFLINE": func(t *testing.T) { t.Setenv(envPIGOffline, "1") },
			"no key":      func(t *testing.T) { os.Unsetenv(envAPIKey) },
		} {
			t.Run(name, func(t *testing.T) {
				state, definition := startup(t)
				diagnostics := collectDiagnostics(t)
				server, calls := proxy(t, 200, nil)
				t.Setenv(envBaseURL, server.URL)
				t.Setenv(envAPIKey, "sk-test")
				t.Setenv(envVerboseDiscovery, "1")
				setup(t)
				if seed := state.seedModels(definition); seed != nil || calls.Load() != 0 {
					t.Fatalf("seed=%v calls=%d", seed, calls.Load())
				}
				if len(*diagnostics) != 1 || !strings.Contains((*diagnostics)[0], "LiteLLM (litellm): startup discovery skipped (") {
					t.Fatalf("diagnostics = %q", *diagnostics)
				}
			})
		}
	})
	t.Run("stays quiet when skipped and not verbose", func(t *testing.T) {
		state, definition := startup(t)
		diagnostics := collectDiagnostics(t)
		if seed := state.seedModels(definition); seed != nil || len(*diagnostics) != 0 {
			t.Fatalf("seed=%v diagnostics=%q", seed, *diagnostics)
		}
	})
	t.Run("never runs an API key helper", func(t *testing.T) {
		state, definition := startup(t)
		marker := filepath.Join(t.TempDir(), "ran")
		t.Setenv(envAPIKeyHelper, "touch "+marker)
		t.Setenv(envBaseURL, "https://proxy.example")
		state.seedModels(definition)
		if _, err := os.Stat(marker); err == nil {
			t.Fatal("seeding executed the helper")
		}
	})
}

func TestExtensionFactory(t *testing.T) {
	t.Run("builds without settings", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv(envOffline, "1")
		if e := Extension(); e == nil || e.Name() != "litellm" {
			t.Fatalf("extension = %v", e)
		}
	})
	t.Run("builds with several providers and their policy sidecar", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		t.Setenv(envOffline, "1")
		writeFileIn(t, dir, "settings.json", `{"litellm":{"providers":{"litellm":{"baseUrl":"https://a.example","apiKey":"k"},"litellm-b":{"baseUrl":"https://b.example","apiKey":"k"}}}}`)
		if e := Extension(); e == nil {
			t.Fatal("nil extension")
		}
	})
}

func TestHelpers(t *testing.T) {
	t.Run("attemptedFallbacks parses like Number()", func(t *testing.T) {
		for value, want := range map[string]int{"1": 1, " 2 ": 2, "0": 0, "-1": 0, "1.5": 0, "x": 0, "": 0, "9007199254740993": 0} {
			if got := attemptedFallbacks(map[string]any{"X-LiteLLM-Attempted-Fallbacks": value}); got != want {
				t.Errorf("%q = %d, want %d", value, got, want)
			}
		}
		if attemptedFallbacks(nil) != 0 {
			t.Error("missing header")
		}
	})
	t.Run("requestBaseURL falls back to the placeholder", func(t *testing.T) {
		hermeticAgentDir(t)
		definition := getProviderDefinitions(nil)[0]
		if got := requestBaseURL(definition); got != defaultLiteLLMBaseURL+"/v1" {
			t.Fatalf("got %q", got)
		}
		t.Setenv(envBaseURL, "https://proxy.example/")
		if got := requestBaseURL(definition); got != "https://proxy.example/v1" {
			t.Fatalf("got %q", got)
		}
	})
	t.Run("missingCredentials names the fix per provider", func(t *testing.T) {
		if got := missingCredentials(providerDefinition{Name: "litellm"}); got != "no credentials for litellm. Run /login litellm or set env vars." {
			t.Fatal(got)
		}
		if got := missingCredentials(providerDefinition{Name: "alias"}); got != "no credentials for alias. Set its baseUrl and apiKey." {
			t.Fatal(got)
		}
	})
	t.Run("discoveryDisabledReason", func(t *testing.T) {
		hermeticAgentDir(t)
		if discoveryDisabledReason() != "" {
			t.Fatal("enabled by default")
		}
		t.Setenv(envTimeout, "0")
		if got := discoveryDisabledReason(); got != "LITELLM_DISCOVERY_TIMEOUT_MS=0" {
			t.Fatal(got)
		}
		t.Setenv(envOffline, "1")
		if got := discoveryDisabledReason(); got != "LITELLM_OFFLINE=1" {
			t.Fatal(got)
		}
	})
}
