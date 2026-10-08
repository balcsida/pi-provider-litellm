package litellm

// Ports the startup cases of tests/index.test.ts as unit tests of the Go factory pieces.

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/MichaelKinsy/PiG/ai"
	sdk "github.com/MichaelKinsy/PiG/extensions/sdk"
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

func TestRuntimeAuthFrom(t *testing.T) {
	definition := providerDefinition{Name: "litellm", BaseURL: "https://registered.example", Headers: map[string]any{"X-Team": "t"}}
	hermeticAgentDir(t)
	t.Run("uses the host's auth baseUrl and headers", func(t *testing.T) {
		auth, err := runtimeAuthFrom(definition, map[string]any{"auth": map[string]any{
			"apiKey": "k", "baseUrl": "https://host.example/v1", "headers": map[string]any{"X-Host": "h", "skip": 3.0},
		}})
		if err != nil || auth == nil || auth.BaseURL != "https://host.example" || auth.APIKey != "k" || len(auth.Headers) != 1 || auth.Headers["X-Host"] != "h" {
			t.Fatalf("auth = %+v err = %v", auth, err)
		}
	})
	t.Run("falls back to the env base URL", func(t *testing.T) {
		auth, err := runtimeAuthFrom(definition, map[string]any{"auth": map[string]any{"apiKey": "k"}, "env": map[string]any{envBaseURL: "https://env.example"}})
		if err != nil || auth == nil || auth.BaseURL != "https://env.example" {
			t.Fatalf("auth = %+v err = %v", auth, err)
		}
	})
	t.Run("falls back to the provider's registered baseUrl and headers", func(t *testing.T) {
		auth, err := runtimeAuthFrom(definition, map[string]any{"auth": map[string]any{"apiKey": "k"}})
		if err != nil || auth == nil || auth.BaseURL != "https://registered.example" || auth.Headers["X-Team"] != "t" {
			t.Fatalf("auth = %+v err = %v", auth, err)
		}
	})
	t.Run("has no auth without a key and rejects a placeholder root", func(t *testing.T) {
		if auth, err := runtimeAuthFrom(definition, map[string]any{"auth": map[string]any{}}); auth != nil || err != nil {
			t.Fatalf("auth = %+v err = %v", auth, err)
		}
		bare := providerDefinition{Name: "litellm"}
		if _, err := runtimeAuthFrom(bare, map[string]any{"auth": map[string]any{"apiKey": "k"}}); err == nil || !strings.Contains(err.Error(), "placeholder") {
			t.Fatalf("err = %v", err)
		}
	})
}

func TestResolveRootPrecedence(t *testing.T) {
	hermeticAgentDir(t)
	state, definition := startup(t)
	definition.UseDefaultEnv = true
	t.Setenv(envBaseURL, "https://env.example")
	oauthCredential := &ai.Credential{Type: ai.CredentialOAuth, Access: "tok", Extra: map[string]json.RawMessage{"baseUrl": json.RawMessage(`"https://sso.example"`)}}

	t.Run("a credential decides, ahead of a request root", func(t *testing.T) {
		if root, err := state.resolveRoot(definition, oauthCredential, "https://request.example", ""); err != nil || root != "https://sso.example" {
			t.Fatalf("root = %q err = %v", root, err)
		}
	})
	t.Run("an explicit request root outranks the remembered OAuth root and the environment", func(t *testing.T) {
		state.oauthRoots[definition.Name] = oauthRuntimeRoot{apiKey: "tok", root: "https://remembered.example"}
		if root, err := state.resolveRoot(definition, nil, "https://request.example/", "tok"); err != nil || root != "https://request.example" {
			t.Fatalf("root = %q err = %v", root, err)
		}
	})
	t.Run("the remembered OAuth root needs the same API key", func(t *testing.T) {
		if root, _ := state.resolveRoot(definition, nil, "", "tok"); root != "https://remembered.example" {
			t.Fatalf("root = %q", root)
		}
		if root, _ := state.resolveRoot(definition, nil, "", "other"); root != "https://env.example" {
			t.Fatalf("a different key must not reuse the remembered root, got %q", root)
		}
	})
	t.Run("falls back to configuration", func(t *testing.T) {
		delete(state.oauthRoots, definition.Name)
		if root, _ := state.resolveRoot(definition, nil, "", ""); root != "https://env.example" {
			t.Fatalf("root = %q", root)
		}
	})
}

func TestOAuthToAuthPinsTheCredentialRoot(t *testing.T) {
	hermeticAgentDir(t)
	previous := defaultLoginFlows
	defaultLoginFlows = loginHooks{
		LoginOAuth: func(context.Context, ai.AuthInteraction, providerDefinition) (ai.Credential, error) {
			return ai.Credential{}, nil
		},
		Refresh: func(_ context.Context, credential ai.Credential, _ providerDefinition) (ai.Credential, error) {
			return credential, nil
		},
	}
	t.Cleanup(func() { defaultLoginFlows = previous })
	state, definition := startup(t)
	definition.EnableOAuth = true
	provider, err := state.newProvider(definition, nil)
	if err != nil {
		t.Fatal(err)
	}
	if provider.Auth.OAuth == nil {
		t.Fatal("OAuth is not offered")
	}
	credential := ai.Credential{Type: ai.CredentialOAuth, Access: "tok", Extra: map[string]json.RawMessage{"baseUrl": json.RawMessage(`"https://sso.example/"`)}}
	resolved, err := provider.Auth.OAuth.ToAuth(credential)
	if err != nil {
		t.Fatal(err)
	}
	if resolved.APIKey != "tok" || resolved.BaseURL != "https://sso.example" {
		t.Fatalf("resolved = %+v, the request host must be pinned to the credential root", resolved)
	}
	if got := state.oauthRoots[definition.Name]; got.apiKey != "tok" || got.root != "https://sso.example" {
		t.Fatalf("remembered = %+v", got)
	}
	for name, bad := range map[string]ai.Credential{
		"no token":    {Type: ai.CredentialOAuth},
		"no root":     {Type: ai.CredentialOAuth, Access: "tok"},
		"placeholder": {Type: ai.CredentialOAuth, Access: "tok", Extra: map[string]json.RawMessage{"baseUrl": json.RawMessage(`"https://LiteLLM.Example.com"`)}},
	} {
		if _, err := provider.Auth.OAuth.ToAuth(bad); err == nil {
			t.Errorf("%s: ToAuth accepted the credential", name)
		}
	}
}

func TestNotifyMcp(t *testing.T) {
	state, _ := startup(t)
	t.Run("reports on the diagnostic stream when there is no terminal", func(t *testing.T) {
		diagnostics := collectDiagnostics(t)
		previous := stderrIsTerminal
		stderrIsTerminal = func() bool { return false }
		t.Cleanup(func() { stderrIsTerminal = previous })
		state.notifyMcp("hello \n", "info")
		if len(*diagnostics) != 1 || (*diagnostics)[0] != "hello" || len(state.pendingMessages) != 0 {
			t.Fatalf("diagnostics = %q pending = %v", *diagnostics, state.pendingMessages)
		}
	})
	t.Run("buffers while the terminal is starting", func(t *testing.T) {
		diagnostics := collectDiagnostics(t)
		previous := stderrIsTerminal
		stderrIsTerminal = func() bool { return true }
		t.Cleanup(func() { stderrIsTerminal = previous })
		state.notifyMcp("early", "warning")
		if len(*diagnostics) != 0 || len(state.pendingMessages) != 1 || state.pendingMessages[0] != (pendingMessage{"early", "warning"}) {
			t.Fatalf("diagnostics = %q pending = %v", *diagnostics, state.pendingMessages)
		}
	})
	t.Run("stops buffering once the session has started", func(t *testing.T) {
		diagnostics := collectDiagnostics(t)
		previous := stderrIsTerminal
		stderrIsTerminal = func() bool { return true }
		t.Cleanup(func() { stderrIsTerminal = previous })
		state.sessionStarted = true
		state.notifyMcp("late", "info")
		if len(*diagnostics) != 1 || (*diagnostics)[0] != "late" {
			t.Fatalf("diagnostics = %q", *diagnostics)
		}
	})
}

// ---- hook wiring: setupHooks over a recording registrar and a fake host ----

type recordedCommand struct {
	description string
	completions func(prefix string) ([]sdk.AutocompleteItem, error)
	handler     func(h hookHost, args string) error
}

// recordingHooks is a hookRegistrar that keeps registrations in order.
type recordingHooks struct {
	events   map[string][]hookFunc
	commands map[string]recordedCommand
	tools    []string
}

func (r *recordingHooks) OnEvent(eventName string, handler hookFunc) {
	if r.events == nil {
		r.events = map[string][]hookFunc{}
	}
	r.events[eventName] = append(r.events[eventName], handler)
}

func (r *recordingHooks) Command(name, description string, completions func(string) ([]sdk.AutocompleteItem, error), handler func(hookHost, string) error) {
	if r.commands == nil {
		r.commands = map[string]recordedCommand{}
	}
	r.commands[name] = recordedCommand{description, completions, handler}
}

func (r *recordingHooks) RegisterTool(definition sdk.ToolDefinition) {
	r.tools = append(r.tools, definition.Name)
}

// emit runs the handlers of an event in registration order and returns what each one answered.
func (r *recordingHooks) emit(t *testing.T, eventName string, host hookHost, data map[string]any) []any {
	t.Helper()
	var results []any
	for _, handler := range r.events[eventName] {
		result, err := handler(host, data)
		if err != nil {
			t.Fatalf("%s handler: %v", eventName, err)
		}
		results = append(results, result)
	}
	return results
}

// only runs the index-th handler of an event.
func (r *recordingHooks) only(t *testing.T, eventName string, index int, host hookHost, data map[string]any) any {
	t.Helper()
	result, err := r.events[eventName][index](host, data)
	if err != nil {
		t.Fatalf("%s handler %d: %v", eventName, index, err)
	}
	return result
}

// fakeHookHost is a hookHost: the budget fake for the UI and credentials, the MCP fake for registrations.
type fakeHookHost struct {
	*fakeBudgetHost
	*fakeRegistrar
	model      string
	sessionID  string
	sessionErr error
	models     map[string]map[string]any
	finds      int
	requestCtx context.Context
}

func newFakeHookHost() *fakeHookHost {
	return &fakeHookHost{
		fakeBudgetHost: &fakeBudgetHost{ui: true, provider: providerName, auths: map[string]*types.LiteLLMRuntimeAuth{}},
		fakeRegistrar:  newFakeRegistrar(),
		model:          "some-model", sessionID: "session-1", models: map[string]map[string]any{}, requestCtx: context.Background(),
	}
}

func (h *fakeHookHost) Model() string                   { return h.model }
func (h *fakeHookHost) GetSessionID() (string, error)   { return h.sessionID, h.sessionErr }
func (h *fakeHookHost) RequestContext() context.Context { return h.requestCtx }
func (h *fakeHookHost) FindModel(provider, modelID string) map[string]any {
	h.finds++
	return h.models[provider+"\x00"+modelID]
}

// wire builds the state from the settings file and registers every handler into a recording registrar.
func wire(t *testing.T, settings string) (*recordingHooks, *extensionState, *fakeHookHost) {
	t.Helper()
	dir := hermeticAgentDir(t)
	if settings == "" {
		settings = `{"litellm":{}}`
	}
	writeFileIn(t, dir, "settings.json", settings)
	parsed := readGlobalLiteLLMSettings()
	state := newExtensionState(parsed, getProviderDefinitions(parsed), newPolicyStore(filepath.Join(dir, policiesFilename)))
	hooks := &recordingHooks{}
	setupHooks(hooks, state)
	return hooks, state, newFakeHookHost()
}

func costHeaders() map[string]any {
	return map[string]any{"x-litellm-response-cost": "0.5"}
}

func TestHookRegistration(t *testing.T) {
	counts := func(hooks *recordingHooks) map[string]int {
		result := map[string]int{}
		for name, handlers := range hooks.events {
			result[name] = len(handlers)
		}
		return result
	}

	t.Run("registers cost tracking and session grouping handlers", func(t *testing.T) {
		hooks, _, _ := wire(t, "")
		for _, name := range []string{sdk.EventBeforeProviderRequest, sdk.EventBeforeProviderHeaders, sdk.EventAfterProviderResponse, sdk.EventMessageEnd} {
			if counts(hooks)[name] == 0 {
				t.Errorf("no %s handler", name)
			}
		}
	})

	t.Run("registers budget hooks after cost tracking, and nothing when budget.enabled is false", func(t *testing.T) {
		hooks, _, host := wire(t, "")
		if _, ok := hooks.commands["litellm-budget"]; !ok {
			t.Fatal("no litellm-budget command")
		}
		got := counts(hooks)
		if got[sdk.EventTurnEnd] != 1 || got[sdk.EventModelSelect] != 1 || got[sdk.EventAfterProviderResponse] != 3 {
			t.Fatalf("handler counts = %v", got)
		}
		// Index 0 is the cost handler: it captures the header, and the cost lands on the message at message_end.
		hooks.only(t, sdk.EventAfterProviderResponse, 0, host, map[string]any{"status": 200.0, "headers": costHeaders()})
		message := map[string]any{"role": "assistant", "provider": providerName, "usage": map[string]any{"input": 1.0, "output": 1.0}}
		result, _ := hooks.only(t, sdk.EventMessageEnd, 0, host, map[string]any{"message": message}).(map[string]any)
		if result == nil || totalOf(t, result["message"].(map[string]any)) != 0.5 {
			t.Fatalf("message_end[0] = %v", result)
		}

		disabled, _, _ := wire(t, `{"litellm":{"budget":{"enabled":false}}}`)
		got = counts(disabled)
		if _, ok := disabled.commands["litellm-budget"]; ok || got[sdk.EventTurnEnd] != 0 || got[sdk.EventModelSelect] != 0 {
			t.Fatalf("budget registered while disabled: %v %v", disabled.commands, got)
		}
	})

	t.Run("runs the cost, budget and fallback after_provider_response handlers in that order", func(t *testing.T) {
		hooks, _, host := wire(t, "")
		hooks.emit(t, sdk.EventSessionStart, host, nil)
		data := map[string]any{"status": 200.0, "headers": map[string]any{
			"x-litellm-response-cost": "0.5", "x-litellm-key-spend": "1", "x-litellm-key-max-budget": "10", "x-litellm-attempted-fallbacks": "1",
		}}
		effect := func(index int) (cost bool, statuses []string, notices int) {
			h := newFakeHookHost()
			hooks.emit(t, sdk.EventSessionStart, h, nil)
			hooks.only(t, sdk.EventAfterProviderResponse, index, h, data)
			message := map[string]any{"role": "assistant", "provider": providerName, "usage": map[string]any{"input": 1.0}}
			result, _ := hooks.only(t, sdk.EventMessageEnd, 0, h, map[string]any{"message": message}).(map[string]any)
			return result != nil, h.texts(), len(h.notifications())
		}
		if cost, statuses, notices := effect(0); !cost || len(statuses) != 0 || notices != 0 {
			t.Errorf("handler 0 is not the cost handler: cost %v statuses %q notices %d", cost, statuses, notices)
		}
		if cost, statuses, notices := effect(1); cost || len(statuses) != 1 || !strings.Contains(statuses[0], "key $1/$10") || notices != 0 {
			t.Errorf("handler 1 is not the budget handler: cost %v statuses %q notices %d", cost, statuses, notices)
		}
		if cost, statuses, notices := effect(2); cost || len(statuses) != 0 || notices != 1 {
			t.Errorf("handler 2 is not the fallback warning: cost %v statuses %q notices %d", cost, statuses, notices)
		}
	})

	t.Run("attaches the cost before think tags are normalized", func(t *testing.T) {
		hooks, state, host := wire(t, "")
		model := discoveredModel("kimi")
		model.LiteLLMPolicy = &types.LiteLLMModelPolicy{NormalizeThinkTags: true}
		if err := state.policies.replace(providerName, []types.DiscoveredModel{model}); err != nil {
			t.Fatal(err)
		}
		hooks.only(t, sdk.EventAfterProviderResponse, 0, host, map[string]any{"status": 200.0, "headers": costHeaders()})
		message := map[string]any{"role": "assistant", "provider": providerName, "model": "kimi",
			"content": []any{map[string]any{"type": "text", "text": "<think>plan</think>answer"}},
			"usage":   map[string]any{"input": 1.0, "output": 1.0}}
		// The host hands each handler the message the previous one returned.
		for index := range hooks.events[sdk.EventMessageEnd] {
			if updated, ok := hooks.only(t, sdk.EventMessageEnd, index, host, map[string]any{"message": message}).(map[string]any); ok {
				message = updated["message"].(map[string]any)
			}
		}
		content := message["content"].([]any)
		if len(content) != 2 || content[0].(map[string]any)["type"] != "thinking" || totalOf(t, message) != 0.5 {
			t.Fatalf("message = %v", message)
		}
	})

	t.Run("registers the session, MCP and Skills handlers in the TypeScript order", func(t *testing.T) {
		hooks, _, _ := wire(t, "")
		got := counts(hooks)
		if got[sdk.EventSessionStart] != 3 || got[sdk.EventSessionShutdown] != 2 || got[sdk.EventBeforeAgentStart] != 3 {
			t.Fatalf("handler counts = %v", got)
		}
		if len(hooks.tools) != 3 {
			t.Fatalf("tools = %v", hooks.tools)
		}
	})
}

func TestHookDataKeysAndScoping(t *testing.T) {
	strictPayload := func() map[string]any {
		return map[string]any{"messages": []any{
			map[string]any{"role": "assistant", "content": nil, "tool_calls": []any{map[string]any{"id": "call_1", "type": "function"}}},
			map[string]any{"role": "tool", "tool_call_id": "call_1", "content": []any{map[string]any{"type": "text", "text": "tool output"}}},
		}}
	}
	seed := func(t *testing.T, state *extensionState, id string, policy *types.LiteLLMModelPolicy) {
		t.Helper()
		model := discoveredModel(id)
		model.LiteLLMPolicy = policy
		if err := state.policies.replace(providerName, []types.DiscoveredModel{model}); err != nil {
			t.Fatal(err)
		}
	}

	t.Run("before_provider_request reads data[payload] and returns the replacement payload", func(t *testing.T) {
		hooks, state, host := wire(t, "")
		seed(t, state, "kimi-k2.6", &types.LiteLLMModelPolicy{NormalizeStrictToolMessages: true})
		host.model = "kimi-k2.6"
		host.models[providerName+"\x00kimi-k2.6"] = map[string]any{"api": "openai-completions"}
		payload := strictPayload()
		result, _ := hooks.only(t, sdk.EventBeforeProviderRequest, 0, host, map[string]any{"payload": payload}).(map[string]any)
		messages, _ := result["messages"].([]any)
		if len(messages) != 2 || messages[0].(map[string]any)["content"] != "" || messages[1].(map[string]any)["content"] != "tool output" {
			t.Fatalf("result = %v", result)
		}
		if payload["messages"].([]any)[0].(map[string]any)["content"] != nil {
			t.Fatal("the input payload was mutated")
		}
	})

	t.Run("leaves strict-schema tool messages untouched for non-Moonshot models", func(t *testing.T) {
		hooks, _, host := wire(t, "")
		host.model = "anthropic/claude-3-5-sonnet"
		if result := hooks.only(t, sdk.EventBeforeProviderRequest, 0, host, map[string]any{"payload": strictPayload()}); result != nil {
			t.Fatalf("result = %v", result)
		}
	})

	t.Run("before_provider_request ignores a payload that is not an object", func(t *testing.T) {
		hooks, _, host := wire(t, "")
		if result := hooks.only(t, sdk.EventBeforeProviderRequest, 0, host, map[string]any{"payload": "text"}); result != nil {
			t.Fatalf("result = %v", result)
		}
	})

	t.Run("before_provider_headers adds the session id to data[headers] and returns them", func(t *testing.T) {
		hooks, _, host := wire(t, "")
		result, _ := hooks.only(t, sdk.EventBeforeProviderHeaders, 0, host, map[string]any{"headers": map[string]any{"x-a": "1"}}).(map[string]any)
		if result["x-a"] != "1" || result[sessionHeader] != "session-1" {
			t.Fatalf("headers = %v", result)
		}
		host.sessionErr = errors.New("no session")
		if result := hooks.only(t, sdk.EventBeforeProviderHeaders, 0, host, map[string]any{"headers": map[string]any{}}); result != nil {
			t.Fatalf("headers without a session = %v", result)
		}
	})

	t.Run("before_agent_start reads data[systemPrompt] and returns the extended prompt", func(t *testing.T) {
		server, _ := skillsServer(t, scriptedReply{200, `{"plugins":[{"name":"terraform","description":"Terraform conventions"}]}`})
		hooks, state, host := wire(t, "")
		host.setAuth(providerName, &types.LiteLLMRuntimeAuth{BaseURL: server.URL, APIKey: "sk-test", AllowInsecureHTTP: true})
		resetSkillsCache()
		var prompt string
		for _, result := range hooks.emit(t, sdk.EventBeforeAgentStart, host, map[string]any{"systemPrompt": "Base prompt"}) {
			if updated, ok := result.(map[string]any); ok {
				prompt, _ = updated["systemPrompt"].(string)
			}
		}
		if !strings.HasPrefix(prompt, "Base prompt\n\n") || !strings.Contains(prompt, "Terraform conventions") || state.defaultRuntimeAuth == nil {
			t.Fatalf("prompt = %q", prompt)
		}
	})

	t.Run("leaves every other provider untouched", func(t *testing.T) {
		hooks, state, host := wire(t, "")
		seed(t, state, "kimi-k2.6", &types.LiteLLMModelPolicy{NormalizeStrictToolMessages: true, NormalizeThinkTags: true})
		host.provider, host.model = "anthropic", "kimi-k2.6"
		host.models["anthropic\x00kimi-k2.6"] = map[string]any{"api": "openai-completions"}
		hooks.emit(t, sdk.EventSessionStart, host, nil)

		if result := hooks.only(t, sdk.EventBeforeProviderRequest, 0, host, map[string]any{"payload": strictPayload()}); result != nil {
			t.Errorf("payload rewritten for another provider: %v", result)
		}
		headers := map[string]any{}
		if result := hooks.only(t, sdk.EventBeforeProviderHeaders, 0, host, map[string]any{"headers": headers}); result != nil || len(headers) != 0 {
			t.Errorf("session header added for another provider: %v %v", result, headers)
		}
		response := map[string]any{"status": 200.0, "headers": map[string]any{
			"x-litellm-response-cost": "0.5", "x-litellm-key-spend": "1", "x-litellm-key-max-budget": "10", "x-litellm-attempted-fallbacks": "1",
		}}
		hooks.emit(t, sdk.EventAfterProviderResponse, host, response)
		if len(host.allStatuses()) != 0 || len(host.notifications()) != 0 {
			t.Errorf("another provider's response reached the footer or fallback warning: %q %v", host.allStatuses(), host.notifications())
		}
		message := map[string]any{"role": "assistant", "provider": "anthropic", "model": "kimi-k2.6",
			"content": []any{map[string]any{"type": "text", "text": "<think>a</think>b"}}, "usage": map[string]any{"input": 1.0}}
		for _, result := range hooks.emit(t, sdk.EventMessageEnd, host, map[string]any{"message": message}) {
			if result != nil {
				t.Errorf("message_end rewrote another provider's message: %v", result)
			}
		}
		if host.finds != 0 {
			t.Errorf("the host registry was queried %d times for another provider's message", host.finds)
		}
	})

	t.Run("message_end does not query the registry for a provider that is not configured", func(t *testing.T) {
		hooks, _, host := wire(t, "")
		message := map[string]any{"role": "assistant", "provider": "openai", "model": "gpt-5", "content": []any{}}
		hooks.only(t, sdk.EventMessageEnd, 1, host, map[string]any{"message": message})
		if host.finds != 0 {
			t.Fatalf("finds = %d", host.finds)
		}
		message["provider"] = providerName
		hooks.only(t, sdk.EventMessageEnd, 1, host, map[string]any{"message": message})
		if host.finds != 1 {
			t.Fatalf("a configured provider's message did not query the registry: finds = %d", host.finds)
		}
	})
}

func TestSessionStateBuffering(t *testing.T) {
	hooks, state, host := wire(t, "")
	diagnostics := collectDiagnostics(t)
	previous := stderrIsTerminal
	stderrIsTerminal = func() bool { return true }
	t.Cleanup(func() { stderrIsTerminal = previous })

	state.notifyMcp("early", "warning")
	if len(*diagnostics) != 0 || len(host.notifications()) != 0 {
		t.Fatalf("a message was shown before the session started: %q %v", *diagnostics, host.notifications())
	}
	hooks.only(t, sdk.EventSessionStart, 0, host, nil)
	if got := host.notifications(); len(got) != 1 || got[0] != (budgetNotice{"early", "warning"}) || len(state.pendingMessages) != 0 {
		t.Fatalf("notices %v, pending %v", got, state.pendingMessages)
	}
	state.notifyMcp("late", "info")
	if got := host.notifications(); len(got) != 2 || got[1] != (budgetNotice{"late", "info"}) {
		t.Fatalf("notices %v", got)
	}
	hooks.only(t, sdk.EventSessionShutdown, 0, host, nil)
	state.notifyMcp("after", "info")
	// Shutdown returns to the starting state: the next session buffers again.
	if len(*diagnostics) != 0 || len(state.pendingMessages) != 1 {
		t.Fatalf("diagnostics = %q, pending %v", *diagnostics, state.pendingMessages)
	}
}

func TestBudgetWiring(t *testing.T) {
	eventually := func(t *testing.T, host *fakeHookHost, want string) {
		t.Helper()
		deadline := time.Now().Add(5 * time.Second)
		for time.Now().Before(deadline) {
			if last, ok := host.lastStatus(); ok && last == want {
				return
			}
			time.Sleep(5 * time.Millisecond)
		}
		t.Fatalf("status never became %q; statuses %q", want, host.allStatuses())
	}
	withAuth := func(t *testing.T) (*recordingHooks, *fakeHookHost) {
		mockBudgetProxy(t, budgetPersonalRoutes(t))
		hooks, _, host := wire(t, "")
		auth := budgetAuth()
		host.setAuth(providerName, &auth)
		return hooks, host
	}
	const personal = "LiteLLM key $0.15/$10 · user $0.34/$100"

	t.Run("model_select tracks the provider named in data[model], not the host's current one", func(t *testing.T) {
		hooks, host := withAuth(t)
		host.setProvider("openai") // the host still reports the previous model
		hooks.only(t, sdk.EventModelSelect, 0, host, map[string]any{"type": "model_select", "model": map[string]any{"provider": providerName, "id": "m"}})
		eventually(t, host, personal)
		host.setProvider(providerName)
		hooks.only(t, sdk.EventModelSelect, 0, host, map[string]any{"type": "model_select", "model": map[string]any{"provider": "openai", "id": "m"}})
		eventually(t, host, "")
		hooks.emit(t, sdk.EventSessionShutdown, host, nil)
	})

	t.Run("before_agent_start still picks up a provider switch without model_select", func(t *testing.T) {
		hooks, host := withAuth(t)
		hooks.only(t, sdk.EventBeforeAgentStart, 0, host, map[string]any{"systemPrompt": ""})
		eventually(t, host, personal)
		hooks.emit(t, sdk.EventSessionShutdown, host, nil)
	})

	t.Run("polls through the provider auth with the configured headers", func(t *testing.T) {
		recorder := mockBudgetProxy(t, budgetPersonalRoutes(t), budgetExpect{"env-key", "g1"})
		hooks, state, host := wire(t, "")
		t.Setenv(envHeaders, `{"x-gateway":"g1"}`)
		auth, err := runtimeAuthFrom(state.definitions[0], map[string]any{"auth": map[string]any{"apiKey": "env-key", "baseUrl": "https://proxy.example.com"}})
		if err != nil || auth == nil {
			t.Fatalf("auth = %v, %v", auth, err)
		}
		host.setAuth(providerName, auth)
		if err := hooks.commands["litellm-budget"].handler(host, ""); err != nil {
			t.Fatal(err)
		}
		notices := host.notifications()
		if recorder.count("/key/info") != 1 || len(notices) != 1 || notices[0].level != "info" {
			t.Fatalf("requests %v, notices %v", recorder.paths(), notices)
		}
	})
}
