package litellm

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/MichaelKinsy/PiG/ai"
)

func contextEnv(values map[string]string) ai.AuthContext {
	return ai.AuthContext{
		Env: func(name string) (string, bool) {
			value, ok := values[name]
			return value, ok && strings.TrimSpace(value) != ""
		},
		FileExists: func(string) bool { return false },
	}
}

func defaultDefinition(t *testing.T) providerDefinition {
	t.Helper()
	return getProviderDefinitions(readGlobalLiteLLMSettings())[0]
}

func resolveWith(t *testing.T, definition providerDefinition, values map[string]string, credential *ai.Credential) (*ai.AuthResult, error) {
	t.Helper()
	auth := createProviderAuth(definition, nil, nil, nil)
	return auth.APIKey.Resolve(context.Background(), ai.APIKeyAuthInput{Ctx: contextEnv(values), Credential: credential})
}

func mustResolve(t *testing.T, definition providerDefinition, values map[string]string, credential *ai.Credential) *ai.AuthResult {
	t.Helper()
	result, err := resolveWith(t, definition, values, credential)
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if result == nil {
		t.Fatal("resolve returned no auth")
	}
	return result
}

func TestPlaceholderHost(t *testing.T) {
	for host, want := range map[string]bool{"litellm.example.com": true, "litellm.example.com.": true, "proxy.example.com": false} {
		if got := isPlaceholderHost(host); got != want {
			t.Errorf("isPlaceholderHost(%q) = %v", host, got)
		}
	}
	if _, err := requireCredentialRoot("", "litellm"); err == nil || !strings.Contains(err.Error(), "no LiteLLM base URL for litellm") {
		t.Errorf("empty root error = %v", err)
	}
	if _, err := requireCredentialRoot(defaultLiteLLMBaseURL, "litellm"); err == nil || !strings.Contains(err.Error(), "placeholder LiteLLM base URL") {
		t.Errorf("placeholder error = %v", err)
	}
	if root, err := requireCredentialRoot("https://proxy.example.com", "litellm"); err != nil || root != "https://proxy.example.com" {
		t.Errorf("root = %q, %v", root, err)
	}
}

func TestResolveCredentialRoot(t *testing.T) {
	hermeticAgentDir(t)
	t.Setenv(envBaseURL, "https://env.example.com")
	definition := providerDefinition{Name: providerName, UseDefaultEnv: true, BaseURL: "https://configured.example.com"}
	oauth := &credentialInfo{Type: "oauth", BaseURL: "https://oauth.example.com/v1/"}
	apiKey := &credentialInfo{Type: "api_key", Env: map[string]string{envBaseURL: "https://key.example.com"}}

	cases := []struct {
		name       string
		definition providerDefinition
		credential *credentialInfo
		request    string
		want       string
	}{
		{"oauth credential wins", definition, oauth, "https://request.example.com", "https://oauth.example.com"},
		{"api_key env wins", definition, apiKey, "", "https://key.example.com"},
		{"request over settings", definition, nil, "https://request.example.com/", "https://request.example.com"},
		{"settings over environment", definition, nil, "", "https://configured.example.com"},
		{"environment last", providerDefinition{Name: providerName, UseDefaultEnv: true}, nil, "", "https://env.example.com"},
		{"alias ignores environment", providerDefinition{Name: "alias"}, nil, "", ""},
	}
	for _, tc := range cases {
		got, err := resolveCredentialRoot(tc.definition, tc.credential, tc.request)
		if err != nil || got != tc.want {
			t.Errorf("%s: got %q, %v; want %q", tc.name, got, err, tc.want)
		}
	}
	if _, err := resolveCredentialRoot(providerDefinition{Name: "alias", BaseURL: "http://insecure.example.com"}, nil, ""); err == nil {
		t.Error("insecure root must be rejected")
	}
}

func TestProviderAuthCheck(t *testing.T) {
	ctx := context.Background()
	check := func(t *testing.T, definition providerDefinition, values map[string]string, credential *ai.Credential) *ai.AuthCheck {
		t.Helper()
		result, err := createProviderAuth(definition, nil, nil, nil).APIKey.Check(ctx, ai.APIKeyAuthInput{Ctx: contextEnv(values), Credential: credential})
		if err != nil {
			t.Fatal(err)
		}
		return result
	}

	t.Run("checks command-backed auth without executing the helper", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		helper, runs := writeHelper(t, dir, []string{"helper-key"}, "h.sh")
		got := check(t, defaultDefinition(t), map[string]string{envBaseURL: "https://proxy.example.com", envAPIKeyHelper: helper}, nil)
		if got == nil || got.Type != ai.CredentialAPIKey || got.Source != envAPIKeyHelper || runs() != 0 {
			t.Fatalf("got %+v runs=%d", got, runs())
		}
	})

	t.Run("reports nothing without a base URL", func(t *testing.T) {
		hermeticAgentDir(t)
		if got := check(t, defaultDefinition(t), map[string]string{envAPIKey: "k"}, nil); got != nil {
			t.Fatalf("got %+v", got)
		}
	})

	t.Run("reports nothing without any key", func(t *testing.T) {
		hermeticAgentDir(t)
		if got := check(t, defaultDefinition(t), map[string]string{envBaseURL: "https://proxy.example.com"}, nil); got != nil {
			t.Fatalf("got %+v", got)
		}
	})

	t.Run("a stored credential with its own base URL", func(t *testing.T) {
		hermeticAgentDir(t)
		credential := &ai.Credential{Type: ai.CredentialAPIKey, Key: "k", Env: map[string]string{envBaseURL: "https://proxy.example.com"}}
		if got := check(t, defaultDefinition(t), nil, credential); got == nil || got.Source != "stored credential" {
			t.Fatalf("got %+v", got)
		}
	})

	t.Run("environment key", func(t *testing.T) {
		hermeticAgentDir(t)
		got := check(t, defaultDefinition(t), map[string]string{envBaseURL: "https://proxy.example.com", envAPIKey: "k"}, nil)
		if got == nil || got.Source != envAPIKey {
			t.Fatalf("got %+v", got)
		}
	})

	t.Run("configured key template resolved from the context", func(t *testing.T) {
		hermeticAgentDir(t)
		definition := providerDefinition{Name: "alias", BaseURL: "https://alias.example.com", APIKeyConfig: "$ALIAS_KEY"}
		if got := check(t, definition, nil, nil); got != nil {
			t.Fatalf("unset template must not count: %+v", got)
		}
		if got := check(t, definition, map[string]string{"ALIAS_KEY": "k"}, nil); got == nil || got.Source != "$ALIAS_KEY" {
			t.Fatalf("got %+v", got)
		}
	})

	t.Run("gcloud ADC outranks the configured key, with no network call", func(t *testing.T) {
		hermeticAgentDir(t)
		calls, _ := tokenServer(t, http.StatusOK, "unused")
		t.Setenv(envGcloudTokenAuth, "1")
		t.Setenv(envGoogleCredentials, writeADCFile(t, authorizedUser(nil)))
		got := check(t, defaultDefinition(t), map[string]string{envBaseURL: "https://proxy.example.com", envAPIKey: "k"}, nil)
		if got == nil || got.Source != gcloudADCSource || calls.Load() != 0 {
			t.Fatalf("got %+v calls=%d", got, calls.Load())
		}
	})

	t.Run("a service-account ADC falls through to the environment key", func(t *testing.T) {
		hermeticAgentDir(t)
		captureDiagnostics(t)
		t.Setenv(envGcloudTokenAuth, "1")
		t.Setenv(envGoogleCredentials, writeADCFile(t, map[string]any{"type": "service_account"}))
		got := check(t, defaultDefinition(t), map[string]string{envBaseURL: "https://proxy.example.com", envAPIKey: "k"}, nil)
		if got == nil || got.Source != envAPIKey {
			t.Fatalf("got %+v", got)
		}
	})
}

func TestProviderAuthResolve(t *testing.T) {
	t.Run("uses explicitly allowed insecure HTTP for a provider", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		writeFileIn(t, dir, "settings.json", `{"litellm":{"providers":{"litellm":{"baseUrl":"http://host.docker.internal","apiKey":"sk-local","allowInsecureHttp":true}}}}`)
		got := mustResolve(t, defaultDefinition(t), nil, nil)
		if got.Auth.APIKey != "sk-local" || got.Auth.BaseURL != "http://host.docker.internal" || got.Env[envBaseURL] != "http://host.docker.internal" {
			t.Fatalf("got %+v", got)
		}
	})

	t.Run("resolves native auth from the injected context instead of process env", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv(envBaseURL, "https://process.example.com")
		t.Setenv(envAPIKey, "process-key")
		got := mustResolve(t, defaultDefinition(t), map[string]string{
			envBaseURL: "https://context.example.com",
			envAPIKey:  "context-key",
			envHeaders: `{"x-tenant":"context"}`,
		}, nil)
		if got.Auth.APIKey != "context-key" || got.Auth.BaseURL != "https://context.example.com" || got.Source != envAPIKey {
			t.Fatalf("got %+v", got)
		}
		if header := got.Auth.Headers["x-tenant"]; header == nil || *header != "context" {
			t.Fatalf("headers = %v", got.Auth.Headers)
		}
	})

	t.Run("executes only the helper supplied by the injected auth context", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		helper, runs := writeHelper(t, dir, []string{"context-helper-key"}, "h.sh")
		got := mustResolve(t, defaultDefinition(t), map[string]string{envBaseURL: "https://context.example.com", envAPIKeyHelper: helper, envAPIKey: "context-env-key"}, nil)
		if got.Auth.APIKey != "context-helper-key" || got.Source != envAPIKeyHelper || runs() != 1 {
			t.Fatalf("got %+v runs=%d", got, runs())
		}
	})

	t.Run("rejects shell expressions in helper commands", func(t *testing.T) {
		hermeticAgentDir(t)
		_, err := resolveWith(t, defaultDefinition(t), map[string]string{envBaseURL: "https://context.example.com", envAPIKeyHelper: "printf safe-key; printf injected-key"}, nil)
		if err == nil || !strings.Contains(err.Error(), "shell syntax is not supported") {
			t.Fatalf("err = %v", err)
		}
	})

	t.Run("preserves backslashes in helper executable paths", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		helper, _ := writeHelper(t, dir, []string{"backslash-key"}, `token\helper.sh`)
		got := mustResolve(t, defaultDefinition(t), map[string]string{envBaseURL: "https://context.example.com", envAPIKeyHelper: `"` + helper + `"`}, nil)
		if got.Auth.APIKey != "backslash-key" {
			t.Fatalf("got %+v", got)
		}
	})

	t.Run("resolves configured key templates from the injected auth context", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		lowerPriority, runs := writeHelper(t, dir, []string{"unexpected-helper-key"}, "h.sh")
		writeFileIn(t, dir, "settings.json", `{"litellm":{"providers":{"litellm":{"apiKey":"$CUSTOM_LITELLM_KEY"}}}}`)
		t.Setenv("CUSTOM_LITELLM_KEY", "process-configured-key")
		got := mustResolve(t, defaultDefinition(t), map[string]string{
			envBaseURL: "https://context.example.com", "CUSTOM_LITELLM_KEY": "context-configured-key",
			envAPIKeyHelper: lowerPriority, envAPIKey: "context-default-key",
		}, nil)
		if got.Auth.APIKey != "context-configured-key" || got.Source != "$CUSTOM_LITELLM_KEY" || runs() != 0 {
			t.Fatalf("got %+v runs=%d", got, runs())
		}
	})

	t.Run("returns nothing without a key", func(t *testing.T) {
		hermeticAgentDir(t)
		got, err := resolveWith(t, defaultDefinition(t), map[string]string{envBaseURL: "https://context.example.com"}, nil)
		if err != nil || got != nil {
			t.Fatalf("got %+v, %v", got, err)
		}
	})

	t.Run("a stored credential wins and exports its root", func(t *testing.T) {
		hermeticAgentDir(t)
		credential := &ai.Credential{Type: ai.CredentialAPIKey, Key: " sk-login ", Env: map[string]string{envBaseURL: "http://127.0.0.1:4000"}}
		got := mustResolve(t, defaultDefinition(t), map[string]string{envAPIKey: "ignored"}, credential)
		if got.Auth.APIKey != "sk-login" || got.Auth.BaseURL != "http://127.0.0.1:4000" || got.Source != "stored credential" {
			t.Fatalf("got %+v", got)
		}
	})

	t.Run("resolves opaque command-backed API keys for each request", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		helper, runs := writeHelper(t, dir, []string{"opaque-first", "opaque-second", "unexpected-third"}, "h.sh")
		credential := &ai.Credential{Type: ai.CredentialAPIKey, Key: "!" + helper, Env: map[string]string{envBaseURL: "https://proxy.example.com"}}
		first := mustResolve(t, defaultDefinition(t), nil, credential)
		second := mustResolve(t, defaultDefinition(t), nil, credential)
		if first.Auth.APIKey != "opaque-first" || second.Auth.APIKey != "opaque-second" || runs() != 2 {
			t.Fatalf("keys %q %q runs=%d", first.Auth.APIKey, second.Auth.APIKey, runs())
		}
	})

	t.Run("resolveAPIKeyAuth with executeHelpers false never runs a helper", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		helper, runs := writeHelper(t, dir, []string{"x"}, "h.sh")
		env := func(name string) (string, bool) {
			return map[string]string{envBaseURL: "https://proxy.example.com", envAPIKeyHelper: helper}[name], name == envBaseURL || name == envAPIKeyHelper
		}
		got, err := resolveAPIKeyAuth(context.Background(), defaultDefinition(t), env, nil, false)
		if runs() != 0 || err != nil || got != nil {
			t.Fatalf("got %+v, %v, runs=%d", got, err, runs())
		}
	})

	t.Run("an alias without an apiKey does not take the default env key", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv(envAPIKey, "default-key")
		alias := providerDefinition{Name: "litellm-alias", BaseURL: "https://alias.example.com"}
		got, err := resolveWith(t, alias, map[string]string{envAPIKey: "default-key"}, nil)
		if err != nil || got != nil {
			t.Fatalf("got %+v, %v", got, err)
		}
	})

	t.Run("mints the key from authorized_user ADC when enabled", func(t *testing.T) {
		hermeticAgentDir(t)
		tokenServer(t, http.StatusOK, "ya29.minted")
		t.Setenv(envGcloudTokenAuth, "1")
		t.Setenv(envGoogleCredentials, writeADCFile(t, authorizedUser(nil)))
		got := mustResolve(t, defaultDefinition(t), map[string]string{envBaseURL: "https://proxy.example.com", envAPIKey: "env-key"}, nil)
		if got.Auth.APIKey != "ya29.minted" || got.Source != gcloudADCSource {
			t.Fatalf("got %+v", got)
		}
	})

	t.Run("rejects a placeholder or insecure base URL", func(t *testing.T) {
		hermeticAgentDir(t)
		_, err := resolveWith(t, defaultDefinition(t), map[string]string{envBaseURL: defaultLiteLLMBaseURL, envAPIKey: "k"}, nil)
		if err == nil || !strings.Contains(err.Error(), "placeholder LiteLLM base URL") {
			t.Fatalf("placeholder err = %v", err)
		}
		_, err = resolveWith(t, defaultDefinition(t), map[string]string{envBaseURL: "http://insecure.example.com", envAPIKey: "k"}, nil)
		if err == nil || !strings.Contains(strings.ToLower(err.Error()), "https") {
			t.Fatalf("insecure err = %v", err)
		}
	})

	t.Run("fails when a key resolves but no root is configured", func(t *testing.T) {
		hermeticAgentDir(t)
		_, err := resolveWith(t, defaultDefinition(t), map[string]string{envAPIKey: "k"}, nil)
		if err == nil || !strings.Contains(err.Error(), "no LiteLLM base URL for litellm") {
			t.Fatalf("err = %v", err)
		}
	})
}

func TestProviderAuthResolveOAuthRoot(t *testing.T) {
	ssoAgentDir := func(t *testing.T) string {
		dir := hermeticAgentDir(t)
		writeFileIn(t, dir, "auth.json", `{"litellm":{"type":"oauth","access":"sk-sso","refresh":"","expires":9007199254740991,"baseUrl":"https://oauth.example.com"}}`)
		return dir
	}
	sessionKey := &ai.Credential{Type: ai.CredentialAPIKey, Key: "sk-sso"}

	t.Run("keeps the OAuth base URL when Pi re-resolves auth with the session's own token", func(t *testing.T) {
		ssoAgentDir(t)
		got := mustResolve(t, defaultDefinition(t), nil, sessionKey)
		if got.Auth.APIKey != "sk-sso" || got.Auth.BaseURL != "https://oauth.example.com" || got.Env[envBaseURL] != "https://oauth.example.com" || got.Source != "OAuth" {
			t.Fatalf("got %+v", got)
		}
	})

	t.Run("keeps this process's root after another process replaces the stored credential", func(t *testing.T) {
		ssoAgentDir(t)
		remembered := func() (string, string, bool) { return "sk-live", "https://live.example.com", true }
		auth := createProviderAuth(defaultDefinition(t), nil, nil, remembered)
		got, err := auth.APIKey.Resolve(context.Background(), ai.APIKeyAuthInput{Ctx: contextEnv(nil), Credential: &ai.Credential{Type: ai.CredentialAPIKey, Key: "sk-live"}})
		if err != nil || got == nil || got.Auth.BaseURL != "https://live.example.com" || got.Env[envBaseURL] != "https://live.example.com" {
			t.Fatalf("got %+v, %v", got, err)
		}
	})

	t.Run("clears the remembered root when API-key auth resolves", func(t *testing.T) {
		hermeticAgentDir(t)
		cleared := 0
		auth := createProviderAuth(defaultDefinition(t), func() { cleared++ }, nil, nil)
		credential := &ai.Credential{Type: ai.CredentialAPIKey, Key: "shared-key", Env: map[string]string{envBaseURL: "https://api-key.example.com"}}
		if _, err := auth.APIKey.Resolve(context.Background(), ai.APIKeyAuthInput{Ctx: contextEnv(nil), Credential: credential}); err != nil || cleared != 1 {
			t.Fatalf("err=%v cleared=%d", err, cleared)
		}
	})

	for name, baseURL := range map[string]string{"placeholder": defaultLiteLLMBaseURL, "insecure": "http://insecure.example.com"} {
		t.Run("still rejects a "+name+" base URL when the session's own token is re-resolved", func(t *testing.T) {
			ssoAgentDir(t)
			t.Setenv(envBaseURL, baseURL)
			// A configured base URL must fail on its own terms, never fall back to the stored OAuth root.
			if _, err := resolveWith(t, defaultDefinition(t), map[string]string{envBaseURL: baseURL}, sessionKey); err == nil {
				t.Fatal("expected rejection")
			}
		})
	}

	t.Run("a stored OAuth token that differs from the session's is not reused", func(t *testing.T) {
		ssoAgentDir(t)
		_, err := resolveWith(t, defaultDefinition(t), nil, &ai.Credential{Type: ai.CredentialAPIKey, Key: "sk-other"})
		if err == nil || !strings.Contains(err.Error(), "no LiteLLM base URL") {
			t.Fatalf("err = %v", err)
		}
	})
}

func TestCreateProviderAuthHooks(t *testing.T) {
	ctx := context.Background()

	t.Run("nil hooks offer no login and no OAuth", func(t *testing.T) {
		hermeticAgentDir(t)
		auth := createProviderAuth(defaultDefinition(t), nil, nil, nil)
		if auth.OAuth != nil || auth.APIKey == nil || auth.APIKey.Login != nil || auth.APIKey.Name != "LiteLLM API key" {
			t.Fatalf("auth = %+v", auth)
		}
	})

	t.Run("OAuth needs both its login and its refresh hook", func(t *testing.T) {
		hermeticAgentDir(t)
		partial := &loginHooks{LoginOAuth: func(context.Context, ai.AuthInteraction, providerDefinition) (ai.Credential, error) {
			return ai.Credential{}, nil
		}}
		if createProviderAuth(defaultDefinition(t), nil, partial, nil).OAuth != nil {
			t.Fatal("OAuth must not be offered without Refresh")
		}
	})

	t.Run("hooks are bracketed by Start and Complete and receive the definition", func(t *testing.T) {
		hermeticAgentDir(t)
		var events []string
		hooks := &loginHooks{
			Start:    func() { events = append(events, "start") },
			Complete: func() { events = append(events, "complete") },
			LoginAPIKey: func(_ context.Context, _ ai.AuthInteraction, definition providerDefinition) (ai.Credential, error) {
				events = append(events, "api:"+definition.Name)
				return ai.Credential{Type: ai.CredentialAPIKey, Key: "k"}, nil
			},
			LoginOAuth: func(_ context.Context, _ ai.AuthInteraction, _ providerDefinition) (ai.Credential, error) {
				return ai.Credential{}, errors.New("denied")
			},
			Refresh: func(_ context.Context, credential ai.Credential, _ providerDefinition) (ai.Credential, error) {
				credential.Access = "refreshed"
				return credential, nil
			},
		}
		auth := createProviderAuth(defaultDefinition(t), nil, hooks, nil)
		if credential, err := auth.APIKey.Login(ctx, ai.AuthInteraction{}); err != nil || credential.Key != "k" {
			t.Fatalf("api login = %+v, %v", credential, err)
		}
		if _, err := auth.OAuth.Login(ctx, ai.AuthInteraction{}, ai.LoginOptions{}); err == nil {
			t.Fatal("oauth login error must propagate")
		}
		if got := strings.Join(events, ","); got != "start,api:litellm,complete,start" {
			t.Fatalf("events = %s", got)
		}
		if auth.OAuth.Name != "LiteLLM SSO" || auth.OAuth.LoginLabel != "Sign in with LiteLLM SSO" {
			t.Fatalf("oauth = %+v", auth.OAuth)
		}
		refreshed, err := auth.OAuth.Refresh(ctx, ai.Credential{Access: "old"})
		if err != nil || refreshed.Access != "refreshed" || refreshed.Type != ai.CredentialOAuth {
			t.Fatalf("refresh = %+v, %v", refreshed, err)
		}
	})

	t.Run("aliases get no API-key login", func(t *testing.T) {
		hooks := &loginHooks{LoginAPIKey: func(context.Context, ai.AuthInteraction, providerDefinition) (ai.Credential, error) {
			return ai.Credential{}, nil
		}}
		auth := createProviderAuth(providerDefinition{Name: "alias", DisplayName: "Alias"}, nil, hooks, nil)
		if auth.APIKey.Login != nil || auth.OAuth != nil {
			t.Fatalf("auth = %+v", auth)
		}
	})

	t.Run("OAuth toAuth uses the access token and resolved headers", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv(envHeaders, `{"x-team":"a"}`)
		hooks := &loginHooks{
			LoginOAuth: func(context.Context, ai.AuthInteraction, providerDefinition) (ai.Credential, error) {
				return ai.Credential{}, nil
			},
			Refresh: func(context.Context, ai.Credential, providerDefinition) (ai.Credential, error) {
				return ai.Credential{}, nil
			},
		}
		auth := createProviderAuth(defaultDefinition(t), nil, hooks, nil)
		got, err := auth.OAuth.ToAuth(ai.Credential{Type: ai.CredentialOAuth, Access: "refreshed-token"})
		if err != nil || got.APIKey != "refreshed-token" || got.Headers["x-team"] == nil || *got.Headers["x-team"] != "a" {
			t.Fatalf("got %+v, %v", got, err)
		}
	})
}

func TestCredentialInfoFromAI(t *testing.T) {
	credential := ai.Credential{Type: ai.CredentialOAuth, Extra: map[string]json.RawMessage{"baseUrl": json.RawMessage(`"https://oauth.example.com"`)}}
	if info := credentialInfoFromAI(&credential); info.BaseURL != "https://oauth.example.com" || info.Type != "oauth" {
		t.Fatalf("info = %+v", info)
	}
	if credentialInfoFromAI(nil) != nil {
		t.Fatal("nil credential")
	}
}

func TestPlaceholderHostIsCaseInsensitive(t *testing.T) {
	for _, host := range []string{"LiteLLM.Example.com", "LITELLM.EXAMPLE.COM.", "litellm.example.com"} {
		if !isPlaceholderHost(host) {
			t.Errorf("isPlaceholderHost(%q) = false", host)
		}
	}
	if _, err := requireCredentialRoot("https://LiteLLM.Example.com", "litellm"); err == nil || !strings.Contains(err.Error(), "placeholder LiteLLM base URL") {
		t.Errorf("mixed-case placeholder root error = %v", err)
	}
}
