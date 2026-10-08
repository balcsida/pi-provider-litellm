package litellm

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// captureDiagnostics replaces the reporter for one test and returns the collected messages.
func captureDiagnostics(t *testing.T) func() string {
	t.Helper()
	var messages []string
	previous := reportDiagnostic
	reportDiagnostic = func(message string) { messages = append(messages, message) }
	t.Cleanup(func() { reportDiagnostic = previous })
	return func() string { return strings.Join(messages, "\n") }
}

func writeADCFile(t *testing.T, body any) string {
	t.Helper()
	data, err := json.Marshal(body)
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "adc.json")
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func authorizedUser(overrides map[string]any) map[string]any {
	body := map[string]any{
		"type":          "authorized_user",
		"client_id":     "client-id",
		"client_secret": "client-secret",
		"refresh_token": "refresh-token",
	}
	for key, value := range overrides {
		body[key] = value
	}
	return body
}

// tokenServer serves the token endpoint and counts requests; each reply is the next entry of tokens.
func tokenServer(t *testing.T, status int, tokens ...string) (*atomic.Int32, *string) {
	t.Helper()
	var calls atomic.Int32
	var lastBody string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := int(calls.Add(1))
		_ = r.ParseForm()
		lastBody = r.PostForm.Encode()
		if r.Method != http.MethodPost || r.Header.Get("Content-Type") != "application/x-www-form-urlencoded" {
			t.Errorf("unexpected request %s %q", r.Method, r.Header.Get("Content-Type"))
		}
		if status != http.StatusOK {
			w.WriteHeader(status)
			_, _ = w.Write([]byte("invalid_grant: remote-secret"))
			return
		}
		token := tokens[min(n, len(tokens))-1]
		_ = json.NewEncoder(w).Encode(map[string]string{"access_token": token})
	}))
	t.Cleanup(server.Close)
	previousURL, previousClient := gcloudTokenURL, gcloudHTTPClient
	gcloudTokenURL, gcloudHTTPClient = server.URL, server.Client()
	t.Cleanup(func() { gcloudTokenURL, gcloudHTTPClient = previousURL, previousClient })
	return &calls, &lastBody
}

func gcloudTest(t *testing.T) {
	t.Helper()
	t.Setenv(envGoogleCredentials, "")
	t.Setenv("APPDATA", "")
	t.Setenv("HOME", t.TempDir())
	resetGcloudTokenCache()
	t.Cleanup(resetGcloudTokenCache)
}

func TestGetGcloudToken(t *testing.T) {
	ctx := context.Background()

	t.Run("detects authorized_user ADC without exchanging a token", func(t *testing.T) {
		gcloudTest(t)
		calls, _ := tokenServer(t, http.StatusOK, "unused")
		t.Setenv(envGoogleCredentials, writeADCFile(t, authorizedUser(nil)))
		if !hasGcloudADCCredentials() || calls.Load() != 0 {
			t.Fatalf("has=false or exchanged (%d calls)", calls.Load())
		}
	})

	for _, field := range []string{"client_id", "client_secret", "refresh_token"} {
		for _, blank := range []string{"", " \t", " \r\n"} {
			t.Run("rejects a blank "+field+" in authorized_user ADC "+strings.TrimSpace(blank), func(t *testing.T) {
				gcloudTest(t)
				calls, _ := tokenServer(t, http.StatusOK, "unused")
				diagnostics := captureDiagnostics(t)
				t.Setenv(envGoogleCredentials, writeADCFile(t, authorizedUser(map[string]any{field: blank})))
				if hasGcloudADCCredentials() || getGcloudToken(ctx) != "" || calls.Load() != 0 {
					t.Fatal("blank field must fail closed without a request")
				}
				if got := diagnostics(); !strings.Contains(got, "authorized_user ADC has invalid or incomplete required fields") || strings.Contains(got, "Unknown credential type") {
					t.Fatalf("diagnostics = %q", got)
				}
			})
		}
	}

	t.Run("preserves the unknown-type diagnostic for unsupported credential types", func(t *testing.T) {
		gcloudTest(t)
		diagnostics := captureDiagnostics(t)
		t.Setenv(envGoogleCredentials, writeADCFile(t, map[string]any{"type": "external_account"}))
		if getGcloudToken(ctx) != "" || !strings.Contains(diagnostics(), "Unknown credential type: external_account") {
			t.Fatalf("diagnostics = %q", diagnostics())
		}
	})

	t.Run("exchanges authorized_user ADC credentials for an access token", func(t *testing.T) {
		gcloudTest(t)
		_, body := tokenServer(t, http.StatusOK, "ya29.token")
		t.Setenv(envGoogleCredentials, writeADCFile(t, authorizedUser(nil)))
		if got := getGcloudToken(ctx); got != "ya29.token" {
			t.Fatalf("token = %q", got)
		}
		if !strings.Contains(*body, "grant_type=refresh_token") || !strings.Contains(*body, "client_id=client-id") {
			t.Fatalf("body = %q", *body)
		}
	})

	t.Run("uses the cached token until the TTL expires", func(t *testing.T) {
		gcloudTest(t)
		calls, _ := tokenServer(t, http.StatusOK, "first-token", "second-token")
		t.Setenv(envGoogleCredentials, writeADCFile(t, authorizedUser(nil)))
		clock := time.Date(2026, 6, 10, 12, 0, 0, 0, time.UTC)
		previous := now
		now = func() time.Time { return clock }
		t.Cleanup(func() { now = previous })

		if getGcloudToken(ctx) != "first-token" || getGcloudToken(ctx) != "first-token" || calls.Load() != 1 {
			t.Fatalf("expected one exchange, got %d", calls.Load())
		}
		clock = clock.Add(gcloudCacheTTL + time.Millisecond)
		if got := getGcloudToken(ctx); got != "second-token" || calls.Load() != 2 {
			t.Fatalf("token = %q after %d calls", got, calls.Load())
		}
	})

	t.Run("invalidates the cached token when only the refresh token rotates", func(t *testing.T) {
		gcloudTest(t)
		calls, _ := tokenServer(t, http.StatusOK, "before-relogin", "after-relogin")
		t.Setenv(envGoogleCredentials, writeADCFile(t, authorizedUser(map[string]any{"refresh_token": "refresh-one"})))
		if got := getGcloudToken(ctx); got != "before-relogin" {
			t.Fatalf("token = %q", got)
		}
		t.Setenv(envGoogleCredentials, writeADCFile(t, authorizedUser(map[string]any{"refresh_token": "refresh-two"})))
		if got := getGcloudToken(ctx); got != "after-relogin" || calls.Load() != 2 {
			t.Fatalf("token = %q after %d calls", got, calls.Load())
		}
	})

	t.Run("does not reuse a cached token after the ADC identity changes", func(t *testing.T) {
		gcloudTest(t)
		calls, _ := tokenServer(t, http.StatusOK, "first-token", "second-token")
		t.Setenv(envGoogleCredentials, writeADCFile(t, authorizedUser(map[string]any{"client_id": "first-client"})))
		getGcloudToken(ctx)
		t.Setenv(envGoogleCredentials, writeADCFile(t, authorizedUser(map[string]any{"client_id": "second-client"})))
		if got := getGcloudToken(ctx); got != "second-token" || calls.Load() != 2 {
			t.Fatalf("token = %q after %d calls", got, calls.Load())
		}
	})

	t.Run("returns empty and warns when no ADC file exists", func(t *testing.T) {
		gcloudTest(t)
		diagnostics := captureDiagnostics(t)
		if getGcloudToken(ctx) != "" || !strings.Contains(diagnostics(), "No Google ADC file found") {
			t.Fatalf("diagnostics = %q", diagnostics())
		}
	})

	t.Run("finds the default gcloud ADC file under HOME", func(t *testing.T) {
		gcloudTest(t)
		tokenServer(t, http.StatusOK, "home-token")
		home := t.TempDir()
		t.Setenv("HOME", home)
		dir := filepath.Join(home, ".config", "gcloud")
		if err := os.MkdirAll(dir, 0o700); err != nil {
			t.Fatal(err)
		}
		data, _ := json.Marshal(authorizedUser(nil))
		if err := os.WriteFile(filepath.Join(dir, gcloudADCFilename), data, 0o600); err != nil {
			t.Fatal(err)
		}
		if got := getGcloudToken(ctx); got != "home-token" {
			t.Fatalf("token = %q", got)
		}
	})

	t.Run("returns empty and warns for service account credentials", func(t *testing.T) {
		gcloudTest(t)
		diagnostics := captureDiagnostics(t)
		t.Setenv(envGoogleCredentials, writeADCFile(t, map[string]any{"type": "service_account"}))
		if getGcloudToken(ctx) != "" || !strings.Contains(diagnostics(), "Service account credentials are not supported") {
			t.Fatalf("diagnostics = %q", diagnostics())
		}
	})

	t.Run("returns empty when the token exchange fails", func(t *testing.T) {
		gcloudTest(t)
		tokenServer(t, http.StatusBadRequest)
		diagnostics := captureDiagnostics(t)
		t.Setenv(envGoogleCredentials, writeADCFile(t, authorizedUser(nil)))
		got := diagnostics
		if getGcloudToken(ctx) != "" || !strings.Contains(got(), "Token exchange failed") || strings.Contains(got(), "remote-secret") {
			t.Fatalf("diagnostics = %q", got())
		}
	})
}

func TestIsGcloudTokenAuthEnabled(t *testing.T) {
	for value, want := range map[string]bool{"": false, "0": false, "1": true, "true": true} {
		t.Setenv(envGcloudTokenAuth, value)
		if got := isGcloudTokenAuthEnabled(); got != want {
			t.Errorf("%q: got %v, want %v", value, got, want)
		}
	}
	os.Unsetenv(envGcloudTokenAuth)
	if isGcloudTokenAuthEnabled() {
		t.Error("unset must be disabled")
	}
}
