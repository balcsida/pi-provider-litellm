package litellm

// Ports the PKCE refresh, direct OIDC login, login base URL reuse and login startup cases of
// tests/index.test.ts. The proxy and the IdP are httptest servers; the flows are driven directly with a
// scripted ai.AuthInteraction. The protocols themselves are github.com/balcsida/litellm-auth-go's and tested
// there, so these cases prove the glue: flow choice, events, the auth.json shapes, refresh classification and
// that a library rejection surfaces as an error.

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/MichaelKinsy/PiG/ai"
	litellmauth "github.com/balcsida/litellm-auth-go"
)

const (
	testNow              = int64(1_800_000_000_000)
	testIssuerF          = "https://idp.example.com"
	cliAuthDiscoveryPath = "/.well-known/litellm-cli-auth"
)

// oauthTest isolates the environment and shortens every login timer.
func oauthTest(t *testing.T) string {
	t.Helper()
	dir := hermeticAgentDir(t)
	savedTimeout, savedPoll, savedCallback, savedTransport, savedNow := loginTimeout, cliSSOPollInterval, callbackTimeout, authTransport, now
	cliSSOPollInterval = 5 * time.Millisecond
	transientBackoffs.Lock()
	transientBackoffs.until = map[string]int64{}
	transientBackoffs.Unlock()
	t.Cleanup(func() {
		loginTimeout, cliSSOPollInterval, callbackTimeout, authTransport, now = savedTimeout, savedPoll, savedCallback, savedTransport, savedNow
	})
	return dir
}

func pinNow(millis int64) {
	now = func() time.Time { return time.UnixMilli(millis) }
}

// scriptedAuth answers prompts and records everything the flow shows.
type scriptedAuth struct {
	mu      sync.Mutex
	answer  func(ai.AuthPrompt) (string, error)
	onEvent func(ai.AuthEvent)
	prompts []ai.AuthPrompt
	events  []ai.AuthEvent
}

func (s *scriptedAuth) interaction() ai.AuthInteraction {
	return ai.AuthInteraction{
		Prompt: func(_ context.Context, prompt ai.AuthPrompt) (string, error) {
			s.mu.Lock()
			s.prompts = append(s.prompts, prompt)
			s.mu.Unlock()
			return s.answer(prompt)
		},
		Notify: func(event ai.AuthEvent) {
			s.mu.Lock()
			s.events = append(s.events, event)
			s.mu.Unlock()
			if s.onEvent != nil {
				s.onEvent(event)
			}
		},
	}
}

func (s *scriptedAuth) promptTypes() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	var types []string
	for _, prompt := range s.prompts {
		types = append(types, prompt.Type())
	}
	return types
}

func (s *scriptedAuth) eventsOf(kind string) []ai.AuthEvent {
	s.mu.Lock()
	defer s.mu.Unlock()
	var matched []ai.AuthEvent
	for _, event := range s.events {
		if event.Type() == kind {
			matched = append(matched, event)
		}
	}
	return matched
}

func promptMessage(prompt ai.AuthPrompt) string {
	switch typed := prompt.(type) {
	case ai.AuthTextPrompt:
		return typed.Message
	case ai.AuthSecretPrompt:
		return typed.Message
	case ai.AuthSelectPrompt:
		return typed.Message
	}
	return ""
}

// answers builds the usual script: the proxy URL for the URL prompt, token for secrets, other for the rest.
func answers(baseURL, secret, other string) func(ai.AuthPrompt) (string, error) {
	return func(prompt ai.AuthPrompt) (string, error) {
		switch typed := prompt.(type) {
		case ai.AuthTextPrompt:
			if typed.Placeholder != "" {
				return baseURL, nil
			}
			return other, nil
		case ai.AuthSecretPrompt:
			return secret, nil
		case ai.AuthSelectPrompt:
			return typed.Options[0].ID, nil
		}
		return "", nil
	}
}

type oauthRequest struct {
	method, path, rawQuery, body string
	header                       http.Header
}

// recorder is an httptest server that logs every request before routing it to its handlers.
type recorder struct {
	*httptest.Server
	mu       sync.Mutex
	requests []oauthRequest
	handlers map[string]http.HandlerFunc
}

func newRecorder(t *testing.T, tls bool) *recorder {
	t.Helper()
	r := &recorder{handlers: map[string]http.HandlerFunc{}}
	handler := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		body, _ := io.ReadAll(req.Body)
		r.mu.Lock()
		r.requests = append(r.requests, oauthRequest{req.Method, req.URL.Path, req.URL.RawQuery, string(body), req.Header.Clone()})
		fn := r.handlers[req.URL.Path]
		r.mu.Unlock()
		if fn == nil {
			http.NotFound(w, req)
			return
		}
		req.Body = io.NopCloser(strings.NewReader(string(body)))
		fn(w, req)
	})
	if tls {
		r.Server = httptest.NewTLSServer(handler)
	} else {
		r.Server = httptest.NewServer(handler)
	}
	t.Cleanup(r.Server.Close)
	return r
}

func (r *recorder) handle(path string, fn http.HandlerFunc) {
	r.mu.Lock()
	r.handlers[path] = fn
	r.mu.Unlock()
}

func (r *recorder) log() []oauthRequest {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]oauthRequest(nil), r.requests...)
}

func (r *recorder) count(path string) int {
	n := 0
	for _, request := range r.log() {
		if request.path == path {
			n++
		}
	}
	return n
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func jsonHandler(status int, value any) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) { writeJSON(w, status, value) }
}

// dropConnection closes the connection without a response, as a network failure.
func dropConnection(w http.ResponseWriter, _ *http.Request) {
	conn, _, _ := w.(http.Hijacker).Hijack()
	_ = conn.Close()
}

// truncatedBody promises more bytes than it sends, so reading the body fails.
func truncatedBody(w http.ResponseWriter, _ *http.Request) {
	conn, buf, _ := w.(http.Hijacker).Hijack()
	_, _ = buf.WriteString("HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{")
	_ = buf.Flush()
	_ = conn.Close()
}

func queryOf(t *testing.T, body string) map[string]string {
	t.Helper()
	values, err := url.ParseQuery(body)
	if err != nil {
		t.Fatal(err)
	}
	out := map[string]string{}
	for name := range values {
		out[name] = values.Get(name)
	}
	return out
}

func equalMaps(a, b map[string]string) bool {
	if len(a) != len(b) {
		return false
	}
	for name, value := range a {
		if other, ok := b[name]; !ok || other != value {
			return false
		}
	}
	return true
}

func extra(t *testing.T, credential ai.Credential, name string) string {
	t.Helper()
	value, _ := extraString(credential, name)
	return value
}

func makeJWT(exp int64) string {
	encode := func(value any) string {
		data, _ := json.Marshal(value)
		return base64.RawURLEncoding.EncodeToString(data)
	}
	return encode(map[string]any{"alg": "none"}) + "." + encode(map[string]any{"exp": exp}) + ".sig"
}

func idToken(claims map[string]any) string {
	encode := func(value any) string {
		data, _ := json.Marshal(value)
		return base64.RawURLEncoding.EncodeToString(data)
	}
	return encode(map[string]any{"alg": "RS256", "typ": "JWT"}) + "." + encode(claims) + ".sig"
}

func merged(base map[string]any, override map[string]any) map[string]any {
	out := map[string]any{}
	for name, value := range base {
		out[name] = value
	}
	for name, value := range override {
		if value == nil {
			delete(out, name)
		} else {
			out[name] = value
		}
	}
	return out
}

// ---- PKCE refresh -----------------------------------------------------------------------------------------

func pkceCredential(proxy *recorder, mutate ...func(*ai.Credential)) ai.Credential {
	credential := newOAuthCredential("access-old", "refresh-old", float64(testNow+60_000), proxy.URL)
	setExtra(&credential, "flow", pkceFlow)
	setExtra(&credential, "clientId", "llm_dcrc_client")
	setExtra(&credential, "tokenEndpoint", proxy.URL+"/token")
	setExtra(&credential, "resource", proxy.URL)
	for _, fn := range mutate {
		fn(&credential)
	}
	return credential
}

func refreshOf(t *testing.T, credential ai.Credential) (ai.Credential, error) {
	t.Helper()
	return refreshLiteLLM(context.Background(), credential, defaultDefinition(t))
}

var goodPkceToken = map[string]any{"access_token": "access-new", "refresh_token": "refresh-new", "token_type": "bearer", "expires_in": 3600}

// assertExpiresAbout checks an expiry the library computed from the real clock against the window the call ran in.
func assertExpiresAbout(t *testing.T, credential ai.Credential, started, ended time.Time, lifetime time.Duration) {
	t.Helper()
	if expires := int64(credential.ExpiresMillis()); expires < started.UnixMilli()+lifetime.Milliseconds() || expires > ended.UnixMilli()+lifetime.Milliseconds() {
		t.Errorf("expires = %d, want within [%d, %d]", expires, started.UnixMilli()+lifetime.Milliseconds(), ended.UnixMilli()+lifetime.Milliseconds())
	}
}

func TestPKCERefresh(t *testing.T) {
	t.Run("rotates PKCE refresh credentials", func(t *testing.T) {
		oauthTest(t)
		pinNow(testNow)
		t.Setenv(envHeaders, `{"x-tenant":"tenant-a"}`)
		proxy := newRecorder(t, false)
		proxy.handle("/token", jsonHandler(200, goodPkceToken))
		credential := pkceCredential(proxy)
		started := time.Now()
		refreshed, err := refreshOf(t, credential)
		ended := time.Now()
		if err != nil {
			t.Fatal(err)
		}
		if refreshed.Access != "access-new" || refreshed.Refresh != "refresh-new" {
			t.Errorf("refreshed = %+v", refreshed)
		}
		assertExpiresAbout(t, refreshed, started, ended, time.Hour)
		if extra(t, refreshed, "clientId") != "llm_dcrc_client" || extra(t, refreshed, "baseUrl") != proxy.URL || extra(t, refreshed, "flow") != pkceFlow {
			t.Errorf("metadata lost: %v", refreshed.Extra)
		}
		if extra(t, credential, "flow") != pkceFlow || credential.Access != "access-old" {
			t.Error("refresh mutated its input")
		}
		requests := proxy.log()
		if len(requests) != 1 || requests[0].method != "POST" || requests[0].header.Get("x-tenant") != "tenant-a" {
			t.Fatalf("requests = %+v", requests)
		}
		want := map[string]string{"grant_type": "refresh_token", "refresh_token": "refresh-old", "client_id": "llm_dcrc_client", "resource": proxy.URL}
		if got := queryOf(t, requests[0].body); !equalMaps(got, want) {
			t.Errorf("form = %v", got)
		}
	})

	t.Run("rejects a refresh that does not rotate the refresh token", func(t *testing.T) {
		oauthTest(t)
		pinNow(testNow)
		proxy := newRecorder(t, false)
		proxy.handle("/token", jsonHandler(200, map[string]any{"access_token": "access-new", "token_type": "bearer", "expires_in": 3600}))
		_, err := refreshOf(t, pkceCredential(proxy))
		if err == nil || err.Error() != "LiteLLM token exchange returned an invalid response; run /login litellm again" {
			t.Errorf("err = %v", err)
		}
	})

	t.Run("sets userId and teamId the proxy names and deletes the ones it omits", func(t *testing.T) {
		oauthTest(t)
		pinNow(testNow)
		proxy := newRecorder(t, false)
		proxy.handle("/token", jsonHandler(200, merged(goodPkceToken, map[string]any{"user_id": "user@example.com", "team_id": "team-a"})))
		credential := pkceCredential(proxy, func(c *ai.Credential) { setExtra(c, "userId", "stale-user") })
		refreshed, err := refreshOf(t, credential)
		if err != nil || extra(t, refreshed, "userId") != "user@example.com" || extra(t, refreshed, "teamId") != "team-a" {
			t.Fatalf("refreshed = %+v, %v", refreshed.Extra, err)
		}
		proxy.handle("/token", jsonHandler(200, goodPkceToken))
		refreshed, err = refreshOf(t, refreshed)
		if err != nil {
			t.Fatal(err)
		}
		if _, hasUser := refreshed.Extra["userId"]; hasUser {
			t.Errorf("userId kept: %v", refreshed.Extra)
		}
		if _, hasTeam := refreshed.Extra["teamId"]; hasTeam {
			t.Errorf("teamId kept: %v", refreshed.Extra)
		}
	})

	for name, handler := range map[string]http.HandlerFunc{
		"429": jsonHandler(429, map[string]any{}), "500": jsonHandler(500, map[string]any{}), "503": jsonHandler(503, map[string]any{}),
		"network": dropConnection, "body": truncatedBody,
	} {
		t.Run("keeps fresh PKCE credentials after "+name, func(t *testing.T) {
			oauthTest(t)
			pinNow(testNow)
			proxy := newRecorder(t, false)
			proxy.handle("/token", handler)
			credential := pkceCredential(proxy)
			refreshed, err := refreshOf(t, credential)
			if err != nil || refreshed.Access != "access-old" || refreshed.Refresh != "refresh-old" {
				t.Errorf("refreshed = %+v, %v", refreshed, err)
			}
		})
	}

	t.Run("keeps fresh PKCE credentials after a redirect and applies the backoff", func(t *testing.T) {
		oauthTest(t)
		pinNow(testNow)
		proxy := newRecorder(t, false)
		proxy.handle("/token", func(w http.ResponseWriter, r *http.Request) { http.Redirect(w, r, "/elsewhere", http.StatusFound) })
		credential := pkceCredential(proxy)
		for range 2 {
			if refreshed, err := refreshOf(t, credential); err != nil || refreshed.Access != "access-old" {
				t.Fatalf("refreshed = %+v, %v", refreshed, err)
			}
		}
		if proxy.count("/token") != 1 {
			t.Errorf("token requests = %d, want 1", proxy.count("/token"))
		}
	})

	for name, handler := range map[string]http.HandlerFunc{
		"a stalled body": func(w http.ResponseWriter, r *http.Request) {
			conn, buf, _ := w.(http.Hijacker).Hijack()
			_, _ = buf.WriteString("HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{")
			_ = buf.Flush()
			<-r.Context().Done()
			_ = conn.Close()
		},
		"a reset mid-body": func(w http.ResponseWriter, _ *http.Request) {
			conn, buf, _ := w.(http.Hijacker).Hijack()
			_, _ = buf.WriteString("HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{")
			_ = buf.Flush()
			_ = conn.(*net.TCPConn).SetLinger(0)
			_ = conn.Close()
		},
	} {
		t.Run("classifies "+name+" as a transient network error", func(t *testing.T) {
			oauthTest(t)
			pinNow(testNow)
			loginTimeout = 50 * time.Millisecond
			proxy := newRecorder(t, false)
			proxy.handle("/token", handler)
			// An expired credential surfaces the classification instead of keeping the old one.
			credential := pkceCredential(proxy, func(c *ai.Credential) { c.SetExpiresMillis(float64(testNow - 1)) })
			_, err := refreshOf(t, credential)
			if err == nil || !strings.Contains(err.Error(), "(network error)") || strings.Contains(err.Error(), "run /login") {
				t.Errorf("err = %v", err)
			}
		})
	}

	t.Run("keeps fresh PKCE credentials when the internal refresh deadline elapses", func(t *testing.T) {
		oauthTest(t)
		pinNow(testNow)
		loginTimeout = 20 * time.Millisecond
		proxy := newRecorder(t, false)
		proxy.handle("/token", func(w http.ResponseWriter, r *http.Request) {
			select {
			case <-r.Context().Done():
			case <-time.After(2 * time.Second):
			}
		})
		refreshed, err := refreshOf(t, pkceCredential(proxy))
		if err != nil || refreshed.Access != "access-old" {
			t.Errorf("refreshed = %+v, %v", refreshed, err)
		}
	})

	t.Run("scopes the transient PKCE refresh backoff to one credential", func(t *testing.T) {
		oauthTest(t)
		pinNow(testNow)
		proxy := newRecorder(t, false)
		var calls atomic.Int32
		proxy.handle("/token", func(w http.ResponseWriter, r *http.Request) {
			if calls.Add(1) == 1 {
				writeJSON(w, 503, map[string]any{})
				return
			}
			writeJSON(w, 200, merged(goodPkceToken, map[string]any{"token_type": "Bearer"}))
		})
		failing := pkceCredential(proxy)
		other := pkceCredential(proxy, func(c *ai.Credential) {
			c.Refresh = "refresh-other"
			setExtra(c, "clientId", "llm_dcrc_other")
		})
		if refreshed, err := refreshOf(t, failing); err != nil || refreshed.Access != "access-old" {
			t.Fatalf("failing = %+v, %v", refreshed, err)
		}
		if refreshed, err := refreshOf(t, other); err != nil || refreshed.Access != "access-new" || refreshed.Refresh != "refresh-new" {
			t.Fatalf("other = %+v, %v", refreshed, err)
		}
		if calls.Load() != 2 {
			t.Errorf("token calls = %d", calls.Load())
		}
		// The failing credential is inside its backoff window: no third request.
		if refreshed, err := refreshOf(t, failing); err != nil || refreshed.Access != "access-old" || calls.Load() != 2 {
			t.Errorf("backoff not honored: %+v, %v, calls %d", refreshed, err, calls.Load())
		}
	})

	t.Run("rejects a transient PKCE refresh failure after expiry without asking for a new login", func(t *testing.T) {
		oauthTest(t)
		pinNow(testNow)
		proxy := newRecorder(t, false)
		proxy.handle("/token", jsonHandler(503, map[string]any{}))
		credential := pkceCredential(proxy)
		credential.SetExpiresMillis(float64(testNow))
		if _, err := refreshOf(t, credential); err == nil || err.Error() != "LiteLLM token exchange failed (HTTP 503)" {
			t.Errorf("err = %v", err)
		}
	})

	t.Run("sanitizes PKCE refresh OAuth errors", func(t *testing.T) {
		oauthTest(t)
		pinNow(testNow)
		proxy := newRecorder(t, false)
		proxy.handle("/token", jsonHandler(400, map[string]any{"error": "invalid_grant", "error_description": "refresh-old\naccess-old\x1b[31m", "token": "unrelated-secret"}))
		_, err := refreshOf(t, pkceCredential(proxy))
		if err == nil || err.Error() != "LiteLLM token exchange rejected (invalid_grant); run /login litellm again" {
			t.Errorf("err = %v", err)
		}
	})

	for name, mutate := range map[string]func(*recorder, *ai.Credential){
		"cross-origin token endpoint": func(_ *recorder, c *ai.Credential) {
			setExtra(c, "tokenEndpoint", "https://attacker.example.com/token")
		},
		"cross-origin resource": func(_ *recorder, c *ai.Credential) { setExtra(c, "resource", "https://attacker.example.com") },
		"token endpoint credentials": func(p *recorder, c *ai.Credential) {
			setExtra(c, "tokenEndpoint", strings.Replace(p.URL, "http://", "http://secret@", 1)+"/token")
		},
		"token endpoint fragment": func(p *recorder, c *ai.Credential) { setExtra(c, "tokenEndpoint", p.URL+"/token#fragment") },
		"blob token endpoint":     func(p *recorder, c *ai.Credential) { setExtra(c, "tokenEndpoint", "blob:"+p.URL+"/token") },
		"base URL credentials": func(p *recorder, c *ai.Credential) {
			setExtra(c, "baseUrl", strings.Replace(p.URL, "http://", "http://secret@", 1))
		},
		"insecure non-loopback base URL": func(_ *recorder, c *ai.Credential) { setExtra(c, "baseUrl", "http://proxy.example.com") },
		"empty client ID":                func(_ *recorder, c *ai.Credential) { setExtra(c, "clientId", "") },
		"empty refresh token":            func(_ *recorder, c *ai.Credential) { c.Refresh = "" },
	} {
		t.Run("rejects malformed PKCE refresh metadata before fetching: "+name, func(t *testing.T) {
			oauthTest(t)
			pinNow(testNow)
			proxy := newRecorder(t, false)
			proxy.handle("/token", jsonHandler(200, goodPkceToken))
			credential := pkceCredential(proxy)
			mutate(proxy, &credential)
			if _, err := refreshOf(t, credential); err == nil {
				t.Error("refresh succeeded")
			}
			if n := len(proxy.log()); n != 0 {
				t.Errorf("%d requests reached the proxy", n)
			}
		})
	}

	t.Run("preserves cancellation while reading the PKCE token response", func(t *testing.T) {
		oauthTest(t)
		pinNow(testNow)
		proxy := newRecorder(t, false)
		ctx, cancel := context.WithCancelCause(context.Background())
		reason := errors.New("cancelled token body")
		proxy.handle("/token", func(w http.ResponseWriter, r *http.Request) {
			cancel(reason)
			<-r.Context().Done()
		})
		_, err := refreshLiteLLM(ctx, pkceCredential(proxy), defaultDefinition(t))
		if !errors.Is(err, reason) {
			t.Errorf("err = %v", err)
		}
	})
}

// ---- login flows with a proxy ----------------------------------------------------------------------------

type pkceProxy struct {
	*recorder
	registrationBody  map[string]any
	callbackResponses chan *http.Response
}

// newPKCEProxy serves the proxy's native CLI auth contract the way litellm-auth-go validates it: every endpoint
// on the proxy origin, the issuer equal to the proxy URL, registration answering with a client_id.
func newPKCEProxy(t *testing.T, override map[string]any) *pkceProxy {
	t.Helper()
	p := &pkceProxy{recorder: newRecorder(t, false), callbackResponses: make(chan *http.Response, 8)}
	var bodyMu sync.Mutex
	p.handle(cliAuthDiscoveryPath, func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, 200, merged(map[string]any{
			"contract_version": 1, "issuer": p.URL, "authorization_endpoint": p.URL + "/authorize?tenant=alpha&client_id=wrong",
			"token_endpoint": p.URL + "/token", "registration_endpoint": p.URL + "/register", "revocation_endpoint": p.URL + "/revoke",
			"resource": p.URL, "code_challenge_methods_supported": []string{"S256"},
		}, override))
	})
	p.handle("/register", func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		bodyMu.Lock()
		p.registrationBody = body
		bodyMu.Unlock()
		writeJSON(w, 201, map[string]any{"client_id": "llm_dcrc_client", "redirect_uris": body["redirect_uris"]})
	})
	p.handle("/token", jsonHandler(200, merged(goodPkceToken, map[string]any{"token_type": "Bearer", "user_id": "user@example.com", "team_id": "team-a"})))
	return p
}

// followCallback visits the redirect URI of an auth_url event the way a browser would.
func followCallback(responses chan *http.Response, query func(authURL *url.URL) string) func(ai.AuthEvent) {
	return func(event ai.AuthEvent) {
		urlEvent, ok := event.(ai.AuthURLEvent)
		if !ok {
			return
		}
		authURL, _ := url.Parse(urlEvent.URL)
		go func() {
			response, err := http.Get(authURL.Query().Get("redirect_uri") + "?" + query(authURL))
			if err == nil {
				responses <- response
			}
		}()
	}
}

func okQuery(authURL *url.URL) string {
	return "code=authorization-code&state=" + url.QueryEscape(authURL.Query().Get("state"))
}

func callbackURLOf(event ai.AuthEvent) string {
	authURL, _ := url.Parse(event.(ai.AuthURLEvent).URL)
	return authURL.Query().Get("redirect_uri")
}

func assertListenerClosed(t *testing.T, redirectURI string) {
	t.Helper()
	if redirectURI == "" {
		t.Fatal("no callback URL was offered")
	}
	client := http.Client{Timeout: time.Second}
	if response, err := client.Get(redirectURI + "?code=authorization-code"); err == nil {
		response.Body.Close()
		t.Error("loopback listener still accepts connections")
	}
}

func TestNativePKCELogin(t *testing.T) {
	t.Run("completes native PKCE login", func(t *testing.T) {
		oauthTest(t)
		t.Setenv(envHeaders, `{"x-tenant":"tenant-a"}`)
		proxy := newPKCEProxy(t, nil)
		script := &scriptedAuth{answer: answers(proxy.URL, "", ""), onEvent: followCallback(proxy.callbackResponses, okQuery)}
		started := time.Now()
		credential, err := loginOAuth(context.Background(), script.interaction(), defaultDefinition(t))
		ended := time.Now()
		if err != nil {
			t.Fatal(err)
		}
		events := script.eventsOf("auth_url")
		if len(events) != 1 || events[0].(ai.AuthURLEvent).Instructions != "Open this URL in a browser to sign in with LiteLLM." {
			t.Fatalf("events = %v", script.events)
		}
		authURL, _ := url.Parse(events[0].(ai.AuthURLEvent).URL)
		query := authURL.Query()
		redirect := query.Get("redirect_uri")
		if !regexp.MustCompile(`^http://127\.0\.0\.1:\d+/callback$`).MatchString(redirect) {
			t.Errorf("redirect_uri = %q", redirect)
		}
		if query.Get("resource") != proxy.URL || query.Get("tenant") != "alpha" || len(query["client_id"]) != 1 || query.Get("client_id") != "llm_dcrc_client" {
			t.Errorf("authorization query = %v", query)
		}
		if registration := proxy.registrationBody; registration["token_endpoint_auth_method"] != "none" ||
			fmt.Sprint(registration["redirect_uris"]) != "["+redirect+"]" || fmt.Sprint(registration["grant_types"]) != "[authorization_code refresh_token]" {
			t.Errorf("registration = %v", registration)
		}
		var token oauthRequest
		for _, request := range proxy.log() {
			if request.path == "/token" {
				token = request
			}
			if request.header.Get("x-tenant") != "tenant-a" {
				t.Errorf("%s did not carry the custom header: %v", request.path, request.header)
			}
		}
		form := queryOf(t, token.body)
		digest := sha256.Sum256([]byte(form["code_verifier"]))
		if base64.RawURLEncoding.EncodeToString(digest[:]) != query.Get("code_challenge") || query.Get("code_challenge_method") != "S256" {
			t.Error("code challenge does not match the verifier")
		}
		if form["redirect_uri"] != redirect || form["code"] != "authorization-code" || form["resource"] != proxy.URL {
			t.Errorf("token form = %v", form)
		}
		response := <-proxy.callbackResponses
		body, _ := io.ReadAll(response.Body)
		response.Body.Close()
		if response.StatusCode != 200 || response.Header.Get("Cache-Control") != "no-store" || !strings.Contains(string(body), "close this window") {
			t.Errorf("callback response = %d %v %q", response.StatusCode, response.Header, body)
		}
		if credential.Type != ai.CredentialOAuth || credential.Access != "access-new" || credential.Refresh != "refresh-new" {
			t.Errorf("credential = %+v", credential)
		}
		for name, want := range map[string]string{"baseUrl": proxy.URL, "flow": pkceFlow, "clientId": "llm_dcrc_client", "tokenEndpoint": proxy.URL + "/token", "resource": proxy.URL, "userId": "user@example.com", "teamId": "team-a"} {
			if got := extra(t, credential, name); got != want {
				t.Errorf("%s = %q, want %q", name, got, want)
			}
		}
		assertExpiresAbout(t, credential, started, ended, time.Hour)
		if proxy.count("/sso/cli/start") != 0 {
			t.Error("CLI SSO was started")
		}
		assertListenerClosed(t, redirect)
	})

	t.Run("omits userId and teamId the proxy does not name", func(t *testing.T) {
		oauthTest(t)
		proxy := newPKCEProxy(t, nil)
		proxy.handle("/token", jsonHandler(200, goodPkceToken))
		script := &scriptedAuth{answer: answers(proxy.URL, "", ""), onEvent: followCallback(proxy.callbackResponses, okQuery)}
		credential, err := loginOAuth(context.Background(), script.interaction(), defaultDefinition(t))
		if err != nil {
			t.Fatal(err)
		}
		if _, hasUser := credential.Extra["userId"]; hasUser {
			t.Errorf("extra = %v", credential.Extra)
		}
		if _, hasTeam := credential.Extra["teamId"]; hasTeam {
			t.Errorf("extra = %v", credential.Extra)
		}
	})

	// The library validates the discovery document; one foreign-origin case proves its rejection surfaces here
	// and stops the login before registration.
	t.Run("rejects native PKCE discovery with a cross-origin token endpoint", func(t *testing.T) {
		oauthTest(t)
		proxy := newPKCEProxy(t, map[string]any{"token_endpoint": "https://other.example.com/token"})
		script := &scriptedAuth{answer: answers(proxy.URL, "", "")}
		_, err := loginOAuth(context.Background(), script.interaction(), defaultDefinition(t))
		if !errors.Is(err, litellmauth.ErrProtocol) || errors.Is(err, litellmauth.ErrPKCEUnsupported) {
			t.Errorf("err = %v", err)
		}
		if proxy.count("/register") != 0 || proxy.count("/sso/cli/start") != 0 {
			t.Error("login went on after a rejected contract")
		}
	})

	for _, baseURL := range []string{"http://secret@127.0.0.1:9", "http://127.0.0.1:9?tenant=other", "http://127.0.0.1:9#fragment"} {
		t.Run("rejects an invalid PKCE proxy URL before discovery: "+baseURL, func(t *testing.T) {
			oauthTest(t)
			script := &scriptedAuth{answer: answers(baseURL, "", "")}
			if _, err := loginOAuth(context.Background(), script.interaction(), defaultDefinition(t)); err == nil {
				t.Error("login succeeded")
			}
		})
	}

	t.Run("closes the native PKCE callback listener when login is aborted", func(t *testing.T) {
		oauthTest(t)
		proxy := newPKCEProxy(t, nil)
		ctx, cancel := context.WithCancelCause(context.Background())
		reason := errors.New("caller cancelled login")
		offered := make(chan string, 1)
		script := &scriptedAuth{answer: answers(proxy.URL, "", ""), onEvent: func(event ai.AuthEvent) {
			if _, ok := event.(ai.AuthURLEvent); ok {
				offered <- callbackURLOf(event)
			}
		}}
		done := make(chan error, 1)
		go func() {
			_, err := loginOAuth(ctx, script.interaction(), defaultDefinition(t))
			done <- err
		}()
		redirect := <-offered
		cancel(reason)
		if err := <-done; !errors.Is(err, reason) {
			t.Errorf("err = %v", err)
		}
		assertListenerClosed(t, redirect)
	})

	t.Run("times out waiting for the callback", func(t *testing.T) {
		oauthTest(t)
		callbackTimeout = 30 * time.Millisecond
		proxy := newPKCEProxy(t, nil)
		script := &scriptedAuth{answer: answers(proxy.URL, "", "")}
		var timeout litellmauth.LoginTimeoutError
		_, err := loginOAuth(context.Background(), script.interaction(), defaultDefinition(t))
		if !errors.As(err, &timeout) || err.Error() != "LiteLLM PKCE login timed out" {
			t.Errorf("err = %v", err)
		}
	})

	for name, override := range map[string]map[string]any{
		"contract_version 2":       {"contract_version": 2},
		"no S256 challenge method": {"code_challenge_methods_supported": []string{"plain"}},
	} {
		t.Run("falls through to CLI SSO for discovery with "+name, func(t *testing.T) {
			oauthTest(t)
			proxy := newPKCEProxy(t, override)
			proxy.handle("/sso/cli/start", jsonHandler(404, map[string]any{}))
			proxy.handle("/sso/key/generate", jsonHandler(200, map[string]any{}))
			script := &scriptedAuth{answer: answers(proxy.URL, "token", "")}
			_, _ = loginOAuth(context.Background(), script.interaction(), defaultDefinition(t))
			if proxy.count("/sso/cli/start") != 1 || proxy.count("/register") != 0 {
				t.Errorf("start = %d, register = %d", proxy.count("/sso/cli/start"), proxy.count("/register"))
			}
		})
	}

	outcomes := []struct {
		name    string
		wantErr func(error) bool
	}{
		{"success", nil},
		{"OAuth denial", func(err error) bool { return errors.Is(err, litellmauth.ErrPKCEDenied) }},
		{"registration failure", func(err error) bool {
			var httpErr *litellmauth.HTTPError
			return errors.As(err, &httpErr) && httpErr.Op == "register" && httpErr.StatusCode == 500
		}},
		{"token failure", func(err error) bool {
			var httpErr *litellmauth.HTTPError
			return errors.As(err, &httpErr) && httpErr.Op == "token" && httpErr.StatusCode == 500
		}},
		{"token network failure", func(err error) bool { return err != nil && strings.Contains(err.Error(), "request failed") }},
		{"registration invalid JSON", func(err error) bool { return errors.Is(err, litellmauth.ErrProtocol) }},
	}
	for _, outcome := range outcomes {
		t.Run("closes the native PKCE callback listener after "+outcome.name, func(t *testing.T) {
			oauthTest(t)
			proxy := newPKCEProxy(t, nil)
			switch outcome.name {
			case "registration failure":
				proxy.handle("/register", jsonHandler(500, map[string]any{}))
			case "registration invalid JSON":
				proxy.handle("/register", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte("untrusted-secret")) })
			case "token failure":
				proxy.handle("/token", jsonHandler(500, map[string]any{}))
			case "token network failure":
				proxy.handle("/token", dropConnection)
			}
			query := okQuery
			if outcome.name == "OAuth denial" {
				query = func(authURL *url.URL) string {
					return "state=" + url.QueryEscape(authURL.Query().Get("state")) + "&error=access_denied&error_description=User%20declined"
				}
			}
			script := &scriptedAuth{answer: answers(proxy.URL, "", ""), onEvent: followCallback(proxy.callbackResponses, query)}
			_, err := loginOAuth(context.Background(), script.interaction(), defaultDefinition(t))
			redirect := registeredRedirect(proxy.recorder)
			if outcome.wantErr == nil {
				if err != nil {
					t.Fatalf("err = %v", err)
				}
			} else if err == nil || !outcome.wantErr(err) {
				t.Errorf("err = %v", err)
			}
			assertListenerClosed(t, redirect)
		})
	}
}

// registeredRedirect is the redirect URI the flow registered, which names its loopback listener.
func registeredRedirect(proxy *recorder) string {
	for _, request := range proxy.log() {
		if request.path == "/register" {
			var body struct {
				RedirectURIs []string `json:"redirect_uris"`
			}
			_ = json.Unmarshal([]byte(request.body), &body)
			if len(body.RedirectURIs) > 0 {
				return body.RedirectURIs[0]
			}
		}
	}
	return ""
}

// ---- CLI SSO and pasted token ----------------------------------------------------------------------------

func TestCliSSOLogin(t *testing.T) {
	startOK := jsonHandler(200, map[string]any{"login_id": "cli-login", "poll_secret": "poll-secret", "user_code": "ABCD-EFGH", "expires_in": 600})

	t.Run("completes CLI SSO with the selected team and the key's own lifetime", func(t *testing.T) {
		oauthTest(t)
		proxy := newRecorder(t, false)
		proxy.handle(cliAuthDiscoveryPath, nil)
		proxy.handle("/sso/cli/start", startOK)
		proxy.handle("/sso/cli/poll/cli-login", func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Query().Get("team_id") == "team-b" {
				writeJSON(w, 200, map[string]any{"status": "ready", "key": "opaque-cli-token", "expires_in": 7200, "team_id": "team-b", "team_details": []any{map[string]any{"id": "team-b", "team_alias": "Beta"}}})
				return
			}
			writeJSON(w, 200, map[string]any{"status": "ready", "requires_team_selection": true, "team_details": []any{map[string]any{"id": "team-b", "team_alias": "Beta"}}})
		})
		script := &scriptedAuth{answer: answers(proxy.URL, "", "")}
		t.Setenv(envHeaders, `{"x-tenant":"tenant-a"}`)
		started := time.Now().UnixMilli()
		credential, err := loginOAuth(context.Background(), script.interaction(), defaultDefinition(t))
		if err != nil {
			t.Fatal(err)
		}
		events := script.eventsOf("device_code")
		if len(events) != 1 {
			t.Fatalf("events = %v", script.events)
		}
		device := events[0].(ai.AuthDeviceCodeEvent)
		if device.UserCode != "ABCD-EFGH" || device.VerificationURI != proxy.URL+"/sso/key/generate?key=cli-login&source=litellm-cli" || device.ExpiresInSeconds == nil || *device.ExpiresInSeconds != 600 {
			t.Errorf("device event = %+v", device)
		}
		if credential.Access != "opaque-cli-token" || credential.Refresh != "" || extra(t, credential, "baseUrl") != proxy.URL || int64(credential.ExpiresMillis()) < started+24*3600*1000 {
			t.Errorf("credential = %+v", credential)
		}
		var polls []string
		for _, request := range proxy.log() {
			if strings.HasPrefix(request.path, "/sso/cli/poll/") {
				polls = append(polls, request.path+"?"+request.rawQuery+"|"+request.header.Get("x-litellm-cli-poll-secret"))
			}
			if request.path == "/sso/cli/start" && (request.method != "POST" || request.header.Get("x-litellm-cli-poll-secret") != "" || request.header.Get("x-tenant") != "tenant-a") {
				t.Errorf("start request = %+v", request)
			}
		}
		if strings.Join(polls, ",") != "/sso/cli/poll/cli-login?|poll-secret,/sso/cli/poll/cli-login?team_id=team-b|poll-secret" {
			t.Errorf("polls = %v", polls)
		}
		if got := script.promptTypes(); len(got) != 2 || got[1] != "select" {
			t.Errorf("prompts = %v", got)
		}
	})

	t.Run("retries pending and transient CLI SSO polls", func(t *testing.T) {
		oauthTest(t)
		proxy := newRecorder(t, false)
		var polls atomic.Int32
		proxy.handle("/sso/cli/start", jsonHandler(200, map[string]any{"login_id": "cli-login", "poll_secret": "poll-secret", "user_code": "ABCD-EFGH"}))
		proxy.handle("/sso/cli/poll/cli-login", func(w http.ResponseWriter, _ *http.Request) {
			switch polls.Add(1) {
			case 1:
				writeJSON(w, 200, map[string]any{"status": "pending"})
			case 2:
				writeJSON(w, 503, map[string]any{"detail": "unavailable"})
			default:
				writeJSON(w, 200, map[string]any{"status": "ready", "key": "opaque-cli-token"})
			}
		})
		script := &scriptedAuth{answer: answers(proxy.URL, "", "")}
		credential, err := loginOAuth(context.Background(), script.interaction(), defaultDefinition(t))
		if err != nil || credential.Access != "opaque-cli-token" || polls.Load() != 3 {
			t.Errorf("credential = %+v, %v, polls %d", credential, err, polls.Load())
		}
	})

	t.Run("preserves cancellation while waiting for a CLI SSO poll", func(t *testing.T) {
		oauthTest(t)
		cliSSOPollInterval = time.Minute
		proxy := newRecorder(t, false)
		polled := make(chan struct{}, 1)
		proxy.handle("/sso/cli/start", jsonHandler(200, map[string]any{"login_id": "cli-login", "poll_secret": "poll-secret", "user_code": "ABCD-EFGH"}))
		proxy.handle("/sso/cli/poll/cli-login", func(w http.ResponseWriter, _ *http.Request) {
			select {
			case polled <- struct{}{}:
			default:
			}
			writeJSON(w, 200, map[string]any{"status": "pending"})
		})
		ctx, cancel := context.WithCancelCause(context.Background())
		reason := errors.New("caller cancelled login")
		script := &scriptedAuth{answer: answers(proxy.URL, "", "")}
		done := make(chan error, 1)
		go func() {
			_, err := loginOAuth(ctx, script.interaction(), defaultDefinition(t))
			done <- err
		}()
		<-polled
		cancel(reason)
		if err := <-done; !errors.Is(err, reason) {
			t.Errorf("err = %v", err)
		}
	})

	t.Run("uses legacy token paste only when CLI SSO start is unavailable", func(t *testing.T) {
		oauthTest(t)
		proxy := newRecorder(t, false)
		var status atomic.Int32
		status.Store(404)
		proxy.handle("/sso/cli/start", func(w http.ResponseWriter, _ *http.Request) { writeJSON(w, int(status.Load()), map[string]any{}) })
		proxy.handle("/key/generate", jsonHandler(200, map[string]any{"key": "sk-legacy"}))
		login := func() (ai.Credential, error) {
			script := &scriptedAuth{answer: answers(proxy.URL, "Bearer legacy-token", "")}
			return loginOAuth(context.Background(), script.interaction(), defaultDefinition(t))
		}
		for _, code := range []int32{404, 405} {
			status.Store(code)
			if credential, err := login(); err != nil || credential.Access != "sk-legacy" {
				t.Errorf("status %d: %+v, %v", code, credential, err)
			}
		}
		status.Store(500)
		if _, err := login(); err == nil || err.Error() != "LiteLLM CLI SSO start: HTTP 500" {
			t.Errorf("err = %v", err)
		}
	})

	t.Run("uses configured and default CLI token lifetimes when poll expiry is absent", func(t *testing.T) {
		oauthTest(t)
		proxy := newRecorder(t, false)
		proxy.handle("/sso/cli/start", jsonHandler(200, map[string]any{"login_id": "cli-login", "poll_secret": "poll-secret", "user_code": "ABCD-EFGH"}))
		proxy.handle("/sso/cli/poll/cli-login", jsonHandler(200, map[string]any{"status": "ready", "key": "opaque-cli-token"}))
		login := func() ai.Credential {
			script := &scriptedAuth{answer: answers(proxy.URL, "", "")}
			credential, err := loginOAuth(context.Background(), script.interaction(), defaultDefinition(t))
			if err != nil {
				t.Fatal(err)
			}
			return credential
		}
		t.Setenv(envCLIJWTExpirationHours, "48")
		start := time.Now().UnixMilli()
		if got := int64(login().ExpiresMillis()); got < start+48*3600*1000 {
			t.Errorf("configured expires = %d", got-start)
		}
		os.Unsetenv(envCLIJWTExpirationHours)
		start = time.Now().UnixMilli()
		if got := int64(login().ExpiresMillis()); got < start+24*3600*1000 {
			t.Errorf("default expires = %d", got-start)
		}
	})
}

func pasteProxy(t *testing.T) *recorder {
	proxy := newRecorder(t, false)
	proxy.handle("/sso/cli/start", jsonHandler(404, map[string]any{}))
	return proxy
}

func TestEnterpriseSSOPaste(t *testing.T) {
	jwt := makeJWT(time.Now().Unix() + 3600)
	sso := func(proxy *recorder, token, other string) *scriptedAuth {
		return &scriptedAuth{answer: answers(proxy.URL, token, other)}
	}

	t.Run("generates a virtual key and uses it as the access token", func(t *testing.T) {
		oauthTest(t)
		proxy := pasteProxy(t)
		proxy.handle("/key/generate", jsonHandler(200, map[string]any{"key": "sk-virtual-abc"}))
		script := sso(proxy, "Bearer "+jwt, "y")
		credential, err := loginOAuth(context.Background(), script.interaction(), defaultDefinition(t))
		if err != nil {
			t.Fatal(err)
		}
		urls := script.eventsOf("auth_url")
		if len(urls) != 1 || urls[0].(ai.AuthURLEvent).URL != proxy.URL+"/sso/key/generate" || urls[0].(ai.AuthURLEvent).Instructions != "Authenticate via SSO, then copy your token from the LiteLLM UI." {
			t.Errorf("auth_url events = %v", urls)
		}
		if credential.Access != "sk-virtual-abc" || credential.Refresh != "" || credential.ExpiresMillis() != float64(permanentTokenExpiresAt) || extra(t, credential, "baseUrl") != proxy.URL {
			t.Errorf("credential = %+v", credential)
		}
		var generated oauthRequest
		for _, request := range proxy.log() {
			if request.path == "/key/generate" {
				generated = request
			}
		}
		if generated.method != "POST" || generated.header.Get("Authorization") != "Bearer "+jwt {
			t.Errorf("key/generate = %+v", generated)
		}
		// The stored credential feeds ToAuth through createProviderAuth.
		auth := createProviderAuth(defaultDefinition(t), nil, &defaultLoginFlows, nil)
		resolved, err := auth.OAuth.ToAuth(credential)
		if err != nil || resolved.APIKey != "sk-virtual-abc" {
			t.Errorf("ToAuth = %+v, %v", resolved, err)
		}
	})

	t.Run("strips the Bearer prefix from the pasted token", func(t *testing.T) {
		oauthTest(t)
		proxy := pasteProxy(t)
		proxy.handle("/key/generate", jsonHandler(200, map[string]any{"key": "sk-stripped"}))
		if _, err := loginOAuth(context.Background(), sso(proxy, "  Bearer  "+jwt+"  ", "y").interaction(), defaultDefinition(t)); err != nil {
			t.Fatal(err)
		}
		for _, request := range proxy.log() {
			if request.path == "/key/generate" && request.header.Get("Authorization") != "Bearer "+jwt {
				t.Errorf("Authorization = %q", request.header.Get("Authorization"))
			}
		}
	})

	t.Run("honors the expiry returned with a generated virtual key", func(t *testing.T) {
		oauthTest(t)
		proxy := pasteProxy(t)
		expiresAt := time.Now().Add(time.Hour).UTC().Truncate(time.Millisecond)
		proxy.handle("/key/generate", jsonHandler(200, map[string]any{"key": "sk-expiring", "expires": expiresAt.Format("2006-01-02T15:04:05.000Z")}))
		credential, err := loginOAuth(context.Background(), sso(proxy, jwt, "y").interaction(), defaultDefinition(t))
		if err != nil || credential.Access != "sk-expiring" || credential.ExpiresMillis() != float64(expiresAt.UnixMilli()-5*60*1000) {
			t.Fatalf("credential = %+v, %v", credential, err)
		}
		// Without a refresh path, the expiring virtual key cannot be refreshed.
		if _, err := refreshOf(t, credential); err == nil || err.Error() != "LiteLLM credential cannot be refreshed; run /login litellm again" {
			t.Errorf("refresh err = %v", err)
		}
	})

	t.Run("falls back to the JWT when virtual key generation times out", func(t *testing.T) {
		oauthTest(t)
		loginTimeout = 30 * time.Millisecond
		proxy := pasteProxy(t)
		proxy.handle("/key/generate", func(w http.ResponseWriter, r *http.Request) { <-r.Context().Done() })
		script := sso(proxy, jwt, "y")
		credential, err := loginOAuth(context.Background(), script.interaction(), defaultDefinition(t))
		if err != nil || credential.Access != jwt || credential.Refresh != "" {
			t.Fatalf("credential = %+v, %v", credential, err)
		}
		progress := script.eventsOf("progress")
		if last := progress[len(progress)-1].(ai.AuthProgressEvent).Message; !strings.Contains(last, "virtual key generation failed") {
			t.Errorf("progress = %v", progress)
		}
	})

	t.Run("rejects when the caller cancels virtual key generation", func(t *testing.T) {
		oauthTest(t)
		proxy := pasteProxy(t)
		started := make(chan struct{})
		proxy.handle("/key/generate", func(w http.ResponseWriter, r *http.Request) {
			close(started)
			<-r.Context().Done()
		})
		ctx, cancel := context.WithCancelCause(context.Background())
		reason := errors.New("caller cancelled login")
		done := make(chan error, 1)
		go func() {
			_, err := loginOAuth(ctx, sso(proxy, jwt, "y").interaction(), defaultDefinition(t))
			done <- err
		}()
		<-started
		cancel(reason)
		if err := <-done; !errors.Is(err, reason) {
			t.Errorf("err = %v", err)
		}
	})

	t.Run("uses the JWT directly when the user answers no", func(t *testing.T) {
		oauthTest(t)
		proxy := pasteProxy(t)
		credential, err := loginOAuth(context.Background(), sso(proxy, jwt, "no").interaction(), defaultDefinition(t))
		if err != nil || credential.Access != jwt || credential.Refresh != "" || credential.ExpiresMillis() >= float64(permanentTokenExpiresAt) {
			t.Fatalf("credential = %+v, %v", credential, err)
		}
		if proxy.count("/key/generate") != 0 {
			t.Error("a virtual key was requested")
		}
	})

	t.Run("falls back to the JWT when virtual key generation fails", func(t *testing.T) {
		oauthTest(t)
		proxy := pasteProxy(t)
		proxy.handle("/key/generate", jsonHandler(403, map[string]any{"error": "forbidden", "token": "remote-secret"}))
		script := sso(proxy, jwt, "y")
		credential, err := loginOAuth(context.Background(), script.interaction(), defaultDefinition(t))
		if err != nil || credential.Access != jwt {
			t.Fatalf("credential = %+v, %v", credential, err)
		}
		var messages []string
		for _, event := range script.eventsOf("progress") {
			messages = append(messages, event.(ai.AuthProgressEvent).Message)
		}
		joined := strings.Join(messages, " ")
		if !strings.Contains(joined, "virtual key generation failed") || strings.Contains(joined, "remote-secret") {
			t.Errorf("progress = %q", joined)
		}
	})

	t.Run("throws when the SSO token is empty", func(t *testing.T) {
		oauthTest(t)
		proxy := pasteProxy(t)
		if _, err := loginOAuth(context.Background(), sso(proxy, "", "").interaction(), defaultDefinition(t)); err == nil || err.Error() != "SSO token is required" {
			t.Errorf("err = %v", err)
		}
	})
}

// ---- direct OIDC ------------------------------------------------------------------------------------------

type oidcOptions struct {
	oidc          any
	hasOIDC       bool
	headers       string
	discovery     map[string]any
	discoveryFunc http.HandlerFunc
	claims        map[string]any
	tokenBody     map[string]any
	token         http.HandlerFunc
	callbackQuery func(state string) string
}

type oidcRun struct {
	credential   ai.Credential
	err          error
	idp          *recorder
	authURL      *url.URL
	issued       []string
	callbackResp *http.Response
	script       *scriptedAuth
}

func runOidcLogin(t *testing.T, options oidcOptions) oidcRun {
	t.Helper()
	oauthTest(t)
	idp := newRecorder(t, true)
	authTransport = idp.Client().Transport
	issuer := idp.URL
	clientID := "example-client-id"
	oidc := options.oidc
	if !options.hasOIDC {
		oidc = map[string]any{"issuer": issuer, "clientId": clientID}
	} else if mapped, ok := oidc.(map[string]any); ok {
		oidc = merged(mapped, nil)
		if mapped["issuer"] == "ISSUER" {
			oidc.(map[string]any)["issuer"] = issuer
		}
	}
	definition := defaultDefinition(t)
	definition.OIDC = oidc
	definition.HasOIDC = true
	t.Setenv(envHeaders, `{"x-gateway-secret":"gateway-secret"}`)
	if options.headers != "" {
		definition.Headers = options.headers
	}
	run := oidcRun{idp: idp}
	var authMu sync.Mutex
	idp.handle(oidcDiscoveryPath, func(w http.ResponseWriter, r *http.Request) {
		if options.discoveryFunc != nil {
			options.discoveryFunc(w, r)
			return
		}
		writeJSON(w, 200, merged(map[string]any{
			"issuer": issuer, "authorization_endpoint": issuer + "/authorize?tenant=alpha&client_id=wrong", "token_endpoint": issuer + "/token",
			"code_challenge_methods_supported": []string{"S256"},
		}, options.discovery))
	})
	idp.handle("/token", func(w http.ResponseWriter, r *http.Request) {
		if options.token != nil {
			options.token(w, r)
			return
		}
		authMu.Lock()
		nonce := run.authURL.Query().Get("nonce")
		authMu.Unlock()
		claims := merged(map[string]any{"iss": issuer, "aud": clientID, "sub": "user-123", "exp": time.Now().Unix() + 3600, "nonce": nonce}, options.claims)
		if claims["iss"] == "SLASH" {
			claims["iss"] = issuer + "/"
		}
		token := idToken(claims)
		authMu.Lock()
		run.issued = append(run.issued, token)
		authMu.Unlock()
		writeJSON(w, 200, merged(map[string]any{"id_token": token, "refresh_token": "refresh-new", "access_token": "idp-access-token", "token_type": "Bearer", "expires_in": 3600}, options.tokenBody))
	})
	responses := make(chan *http.Response, 1)
	script := &scriptedAuth{answer: answers("https://proxy.example.com", "", "")}
	script.onEvent = func(event ai.AuthEvent) {
		urlEvent, ok := event.(ai.AuthURLEvent)
		if !ok {
			return
		}
		parsed, _ := url.Parse(urlEvent.URL)
		authMu.Lock()
		run.authURL = parsed
		authMu.Unlock()
		state := parsed.Query().Get("state")
		query := "code=authorization-code&state=" + url.QueryEscape(state)
		if options.callbackQuery != nil {
			query = options.callbackQuery(state)
		}
		go func() {
			if response, err := http.Get(parsed.Query().Get("redirect_uri") + "?" + query); err == nil {
				responses <- response
			}
		}()
	}
	run.script = script
	run.credential, run.err = loginOAuth(context.Background(), script.interaction(), definition)
	select {
	case run.callbackResp = <-responses:
	case <-time.After(300 * time.Millisecond):
	}
	return run
}

func TestDirectOIDCLogin(t *testing.T) {
	t.Run("signs in with the identity provider and stores its id_token", func(t *testing.T) {
		exp := time.Now().Unix() + 3600
		run := runOidcLogin(t, oidcOptions{claims: map[string]any{"exp": exp}})
		if run.err != nil {
			t.Fatal(run.err)
		}
		issuer := run.idp.URL
		requests := run.idp.log()
		if len(requests) != 2 || requests[0].path != oidcDiscoveryPath || requests[0].method != "GET" || requests[1].path != "/token" || requests[1].method != "POST" {
			t.Fatalf("requests = %+v", requests)
		}
		for _, request := range requests {
			if request.header.Get("x-gateway-secret") != "" || request.header.Get("Authorization") != "" {
				t.Errorf("gateway headers reached the IdP: %v", request.header)
			}
		}
		query := run.authURL.Query()
		redirect := query.Get("redirect_uri")
		if !regexp.MustCompile(`^http://127\.0\.0\.1:\d+/callback$`).MatchString(redirect) || run.authURL.Path != "/authorize" {
			t.Errorf("auth URL = %s", run.authURL)
		}
		pattern := regexp.MustCompile(`^[\w-]{43}$`)
		for _, name := range []string{"state", "nonce", "code_challenge"} {
			if !pattern.MatchString(query.Get(name)) {
				t.Errorf("%s = %q", name, query.Get(name))
			}
		}
		if query.Get("tenant") != "alpha" || query.Get("response_type") != "code" || query.Get("scope") != "openid" || query.Get("code_challenge_method") != "S256" || len(query["client_id"]) != 1 || query.Get("client_id") != "example-client-id" {
			t.Errorf("query = %v", query)
		}
		form := queryOf(t, requests[1].body)
		// No client_secret: the exact form proves a public client.
		if len(form) != 5 || form["grant_type"] != "authorization_code" || form["code"] != "authorization-code" || form["redirect_uri"] != redirect || form["client_id"] != "example-client-id" {
			t.Errorf("form = %v", form)
		}
		digest := sha256.Sum256([]byte(form["code_verifier"]))
		if base64.RawURLEncoding.EncodeToString(digest[:]) != query.Get("code_challenge") {
			t.Error("verifier does not match the challenge")
		}
		if run.callbackResp == nil || run.callbackResp.StatusCode != 200 || run.callbackResp.Header.Get("Cache-Control") != "no-store" {
			t.Errorf("callback = %+v", run.callbackResp)
		}
		credential := run.credential
		if credential.Access != run.issued[0] || credential.Refresh != "refresh-new" || credential.ExpiresMillis() != float64(exp*1000-5*60*1000) {
			t.Errorf("credential = %+v", credential)
		}
		for name, want := range map[string]string{"baseUrl": "https://proxy.example.com", "flow": oidcFlow, "issuer": issuer, "clientId": "example-client-id", "tokenEndpoint": issuer + "/token", "subject": "user-123"} {
			if got := extra(t, credential, name); got != want {
				t.Errorf("%s = %q, want %q", name, got, want)
			}
		}
		auth := createProviderAuth(defaultDefinition(t), nil, &defaultLoginFlows, nil)
		if resolved, err := auth.OAuth.ToAuth(credential); err != nil || resolved.APIKey != run.issued[0] {
			t.Errorf("ToAuth = %+v, %v", resolved, err)
		}
	})

	t.Run("keeps provider headers off IdP requests", func(t *testing.T) {
		run := runOidcLogin(t, oidcOptions{headers: `{"x-provider-secret":"provider-secret"}`})
		if run.err != nil || len(run.idp.log()) != 2 {
			t.Fatalf("err = %v, requests %d", run.err, len(run.idp.log()))
		}
		for _, request := range run.idp.log() {
			if request.header.Get("x-provider-secret") != "" {
				t.Error("provider header reached the IdP")
			}
		}
	})

	t.Run("stores an empty refresh token when the IdP issues none", func(t *testing.T) {
		run := runOidcLogin(t, oidcOptions{tokenBody: map[string]any{"refresh_token": nil}})
		if run.err != nil || run.credential.Refresh != "" || extra(t, run.credential, "flow") != oidcFlow {
			t.Errorf("credential = %+v, %v", run.credential, run.err)
		}
	})

	t.Run("uses the first free configured redirect port", func(t *testing.T) {
		occupied, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		defer occupied.Close()
		free, _ := net.Listen("tcp", "127.0.0.1:0")
		freePort := free.Addr().(*net.TCPAddr).Port
		free.Close()
		run := runOidcLogin(t, oidcOptions{hasOIDC: true, oidc: map[string]any{
			"issuer": "ISSUER", "clientId": "example-client-id", "redirectPorts": []any{float64(occupied.Addr().(*net.TCPAddr).Port), float64(freePort)},
		}})
		if run.err != nil {
			t.Fatal(run.err)
		}
		if got := run.authURL.Query().Get("redirect_uri"); got != fmt.Sprintf("http://127.0.0.1:%d/callback", freePort) {
			t.Errorf("redirect_uri = %q", got)
		}
	})

	t.Run("accepts an audience array that contains the client ID and a configured scope", func(t *testing.T) {
		run := runOidcLogin(t, oidcOptions{
			hasOIDC: true, oidc: map[string]any{"issuer": "ISSUER", "clientId": "example-client-id", "scope": "openid offline_access"},
			claims: map[string]any{"aud": []any{"other-client", "example-client-id"}, "azp": "example-client-id"},
		})
		if run.err != nil || run.authURL.Query().Get("scope") != "openid offline_access" {
			t.Errorf("err = %v, scope %q", run.err, run.authURL.Query().Get("scope"))
		}
	})

	redirect302 := func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Location", "https://other.example.com/next")
		w.WriteHeader(302)
	}
	// Discovery is this package's; the id_token, token endpoint and callback checks are the library's.
	rejections := []struct {
		name    string
		options oidcOptions
		message string
	}{
		{"a discovery issuer mismatch", oidcOptions{discovery: map[string]any{"issuer": "https://other.example.com"}}, "OIDC discovery issuer does not match the configured issuer"},
		{"a non-https authorization endpoint", oidcOptions{discovery: map[string]any{"authorization_endpoint": "http://idp.example.com/authorize"}}, "OIDC discovery has invalid authorization_endpoint"},
		{"a non-https token endpoint", oidcOptions{discovery: map[string]any{"token_endpoint": "http://idp.example.com/token"}}, "OIDC discovery has invalid token_endpoint"},
		{"no S256 support", oidcOptions{discovery: map[string]any{"code_challenge_methods_supported": []string{"plain"}}}, "OIDC discovery does not support S256"},
		{"a discovery redirect", oidcOptions{discoveryFunc: redirect302}, "OIDC discovery failed (HTTP 302)"},
	}
	for _, c := range rejections {
		t.Run("rejects "+c.name, func(t *testing.T) {
			run := runOidcLogin(t, c.options)
			if run.err == nil || run.err.Error() != c.message {
				t.Errorf("err = %v, want %q", run.err, c.message)
			}
			if run.credential.Access != "" {
				t.Error("a credential was returned")
			}
		})
	}

	t.Run("rejects an id_token with the wrong nonce", func(t *testing.T) {
		run := runOidcLogin(t, oidcOptions{claims: map[string]any{"nonce": "other-nonce"}})
		if !errors.Is(run.err, litellmauth.ErrProtocol) || !strings.Contains(run.err.Error(), "nonce") {
			t.Errorf("err = %v", run.err)
		}
		if run.credential.Access != "" {
			t.Error("a credential was returned")
		}
	})

	invalidConfig := []struct {
		name  string
		oidc  any
		field string
	}{
		{"a string", testIssuerF, "oidc"},
		{"null", nil, "oidc"},
		{"a missing issuer", map[string]any{"clientId": "c"}, "oidc.issuer"},
		{"an http issuer", map[string]any{"issuer": "http://idp.example.com", "clientId": "c"}, "oidc.issuer"},
		{"issuer credentials", map[string]any{"issuer": "https://user:pass@idp.example.com", "clientId": "c"}, "oidc.issuer"},
		{"an issuer query", map[string]any{"issuer": testIssuerF + "?tenant=alpha", "clientId": "c"}, "oidc.issuer"},
		{"an issuer fragment", map[string]any{"issuer": testIssuerF + "#alpha", "clientId": "c"}, "oidc.issuer"},
		{"a missing client ID", map[string]any{"issuer": testIssuerF}, "oidc.clientId"},
		{"an empty client ID", map[string]any{"issuer": testIssuerF, "clientId": ""}, "oidc.clientId"},
		{"a scope without openid", map[string]any{"issuer": testIssuerF, "clientId": "c", "scope": "profile email"}, "oidc.scope"},
		{"a non-array redirectPorts", map[string]any{"issuer": testIssuerF, "clientId": "c", "redirectPorts": float64(8400)}, "oidc.redirectPorts"},
		{"a zero port", map[string]any{"issuer": testIssuerF, "clientId": "c", "redirectPorts": []any{float64(0)}}, "oidc.redirectPorts"},
		{"an out-of-range port", map[string]any{"issuer": testIssuerF, "clientId": "c", "redirectPorts": []any{float64(8400), float64(65536)}}, "oidc.redirectPorts"},
		{"a fractional port", map[string]any{"issuer": testIssuerF, "clientId": "c", "redirectPorts": []any{8400.5}}, "oidc.redirectPorts"},
	}
	for _, c := range invalidConfig {
		t.Run("fails login without reaching LiteLLM when oidc is "+c.name, func(t *testing.T) {
			oauthTest(t)
			proxy := newRecorder(t, false)
			definition := defaultDefinition(t)
			definition.OIDC = c.oidc
			definition.HasOIDC = true
			script := &scriptedAuth{answer: answers(proxy.URL, "", "")}
			_, err := loginOAuth(context.Background(), script.interaction(), definition)
			if err == nil || !strings.HasPrefix(err.Error(), "Invalid LiteLLM "+c.field+" setting: ") {
				t.Errorf("err = %v", err)
			}
			if len(proxy.log()) != 0 || len(script.prompts) != 0 {
				t.Errorf("login progressed: requests %d prompts %d", len(proxy.log()), len(script.prompts))
			}
		})
	}
}

// ---- OIDC refresh -----------------------------------------------------------------------------------------

func TestOIDCRefresh(t *testing.T) {
	const clientID = "example-client-id"
	exp := testNow/1000 + 3600
	setup := func(t *testing.T) (*recorder, string) {
		oauthTest(t)
		pinNow(testNow)
		t.Setenv(envHeaders, `{"x-gateway-secret":"gateway-secret"}`)
		idp := newRecorder(t, true)
		authTransport = idp.Client().Transport
		return idp, idp.URL
	}
	credentialFor := func(issuer string, mutate ...func(*ai.Credential)) ai.Credential {
		credential := newOAuthCredential(idToken(map[string]any{"iss": issuer, "aud": clientID, "sub": "user-123", "exp": testNow/1000 + 60}), "refresh-old", float64(testNow+60_000), "https://proxy.example.com")
		setExtra(&credential, "flow", oidcFlow)
		setExtra(&credential, "issuer", issuer)
		setExtra(&credential, "clientId", clientID)
		setExtra(&credential, "tokenEndpoint", issuer+"/token")
		setExtra(&credential, "subject", "user-123")
		for _, fn := range mutate {
			fn(&credential)
		}
		return credential
	}
	fresh := func(issuer, sub string, lifetime int64) string {
		return idToken(map[string]any{"iss": issuer, "aud": clientID, "sub": sub, "exp": testNow/1000 + lifetime})
	}

	t.Run("rotates the id_token and refresh token", func(t *testing.T) {
		idp, issuer := setup(t)
		token := fresh(issuer, "user-123", 3600)
		idp.handle("/token", jsonHandler(200, map[string]any{"id_token": token, "refresh_token": "refresh-new", "token_type": "Bearer", "expires_in": 3600}))
		refreshed, err := refreshOf(t, credentialFor(issuer))
		if err != nil || refreshed.Access != token || refreshed.Refresh != "refresh-new" || refreshed.ExpiresMillis() != float64(exp*1000-300_000) {
			t.Fatalf("refreshed = %+v, %v", refreshed, err)
		}
		if extra(t, refreshed, "subject") != "user-123" || extra(t, refreshed, "flow") != oidcFlow || extra(t, refreshed, "baseUrl") != "https://proxy.example.com" {
			t.Errorf("metadata = %v", refreshed.Extra)
		}
		requests := idp.log()
		want := map[string]string{"grant_type": "refresh_token", "refresh_token": "refresh-old", "client_id": clientID}
		if len(requests) != 1 || !equalMaps(queryOf(t, requests[0].body), want) || requests[0].header.Get("x-gateway-secret") != "" {
			t.Errorf("requests = %+v", requests)
		}
	})

	t.Run("keeps the refresh token when the IdP does not rotate it", func(t *testing.T) {
		idp, issuer := setup(t)
		token := fresh(issuer, "user-123", 3600)
		idp.handle("/token", jsonHandler(200, map[string]any{"id_token": token}))
		refreshed, err := refreshOf(t, credentialFor(issuer))
		if err != nil || refreshed.Access != token || refreshed.Refresh != "refresh-old" {
			t.Errorf("refreshed = %+v, %v", refreshed, err)
		}
	})

	for name, c := range map[string]struct {
		lifetime int64
		expires  int64
	}{
		"a long-lived":  {3600, testNow + 3_600_000 - 300_000},
		"a short-lived": {120, testNow + 60_000},
	} {
		t.Run("schedules the next refresh of "+name+" id_token before it expires", func(t *testing.T) {
			idp, issuer := setup(t)
			idp.handle("/token", jsonHandler(200, map[string]any{"id_token": fresh(issuer, "user-123", c.lifetime)}))
			refreshed, err := refreshOf(t, credentialFor(issuer))
			if err != nil || int64(refreshed.ExpiresMillis()) != c.expires {
				t.Errorf("expires = %v, want %d (%v)", refreshed.ExpiresMillis(), c.expires, err)
			}
		})
	}

	renewCases := []struct {
		name    string
		respond func(issuer string) http.HandlerFunc
		message string
	}{
		{"a changed subject", func(i string) http.HandlerFunc {
			return jsonHandler(200, map[string]any{"id_token": fresh(i, "user-456", 3600)})
		}, "returned an invalid response"},
		{"a missing id_token", func(string) http.HandlerFunc {
			return jsonHandler(200, map[string]any{"access_token": "idp-access-token"})
		}, "returned an invalid response"},
		{"an id_token for another client", func(i string) http.HandlerFunc {
			return jsonHandler(200, map[string]any{"id_token": idToken(map[string]any{"iss": i, "aud": "other-client", "sub": "user-123", "exp": exp})})
		}, "returned an invalid response"},
		{"invalid_grant", func(string) http.HandlerFunc {
			return jsonHandler(400, map[string]any{"error": "invalid_grant", "error_description": "refresh-old secret-description"})
		}, "OIDC token exchange rejected (invalid_grant)"},
	}
	for _, c := range renewCases {
		t.Run("requires a new login after "+c.name, func(t *testing.T) {
			idp, issuer := setup(t)
			idp.handle("/token", c.respond(issuer))
			_, err := refreshOf(t, credentialFor(issuer))
			if err == nil || !strings.Contains(err.Error(), c.message) || !strings.HasSuffix(err.Error(), "; run /login litellm again") ||
				strings.Contains(err.Error(), "refresh-old") || strings.Contains(err.Error(), "secret-description") {
				t.Errorf("err = %v", err)
			}
		})
	}

	failures := map[string]struct {
		handler http.HandlerFunc
		reason  string
	}{
		"429":     {jsonHandler(429, map[string]any{}), "HTTP 429"},
		"503":     {jsonHandler(503, map[string]any{}), "HTTP 503"},
		"network": {dropConnection, "network error"},
	}
	for name, c := range failures {
		t.Run("keeps the credential before expiry after "+name, func(t *testing.T) {
			idp, issuer := setup(t)
			idp.handle("/token", c.handler)
			credential := credentialFor(issuer)
			refreshed, err := refreshOf(t, credential)
			if err != nil || refreshed.Access != credential.Access || refreshed.Refresh != "refresh-old" {
				t.Errorf("refreshed = %+v, %v", refreshed, err)
			}
		})
		t.Run("fails without asking for a new login after "+name+" once the id_token has expired", func(t *testing.T) {
			idp, issuer := setup(t)
			idp.handle("/token", c.handler)
			_, err := refreshOf(t, credentialFor(issuer, func(c *ai.Credential) { c.SetExpiresMillis(float64(testNow)) }))
			if err == nil || err.Error() != "OIDC token exchange failed ("+c.reason+")" {
				t.Errorf("err = %v", err)
			}
		})
	}

	stored := map[string]func(*ai.Credential){
		"empty refresh token": func(c *ai.Credential) { c.Refresh = "" },
		"http token endpoint": func(c *ai.Credential) { setExtra(c, "tokenEndpoint", "http://idp.example.com/token") },
		"token endpoint user": func(c *ai.Credential) { setExtra(c, "tokenEndpoint", "https://user@idp.example.com/token") },
		"empty client ID":     func(c *ai.Credential) { setExtra(c, "clientId", "") },
		"missing issuer":      func(c *ai.Credential) { delete(c.Extra, "issuer") },
		"empty subject":       func(c *ai.Credential) { setExtra(c, "subject", "") },
	}
	for name, mutate := range stored {
		t.Run("requires a new login without contacting the IdP when stored "+name, func(t *testing.T) {
			idp, issuer := setup(t)
			idp.handle("/token", jsonHandler(200, map[string]any{}))
			_, err := refreshOf(t, credentialFor(issuer, mutate))
			if err == nil || !strings.HasSuffix(err.Error(), "; run /login litellm again") || len(idp.log()) != 0 {
				t.Errorf("err = %v, requests %d", err, len(idp.log()))
			}
		})
	}

	t.Run("never executes a refresh token that starts with !", func(t *testing.T) {
		idp, issuer := setup(t)
		helper, runs := writeHelper(t, t.TempDir(), []string{"executed-token"}, "helper.sh")
		token := fresh(issuer, "user-123", 3600)
		idp.handle("/token", jsonHandler(200, map[string]any{"id_token": token}))
		credential := credentialFor(issuer, func(c *ai.Credential) { c.Refresh = "!" + helper })
		refreshed, err := refreshOf(t, credential)
		if err != nil || refreshed.Access != token || refreshed.Refresh != "!"+helper {
			t.Fatalf("refreshed = %+v, %v", refreshed, err)
		}
		if got := queryOf(t, idp.log()[0].body)["refresh_token"]; got != "!"+helper {
			t.Errorf("refresh_token = %q", got)
		}
		idp.handle("/token", jsonHandler(400, map[string]any{"error": "invalid_grant"}))
		if _, err := refreshOf(t, credential); err == nil || err.Error() != "OIDC token exchange rejected (invalid_grant); run /login litellm again" {
			t.Errorf("err = %v", err)
		}
		if runs() != 0 {
			t.Errorf("helper ran %d times", runs())
		}
	})
}

// ---- login base URL reuse and startup ---------------------------------------------------------------------

func writeStoredOAuth(t *testing.T, dir, baseURL string) {
	t.Helper()
	data, _ := json.Marshal(map[string]any{"litellm": map[string]any{"type": "oauth", "access": "expired-token", "refresh": "", "expires": 0, "baseUrl": baseURL}})
	if err := os.WriteFile(filepath.Join(dir, "auth.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func TestLoginBaseURLReuse(t *testing.T) {
	const storedURL = "https://stored.example.com"
	// An unreachable proxy: every flow below stops at the paste prompt only after CLI SSO is unavailable, so
	// give them a server that answers 404 to every probe.
	notFound := func(t *testing.T) string { return newRecorder(t, false).URL }

	t.Run("offers the stored credential's base URL instead of asking for it again", func(t *testing.T) {
		dir := oauthTest(t)
		writeStoredOAuth(t, dir, storedURL)
		baseURL, err := promptBaseURL(context.Background(), (&scriptedAuth{answer: answers("", "", "")}).interaction(), defaultDefinition(t))
		if err != nil || baseURL != storedURL {
			t.Errorf("baseURL = %q, %v", baseURL, err)
		}
		script := &scriptedAuth{answer: answers("", "", "")}
		_, _ = promptBaseURL(context.Background(), script.interaction(), defaultDefinition(t))
		for _, prompt := range script.prompts {
			if strings.Contains(promptMessage(prompt), "Enter LiteLLM proxy URL") {
				t.Error("asked for the URL again")
			}
		}
	})

	t.Run("names where the offered base URL came from", func(t *testing.T) {
		dir := oauthTest(t)
		writeStoredOAuth(t, dir, storedURL)
		script := &scriptedAuth{answer: answers("", "", "")}
		if _, err := promptBaseURL(context.Background(), script.interaction(), defaultDefinition(t)); err != nil {
			t.Fatal(err)
		}
		options := script.prompts[0].(ai.AuthSelectPrompt).Options
		if len(options) != 2 || options[0].Label != storedURL+" (previous login)" || options[0].ID != storedURL || options[1].Label != "Enter a different URL…" {
			t.Errorf("options = %+v", options)
		}
	})

	t.Run("still asks for a URL when the offered one is declined", func(t *testing.T) {
		dir := oauthTest(t)
		writeStoredOAuth(t, dir, storedURL)
		other := notFound(t)
		script := &scriptedAuth{answer: func(prompt ai.AuthPrompt) (string, error) {
			if selection, ok := prompt.(ai.AuthSelectPrompt); ok {
				return selection.Options[1].ID, nil
			}
			return other, nil
		}}
		baseURL, err := promptBaseURL(context.Background(), script.interaction(), defaultDefinition(t))
		if got := script.promptTypes(); err != nil || baseURL != other || len(got) != 2 || got[0] != "select" || got[1] != "text" {
			t.Errorf("baseURL = %q, %v, prompts %v", baseURL, err, got)
		}
	})

	t.Run("offers LITELLM_BASE_URL to the API-key login", func(t *testing.T) {
		oauthTest(t)
		t.Setenv(envBaseURL, "https://env.example.com")
		script := &scriptedAuth{answer: answers("", "sk-typed", "")}
		credential, err := loginAPIKey(context.Background(), script.interaction(), defaultDefinition(t))
		if err != nil || credential.Env[envBaseURL] != "https://env.example.com" || credential.Key != "sk-typed" || credential.Type != ai.CredentialAPIKey {
			t.Errorf("credential = %+v, %v", credential, err)
		}
		if got := script.promptTypes(); len(got) != 2 || got[0] != "select" || got[1] != "secret" {
			t.Errorf("prompts = %v", got)
		}
		if label := script.prompts[0].(ai.AuthSelectPrompt).Options[0].Label; label != "https://env.example.com ($LITELLM_BASE_URL)" {
			t.Errorf("label = %q", label)
		}
	})

	t.Run("ignores a stored base URL that is no longer usable", func(t *testing.T) {
		dir := oauthTest(t)
		data, _ := json.Marshal(map[string]any{"litellm": map[string]any{"type": "api_key", "key": "sk-old", "env": map[string]string{envBaseURL: "http://insecure.example.com"}}})
		os.WriteFile(filepath.Join(dir, "auth.json"), data, 0o600)
		script := &scriptedAuth{answer: answers("https://typed.example.com", "sk-typed", "")}
		credential, err := loginAPIKey(context.Background(), script.interaction(), defaultDefinition(t))
		if got := script.promptTypes(); err != nil || len(got) != 2 || got[0] != "text" || got[1] != "secret" || credential.Env[envBaseURL] != "https://typed.example.com" {
			t.Errorf("credential = %+v, %v, prompts %v", credential, err, got)
		}
	})

	t.Run("asks for the proxy URL when nothing is configured yet", func(t *testing.T) {
		oauthTest(t)
		script := &scriptedAuth{answer: answers("https://typed.example.com", "sk-typed", "")}
		credential, err := loginAPIKey(context.Background(), script.interaction(), defaultDefinition(t))
		if got := script.promptTypes(); err != nil || len(got) != 2 || got[0] != "text" || credential.Env[envBaseURL] != "https://typed.example.com" {
			t.Errorf("credential = %+v, %v, prompts %v", credential, err, got)
		}
	})

	t.Run("requires a base URL and an API key", func(t *testing.T) {
		oauthTest(t)
		if _, err := loginAPIKey(context.Background(), (&scriptedAuth{answer: answers("  ", "", "")}).interaction(), defaultDefinition(t)); err == nil || err.Error() != "Base URL is required" {
			t.Errorf("err = %v", err)
		}
		if _, err := loginAPIKey(context.Background(), (&scriptedAuth{answer: answers("https://typed.example.com", " ", "")}).interaction(), defaultDefinition(t)); err == nil || err.Error() != "Both base URL and API key are required" {
			t.Errorf("err = %v", err)
		}
	})
}

func TestLoginStartup(t *testing.T) {
	t.Run("leaves /login litellm to the registered OAuth provider", func(t *testing.T) {
		oauthTest(t)
		if defaultLoginFlows.LoginAPIKey == nil || defaultLoginFlows.LoginOAuth == nil || defaultLoginFlows.Refresh == nil {
			t.Fatalf("flows not plugged in: %+v", defaultLoginFlows)
		}
		auth := createProviderAuth(defaultDefinition(t), nil, &defaultLoginFlows, nil)
		if auth.OAuth == nil || auth.APIKey == nil || auth.APIKey.Login == nil {
			t.Errorf("auth = %+v", auth)
		}
		alias := providerDefinition{Name: "other", DisplayName: "Other"}
		if aliasAuth := createProviderAuth(alias, nil, &defaultLoginFlows, nil); aliasAuth.OAuth != nil || aliasAuth.APIKey.Login != nil {
			t.Error("an alias offers interactive login")
		}
	})

	t.Run("API-key login normalizes input and makes no network call", func(t *testing.T) {
		oauthTest(t)
		script := &scriptedAuth{answer: answers(" http://127.0.0.1:4000/v1 ", " sk-login ", "")}
		credential, err := defaultLoginFlows.LoginAPIKey(context.Background(), script.interaction(), defaultDefinition(t))
		if err != nil || credential.Key != "sk-login" || credential.Env[envBaseURL] != "http://127.0.0.1:4000" {
			t.Errorf("credential = %+v, %v", credential, err)
		}
	})

	t.Run("does not re-run command-backed helpers after refreshing login credentials", func(t *testing.T) {
		dir := oauthTest(t)
		helper, runs := writeHelper(t, dir, []string{"first", "second", "unexpected-third"}, "litellm-token-helper.sh")
		script := &scriptedAuth{answer: answers("https://proxy.example.com", "!"+helper, "")}
		credential, err := defaultLoginFlows.LoginAPIKey(context.Background(), script.interaction(), defaultDefinition(t))
		if err != nil {
			t.Fatal(err)
		}
		auth := createProviderAuth(defaultDefinition(t), nil, &defaultLoginFlows, nil)
		resolve := func() string {
			result, err := auth.APIKey.Resolve(context.Background(), ai.APIKeyAuthInput{Ctx: contextEnv(nil), Credential: &credential})
			if err != nil || result == nil {
				t.Fatalf("resolve = %+v, %v", result, err)
			}
			return result.Auth.APIKey
		}
		if first, second := resolve(), resolve(); first != "first" || second != "second" || runs() != 2 {
			t.Errorf("keys = %q %q, runs %d", first, second, runs())
		}
	})

	t.Run("executes an OAuth refresh command only during refresh", func(t *testing.T) {
		dir := oauthTest(t)
		helper, runs := writeHelper(t, dir, []string{"refreshed-token", "unexpected-second-run"}, "litellm-token-helper.sh")
		credential := newOAuthCredential("expired-token", "!"+helper, 0, "https://proxy.example.com")
		refreshed, err := refreshOf(t, credential)
		if err != nil || refreshed.Access != "refreshed-token" || runs() != 1 {
			t.Fatalf("refreshed = %+v, %v, runs %d", refreshed, err, runs())
		}
		auth := createProviderAuth(defaultDefinition(t), nil, &defaultLoginFlows, nil)
		resolved, err := auth.OAuth.Refresh(context.Background(), credential)
		if err != nil || resolved.Type != ai.CredentialOAuth {
			t.Errorf("auth refresh = %+v, %v", resolved, err)
		}
		before := runs()
		if model, err := auth.OAuth.ToAuth(refreshed); err != nil || model.APIKey != "refreshed-token" || runs() != before {
			t.Errorf("ToAuth = %+v, %v, runs %d->%d", model, err, before, runs())
		}
	})

	t.Run("uses the login cache timestamp for the stale auto-refresh deadline", func(t *testing.T) {
		oauthTest(t)
		loginTime := time.Date(2026, 5, 1, 0, 0, 0, 0, time.UTC).UnixMilli()
		pinNow(loginTime)
		proxy := pasteProxy(t)
		script := &scriptedAuth{answer: answers(proxy.URL, "opaque-sso-token", "n")}
		credential, err := loginOAuth(context.Background(), script.interaction(), defaultDefinition(t))
		if err != nil {
			t.Fatal(err)
		}
		// An opaque token with no virtual key never expires; a JWT-less CLI token uses the login time.
		if credential.ExpiresMillis() != float64(permanentTokenExpiresAt) {
			t.Errorf("expires = %v", credential.ExpiresMillis())
		}
		proxy.handle("/sso/cli/start", jsonHandler(200, map[string]any{"login_id": "login-1", "poll_secret": "poll-secret", "user_code": "ABCD-EFGH"}))
		proxy.handle("/sso/cli/poll/login-1", jsonHandler(200, map[string]any{"status": "ready", "key": "opaque-cli-token"}))
		cli, err := loginOAuth(context.Background(), (&scriptedAuth{answer: answers(proxy.URL, "", "")}).interaction(), defaultDefinition(t))
		if want := float64(loginTime + 24*60*60*1000); err != nil || cli.ExpiresMillis() != want {
			t.Errorf("cli expires = %v, want %v (%v)", cli.ExpiresMillis(), want, err)
		}
	})
}

// ---- security review follow-ups ---------------------------------------------------------------------------

func TestOidcNullSettingMessage(t *testing.T) {
	oauthTest(t)
	definition := defaultDefinition(t)
	definition.HasOIDC = true
	_, err := loginOAuth(context.Background(), (&scriptedAuth{}).interaction(), definition)
	if err == nil || err.Error() != "Invalid LiteLLM oidc setting: expected an object" {
		t.Errorf("err = %v", err)
	}
}

func TestAuthResponseBodyLimit(t *testing.T) {
	oauthTest(t)
	proxy := newRecorder(t, false)
	proxy.handle("/big", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = io.WriteString(w, `{"pad":"`+strings.Repeat("a", maxAuthBodyBytes)+`"}`)
	})
	response, err := doAuth(context.Background(), http.MethodGet, proxy.URL+"/big", http.Header{}, nil)
	if err != nil || !errors.Is(response.bodyErr, errAuthBodyTooLarge) || response.object() != nil {
		t.Errorf("response = %+v, %v", response, err)
	}
}

func TestAuthRedirects(t *testing.T) {
	oauthTest(t)
	proxy := newRecorder(t, false)
	other := newRecorder(t, false)
	other.handle("/target", jsonHandler(200, map[string]any{"ok": true}))
	proxy.handle("/target", jsonHandler(200, map[string]any{"ok": true}))
	proxy.handle("/same", func(w http.ResponseWriter, r *http.Request) { http.Redirect(w, r, "/target", http.StatusFound) })
	proxy.handle("/cross", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, other.URL+"/target", http.StatusFound)
	})
	proxy.handle("/loop", func(w http.ResponseWriter, r *http.Request) { http.Redirect(w, r, "/loop", http.StatusFound) })
	get := func(do func(context.Context, string, string, http.Header, []byte) (*authResponse, error), path string) int {
		response, err := do(context.Background(), http.MethodGet, proxy.URL+path, http.Header{}, nil)
		if err != nil {
			return -1
		}
		return response.status
	}
	for _, c := range []struct {
		name string
		do   func(context.Context, string, string, http.Header, []byte) (*authResponse, error)
		path string
		want int
	}{
		{"follows a same-origin redirect", doAuthFollowing, "/same", 200},
		{"stops at a cross-origin redirect", doAuthFollowing, "/cross", 302},
		{"stops after five hops", doAuthFollowing, "/loop", 302},
		{"does not follow when manual", doAuth, "/same", 302},
	} {
		if got := get(c.do, c.path); got != c.want {
			t.Errorf("%s: status %d, want %d", c.name, got, c.want)
		}
	}
	if len(other.log()) != 0 {
		t.Error("cross-origin redirect target was contacted")
	}
}

func TestPkceRefreshRejectsEmptyBaseURL(t *testing.T) {
	oauthTest(t)
	pinNow(testNow)
	proxy := newRecorder(t, false)
	credential := pkceCredential(proxy, func(c *ai.Credential) { setExtra(c, "baseUrl", "") })
	credential.SetExpiresMillis(float64(testNow))
	if _, err := refreshOf(t, credential); err == nil || err.Error() != "Invalid LiteLLM PKCE credential; run /login litellm again" {
		t.Errorf("err = %v", err)
	}
	if len(proxy.log()) != 0 {
		t.Error("a request was sent")
	}
}

func TestSameOriginURLTrimsBeforeParsing(t *testing.T) {
	issuer, _ := url.Parse("https://proxy.example.com")
	got, err := sameOriginURL("  https://proxy.example.com/token \n", issuer, "token endpoint")
	if err != nil || got != "https://proxy.example.com/token" {
		t.Errorf("got %q, %v", got, err)
	}
}

func TestHeaderTransportKeepsRequestHeaders(t *testing.T) {
	oauthTest(t)
	proxy := newRecorder(t, false)
	client := &http.Client{Transport: headerTransport{headers: map[string]string{"x-tenant": "tenant-a", "x-other": "other-a"}}}
	request, _ := http.NewRequest(http.MethodGet, proxy.URL+"/probe", nil)
	request.Header.Set("X-Tenant", "set-by-request")
	response, err := client.Do(request)
	if err != nil {
		t.Fatal(err)
	}
	response.Body.Close()
	header := proxy.log()[0].header
	if header.Get("x-tenant") != "set-by-request" || header.Get("x-other") != "other-a" {
		t.Errorf("header = %v", header)
	}
	if request.Header.Get("x-other") != "" {
		t.Error("the caller's request was mutated")
	}
}

func TestLoginErrorLabelsAndRedaction(t *testing.T) {
	for _, flow := range []string{"LiteLLM PKCE", "OIDC", "LiteLLM CLI SSO"} {
		err := loginError(context.Background(), litellmauth.LoginTimeoutError{}, flow, "https://proxy.example.com")
		if err.Error() != flow+" login timed out" || !errors.Is(err, context.DeadlineExceeded) {
			t.Errorf("%s: err = %v", flow, err)
		}
	}

	t.Run("names the proxy instead of its URL when CLI SSO start fails at the transport", func(t *testing.T) {
		oauthTest(t)
		proxy := newRecorder(t, false)
		proxy.handle("/sso/cli/start", dropConnection)
		script := &scriptedAuth{answer: answers(proxy.URL, "", "")}
		_, err := loginCliSSO(context.Background(), script.interaction(), proxy.URL, mustAuthClient(t, proxy.URL))
		if err == nil || strings.Contains(err.Error(), proxy.URL) || !strings.Contains(err.Error(), "the proxy") {
			t.Errorf("err = %v", err)
		}
	})
}

func mustAuthClient(t *testing.T, baseURL string) *litellmauth.Client {
	t.Helper()
	client, err := newAuthClient(baseURL, defaultDefinition(t), nil)
	if err != nil {
		t.Fatal(err)
	}
	return client
}
