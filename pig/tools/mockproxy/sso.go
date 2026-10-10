package main

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"sync"
	"time"
)

var (
	ssoMode         = flag.String("sso", "off", "SSO login mode: off, cli, pkce or paste")
	ssoPendingPolls = flag.Int("sso-pending-polls", 1, "CLI SSO polls answered pending before ready")
	ssoTeams        = flag.Bool("sso-teams", false, "CLI SSO asks for team selection")
	ssoDeny         = flag.Bool("sso-deny", false, "PKCE authorize answers access_denied")
)

// issued holds every token the mock handed out; auth accepts them like the configured key.
var (
	issuedMu sync.Mutex
	issued   = map[string]bool{}
)

func randomHex(n int) string {
	b := make([]byte, n)
	rand.Read(b)
	return hex.EncodeToString(b)
}

func issueToken() string {
	t := "sk-mock-" + randomHex(16)
	issuedMu.Lock()
	issued[t] = true
	issuedMu.Unlock()
	return t
}

func isIssued(t string) bool {
	issuedMu.Lock()
	defer issuedMu.Unlock()
	return issued[t]
}

// ssoLog records an SSO event; callers pass no secrets.
func ssoLog(format string, args ...any) {
	if *dump {
		fmt.Fprintf(os.Stderr, "[SSO] "+format+"\n", args...)
	}
}

func baseURL(r *http.Request) string {
	scheme := "http"
	if r.TLS != nil {
		scheme = "https"
	}
	return scheme + "://" + r.Host
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func handleSSOPage(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	fmt.Fprint(w, "<!doctype html><html><body><h1>Mock LiteLLM SSO</h1><p>You are signed in. Return to the terminal.</p></body></html>")
}

type cliSession struct {
	secret string
	polls  int
}

type pkceCode struct {
	clientID, redirectURI, resource, challenge string
}

type ssoState struct {
	pending int
	teams   bool
	deny    bool

	mu       sync.Mutex
	sessions map[string]*cliSession
	clients  map[string][]string  // client_id -> redirect URIs
	codes    map[string]*pkceCode // authorization code -> binding
	refresh  map[string]string    // live refresh token -> client_id
}

// registerSSO adds the routes of one SSO mode; "off" (or anything unknown) adds none, so they answer 404.
func registerSSO(mux *http.ServeMux, mode string, pending int, teams, deny bool) {
	s := &ssoState{
		pending: pending, teams: teams, deny: deny,
		sessions: map[string]*cliSession{},
		clients:  map[string][]string{},
		codes:    map[string]*pkceCode{},
		refresh:  map[string]string{},
	}
	switch mode {
	case "cli":
		mux.HandleFunc("POST /sso/cli/start", s.cliStart)
		mux.HandleFunc("GET /sso/cli/poll/{login_id}", s.cliPoll)
		mux.HandleFunc("GET /sso/key/generate", handleSSOPage)
	case "pkce":
		mux.HandleFunc("GET /.well-known/litellm-cli-auth", s.discovery)
		mux.HandleFunc("POST /oauth/register", s.register)
		mux.HandleFunc("GET /oauth/authorize", s.authorize)
		mux.HandleFunc("POST /oauth/token", s.token)
		mux.HandleFunc("POST /oauth/revoke", func(w http.ResponseWriter, r *http.Request) {
			ssoLog("revoke")
			w.WriteHeader(http.StatusOK)
		})
	case "paste":
		mux.HandleFunc("POST /key/generate", s.generateKey)
		mux.HandleFunc("GET /sso/key/generate", handleSSOPage)
	}
}

func (s *ssoState) cliStart(w http.ResponseWriter, r *http.Request) {
	id, secret := "login-"+randomHex(8), randomHex(16)
	s.mu.Lock()
	s.sessions[id] = &cliSession{secret: secret}
	s.mu.Unlock()
	ssoLog("cli start")
	writeJSON(w, http.StatusOK, map[string]any{
		"login_id":                  id,
		"poll_secret":               secret,
		"user_code":                 "MOCK-" + randomHex(2),
		"expires_in":                600,
		"verification_uri_complete": baseURL(r) + "/sso/key/generate?source=litellm-cli&key=" + url.QueryEscape(id),
	})
}

func (s *ssoState) cliPoll(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	sess := s.sessions[r.PathValue("login_id")]
	if sess == nil {
		s.mu.Unlock()
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "unknown login"})
		return
	}
	if r.Header.Get("x-litellm-cli-poll-secret") != sess.secret {
		s.mu.Unlock()
		writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "invalid poll secret"})
		return
	}
	sess.polls++
	waiting := sess.polls <= s.pending
	s.mu.Unlock()
	if waiting {
		ssoLog("cli poll pending")
		writeJSON(w, http.StatusOK, map[string]string{"status": "pending"})
		return
	}
	team := r.URL.Query().Get("team_id")
	if s.teams && team == "" {
		ssoLog("cli poll ready, team selection required")
		writeJSON(w, http.StatusOK, map[string]any{
			"status":                  "ready",
			"requires_team_selection": true,
			"teams":                   []string{"team-a", "team-b"},
			"team_details": []map[string]string{
				{"team_id": "team-a", "team_alias": "Team A"},
				{"team_id": "team-b", "team_alias": "Team B"},
			},
		})
		return
	}
	if team == "" {
		team = "team-a"
	}
	ssoLog("cli poll ready")
	writeJSON(w, http.StatusOK, map[string]any{
		"status": "ready", "key": issueToken(), "expires_in": 3600, "user_id": "mock-user", "team_id": team,
	})
}

func (s *ssoState) discovery(w http.ResponseWriter, r *http.Request) {
	b := baseURL(r)
	writeJSON(w, http.StatusOK, map[string]any{
		"contract_version":                 1,
		"issuer":                           b,
		"authorization_endpoint":           b + "/oauth/authorize",
		"token_endpoint":                   b + "/oauth/token",
		"registration_endpoint":            b + "/oauth/register",
		"revocation_endpoint":              b + "/oauth/revoke",
		"resource":                         b,
		"code_challenge_methods_supported": []string{"S256"},
	})
}

func (s *ssoState) register(w http.ResponseWriter, r *http.Request) {
	var req struct {
		RedirectURIs []string `json:"redirect_uris"`
	}
	if json.NewDecoder(r.Body).Decode(&req) != nil || len(req.RedirectURIs) == 0 {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid_client_metadata"})
		return
	}
	id := "mock-client-" + randomHex(6)
	s.mu.Lock()
	s.clients[id] = req.RedirectURIs
	s.mu.Unlock()
	ssoLog("register client")
	writeJSON(w, http.StatusCreated, map[string]any{
		"client_id": id, "redirect_uris": req.RedirectURIs, "token_endpoint_auth_method": "none",
	})
}

func (s *ssoState) authorize(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()
	redirect := q.Get("redirect_uri")
	s.mu.Lock()
	registered := false
	for _, u := range s.clients[q.Get("client_id")] {
		registered = registered || u == redirect
	}
	s.mu.Unlock()
	if !registered || q.Get("response_type") != "code" || q.Get("code_challenge") == "" ||
		q.Get("code_challenge_method") != "S256" || q.Get("resource") == "" || q.Get("state") == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid_request"})
		return
	}
	target, err := url.Parse(redirect)
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid_request"})
		return
	}
	out := target.Query()
	out.Set("state", q.Get("state"))
	if s.deny {
		ssoLog("authorize denied")
		out.Set("error", "access_denied")
	} else {
		code := "code-" + randomHex(16)
		s.mu.Lock()
		s.codes[code] = &pkceCode{q.Get("client_id"), redirect, q.Get("resource"), q.Get("code_challenge")}
		s.mu.Unlock()
		ssoLog("authorize granted")
		out.Set("code", code)
	}
	target.RawQuery = out.Encode()
	http.Redirect(w, r, target.String(), http.StatusFound)
}

func (s *ssoState) token(w http.ResponseWriter, r *http.Request) {
	invalid := func() { writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid_grant"}) }
	if r.ParseForm() != nil {
		invalid()
		return
	}
	grant, clientID := r.PostForm.Get("grant_type"), r.PostForm.Get("client_id")
	s.mu.Lock()
	switch grant {
	case "authorization_code":
		ssoLog("token grant authorization_code")
		c := s.codes[r.PostForm.Get("code")]
		delete(s.codes, r.PostForm.Get("code")) // single use
		sum := sha256.Sum256([]byte(r.PostForm.Get("code_verifier")))
		if c == nil || c.clientID != clientID || c.redirectURI != r.PostForm.Get("redirect_uri") ||
			c.resource != r.PostForm.Get("resource") || c.challenge != base64.RawURLEncoding.EncodeToString(sum[:]) {
			s.mu.Unlock()
			invalid()
			return
		}
	case "refresh_token":
		ssoLog("token grant refresh_token (rotation)")
		old := r.PostForm.Get("refresh_token")
		if owner, ok := s.refresh[old]; !ok || owner != clientID {
			s.mu.Unlock()
			invalid()
			return
		}
		delete(s.refresh, old)
	default:
		s.mu.Unlock()
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "unsupported_grant_type"})
		return
	}
	refresh := issueToken()
	s.refresh[refresh] = clientID
	s.mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]any{
		"access_token": issueToken(), "refresh_token": refresh, "token_type": "Bearer",
		"expires_in": 3600, "user_id": "mock-user", "team_id": "team-a",
	})
}

func (s *ssoState) generateKey(w http.ResponseWriter, r *http.Request) {
	h := r.Header.Get("Authorization")
	if len(h) <= len("Bearer ") || h[:7] != "Bearer " {
		writeJSON(w, http.StatusUnauthorized, map[string]any{"error": map[string]string{"message": "missing token"}})
		return
	}
	ssoLog("virtual key generated")
	writeJSON(w, http.StatusOK, map[string]string{
		"key": issueToken(), "expires": time.Now().Add(time.Hour).UTC().Format(time.RFC3339),
	})
}
