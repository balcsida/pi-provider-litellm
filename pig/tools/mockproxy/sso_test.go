package main

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

func ssoServer(t *testing.T, mode string, pending int, teams, deny bool) string {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("GET /model/info", makeHandleModelInfo("admin"))
	registerSSO(mux, mode, pending, teams, deny)
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv.URL
}

func do(t *testing.T, req *http.Request) (*http.Response, map[string]any) {
	t.Helper()
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	var out map[string]any
	json.NewDecoder(resp.Body).Decode(&out)
	return resp, out
}

func call(t *testing.T, method, u, bearer string, form url.Values, hdr map[string]string) (*http.Response, map[string]any) {
	t.Helper()
	var body *strings.Reader
	ct := ""
	switch {
	case form != nil:
		body, ct = strings.NewReader(form.Encode()), "application/x-www-form-urlencoded"
	default:
		body, ct = strings.NewReader("{}"), "application/json"
	}
	req, _ := http.NewRequest(method, u, body)
	req.Header.Set("Content-Type", ct)
	if bearer != "" {
		req.Header.Set("Authorization", "Bearer "+bearer)
	}
	for k, v := range hdr {
		req.Header.Set(k, v)
	}
	return do(t, req)
}

func wantStatus(t *testing.T, resp *http.Response, want int) {
	t.Helper()
	if resp.StatusCode != want {
		t.Fatalf("%s %s: want %d, got %d", resp.Request.Method, resp.Request.URL.Path, want, resp.StatusCode)
	}
}

func TestSSOOffMode(t *testing.T) {
	base := ssoServer(t, "off", 1, false, false)
	for _, c := range [][2]string{{"GET", "/.well-known/litellm-cli-auth"}, {"POST", "/sso/cli/start"}, {"POST", "/key/generate"}} {
		resp, _ := call(t, c[0], base+c[1], "admin", nil, nil)
		wantStatus(t, resp, 404)
	}
}

func cliLogin(t *testing.T, base, query string) (map[string]any, map[string]any) {
	t.Helper()
	resp, start := call(t, "POST", base+"/sso/cli/start", "", nil, nil)
	wantStatus(t, resp, 200)
	id, secret := start["login_id"].(string), start["poll_secret"].(string)
	if !strings.HasPrefix(start["verification_uri_complete"].(string), base+"/sso/key/generate?source=litellm-cli&key=") {
		t.Fatalf("bad verification uri %v", start["verification_uri_complete"])
	}
	poll := func(secret, q string) (*http.Response, map[string]any) {
		return call(t, "GET", base+"/sso/cli/poll/"+id+q, "", nil, map[string]string{"x-litellm-cli-poll-secret": secret})
	}
	resp, _ = poll("wrong", "")
	wantStatus(t, resp, 401)
	resp, _ = call(t, "GET", base+"/sso/cli/poll/nope", "", nil, nil)
	wantStatus(t, resp, 400)
	resp, got := poll(secret, "")
	wantStatus(t, resp, 200)
	if got["status"] != "pending" {
		t.Fatalf("want pending, got %v", got)
	}
	_, ready := poll(secret, "")
	_, second := poll(secret, query)
	return ready, second
}

func TestSSOCliMode(t *testing.T) {
	base := ssoServer(t, "cli", 1, false, false)
	ready, _ := cliLogin(t, base, "")
	key, _ := ready["key"].(string)
	if ready["status"] != "ready" || !strings.HasPrefix(key, "sk-mock-") || ready["team_id"] != "team-a" {
		t.Fatalf("bad ready %v", ready)
	}
	resp, _ := call(t, "GET", base+"/model/info", key, nil, nil)
	wantStatus(t, resp, 200)
	resp, _ = call(t, "GET", base+"/model/info", "sk-mock-bogus", nil, nil)
	wantStatus(t, resp, 401)
	resp, _ = call(t, "GET", base+"/sso/key/generate", "", nil, nil)
	wantStatus(t, resp, 200)
	resp, _ = call(t, "GET", base+"/.well-known/litellm-cli-auth", "", nil, nil)
	wantStatus(t, resp, 404)
}

func TestSSOCliTeams(t *testing.T) {
	base := ssoServer(t, "cli", 1, true, false)
	ready, again := cliLogin(t, base, "?team_id=team-b")
	if ready["requires_team_selection"] != true || len(ready["teams"].([]any)) != 2 || len(ready["team_details"].([]any)) != 2 {
		t.Fatalf("bad selection %v", ready)
	}
	key, _ := again["key"].(string)
	if again["team_id"] != "team-b" || !strings.HasPrefix(key, "sk-mock-") {
		t.Fatalf("bad team ready %v", again)
	}
	resp, _ := call(t, "GET", base+"/model/info", key, nil, nil)
	wantStatus(t, resp, 200)
}

// pkceAuthorize registers a client and runs /oauth/authorize, returning the redirect query.
func pkceAuthorize(t *testing.T, base, challenge string) (clientID, redirect string, q url.Values) {
	t.Helper()
	redirect = "http://127.0.0.1:9999/callback"
	resp, _ := call(t, "GET", base+"/.well-known/litellm-cli-auth", "", nil, nil)
	wantStatus(t, resp, 200)
	body := strings.NewReader(`{"redirect_uris":["` + redirect + `"]}`)
	req, _ := http.NewRequest("POST", base+"/oauth/register", body)
	req.Header.Set("Content-Type", "application/json")
	resp, reg := do(t, req)
	wantStatus(t, resp, 201)
	clientID = reg["client_id"].(string)

	authURL := base + "/oauth/authorize?" + url.Values{
		"client_id": {clientID}, "redirect_uri": {redirect}, "response_type": {"code"}, "resource": {base},
		"code_challenge": {challenge}, "code_challenge_method": {"S256"}, "state": {"st"},
	}.Encode()
	noFollow := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
	r, err := noFollow.Get(authURL)
	if err != nil {
		t.Fatal(err)
	}
	r.Body.Close()
	wantStatus(t, r, 302)
	loc, _ := url.Parse(r.Header.Get("Location"))
	if !strings.HasPrefix(loc.String(), redirect+"?") || loc.Query().Get("state") != "st" {
		t.Fatalf("bad redirect %s", loc)
	}
	return clientID, redirect, loc.Query()
}

func TestSSOPkceDiscoveryAndWrongVerifier(t *testing.T) {
	base := ssoServer(t, "pkce", 1, false, false)
	verifier := "verifier-verifier-verifier-verifier-1234"
	sum := sha256.Sum256([]byte(verifier))
	clientID, redirect, q := pkceAuthorize(t, base, base64.RawURLEncoding.EncodeToString(sum[:]))

	_, disc := call(t, "GET", base+"/.well-known/litellm-cli-auth", "", nil, nil)
	if disc["issuer"] != base || disc["token_endpoint"] != base+"/oauth/token" || disc["resource"] != base {
		t.Fatalf("bad discovery %v", disc)
	}
	resp, _ := call(t, "POST", base+"/sso/cli/start", "", nil, nil)
	wantStatus(t, resp, 404)

	token := func(v url.Values) (*http.Response, map[string]any) {
		return call(t, "POST", base+"/oauth/token", "", v, nil)
	}
	grant := func(verifier string) url.Values {
		return url.Values{"grant_type": {"authorization_code"}, "code": {q.Get("code")}, "redirect_uri": {redirect},
			"client_id": {clientID}, "code_verifier": {verifier}, "resource": {base}}
	}
	resp, bad := token(grant("wrong-verifier"))
	wantStatus(t, resp, 400)
	if bad["error"] != "invalid_grant" {
		t.Fatalf("want invalid_grant, got %v", bad)
	}
}

func TestSSOPkceTokenAndRefresh(t *testing.T) {
	base := ssoServer(t, "pkce", 1, false, false)
	verifier := "verifier-verifier-verifier-verifier-1234"
	sum := sha256.Sum256([]byte(verifier))
	clientID, redirect, q := pkceAuthorize(t, base, base64.RawURLEncoding.EncodeToString(sum[:]))
	token := func(v url.Values) (*http.Response, map[string]any) {
		return call(t, "POST", base+"/oauth/token", "", v, nil)
	}
	resp, tok := token(url.Values{"grant_type": {"authorization_code"}, "code": {q.Get("code")}, "redirect_uri": {redirect},
		"client_id": {clientID}, "code_verifier": {verifier}, "resource": {base}})
	wantStatus(t, resp, 200)
	access, refresh := tok["access_token"].(string), tok["refresh_token"].(string)
	if !strings.HasPrefix(access, "sk-mock-") || tok["token_type"] != "Bearer" || tok["team_id"] != "team-a" {
		t.Fatalf("bad token %v", tok)
	}
	resp, _ = call(t, "GET", base+"/model/info", access, nil, nil)
	wantStatus(t, resp, 200)

	rf := func(rt, client string) (*http.Response, map[string]any) {
		return token(url.Values{"grant_type": {"refresh_token"}, "refresh_token": {rt}, "client_id": {client}})
	}
	resp, _ = rf(refresh, "other-client")
	wantStatus(t, resp, 400)
	resp, rotated := rf(refresh, clientID)
	wantStatus(t, resp, 200)
	if rotated["refresh_token"] == refresh || rotated["access_token"] == access {
		t.Fatal("tokens were not rotated")
	}
	resp, failed := rf(refresh, clientID)
	wantStatus(t, resp, 400)
	if failed["error"] != "invalid_grant" {
		t.Fatalf("old refresh token accepted: %v", failed)
	}
	resp, unsupported := token(url.Values{"grant_type": {"password"}})
	wantStatus(t, resp, 400)
	if unsupported["error"] != "unsupported_grant_type" {
		t.Fatalf("got %v", unsupported)
	}
	resp, _ = call(t, "POST", base+"/oauth/revoke", "", url.Values{"token": {refresh}}, nil)
	wantStatus(t, resp, 200)
}

func TestSSOPkceDeny(t *testing.T) {
	base := ssoServer(t, "pkce", 1, false, true)
	_, _, q := pkceAuthorize(t, base, "challenge")
	if q.Get("error") != "access_denied" || q.Get("state") != "st" || q.Get("code") != "" {
		t.Fatalf("bad denial %v", q)
	}
}

func TestSSOPasteMode(t *testing.T) {
	base := ssoServer(t, "paste", 1, false, false)
	resp, _ := call(t, "POST", base+"/key/generate", "", nil, nil)
	wantStatus(t, resp, 401)
	resp, gen := call(t, "POST", base+"/key/generate", "pasted-sso-token", nil, nil)
	wantStatus(t, resp, 200)
	key, _ := gen["key"].(string)
	if !strings.HasPrefix(key, "sk-mock-") || gen["expires"] == "" {
		t.Fatalf("bad key %v", gen)
	}
	resp, _ = call(t, "GET", base+"/model/info", key, nil, nil)
	wantStatus(t, resp, 200)
	resp, _ = call(t, "GET", base+"/sso/key/generate", "", nil, nil)
	wantStatus(t, resp, 200)
	resp, _ = call(t, "POST", base+"/sso/cli/start", "", nil, nil)
	wantStatus(t, resp, 404)
}
