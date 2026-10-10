package litellm

// Ports src/index.ts (promptBaseUrl through refreshLiteLLM): the API-key login, the native PKCE, direct OIDC,
// CLI SSO and pasted-token OAuth flows, and the OAuth refresh with its transient-failure backoff.

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/protocols"
)

const (
	cliSSOExpiresInSeconds  = 600
	cliAuthDiscoveryPath    = "/.well-known/litellm-cli-auth"
	pkceFlow                = "litellm_cli_pkce"
	oidcFlow                = "oidc_pkce" // persisted in auth.json; never rename
	oidcDiscoveryPath       = "/.well-known/openid-configuration"
	differentBaseURL        = "enter-a-different-url"
	pkceTransientBackoffMs  = 5_000
	expireTokenImmediately  = 0
	maxSafeInteger          = float64(1<<53 - 1)
	pkceClientName          = "pi-provider-litellm"
	loginCompletePage       = "<!doctype html><title>LiteLLM login complete</title><p>You can close this window.</p>"
	virtualKeyProgressStart = "Generating virtual key..."
)

// Timing knobs that tests shorten.
var (
	loginTimeout       = 10 * time.Second
	cliSSOPollInterval = 2 * time.Second
	callbackTimeout    = 10 * time.Minute
)

// authTransport is the transport of every login and refresh request; nil means http.DefaultTransport.
// Tests inject a transport that trusts their httptest servers.
var authTransport http.RoundTripper

var (
	authTokenPattern  = regexp.MustCompile(`^[\x21-\x7e]+$`)
	oauthCodePattern  = regexp.MustCompile(`^[a-z_]{1,64}$`)
	jwtPartPattern    = regexp.MustCompile(`^[\w-]+$`)
	bearerPrefix      = regexp.MustCompile(`(?i)^Bearer\s+`)
	errAuthNetwork    = errors.New("network error")
	transientBackoffs = struct {
		sync.Mutex
		until map[string]int64
	}{until: map[string]int64{}}
)

func init() {
	defaultLoginFlows = loginHooks{LoginAPIKey: loginAPIKey, LoginOAuth: loginOAuth, Refresh: refreshLiteLLM}
}

// ---- HTTP plumbing --------------------------------------------------------------------------------------

const (
	maxAuthBodyBytes = 1 << 20
	maxAuthRedirects = 5
	// maxCliSSOExpiresInSeconds bounds a server-supplied CLI SSO lifetime so the deadline arithmetic cannot overflow.
	maxCliSSOExpiresInSeconds = 3600
)

var errAuthBodyTooLarge = errors.New("response body too large")

// authClient never follows redirects unless follow is set; then only same-origin ones, at most maxAuthRedirects.
func authClient(follow bool) *http.Client {
	return &http.Client{
		Transport: authTransport,
		CheckRedirect: func(request *http.Request, via []*http.Request) error {
			if !follow || len(via) > maxAuthRedirects || urlOrigin(request.URL) != urlOrigin(via[0].URL) {
				return http.ErrUseLastResponse
			}
			return nil
		},
	}
}

type authResponse struct {
	status  int
	body    []byte
	bodyErr error
}

func (r *authResponse) ok() bool { return r.status >= 200 && r.status < 300 }

// object decodes the body as a JSON object; nil when it is anything else.
func (r *authResponse) object() map[string]any {
	if r.bodyErr != nil {
		return nil
	}
	var parsed any
	if json.Unmarshal(r.body, &parsed) != nil {
		return nil
	}
	object, _ := parsed.(map[string]any)
	return object
}

// doAuth sends one request without following redirects, bounded by loginTimeout. A cancelled ctx returns its
// cause; any other transport failure returns errAuthNetwork, whose text never carries the URL.
func doAuth(ctx context.Context, method, endpoint string, header http.Header, body []byte) (*authResponse, error) {
	return doAuthWith(ctx, false, method, endpoint, header, body)
}

// doAuthFollowing is doAuth for the proxy requests the TypeScript leaves on fetch's default redirect handling
// (/sso/cli/start, the CLI SSO poll, /key/generate): same-origin redirects are followed.
func doAuthFollowing(ctx context.Context, method, endpoint string, header http.Header, body []byte) (*authResponse, error) {
	return doAuthWith(ctx, true, method, endpoint, header, body)
}

func doAuthWith(ctx context.Context, follow bool, method, endpoint string, header http.Header, body []byte) (*authResponse, error) {
	if err := ctx.Err(); err != nil {
		return nil, context.Cause(ctx)
	}
	requestCtx, cancel := context.WithTimeout(ctx, loginTimeout)
	defer cancel()
	var reader *bytes.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}
	var request *http.Request
	var err error
	if reader != nil {
		request, err = http.NewRequestWithContext(requestCtx, method, endpoint, reader)
	} else {
		request, err = http.NewRequestWithContext(requestCtx, method, endpoint, nil)
	}
	if err != nil {
		return nil, errAuthNetwork
	}
	request.Header = header
	response, err := authClient(follow).Do(request)
	if err != nil {
		if ctx.Err() != nil {
			return nil, context.Cause(ctx)
		}
		return nil, errAuthNetwork
	}
	defer response.Body.Close()
	var buffer bytes.Buffer
	_, readErr := buffer.ReadFrom(io.LimitReader(response.Body, maxAuthBodyBytes+1))
	if readErr == nil && buffer.Len() > maxAuthBodyBytes {
		readErr = errAuthBodyTooLarge
	}
	if ctx.Err() != nil {
		return nil, context.Cause(ctx)
	}
	return &authResponse{status: response.StatusCode, body: buffer.Bytes(), bodyErr: readErr}, nil
}

// authRequestHeaders is authRequestHeaders: Accept is always JSON.
func authRequestHeaders(headers map[string]string, contentType string) http.Header {
	result := http.Header{}
	for name, value := range headers {
		result.Set(name, value)
	}
	result.Set("Accept", "application/json")
	if contentType != "" {
		result.Set("Content-Type", contentType)
	}
	return result
}

func rawHeaders(headers map[string]string) http.Header {
	result := http.Header{}
	for name, value := range headers {
		result.Set(name, value)
	}
	return result
}

// readAuthJSON decodes the whole body or fails with "<stage> returned invalid JSON".
func readAuthJSON(response *authResponse, stage string) (any, error) {
	var parsed any
	if response.bodyErr != nil || json.Unmarshal(response.body, &parsed) != nil {
		return nil, fmt.Errorf("%s returned invalid JSON", stage)
	}
	return parsed, nil
}

// ---- validation helpers ---------------------------------------------------------------------------------

func isAuthToken(value any) bool {
	text, ok := value.(string)
	return ok && authTokenPattern.MatchString(text)
}

// oauthErrorCode is the OAuth `error` code when it is a plain code; descriptions are never echoed.
func oauthErrorCode(value any) string {
	if text, ok := value.(string); ok && oauthCodePattern.MatchString(text) {
		return text
	}
	return ""
}

func constantTimeEqual(value any, expected string) bool {
	text, ok := value.(string)
	return ok && subtle.ConstantTimeCompare([]byte(text), []byte(expected)) == 1
}

func isSafeInteger(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value == math.Trunc(value) && math.Abs(value) <= maxSafeInteger
}

func randomToken() string {
	buffer := make([]byte, 32)
	_, _ = rand.Read(buffer)
	return base64.RawURLEncoding.EncodeToString(buffer)
}

func pkceChallenge(verifier string) string {
	digest := sha256.Sum256([]byte(verifier))
	return base64.RawURLEncoding.EncodeToString(digest[:])
}

// urlOrigin is scheme://host[:port] with the scheme's default port dropped, as URL.origin.
func urlOrigin(u *url.URL) string {
	host := strings.ToLower(u.Hostname())
	port := u.Port()
	if (u.Scheme == "http" && port == "80") || (u.Scheme == "https" && port == "443") {
		port = ""
	}
	if strings.Contains(host, ":") {
		host = "[" + host + "]"
	}
	if port != "" {
		host += ":" + port
	}
	return strings.ToLower(u.Scheme) + "://" + host
}

func canonicalIssuer(value string) (string, error) {
	u, err := url.Parse(value)
	if err != nil || u.Host == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" {
		return "", errors.New("LiteLLM CLI auth has invalid issuer")
	}
	return urlOrigin(u) + firstNonEmpty(strings.TrimRight(u.EscapedPath(), "/"), "/"), nil
}

func sameOriginURL(value any, issuer *url.URL, field string) (string, error) {
	text, ok := value.(string)
	if !ok || strings.TrimSpace(text) == "" {
		return "", fmt.Errorf("LiteLLM CLI auth discovery has invalid %s", field)
	}
	text = strings.TrimSpace(text)
	u, err := url.Parse(text)
	if err != nil {
		return "", fmt.Errorf("LiteLLM CLI auth discovery has invalid %s", field)
	}
	if u.Scheme != issuer.Scheme || u.Host == "" || urlOrigin(u) != urlOrigin(issuer) {
		return "", fmt.Errorf("LiteLLM CLI auth discovery has cross-origin %s", field)
	}
	if u.User != nil || u.Fragment != "" {
		return "", fmt.Errorf("LiteLLM CLI auth discovery has invalid %s", field)
	}
	return text, nil
}

// httpsURL is an https URL without credentials or fragment, returned verbatim.
func httpsURL(value any, allowQuery bool) (string, bool) {
	text, ok := value.(string)
	if !ok || strings.Contains(text, "#") || (!allowQuery && strings.Contains(text, "?")) {
		return "", false
	}
	u, err := url.Parse(text)
	if err != nil || u.Scheme != "https" || u.Host == "" || u.User != nil {
		return "", false
	}
	return text, true
}

func jwtClaims(token string) map[string]any {
	parts := strings.Split(token, ".")
	if len(parts) != 3 || parts[1] == "" || !jwtPartPattern.MatchString(parts[1]) {
		return nil
	}
	payload, err := base64.RawURLEncoding.DecodeString(parts[1])
	if err != nil {
		return nil
	}
	var claims any
	if json.Unmarshal(payload, &claims) != nil {
		return nil
	}
	object, _ := claims.(map[string]any)
	return object
}

// ---- credential fields ----------------------------------------------------------------------------------

func setExtra(credential *ai.Credential, name string, value any) {
	if credential.Extra == nil {
		credential.Extra = map[string]json.RawMessage{}
	}
	raw, _ := json.Marshal(value)
	credential.Extra[name] = raw
}

func extraString(credential ai.Credential, name string) (string, bool) {
	raw, ok := credential.Extra[name]
	if !ok {
		return "", false
	}
	var text string
	if json.Unmarshal(raw, &text) != nil {
		return "", false
	}
	return text, true
}

func cloneExtra(credential ai.Credential) ai.Credential {
	extra := make(map[string]json.RawMessage, len(credential.Extra))
	for name, raw := range credential.Extra {
		extra[name] = raw
	}
	credential.Extra = extra
	return credential
}

func newOAuthCredential(access, refresh string, expires float64, baseURL string) ai.Credential {
	credential := ai.Credential{Type: ai.CredentialOAuth, Access: access, Refresh: refresh}
	credential.SetExpiresMillis(expires)
	setExtra(&credential, "baseUrl", baseURL)
	return credential
}

func notifyAuth(interaction ai.AuthInteraction, event ai.AuthEvent) {
	if interaction.Notify != nil {
		interaction.Notify(event)
	}
}

// ---- base URL and API-key login -------------------------------------------------------------------------

// promptBaseURL offers whatever proxy URL is already known, then asks for one.
func promptBaseURL(ctx context.Context, interaction ai.AuthInteraction, definition providerDefinition) (string, error) {
	if known, source, ok := knownBaseURL(definition); ok {
		choice, err := interaction.Prompt(ctx, ai.AuthSelectPrompt{
			Message: "LiteLLM proxy URL:",
			// Pi's login selector renders labels only, so the source has to ride along in the label.
			Options: []ai.AuthSelectOption{
				{ID: known, Label: known + " (" + source + ")"},
				{ID: differentBaseURL, Label: "Enter a different URL…"},
			},
		})
		if err != nil {
			return "", err
		}
		if choice == known {
			return known, nil
		}
	}
	raw, err := interaction.Prompt(ctx, ai.AuthTextPrompt{
		Message:     "Enter LiteLLM proxy URL (no trailing /v1):",
		Placeholder: defaultLiteLLMBaseURL,
	})
	if err != nil {
		return "", err
	}
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", errors.New("Base URL is required")
	}
	return protocols.NormalizeBaseURL(raw, definition.AllowInsecureHTTP)
}

func loginAPIKey(ctx context.Context, interaction ai.AuthInteraction, definition providerDefinition) (ai.Credential, error) {
	baseURL, err := promptBaseURL(ctx, interaction, definition)
	if err != nil {
		return ai.Credential{}, err
	}
	key, err := interaction.Prompt(ctx, ai.AuthSecretPrompt{Message: "Enter API key:"})
	if err != nil {
		return ai.Credential{}, err
	}
	key = strings.TrimSpace(key)
	if key == "" {
		return ai.Credential{}, errors.New("Both base URL and API key are required")
	}
	return ai.Credential{Type: ai.CredentialAPIKey, Key: key, Env: map[string]string{envBaseURL: baseURL}}, nil
}

// ---- token endpoints ------------------------------------------------------------------------------------

// tokenFailure is a token-endpoint outcome that is not a success. Transient failures may be retried.
type tokenFailure struct {
	transient bool
	message   string
}

// postTokenForm POSTs a token form without following redirects; each flow validates its own success fields.
// A non-nil error is a cancellation; failure describes everything else.
func postTokenForm(ctx context.Context, endpoint string, form url.Values, headers map[string]string, label string) (map[string]any, *tokenFailure, error) {
	response, err := doAuth(ctx, http.MethodPost, endpoint, authRequestHeaders(headers, "application/x-www-form-urlencoded"), []byte(form.Encode()))
	if err != nil {
		if errors.Is(err, errAuthNetwork) {
			return nil, &tokenFailure{true, label + " token exchange failed (network error)"}, nil
		}
		return nil, nil, err
	}
	if response.bodyErr != nil && response.ok() {
		return nil, &tokenFailure{true, label + " token exchange failed (network error)"}, nil
	}
	data := response.object()
	if !response.ok() {
		message := fmt.Sprintf("%s token exchange failed (HTTP %d)", label, response.status)
		if code := oauthErrorCode(data["error"]); code != "" {
			message = fmt.Sprintf("%s token exchange rejected (%s)", label, code)
		}
		return nil, &tokenFailure{response.status == http.StatusTooManyRequests || response.status >= 500, message}, nil
	}
	if data == nil {
		data = map[string]any{}
	}
	return data, nil, nil
}

type pkceToken struct {
	access, refresh string
	expires         int64
	userID, teamID  string
}

func optionalString(data map[string]any, name string) string {
	text, _ := data[name].(string)
	return text
}

func requestPkceToken(ctx context.Context, endpoint string, form url.Values, headers map[string]string, existingRefresh string) (*pkceToken, *tokenFailure, error) {
	data, failure, err := postTokenForm(ctx, endpoint, form, headers, "LiteLLM")
	if err != nil || failure != nil {
		return nil, failure, err
	}
	invalid := &tokenFailure{false, "LiteLLM token exchange returned an invalid response"}
	expiresIn, hasLifetime := data["expires_in"].(float64)
	if !hasLifetime || math.IsNaN(expiresIn) || math.IsInf(expiresIn, 0) || expiresIn <= 0 {
		return nil, invalid, nil
	}
	expires := float64(now().UnixMilli()) + expiresIn*1000
	tokenType, _ := data["token_type"].(string)
	if !isAuthToken(data["access_token"]) || (!isAuthToken(data["refresh_token"]) && !isAuthToken(existingRefresh)) ||
		strings.ToLower(tokenType) != "bearer" || !isSafeInteger(expires) {
		return nil, invalid, nil
	}
	refresh := existingRefresh // a refresh may not rotate the refresh token
	if isAuthToken(data["refresh_token"]) {
		refresh = data["refresh_token"].(string)
	}
	return &pkceToken{
		access: data["access_token"].(string), refresh: refresh, expires: int64(expires),
		userID: optionalString(data, "user_id"), teamID: optionalString(data, "team_id"),
	}, nil, nil
}

// ---- loopback redirect ----------------------------------------------------------------------------------

type loopbackCallback struct {
	RedirectURI string
	State       string
	// Code returns the authorization code of the first callback carrying the expected state.
	Code func() (string, error)
}

type callbackResult struct {
	code string
	err  error
}

// withLoopbackCallback serves an RFC 8252 loopback redirect on 127.0.0.1 for the lifetime of run and shuts the
// listener down on every exit path. Each port is tried in order; without any, the OS assigns one.
func withLoopbackCallback[T any](ctx context.Context, label string, ports []int, run func(loopbackCallback) (T, error)) (T, error) {
	var zero T
	state := randomToken()
	results := make(chan callbackResult, 1)
	settle := func(result callbackResult) {
		select {
		case results <- result:
		default:
		}
	}
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-store")
		if r.URL.Path != "/callback" {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", "GET")
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		query := r.URL.Query()
		if !constantTimeEqual(firstQuery(query, "state"), state) {
			http.Error(w, "Invalid OAuth state", http.StatusBadRequest)
			return
		}
		if query.Get("error") != "" {
			http.Error(w, "OAuth login failed", http.StatusBadRequest)
			message := label + " login was denied"
			if code := oauthErrorCode(query.Get("error")); code != "" {
				message += " (" + code + ")"
			}
			settle(callbackResult{err: errors.New(message)})
			return
		}
		code := query.Get("code")
		if code == "" {
			http.Error(w, "Missing OAuth code", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write([]byte(loginCompletePage))
		settle(callbackResult{code: code})
	})

	candidates := ports
	if len(candidates) == 0 {
		candidates = []int{0}
	}
	var listener net.Listener
	var listenErr error
	for _, port := range candidates {
		if listener, listenErr = net.Listen("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port))); listenErr == nil {
			break
		}
	}
	if listenErr != nil {
		return zero, listenErr
	}
	server := &http.Server{Handler: handler, ReadHeaderTimeout: loginTimeout}
	served := make(chan struct{})
	go func() {
		defer close(served)
		_ = server.Serve(listener)
	}()
	defer func() {
		_ = server.Close()
		<-served
	}()

	code := func() (string, error) {
		timer := time.NewTimer(callbackTimeout)
		defer timer.Stop()
		select {
		case result := <-results:
			return result.code, result.err
		case <-timer.C:
			return "", fmt.Errorf("%s login timed out", label)
		case <-ctx.Done():
			return "", context.Cause(ctx)
		}
	}
	port := listener.Addr().(*net.TCPAddr).Port
	return run(loopbackCallback{RedirectURI: fmt.Sprintf("http://127.0.0.1:%d/callback", port), State: state, Code: code})
}

func firstQuery(query url.Values, name string) any {
	values, ok := query[name]
	if !ok || len(values) == 0 {
		return nil
	}
	return values[0]
}

// ---- native PKCE ----------------------------------------------------------------------------------------

type cliAuthDiscovery struct {
	issuer, authorizationEndpoint, tokenEndpoint, registrationEndpoint, resource string
}

func discoverPkce(ctx context.Context, baseURL string, headers map[string]string) (*cliAuthDiscovery, error) {
	baseIssuer, err := canonicalIssuer(baseURL)
	if err != nil {
		return nil, err
	}
	response, err := doAuth(ctx, http.MethodGet, baseURL+cliAuthDiscoveryPath, authRequestHeaders(headers, ""), nil)
	if err != nil {
		if errors.Is(err, errAuthNetwork) {
			return nil, errors.New("LiteLLM CLI auth discovery failed (network error)")
		}
		return nil, err
	}
	if response.status == http.StatusNotFound {
		return nil, nil
	}
	if !response.ok() {
		return nil, fmt.Errorf("LiteLLM CLI auth discovery failed (HTTP %d)", response.status)
	}
	parsed, err := readAuthJSON(response, "LiteLLM CLI auth discovery")
	if err != nil {
		return nil, err
	}
	data, _ := parsed.(map[string]any)
	if version, _ := data["contract_version"].(float64); data == nil || version != 1 {
		return nil, errors.New("LiteLLM CLI auth discovery has unsupported contract version")
	}
	if !includesString(data["code_challenge_methods_supported"], "S256") {
		return nil, errors.New("LiteLLM CLI auth discovery does not support S256")
	}
	issuerText, _ := data["issuer"].(string)
	if strings.TrimSpace(issuerText) == "" {
		return nil, errors.New("LiteLLM CLI auth discovery has invalid issuer")
	}
	issuer, err := url.Parse(issuerText)
	if err != nil {
		return nil, errors.New("LiteLLM CLI auth discovery has invalid issuer")
	}
	canonical, err := canonicalIssuer(issuerText)
	if err != nil {
		return nil, err
	}
	if canonical != baseIssuer {
		return nil, errors.New("LiteLLM CLI auth discovery issuer does not match the proxy URL")
	}
	discovery := &cliAuthDiscovery{issuer: issuerText}
	for _, field := range []struct {
		key, label string
		target     *string
	}{
		{"authorization_endpoint", "authorization endpoint", &discovery.authorizationEndpoint},
		{"token_endpoint", "token endpoint", &discovery.tokenEndpoint},
		{"registration_endpoint", "registration endpoint", &discovery.registrationEndpoint},
		{"resource", "resource", &discovery.resource},
	} {
		if *field.target, err = sameOriginURL(data[field.key], issuer, field.label); err != nil {
			return nil, err
		}
	}
	return discovery, nil
}

func includesString(value any, want string) bool {
	items, ok := value.([]any)
	if !ok {
		return false
	}
	for _, item := range items {
		if text, ok := item.(string); ok && text == want {
			return true
		}
	}
	return false
}

func loginPkce(ctx context.Context, interaction ai.AuthInteraction, baseURL string, discovery *cliAuthDiscovery, headers map[string]string) (ai.Credential, error) {
	verifier := randomToken()
	return withLoopbackCallback(ctx, "LiteLLM PKCE", nil, func(callback loopbackCallback) (ai.Credential, error) {
		body, _ := json.Marshal(map[string]any{
			"client_name":                pkceClientName,
			"redirect_uris":              []string{callback.RedirectURI},
			"token_endpoint_auth_method": "none",
			"grant_types":                []string{"authorization_code", "refresh_token"},
			"response_types":             []string{"code"},
		})
		registrationResponse, err := doAuth(ctx, http.MethodPost, discovery.registrationEndpoint, authRequestHeaders(headers, "application/json"), body)
		if err != nil {
			if errors.Is(err, errAuthNetwork) {
				return ai.Credential{}, errors.New("LiteLLM PKCE client registration failed (network error)")
			}
			return ai.Credential{}, err
		}
		if !registrationResponse.ok() {
			return ai.Credential{}, fmt.Errorf("LiteLLM PKCE client registration failed (HTTP %d)", registrationResponse.status)
		}
		parsed, err := readAuthJSON(registrationResponse, "LiteLLM CLI auth registration")
		if err != nil {
			return ai.Credential{}, err
		}
		registration, _ := parsed.(map[string]any)
		clientID, _ := registration["client_id"].(string)
		if registration == nil || !isAuthToken(clientID) || !includesString(registration["redirect_uris"], callback.RedirectURI) {
			return ai.Credential{}, errors.New("LiteLLM PKCE client registration returned an invalid response")
		}
		authorizationURL, err := authorizationURL(discovery.authorizationEndpoint, map[string]string{
			"client_id":             clientID,
			"redirect_uri":          callback.RedirectURI,
			"response_type":         "code",
			"resource":              discovery.resource,
			"code_challenge":        pkceChallenge(verifier),
			"code_challenge_method": "S256",
			"state":                 callback.State,
		})
		if err != nil {
			return ai.Credential{}, err
		}
		notifyAuth(interaction, ai.AuthURLEvent{URL: authorizationURL, Instructions: "Open this URL in a browser to sign in with LiteLLM."})
		code, err := callback.Code()
		if err != nil {
			return ai.Credential{}, err
		}
		token, failure, err := requestPkceToken(ctx, discovery.tokenEndpoint, url.Values{
			"grant_type":    {"authorization_code"},
			"code":          {code},
			"redirect_uri":  {callback.RedirectURI},
			"client_id":     {clientID},
			"code_verifier": {verifier},
			"resource":      {discovery.resource},
		}, headers, "")
		if err != nil {
			return ai.Credential{}, err
		}
		if failure != nil {
			return ai.Credential{}, errors.New(failure.message)
		}
		credential := newOAuthCredential(token.access, token.refresh, float64(token.expires), baseURL)
		setExtra(&credential, "flow", pkceFlow)
		setExtra(&credential, "clientId", clientID)
		setExtra(&credential, "tokenEndpoint", discovery.tokenEndpoint)
		setExtra(&credential, "resource", discovery.resource)
		if token.userID != "" {
			setExtra(&credential, "userId", token.userID)
		}
		if token.teamID != "" {
			setExtra(&credential, "teamId", token.teamID)
		}
		return credential, nil
	})
}

// authorizationURL sets params on the endpoint's own query, replacing any parameter it already carries.
func authorizationURL(endpoint string, params map[string]string) (string, error) {
	u, err := url.Parse(endpoint)
	if err != nil {
		return "", errors.New("authorization endpoint is invalid")
	}
	query := u.Query()
	for name, value := range params {
		query.Set(name, value)
	}
	u.RawQuery = query.Encode()
	return u.String(), nil
}

// ---- direct OIDC ----------------------------------------------------------------------------------------

type oidcConfig struct {
	issuer, clientID, scope string
	redirectPorts           []int
}

type oidcDiscovery struct{ issuer, authorizationEndpoint, tokenEndpoint string }

func parseOidcConfig(raw any) (oidcConfig, error) {
	invalid := func(field, expected string) error {
		return fmt.Errorf("Invalid LiteLLM %s setting: expected %s", field, expected)
	}
	object, ok := raw.(map[string]any)
	if !ok || object == nil {
		return oidcConfig{}, invalid("oidc", "an object")
	}
	issuer, ok := httpsURL(object["issuer"], false)
	if !ok {
		return oidcConfig{}, invalid("oidc.issuer", "an https URL without credentials, query, or fragment")
	}
	if !isAuthToken(object["clientId"]) {
		return oidcConfig{}, invalid("oidc.clientId", "a non-empty string without spaces")
	}
	scope := "openid"
	if value := object["scope"]; value != nil {
		text, isString := value.(string)
		if !isString || !contains(strings.Split(text, " "), "openid") {
			return oidcConfig{}, invalid("oidc.scope", `a space-separated scope list that includes "openid"`)
		}
		scope = text
	}
	var ports []int
	if value := object["redirectPorts"]; value != nil {
		items, isArray := value.([]any)
		if !isArray {
			return oidcConfig{}, invalid("oidc.redirectPorts", "an array of port numbers from 1 to 65535")
		}
		for _, item := range items {
			port, isNumber := item.(float64)
			if !isNumber || port != math.Trunc(port) || port < 1 || port > 65535 {
				return oidcConfig{}, invalid("oidc.redirectPorts", "an array of port numbers from 1 to 65535")
			}
			ports = append(ports, int(port))
		}
	}
	return oidcConfig{issuer: issuer, clientID: object["clientId"].(string), scope: scope, redirectPorts: ports}, nil
}

func contains(items []string, want string) bool {
	for _, item := range items {
		if item == want {
			return true
		}
	}
	return false
}

// discoverOidc never sends the proxy's headers: LITELLM_HEADERS and provider headers can hold gateway credentials.
func discoverOidc(ctx context.Context, issuer string) (*oidcDiscovery, error) {
	withoutSlash := func(value string) string { return strings.TrimRight(value, "/") }
	response, err := doAuth(ctx, http.MethodGet, withoutSlash(issuer)+oidcDiscoveryPath, authRequestHeaders(nil, ""), nil)
	if err != nil {
		if errors.Is(err, errAuthNetwork) {
			return nil, errors.New("OIDC discovery failed (network error)")
		}
		return nil, err
	}
	if !response.ok() {
		return nil, fmt.Errorf("OIDC discovery failed (HTTP %d)", response.status)
	}
	parsed, err := readAuthJSON(response, "OIDC discovery")
	if err != nil {
		return nil, err
	}
	data, _ := parsed.(map[string]any)
	advertised, _ := data["issuer"].(string)
	if data == nil || withoutSlash(advertised) != withoutSlash(issuer) {
		return nil, errors.New("OIDC discovery issuer does not match the configured issuer")
	}
	if methods, present := data["code_challenge_methods_supported"]; present && !includesString(methods, "S256") {
		return nil, errors.New("OIDC discovery does not support S256")
	}
	authorizationEndpoint, ok := httpsURL(data["authorization_endpoint"], true)
	if !ok {
		return nil, errors.New("OIDC discovery has invalid authorization_endpoint")
	}
	tokenEndpoint, ok := httpsURL(data["token_endpoint"], true)
	if !ok {
		return nil, errors.New("OIDC discovery has invalid token_endpoint")
	}
	return &oidcDiscovery{issuer: advertised, authorizationEndpoint: authorizationEndpoint, tokenEndpoint: tokenEndpoint}, nil
}

type oidcToken struct {
	access, refresh string
	expires         int64
	subject         string
}

// expectedIDToken is what requestOidcToken checks an id_token against. A nil nonce or subject is not checked.
type expectedIDToken struct {
	issuer, clientID string
	nonce, subject   *string
}

// requestOidcToken exchanges a grant for an id_token and checks its claims (OIDC Core §3.1.3.7). The signature
// is deliberately not verified here: the proxy verifies it against the IdP's JWKS before honouring the bearer,
// so these checks only stop the extension from storing a token meant for another client or login.
func requestOidcToken(ctx context.Context, endpoint string, form url.Values, expected expectedIDToken, existingRefresh string) (*oidcToken, *tokenFailure, error) {
	data, failure, err := postTokenForm(ctx, endpoint, form, nil, "OIDC")
	if err != nil || failure != nil {
		return nil, failure, err
	}
	idToken, _ := data["id_token"].(string)
	var claims map[string]any
	if isAuthToken(idToken) {
		claims = jwtClaims(idToken)
	}
	if claims == nil {
		return nil, &tokenFailure{false, "OIDC token response has no valid id_token"}, nil
	}
	invalid := func(claim string) (*oidcToken, *tokenFailure, error) {
		return nil, &tokenFailure{false, "OIDC id_token has invalid " + claim}, nil
	}
	if issuer, ok := claims["iss"].(string); !ok || issuer != expected.issuer {
		return invalid("iss")
	}
	audiences, isList := claims["aud"].([]any)
	if !isList {
		audiences = []any{claims["aud"]}
	}
	audienceMatch := false
	for _, audience := range audiences {
		if text, ok := audience.(string); ok && text == expected.clientID {
			audienceMatch = true
		}
	}
	if !audienceMatch {
		return invalid("aud")
	}
	// OIDC Core §3.1.3.7 (errata set 2): azp is optional, even with several audiences, but when present it must name us.
	if azp, present := claims["azp"]; present {
		if text, ok := azp.(string); !ok || text != expected.clientID {
			return invalid("azp")
		}
	}
	subject, _ := claims["sub"].(string)
	if subject == "" || (expected.subject != nil && subject != *expected.subject) {
		return invalid("sub")
	}
	current := float64(now().UnixMilli())
	exp, isNumber := claims["exp"].(float64)
	expiresAt := math.Floor(exp * 1000)
	if !isNumber || !isSafeInteger(expiresAt) || expiresAt <= current {
		return invalid("exp")
	}
	if expected.nonce != nil && !constantTimeEqual(claims["nonce"], *expected.nonce) {
		return invalid("nonce")
	}
	refresh := existingRefresh // the refresh token is optional, and an IdP that does not rotate it keeps the existing one valid
	if isAuthToken(data["refresh_token"]) {
		refresh = data["refresh_token"].(string)
	}
	// Refresh ahead of expiry, but by at most half the remaining lifetime, so a short-lived id_token is still
	// used rather than treated as already expired.
	lead := math.Ceil(math.Min(tokenRefreshLeadMs, (expiresAt-current)/2))
	return &oidcToken{access: idToken, refresh: refresh, expires: int64(expiresAt - lead), subject: subject}, nil, nil
}

func loginOidc(ctx context.Context, interaction ai.AuthInteraction, baseURL string, config oidcConfig) (ai.Credential, error) {
	discovery, err := discoverOidc(ctx, config.issuer)
	if err != nil {
		return ai.Credential{}, err
	}
	verifier, nonce := randomToken(), randomToken()
	return withLoopbackCallback(ctx, "OIDC", config.redirectPorts, func(callback loopbackCallback) (ai.Credential, error) {
		authorizationURL, err := authorizationURL(discovery.authorizationEndpoint, map[string]string{
			"response_type":         "code",
			"client_id":             config.clientID,
			"redirect_uri":          callback.RedirectURI,
			"scope":                 config.scope,
			"state":                 callback.State,
			"nonce":                 nonce,
			"code_challenge":        pkceChallenge(verifier),
			"code_challenge_method": "S256",
		})
		if err != nil {
			return ai.Credential{}, err
		}
		notifyAuth(interaction, ai.AuthURLEvent{URL: authorizationURL, Instructions: "Open this URL in a browser to sign in with your identity provider."})
		code, err := callback.Code()
		if err != nil {
			return ai.Credential{}, err
		}
		token, failure, err := requestOidcToken(ctx, discovery.tokenEndpoint, url.Values{
			"grant_type":    {"authorization_code"},
			"code":          {code},
			"redirect_uri":  {callback.RedirectURI},
			"client_id":     {config.clientID},
			"code_verifier": {verifier},
		}, expectedIDToken{issuer: discovery.issuer, clientID: config.clientID, nonce: &nonce}, "")
		if err != nil {
			return ai.Credential{}, err
		}
		if failure != nil {
			return ai.Credential{}, errors.New(failure.message)
		}
		credential := newOAuthCredential(token.access, token.refresh, float64(token.expires), baseURL)
		setExtra(&credential, "flow", oidcFlow)
		setExtra(&credential, "issuer", discovery.issuer)
		setExtra(&credential, "clientId", config.clientID)
		setExtra(&credential, "tokenEndpoint", discovery.tokenEndpoint)
		setExtra(&credential, "subject", token.subject)
		return credential, nil
	})
}

// ---- CLI SSO and pasted token ---------------------------------------------------------------------------

type cliSSOStart struct {
	loginID, pollSecret, userCode string
	expiresInSeconds              float64
}

func positiveNumber(value any) (float64, bool) {
	number, ok := value.(float64)
	return number, ok && !math.IsNaN(number) && !math.IsInf(number, 0) && number > 0
}

func startCliSSO(ctx context.Context, baseURL string, headers map[string]string) (*cliSSOStart, error) {
	response, err := doAuthFollowing(ctx, http.MethodPost, baseURL+"/sso/cli/start", rawHeaders(headers), []byte{})
	if err != nil {
		if errors.Is(err, errAuthNetwork) {
			return nil, errors.New("LiteLLM CLI SSO start failed (network error)")
		}
		return nil, err
	}
	if response.status == http.StatusNotFound || response.status == http.StatusMethodNotAllowed {
		return nil, nil
	}
	if !response.ok() {
		return nil, fmt.Errorf("LiteLLM CLI SSO start failed (HTTP %d)", response.status)
	}
	data := response.object()
	loginID, _ := data["login_id"].(string)
	pollSecret, _ := data["poll_secret"].(string)
	userCode, _ := data["user_code"].(string)
	if data == nil || loginID == "" || pollSecret == "" || userCode == "" {
		return nil, errors.New("LiteLLM CLI SSO start returned an invalid response")
	}
	expiresIn, ok := positiveNumber(data["expires_in"])
	if !ok {
		expiresIn = cliSSOExpiresInSeconds
	}
	expiresIn = min(expiresIn, maxCliSSOExpiresInSeconds)
	return &cliSSOStart{loginID: loginID, pollSecret: pollSecret, userCode: userCode, expiresInSeconds: expiresIn}, nil
}

func waitForNextCliSSOPoll(ctx context.Context) error {
	timer := time.NewTimer(cliSSOPollInterval)
	defer timer.Stop()
	select {
	case <-timer.C:
		return nil
	case <-ctx.Done():
		return context.Cause(ctx)
	}
}

type cliSSOResult struct {
	access           string
	expiresInSeconds float64 // 0 when the server named no lifetime
}

type cliSSOTeam struct{ id, label string }

func cliSSOTeams(data map[string]any) []cliSSOTeam {
	var teams []cliSSOTeam
	details, _ := data["team_details"].([]any)
	for _, item := range details {
		record, ok := item.(map[string]any)
		if !ok {
			continue
		}
		id, isString := record["team_id"].(string)
		if !isString {
			id, isString = record["id"].(string)
		}
		if !isString || id == "" {
			continue
		}
		label, _ := record["team_alias"].(string)
		teams = append(teams, cliSSOTeam{id, firstNonEmpty(label, id)})
	}
	if len(teams) == 0 {
		ids, _ := data["teams"].([]any)
		for _, item := range ids {
			if id, ok := item.(string); ok && id != "" {
				teams = append(teams, cliSSOTeam{id, id})
			}
		}
	}
	return teams
}

func pollCliSSO(ctx context.Context, baseURL string, start *cliSSOStart, interaction ai.AuthInteraction, headers map[string]string) (cliSSOResult, error) {
	deadline := now().Add(time.Duration(start.expiresInSeconds * float64(time.Second)))
	pollURL := baseURL + "/sso/cli/poll/" + url.PathEscape(start.loginID)
	var teamID string
	for now().Before(deadline) {
		endpoint := pollURL
		if teamID != "" {
			endpoint += "?" + url.Values{"team_id": {teamID}}.Encode()
		}
		header := rawHeaders(headers)
		header.Set("x-litellm-cli-poll-secret", start.pollSecret)
		response, err := doAuthFollowing(ctx, http.MethodGet, endpoint, header, nil)
		if err != nil {
			if !errors.Is(err, errAuthNetwork) {
				return cliSSOResult{}, err
			}
			if err := waitForNextCliSSOPoll(ctx); err != nil {
				return cliSSOResult{}, err
			}
			continue
		}
		if response.status == http.StatusTooManyRequests || response.status >= 500 {
			if err := waitForNextCliSSOPoll(ctx); err != nil {
				return cliSSOResult{}, err
			}
			continue
		}
		if response.status == http.StatusBadRequest {
			return cliSSOResult{}, errors.New("LiteLLM CLI SSO login expired or is invalid")
		}
		if !response.ok() {
			return cliSSOResult{}, fmt.Errorf("LiteLLM CLI SSO polling failed (HTTP %d)", response.status)
		}
		data := response.object()
		if data == nil {
			return cliSSOResult{}, errors.New("LiteLLM CLI SSO polling returned an invalid response")
		}
		status, _ := data["status"].(string)
		if key, _ := data["key"].(string); status == "ready" && key != "" {
			expiresIn, _ := positiveNumber(data["expires_in"])
			return cliSSOResult{access: key, expiresInSeconds: expiresIn}, nil
		}
		if selection, _ := data["requires_team_selection"].(bool); status == "ready" && selection && teamID == "" {
			teams := cliSSOTeams(data)
			if len(teams) == 0 {
				return cliSSOResult{}, errors.New("LiteLLM CLI SSO requested team selection without any teams")
			}
			options := make([]ai.AuthSelectOption, len(teams))
			for index, team := range teams {
				options[index] = ai.AuthSelectOption{ID: team.id, Label: team.label}
			}
			selected, err := interaction.Prompt(ctx, ai.AuthSelectPrompt{Message: "Select a LiteLLM team:", Options: options})
			if err != nil {
				return cliSSOResult{}, err
			}
			valid := false
			for _, team := range teams {
				valid = valid || team.id == selected
			}
			if !valid {
				return cliSSOResult{}, errors.New("Invalid LiteLLM team selection")
			}
			teamID = selected
			continue
		}
		if status != "pending" {
			return cliSSOResult{}, errors.New("LiteLLM CLI SSO polling returned an invalid response")
		}
		if err := waitForNextCliSSOPoll(ctx); err != nil {
			return cliSSOResult{}, err
		}
	}
	return cliSSOResult{}, errors.New("LiteLLM CLI SSO login expired")
}

// generateVirtualKey exchanges an SSO token for a virtual key. expiresAt is nil for a key that does not expire.
func generateVirtualKey(ctx context.Context, baseURL, userToken string, headers map[string]string) (key string, expiresAt *int64, err error) {
	header := rawHeaders(headers)
	header.Set("Authorization", "Bearer "+userToken)
	header.Set("Content-Type", "application/json")
	response, err := doAuthFollowing(ctx, http.MethodPost, baseURL+"/key/generate", header, []byte("{}"))
	if err != nil {
		return "", nil, err
	}
	if !response.ok() {
		return "", nil, fmt.Errorf("Virtual key generation failed (%d)", response.status)
	}
	data := response.object()
	key, _ = data["key"].(string)
	if key == "" {
		return "", nil, errors.New("No key in response from /key/generate")
	}
	if text, ok := data["expires"].(string); ok {
		if parsed, parseErr := time.Parse(time.RFC3339Nano, text); parseErr == nil {
			millis := parsed.UnixMilli()
			expiresAt = &millis
		}
	}
	return key, expiresAt, nil
}

func loginWithPastedToken(ctx context.Context, interaction ai.AuthInteraction, baseURL string, headers map[string]string) (ai.Credential, error) {
	notifyAuth(interaction, ai.AuthURLEvent{URL: baseURL + "/sso/key/generate", Instructions: "Authenticate via SSO, then copy your token from the LiteLLM UI."})
	pasted, err := interaction.Prompt(ctx, ai.AuthSecretPrompt{Message: "Paste your SSO token from the LiteLLM UI:"})
	if err != nil {
		return ai.Credential{}, err
	}
	rawToken := strings.TrimSpace(bearerPrefix.ReplaceAllString(strings.TrimSpace(pasted), ""))
	if rawToken == "" {
		return ai.Credential{}, errors.New("SSO token is required")
	}
	answer, err := interaction.Prompt(ctx, ai.AuthTextPrompt{Message: "Generate a LiteLLM virtual key from this token? (y/n):"})
	if err != nil {
		return ai.Credential{}, err
	}
	answer = strings.ToLower(strings.TrimSpace(answer))
	access := rawToken
	expires := float64(tokenExpiresAt(rawToken, permanentTokenExpiresAt))
	if answer != "n" && answer != "no" {
		notifyAuth(interaction, ai.AuthProgressEvent{Message: virtualKeyProgressStart})
		key, expiresAt, err := generateVirtualKey(ctx, baseURL, rawToken, headers)
		switch {
		case err == nil:
			access = key
			expires = float64(permanentTokenExpiresAt)
			if expiresAt != nil {
				expires = math.Max(float64(now().UnixMilli()), float64(*expiresAt-tokenRefreshLeadMs))
			}
			notifyAuth(interaction, ai.AuthProgressEvent{Message: "Virtual key generated and will be used for API calls."})
		case ctx.Err() != nil:
			return ai.Credential{}, context.Cause(ctx)
		default:
			message := err.Error()
			if errors.Is(err, errAuthNetwork) {
				message = "network error"
			}
			notifyAuth(interaction, ai.AuthProgressEvent{Message: fmt.Sprintf("LiteLLM: virtual key generation failed (%s); using SSO token directly.", message)})
		}
	}
	return newOAuthCredential(access, "", expires, baseURL), nil
}

// ---- flow chooser ---------------------------------------------------------------------------------------

func loginOAuth(ctx context.Context, interaction ai.AuthInteraction, definition providerDefinition) (ai.Credential, error) {
	var oidc *oidcConfig
	if definition.HasOIDC {
		config, err := parseOidcConfig(definition.OIDC)
		if err != nil {
			return ai.Credential{}, err
		}
		oidc = &config
	}
	baseURL, err := promptBaseURL(ctx, interaction, definition)
	if err != nil {
		return ai.Credential{}, err
	}
	// Direct OIDC sends nothing to the proxy. The IdP is never derived from it either: the proxy's own OAuth
	// metadata describes LiteLLM's authorization server, not the IdP that signs the JWTs it accepts.
	if oidc != nil {
		return loginOidc(ctx, interaction, baseURL, *oidc)
	}
	headers := resolveHeaders(definition)
	discovery, err := discoverPkce(ctx, baseURL, headers)
	if err != nil {
		return ai.Credential{}, err
	}
	if discovery != nil {
		return loginPkce(ctx, interaction, baseURL, discovery, headers)
	}
	cliSSO, err := startCliSSO(ctx, baseURL, headers)
	if err != nil {
		return ai.Credential{}, err
	}
	if cliSSO == nil {
		return loginWithPastedToken(ctx, interaction, baseURL, headers)
	}
	expiresIn := cliSSO.expiresInSeconds
	notifyAuth(interaction, ai.AuthDeviceCodeEvent{
		UserCode:         cliSSO.userCode,
		VerificationURI:  baseURL + "/sso/key/generate?source=litellm-cli&key=" + url.QueryEscape(cliSSO.loginID),
		ExpiresInSeconds: &expiresIn,
	})
	result, err := pollCliSSO(ctx, baseURL, cliSSO, interaction, headers)
	if err != nil {
		return ai.Credential{}, err
	}
	expires := float64(tokenExpiresAt(result.access, configuredCLIJWTExpiresAt()))
	if result.expiresInSeconds > 0 {
		expires = float64(now().UnixMilli()) + result.expiresInSeconds*1000
	}
	return newOAuthCredential(result.access, "", expires, baseURL), nil
}

// ---- refresh --------------------------------------------------------------------------------------------

// refreshWithBackoff runs request, and after a transient failure of a still-valid credential keeps that
// credential and skips the next attempts for pkceTransientBackoffMs, keyed by refresh token so one credential's
// failures never delay another's refresh. request returns the change to apply, a failure, or a cancellation.
func refreshWithBackoff(credential ai.Credential, request func() (func(*ai.Credential), *tokenFailure, error)) (ai.Credential, error) {
	current := func() float64 { return float64(now().UnixMilli()) }
	stillValid := func() bool { return current() < credential.ExpiresMillis() }
	if stillValid() {
		transientBackoffs.Lock()
		until, ok := transientBackoffs.until[credential.Refresh]
		transientBackoffs.Unlock()
		if ok && current() < float64(until) {
			return credential, nil
		}
	}
	apply, failure, err := request()
	if err != nil {
		return ai.Credential{}, err
	}
	transientBackoffs.Lock()
	defer transientBackoffs.Unlock()
	if failure != nil && failure.transient && stillValid() {
		transientBackoffs.until[credential.Refresh] = now().UnixMilli() + pkceTransientBackoffMs
		return credential, nil
	}
	delete(transientBackoffs.until, credential.Refresh)
	// A transient failure leaves the stored credential intact, and a new login would need the same endpoint.
	if failure != nil {
		if failure.transient {
			return ai.Credential{}, errors.New(failure.message)
		}
		return ai.Credential{}, errors.New(failure.message + "; run /login litellm again")
	}
	refreshed := cloneExtra(credential)
	apply(&refreshed)
	return refreshed, nil
}

func refreshLiteLLM(ctx context.Context, credential ai.Credential, definition providerDefinition) (ai.Credential, error) {
	if err := ctx.Err(); err != nil {
		return ai.Credential{}, context.Cause(ctx)
	}
	flow, _ := extraString(credential, "flow")
	switch flow {
	case pkceFlow:
		return refreshPkce(ctx, credential, definition)
	// Must precede the `!command` fallthrough below: an IdP refresh token is never executed.
	case oidcFlow:
		return refreshOidc(ctx, credential)
	}
	if !strings.HasPrefix(credential.Refresh, "!") {
		if credential.ExpiresMillis() < float64(permanentTokenExpiresAt) {
			return ai.Credential{}, errors.New("LiteLLM credential cannot be refreshed; run /login litellm again")
		}
		return credential, nil
	}
	access, err := executeAPIKeyCommand(ctx, credential.Refresh)
	if err != nil {
		return ai.Credential{}, err
	}
	refreshed := cloneExtra(credential)
	refreshed.Access = access
	refreshed.SetExpiresMillis(float64(tokenExpiresAt(access, expireTokenImmediately)))
	return refreshed, nil
}

func refreshPkce(ctx context.Context, credential ai.Credential, definition providerDefinition) (ai.Credential, error) {
	baseURL, hasBaseURL := extraString(credential, "baseUrl")
	clientID, _ := extraString(credential, "clientId")
	if !hasBaseURL || baseURL == "" || !isAuthToken(clientID) || !isAuthToken(credential.Access) || !isAuthToken(credential.Refresh) || !isSafeInteger(credential.ExpiresMillis()) {
		return ai.Credential{}, errors.New("Invalid LiteLLM PKCE credential; run /login litellm again")
	}
	return refreshWithBackoff(credential, func() (func(*ai.Credential), *tokenFailure, error) {
		root, err := credentialRoot(definition, &credentialInfo{Type: "oauth", BaseURL: baseURL})
		if err != nil {
			return nil, nil, err
		}
		if _, err := canonicalIssuer(root); err != nil {
			return nil, nil, err
		}
		issuer, _ := url.Parse(root)
		storedEndpoint, _ := extraString(credential, "tokenEndpoint")
		storedResource, _ := extraString(credential, "resource")
		tokenEndpoint, err := sameOriginURL(storedEndpoint, issuer, "token endpoint")
		if err != nil {
			return nil, nil, err
		}
		resource, err := sameOriginURL(storedResource, issuer, "resource")
		if err != nil {
			return nil, nil, err
		}
		token, failure, err := requestPkceToken(ctx, tokenEndpoint, url.Values{
			"grant_type":    {"refresh_token"},
			"refresh_token": {credential.Refresh},
			"client_id":     {clientID},
			"resource":      {resource},
		}, resolveHeaders(definition), credential.Refresh)
		if err != nil || failure != nil {
			return nil, failure, err
		}
		return func(refreshed *ai.Credential) {
			refreshed.Access, refreshed.Refresh = token.access, token.refresh
			refreshed.SetExpiresMillis(float64(token.expires))
			for name, value := range map[string]string{"userId": token.userID, "teamId": token.teamID} {
				if value == "" {
					delete(refreshed.Extra, name)
				} else {
					setExtra(refreshed, name, value)
				}
			}
		}, nil, nil
	})
}

func refreshOidc(ctx context.Context, credential ai.Credential) (ai.Credential, error) {
	if !isAuthToken(credential.Refresh) {
		return ai.Credential{}, errors.New("LiteLLM OIDC credential has no refresh token; run /login litellm again")
	}
	storedEndpoint, _ := extraString(credential, "tokenEndpoint")
	tokenEndpoint, endpointOK := httpsURL(storedEndpoint, true)
	issuer, hasIssuer := extraString(credential, "issuer")
	clientID, _ := extraString(credential, "clientId")
	subject, _ := extraString(credential, "subject")
	if !endpointOK || !hasIssuer || !isAuthToken(clientID) || subject == "" || !isSafeInteger(credential.ExpiresMillis()) {
		return ai.Credential{}, errors.New("Invalid LiteLLM OIDC credential; run /login litellm again")
	}
	return refreshWithBackoff(credential, func() (func(*ai.Credential), *tokenFailure, error) {
		token, failure, err := requestOidcToken(ctx, tokenEndpoint, url.Values{
			"grant_type":    {"refresh_token"},
			"refresh_token": {credential.Refresh},
			"client_id":     {clientID},
		}, expectedIDToken{issuer: issuer, clientID: clientID, subject: &subject}, credential.Refresh)
		if err != nil || failure != nil {
			return nil, failure, err
		}
		return func(refreshed *ai.Credential) {
			refreshed.Access, refreshed.Refresh = token.access, token.refresh
			refreshed.SetExpiresMillis(float64(token.expires))
			setExtra(refreshed, "subject", token.subject)
		}, nil, nil
	})
}
