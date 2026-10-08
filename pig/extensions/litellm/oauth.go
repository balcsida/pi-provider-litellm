package litellm

// Ports src/index.ts (promptBaseUrl through refreshLiteLLM): the API-key login, the native PKCE, direct OIDC,
// CLI SSO and pasted-token OAuth flows, and the OAuth refresh with its transient-failure backoff. The login
// protocols (CLI SSO, proxy PKCE, IdP-direct OIDC and their refreshes) come from github.com/balcsida/litellm-auth-go;
// this file is the glue that picks the flow, talks to the user and maps results to the auth.json shapes.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"slices"
	"strings"
	"sync"
	"syscall"
	"time"

	litellmauth "github.com/balcsida/litellm-auth-go"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/protocols"
)

const (
	pkceFlow                = "litellm_cli_pkce"
	oidcFlow                = "oidc_pkce" // persisted in auth.json; never rename
	oidcDiscoveryPath       = "/.well-known/openid-configuration"
	differentBaseURL        = "enter-a-different-url"
	pkceTransientBackoffMs  = 5_000
	expireTokenImmediately  = 0
	maxSafeInteger          = float64(1<<53 - 1)
	virtualKeyProgressStart = "Generating virtual key..."
	// oidcClientBaseURL is the base the OIDC client is built on: the IdP flow never contacts the proxy, but the
	// library insists on a valid one.
	oidcClientBaseURL = "https://litellm.invalid"
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

// The raw helpers below serve the requests the library does not own: OIDC discovery and virtual key generation.

const (
	maxAuthBodyBytes = 1 << 20
	maxAuthRedirects = 5
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

// authRequestHeaders copies headers and sets Accept to JSON, plus Content-Type when contentType is set.
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

func isSafeInteger(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0) && value == math.Trunc(value) && math.Abs(value) <= maxSafeInteger
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

// ---- token failures -------------------------------------------------------------------------------------

// tokenFailure is a refresh outcome that is not a success. Transient failures may be retried.
type tokenFailure struct {
	transient bool
	message   string
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

// ---- library clients ------------------------------------------------------------------------------------

// headerTransport adds the resolved custom headers to each request, never overriding one the request sets.
type headerTransport struct {
	base    http.RoundTripper
	headers map[string]string
}

func (t headerTransport) RoundTrip(request *http.Request) (*http.Response, error) {
	request = request.Clone(request.Context())
	for name, value := range t.headers {
		if _, set := request.Header[http.CanonicalHeaderKey(name)]; !set {
			request.Header.Set(name, value)
		}
	}
	base := t.base
	if base == nil {
		base = http.DefaultTransport
	}
	return base.RoundTrip(request)
}

// newAuthClient builds a library client over authTransport. headers are added to every request: the proxy
// flows pass the resolved custom headers, the IdP flow passes none.
func newAuthClient(baseURL string, definition providerDefinition, headers map[string]string) (*litellmauth.Client, error) {
	options := []litellmauth.Option{
		litellmauth.WithHTTPClient(&http.Client{Transport: headerTransport{base: authTransport, headers: headers}}),
		litellmauth.WithRequestTimeout(loginTimeout),
		litellmauth.WithMaxWait(callbackTimeout),
		litellmauth.WithPollInterval(cliSSOPollInterval),
	}
	if definition.AllowInsecureHTTP {
		options = append(options, litellmauth.WithAllowInsecureHTTP())
	}
	return litellmauth.New(baseURL, options...)
}

// loginError reports a cancelled ctx as its cause, a library timeout as "<flow> login timed out" and a typed
// library error as it is. Any other library error may carry the proxy URL, which this extension never prints,
// so its text names "the proxy" instead.
func loginError(ctx context.Context, err error, flow, baseURL string) error {
	if ctx.Err() != nil {
		return context.Cause(ctx)
	}
	var timeout litellmauth.LoginTimeoutError
	var timeoutPointer *litellmauth.LoginTimeoutError
	var httpErr *litellmauth.HTTPError
	switch {
	case errors.As(err, &timeout), errors.As(err, &timeoutPointer):
		return &loginTimeoutError{flow: flow, err: err}
	case errors.Is(err, litellmauth.ErrPKCEUnsupported), errors.Is(err, litellmauth.ErrUnsupportedProxy),
		errors.Is(err, litellmauth.ErrProtocol), errors.As(err, &httpErr):
		return err
	}
	return &redactedError{text: strings.ReplaceAll(err.Error(), baseURL, "the proxy"), err: err}
}

// loginTimeoutError keeps the library timeout (and its context.DeadlineExceeded chain) under the flow's own text.
type loginTimeoutError struct {
	flow string
	err  error
}

func (e *loginTimeoutError) Error() string { return e.flow + " login timed out" }
func (e *loginTimeoutError) Unwrap() error { return e.err }

// redactedError carries err's chain under text, which has the proxy URL removed.
type redactedError struct {
	text string
	err  error
}

func (e *redactedError) Error() string { return e.text }
func (e *redactedError) Unwrap() error { return e.err }

func millis(t time.Time) float64 { return float64(t.UnixMilli()) }

// ---- native PKCE ----------------------------------------------------------------------------------------

// loginPkce returns litellmauth.ErrPKCEUnsupported, before anything is shown, for a proxy without the contract.
func loginPkce(ctx context.Context, interaction ai.AuthInteraction, baseURL string, client *litellmauth.Client) (ai.Credential, error) {
	token, err := client.AuthenticatePKCE(ctx, litellmauth.PKCEOptions{
		OnSession: func(_ context.Context, session litellmauth.PKCESession) error {
			notifyAuth(interaction, ai.AuthURLEvent{URL: session.AuthorizeURL.String(), Instructions: "Open this URL in a browser to sign in with LiteLLM."})
			return nil
		},
	})
	if err != nil {
		return ai.Credential{}, loginError(ctx, err, "LiteLLM PKCE", baseURL)
	}
	credential := newOAuthCredential(token.Key, token.RefreshToken, millis(token.ExpiresAt), baseURL)
	setExtra(&credential, "flow", pkceFlow)
	setExtra(&credential, "clientId", token.ClientID)
	setExtra(&credential, "tokenEndpoint", token.TokenEndpoint)
	setExtra(&credential, "resource", token.Resource)
	if token.UserID != "" {
		setExtra(&credential, "userId", token.UserID)
	}
	if token.TeamID != "" {
		setExtra(&credential, "teamId", token.TeamID)
	}
	return credential, nil
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
		if !isString || !slices.Contains(strings.Split(text, " "), "openid") {
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

// oidcExpires refreshes ahead of expiry, but by at most half the remaining lifetime, so a short-lived id_token
// is still used rather than treated as already expired.
func oidcExpires(expiresAt time.Time) float64 {
	expires := millis(expiresAt)
	lead := math.Ceil(math.Min(tokenRefreshLeadMs, (expires-millis(now()))/2))
	return expires - lead
}

// loginOidc sends nothing to the proxy and, unlike the proxy flows, no custom header to the IdP. The library
// checks the id_token's nonce, issuer and audience; the proxy verifies its signature.
func loginOidc(ctx context.Context, interaction ai.AuthInteraction, baseURL string, config oidcConfig, definition providerDefinition) (ai.Credential, error) {
	discovery, err := discoverOidc(ctx, config.issuer)
	if err != nil {
		return ai.Credential{}, err
	}
	client, err := newAuthClient(baseURL, definition, nil)
	if err != nil {
		return ai.Credential{}, err
	}
	token, err := client.AuthenticateOIDC(ctx, litellmauth.OIDCOptions{
		Provider: litellmauth.OIDCProvider{
			Issuer: discovery.issuer, ClientID: config.clientID, Scope: config.scope,
			AuthorizeURL: discovery.authorizationEndpoint, TokenURL: discovery.tokenEndpoint,
		},
		OnSession: func(_ context.Context, session litellmauth.PKCESession) error {
			notifyAuth(interaction, ai.AuthURLEvent{URL: session.AuthorizeURL.String(), Instructions: "Open this URL in a browser to sign in with your identity provider."})
			return nil
		},
		RedirectPorts: config.redirectPorts,
	})
	if err != nil {
		return ai.Credential{}, loginError(ctx, err, "OIDC", baseURL)
	}
	credential := newOAuthCredential(token.Key, token.RefreshToken, oidcExpires(token.ExpiresAt), baseURL)
	setExtra(&credential, "flow", oidcFlow)
	setExtra(&credential, "issuer", token.Issuer)
	setExtra(&credential, "clientId", token.ClientID)
	setExtra(&credential, "tokenEndpoint", token.TokenEndpoint)
	setExtra(&credential, "subject", token.Subject)
	return credential, nil
}

// ---- CLI SSO and pasted token ---------------------------------------------------------------------------

// loginCliSSO returns litellmauth.ErrUnsupportedProxy, before anything is shown, for a proxy without CLI SSO.
// The library does not model the poll response's expires_in, so the lifetime comes from the key itself.
func loginCliSSO(ctx context.Context, interaction ai.AuthInteraction, baseURL string, client *litellmauth.Client) (ai.Credential, error) {
	token, err := client.Authenticate(ctx, litellmauth.AuthenticateOptions{
		OnSession: func(_ context.Context, session litellmauth.Session) error {
			expiresIn := session.ExpiresIn.Seconds()
			notifyAuth(interaction, ai.AuthDeviceCodeEvent{
				UserCode:         session.UserCode,
				VerificationURI:  session.VerificationURL.String(),
				ExpiresInSeconds: &expiresIn,
			})
			return nil
		},
		SelectTeam: func(ctx context.Context, teams []litellmauth.Team) (string, error) {
			options := make([]ai.AuthSelectOption, len(teams))
			for index, team := range teams {
				options[index] = ai.AuthSelectOption{ID: team.ID, Label: firstNonEmpty(team.Alias, team.ID)}
			}
			return interaction.Prompt(ctx, ai.AuthSelectPrompt{Message: "Select a LiteLLM team:", Options: options})
		},
	})
	if err != nil {
		return ai.Credential{}, loginError(ctx, err, "LiteLLM CLI SSO", baseURL)
	}
	expires := float64(tokenExpiresAt(token.Key, configuredCLIJWTExpiresAt()))
	return newOAuthCredential(token.Key, "", expires, baseURL), nil
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
			expiry := parsed.UnixMilli()
			expiresAt = &expiry
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
		return loginOidc(ctx, interaction, baseURL, *oidc, definition)
	}
	client, err := newAuthClient(baseURL, definition, resolveHeaders(definition))
	if err != nil {
		return ai.Credential{}, err
	}
	credential, err := loginPkce(ctx, interaction, baseURL, client)
	if !errors.Is(err, litellmauth.ErrPKCEUnsupported) {
		return credential, err
	}
	credential, err = loginCliSSO(ctx, interaction, baseURL, client)
	if !errors.Is(err, litellmauth.ErrUnsupportedProxy) {
		return credential, err
	}
	return loginWithPastedToken(ctx, interaction, baseURL, resolveHeaders(definition))
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
		return refreshOidc(ctx, credential, definition)
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

// refreshFailure classifies a library refresh error. A cancelled ctx is returned as its cause. A rejected
// refresh token or an invalid response can only be fixed by a new login; an unavailable proxy, a retryable or
// 5xx status and every transport failure are transient.
func refreshFailure(ctx context.Context, label string, err error) (*tokenFailure, error) {
	if ctx.Err() != nil {
		return nil, context.Cause(ctx)
	}
	prefix := label + " token exchange "
	var httpErr *litellmauth.HTTPError
	switch {
	case errors.Is(err, litellmauth.ErrRefreshRejected):
		return &tokenFailure{false, prefix + "rejected (invalid_grant)"}, nil
	case errors.Is(err, io.ErrUnexpectedEOF), errors.Is(err, io.EOF), errors.Is(err, context.DeadlineExceeded),
		errors.Is(err, syscall.ECONNRESET), errors.As(err, new(net.Error)):
		// a response body cut off, reset or timed out mid-read is a network failure, though the library wraps it as a protocol error
		return &tokenFailure{true, prefix + "failed (network error)"}, nil
	case errors.Is(err, litellmauth.ErrOriginMismatch), errors.Is(err, litellmauth.ErrProtocol):
		return &tokenFailure{false, prefix + "returned an invalid response"}, nil
	case errors.Is(err, litellmauth.ErrProxyUnavailable):
		return &tokenFailure{true, prefix + "failed (HTTP 503)"}, nil
	case errors.As(err, &httpErr):
		return &tokenFailure{httpErr.Retryable || httpErr.StatusCode >= 500, fmt.Sprintf("%sfailed (HTTP %d)", prefix, httpErr.StatusCode)}, nil
	}
	return &tokenFailure{true, prefix + "failed (network error)"}, nil
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
		issuer, err := url.Parse(root)
		if err != nil || issuer.Host == "" {
			return nil, nil, errors.New("LiteLLM CLI auth has invalid issuer")
		}
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
		client, err := newAuthClient(root, definition, resolveHeaders(definition))
		if err != nil {
			return nil, nil, err
		}
		token, err := client.RefreshPKCE(ctx, litellmauth.Credential{
			BaseURL: root, AuthMethod: litellmauth.AuthMethodPKCE, Key: credential.Access, RefreshToken: credential.Refresh,
			ClientID: clientID, TokenEndpoint: tokenEndpoint, Resource: resource,
		})
		if err != nil {
			failure, cancelled := refreshFailure(ctx, "LiteLLM", err)
			return nil, failure, cancelled
		}
		return func(refreshed *ai.Credential) {
			refreshed.Access, refreshed.Refresh = token.Key, token.RefreshToken
			refreshed.SetExpiresMillis(millis(token.ExpiresAt))
			for name, value := range map[string]string{"userId": token.UserID, "teamId": token.TeamID} {
				if value == "" {
					delete(refreshed.Extra, name)
				} else {
					setExtra(refreshed, name, value)
				}
			}
		}, nil, nil
	})
}

func refreshOidc(ctx context.Context, credential ai.Credential, definition providerDefinition) (ai.Credential, error) {
	if !isAuthToken(credential.Refresh) {
		return ai.Credential{}, errors.New("LiteLLM OIDC credential has no refresh token; run /login litellm again")
	}
	storedEndpoint, _ := extraString(credential, "tokenEndpoint")
	tokenEndpoint, endpointOK := httpsURL(storedEndpoint, true)
	issuer, hasIssuer := extraString(credential, "issuer")
	_, issuerOK := httpsURL(issuer, false)
	clientID, _ := extraString(credential, "clientId")
	subject, _ := extraString(credential, "subject")
	if !endpointOK || !hasIssuer || !issuerOK || !isAuthToken(clientID) || subject == "" || !isSafeInteger(credential.ExpiresMillis()) {
		return ai.Credential{}, errors.New("Invalid LiteLLM OIDC credential; run /login litellm again")
	}
	storedBase, _ := extraString(credential, "baseUrl")
	return refreshWithBackoff(credential, func() (func(*ai.Credential), *tokenFailure, error) {
		client, err := newAuthClient(oidcClientBaseURL, definition, nil)
		if err != nil {
			return nil, nil, err
		}
		token, err := client.RefreshOIDC(ctx,
			litellmauth.OIDCProvider{Issuer: issuer, ClientID: clientID, Scope: "openid", TokenURL: tokenEndpoint},
			litellmauth.Credential{
				BaseURL: storedBase, AuthMethod: litellmauth.AuthMethodOIDC, Key: credential.Access, RefreshToken: credential.Refresh,
				Issuer: issuer, Subject: subject,
			})
		if err != nil {
			failure, cancelled := refreshFailure(ctx, "OIDC", err)
			return nil, failure, cancelled
		}
		return func(refreshed *ai.Credential) {
			refreshed.Access, refreshed.Refresh = token.Key, token.RefreshToken
			refreshed.SetExpiresMillis(oidcExpires(token.ExpiresAt))
			setExtra(refreshed, "subject", token.Subject)
		}, nil, nil
	})
}
