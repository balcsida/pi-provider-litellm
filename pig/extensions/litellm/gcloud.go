package litellm

// Ports src/gcloud-token.ts: Google ADC token minting. Only authorized_user credentials are supported;
// service accounts warn and fail closed.

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const (
	gcloudCacheTTL       = 50 * time.Minute
	gcloudTokenCacheKey  = "gcloud-adc"
	gcloudADCFilename    = "application_default_credentials.json"
	envGcloudTokenAuth   = "LITELLM_GCLOUD_TOKEN_AUTH"
	envGoogleCredentials = "GOOGLE_APPLICATION_CREDENTIALS"
	gcloudExchangeLimit  = 10 * time.Second
)

// gcloudTokenURL and gcloudHTTPClient are replaced by tests; production never changes them.
var (
	gcloudTokenURL   = "https://oauth2.googleapis.com/token"
	gcloudHTTPClient = http.DefaultClient
)

var gcloudCache struct {
	sync.Mutex
	token string
	key   string
	at    time.Time
}

type authorizedUserADC struct {
	ClientID     string
	ClientSecret string
	RefreshToken string
}

// isGcloudTokenAuthEnabled is true when LITELLM_GCLOUD_TOKEN_AUTH is set to anything but "" or "0".
func isGcloudTokenAuthEnabled() bool {
	raw, ok := os.LookupEnv(envGcloudTokenAuth)
	return ok && raw != "" && raw != "0"
}

func gcloudADCPath() string {
	if path := os.Getenv(envGoogleCredentials); path != "" {
		return path
	}
	var candidates []string
	if home, err := os.UserHomeDir(); err == nil {
		candidates = append(candidates, filepath.Join(home, ".config", "gcloud", gcloudADCFilename))
	}
	if appData := os.Getenv("APPDATA"); appData != "" {
		candidates = append(candidates, filepath.Join(appData, "gcloud", gcloudADCFilename))
	}
	for _, path := range candidates {
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	return ""
}

// resolveAuthorizedUserADC locates and reads the ADC file, returning usable authorized_user credentials or
// reporting the specific reason they are not.
func resolveAuthorizedUserADC() (authorizedUserADC, bool) {
	path := gcloudADCPath()
	if path == "" {
		reportDiagnostic("LiteLLM gcloud auth: No Google ADC file found. Set GOOGLE_APPLICATION_CREDENTIALS or run `gcloud auth application-default login`.")
		return authorizedUserADC{}, false
	}
	raw, err := os.ReadFile(path)
	var fields map[string]any
	if err != nil || json.Unmarshal(raw, &fields) != nil || fields == nil {
		reportDiagnostic("LiteLLM gcloud auth: Failed to read ADC file: " + path)
		return authorizedUserADC{}, false
	}
	credentialType, _ := fields["type"].(string)
	switch credentialType {
	case "authorized_user":
		field := func(name string) string {
			value, _ := fields[name].(string)
			if strings.TrimSpace(value) == "" {
				return ""
			}
			return value
		}
		credentials := authorizedUserADC{field("client_id"), field("client_secret"), field("refresh_token")}
		if credentials.ClientID == "" || credentials.ClientSecret == "" || credentials.RefreshToken == "" {
			reportDiagnostic("LiteLLM gcloud auth: authorized_user ADC has invalid or incomplete required fields.")
			return authorizedUserADC{}, false
		}
		return credentials, true
	case "service_account":
		reportDiagnostic("LiteLLM gcloud auth: Service account credentials are not supported; use authorized_user ADC.")
	default:
		shown := credentialType
		if _, present := fields["type"]; !present || credentialType == "" {
			shown = "missing"
		}
		reportDiagnostic("LiteLLM gcloud auth: Unknown credential type: " + shown)
	}
	return authorizedUserADC{}, false
}

// hasGcloudADCCredentials reports whether a complete authorized_user ADC file is present. It checks the
// credential's shape only and never contacts Google.
func hasGcloudADCCredentials() bool {
	_, ok := resolveAuthorizedUserADC()
	return ok
}

func exchangeRefreshToken(ctx context.Context, credentials authorizedUserADC) string {
	ctx, cancel := context.WithTimeout(ctx, gcloudExchangeLimit)
	defer cancel()
	form := url.Values{
		"grant_type":    {"refresh_token"},
		"client_id":     {credentials.ClientID},
		"client_secret": {credentials.ClientSecret},
		"refresh_token": {credentials.RefreshToken},
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, gcloudTokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		reportDiagnostic("LiteLLM gcloud auth: Token exchange failed.")
		return ""
	}
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	response, err := gcloudHTTPClient.Do(request)
	if err != nil {
		// A *url.Error embeds the request URL; report only the underlying cause.
		cause := error(err)
		if urlErr, ok := err.(*url.Error); ok {
			cause = urlErr.Err
		}
		reportDiagnostic("LiteLLM gcloud auth: Token exchange failed: " + cause.Error())
		return ""
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode > 299 {
		reportDiagnostic(fmt.Sprintf("LiteLLM gcloud auth: Token exchange failed (%d)", response.StatusCode))
		return ""
	}
	var data struct {
		AccessToken any `json:"access_token"`
	}
	body, err := io.ReadAll(io.LimitReader(response.Body, 1<<20))
	if err != nil || json.Unmarshal(body, &data) != nil {
		reportDiagnostic("LiteLLM gcloud auth: Token exchange failed: invalid response")
		return ""
	}
	token, _ := data.AccessToken.(string)
	return token
}

// getGcloudToken mints (or returns the cached) access token for the authorized_user ADC; "" on any failure.
func getGcloudToken(ctx context.Context) string {
	credentials, ok := resolveAuthorizedUserADC()
	if !ok {
		return ""
	}
	cacheKey := gcloudTokenCacheKey + ":authorized_user:" + credentials.ClientID + ":" + credentials.RefreshToken
	gcloudCache.Lock()
	defer gcloudCache.Unlock()
	if gcloudCache.token != "" && gcloudCache.key == cacheKey && now().Sub(gcloudCache.at) < gcloudCacheTTL {
		return gcloudCache.token
	}
	token := exchangeRefreshToken(ctx, credentials)
	if token != "" {
		gcloudCache.token, gcloudCache.key, gcloudCache.at = token, cacheKey, now()
	}
	return token
}

func resetGcloudTokenCache() {
	gcloudCache.Lock()
	defer gcloudCache.Unlock()
	gcloudCache.token, gcloudCache.key, gcloudCache.at = "", "", time.Time{}
}
