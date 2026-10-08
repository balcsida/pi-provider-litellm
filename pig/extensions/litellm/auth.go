package litellm

// Ports src/index.ts: credential roots, API-key auth resolution and createProviderAuth.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"path/filepath"
	"strings"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/protocols"
)

// defaultLiteLLMBaseURL is the placeholder root used before any proxy is configured.
const defaultLiteLLMBaseURL = "https://litellm.example.com"

// isPlaceholderHost reports whether hostname is the placeholder host (a trailing dot is ignored).
func isPlaceholderHost(hostname string) bool {
	placeholder, _ := url.Parse(defaultLiteLLMBaseURL)
	return strings.TrimSuffix(hostname, ".") == strings.TrimSuffix(placeholder.Hostname(), ".")
}

// loginHooks plugs the interactive flows into createProviderAuth. A nil flow means that auth method is not
// offered: without LoginAPIKey the API-key method has no login, and OAuth needs LoginOAuth and Refresh.
type loginHooks struct {
	LoginAPIKey func(ctx context.Context, interaction ai.AuthInteraction, definition providerDefinition) (ai.Credential, error)
	LoginOAuth  func(ctx context.Context, interaction ai.AuthInteraction, definition providerDefinition) (ai.Credential, error)
	Refresh     func(ctx context.Context, credential ai.Credential, definition providerDefinition) (ai.Credential, error)
	// Start and Complete bracket a login; either may be nil.
	Start    func()
	Complete func()
}

// credentialInfo is the part of a credential that names a proxy root.
type credentialInfo struct {
	Type    string
	BaseURL string            // oauth
	Env     map[string]string // api_key
}

func credentialInfoFromAI(credential *ai.Credential) *credentialInfo {
	if credential == nil {
		return nil
	}
	info := &credentialInfo{Type: string(credential.Type), Env: credential.Env}
	if raw, ok := credential.Extra["baseUrl"]; ok {
		_ = json.Unmarshal(raw, &info.BaseURL)
	}
	return info
}

func credentialInfoFromStored(stored *storedCredential) *credentialInfo {
	if stored == nil {
		return nil
	}
	return &credentialInfo{Type: stored.Type, BaseURL: stored.BaseURL, Env: stored.Env}
}

// resolveCredentialRoot picks the proxy root: credential, then request, then settings, then environment.
// It returns "" when none is configured.
func resolveCredentialRoot(definition providerDefinition, credential *credentialInfo, requestBaseURL string) (string, error) {
	var baseURL string
	if credential != nil {
		switch credential.Type {
		case "oauth":
			baseURL = cleanConfig(credential.BaseURL)
		case "api_key":
			baseURL = cleanConfig(credential.Env[envBaseURL])
		}
	}
	baseURL = firstNonEmpty(baseURL, cleanConfig(requestBaseURL), cleanConfig(definition.BaseURL))
	if baseURL == "" && definition.UseDefaultEnv {
		baseURL = cleanConfig(processEnvValue(envBaseURL))
	}
	if baseURL == "" {
		return "", nil
	}
	return protocols.NormalizeBaseURL(baseURL, definition.AllowInsecureHTTP)
}

func processEnvValue(name string) string {
	value, _ := processEnv(name)
	return value
}

func requireCredentialRoot(root, name string) (string, error) {
	if root == "" {
		return "", fmt.Errorf("no LiteLLM base URL for %s. Run /login litellm or set env vars.", name)
	}
	parsed, err := url.Parse(root)
	if err != nil || isPlaceholderHost(parsed.Hostname()) {
		return "", fmt.Errorf("placeholder LiteLLM base URL for %s. Run /login litellm or set env vars.", name)
	}
	return root, nil
}

// configuredBaseURL is the unnormalized base URL: credential env, settings, then the context environment.
func configuredBaseURL(definition providerDefinition, env envLookup, credential *ai.Credential) string {
	if credential != nil {
		if value := cleanConfig(credential.Env[envBaseURL]); value != "" {
			return value
		}
	}
	if value := cleanConfig(definition.BaseURL); value != "" {
		return value
	}
	if definition.UseDefaultEnv {
		value, _ := env(envBaseURL)
		return cleanConfig(value)
	}
	return ""
}

func providerHeaders(headers map[string]string) ai.ProviderHeaders {
	if headers == nil {
		return nil
	}
	out := make(ai.ProviderHeaders, len(headers))
	for name, value := range headers {
		out[name] = new(value)
	}
	return out
}

// resolveAPIKeyAuth resolves the API-key auth of a definition through the context environment. With
// executeHelpers false no `!command` or ADC exchange runs. It returns nil when no key is configured.
func resolveAPIKeyAuth(ctx context.Context, definition providerDefinition, env envLookup, credential *ai.Credential, executeHelpers bool) (*ai.AuthResult, error) {
	baseURL := configuredBaseURL(definition, env, credential)
	var stored string
	if credential != nil && credential.Key != "" {
		value, ok, err := resolveConfigValue(ctx, credential.Key, executeHelpers)
		if err != nil {
			return nil, err
		}
		if ok {
			stored = strings.TrimSpace(value)
		}
	}
	var source, apiKey, apiKeyConfig string
	var fromGcloud bool
	if stored != "" {
		source, apiKey = "stored credential", stored
	} else {
		stripped := definition
		stripped.APIKeyConfig, stripped.UseDefaultEnv = "", false
		creds, err := resolveCredentials(ctx, stripped, executeHelpers)
		if err != nil {
			return nil, err
		}
		apiKey, apiKeyConfig, fromGcloud = creds.APIKey, creds.APIKeyConfig, creds.APIKeyFromGcloudADC
		if apiKey == "" && definition.APIKeyConfig != "" {
			var configured string
			if strings.HasPrefix(definition.APIKeyConfig, "!") {
				if executeHelpers {
					if configured, err = executeAPIKeyCommand(ctx, definition.APIKeyConfig); err != nil {
						return nil, err
					}
				}
			} else {
				configured, _ = resolveTemplateConfigValue(definition.APIKeyConfig, env)
			}
			if configured != "" {
				apiKey, apiKeyConfig = configured, definition.APIKeyConfig
			} else if !strings.HasPrefix(definition.APIKeyConfig, "!") {
				warnUnresolvedAPIKeyConfig(definition.Name, definition.APIKeyConfig)
			}
		}
		if apiKey == "" && definition.UseDefaultEnv {
			helperValue, _ := env(envAPIKeyHelper)
			if helper := normalizeCommand(helperValue); helper != "" {
				if executeHelpers {
					if apiKey, err = executeAPIKeyCommand(ctx, helper); err != nil {
						return nil, err
					}
				}
				apiKeyConfig, source = helper, envAPIKeyHelper
			} else {
				keyValue, _ := env(envAPIKey)
				if envKey := cleanConfig(keyValue); envKey != "" {
					apiKey, apiKeyConfig, source = envKey, envAPIKey, envAPIKey
				}
			}
		}
	}
	if apiKey == "" {
		return nil, nil
	}
	var normalizedRoot string
	if baseURL != "" {
		root, err := protocols.NormalizeBaseURL(baseURL, definition.AllowInsecureHTTP)
		if err != nil {
			return nil, err
		}
		normalizedRoot = root
	}
	// Pin the request host to the root resolved from the credential, so the key never reaches a stale or
	// foreign model base URL. Fail rather than return a key with no pinned root.
	root, err := resolveCredentialRoot(definition, credentialInfoFromAI(credential), normalizedRoot)
	if err != nil {
		return nil, err
	}
	pinnedRoot, err := requireCredentialRoot(root, definition.Name)
	if err != nil {
		return nil, err
	}
	result := &ai.AuthResult{
		Auth: ai.ModelAuth{
			APIKey:  apiKey,
			Headers: providerHeaders(resolveHeadersFrom(definition, env)),
			BaseURL: pinnedRoot,
		},
		Source: firstNonEmpty(source, map[bool]string{true: gcloudADCSource}[fromGcloud], apiKeyConfig, envAPIKey),
	}
	if normalizedRoot != "" {
		result.Env = map[string]string{envBaseURL: normalizedRoot}
	}
	return result, nil
}

// authEnv returns the context environment accessor, or the process one when the host supplies none.
func authEnv(authCtx ai.AuthContext) envLookup {
	if authCtx.Env == nil {
		return processEnv
	}
	return envLookup(authCtx.Env)
}

// createProviderAuth builds the provider's auth methods. clearOAuthRuntimeRoot and oauthRuntimeRoot may be
// nil; oauthRuntimeRoot returns the key and root this process last resolved for an OAuth login.
func createProviderAuth(
	definition providerDefinition,
	clearOAuthRuntimeRoot func(),
	hooks *loginHooks,
	oauthRuntimeRoot func() (apiKey, root string, ok bool),
) ai.ProviderAuth {
	if hooks == nil {
		hooks = &loginHooks{}
	}
	start := func() {
		if hooks.Start != nil {
			hooks.Start()
		}
	}
	complete := func(credential ai.Credential, err error) (ai.Credential, error) {
		if err == nil && hooks.Complete != nil {
			hooks.Complete()
		}
		return credential, err
	}

	apiKey := &ai.APIKeyAuth{Name: definition.DisplayName + " API key"}
	if definition.Name == providerName && hooks.LoginAPIKey != nil {
		apiKey.Login = func(ctx context.Context, interaction ai.AuthInteraction) (ai.Credential, error) {
			start()
			return complete(hooks.LoginAPIKey(ctx, interaction, definition))
		}
	}
	apiKey.Check = func(ctx context.Context, input ai.APIKeyAuthInput) (*ai.AuthCheck, error) {
		env := authEnv(input.Ctx)
		var baseURL string
		if input.Credential != nil {
			baseURL = input.Credential.Env[envBaseURL]
		}
		if baseURL == "" {
			baseURL = definition.BaseURL
		}
		if baseURL == "" && definition.UseDefaultEnv {
			baseURL, _ = env(envBaseURL)
		}
		if cleanConfig(baseURL) == "" {
			return nil, nil
		}
		if input.Credential != nil && input.Credential.Key != "" {
			return &ai.AuthCheck{Type: ai.CredentialAPIKey, Source: "stored credential"}, nil
		}
		// Mirror the precedence in resolveCredentials, where ADC outranks the config key, the helper and the
		// environment key. Whether the refresh token still mints is only knowable at request time, and this
		// must not make a network call; if it fails, Resolve falls back and reports what it used.
		if definition.UseGcloudTokenAuth && isGcloudTokenAuthEnabled() && hasGcloudADCCredentials() {
			return &ai.AuthCheck{Type: ai.CredentialAPIKey, Source: gcloudADCSource}, nil
		}
		if definition.APIKeyConfig != "" {
			configured := definition.APIKeyConfig
			if !strings.HasPrefix(configured, "!") {
				configured, _ = resolveTemplateConfigValue(configured, env)
			}
			if configured != "" {
				return &ai.AuthCheck{Type: ai.CredentialAPIKey, Source: definition.APIKeyConfig}, nil
			}
		}
		if definition.UseDefaultEnv {
			if value, _ := env(envAPIKeyHelper); cleanConfig(value) != "" {
				return &ai.AuthCheck{Type: ai.CredentialAPIKey, Source: envAPIKeyHelper}, nil
			}
			if value, _ := env(envAPIKey); cleanConfig(value) != "" {
				return &ai.AuthCheck{Type: ai.CredentialAPIKey, Source: envAPIKey}, nil
			}
		}
		return nil, nil
	}
	apiKey.Resolve = func(ctx context.Context, input ai.APIKeyAuthInput) (*ai.AuthResult, error) {
		env := authEnv(input.Ctx)
		credential := input.Credential
		// Pi re-resolves auth through this path whenever a caller passes an explicit apiKey override, which
		// bypasses the stored OAuth credential carrying the base URL. Read the root back from auth.json, or
		// from the root this process resolved for that exact token, but only when nothing configures a root:
		// a configured base URL still resolves, and fails, on its own terms.
		if credential != nil && credential.Key != "" && configuredBaseURL(definition, env, credential) == "" {
			stored := readStoredCredential(definition.Name, filepath.Join(agentDir(), "auth.json"))
			var root string
			if stored != nil && stored.Type == "oauth" && stored.Access == credential.Key {
				resolved, err := resolveCredentialRoot(definition, credentialInfoFromStored(stored), "")
				if err != nil {
					return nil, err
				}
				if root, err = requireCredentialRoot(resolved, definition.Name); err != nil {
					return nil, err
				}
			} else if oauthRuntimeRoot != nil {
				if rememberedKey, rememberedRoot, ok := oauthRuntimeRoot(); ok && rememberedKey == credential.Key {
					root = rememberedRoot
				}
			}
			if root != "" {
				return &ai.AuthResult{
					Auth:   ai.ModelAuth{APIKey: credential.Key, Headers: providerHeaders(resolveHeaders(definition)), BaseURL: root},
					Env:    map[string]string{envBaseURL: root},
					Source: "OAuth",
				}, nil
			}
		}
		if clearOAuthRuntimeRoot != nil {
			clearOAuthRuntimeRoot()
		}
		return resolveAPIKeyAuth(ctx, definition, env, credential, true)
	}

	auth := ai.ProviderAuth{APIKey: apiKey}
	if definition.EnableOAuth && hooks.LoginOAuth != nil && hooks.Refresh != nil {
		auth.OAuth = &ai.OAuthAuth{
			Name:       "LiteLLM SSO",
			LoginLabel: "Sign in with LiteLLM SSO",
			Login: func(ctx context.Context, interaction ai.AuthInteraction, _ ai.LoginOptions) (ai.Credential, error) {
				start()
				return complete(hooks.LoginOAuth(ctx, interaction, definition))
			},
			Refresh: func(ctx context.Context, credential ai.Credential) (ai.Credential, error) {
				refreshed, err := hooks.Refresh(ctx, credential, definition)
				if err != nil {
					return ai.Credential{}, err
				}
				refreshed.Type = ai.CredentialOAuth
				return refreshed, nil
			},
			ToAuth: func(credential ai.Credential) (ai.ModelAuth, error) {
				if credential.Access == "" {
					return ai.ModelAuth{}, errors.New("LiteLLM SSO credential has no access token")
				}
				return ai.ModelAuth{APIKey: credential.Access, Headers: providerHeaders(resolveHeaders(definition))}, nil
			},
		}
	}
	return auth
}
