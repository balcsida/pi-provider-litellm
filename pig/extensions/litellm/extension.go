package litellm

// Ports src/index.ts: the extension factory up to the /litellm-refresh command, plus the
// before_provider_headers and after_provider_response hooks. Cost, budget, skills, MCP and the request
// and message hooks plug in later through setup functions that take the shared *extensionState.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/MichaelKinsy/PiG/ai"
	sdk "github.com/MichaelKinsy/PiG/extensions/sdk"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/bridge"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/discover"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/protocols"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

const sessionHeader = "x-litellm-session-id"

// defaultLoginFlows holds the interactive login flows; the login phase assigns it before Extension runs.
var defaultLoginFlows loginHooks

type oauthRuntimeRoot struct{ apiKey, root string }

// extensionState is the state the factory shares with every setup function.
type extensionState struct {
	settings      *litellmSettings
	definitions   []providerDefinition
	providerNames map[string]bool
	policies      *policyStore

	mu                  sync.Mutex
	oauthRoots          map[string]oauthRuntimeRoot
	defaultRuntimeAuth  *types.LiteLLMRuntimeAuth
	loginGeneration     int
	networkAttempts     map[string]int
	onLoginStart        []func(providerDefinition)
	warnedFallbackRoute map[string]bool

	// MCP message buffering: Pi starts the terminal before supplying a context.
	sessionStarted  bool
	ui              *sdk.Context
	pendingMessages []pendingMessage
}

type pendingMessage struct{ text, level string }

func newExtensionState(settings *litellmSettings, definitions []providerDefinition, policies *policyStore) *extensionState {
	names := map[string]bool{}
	for _, definition := range definitions {
		names[definition.Name] = true
	}
	return &extensionState{
		settings: settings, definitions: definitions, providerNames: names, policies: policies,
		oauthRoots: map[string]oauthRuntimeRoot{}, networkAttempts: map[string]int{}, warnedFallbackRoute: map[string]bool{},
	}
}

func discoveryDisabledReason() string {
	if isOffline() {
		return envOffline + "=1"
	}
	if discoveryTimeoutMs() == 0 {
		return envTimeout + "=0"
	}
	return ""
}

func missingCredentials(definition providerDefinition) string {
	fix := "Set its baseUrl and apiKey"
	if definition.Name == providerName {
		fix = "Run /login litellm or set env vars"
	}
	return fmt.Sprintf("no credentials for %s. %s.", definition.Name, fix)
}

func requestBaseURL(definition providerDefinition) string {
	root, err := resolveCredentialRoot(definition, nil, "")
	if err != nil || root == "" {
		root = defaultLiteLLMBaseURL
	}
	return root + "/v1"
}

// credentialRoot is requireCredentialRoot(resolveCredentialRoot(definition, credential), name).
func credentialRoot(definition providerDefinition, credential *credentialInfo) (string, error) {
	root, err := resolveCredentialRoot(definition, credential, "")
	if err != nil {
		return "", err
	}
	return requireCredentialRoot(root, definition.Name)
}

func credentialFromStored(stored *storedCredential) *ai.Credential {
	if stored == nil {
		return nil
	}
	credential := &ai.Credential{Type: ai.CredentialType(stored.Type), Key: stored.Key, Env: stored.Env, Access: stored.Access, Refresh: stored.Refresh}
	if stored.Expires != nil {
		credential.Expires = *stored.Expires
	}
	if stored.BaseURL != "" {
		raw, _ := json.Marshal(stored.BaseURL)
		credential.Extra = map[string]json.RawMessage{"baseUrl": raw}
	}
	return credential
}

func (s *extensionState) resolveRoot(definition providerDefinition, credential *ai.Credential, requestRoot, apiKey string) (string, error) {
	if credential != nil {
		return resolveCredentialRoot(definition, credentialInfoFromAI(credential), requestRoot)
	}
	if explicit := cleanConfig(requestRoot); explicit != "" {
		return protocols.NormalizeBaseURL(explicit, definition.AllowInsecureHTTP)
	}
	s.mu.Lock()
	runtimeRoot, ok := s.oauthRoots[definition.Name]
	s.mu.Unlock()
	if ok && apiKey != "" && runtimeRoot.apiKey == apiKey {
		return runtimeRoot.root, nil
	}
	return resolveCredentialRoot(definition, nil, "")
}

func (s *extensionState) authForCredential(ctx context.Context, definition providerDefinition, credential *ai.Credential, executeHelpers bool) (types.LiteLLMRuntimeAuth, error) {
	if credential != nil && credential.Type == ai.CredentialOAuth {
		root, err := credentialRoot(definition, credentialInfoFromAI(credential))
		if err != nil {
			return types.LiteLLMRuntimeAuth{}, err
		}
		return types.LiteLLMRuntimeAuth{BaseURL: root, APIKey: credential.Access, Headers: resolveHeaders(definition), AllowInsecureHTTP: definition.AllowInsecureHTTP}, nil
	}
	resolved, err := resolveAPIKeyAuth(ctx, definition, processEnv, credential, executeHelpers)
	if err != nil {
		return types.LiteLLMRuntimeAuth{}, err
	}
	if resolved == nil || resolved.Auth.APIKey == "" || resolved.Auth.BaseURL == "" {
		return types.LiteLLMRuntimeAuth{}, fmt.Errorf("no credentials for %s. Run /login litellm or set env vars.", definition.Name)
	}
	var headers map[string]string
	for name, value := range resolved.Auth.Headers {
		if value != nil {
			if headers == nil {
				headers = map[string]string{}
			}
			headers[name] = *value
		}
	}
	return types.LiteLLMRuntimeAuth{BaseURL: resolved.Auth.BaseURL, APIKey: resolved.Auth.APIKey, Headers: headers, AllowInsecureHTTP: definition.AllowInsecureHTTP}, nil
}

// runtimeAuth is getRuntimeAuth: the credential the host resolves for a provider, or nil when it has none.
func (s *extensionState) runtimeAuth(ctx sdk.Context, definition providerDefinition) (*types.LiteLLMRuntimeAuth, error) {
	resolved, err := ctx.ModelRegistry().GetProviderAuth(definition.Name)
	if err != nil || resolved == nil {
		return nil, err
	}
	auth, _ := resolved["auth"].(map[string]any)
	env, _ := resolved["env"].(map[string]any)
	apiKey, _ := auth["apiKey"].(string)
	baseURL, _ := auth["baseUrl"].(string)
	baseURL = cleanConfig(baseURL)
	if baseURL == "" {
		envURL, _ := env[envBaseURL].(string)
		baseURL = cleanConfig(envURL)
	}
	if baseURL == "" || apiKey == "" {
		return nil, nil
	}
	normalized, err := protocols.NormalizeBaseURL(baseURL, definition.AllowInsecureHTTP)
	if err != nil {
		return nil, err
	}
	root, err := requireCredentialRoot(normalized, definition.Name)
	if err != nil {
		return nil, err
	}
	var headers map[string]string
	rawHeaders, _ := auth["headers"].(map[string]any)
	for name, value := range rawHeaders {
		if text, ok := value.(string); ok {
			if headers == nil {
				headers = map[string]string{}
			}
			headers[name] = text
		}
	}
	return &types.LiteLLMRuntimeAuth{BaseURL: root, APIKey: apiKey, Headers: headers, AllowInsecureHTTP: definition.AllowInsecureHTTP}, nil
}

// requireRuntimeAuth is requireRuntimeAuth.
func (s *extensionState) requireRuntimeAuth(ctx sdk.Context, definition providerDefinition) (types.LiteLLMRuntimeAuth, error) {
	auth, err := s.runtimeAuth(ctx, definition)
	if err != nil {
		return types.LiteLLMRuntimeAuth{}, err
	}
	if auth == nil {
		return types.LiteLLMRuntimeAuth{}, errors.New(missingCredentials(definition))
	}
	return *auth, nil
}

// resolveDefaultRuntimeAuth is resolveDefaultRuntimeAuth; ctx is nil when no host context is at hand.
func (s *extensionState) resolveDefaultRuntimeAuth(ctx *sdk.Context) (types.LiteLLMRuntimeAuth, error) {
	if ctx != nil {
		return s.requireRuntimeAuth(*ctx, s.definitions[0])
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.defaultRuntimeAuth == nil {
		return types.LiteLLMRuntimeAuth{}, errors.New("no credentials for litellm. Run /login litellm or set env vars.")
	}
	return *s.defaultRuntimeAuth, nil
}

func (s *extensionState) runDiscovery(ctx context.Context, auth types.LiteLLMRuntimeAuth) (*discoveredCatalog, error) {
	timeout := time.Duration(discoveryTimeoutMs()) * time.Millisecond
	modelsDev, cachePath := modelsDevDiscoveryOptions()
	options := discover.Options{
		DiscoveryOptions: types.DiscoveryOptions{
			Timeout: &timeout, Headers: auth.Headers, AllowInsecureHTTP: auth.AllowInsecureHTTP,
			ModelsDev: &modelsDev, ModelsDevCachePath: cachePath,
		},
		Silent: !isVerboseDiscovery(),
		Report: func(message string) { reportDiagnostic(strings.TrimSuffix(message, "\n")) },
	}
	if isVerboseDiscovery() {
		options.OnProgress = func(message string) { reportDiagnostic("LiteLLM: " + message) }
	}
	result, err := discover.DiscoverModels(ctx, auth.BaseURL, auth.APIKey, options)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	root, err := protocols.NormalizeBaseURL(auth.BaseURL, auth.AllowInsecureHTTP)
	if err != nil {
		return nil, err
	}
	return &discoveredCatalog{Models: result.Models, BaseURL: root}, nil
}

// seedModels discovers a provider's catalog at activation, the way 1.x did, because the host's startup
// refresh never allows network access.
func (s *extensionState) seedModels(definition providerDefinition) []types.DiscoveredModel {
	skip := func(reason string) []types.DiscoveredModel {
		if isVerboseDiscovery() {
			reportDiagnostic(fmt.Sprintf("LiteLLM (%s): startup discovery skipped (%s).", definition.Name, reason))
		}
		return nil
	}
	reason := discoveryDisabledReason()
	if reason == "" && isHostOffline() {
		reason = envPIOffline
		if _, pig := os.LookupEnv(envPIGOffline); pig {
			reason = envPIGOffline
		}
	}
	if reason != "" {
		return skip(reason)
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(seedTimeoutMsBudget())*time.Millisecond)
	defer cancel()
	stored := readStoredCredential(definition.Name, filepath.Join(agentDir(), "auth.json"))
	// executeHelpers false: activation must never run the user's key helper as a side effect, so
	// helper-backed setups keep waiting for the host's own refresh.
	auth, err := s.authForCredential(ctx, definition, credentialFromStored(stored), false)
	if err != nil {
		return skip(err.Error())
	}
	result, err := s.runDiscovery(ctx, auth)
	if err != nil {
		return skip(err.Error())
	}
	models, err := toNativeModels(definition.Name, result.BaseURL, result.Models, definition.AllowInsecureHTTP)
	if err != nil {
		return skip(err.Error())
	}
	if err := s.policies.replace(definition.Name, models); err != nil {
		reportDiagnostic(fmt.Sprintf("LiteLLM (%s): could not save model policies: %s", definition.Name, err))
	}
	return models
}

func (s *extensionState) loginGenerationNow() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.loginGeneration
}

// newProvider builds the native provider of one definition, with the refresh bookkeeping around it.
func (s *extensionState) newProvider(definition providerDefinition, seed []types.DiscoveredModel) (*ai.ModelsProvider, error) {
	var hooks *loginHooks
	if definition.Name == providerName {
		flows := defaultLoginFlows
		// Start drops anything bound to the old proxy root before the login can store a new credential.
		flows.Start = func() {
			s.mu.Lock()
			callbacks := append([]func(providerDefinition){}, s.onLoginStart...)
			s.mu.Unlock()
			for _, callback := range callbacks {
				callback(definition)
			}
		}
		flows.Complete = func() {
			s.mu.Lock()
			s.loginGeneration++
			s.mu.Unlock()
		}
		hooks = &flows
	}
	auth := createProviderAuth(definition,
		func() { s.mu.Lock(); delete(s.oauthRoots, definition.Name); s.mu.Unlock() },
		hooks,
		func() (string, string, bool) {
			s.mu.Lock()
			defer s.mu.Unlock()
			runtimeRoot, ok := s.oauthRoots[definition.Name]
			return runtimeRoot.apiKey, runtimeRoot.root, ok
		})
	if auth.OAuth != nil {
		toAuth := auth.OAuth.ToAuth
		auth.OAuth.ToAuth = func(credential ai.Credential) (ai.ModelAuth, error) {
			resolved, err := toAuth(credential)
			if err != nil {
				return resolved, err
			}
			root, err := credentialRoot(definition, credentialInfoFromAI(&credential))
			if err != nil {
				return ai.ModelAuth{}, err
			}
			s.mu.Lock()
			s.oauthRoots[definition.Name] = oauthRuntimeRoot{apiKey: credential.Access, root: root}
			s.mu.Unlock()
			// Pin the request host to the credential root so the host's fallback for an off-catalog model
			// cannot send this credential to a stale model base URL.
			resolved.BaseURL = root
			return resolved, nil
		}
	}
	provider, err := createLiteLLMProvider(providerOptions{
		ID:                definition.Name,
		Name:              definition.DisplayName,
		BaseURL:           requestBaseURL(definition),
		Headers:           resolveHeaders(definition),
		Auth:              auth,
		Models:            seed,
		AllowInsecureHTTP: definition.AllowInsecureHTTP,
		Policies:          s.policies,
		ResolveCredentialRoot: func(credential *ai.Credential, requestRoot, apiKey string) (string, error) {
			return s.resolveRoot(definition, credential, requestRoot, apiKey)
		},
		Discover: func(ctx context.Context, credential ai.Credential) (*discoveredCatalog, error) {
			if reason := discoveryDisabledReason(); reason != "" {
				return nil, fmt.Errorf("discovery disabled (%s)", reason)
			}
			runtimeAuth, err := s.authForCredential(ctx, definition, &credential, true)
			if err != nil {
				return nil, err
			}
			return s.runDiscovery(ctx, runtimeAuth)
		},
	})
	if err != nil {
		return nil, err
	}
	inner := provider.RefreshModels
	provider.RefreshModels = func(refresh ai.RefreshModelsContext) error {
		generation := s.loginGenerationNow()
		if refresh.AllowNetwork {
			s.mu.Lock()
			s.networkAttempts[definition.Name]++
			s.mu.Unlock()
		}
		err := inner(refresh)
		if definition.Name == providerName && refresh.AllowNetwork && discoveryDisabledReason() == "" &&
			generation == s.loginGenerationNow() && refresh.Credential != nil {
			// Best-effort: caching the default auth must not let a bad credential override the
			// refresh outcome.
			signal := refresh.Signal
			if signal == nil {
				signal = context.Background()
			}
			if cached, authErr := s.authForCredential(signal, definition, refresh.Credential, true); authErr == nil {
				s.mu.Lock()
				if generation == s.loginGeneration {
					s.defaultRuntimeAuth = &cached
				}
				s.mu.Unlock()
			}
		}
		return err
	}
	return provider, nil
}

// notify shows a message in the UI, or on the diagnostic stream when there is none.
func notify(ctx sdk.Context, message, level string) {
	if ctx.HasUI() {
		ctx.Notify(message, level)
		return
	}
	reportDiagnostic(message)
}

// notifyMcp is notifyMcp: it buffers until session_start while the terminal is starting.
func (s *extensionState) notifyMcp(message, level string) {
	text := strings.TrimRight(message, " \t\r\n")
	s.mu.Lock()
	ui, started := s.ui, s.sessionStarted
	if ui == nil && !started && stderrIsTerminal() {
		s.pendingMessages = append(s.pendingMessages, pendingMessage{text, level})
		s.mu.Unlock()
		return
	}
	s.mu.Unlock()
	if ui != nil {
		ui.Notify(text, level)
	} else {
		reportDiagnostic(text)
	}
}

func stderrIsTerminal() bool {
	info, err := os.Stderr.Stat()
	return err == nil && info.Mode()&os.ModeCharDevice != 0
}

func setupSessionState(e *sdk.Extension, s *extensionState) {
	e.OnSessionStart(func(ctx sdk.Context, _ map[string]any) (any, error) {
		s.mu.Lock()
		s.sessionStarted = true
		s.ui = nil
		if ctx.HasUI() {
			s.ui = &ctx
		}
		pending := s.pendingMessages
		s.pendingMessages = nil
		s.mu.Unlock()
		for _, message := range pending {
			s.notifyMcp(message.text, message.level)
		}
		return nil, nil
	})
	e.OnSessionShutdown(func(sdk.Context, map[string]any) (any, error) {
		s.mu.Lock()
		s.ui, s.sessionStarted, s.pendingMessages = nil, false, nil
		s.mu.Unlock()
		return nil, nil
	})
}

func (s *extensionState) inScope(ctx sdk.Context) bool {
	return s.providerNames[ctx.ModelProvider()]
}

func setupRefreshCommand(e *sdk.Extension, s *extensionState) {
	e.RegisterCommand("litellm-refresh", sdk.CommandOptions{
		Description: "Refresh LiteLLM model catalogs from the proxy, even with PI_OFFLINE set",
		GetArgumentCompletions: func(prefix string) ([]sdk.AutocompleteItem, error) {
			var items []sdk.AutocompleteItem
			for _, definition := range s.definitions {
				if strings.HasPrefix(definition.Name, prefix) {
					items = append(items, sdk.AutocompleteItem{Value: definition.Name, Label: definition.Name})
				}
			}
			return items, nil
		},
		Handler: func(ctx sdk.Context, args string) error {
			if reason := discoveryDisabledReason(); reason != "" {
				notify(ctx, fmt.Sprintf("LiteLLM: model refresh skipped (%s).", reason), "warning")
				return nil
			}
			requested := strings.TrimSpace(args)
			var selected []providerDefinition
			for _, definition := range s.definitions {
				if requested == "" || definition.Name == requested {
					selected = append(selected, definition)
				}
			}
			if len(selected) == 0 {
				configured := make([]string, 0, len(s.definitions))
				for _, definition := range s.definitions {
					configured = append(configured, jsonString(definition.Name))
				}
				notify(ctx, fmt.Sprintf("LiteLLM: unknown provider %s; configured: %s.", jsonString(requested), strings.Join(configured, ", ")), "warning")
				return nil
			}
			attempts := map[string]int{}
			names := make([]string, 0, len(selected))
			s.mu.Lock()
			for _, definition := range selected {
				attempts[definition.Name] = s.networkAttempts[definition.Name]
				names = append(names, definition.Name)
			}
			s.mu.Unlock()
			yes := true
			result, err := ctx.ModelRegistry().Refresh(sdk.ModelsRefreshOptions{Providers: names, AllowNetwork: &yes, Force: &yes})
			if err != nil {
				return err
			}
			models, err := ctx.ModelRegistry().GetAll()
			if err != nil {
				return err
			}
			for _, definition := range selected {
				label := "LiteLLM (" + jsonString(definition.Name) + ")"
				s.mu.Lock()
				attempted := s.networkAttempts[definition.Name]
				s.mu.Unlock()
				if message, failed := result.Errors[definition.Name]; failed {
					notify(ctx, fmt.Sprintf("%s: model refresh failed (%s).", label, message), "warning")
				} else if attempted == attempts[definition.Name] {
					// The host runs a provider's network phase only once a credential resolves, so an
					// unchanged count means "no credentials" rather than a successful refresh.
					notify(ctx, fmt.Sprintf("%s: model refresh skipped; %s", label, missingCredentials(definition)), "warning")
				} else {
					count := 0
					for _, model := range models {
						if model["provider"] == definition.Name {
							count++
						}
					}
					plural := "s"
					if count == 1 {
						plural = ""
					}
					notify(ctx, fmt.Sprintf("%s: refreshed %d model%s.", label, count, plural), "info")
				}
			}
			return nil
		},
	})
}

func jsonString(value string) string {
	data, _ := json.Marshal(value)
	return string(data)
}

func setupSessionHeader(e *sdk.Extension, s *extensionState) {
	e.OnEvent(sdk.EventBeforeProviderHeaders, func(ctx sdk.Context, data map[string]any) (any, error) {
		if !s.inScope(ctx) {
			return nil, nil
		}
		headers, ok := data["headers"].(map[string]any)
		if !ok {
			return nil, nil
		}
		sessionID, err := ctx.GetSessionID()
		if err != nil {
			return nil, nil
		}
		headers[sessionHeader] = sessionID
		// The host applies the headers a handler returns, as its Node runtime does.
		return headers, nil
	})
}

// attemptedFallbacks parses x-litellm-attempted-fallbacks like Number(): a positive safe integer, else 0.
func attemptedFallbacks(headers map[string]any) int {
	for name, value := range headers {
		if !strings.EqualFold(name, "x-litellm-attempted-fallbacks") {
			continue
		}
		text, ok := value.(string)
		if !ok {
			return 0
		}
		number, err := strconv.ParseFloat(strings.TrimSpace(text), 64)
		if err != nil || number != math.Trunc(number) || number <= 0 || number > 1<<53-1 {
			return 0
		}
		return int(number)
	}
	return 0
}

// setupFallbackWarning names the route only: router fallbacks are invisible to discovery, and header text is proxy-supplied.
func setupFallbackWarning(e *sdk.Extension, s *extensionState) {
	e.OnEvent(sdk.EventAfterProviderResponse, func(ctx sdk.Context, data map[string]any) (any, error) {
		if !s.inScope(ctx) {
			return nil, nil
		}
		if status, _ := data["status"].(float64); status < 200 || status >= 300 {
			return nil, nil
		}
		headers, _ := data["headers"].(map[string]any)
		if attemptedFallbacks(headers) <= 0 {
			return nil, nil
		}
		provider, modelID := ctx.ModelProvider(), ctx.Model()
		key := provider + "\x00" + modelID
		s.mu.Lock()
		warned := s.warnedFallbackRoute[key]
		s.warnedFallbackRoute[key] = true
		s.mu.Unlock()
		if warned {
			return nil, nil
		}
		route := jsonString(modelID)
		notify(ctx, fmt.Sprintf("LiteLLM (%s): the request for %s was answered by a fallback, but protocol and model handling were chosen "+
			"for %s's own deployments, not the fallback's. If its fallbacks cross model families, pin the protocol with "+
			"`model_info.supported_endpoints`.", jsonString(provider), route, route), "warning")
		return nil, nil
	})
}

// Extension is the extension factory.
func Extension() *sdk.Extension {
	e := sdk.New("litellm")
	settings := readGlobalLiteLLMSettings()
	definitions := getProviderDefinitions(settings)
	state := newExtensionState(settings, definitions, newPolicyStore(filepath.Join(agentDir(), policiesFilename)))

	type seeded struct {
		index  int
		models []types.DiscoveredModel
	}
	results := make(chan seeded, len(definitions))
	for i, definition := range definitions {
		go func() { results <- seeded{i, state.seedModels(definition)} }()
	}
	seeds := make([][]types.DiscoveredModel, len(definitions))
	for range definitions {
		result := <-results
		seeds[result.index] = result.models
	}

	for i, definition := range definitions {
		provider, err := state.newProvider(definition, seeds[i])
		if err == nil {
			err = e.RegisterNativeProvider(bridge.Wrap(provider))
		}
		if err != nil {
			reportDiagnostic(fmt.Sprintf("LiteLLM (%s): provider not registered (%s).", definition.Name, err))
		}
	}

	setupSessionState(e, state)
	setupRefreshCommand(e, state)
	setupSessionHeader(e, state)
	setupFallbackWarning(e, state)
	return e
}
