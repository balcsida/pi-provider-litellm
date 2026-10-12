package litellm

// Ports src/provider.ts: the LiteLLM native provider on ai.CreateProvider, its availability filter,
// the request guard around the stream functions, and the discovery-version aware refreshModels.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"sync"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/discover"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/protocols"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

const chatToolCap = 128

// discoveredCatalog is a discovery result together with the normalized root it was discovered from.
type discoveredCatalog struct {
	Models  []types.DiscoveredModel
	BaseURL string
}

// providerOptions is LiteLLMProviderOptions.
type providerOptions struct {
	ID      string
	Name    string
	BaseURL string
	Headers map[string]string
	Auth    ai.ProviderAuth
	// Models is the catalog discovered during activation, already projected by toNativeModels.
	Models            []types.DiscoveredModel
	AllowInsecureHTTP bool
	Policies          *policyStore
	// ResolveCredentialRoot returns "" when nothing configures a root.
	ResolveCredentialRoot func(credential *ai.Credential, requestRoot, apiKey string) (string, error)
	Discover              func(ctx context.Context, credential ai.Credential) (*discoveredCatalog, error)
}

// toNativeModels projects discovered models onto a provider and its normalized proxy root.
func toNativeModels(provider, baseURL string, models []types.DiscoveredModel, allowInsecureHTTP bool) ([]types.DiscoveredModel, error) {
	out := make([]types.DiscoveredModel, 0, len(models))
	for _, model := range models {
		resolved, err := protocols.ResolveModelBaseURL(baseURL, types.LiteLLMApi(model.API), allowInsecureHTTP)
		if err != nil {
			return nil, err
		}
		model.Provider, model.BaseURL = provider, resolved
		out = append(out, model)
	}
	return out, nil
}

// toAIModels converts discovered models to PiG models through the catalog JSON codec.
func toAIModels(provider string, models []types.DiscoveredModel) ([]ai.AnyModel, error) {
	raw := make([]json.RawMessage, 0, len(models))
	for _, model := range models {
		data, err := json.Marshal(model)
		if err != nil {
			return nil, err
		}
		raw = append(raw, data)
	}
	return ai.DecodeModelsCatalog(raw, provider)
}

type proxyRootInfo struct{ root, canonical, hostname string }

func refreshRequired(message string) error {
	return errors.New(message + "; a network refresh with a valid LiteLLM base URL is required")
}

// canonicalURL is the WHATWG href of a parsed root: lowercase host, default port dropped, "/" for an empty path.
func canonicalURL(u *url.URL) string {
	host := strings.ToLower(u.Hostname())
	if strings.Contains(host, ":") {
		host = "[" + host + "]"
	}
	if port := u.Port(); port != "" && !(port == "80" && u.Scheme == "http") && !(port == "443" && u.Scheme == "https") {
		host += ":" + port
	}
	path := u.EscapedPath()
	if path == "" {
		path = "/"
	}
	out := u.Scheme + "://" + host + path
	if u.RawQuery != "" {
		out += "?" + u.RawQuery
	}
	return out
}

func proxyRoot(baseURL, subject string, allowInsecureHTTP bool, verb string) (proxyRootInfo, error) {
	invalid := refreshRequired(subject + " " + verb + " an invalid LiteLLM model URL")
	root, err := protocols.NormalizeBaseURL(baseURL, allowInsecureHTTP)
	if err != nil {
		return proxyRootInfo{}, invalid
	}
	parsed, err := url.Parse(root)
	if err != nil || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return proxyRootInfo{}, invalid
	}
	return proxyRootInfo{root: root, canonical: canonicalURL(parsed), hostname: parsed.Hostname()}, nil
}

func activeCredentialRoot(root string, allowInsecureHTTP bool) (proxyRootInfo, error) {
	active, err := proxyRoot(root, "Active credentials", allowInsecureHTTP, "have")
	if err != nil {
		return active, err
	}
	if isPlaceholderHost(active.hostname) {
		return active, refreshRequired("Active credentials use a placeholder LiteLLM model host")
	}
	return active, nil
}

func modelRootError(model *ai.Model, active proxyRootInfo, allowInsecureHTTP bool) error {
	api := string(model.ProviderMeta.API)
	if !protocols.IsLiteLLMAPI(api) {
		names := make([]string, len(protocols.LiteLLMAPINames))
		for i, name := range protocols.LiteLLMAPINames {
			names[i] = string(name)
		}
		return fmt.Errorf("LiteLLM model %s declares unsupported protocol %q; set \"api\" to one of %s in models.json",
			model.ID, api, strings.Join(names, ", "))
	}
	stored, err := proxyRoot(model.ProviderMeta.BaseURL, "Cached model", allowInsecureHTTP, "has")
	if err != nil {
		return err
	}
	if isPlaceholderHost(stored.hostname) {
		return refreshRequired("Cached model uses a placeholder LiteLLM model host")
	}
	if stored.canonical != active.canonical {
		return refreshRequired(fmt.Sprintf("Cached model has stale LiteLLM model root %s; active credentials use %s", stored.root, active.root))
	}
	return nil
}

// requestModel validates a model against the active credential root and returns it projected onto that root.
func requestModel(provider string, model *ai.Model, transcript ai.TranscriptContext, credentialRoot string, allowInsecureHTTP bool, family types.BackendFamily) (*ai.Model, error) {
	if credentialRoot == "" {
		return nil, refreshRequired("Active credentials do not identify a LiteLLM model host")
	}
	active, err := activeCredentialRoot(credentialRoot, allowInsecureHTTP)
	if err != nil {
		return nil, err
	}
	if err := modelRootError(model, active, allowInsecureHTTP); err != nil {
		return nil, err
	}
	api := model.ProviderMeta.API
	if api == ai.APIOpenAICompletions && family == types.FamilyOpenAI {
		if count := len(ai.GetCurrentTools(transcript.Messages())); count > chatToolCap {
			return nil, fmt.Errorf("LiteLLM model %s uses Chat Completions with %d tools, exceeding the %d-tool cap; "+
				"route this model via Responses or reduce enabled extensions", model.ID, count, chatToolCap)
		}
	}
	baseURL, err := protocols.ResolveModelBaseURL(active.root, types.LiteLLMApi(api), allowInsecureHTTP)
	if err != nil {
		return nil, err
	}
	out := *model
	out.ProviderMeta.ProviderID, out.ProviderMeta.BaseURL = provider, baseURL
	return &out, nil
}

// isChatModelRecord reports whether a stored record is a chat model (type absent or "chat").
func isChatModelRecord(raw json.RawMessage) bool {
	var kind struct {
		Type string `json:"type"`
	}
	return json.Unmarshal(raw, &kind) == nil && (kind.Type == "" || kind.Type == string(ai.ModelTypeChat))
}

// createLiteLLMProvider is createLiteLLMProvider.
func createLiteLLMProvider(options providerOptions) (*ai.ModelsProvider, error) {
	var reportedMu sync.Mutex
	reported := map[string]bool{}
	reportUnavailable := func(message string) {
		reportedMu.Lock()
		first := !reported[message]
		reported[message] = true
		reportedMu.Unlock()
		if first {
			reportDiagnostic(fmt.Sprintf("LiteLLM (%s): %s", options.ID, message))
		}
	}
	policies := options.Policies
	if policies == nil {
		policies = newPolicyStore("")
	}
	baseline, err := toAIModels(options.ID, options.Models)
	if err != nil {
		return nil, err
	}
	name := options.Name
	provider := ai.CreateProvider(ai.CreateProviderOptions{
		ID:      options.ID,
		Name:    &name,
		BaseURL: options.BaseURL,
		Headers: providerHeaders(options.Headers),
		Auth:    options.Auth,
		Models:  baseline,
		FetchModels: func(refresh ai.RefreshModelsContext) ([]ai.AnyModel, error) {
			if refresh.Credential == nil {
				return nil, errors.New("LiteLLM model discovery requires a credential")
			}
			result, err := options.Discover(refresh.Signal, *refresh.Credential)
			if err != nil {
				return nil, err
			}
			baseURL := result.BaseURL
			if baseURL == "" {
				baseURL = options.BaseURL
			}
			native, err := toNativeModels(options.ID, baseURL, result.Models, options.AllowInsecureHTTP)
			if err != nil {
				return nil, err
			}
			if err := policies.replace(options.ID, native); err != nil {
				reportUnavailable("could not save model policies: " + err.Error())
			}
			return toAIModels(options.ID, native)
		},
		FilterModels: func(models []*ai.Model, credential *ai.Credential) []*ai.Model {
			root, err := options.ResolveCredentialRoot(credential, "", "")
			if err == nil && root == "" {
				if len(models) > 0 {
					reportUnavailable(fmt.Sprintf("%d model(s) hidden because no LiteLLM base URL is configured; "+
						"set LITELLM_BASE_URL or run /login litellm", len(models)))
				}
				return []*ai.Model{}
			}
			var active proxyRootInfo
			if err == nil {
				active, err = activeCredentialRoot(root, options.AllowInsecureHTTP)
			}
			if err != nil {
				reportUnavailable(err.Error())
				return []*ai.Model{}
			}
			available := []*ai.Model{}
			for _, model := range models {
				if err := modelRootError(model, active, options.AllowInsecureHTTP); err != nil {
					reportUnavailable(err.Error())
					continue
				}
				baseURL, err := protocols.ResolveModelBaseURL(active.root, types.LiteLLMApi(model.ProviderMeta.API), options.AllowInsecureHTTP)
				if err != nil {
					reportUnavailable(err.Error())
					continue
				}
				if baseURL != model.ProviderMeta.BaseURL {
					projected := *model
					projected.ProviderMeta.BaseURL = baseURL
					model = &projected
				}
				available = append(available, model)
			}
			return available
		},
		// Both entries map to ai.StreamSimple, as PiG's builtin providers do (ai/builtin_providers.go:85);
		// the guard below wraps each.
		API: &ai.ProviderStreams{Stream: ai.StreamSimple, StreamSimple: ai.StreamSimple},
	})

	guard := func(next ai.ModelsStreamFunction) ai.ModelsStreamFunction {
		return func(ctx context.Context, model *ai.Model, transcript ai.TranscriptContext, requestOptions ai.StreamOptions) (*ai.AssistantMessageEventStream, error) {
			root, err := options.ResolveCredentialRoot(nil, requestOptions.Env[envBaseURL], requestOptions.APIKey)
			if err != nil {
				return nil, err
			}
			_, family, _ := policies.policyFor(options.ID, model.ID)
			guarded, err := requestModel(options.ID, model, transcript, root, options.AllowInsecureHTTP, family)
			if err != nil {
				return nil, err
			}
			return next(ctx, guarded, transcript, requestOptions)
		}
	}
	provider.Stream, provider.StreamSimple = guard(provider.Stream), guard(provider.StreamSimple)

	inner := provider.RefreshModels
	seedKey := func(id, baseURL string) string { return id + "\x00" + baseURL }
	seeded := map[string]types.DiscoveredModel{}
	for _, model := range options.Models {
		seeded[seedKey(model.ID, model.BaseURL)] = model
	}
	provider.RefreshModels = func(refresh ai.RefreshModelsContext) error {
		legacyCount := 0
		var stored *ai.ModelsStoreEntry
		if refresh.Stored != nil {
			// PiG persists models of every type; this provider only publishes chat models, and the
			// policy helpers read chat-only fields.
			records := make([]json.RawMessage, 0, len(refresh.Stored.Models))
			for _, raw := range refresh.Stored.Models {
				if !isChatModelRecord(raw) {
					continue
				}
				var model types.DiscoveredModel
				if err := json.Unmarshal(raw, &model); err != nil {
					// One undecodable record must not abort the offline restore of the others.
					legacyCount++
					reportUnavailable("skipping an undecodable stored model: " + err.Error())
					continue
				}
				entry, hasEntry := policies.entry(options.ID, model.ID)
				version, known := policies.storedVersion(options.ID, model.ID)
				if !known || version != types.DiscoveryVersion {
					// PiG applies stored models over the seed, so a stale entry would otherwise mask what
					// startup discovery just proved for the same route.
					if fresh, ok := seeded[seedKey(model.ID, model.BaseURL)]; ok {
						model = fresh
					} else {
						legacyCount++
						if hasEntry {
							model = entry.apply(model)
						}
						model = discover.RestoreCachedModelPolicy(model)
						policies.remember(options.ID, model)
					}
				} else {
					model = discover.EnrichCachedModel(entry.apply(model))
				}
				data, err := json.Marshal(model)
				if err != nil {
					return err
				}
				records = append(records, data)
			}
			entry := *refresh.Stored
			entry.Models = records
			stored = &entry
		}
		forced := refresh.Force != nil && *refresh.Force
		if refresh.AllowNetwork && legacyCount > 0 {
			forced = true
		}
		next := refresh
		next.Stored = stored
		if forced {
			next.Force = &forced
		}
		err := inner(next)
		if err != nil && refresh.AllowNetwork && legacyCount > 0 {
			reportUnavailable(fmt.Sprintf("keeping cached models whose discovery metadata version does not match %d "+
				"because the required network refresh failed", types.DiscoveryVersion))
		}
		return err
	}
	return provider, nil
}
