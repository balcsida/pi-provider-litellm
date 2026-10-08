// Ports src/public-catalog.ts
package catalog

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/MichaelKinsy/PiG/ai"
)

const (
	ModelsDevURL          = "https://models.dev/api.json"
	modelsDevCacheTTL     = 28 * 24 * time.Hour
	defaultTimeout        = 5 * time.Second
	piAzureLegacyProvider = "azure-openai-responses"
)

// piModels reads PiG's generated catalog; tests replace it.
var piModels = ai.ListModels

var knownProviders = func() map[string]bool {
	known := map[string]bool{}
	for _, provider := range ai.ListProviders() {
		known[provider] = true
	}
	return known
}()

// PiAzureProvider is the Pi catalog's Azure provider id (Pi 1.0.3 renamed `azure-openai-responses` to `azure`).
var PiAzureProvider = func() string {
	if knownProviders["azure"] {
		return "azure"
	}
	return piAzureLegacyProvider
}()

// Limits are token limits; a nil field is unknown.
type Limits struct {
	Context *float64
	Output  *float64
}

// Cost is USD per million tokens; a nil field is unknown.
type Cost struct {
	Input      *float64
	Output     *float64
	CacheRead  *float64
	CacheWrite *float64
}

// PublicCatalogRecord is one public-catalog answer. Source is "models.dev", "pi-vendor" or "pi-adapter".
type PublicCatalogRecord struct {
	Source   string
	Provider string
	// PiProvider names the Pi catalog for fields a models.dev record omits, when the record was found
	// under another vendor's key (ChatGPT routes read OpenAI's but bill as Codex).
	PiProvider       string
	ModelID          string
	Limits           *Limits
	Cost             *Cost
	Modalities       []string
	EffortLevels     []string
	ThinkingLevelMap map[string]*string
}

// PublicCatalog answers lookups; Lookup returns nil for "no opinion".
type PublicCatalog interface {
	Lookup(provider, id string) *PublicCatalogRecord
}

// LoadPublicCatalogOptions mirrors the TypeScript options. A nil Offline falls back to LITELLM_OFFLINE=1.
// CachePath empty keeps the models.dev document in memory only.
type LoadPublicCatalogOptions struct {
	CachePath string
	Offline   *bool
	Timeout   time.Duration
	Client    *http.Client
	// URL overrides ModelsDevURL (tests point it at an httptest server).
	URL string
}

type modelsDevCatalog map[string]map[string]map[string]any // provider → model id → raw record

type cacheFile struct {
	fetchedAt float64 // epoch milliseconds
	catalog   modelsDevCatalog
}

type refreshCall struct {
	done    chan struct{}
	catalog modelsDevCatalog
}

var (
	cacheMu      sync.Mutex
	memoryCaches = map[string]*cacheFile{}
	refreshes    = map[string]*refreshCall{}
)

var providerAliases = map[string][]string{
	"anthropic":        {"anthropic"},
	"azure":            {"azure", "openai"},
	"azure_ai":         {"azure", "openai"},
	"bedrock":          {"amazon-bedrock"},
	"bedrock_converse": {"amazon-bedrock"},
	// models.dev has no ChatGPT subscription provider; its routes serve OpenAI models.
	"chatgpt":      {"chatgpt", "openai"},
	"deepseek":     {"deepseek"},
	"fireworks":    {"fireworks-ai"},
	"fireworks_ai": {"fireworks-ai"},
	"gemini":       {"google"},
	"moonshot":     {"moonshotai"},
	"nvidia_nim":   {"nvidia"},
	"openai":       {"openai"},
	"together_ai":  {"together"},
	"vertex_ai":    {"google-vertex"},
}

var piProviderAliases = map[string][]string{
	"amazon-bedrock": {"amazon-bedrock"},
	"azure":          {PiAzureProvider},
	"azure_ai":       {PiAzureProvider},
	"chatgpt":        {"openai-codex"},
	"fireworks-ai":   {"fireworks"},
	"openai":         {"openai"},
}

func piAliases(provider string) []string {
	if aliases, ok := piProviderAliases[provider]; ok {
		return aliases
	}
	return []string{provider}
}

func record(value any) map[string]any {
	m, _ := value.(map[string]any)
	return m
}

func finite(value any) *float64 {
	if n, ok := value.(float64); ok && !math.IsNaN(n) && !math.IsInf(n, 0) {
		return &n
	}
	return nil
}

// A token price below zero is malformed catalog data, not a discount; treating it as
// missing lets the normal incomplete-cost handling apply instead of publishing it.
func price(value any) *float64 {
	if parsed := finite(value); parsed != nil && *parsed >= 0 {
		return parsed
	}
	return nil
}

func normalizeCatalog(value any) modelsDevCatalog {
	root := record(value)
	if root == nil {
		return nil
	}
	result := modelsDevCatalog{}
	for provider, rawProvider := range root {
		models := record(record(rawProvider)["models"])
		if models == nil {
			continue
		}
		entries := map[string]map[string]any{}
		for id, rawModel := range models {
			if model := record(rawModel); model != nil {
				entries[id] = model
			}
		}
		result[provider] = entries
	}
	return result
}

func nowMillis() float64 { return float64(time.Now().UnixMilli()) }

func writeJSONAtomic(path string, cache *cacheFile) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o777); err != nil {
		return err
	}
	providers := map[string]any{}
	for provider, models := range cache.catalog {
		providers[provider] = map[string]any{"models": models}
	}
	data, err := json.MarshalIndent(map[string]any{"fetchedAt": cache.fetchedAt, "catalog": providers}, "", "  ")
	if err != nil {
		return err
	}
	temporaryPath := fmt.Sprintf("%s.%d.%d.tmp", path, os.Getpid(), time.Now().UnixMilli())
	if err := os.WriteFile(temporaryPath, data, 0o666); err != nil {
		return err
	}
	if err := os.Rename(temporaryPath, path); err != nil {
		_ = os.Remove(temporaryPath)
		return err
	}
	return nil
}

func readCache(path string) *cacheFile {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var parsed any
	if json.Unmarshal(data, &parsed) != nil {
		return nil
	}
	root := record(parsed)
	if root == nil {
		return nil
	}
	fetchedAt := finite(root["fetchedAt"])
	if fetchedAt == nil || *fetchedAt < 0 || *fetchedAt > nowMillis() {
		return nil
	}
	catalog := normalizeCatalog(root["catalog"])
	if catalog == nil {
		return nil
	}
	return &cacheFile{fetchedAt: *fetchedAt, catalog: catalog}
}

func refreshCatalog(ctx context.Context, key string, options LoadPublicCatalogOptions) modelsDevCatalog {
	cacheMu.Lock()
	if active, ok := refreshes[key]; ok {
		cacheMu.Unlock()
		<-active.done
		return active.catalog
	}
	call := &refreshCall{done: make(chan struct{})}
	refreshes[key] = call
	cacheMu.Unlock()
	defer func() {
		cacheMu.Lock()
		delete(refreshes, key)
		cacheMu.Unlock()
		close(call.done)
	}()
	call.catalog = fetchCatalog(ctx, key, options)
	return call.catalog
}

func fetchCatalog(ctx context.Context, key string, options LoadPublicCatalogOptions) modelsDevCatalog {
	timeout := options.Timeout
	if timeout == 0 {
		timeout = defaultTimeout
	}
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	url := options.URL
	if url == "" {
		url = ModelsDevURL
	}
	client := options.Client
	if client == nil {
		client = http.DefaultClient
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil
	}
	request.Header.Set("Accept", "application/json")
	response, err := client.Do(request)
	if err != nil {
		return nil
	}
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode > 299 {
		return nil
	}
	body, err := io.ReadAll(response.Body)
	if err != nil {
		return nil
	}
	var parsed any
	if json.Unmarshal(body, &parsed) != nil {
		return nil
	}
	catalog := normalizeCatalog(parsed)
	if catalog == nil {
		return nil
	}
	cache := &cacheFile{fetchedAt: nowMillis(), catalog: catalog}
	cacheMu.Lock()
	memoryCaches[key] = cache
	cacheMu.Unlock()
	if options.CachePath != "" {
		_ = writeJSONAtomic(options.CachePath, cache)
	}
	return catalog
}

func providerCandidates(provider string) []string {
	if provider == "" {
		return nil
	}
	normalized := strings.ToLower(strings.TrimSpace(provider))
	if aliases, ok := providerAliases[normalized]; ok {
		return slices.Clone(aliases)
	}
	return []string{normalized}
}

func lookupIDs(id string) []string {
	if i := strings.Index(id, "/"); i >= 0 {
		return []string{id, id[i+1:]}
	}
	return []string{id}
}

func mapModelsDev(provider, modelID string, model map[string]any) *PublicCatalogRecord {
	limit := record(model["limit"])
	cost := record(model["cost"])
	context := finite(limit["context"])
	if context == nil {
		context = finite(limit["input"])
	}
	output := finite(limit["output"])
	result := &PublicCatalogRecord{Source: "models.dev", Provider: provider, ModelID: modelID}
	if context != nil || output != nil {
		result.Limits = &Limits{Context: context, Output: output}
	}
	c := Cost{Input: price(cost["input"]), Output: price(cost["output"]), CacheRead: price(cost["cache_read"]), CacheWrite: price(cost["cache_write"])}
	if c != (Cost{}) {
		result.Cost = &c
	}
	if input, ok := record(model["modalities"])["input"].([]any); ok {
		result.Modalities = []string{"text"}
		for _, item := range input {
			if item == "image" {
				result.Modalities = []string{"text", "image"}
				break
			}
		}
	}
	options, isArray := model["reasoning_options"].([]any)
	if !isArray {
		options = []any{model["reasoning_options"]}
	}
	for _, option := range options {
		o := record(option)
		if o["type"] != "effort" {
			continue
		}
		values, _ := o["values"].([]any)
		for _, value := range values {
			if s, ok := value.(string); ok {
				result.EffortLevels = append(result.EffortLevels, s)
			}
		}
	}
	return result
}

func ptr(n int) *float64 {
	f := float64(n)
	return &f
}

func toPiRecord(source, provider string, model ai.GeneratedModel) *PublicCatalogRecord {
	modalities := []string{"text"}
	if slices.Contains(model.Capabilities, "image") {
		modalities = []string{"text", "image"}
	}
	result := &PublicCatalogRecord{
		Source:     source,
		Provider:   provider,
		ModelID:    model.ID,
		Limits:     &Limits{Context: ptr(model.ContextWindow), Output: ptr(model.MaxOutputTokens)},
		Cost:       &Cost{Input: &model.InputCostPerMTokens, Output: &model.OutputCostPerMTokens, CacheRead: &model.CacheReadCost, CacheWrite: &model.CacheWriteCost},
		Modalities: modalities,
	}
	if model.ThinkingLevelMap != nil {
		result.ThinkingLevelMap = map[string]*string{}
		for level, mapped := range model.ThinkingLevelMap {
			result.ThinkingLevelMap[string(level)] = mapped
		}
	}
	return result
}

func findPiModel(provider string, ids []string) *ai.GeneratedModel {
	if !knownProviders[provider] {
		return nil
	}
	models := piModels(provider)
	for i := range models {
		if slices.Contains(ids, models[i].ID) || slices.Contains(ids, provider+"/"+models[i].ID) {
			return &models[i]
		}
	}
	return nil
}

var (
	backgroundMu       sync.Mutex
	backgroundInFlight = map[*refreshCall]struct{}{}
)

// backgroundRefresh is `void refreshCatalog(...)`: the caller does not wait, and the stale cache
// keeps serving. The fetch outlives ctx's cancellation but is bounded by options.Timeout, and
// Drain waits for it.
func backgroundRefresh(ctx context.Context, key string, options LoadPublicCatalogOptions) {
	call := &refreshCall{done: make(chan struct{})}
	backgroundMu.Lock()
	backgroundInFlight[call] = struct{}{}
	backgroundMu.Unlock()
	ctx = context.WithoutCancel(ctx)
	go func() {
		defer func() {
			backgroundMu.Lock()
			delete(backgroundInFlight, call)
			backgroundMu.Unlock()
			close(call.done)
		}()
		refreshCatalog(ctx, key, options)
	}()
}

// Drain waits for in-flight background refreshes, returning ctx.Err() if ctx ends first.
// The extension calls it on session shutdown.
func Drain(ctx context.Context) error {
	backgroundMu.Lock()
	pending := make([]*refreshCall, 0, len(backgroundInFlight))
	for call := range backgroundInFlight {
		pending = append(pending, call)
	}
	backgroundMu.Unlock()
	for _, call := range pending {
		select {
		case <-call.done:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

type loaded struct{ catalog modelsDevCatalog }

// LoadPublicCatalog reads the models.dev cache and, unless offline, refreshes it. ctx bounds the fetches.
func LoadPublicCatalog(ctx context.Context, options LoadPublicCatalogOptions) PublicCatalog {
	key := options.CachePath
	if key == "" {
		key = options.URL
	}
	if key == "" {
		key = ModelsDevURL
	}
	cacheMu.Lock()
	cache := memoryCaches[key]
	cacheMu.Unlock()
	if cache == nil && options.CachePath != "" {
		if cache = readCache(options.CachePath); cache != nil {
			cacheMu.Lock()
			memoryCaches[key] = cache
			cacheMu.Unlock()
		}
	}
	offline := os.Getenv("LITELLM_OFFLINE") == "1"
	if options.Offline != nil {
		offline = *options.Offline
	}
	var catalog modelsDevCatalog
	if cache != nil {
		catalog = cache.catalog
	}
	if !offline {
		if cache == nil {
			catalog = refreshCatalog(ctx, key, options)
		} else if nowMillis()-cache.fetchedAt >= float64(modelsDevCacheTTL.Milliseconds()) {
			backgroundRefresh(ctx, key, options)
		}
	}
	return loaded{catalog}
}

func (l loaded) Lookup(provider, id string) *PublicCatalogRecord {
	providers := providerCandidates(provider)
	ids := lookupIDs(id)
	vendor := ""
	for _, candidate := range providers {
		if candidate != "azure" {
			vendor = candidate
			break
		}
	}
	if vendor == "" && len(providers) > 0 {
		vendor = providers[0]
	}
	adapter := strings.ToLower(strings.TrimSpace(provider))
	for _, candidate := range providers {
		models := l.catalog[candidate]
		for _, modelID := range ids {
			model, ok := models[modelID]
			if !ok {
				continue
			}
			result := mapModelsDev(candidate, modelID, model)
			if candidate == providers[0] || vendor == "" {
				return result
			}
			fallback := vendor
			if adapter != "" {
				for _, piProvider := range piAliases(adapter) {
					if findPiModel(piProvider, ids) != nil {
						fallback = adapter
						break
					}
				}
			}
			result.PiProvider = piAliases(fallback)[0]
			return result
		}
	}
	// Provider-specific limits and pricing must not be replaced by the generic
	// vendor's entry just because both catalogs recognize the model id.
	if adapter != "" && adapter != vendor {
		for _, piProvider := range piAliases(adapter) {
			if model := findPiModel(piProvider, ids); model != nil {
				return toPiRecord("pi-adapter", piProvider, *model)
			}
		}
	}
	if vendor != "" {
		for _, piProvider := range piAliases(vendor) {
			if model := findPiModel(piProvider, ids); model != nil {
				return toPiRecord("pi-vendor", piProvider, *model)
			}
		}
	}
	return nil
}
