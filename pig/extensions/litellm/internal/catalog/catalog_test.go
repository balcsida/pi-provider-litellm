// Ports tests/public-catalog.test.ts
package catalog

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/MichaelKinsy/PiG/ai"
)

var modelsDevFixture = map[string]any{
	"amazon-bedrock": map[string]any{"models": map[string]any{
		"anthropic.claude-sonnet-4-6-v1:0": map[string]any{
			"modalities":        map[string]any{"input": []any{"text", "image"}},
			"limit":             map[string]any{"context": 200000, "output": 64000},
			"cost":              map[string]any{"input": 3, "output": 15},
			"reasoning_options": map[string]any{"type": "effort", "values": []any{"low", "medium", "high"}},
		},
	}},
	"fireworks-ai": map[string]any{"models": map[string]any{
		"accounts/fireworks/models/kimi-k3": map[string]any{"limit": map[string]any{"context": 262144, "output": 32768}},
	}},
}

// serve returns a client+URL pair for an httptest server answering body and a request counter.
func serve(t *testing.T, body any) (LoadPublicCatalogOptions, *atomic.Int32) {
	t.Helper()
	var hits atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(body)
	}))
	t.Cleanup(server.Close)
	return LoadPublicCatalogOptions{Client: server.Client(), URL: server.URL, CachePath: filepath.Join(t.TempDir(), "models-dev.json")}, &hits
}

func loadFresh(t *testing.T, body any) PublicCatalog {
	t.Helper()
	options, _ := serve(t, body)
	return LoadPublicCatalog(context.Background(), options)
}

func f(n float64) *float64 { return &n }

func mustLookup(t *testing.T, c PublicCatalog, provider, id string) *PublicCatalogRecord {
	t.Helper()
	r := c.Lookup(provider, id)
	if r == nil {
		t.Fatalf("Lookup(%q, %q) = nil", provider, id)
	}
	return r
}

func TestLoadPublicCatalog_ChatGPTCodexCatalog(t *testing.T) {
	for _, id := range []string{"gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"} {
		t.Run(id, func(t *testing.T) {
			r := mustLookup(t, loadFresh(t, map[string]any{}), "chatgpt", id)
			if r.Provider != "openai-codex" || r.ModelID != id || *r.Limits.Context != 272000 || *r.Limits.Output != 128000 {
				t.Fatalf("record = %+v limits %+v", r, r.Limits)
			}
		})
	}
}

func TestLoadPublicCatalog_PrefersAzureSpecificPiCatalog(t *testing.T) {
	for _, provider := range []string{"azure", "azure_ai"} {
		t.Run(provider, func(t *testing.T) {
			r := mustLookup(t, loadFresh(t, map[string]any{}), provider, "gpt-5.6-sol")
			if r.Source != "pi-adapter" || r.Provider != PiAzureProvider || *r.Limits.Context != 1050000 || *r.Limits.Output != 128000 {
				t.Fatalf("record = %+v limits %+v", r, r.Limits)
			}
		})
	}
}

func TestLoadPublicCatalog_AzureFallbackForSparseOpenAIRecord(t *testing.T) {
	body := map[string]any{"openai": map[string]any{"models": map[string]any{
		"gpt-5.6-sol": map[string]any{"reasoning_options": []any{map[string]any{"type": "effort", "values": []any{"low", "high"}}}},
	}}}
	for _, provider := range []string{"azure", "azure_ai"} {
		t.Run(provider, func(t *testing.T) {
			r := mustLookup(t, loadFresh(t, body), provider, "gpt-5.6-sol")
			if r.Source != "models.dev" || r.Provider != "openai" || r.PiProvider != "azure" || !reflect.DeepEqual(r.EffortLevels, []string{"low", "high"}) {
				t.Fatalf("record = %+v", r)
			}
		})
	}
}

// withoutAzureCatalog makes the Pi catalog answer no models for the Azure provider once, like mockReturnValueOnce([]).
func withoutAzureCatalog(t *testing.T) {
	t.Helper()
	original := piModels
	piModels = func(provider string) []ai.GeneratedModel {
		if provider == PiAzureProvider {
			return nil
		}
		return original(provider)
	}
	t.Cleanup(func() { piModels = original })
}

func TestLoadPublicCatalog_FallsBackToOpenAIWhenAzureDoesNotKnowModel(t *testing.T) {
	withoutAzureCatalog(t)
	r := mustLookup(t, loadFresh(t, map[string]any{}), "azure", "gpt-4")
	if r.Source != "pi-vendor" || r.Provider != "openai" || *r.Limits.Context != 8192 {
		t.Fatalf("record = %+v limits %+v", r, r.Limits)
	}
}

func TestLoadPublicCatalog_VendorFallbackForSparseEnrichment(t *testing.T) {
	withoutAzureCatalog(t)
	body := map[string]any{"openai": map[string]any{"models": map[string]any{"gpt-4": map[string]any{"cost": map[string]any{"input": 30}}}}}
	r := mustLookup(t, loadFresh(t, body), "azure", "gpt-4")
	if r.Source != "models.dev" || r.PiProvider != "openai" {
		t.Fatalf("record = %+v", r)
	}
}

func TestLoadPublicCatalog_ChatGPTUnderOpenAIOnModelsDev(t *testing.T) {
	levels := []any{"none", "low", "medium", "high", "xhigh", "max"}
	body := map[string]any{"openai": map[string]any{"models": map[string]any{
		"gpt-5.6-sol": map[string]any{"reasoning_options": []any{map[string]any{"type": "effort", "values": levels}}},
	}}}
	r := mustLookup(t, loadFresh(t, body), "chatgpt", "gpt-5.6-sol")
	if r.Source != "models.dev" || r.Provider != "openai" || r.PiProvider != "openai-codex" ||
		!reflect.DeepEqual(r.EffortLevels, []string{"none", "low", "medium", "high", "xhigh", "max"}) {
		t.Fatalf("record = %+v", r)
	}
}

func TestLoadPublicCatalog_CarriesThinkingLevelMap(t *testing.T) {
	r := mustLookup(t, loadFresh(t, map[string]any{}), "chatgpt", "gpt-5.6-sol")
	got := map[string]string{}
	for level, mapped := range r.ThinkingLevelMap {
		if mapped == nil {
			t.Fatalf("level %s is null", level)
		}
		got[level] = *mapped
	}
	if want := map[string]string{"xhigh": "xhigh", "max": "max", "minimal": "low"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("map = %v, want %v", got, want)
	}
	if r.EffortLevels != nil {
		t.Fatalf("effortLevels = %v", r.EffortLevels)
	}
}

func TestLoadPublicCatalog_BedrockClaudeEvidence(t *testing.T) {
	got := mustLookup(t, loadFresh(t, modelsDevFixture), "bedrock", "anthropic.claude-sonnet-4-6-v1:0")
	want := &PublicCatalogRecord{
		Source: "models.dev", Provider: "amazon-bedrock", ModelID: "anthropic.claude-sonnet-4-6-v1:0",
		Limits:       &Limits{Context: f(200000), Output: f(64000)},
		Cost:         &Cost{Input: f(3), Output: f(15)},
		Modalities:   []string{"text", "image"},
		EffortLevels: []string{"low", "medium", "high"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %+v\nwant %+v", got, want)
	}
}

func TestLoadPublicCatalog_FireworksBaseModelHint(t *testing.T) {
	r := mustLookup(t, loadFresh(t, modelsDevFixture), "fireworks", "accounts/fireworks/models/kimi-k3")
	if r.Source != "models.dev" || r.Provider != "fireworks-ai" || *r.Limits.Context != 262144 || *r.Limits.Output != 32768 {
		t.Fatalf("record = %+v", r)
	}
}

func TestLoadPublicCatalog_WithholdsModalitiesForNonArrayInput(t *testing.T) {
	body := map[string]any{"moonshotai": map[string]any{"models": map[string]any{
		"kimi-k3-preview": map[string]any{"modalities": map[string]any{"input": 7}, "limit": map[string]any{"context": 262144, "output": 32768}},
	}}}
	r := mustLookup(t, loadFresh(t, body), "moonshot", "kimi-k3-preview")
	if r.Source != "models.dev" || *r.Limits.Context != 262144 || r.Modalities != nil {
		t.Fatalf("record = %+v", r)
	}
}

func TestLoadPublicCatalog_OmitsNegativePricesPreservingZero(t *testing.T) {
	body := map[string]any{"private": map[string]any{"models": map[string]any{
		"priced": map[string]any{"cost": map[string]any{"input": -1, "output": 2, "cache_read": 0, "cache_write": 0}},
	}}}
	r := mustLookup(t, loadFresh(t, body), "private", "priced")
	if want := (&Cost{Output: f(2), CacheRead: f(0), CacheWrite: f(0)}); !reflect.DeepEqual(r.Cost, want) {
		t.Fatalf("cost = %+v, want %+v", r.Cost, want)
	}
}

func TestLoadPublicCatalog_UnknownModelNoOpinion(t *testing.T) {
	if r := loadFresh(t, map[string]any{}).Lookup("unknown", "not-real"); r != nil {
		t.Fatalf("record = %+v", r)
	}
}

func TestLoadPublicCatalog_ProcessUniqueTemporaryCachePath(t *testing.T) {
	options, _ := serve(t, modelsDevFixture)
	LoadPublicCatalog(context.Background(), options)
	if _, err := os.Stat(options.CachePath); err != nil {
		t.Fatalf("cache not written: %v", err)
	}
	entries, _ := os.ReadDir(filepath.Dir(options.CachePath))
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".tmp") {
			t.Fatalf("temporary file left behind: %s", e.Name())
		}
	}
	if !regexpTmp(options.CachePath) {
		t.Fatal("temporary path shape")
	}
}

// regexpTmp checks the `<path>.<pid>.<ms>.tmp` shape writeJSONAtomic uses by writing to an unwritable rename target.
func regexpTmp(path string) bool {
	dir := filepath.Join(filepath.Dir(path), "target-dir")
	_ = os.Mkdir(dir, 0o755)
	_ = os.WriteFile(filepath.Join(dir, "x"), nil, 0o644)
	// renaming a file over a non-empty directory fails, so the temp file must be cleaned and the error returned.
	err := writeJSONAtomic(dir, &cacheFile{fetchedAt: 1, catalog: modelsDevCatalog{}})
	entries, _ := os.ReadDir(filepath.Dir(path))
	for _, e := range entries {
		if strings.HasSuffix(e.Name(), ".tmp") {
			return false
		}
	}
	return err != nil
}

func TestLoadPublicCatalog_StaleDiskCacheOffline(t *testing.T) {
	options, hits := serve(t, modelsDevFixture)
	data, _ := json.Marshal(map[string]any{"fetchedAt": 1, "catalog": modelsDevFixture})
	if err := os.WriteFile(options.CachePath, data, 0o644); err != nil {
		t.Fatal(err)
	}
	offline := true
	options.Offline = &offline
	c := LoadPublicCatalog(context.Background(), options)
	mustLookup(t, c, "bedrock", "anthropic.claude-sonnet-4-6-v1:0")
	time.Sleep(50 * time.Millisecond)
	if hits.Load() != 0 {
		t.Fatalf("fetched %d times while offline", hits.Load())
	}
}

func TestLoadPublicCatalog_RefreshesFutureDatedDiskCache(t *testing.T) {
	options, hits := serve(t, modelsDevFixture)
	data, _ := json.Marshal(map[string]any{"fetchedAt": float64(time.Now().UnixMilli() + 60000), "catalog": map[string]any{}})
	if err := os.WriteFile(options.CachePath, data, 0o644); err != nil {
		t.Fatal(err)
	}
	c := LoadPublicCatalog(context.Background(), options)
	if hits.Load() != 1 {
		t.Fatalf("fetches = %d", hits.Load())
	}
	mustLookup(t, c, "bedrock", "anthropic.claude-sonnet-4-6-v1:0")
}
