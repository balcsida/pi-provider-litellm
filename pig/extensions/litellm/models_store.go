package litellm

// Ports src/models-store.ts: the stored-catalog seed. The seed reads PiG's models-store.json and hands those models
// over instead of discovering them again, when every record is one PiG can restore and the store is fresh, for this
// proxy root, and from the current discovery version. It only reads; PiG owns the file. See src/models-store.ts for
// why the models are returned and why the credential is not checked.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"time"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/discover"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/protocols"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

const modelsStoreFilename = "models-store.json"

// storeFreshness matches PiG's RemoteCatalogRefreshInterval (internal/codingagent/remote_catalog_provider.go), which
// extensions cannot import.
const storeFreshness = 4 * time.Hour

// readStoredModels is PiG's stored models for provider, when every one of them can be used as it is, and nil
// otherwise. nil means "discover"; it never fails.
func readStoredModels(agentDir, provider, baseURL string, allowInsecureHTTP bool, policies *policyStore) []types.DiscoveredModel {
	raw, err := os.ReadFile(filepath.Join(agentDir, modelsStoreFilename))
	if err != nil {
		return nil
	}
	var store map[string]ai.ModelsStoreEntry
	if json.Unmarshal(raw, &store) != nil {
		return nil
	}
	entry, ok := store[provider]
	if !ok || len(entry.Models) == 0 || entry.CheckedAt == nil {
		return nil
	}
	if time.Since(time.UnixMilli(int64(*entry.CheckedAt))) >= storeFreshness {
		return nil
	}
	models := make([]types.DiscoveredModel, 0, len(entry.Models))
	for _, record := range entry.Models {
		var model types.DiscoveredModel
		// An undecodable record is the restore failure: PiG skips it, so the seed has to be the catalog.
		if json.Unmarshal(record, &model) != nil || model.Provider != provider {
			return nil
		}
		policy, known := policies.entry(provider, model.ID)
		if !known || policy.Version != types.DiscoveryVersion || !protocols.IsLiteLLMAPI(string(model.API)) {
			return nil
		}
		want, err := protocols.ResolveModelBaseURL(baseURL, types.LiteLLMApi(model.API), allowInsecureHTTP)
		if err != nil || model.BaseURL != want {
			return nil
		}
		models = append(models, discover.EnrichCachedModel(policy.apply(model)))
	}
	return models
}
