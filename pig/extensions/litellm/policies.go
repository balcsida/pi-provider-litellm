package litellm

// Ports src/types.ts model fields that PiG's closed model struct drops (litellmPolicy, litellmBackendFamily,
// litellmResponsesReasoningControl, suppressReasoningContent, litellmDiscoveryVersion). Discovery results
// are kept here, keyed by provider and model id, and persisted beside PiG's model store.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"

	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

const policiesFilename = "litellm-model-policies.json"

// modelPolicy is the extension-owned part of a discovered model.
type modelPolicy struct {
	// Version is the discovery version that produced the entry; an older one marks a legacy model.
	Version                   int                       `json:"version"`
	Policy                    *types.LiteLLMModelPolicy `json:"policy,omitempty"`
	Family                    types.BackendFamily       `json:"family,omitempty"`
	ResponsesReasoningControl bool                      `json:"responsesReasoningControl,omitempty"`
	SuppressReasoningContent  *bool                     `json:"suppressReasoningContent,omitempty"`
}

type policyFile struct {
	Models map[string]modelPolicy `json:"models"`
}

// policyStore is safe for concurrent use. An empty path keeps it in memory only.
type policyStore struct {
	path   string
	mu     sync.Mutex
	models map[string]modelPolicy
}

func policyKey(provider, modelID string) string { return provider + "\x00" + modelID }

// newPolicyStore loads path; a missing or unreadable file starts empty.
func newPolicyStore(path string) *policyStore {
	store := &policyStore{path: path, models: map[string]modelPolicy{}}
	if path == "" {
		return store
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return store
	}
	var file policyFile
	if json.Unmarshal(raw, &file) == nil && file.Models != nil {
		store.models = file.Models
	}
	return store
}

func policyOf(model types.DiscoveredModel) modelPolicy {
	return modelPolicy{
		Version:                   model.LiteLLMDiscoveryVersion,
		Policy:                    model.LiteLLMPolicy,
		Family:                    model.LiteLLMBackendFamily,
		ResponsesReasoningControl: model.LiteLLMResponsesReasoningControl,
		SuppressReasoningContent:  model.SuppressReasoningContent,
	}
}

// apply copies the entry onto a model decoded from PiG's store, which carries none of these fields.
func (entry modelPolicy) apply(model types.DiscoveredModel) types.DiscoveredModel {
	model.LiteLLMDiscoveryVersion = entry.Version
	model.LiteLLMPolicy = entry.Policy
	model.LiteLLMBackendFamily = entry.Family
	model.LiteLLMResponsesReasoningControl = entry.ResponsesReasoningControl
	model.SuppressReasoningContent = entry.SuppressReasoningContent
	return model
}

// policyFor is the lookup request and response hooks use.
func (s *policyStore) policyFor(provider, modelID string) (*types.LiteLLMModelPolicy, types.BackendFamily, bool) {
	entry, ok := s.entry(provider, modelID)
	return entry.Policy, entry.Family, ok
}

func (s *policyStore) entry(provider, modelID string) (modelPolicy, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	entry, ok := s.models[policyKey(provider, modelID)]
	return entry, ok
}

// replace swaps every entry of provider for the discovered models, then persists.
func (s *policyStore) replace(provider string, models []types.DiscoveredModel) error {
	s.mu.Lock()
	prefix := policyKey(provider, "")
	for key := range s.models {
		if len(key) >= len(prefix) && key[:len(prefix)] == prefix {
			delete(s.models, key)
		}
	}
	for _, model := range models {
		entry := policyOf(model)
		if entry.Version == 0 {
			entry.Version = types.DiscoveryVersion
		}
		s.models[policyKey(provider, model.ID)] = entry
	}
	s.mu.Unlock()
	return s.flush()
}

// remember records the policy a legacy model was restored with, without persisting it and without
// touching the entry's version, so the model stays legacy until a network refresh replaces it.
func (s *policyStore) remember(provider string, model types.DiscoveredModel) {
	s.mu.Lock()
	defer s.mu.Unlock()
	key := policyKey(provider, model.ID)
	entry := policyOf(model)
	if existing, ok := s.models[key]; ok {
		entry.Version = existing.Version
	} else {
		entry.Version = 0
	}
	s.models[key] = entry
}

// flush writes the store atomically (temp file and rename, mode 0600).
func (s *policyStore) flush() error {
	if s.path == "" {
		return nil
	}
	s.mu.Lock()
	data, err := json.Marshal(policyFile{Models: s.models})
	s.mu.Unlock()
	if err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Dir(s.path), 0o700); err != nil {
		return err
	}
	tmp, err := os.CreateTemp(filepath.Dir(s.path), policiesFilename+".*.tmp")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return err
	}
	if _, err := tmp.Write(data); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmp.Name(), s.path)
}
