package litellm

// Tests the policy sidecar (new in the PiG port; PiG's closed model struct drops these fields).

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

func TestPolicyStore(t *testing.T) {
	policy := &types.LiteLLMModelPolicy{DropToolReasoning: true, ExplicitToolReasoningOff: true}
	suppress := true
	model := discoveredModel("gpt-6")
	model.LiteLLMPolicy, model.LiteLLMBackendFamily = policy, types.FamilyOpenAI
	model.LiteLLMResponsesReasoningControl, model.SuppressReasoningContent = true, &suppress

	t.Run("round trips through an atomically written 0600 file", func(t *testing.T) {
		path := filepath.Join(t.TempDir(), "agent", policiesFilename)
		store := newPolicyStore(path)
		if err := store.replace("litellm", []types.DiscoveredModel{model}); err != nil {
			t.Fatal(err)
		}
		info, err := os.Stat(path)
		if err != nil || info.Mode().Perm() != 0o600 {
			t.Fatalf("stat = %v, %v", info, err)
		}
		entries, _ := os.ReadDir(filepath.Dir(path))
		if len(entries) != 1 {
			t.Fatalf("temp files left behind: %v", entries)
		}
		got, family, ok := newPolicyStore(path).policyFor("litellm", "gpt-6")
		if !ok || family != types.FamilyOpenAI || got == nil || !got.DropToolReasoning || !got.ExplicitToolReasoningOff {
			t.Fatalf("policyFor = %+v %q %v", got, family, ok)
		}
		entry, _ := newPolicyStore(path).entry("litellm", "gpt-6")
		restored := entry.apply(discoveredModel("gpt-6"))
		if restored.LiteLLMDiscoveryVersion != types.DiscoveryVersion || !restored.LiteLLMResponsesReasoningControl ||
			restored.SuppressReasoningContent == nil || !*restored.SuppressReasoningContent {
			t.Fatalf("restored = %+v", restored)
		}
	})
	t.Run("keys by provider and replaces only that provider's entries", func(t *testing.T) {
		store := newPolicyStore("")
		store.replace("litellm", []types.DiscoveredModel{model})
		store.replace("alias", []types.DiscoveredModel{discoveredModel("other")})
		store.replace("alias", nil)
		if _, _, ok := store.policyFor("litellm", "gpt-6"); !ok {
			t.Fatal("replacing alias dropped litellm")
		}
		if _, _, ok := store.policyFor("alias", "other"); ok {
			t.Fatal("alias entry survived replace")
		}
		if _, _, ok := store.policyFor("alias", "gpt-6"); ok {
			t.Fatal("entries leaked across providers")
		}
	})
	t.Run("remember keeps the stored version so the model stays legacy", func(t *testing.T) {
		store := newPolicyStore("")
		legacy := model
		legacy.LiteLLMDiscoveryVersion = 0
		store.remember("litellm", legacy)
		if entry, ok := store.entry("litellm", "gpt-6"); !ok || entry.Version == types.DiscoveryVersion {
			t.Fatalf("entry = %+v", entry)
		}
		store.replace("litellm", []types.DiscoveredModel{model})
		store.remember("litellm", legacy)
		if entry, _ := store.entry("litellm", "gpt-6"); entry.Version != types.DiscoveryVersion {
			t.Fatalf("remember lowered the version: %+v", entry)
		}
	})
	t.Run("a missing or corrupt file starts empty", func(t *testing.T) {
		dir := t.TempDir()
		if _, _, ok := newPolicyStore(filepath.Join(dir, "missing.json")).policyFor("litellm", "x"); ok {
			t.Fatal("missing file produced an entry")
		}
		path := filepath.Join(dir, "corrupt.json")
		os.WriteFile(path, []byte("{not json"), 0o600)
		if _, _, ok := newPolicyStore(path).policyFor("litellm", "x"); ok {
			t.Fatal("corrupt file produced an entry")
		}
	})
}

func TestPolicyStoreFlushSerialization(t *testing.T) {
	path := filepath.Join(t.TempDir(), policiesFilename)
	store := newPolicyStore(path)
	const providers = 24
	var wg sync.WaitGroup
	for i := range providers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if err := store.replaceSeed(fmt.Sprintf("p%d", i), []types.DiscoveredModel{discoveredModel("m")}); err != nil {
				t.Error(err)
			}
		}()
	}
	wg.Wait()
	reloaded := newPolicyStore(path)
	for i := range providers {
		if _, ok := reloaded.entry(fmt.Sprintf("p%d", i), "m"); !ok {
			t.Fatalf("provider p%d is missing from the persisted snapshot: an older snapshot was written last", i)
		}
	}
}

func TestPolicyStoreStoredVersion(t *testing.T) {
	store := newPolicyStore("")
	stale := discoveredModel("m")
	stale.LiteLLMDiscoveryVersion = 2
	store.replace("litellm", []types.DiscoveredModel{stale})
	if v, known := store.storedVersion("litellm", "m"); v != 2 || !known {
		t.Fatalf("unseeded = %d %v", v, known)
	}
	store.replaceSeed("litellm", []types.DiscoveredModel{discoveredModel("m"), discoveredModel("new")})
	if v, known := store.storedVersion("litellm", "m"); v != 2 || !known {
		t.Fatalf("seeded keeps the pre-seed version, got %d %v", v, known)
	}
	if _, known := store.storedVersion("litellm", "new"); known {
		t.Fatal("a route first seen by the seed was unknown before it")
	}
	store.replace("litellm", []types.DiscoveredModel{discoveredModel("m")})
	if v, _ := store.storedVersion("litellm", "m"); v != types.DiscoveryVersion {
		t.Fatalf("a network refresh must end the seed baseline, got %d", v)
	}
}
