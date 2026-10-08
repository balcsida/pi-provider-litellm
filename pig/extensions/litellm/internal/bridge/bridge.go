// Package bridge adapts PiG's typed ai.ModelsProvider to the wire-shaped sdk.Provider that
// sdk.Extension.RegisterNativeProvider takes. PiG has no equivalent adapter.
//
// Request credentials: ai.StreamSimple and the built-in drivers read the API key and extra
// headers from ai.StreamOptions (APIKey, Headers), never from Provider.Auth. Auth is resolved by
// the host (ai.ResolveProviderAuth) and arrives here already folded into the wire options
// (options.apiKey, options.headers), so Wrap forwards those values untouched.
package bridge

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/MichaelKinsy/PiG/extensions/sdk"
)

// Wrap exposes p through the SDK's native-provider callbacks. GenerateImages, Classify,
// FetchDeferred, CancelDeferred and GetAllModels (image and classifier models) are not mapped.
func Wrap(p *ai.ModelsProvider) *sdk.Provider {
	out := &sdk.Provider{
		ID:      p.ID,
		Name:    p.Name,
		Headers: stringHeaders(p.Headers),
		Auth:    wrapAuth(p.Auth),
		GetModels: func() ([]map[string]any, error) {
			models, err := p.GetModels()
			if err != nil {
				return nil, err
			}
			return encodeModels(models)
		},
		Stream:       wrapStream(p.Stream),
		StreamSimple: wrapStream(p.StreamSimple),
	}
	if p.BaseURL != "" {
		out.BaseURL = &p.BaseURL
	}
	if p.FilterModels != nil {
		out.FilterModels = func(wire []map[string]any, credential map[string]any) ([]map[string]any, error) {
			return filterModels(p, wire, credential)
		}
	}
	if p.RefreshModels != nil {
		out.RefreshModels = func(in sdk.RefreshModelsContext) error { return refreshModels(p, in) }
	}
	return out
}

// stringHeaders drops nil entries: sdk.Provider.Headers has no way to say "delete this header".
func stringHeaders(headers ai.ProviderHeaders) map[string]string {
	if headers == nil {
		return nil
	}
	out := make(map[string]string, len(headers))
	for name, value := range headers {
		if value != nil {
			out[name] = *value
		}
	}
	return out
}

// toMap and fromMap convert between wire maps and typed values through JSON.
func toMap(value any) (map[string]any, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	var out map[string]any
	err = json.Unmarshal(data, &out)
	return out, err
}

func fromMap(in map[string]any, target any) error {
	data, err := json.Marshal(in)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, target)
}

func decodeCredential(in map[string]any) (*ai.Credential, error) {
	if in == nil {
		return nil, nil
	}
	var credential ai.Credential
	if err := fromMap(in, &credential); err != nil {
		return nil, fmt.Errorf("decode credential: %w", err)
	}
	return &credential, nil
}

// modelsCatalogRecord mirrors the unexported record of ai/models_catalog_codec.go:15, which is
// the only chat-model JSON shape PiG defines (only DecodeModelsCatalog is exported).
type modelsCatalogRecord struct {
	Type                          ai.ModelType                     `json:"type,omitempty"`
	ID                            string                           `json:"id"`
	Name                          string                           `json:"name"`
	API                           ai.API                           `json:"api"`
	Provider                      string                           `json:"provider"`
	BaseURL                       string                           `json:"baseUrl"`
	Reasoning                     bool                             `json:"reasoning"`
	ThinkingLevelMap              ai.ThinkingLevelMap              `json:"thinkingLevelMap,omitempty"`
	Input                         []string                         `json:"input"`
	InputLimits                   *ai.ModelInputLimits             `json:"inputLimits,omitempty"`
	Cost                          ai.ModelCost                     `json:"cost"`
	PromptCache                   ai.ModelPromptCache              `json:"promptCache,omitempty"`
	ContextWindow                 int                              `json:"contextWindow"`
	MaxTokens                     int                              `json:"maxTokens"`
	SamplingParams                map[string]any                   `json:"samplingParams,omitempty"`
	SamplingParamsByThinkingLevel ai.SamplingParamsByThinkingLevel `json:"samplingParamsByThinkingLevel,omitempty"`
	Headers                       map[string]string                `json:"headers,omitempty"`
	Compat                        *ai.ModelCompat                  `json:"compat,omitempty"`
}

func encodeModel(m *ai.Model) (map[string]any, error) {
	caps := m.Capabilities
	return toMap(modelsCatalogRecord{
		Type: m.Type, ID: m.ID, Name: m.DisplayName, API: m.ProviderMeta.API, Provider: m.ProviderMeta.ProviderID,
		BaseURL: m.ProviderMeta.BaseURL, Reasoning: m.ProviderMeta.Reasoning, ThinkingLevelMap: m.ThinkingLevelMap,
		Input: m.Input, InputLimits: m.InputLimits,
		Cost: ai.ModelCost{Input: caps.InputCostPer1M, Output: caps.OutputCostPer1M, CacheRead: caps.CacheReadCostPer1M,
			CacheWrite: caps.CacheWriteCostPer1M, Tiers: caps.CostTiers},
		PromptCache: m.PromptCache, ContextWindow: caps.ContextWindow, MaxTokens: caps.MaxOutputTokens,
		SamplingParams: m.SamplingParams, SamplingParamsByThinkingLevel: m.SamplingParamsByThinkingLevel,
		Headers: m.ProviderMeta.Headers, Compat: m.ProviderMeta.Compat,
	})
}

func encodeModels(models []*ai.Model) ([]map[string]any, error) {
	out := make([]map[string]any, 0, len(models))
	for _, m := range models {
		wire, err := encodeModel(m)
		if err != nil {
			return nil, err
		}
		out = append(out, wire)
	}
	return out, nil
}

// decodeModel restores one chat model. ai.DecodeModelsCatalog drops records whose provider differs
// from providerID and every field the catalog record lacks (see TestFilterModelsDropsUnknownFields).
func decodeModel(wire map[string]any, providerID string) (*ai.Model, error) {
	data, err := json.Marshal(wire)
	if err != nil {
		return nil, err
	}
	models, err := ai.DecodeModelsCatalog([]json.RawMessage{data}, providerID)
	if err != nil {
		return nil, err
	}
	if len(models) != 1 {
		return nil, fmt.Errorf("model %v is not a chat model of provider %q", wire["id"], providerID)
	}
	chat, ok := models[0].(*ai.Model)
	if !ok {
		return nil, fmt.Errorf("model %v is not a chat model", wire["id"])
	}
	return chat, nil
}

// streamModel decodes the model of a stream request under its own provider id, which is what the
// host sends (extension.ModelInfo) and may be an alias of p.ID.
func streamModel(wire map[string]any) (*ai.Model, error) {
	providerID, _ := wire["provider"].(string)
	if providerID == "" {
		return nil, errors.New("model has no provider")
	}
	return decodeModel(wire, providerID)
}

// filterModels keeps the caller's own map for every model p keeps, so the host's pointer-identity
// index match survives and fields the catalog record lacks stay on the map. A model p replaced
// with a new value is encoded fresh and reported with index -1.
func filterModels(p *ai.ModelsProvider, wire []map[string]any, credentialWire map[string]any) ([]map[string]any, error) {
	credential, err := decodeCredential(credentialWire)
	if err != nil {
		return nil, err
	}
	decoded := make([]*ai.Model, len(wire))
	byPointer := make(map[*ai.Model]map[string]any, len(wire))
	for i, w := range wire {
		if decoded[i], err = decodeModel(w, p.ID); err != nil {
			return nil, err
		}
		byPointer[decoded[i]] = w
	}
	kept := p.FilterModels(decoded, credential)
	out := make([]map[string]any, 0, len(kept))
	for _, m := range kept {
		encoded, err := encodeModel(m)
		if err != nil {
			return nil, err
		}
		if original, ok := byPointer[m]; ok {
			for key, value := range encoded {
				original[key] = value
			}
			encoded = original
		}
		out = append(out, encoded)
	}
	return out, nil
}

func refreshModels(p *ai.ModelsProvider, in sdk.RefreshModelsContext) error {
	credential, err := decodeCredential(in.Credential)
	if err != nil {
		return err
	}
	var stored *ai.ModelsStoreEntry
	if in.Stored != nil {
		stored = new(ai.ModelsStoreEntry)
		if err := fromMap(in.Stored, stored); err != nil {
			return fmt.Errorf("decode stored models: %w", err)
		}
	}
	signal := in.Signal
	if signal == nil {
		signal = context.Background()
	}
	return p.RefreshModels(ai.RefreshModelsContext{
		Credential: credential, Stored: stored, AllowNetwork: in.AllowNetwork, Force: in.Force, Signal: signal,
		Publish: func(publication ai.ModelsPublication) (bool, error) {
			wire := sdk.ModelsPublication{}
			if publication.Persist != nil {
				data, err := json.Marshal(publication.Persist)
				if err != nil {
					return false, err
				}
				wire.Persist = data
			} else if publication.PersistSet {
				wire.Persist = json.RawMessage("null")
			}
			if publication.Update != nil {
				wire.Update = func() error { publication.Update(); return nil }
			}
			return in.Publish(wire)
		},
	})
}
