package discover

// Ports the "discoverModels via /health", "discoverModels wildcard expansion via /v1/models",
// "discoverModels proxy version gate", "discoverModels response-mode models", "discoverModels fallback to
// /v1/models", "discoverModels fallback to /health", "native Messages discovery" and "Moonshot transport
// suppression" describe blocks of tests/discover.test.ts.

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/modelgroups"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

// fbOK serves a JSON literal with status 200.
func fbOK(body string) http.HandlerFunc { return jsonResponse(200, json.RawMessage(body)) }

// fbStatus serves an empty JSON object with the given status.
func fbStatus(status int) http.HandlerFunc { return jsonResponse(status, map[string]any{}) }

// fbData wraps rows in a `{"data":[...]}` envelope (also the /v1/models shape).
func fbData(rows ...string) http.HandlerFunc { return fbOK(string(modelInfoBody(rows...))) }

// fbHealth serves a /health body listing the given entries.
func fbHealth(entries ...string) http.HandlerFunc {
	return fbOK(`{"healthy_endpoints":[` + strings.Join(entries, ",") + `]}`)
}

// fbNormalize round-trips v through JSON so structs, maps and literals compare alike.
func fbNormalize(t testing.TB, v any) any {
	t.Helper()
	var encoded []byte
	switch value := v.(type) {
	case string:
		encoded = []byte(value)
	default:
		var err error
		if encoded, err = json.Marshal(v); err != nil {
			t.Fatal(err)
		}
	}
	var out any
	if err := json.Unmarshal(encoded, &out); err != nil {
		t.Fatalf("decode %s: %v", encoded, err)
	}
	return out
}

// fbMatches is `toMatchObject`: objects match when every wanted key matches, arrays and scalars exactly.
func fbMatches(got, want any) bool {
	switch w := want.(type) {
	case map[string]any:
		g, ok := got.(map[string]any)
		if !ok {
			return false
		}
		for key, wantValue := range w {
			gotValue, present := g[key]
			if !present || !fbMatches(gotValue, wantValue) {
				return false
			}
		}
		return true
	case []any:
		g, ok := got.([]any)
		if !ok || len(g) != len(w) {
			return false
		}
		for i := range w {
			if !fbMatches(g[i], w[i]) {
				return false
			}
		}
		return true
	case float64:
		g, ok := got.(float64)
		return ok && math.Abs(g-w) <= 1e-9*math.Max(1, math.Abs(w))
	}
	return reflect.DeepEqual(got, want)
}

// fbMatch asserts `expect(got).toMatchObject(want)` for a JSON literal want.
func fbMatch(t *testing.T, got any, want string) {
	t.Helper()
	g := fbNormalize(t, got)
	if !fbMatches(g, fbNormalize(t, want)) {
		encoded, _ := json.Marshal(g)
		t.Errorf("got %s\nwant to match %s", encoded, want)
	}
}

// fbEqual asserts `expect(got).toEqual(want)` for a JSON literal want.
func fbEqual(t *testing.T, got any, want string) {
	t.Helper()
	g, w := fbNormalize(t, got), fbNormalize(t, want)
	if !reflect.DeepEqual(g, w) {
		encoded, _ := json.Marshal(g)
		t.Errorf("got %s\nwant %s", encoded, want)
	}
}

// fbLacks asserts `expect(got).not.toHaveProperty(key)`.
func fbLacks(t *testing.T, got any, key string) {
	t.Helper()
	if object, _ := fbNormalize(t, got).(map[string]any); object != nil {
		if value, present := object[key]; present {
			t.Errorf("unexpected property %s = %v", key, value)
		}
	}
}

func fbIDs(models []types.DiscoveredModel) []string {
	ids := make([]string, 0, len(models))
	for _, model := range models {
		ids = append(ids, model.ID)
	}
	return ids
}

// fbIDAPIs lists "id=api" pairs, the Go form of `models.map(({ id, api }) => ({ id, api }))`.
func fbIDAPIs(models []types.DiscoveredModel) []string {
	pairs := make([]string, 0, len(models))
	for _, model := range models {
		pairs = append(pairs, fmt.Sprintf("%s=%s", model.ID, model.API))
	}
	return pairs
}

func fbExpectStrings(t *testing.T, got []string, want ...string) {
	t.Helper()
	if want == nil {
		want = []string{}
	}
	if got == nil {
		got = []string{}
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got %q, want %q", got, want)
	}
}

func fbCount(requests []string, suffix string) int {
	count := 0
	for _, request := range requests {
		if strings.HasSuffix(request, suffix) {
			count++
		}
	}
	return count
}

// fbModelsDevCache writes a models.dev cache document and returns its path.
func fbModelsDevCache(t *testing.T, catalog string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "models-dev.json")
	if err := os.WriteFile(path, []byte(`{"fetchedAt":1,"catalog":`+catalog+`}`), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// fbDiscover runs DiscoverModels against routes with offline options.
func fbDiscover(t *testing.T, routes map[string]http.HandlerFunc) (*types.DiscoveryResult, *reports) {
	t.Helper()
	base := mockEndpoints(t, routes)
	options, stderr := testOptions(t)
	return discover(t, base, options), stderr
}

// fbDiscoverProxy is fbDiscover that also returns the proxy to inspect its requests.
func fbDiscoverProxy(t *testing.T, routes map[string]http.HandlerFunc) (*types.DiscoveryResult, *mockProxy, *reports) {
	t.Helper()
	proxy := newMockProxy(t, routes)
	options, stderr := testOptions(t)
	return discover(t, proxy.URL, options), proxy, stderr
}

// fbJoined is `stderr.mock.calls.flat().join(" ")`.
func fbJoined(stderr *reports) string { return strings.Join(stderr.Lines(), " ") }

// ---- discoverModels via /health ----

func TestFallbacks_HealthEnrichesDeploymentDetailsWithoutGrantingCatalogMetadataToHealthOnlyRoutes(t *testing.T) {
	cachePath := fbModelsDevCache(t, `{"private":{"models":{"priced-model":{"modalities":{"input":["text","image"]},`+
		`"limit":{"context":64000,"output":8000},"cost":{"input":7,"output":9,"cache_read":1,"cache_write":2}}}}}`)
	base := mockEndpoints(t, map[string]http.HandlerFunc{
		"/model/info": fbStatus(404),
		"/v1/models":  fbStatus(404),
		"/health": fbHealth(`{"model":"detailed-route","model_id":"detailed-deployment"}`,
			`{"model":"private/priced-model"}`),
		"/model/info?litellm_model_id=detailed-deployment": fbData(`{"model_name":"detailed-route",` +
			`"litellm_params":{"model":"private/priced-model"},` +
			`"model_info":{"mode":"chat","supports_reasoning":false,"supports_vision":true}}`),
	})
	options, _ := testOptions(t)
	options.ModelsDevCachePath = cachePath

	result := discover(t, base, options)

	byID := map[string]types.DiscoveredModel{}
	for _, model := range result.Models {
		byID[model.ID] = model
	}
	fbMatch(t, byID["detailed-route"], `{"name":"detailed-route","input":["text","image"],"contextWindow":64000,"maxTokens":8000,`+
		`"cost":{"input":7,"output":9,"cacheRead":1,"cacheWrite":2}}`)
	fbMatch(t, byID["private/priced-model"], `{"name":"private/priced-model (no metadata)","input":["text"],"contextWindow":128000,`+
		`"maxTokens":16384,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0}}`)
}

func TestFallbacks_HealthWithholdsMixedChatAndEmbeddingDetailRowsWithOneDiagnostic(t *testing.T) {
	route := "mixed-health-route"
	result, stderr := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbStatus(404),
		"/v1/models":  fbStatus(404),
		"/health": fbHealth(`{"model":"`+route+`","model_id":"chat-detail"}`,
			`{"model":"`+route+`","model_id":"embedding-detail"}`, `{"model":"safe-health-route"}`),
		"litellm_model_id=chat-detail": fbData(`{"model_name":"` + route + `","model_info":{"id":"chat","mode":"chat"}}`),
		"litellm_model_id=embedding-detail": fbData(`{"model_name":"` + route +
			`","model_info":{"id":"embedding","mode":"embedding"}}`),
	})

	fbExpectStrings(t, fbIDs(result.Models), "safe-health-route")
	lines := stderr.Lines()
	if len(lines) != 1 {
		t.Fatalf("stderr lines = %q, want 1", lines)
	}
	want := "1 route group(s) mix chat-style and explicitly incompatible deployment modes; the routes are withheld " +
		"because not every deployment can accept chat requests: " + route
	if !strings.Contains(lines[0], want) {
		t.Errorf("message %q lacks %q", lines[0], want)
	}
}

func TestFallbacks_HealthWithholdsAnEmbeddingOnlyRouteWithoutADiagnostic(t *testing.T) {
	route := "embedding-health-route"
	result, stderr := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbStatus(404),
		"/v1/models":  fbStatus(404),
		"/health": fbHealth(`{"model":"`+route+`","model_id":"embedding-detail"}`,
			`{"model":"safe-health-route"}`),
		"litellm_model_id=embedding-detail": fbData(`{"model_name":"` + route +
			`","model_info":{"id":"embedding","mode":"embedding"}}`),
	})

	fbExpectStrings(t, fbIDs(result.Models), "safe-health-route")
	if lines := stderr.Lines(); len(lines) != 0 {
		t.Errorf("stderr = %q, want none", lines)
	}
}

func TestFallbacks_HealthReducesConflictingDeploymentDetailsConservatively(t *testing.T) {
	details := map[string]string{
		"responses": `{"model_name":"shared-health-route","model_info":{"id":"responses","mode":"responses",` +
			`"supports_reasoning":true,"supports_vision":true,"max_input_tokens":200000,"max_output_tokens":64000,` +
			`"input_cost_per_token":0.000001,"output_cost_per_token":0.00001,"cache_read_input_token_cost":0.0000001,` +
			`"cache_creation_input_token_cost":0.000002}}`,
		"chat": `{"model_name":"shared-health-route","model_info":{"id":"chat","mode":"chat",` +
			`"supports_reasoning":false,"supports_vision":false,"max_input_tokens":8000,"max_output_tokens":1000,` +
			`"input_cost_per_token":0.000002,"output_cost_per_token":0.00002,"cache_read_input_token_cost":0.0000002,` +
			`"cache_creation_input_token_cost":0.000003}}`,
	}
	for _, tc := range []struct {
		name  string
		modes []string
	}{
		{"Responses first", []string{"responses", "chat"}},
		{"Chat first", []string{"chat", "responses"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			entries := make([]string, 0, len(tc.modes))
			routes := map[string]http.HandlerFunc{"/model/info": fbStatus(404), "/v1/models": fbStatus(404)}
			for _, mode := range tc.modes {
				entries = append(entries, `{"model":"shared-health-route","model_id":"`+mode+`"}`)
				routes["litellm_model_id="+mode] = fbData(details[mode])
			}
			routes["/health"] = fbHealth(entries...)

			result, _ := fbDiscover(t, routes)

			if len(result.Models) != 1 {
				t.Fatalf("models = %v, want 1", fbIDs(result.Models))
			}
			model := result.Models[0]
			fbMatch(t, model, `{"id":"shared-health-route","name":"shared-health-route","api":"openai-completions",`+
				`"reasoning":false,"input":["text"],"contextWindow":8000,"maxTokens":1000,`+
				`"cost":{"input":2,"output":20,"cacheWrite":3}}`)
			if math.Abs(model.Cost.CacheRead-0.2) > 1e-9 {
				t.Errorf("cacheRead = %v, want ~0.2", model.Cost.CacheRead)
			}
		})
	}
}

func TestFallbacks_HealthDoesNotSuppressDuplicateRoutesWithDifferentBackendFamilies(t *testing.T) {
	for _, tc := range []struct {
		name   string
		routes []string
	}{
		{"Moonshot first", []string{"moonshot/kimi-k3", "azure/gpt-4o"}},
		{"Azure first", []string{"azure/gpt-4o", "moonshot/kimi-k3"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			routes := map[string]http.HandlerFunc{"/model/info": fbStatus(403), "/v1/models": fbStatus(403)}
			entries := make([]string, 0, len(tc.routes))
			for index, route := range tc.routes {
				entries = append(entries, fmt.Sprintf(`{"model":"kimi-prod","model_id":"route-%d"}`, index))
				routes[fmt.Sprintf("litellm_model_id=route-%d", index)] = fbData(
					`{"litellm_params":{"model":"` + route + `"},"model_info":{"mode":"chat"}}`)
			}
			routes["/health"] = fbHealth(entries...)

			result, _ := fbDiscover(t, routes)

			if result.Source != types.SourceHealth || len(result.Models) != 1 {
				t.Fatalf("source = %s, models = %v", result.Source, fbIDs(result.Models))
			}
			if policy := result.Models[0].LiteLLMPolicy; policy != nil && policy.SuppressReasoningVisibility {
				t.Errorf("policy = %+v, want no suppression", policy)
			}
		})
	}
}

func TestFallbacks_HealthSuppressesDuplicateProvenNonForcedMoonshotRoutes(t *testing.T) {
	// The Vitest mock answers every non-listing URL with the same Moonshot row.
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbStatus(403),
		"/v1/models":  fbStatus(403),
		"/health": fbHealth(`{"model":"kimi-prod","model_id":"route-1"}`,
			`{"model":"kimi-prod","model_id":"route-2"}`),
		"litellm_model_id=route-1": fbData(`{"litellm_params":{"model":"moonshot/kimi-k3"},"model_info":{"mode":"chat"}}`),
		"litellm_model_id=route-2": fbData(`{"litellm_params":{"model":"moonshot/kimi-k3"},"model_info":{"mode":"chat"}}`),
	})

	if policy := result.Models[0].LiteLLMPolicy; policy == nil || !policy.SuppressReasoningVisibility {
		t.Errorf("policy = %+v, want suppressReasoningVisibility", policy)
	}
}

// ---- shared helpers for the remaining blocks ----

const fbNoReasoningLevels = `{"off":null,"minimal":null,"low":null,"medium":null,"high":null,"xhigh":null,"max":null}`

// fbContaining is `expect.objectContaining`: top-level keys match, nested values compare exactly except
// for the keys listed in subset, which match as objects.
func fbContaining(t *testing.T, got any, want string, subset ...string) {
	t.Helper()
	g, _ := fbNormalize(t, got).(map[string]any)
	w, _ := fbNormalize(t, want).(map[string]any)
	for key, wantValue := range w {
		gotValue, present := g[key]
		matched := present && reflect.DeepEqual(gotValue, wantValue)
		if present && !matched {
			if slices.Contains(subset, key) {
				matched = fbMatches(gotValue, wantValue)
			} else {
				matched = fbEqualValues(gotValue, wantValue)
			}
		}
		if !matched {
			encoded, _ := json.Marshal(g)
			t.Errorf("property %s: got %s\nwant %v", key, encoded, wantValue)
		}
	}
}

// fbEqualValues is reflect.DeepEqual with the float tolerance of fbMatches.
func fbEqualValues(got, want any) bool {
	switch w := want.(type) {
	case map[string]any:
		g, ok := got.(map[string]any)
		if !ok || len(g) != len(w) {
			return false
		}
		for key, wantValue := range w {
			if gotValue, present := g[key]; !present || !fbEqualValues(gotValue, wantValue) {
				return false
			}
		}
		return true
	case []any:
		g, ok := got.([]any)
		if !ok || len(g) != len(w) {
			return false
		}
		for i := range w {
			if !fbEqualValues(g[i], w[i]) {
				return false
			}
		}
		return true
	}
	return fbMatches(got, want)
}

// fbNotMatch asserts `expect(got).not.toMatchObject(want)`.
func fbNotMatch(t *testing.T, got any, want string) {
	t.Helper()
	if fbMatches(fbNormalize(t, got), fbNormalize(t, want)) {
		t.Errorf("got %v, want it not to match %s", got, want)
	}
}

// fbPiModel is the Pi catalog's `getModel(provider, id)`.
func fbPiModel(t *testing.T, provider, id string) ai.GeneratedModel {
	t.Helper()
	for _, model := range ai.GeneratedModels {
		if model.Provider == provider && model.ID == id {
			return model
		}
	}
	t.Fatalf("Pi catalog lacks %s/%s", provider, id)
	return ai.GeneratedModel{}
}

func fbPiCost(model ai.GeneratedModel) ai.ModelCost {
	return ai.ModelCost{
		Input: model.InputCostPerMTokens, Output: model.OutputCostPerMTokens,
		CacheRead: model.CacheReadCost, CacheWrite: model.CacheWriteCost, Tiers: model.Tiers,
	}
}

func fbPiInput(model ai.GeneratedModel) []string {
	if slices.Contains(model.Capabilities, "image") {
		return []string{"text", "image"}
	}
	return []string{"text"}
}

// fbSupportedLevels is Pi's getSupportedThinkingLevels for a discovered model.
func fbSupportedLevels(model types.DiscoveredModel) []ai.ThinkingLevel {
	return ai.GetSupportedThinkingLevels(&ai.Model{
		ProviderMeta:     ai.ProviderMetadata{Reasoning: model.Reasoning},
		ThinkingLevelMap: model.ThinkingLevelMap,
	})
}

// ---- discoverModels wildcard expansion via /v1/models ----

func TestFallbacks_WildcardRestoresChatReasoningCarriersAndCacheMarkersAfterProtocolSelection(t *testing.T) {
	row := `{"model_name":"team/*","litellm_params":{"model":"openai/gpt-*"},"model_info":{"supports_reasoning":true,` +
		`"supported_openai_params":["reasoning_effort"],"supports_low_reasoning_effort":true}}`
	if api := ModelProtocol("team/*", entryFrom(t, row), nil).API; api != ai.APIOpenAIResponses {
		t.Fatalf("wildcard protocol = %s, want openai-responses", api)
	}

	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(row),
		"/v1/models":  fbData(`{"id":"team/claude-sonnet-4-5"}`),
	})

	if len(result.Models) != 1 {
		t.Fatalf("models = %v, want 1", fbIDs(result.Models))
	}
	fbMatch(t, result.Models[0], `{"id":"team/claude-sonnet-4-5","api":"openai-completions","litellmBackendFamily":"claude",`+
		`"thinkingLevelMap":{"low":"low"},"compat":{"supportsReasoningEffort":true,"cacheControlFormat":"anthropic"}}`)
	fbLacks(t, result.Models[0], "litellmResponsesReasoningControl")
}

func TestFallbacks_WildcardExpandsAModelInfoEntryAndDropsTheLiteralWildcardID(t *testing.T) {
	result, proxy, _ := fbDiscoverProxy(t, map[string]http.HandlerFunc{
		"/model/info": fbData(`{"model_name":"lemonade/*","model_info":{"mode":"chat"}}`,
			`{"model_name":"lemonade/Qwen3.6-35B-A3B-MTP-GGUF-UD-IQ4_NL","model_info":{"mode":"chat"}}`),
		"/v1/models": fbData(
			`{"id":"lemonade/*","object":"model","owned_by":"openai"}`,
			`{"id":"lemonade/Bonsai-1.7B-gguf","object":"model","owned_by":"openai"}`,
			`{"id":"lemonade/Laguna-S-2.1-GGUF-UD-IQ4_NL","object":"model","owned_by":"openai"}`,
			`{"id":"lemonade/Qwen3.6-35B-A3B-MTP-GGUF-UD-IQ4_NL","object":"model","owned_by":"openai"}`),
	})

	if result.Source != types.SourceModelInfo {
		t.Errorf("source = %s", result.Source)
	}
	ids := fbIDs(result.Models)
	sorted := slices.Sorted(slices.Values(ids))
	fbExpectStrings(t, sorted, "lemonade/Bonsai-1.7B-gguf", "lemonade/Laguna-S-2.1-GGUF-UD-IQ4_NL",
		"lemonade/Qwen3.6-35B-A3B-MTP-GGUF-UD-IQ4_NL")
	for _, model := range result.Models {
		if strings.Contains(model.ID, "*") {
			t.Errorf("literal wildcard %q surfaced", model.ID)
		}
		if (model.ID == "lemonade/Bonsai-1.7B-gguf" || model.ID == "lemonade/Laguna-S-2.1-GGUF-UD-IQ4_NL") &&
			model.API != ai.APIOpenAICompletions {
			t.Errorf("%s api = %s", model.ID, model.API)
		}
	}
	if fbCount(ids, "lemonade/Qwen3.6-35B-A3B-MTP-GGUF-UD-IQ4_NL") != 1 {
		t.Errorf("concrete entry duplicated: %v", ids)
	}
	if fbCount(proxy.Requests(), "/v1/models") == 0 {
		t.Error("/v1/models was not queried")
	}
}

func TestFallbacks_WildcardKeepsAChildOnChatWhenItsRowPinsAnOlderAzureAPIVersion(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(`{"model_name":"azure/*","litellm_params":{"model":"azure/*","api_version":"2024-10-21"},` +
			`"model_info":{"mode":"chat","litellm_provider":"azure"}}`),
		"/v1/models": fbData(`{"id":"azure/gpt-5.5","object":"model","owned_by":"openai"}`),
	})

	if len(result.Models) != 1 {
		t.Fatalf("models = %v, want 1", fbIDs(result.Models))
	}
	fbMatch(t, result.Models[0], `{"id":"azure/gpt-5.5","api":"openai-completions","litellmBackendFamily":"openai"}`)
}

func TestFallbacks_WildcardUsesOnlyTheRouteLiteLLMWouldSelectForAChild(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(
			`{"model_name":"*","litellm_params":{"model":"anthropic/*"},"model_info":{"supported_endpoints":["/v1/chat/completions"]}}`,
			`{"model_name":"openai/*","litellm_params":{"model":"openai/*"},"model_info":{"supported_endpoints":["/v1/responses"]}}`),
		"/v1/models": fbData(`{"id":"openai/gpt-5","object":"model","owned_by":"openai"}`),
	})

	if len(result.Models) != 1 {
		t.Fatalf("models = %v, want 1", fbIDs(result.Models))
	}
	fbMatch(t, result.Models[0], `{"id":"openai/gpt-5","api":"openai-responses","litellmBackendFamily":"openai"}`)
}

func TestFallbacks_WildcardResolvesEquallySpecificRoutesIndependentlyOfRowOrder(t *testing.T) {
	chat := `{"model_name":"team-*","model_info":{"mode":"chat"}}`
	embedding := `{"model_name":"*-team","model_info":{"mode":"embedding"}}`
	for _, rows := range [][]string{{chat, embedding}, {embedding, chat}} {
		result, _ := fbDiscover(t, map[string]http.HandlerFunc{
			"/model/info": fbData(rows...),
			"/v1/models":  fbData(`{"id":"team-x-team","object":"model"}`),
		})
		if len(result.Models) != 0 {
			t.Errorf("rows %v: models = %v, want none", rows, fbIDs(result.Models))
		}
	}
}

func TestFallbacks_WildcardLetsRowsSharingTheSelectedRouteVoteTogether(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(
			`{"model_name":"team/*","litellm_params":{"model":"openai/*"},"model_info":{"supported_endpoints":["/v1/responses"]}}`,
			`{"model_name":"team/*","litellm_params":{"model":"openai/*"},"model_info":{"supported_endpoints":["/v1/chat/completions"]}}`),
		"/v1/models": fbData(`{"id":"team/production","object":"model","owned_by":"openai"}`),
	})

	if len(result.Models) != 1 {
		t.Fatalf("models = %v, want 1", fbIDs(result.Models))
	}
	fbMatch(t, result.Models[0], `{"id":"team/production","api":"openai-completions"}`)
	fbLacks(t, result.Models[0], "litellmBackendFamily")
}

func TestFallbacks_WildcardDoesNotLetARejectedEmbeddingWildcardAuthorizeAChild(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(`{"model_name":"chat/*","model_info":{"mode":"chat"}}`,
			`{"model_name":"embed/*","model_info":{"mode":"embedding"}}`),
		"/v1/models": fbData(`{"id":"chat/a","object":"model"}`, `{"id":"embed/x","object":"model"}`),
	})

	fbExpectStrings(t, fbIDs(result.Models), "chat/a")
}

func TestFallbacks_WildcardDoesNotFallThroughToACatchAllWhenTheSelectedRouteIsRejected(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(`{"model_name":"*","litellm_params":{"model":"openai/*"},"model_info":{"mode":"chat"}}`,
			`{"model_name":"embed/*","model_info":{"mode":"embedding"}}`),
		"/v1/models": fbData(`{"id":"embed/text-embedding-3-large","object":"model"}`),
	})

	fbExpectStrings(t, fbIDs(result.Models))
}

func TestFallbacks_WildcardRecomputesConcreteIDCompatibilityWhilePreservingConservativeMetadata(t *testing.T) {
	wildcard := "openai/nonce/*"
	claudeID := "openai/nonce/claude-sonnet-4-6"
	kimiID := "openai/nonce/kimi-k2.6"
	result, stderr := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(
			`{"model_name":"`+wildcard+`","litellm_params":{"model":"openai/gpt-4o"},"model_info":{"id":"openai-route",`+
				`"mode":"responses","supports_vision":true,"max_input_tokens":32000,"max_output_tokens":4000,`+
				`"input_cost_per_token":0.000004,"output_cost_per_token":0.00002,"cache_read_input_token_cost":0.0000004,`+
				`"cache_creation_input_token_cost":0.000005}}`,
			`{"model_name":"`+wildcard+`","litellm_params":{"model":"anthropic/claude-sonnet-4-6"},"model_info":{"id":"anthropic-route",`+
				`"mode":"responses","supports_reasoning":false,"supports_vision":false,"max_input_tokens":16000,`+
				`"max_output_tokens":2000,"input_cost_per_token":0.000003,"output_cost_per_token":0.000015,`+
				`"cache_read_input_token_cost":0.0000003,"cache_creation_input_token_cost":0.00000375}}`),
		"/v1/models": fbData(`{"id":"`+claudeID+`","owned_by":"openai"}`, `{"id":"`+kimiID+`","owned_by":"openai"}`),
	})

	if len(result.Models) != 2 {
		t.Fatalf("models = %v, want 2", fbIDs(result.Models))
	}
	claudeName := fbPiModel(t, "anthropic", "claude-sonnet-4-6").DisplayName + " (incomplete metadata)"
	fbContaining(t, result.Models[0], fmt.Sprintf(`{"id":%q,"name":%q,"api":"openai-responses","reasoning":false,`+
		`"input":["text"],"contextWindow":16000,"maxTokens":2000,"cost":{"input":4,"output":20,"cacheWrite":5}}`,
		claudeID, claudeName), "cost")
	fbLacks(t, result.Models[0], "compat")
	fbContaining(t, result.Models[1], fmt.Sprintf(`{"id":%q,"name":%q,"reasoning":false,"input":["text"],`+
		`"contextWindow":16000,"maxTokens":2000}`, kimiID, kimiID+" (incomplete metadata)"))
	fbLacks(t, result.Models[1], "compat")
	if len(stderr.Lines()) == 0 {
		t.Error("expected a stderr diagnostic")
	}
}

func TestFallbacks_WildcardRecomputesKimiCompatibilityFromACatchAllExpansionID(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(`{"model_name":"*","litellm_params":{"model":"private/unknown"},"model_info":{"mode":"responses",` +
			`"supports_reasoning":false,"max_input_tokens":8000,"max_output_tokens":1000,"input_cost_per_token":0.000001}}`),
		"/v1/models": fbData(`{"id":"kimi-k2.6"}`),
	})

	if len(result.Models) != 1 {
		t.Fatalf("models = %v, want 1", fbIDs(result.Models))
	}
	fbContaining(t, result.Models[0], `{"id":"kimi-k2.6","name":"kimi-k2.6 (incomplete metadata)","api":"openai-responses",`+
		`"reasoning":false,"contextWindow":8000,"maxTokens":1000,"cost":{"input":1,"output":0,"cacheRead":0,"cacheWrite":0},`+
		`"compat":{"supportsDeveloperRole":false,"supportsStrictMode":false}}`)
}

func TestFallbacks_WildcardPreservesUnresolvedConservativeMetadataOnExpandedCatalogMatchingIDs(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(`{"model_name":"openai/*","litellm_params":{"model":"private/unknown"},"model_info":{"mode":"chat",` +
			`"supports_reasoning":false,"max_input_tokens":8000,"input_cost_per_token":0.000001}}`),
		"/v1/models": fbData(`{"id":"openai/gpt-4o","owned_by":"openai"}`),
	})

	if len(result.Models) != 1 {
		t.Fatalf("models = %v, want 1", fbIDs(result.Models))
	}
	fbContaining(t, result.Models[0], `{"id":"openai/gpt-4o","name":"GPT-4o (incomplete metadata)","reasoning":false,`+
		`"input":["text"],"contextWindow":8000,"maxTokens":16384,"cost":{"input":1,"output":0,"cacheRead":0,"cacheWrite":0}}`)
}

func TestFallbacks_WildcardAggregatesSuppressionEvidenceOntoExpansions(t *testing.T) {
	for _, tc := range []struct {
		name, backend string
		suppress      bool
	}{
		{"Moonshot", "moonshot/*", true},
		{"non-Moonshot", "openai/*", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			result, _ := fbDiscover(t, map[string]http.HandlerFunc{
				"/model/info": fbData(`{"model_name":"team/*","litellm_params":{"model":"` + tc.backend +
					`"},"model_info":{"id":"wildcard","mode":"chat"}}`),
				"/v1/models": fbData(`{"id":"team/model-a"}`),
			})

			policy := result.Models[0].LiteLLMPolicy
			if got := policy != nil && policy.SuppressReasoningVisibility; got != tc.suppress {
				t.Errorf("suppressReasoningVisibility = %v, want %v", got, tc.suppress)
			}
		})
	}
}

func TestFallbacks_WildcardDoesNotSuppressAForcedThinkingIDExpandedFromAMoonshotWildcard(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(`{"model_name":"*","litellm_params":{"model":"moonshot/*"},"model_info":{"id":"wildcard","mode":"chat"}}`),
		"/v1/models":  fbData(`{"id":"kimi-k2-thinking"}`),
	})

	if result.Models[0].ID != "kimi-k2-thinking" {
		t.Errorf("id = %s", result.Models[0].ID)
	}
	if policy := result.Models[0].LiteLLMPolicy; policy != nil && policy.SuppressReasoningVisibility {
		t.Error("forced-thinking id was suppressed")
	}
}

func TestFallbacks_WildcardRequiresUnanimousSuppressionEvidenceAcrossMatchingGroups(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(
			`{"model_name":"team/*","litellm_params":{"model":"moonshot/*"},"model_info":{"id":"team-wildcard","mode":"chat"}}`,
			`{"model_name":"*","litellm_params":{"model":"openai/*"},"model_info":{"id":"catch-all-wildcard","mode":"chat"}}`),
		"/v1/models": fbData(`{"id":"team/model-a"}`),
	})

	if result.Models[0].ID != "team/model-a" {
		t.Errorf("id = %s", result.Models[0].ID)
	}
	if policy := result.Models[0].LiteLLMPolicy; policy != nil && policy.SuppressReasoningVisibility {
		t.Error("suppression was not unanimous but was applied")
	}
}

func TestFallbacks_WildcardPreservesAWildcardResponsesRouteModeOnExpandedModels(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(`{"model_name":"responses/*","model_info":{"mode":"responses"}}`,
			`{"model_name":"chat/*","model_info":{"mode":"chat"}}`),
		"/v1/models": fbData(`{"id":"responses/model-a","owned_by":"openai"}`, `{"id":"chat/model-b","owned_by":"openai"}`,
			`{"id":"other/model-c","owned_by":"openai"}`),
	})

	pairs := fbIDAPIs(result.Models)
	for _, want := range []string{"responses/model-a=openai-responses", "chat/model-b=openai-completions"} {
		if !slices.Contains(pairs, want) {
			t.Errorf("missing %s in %v", want, pairs)
		}
	}
	if slices.Contains(fbIDs(result.Models), "other/model-c") {
		t.Error("other/model-c was published")
	}
}

func TestFallbacks_WildcardUsesThinkingLevelsFromTheSelectedMostSpecificWildcard(t *testing.T) {
	wildcard := func(name, id, supports string) string {
		return `{"model_name":"` + name + `","litellm_params":{"model":"internal/` + id +
			`","allowed_openai_params":["reasoning_effort"]},"model_info":{"id":"` + id + `","mode":"chat","supports_reasoning":true,` +
			supports + `,"input_cost_per_token":0,"output_cost_per_token":0,"cache_read_input_token_cost":0,` +
			`"cache_creation_input_token_cost":0,"max_input_tokens":8000,"max_output_tokens":1000,"supports_vision":false}}`
	}
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(wildcard("*", "broad", `"supports_low_reasoning_effort":true`),
			wildcard("team/*", "narrow", `"supports_low_reasoning_effort":false,"supports_max_reasoning_effort":true`)),
		"/v1/models": fbData(`{"id":"team/model"}`),
	})

	fbEqual(t, result.Models[0].ThinkingLevelMap, `{"low":null,"xhigh":null,"max":"max"}`)
}

func TestFallbacks_WildcardPreservesTieredPricingFromASingleWildcardMatch(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(`{"model_name":"team/*","litellm_params":{"model":"openai/gpt-5.5"},"model_info":{"id":"wildcard","mode":"chat"}}`),
		"/v1/models":  fbData(`{"id":"team/model"}`),
	})

	fbEqual(t, result.Models[0].Cost.Tiers,
		`[{"inputTokensAbove":272000,"input":10,"output":45,"cacheRead":1,"cacheWrite":0}]`)
}

func TestFallbacks_WildcardCombinesCompatibleTieredPricingAcrossOverlappingMatchesConservatively(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(
			`{"model_name":"*","litellm_params":{"model":"openai/gpt-5.5"},"model_info":{"id":"broad","mode":"chat"}}`,
			`{"model_name":"team/*","litellm_params":{"model":"openai/gpt-5.5-pro"},"model_info":{"id":"narrow","mode":"chat"}}`),
		"/v1/models": fbData(`{"id":"team/model"}`),
	})

	fbEqual(t, result.Models[0].Cost.Tiers,
		`[{"inputTokensAbove":272000,"input":60,"output":270,"cacheRead":1,"cacheWrite":0}]`)
}

func TestFallbacks_WildcardConstructsASafeTierEnvelopeWhenOverlappingThresholdsDiffer(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(
			`{"model_name":"*","litellm_params":{"model":"openai/gpt-5.5"},"model_info":{"id":"broad","mode":"chat"}}`,
			`{"model_name":"team/*","litellm_params":{"model":"github-copilot/grok-4.5"},"model_info":{"id":"narrow","mode":"chat"}}`),
		"/v1/models": fbData(`{"id":"team/model"}`),
	})

	fbEqual(t, result.Models[0].Cost.Tiers, `[{"inputTokensAbove":200000,"input":5,"output":30,"cacheRead":1,"cacheWrite":0},`+
		`{"inputTokensAbove":272000,"input":10,"output":45,"cacheRead":1,"cacheWrite":0}]`)
}

func TestFallbacks_WildcardRetainsKnownHigherTiersWhenAnOverlappingWildcardHasIncompleteMetadata(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(
			`{"model_name":"*","litellm_params":{"model":"openai/gpt-5.5"},"model_info":{"id":"complete","mode":"chat"}}`,
			`{"model_name":"team/*","litellm_params":{"model":"internal/unknown"},"model_info":{"id":"incomplete","mode":"chat",`+
				`"input_cost_per_token":0.000002,"output_cost_per_token":0.00001,"cache_read_input_token_cost":0.0000002,`+
				`"cache_creation_input_token_cost":0.0000025}}`),
		"/v1/models": fbData(`{"id":"team/model"}`),
	})

	if name := result.Models[0].Name; name != "team/model (incomplete metadata)" {
		t.Errorf("name = %q", name)
	}
	fbEqual(t, result.Models[0].Cost, `{"input":5,"output":30,"cacheRead":0.5,"cacheWrite":2.5,"tiers":`+
		`[{"inputTokensAbove":272000,"input":10,"output":45,"cacheRead":1,"cacheWrite":2.5}]}`)
}

func TestFallbacks_WildcardCombinesOverlappingMetadataConservativelyRegardlessOfRouteOrder(t *testing.T) {
	broad := `{"model_name":"*","litellm_params":{"model":"internal/broad"},"model_info":{"id":"broad","mode":"responses",` +
		`"supports_reasoning":true,"supports_vision":true,"supports_low_reasoning_effort":true,"max_input_tokens":80000,` +
		`"max_output_tokens":8000,"input_cost_per_token":0.000002,"output_cost_per_token":0.00001,` +
		`"cache_read_input_token_cost":0.0000002,"cache_creation_input_token_cost":0.0000025}}`
	narrow := `{"model_name":"team/*","litellm_params":{"model":"internal/narrow"},"model_info":{"id":"narrow","mode":"chat",` +
		`"supports_reasoning":false,"supports_vision":false,"supports_low_reasoning_effort":false,"max_input_tokens":40000,` +
		`"max_output_tokens":4000,"input_cost_per_token":0.000003,"output_cost_per_token":0.000015,` +
		`"cache_read_input_token_cost":0.0000003,"cache_creation_input_token_cost":0.00000375}}`
	for _, rows := range [][]string{{broad, narrow}, {narrow, broad}} {
		result, _ := fbDiscover(t, map[string]http.HandlerFunc{
			"/model/info": fbData(rows...),
			"/v1/models":  fbData(`{"id":"team/claude-sonnet-4-6"}`),
		})

		fbEqual(t, result.Models, `[{"id":"team/claude-sonnet-4-6","name":"Claude Sonnet 4.6","api":"openai-completions",`+
			`"litellmDiscoveryVersion":6,"reasoning":false,"input":["text"],"contextWindow":40000,"maxTokens":4000,`+
			`"cost":{"input":3,"output":15,"cacheRead":0.3,"cacheWrite":3.75},`+
			`"compat":{"supportsStore":false,"cacheControlFormat":"anthropic"}}]`)
	}
}

func TestFallbacks_WildcardKeepsADroppedConcreteGroupFromBeingReAdmitted(t *testing.T) {
	for _, wildcardRoute := range []string{"team/*", "*"} {
		t.Run(wildcardRoute, func(t *testing.T) {
			result, _ := fbDiscover(t, map[string]http.HandlerFunc{
				"/model/info": fbData(`{"model_name":"`+wildcardRoute+`","model_info":{"id":"wildcard","mode":"chat"}}`,
					`{"model_name":"team/embed","model_info":{"id":"embedding","mode":"embedding"}}`),
				"/v1/models": fbData(`{"id":"team/chat"}`, `{"id":"team/embed"}`),
			})

			fbExpectStrings(t, fbIDs(result.Models), "team/chat")
		})
	}
}

func TestFallbacks_WildcardKeepsADeliberatelyDroppedWildcardGroupFromBeingReAdmitted(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(`{"model_name":"team/*","model_info":{"id":"wildcard","mode":"chat"}}`,
			`{"model_name":"blocked/*","model_info":{"id":"blocked","mode":"embedding"}}`),
		"/v1/models": fbData(`{"id":"team/chat"}`, `{"id":"blocked/chat"}`),
	})

	fbExpectStrings(t, fbIDs(result.Models), "team/chat")
}

func TestFallbacks_WildcardNeverExposesALiteralWildcardWhenExpansionFails(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(`{"model_name":"team/*","model_info":{"mode":"chat"}}`),
		"/v1/models":  jsonResponse(500, map[string]any{"error": "unavailable"}),
	})

	fbExpectStrings(t, fbIDs(result.Models))
}

func TestFallbacks_WildcardDropsLiteralWildcardIDsWhenExpansionFails(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(`{"model_name":"*","model_info":{"mode":"chat"}}`),
		"/v1/models":  func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(500) },
	})

	if result.Source != types.SourceModelInfo || len(result.Models) != 0 || result.ProxyVersion != nil {
		t.Errorf("result = %+v, want {source: model_info, models: []}", result)
	}
}

func TestFallbacks_WildcardDoesNotPublishListIDsThatMatchNoWildcardRoute(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(`{"model_name":"team/*","litellm_params":{"model":"openai/*"},"model_info":{"mode":"chat"}}`),
		"/v1/models": fbData(`{"id":"team/gpt-production","object":"model","owned_by":"openai"}`,
			`{"id":"unrelated/model","object":"model","owned_by":"openai"}`),
	})

	fbExpectStrings(t, fbIDs(result.Models), "team/gpt-production")
}

func TestFallbacks_WildcardGivesAChildResponsesWhenItsRowIsACurrentOpenAIDeployment(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(`{"model_name":"team/*","litellm_params":{"model":"openai/*"},"model_info":{"mode":"chat"}}`),
		"/v1/models":  fbData(`{"id":"team/gpt-production","object":"model","owned_by":"openai"}`),
	})

	fbMatch(t, result.Models[0], `{"id":"team/gpt-production","api":"openai-responses"}`)
}

func TestFallbacks_WildcardDoesNotQueryV1ModelsWhenModelInfoHasNoWildcards(t *testing.T) {
	result, proxy, _ := fbDiscoverProxy(t, map[string]http.HandlerFunc{
		"/model/info": fbData(`{"model_name":"openai/gpt-4o","model_info":{"mode":"chat"}}`),
	})

	if result.Source != types.SourceModelInfo {
		t.Errorf("source = %s", result.Source)
	}
	fbExpectStrings(t, fbIDs(result.Models), "openai/gpt-4o")
	if fbCount(proxy.Requests(), "/v1/models") != 0 {
		t.Errorf("requests = %v", proxy.Requests())
	}
}

func TestFallbacks_WildcardCarriesAParentsTieredPricingIntoItsChild(t *testing.T) {
	entry := `{"model_name":"team/*","litellm_params":{"model":"openai/gpt-5.4"},"model_info":{"id":"wildcard","mode":"chat"}}`
	resolution := ResolveModelInfoCatalog(entryFrom(t, entry), nil)
	if resolution == nil || resolution.Cost == nil || resolution.Cost.Tiers == nil {
		t.Fatalf("resolution = %+v, want tiers", resolution)
	}
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(entry),
		"/v1/models":  fbData(`{"id":"team/assistant"}`),
	})

	fbEqual(t, result.Models[0].Cost.Tiers, fbJSONString(t, resolution.Cost.Tiers))
}

func fbJSONString(t *testing.T, v any) string {
	t.Helper()
	encoded, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(encoded)
}

func TestFallbacks_WildcardDoesNotReintroduceAWithheldModelInfoRoute(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(`{"model_name":"team/*","model_info":{"id":"wildcard","mode":"chat"}}`,
			`{"model_name":"blocked/model","model_info":{"id":"chat","mode":"chat"}}`,
			`{"model_name":"blocked/model","model_info":{"id":"embedding","mode":"embedding"}}`),
		"/v1/models": fbData(`{"id":"team/expanded","owned_by":"openai"}`, `{"id":"blocked/model","owned_by":"openai"}`),
	})

	fbExpectStrings(t, fbIDs(result.Models), "team/expanded")
}

func TestFallbacks_WildcardKeepsToolRepairWithholdingOnTheExpandedModel(t *testing.T) {
	result, stderr := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(
			`{"model_name":"team/*","litellm_params":{"model":"moonshot/kimi-k2.6"},"model_info":{"id":"kimi","mode":"chat"}}`,
			`{"model_name":"team/*","model_info":{"id":"unidentified","mode":"chat"}}`),
		"/v1/models": fbData(`{"id":"team/assistant"}`),
	})

	fbEqual(t, result.Models[0].LiteLLMPolicy,
		`{"normalizeStrictToolMessages":false,"normalizeThinkTags":false,"suppressReasoningVisibility":false}`)
	if !strings.Contains(fbJoined(stderr), "team/assistant") {
		t.Errorf("stderr %q lacks team/assistant", fbJoined(stderr))
	}
}

func TestFallbacks_WildcardWarnsWhenARouteOnlyWildcardExpandsToAMoonshotID(t *testing.T) {
	result, stderr := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(`{"model_name":"*","model_info":{"id":"wildcard","mode":"chat"}}`),
		"/v1/models":  fbData(`{"id":"kimi-wildcard-review"}`),
	})

	if policy := result.Models[0].LiteLLMPolicy; policy == nil || policy.NormalizeStrictToolMessages {
		t.Errorf("policy = %+v, want normalizeStrictToolMessages false", policy)
	}
	if !strings.Contains(fbJoined(stderr), "kimi-wildcard-review") {
		t.Errorf("stderr %q lacks kimi-wildcard-review", fbJoined(stderr))
	}
}

func TestFallbacks_WildcardDoesNotWarnForACandidateAlreadyPublishedByAnExactRoute(t *testing.T) {
	result, stderr := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(`{"model_name":"*","model_info":{"id":"wildcard","mode":"chat"}}`,
			`{"model_name":"kimi-k2.6-exact-authority","litellm_params":{"model":"moonshot/kimi-k2.6","allowed_openai_params":["thinking"]},`+
				`"model_info":{"id":"exact","mode":"chat","supports_reasoning":true}}`),
		"/v1/models": fbData(`{"id":"kimi-k2.6-exact-authority"}`),
	})

	if len(result.Models) != 1 {
		t.Fatalf("models = %v, want 1", fbIDs(result.Models))
	}
	fbMatch(t, result.Models[0], `{"id":"kimi-k2.6-exact-authority","litellmPolicy":{"normalizeStrictToolMessages":true}}`)
	if strings.Contains(fbJoined(stderr), "kimi-k2.6-exact-authority") {
		t.Errorf("stderr %q mentions the exact route", fbJoined(stderr))
	}
}

func TestFallbacks_WildcardTakesOnlyCatalogPresentationMetadataForAWildcardChild(t *testing.T) {
	for _, tc := range []struct {
		label, pricing, name string
		cost                 string
	}{
		{"complete", `,"input_cost_per_token":0.000001,"output_cost_per_token":0.000002,"cache_read_input_token_cost":0.0000001,` +
			`"cache_creation_input_token_cost":0.0000002`, "GPT-5.5",
			`{"input":1,"output":2,"cacheRead":0.09999999999999999,"cacheWrite":0.19999999999999998}`},
		{"incomplete", ``, "GPT-5.5 (incomplete metadata)", `{"input":0,"output":0,"cacheRead":0,"cacheWrite":0}`},
	} {
		t.Run(tc.label, func(t *testing.T) {
			result, _ := fbDiscover(t, map[string]http.HandlerFunc{
				"/model/info": fbData(`{"model_name":"openai/*","litellm_params":{"model":"internal/proxy"},"model_info":{"id":"wildcard",` +
					`"mode":"chat","supported_openai_params":[],"supports_reasoning":true,"supports_vision":false,` +
					`"max_input_tokens":200000,"max_output_tokens":100000` + tc.pricing + `}}`),
				"/v1/models": fbData(`{"id":"openai/gpt-5.5","owned_by":"openai"}`, `{"id":"unmatched/catalog-model","owned_by":"openai"}`),
			})

			if len(result.Models) != 1 {
				t.Fatalf("models = %v, want 1", fbIDs(result.Models))
			}
			fbMatch(t, result.Models[0], fmt.Sprintf(`{"id":"openai/gpt-5.5","name":%q,"input":["text"],"cost":%s,`+
				`"contextWindow":200000,"maxTokens":100000,"reasoning":true,"thinkingLevelMap":%s}`,
				tc.name, tc.cost, fbNoReasoningLevels))
			fbNotMatch(t, result.Models[0].Compat, `{"supportsReasoningEffort":true}`)
		})
	}
}

func TestFallbacks_WildcardKeepsParentDeploymentLimitsWhenAChildHasSmallerCatalogLimits(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(`{"model_name":"openai/*","model_info":{"id":"wildcard","mode":"chat",` +
			`"max_input_tokens":200000,"max_output_tokens":100000}}`),
		"/v1/models": fbData(`{"id":"openai/gpt-4o","owned_by":"openai"}`),
	})

	fbMatch(t, result.Models[0], `{"id":"openai/gpt-4o","contextWindow":200000,"maxTokens":100000}`)
}

// ---- discoverModels proxy version gate ----

const fbVersionProbe = "/v1/responses/resp_version_probe"

// fbAzureAIRow builds the Vitest `azureAiRoute` with optional overrides.
func fbAzureAIRow(name, model, info string) string {
	return `{"model_name":"` + name + `","litellm_params":{"model":"` + model + `"},"model_info":` + info + `}`
}

const (
	fbAzureAIInfo     = `{"mode":"chat","supported_endpoints":["/v1/chat/completions","/v1/responses"]}`
	fbAzureAIRoute    = `{"model_name":"gpt-6-astra","litellm_params":{"model":"azure_ai/gpt-6-astra"},"model_info":` + fbAzureAIInfo + `}`
	fbAzureAIWildcard = `{"model_name":"foundry/*","litellm_params":{"model":"azure_ai/*"},"model_info":` + fbAzureAIInfo + `}`
)

// fbVersionReply answers the version probe the way LiteLLM does: a 400 carrying x-litellm-version.
func fbVersionReply(version string) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		if version != "" {
			w.Header().Set("x-litellm-version", version)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		_, _ = w.Write([]byte(`{"error":{"message":"no healthy deployments"}}`))
	}
}

// fbDropConnection never answers: the Go form of a fetch that rejects.
func fbDropConnection(http.ResponseWriter, *http.Request) { panic(http.ErrAbortHandler) }

func TestFallbacks_VersionGateSelectsTheAzureAITransportForTheReportedProxyVersion(t *testing.T) {
	for _, tc := range []struct {
		version string
		api     ai.API
	}{
		{"1.103.0", ai.APIOpenAIResponses},
		{"1.104.0", ai.APIOpenAIResponses},
		{"1.102.1", ai.APIOpenAICompletions},
		{"1.103.0-rc.1", ai.APIOpenAICompletions},
		{"not-a-version", ai.APIOpenAICompletions},
	} {
		t.Run(tc.version, func(t *testing.T) {
			result, _ := fbDiscover(t, map[string]http.HandlerFunc{
				"/model/info":  fbData(fbAzureAIRoute),
				fbVersionProbe: fbVersionReply(tc.version),
			})

			fbExpectStrings(t, fbIDAPIs(result.Models), "gpt-6-astra="+string(tc.api))
		})
	}
}

func TestFallbacks_VersionGateKeepsAzureAIOnChatWhenTheProxyCannotReportAVersion(t *testing.T) {
	for _, tc := range []struct {
		label string
		probe http.HandlerFunc
	}{
		{"omits the version header", fbVersionReply("")},
		{"does not answer the version probe", fbDropConnection},
	} {
		t.Run(tc.label, func(t *testing.T) {
			result, _ := fbDiscover(t, map[string]http.HandlerFunc{
				"/model/info":  fbData(fbAzureAIRoute),
				fbVersionProbe: tc.probe,
			})

			if result.Source != types.SourceModelInfo {
				t.Errorf("source = %s", result.Source)
			}
			fbExpectStrings(t, fbIDAPIs(result.Models), "gpt-6-astra=openai-completions")
		})
	}
}

func TestFallbacks_VersionGateKeepsAGroupOnChatWhenOneDeploymentNeedsItOnACurrentProxy(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(fbAzureAIRoute,
			fbAzureAIRow("gpt-6-astra", "azure_ai/gpt-6-astra", `{"mode":"chat","supported_endpoints":["/v1/chat/completions"]}`)),
		fbVersionProbe: fbVersionReply("1.103.0"),
	})

	fbExpectStrings(t, fbIDAPIs(result.Models), "gpt-6-astra=openai-completions")
}

func TestFallbacks_VersionGateAsksForTheVersionOnceHoweverManyDeploymentsDependOnIt(t *testing.T) {
	_, proxy, _ := fbDiscoverProxy(t, map[string]http.HandlerFunc{
		"/model/info": fbData(fbAzureAIRoute, fbAzureAIRoute,
			fbAzureAIRow("gpt-5.5", "azure_ai/gpt-6-astra", fbAzureAIInfo)),
		fbVersionProbe: fbVersionReply("1.103.0"),
	})

	if got := fbCount(proxy.Requests(), fbVersionProbe); got != 1 {
		t.Errorf("version probes = %d, want 1", got)
	}
}

func TestFallbacks_VersionGateDoesNotAskForTheVersionWhenOnlyAnIndependentRouteIsPublished(t *testing.T) {
	for _, tc := range []struct{ label, row string }{
		{"an OpenAI route", `{"model_name":"gpt","litellm_params":{"model":"openai/gpt-5"}}`},
		{"a native azure route listing Responses", `{"model_name":"gpt","litellm_params":{"model":"azure/gpt-5"},` +
			`"model_info":{"supported_endpoints":["/v1/responses"]}}`},
		{"an azure_ai route in explicit Responses mode", fbAzureAIRow("gpt-6-astra", "azure_ai/gpt-6-astra",
			`{"mode":"responses","supported_endpoints":["/v1/chat/completions","/v1/responses"]}`)},
		{"an azure_ai route whose endpoint list omits Responses", fbAzureAIRow("gpt-6-astra", "azure_ai/gpt-6-astra",
			`{"mode":"chat","supported_endpoints":["/v1/chat/completions"]}`)},
		{"an azure_ai route that lists no endpoints", fbAzureAIRow("gpt-6-astra", "azure_ai/gpt-6-astra", `{"mode":"chat"}`)},
	} {
		t.Run(tc.label, func(t *testing.T) {
			_, proxy, _ := fbDiscoverProxy(t, map[string]http.HandlerFunc{"/model/info": fbData(tc.row)})

			fbExpectStrings(t, proxy.Requests(), "/model/info")
		})
	}
}

func TestFallbacks_VersionGateDoesNotAskForTheVersionWhenAnAzureAIWildcardRouteListsNothingUseful(t *testing.T) {
	for _, tc := range []struct {
		label      string
		listModels http.HandlerFunc
	}{
		{"cannot be listed", fbStatus(500)},
		{"lists nothing", fbData()},
		{"lists only ids the wildcard does not match", fbData(`{"id":"other/gpt-6-astra"}`)},
	} {
		t.Run(tc.label, func(t *testing.T) {
			result, proxy, _ := fbDiscoverProxy(t, map[string]http.HandlerFunc{
				"/model/info":  fbData(fbAzureAIWildcard),
				"/v1/models":   tc.listModels,
				fbVersionProbe: fbVersionReply("1.103.0"),
			})

			fbExpectStrings(t, fbIDs(result.Models))
			if got := fbCount(proxy.Requests(), fbVersionProbe); got != 0 {
				t.Errorf("version probes = %d, want 0", got)
			}
		})
	}
}

func TestFallbacks_VersionGateDoesNotAskForTheVersionWhenTheOnlyDependentRouteIsWithheld(t *testing.T) {
	result, proxy, _ := fbDiscoverProxy(t, map[string]http.HandlerFunc{
		"/model/info":  fbData(fbAzureAIRoute, `{"model_name":"gpt-6-astra","model_info":{"mode":"embedding"}}`),
		fbVersionProbe: fbVersionReply("1.103.0"),
	})

	fbExpectStrings(t, fbIDs(result.Models))
	if got := fbCount(proxy.Requests(), fbVersionProbe); got != 0 {
		t.Errorf("version probes = %d, want 0", got)
	}
}

func TestFallbacks_VersionGateAsksOnceWhenAnExactRouteAndAWildcardExpansionBothDependOnTheVersion(t *testing.T) {
	result, proxy, _ := fbDiscoverProxy(t, map[string]http.HandlerFunc{
		"/model/info":  fbData(fbAzureAIRoute, fbAzureAIWildcard),
		"/v1/models":   fbData(`{"id":"gpt-6-astra"}`, `{"id":"foundry/gpt-5.5"}`),
		fbVersionProbe: fbVersionReply("1.103.0"),
	})

	fbExpectStrings(t, fbIDAPIs(result.Models), "gpt-6-astra=openai-responses", "foundry/gpt-5.5=openai-responses")
	if got := fbCount(proxy.Requests(), fbVersionProbe); got != 1 {
		t.Errorf("version probes = %d, want 1", got)
	}
}

func TestFallbacks_VersionGateReportsTheVersionItReadAndNoneWhenItDidNotAsk(t *testing.T) {
	asked, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info":  fbData(fbAzureAIRoute),
		fbVersionProbe: fbVersionReply("1.103.0"),
	})
	notAsked, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info":  fbData(`{"model_name":"gpt","litellm_params":{"model":"openai/gpt-5"}}`),
		fbVersionProbe: fbVersionReply("1.103.0"),
	})

	fbEqual(t, asked.ProxyVersion, `{"major":1,"minor":103,"patch":0,"prerelease":false}`)
	if notAsked.ProxyVersion != nil {
		t.Errorf("proxyVersion = %+v, want none", notAsked.ProxyVersion)
	}
}

func TestFallbacks_VersionGateDoesNotAskForTheVersionWhenTheOnlyDependentHealthRouteIsWithheld(t *testing.T) {
	route := "withheld-health-azure-ai"
	result, proxy, _ := fbDiscoverProxy(t, map[string]http.HandlerFunc{
		"/model/info": fbStatus(404),
		"/v1/models":  fbStatus(404),
		"/health": fbHealth(`{"model":"`+route+`","model_id":"astra"}`, `{"model":"`+route+`","model_id":"embed"}`,
			`{"model":"safe-health-route"}`),
		"litellm_model_id=astra": fbData(fbAzureAIRow(route, "azure_ai/gpt-6-astra", fbAzureAIInfo)),
		"litellm_model_id=embed": fbData(`{"model_name":"` + route + `","model_info":{"mode":"embedding"}}`),
		fbVersionProbe:           fbVersionReply("1.103.0"),
	})

	fbExpectStrings(t, fbIDs(result.Models), "safe-health-route")
	if got := fbCount(proxy.Requests(), fbVersionProbe); got != 0 {
		t.Errorf("version probes = %d, want 0", got)
	}
}

func TestFallbacks_VersionGateAppliesTheProxyVersionToDeploymentDetailsFoundThroughHealth(t *testing.T) {
	for _, tc := range []struct {
		version string
		api     ai.API
	}{
		{"1.103.0", ai.APIOpenAIResponses},
		{"1.102.0", ai.APIOpenAICompletions},
	} {
		t.Run(tc.version, func(t *testing.T) {
			result, _ := fbDiscover(t, map[string]http.HandlerFunc{
				"/model/info":            fbStatus(404),
				"/v1/models":             fbStatus(404),
				"/health":                fbHealth(`{"model":"gpt-6-astra","model_id":"astra"}`),
				"litellm_model_id=astra": fbData(fbAzureAIRoute),
				fbVersionProbe:           fbVersionReply(tc.version),
			})

			if result.Source != types.SourceHealth {
				t.Errorf("source = %s", result.Source)
			}
			fbExpectStrings(t, fbIDAPIs(result.Models), "gpt-6-astra="+string(tc.api))
		})
	}
}

func TestFallbacks_VersionGateAppliesTheVersionToIDsExpandedFromAnAzureAIWildcardRoute(t *testing.T) {
	expand := func(version string) *types.DiscoveryResult {
		result, _ := fbDiscover(t, map[string]http.HandlerFunc{
			"/model/info":  fbData(fbAzureAIWildcard),
			"/v1/models":   fbData(`{"id":"foundry/gpt-6-astra"}`),
			fbVersionProbe: fbVersionReply(version),
		})
		return result
	}

	current := expand("1.103.0")
	older := expand("1.102.0")

	fbExpectStrings(t, fbIDAPIs(current.Models), "foundry/gpt-6-astra=openai-responses")
	fbExpectStrings(t, fbIDAPIs(older.Models), "foundry/gpt-6-astra=openai-completions")
}

// ---- discoverModels response-mode models ----

func TestFallbacks_ResponseModeRetainsUpstreamAutomaticAPIChoicesAndNeverSelectsMessages(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(
			`{"model_name":"anthropic/claude-sonnet-4-6","model_info":{"mode":"chat"}}`,
			`{"model_name":"openai/gpt-5.3-codex-openai","model_info":{"mode":"responses"}}`,
			`{"model_name":"unknown-mode","model_info":{"mode":"messages"}}`),
	})

	fbExpectStrings(t, fbIDAPIs(result.Models),
		"anthropic/claude-sonnet-4-6=openai-completions", "openai/gpt-5.3-codex-openai=openai-responses")
}

func TestFallbacks_ResponseModeKeepsModelInfoResponseModeModelsWithResponsesSpecificCompatibility(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(
			`{"model_name":"openai/gpt-5.3-codex-openai","model_info":{"mode":"responses","max_input_tokens":272000,"max_output_tokens":128000}}`,
			`{"model_name":"anthropic/claude-sonnet-4-6","model_info":{"mode":"responses"}}`,
			`{"model_name":"sonnet-4.6","model_info":{"mode":"responses"}}`),
	})

	if result.Source != types.SourceModelInfo || len(result.Models) != 3 {
		t.Fatalf("source = %s, models = %v", result.Source, fbIDs(result.Models))
	}
	fbMatch(t, result.Models[0], `{"id":"openai/gpt-5.3-codex-openai","api":"openai-responses","contextWindow":272000,"maxTokens":128000}`)
	for _, id := range []string{"anthropic/claude-sonnet-4-6", "sonnet-4.6"} {
		index := slices.Index(fbIDs(result.Models), id)
		if index < 0 {
			t.Fatalf("%s missing from %v", id, fbIDs(result.Models))
		}
		fbMatch(t, result.Models[index], `{"api":"openai-responses"}`)
		fbLacks(t, result.Models[index], "compat")
	}
}

func TestFallbacks_ResponseModeKeepsAHealthRouteOnChatWhenAnyDeploymentNeedsChatRegardlessOfOrder(t *testing.T) {
	detail := func(id, apiVersion string) string {
		version := ""
		if apiVersion != "" {
			version = `,"api_version":"` + apiVersion + `"`
		}
		return `{"model_name":"shared-route","litellm_params":{"model":"azure/gpt-5.5"` + version +
			`},"model_info":{"id":"` + id + `","litellm_provider":"azure"}}`
	}
	run := func(order ...string) types.DiscoveredModel {
		entries := make([]string, 0, len(order))
		for _, id := range order {
			entries = append(entries, `{"model":"azure/gpt-5.5","model_id":"`+id+`"}`)
		}
		result, _ := fbDiscover(t, map[string]http.HandlerFunc{
			"/model/info":                      fbStatus(404),
			"/v1/models":                       fbStatus(404),
			"/health":                          fbHealth(entries...),
			"/model/info?litellm_model_id=new": fbData(detail("new", "")),
			"/model/info?litellm_model_id=old": fbData(detail("old", "2024-10-21")),
		})
		if result.Source != types.SourceHealth {
			t.Errorf("source = %s", result.Source)
		}
		index := slices.Index(fbIDs(result.Models), "shared-route")
		if index < 0 {
			t.Fatalf("shared-route missing from %v", fbIDs(result.Models))
		}
		return result.Models[index]
	}

	fbMatch(t, run("new", "old"), `{"api":"openai-completions"}`)
	fbMatch(t, run("old", "new"), `{"api":"openai-completions"}`)
}

func TestFallbacks_ResponseModeDropsTheBackendFamilyWhenHealthDeploymentsOfOneRouteDisagree(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbStatus(404),
		"/v1/models":  fbStatus(404),
		"/health": fbHealth(`{"model":"openai/gpt-5.5","model_id":"gpt"}`,
			`{"model":"anthropic/claude-sonnet-4-6","model_id":"claude"}`),
		"/model/info?litellm_model_id=gpt": fbData(
			`{"model_name":"mixed-route","litellm_params":{"model":"openai/gpt-5.5"},"model_info":{"id":"gpt"}}`),
		"/model/info?litellm_model_id=claude": fbData(
			`{"model_name":"mixed-route","litellm_params":{"model":"anthropic/claude-sonnet-4-6"},"model_info":{"id":"claude"}}`),
	})

	index := slices.Index(fbIDs(result.Models), "mixed-route")
	if index < 0 {
		t.Fatalf("mixed-route missing from %v", fbIDs(result.Models))
	}
	fbMatch(t, result.Models[index], `{"api":"openai-completions"}`)
	fbLacks(t, result.Models[index], "litellmBackendFamily")
}

func TestFallbacks_ResponseModeKeepsHealthResponseModeModelInfoFallbacksWithAResponsesAPIOverride(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbStatus(404),
		"/v1/models":  fbStatus(404),
		"/health":     fbHealth(`{"model":"openai/gpt-5.3-codex-openai","model_id":"uuid-1"}`),
		"/model/info?litellm_model_id=uuid-1": fbData(
			`{"model_name":"openai/gpt-5.3-codex-openai","model_info":{"mode":"response"}}`),
	})

	if result.Source != types.SourceHealth || len(result.Models) != 1 {
		t.Fatalf("source = %s, models = %v", result.Source, fbIDs(result.Models))
	}
	fbMatch(t, result.Models[0], `{"id":"openai/gpt-5.3-codex-openai","api":"openai-responses"}`)
}

func TestFallbacks_ResponseModeDoesNotDeriveThinkingControlsFromAHealthOnlyRouteName(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info":                         fbStatus(404),
		"/v1/models":                          fbStatus(404),
		"/health":                             fbHealth(`{"model":"openai/gpt-5.5","model_id":"uuid-1"}`),
		"/model/info?litellm_model_id=uuid-1": fbData(`{"model_info":{"mode":"chat"}}`),
	})

	if result.Source != types.SourceHealth {
		t.Errorf("source = %s", result.Source)
	}
	fbLacks(t, result.Models[0], "thinkingLevelMap")
}

func TestFallbacks_ResponseModeUsesPiCatalogProtocolAndPresentationForAHealthEndpointWithoutDeploymentDetail(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbStatus(404),
		"/v1/models":  fbStatus(404),
		"/health":     fbHealth(`{"model":"openai/gpt-5.5"}`),
	})
	catalog := fbPiModel(t, "openai", "gpt-5.5")

	if result.Source != types.SourceHealth {
		t.Errorf("source = %s", result.Source)
	}
	fbMatch(t, result.Models[0], fmt.Sprintf(`{"id":"openai/gpt-5.5","name":%q,"api":"openai-responses","reasoning":%v,`+
		`"thinkingLevelMap":%s,"input":%s,"cost":%s,"contextWindow":%d,"maxTokens":%d}`,
		catalog.DisplayName, catalog.Reasoning, fbNoReasoningLevels, fbJSONString(t, fbPiInput(catalog)),
		fbJSONString(t, fbPiCost(catalog)), catalog.ContextWindow, catalog.MaxOutputTokens))
	fbLacks(t, result.Models[0], "litellmBackendFamily")
}

func TestFallbacks_ResponseModeKeepsAnUnknownHealthOnlyIDOnChat(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbStatus(404),
		"/v1/models":  fbStatus(404),
		"/health":     fbHealth(`{"model":"unknown-health-route"}`),
	})

	fbMatch(t, result.Models[0], `{"id":"unknown-health-route","name":"unknown-health-route (no metadata)","api":"openai-completions"}`)
}

func TestFallbacks_ResponseModeDiagnosesAHealthRouteThatMixesChatAndIncompatibleDeploymentModes(t *testing.T) {
	proxy := newMockProxy(t, map[string]http.HandlerFunc{
		"/model/info?litellm_model_id=chat-id":  fbData(`{"model_name":"health-mixed-mode","model_info":{"mode":"chat"}}`),
		"/model/info?litellm_model_id=embed-id": fbData(`{"model_name":"health-mixed-mode","model_info":{"mode":"embedding"}}`),
		"/model/info":                           fbStatus(404),
		"/v1/models":                            fbStatus(404),
		"/health": fbHealth(`{"model":"health-mixed-mode","model_id":"chat-id"}`,
			`{"model":"health-mixed-mode","model_id":"embed-id"}`),
	})
	options, stderr := testOptions(t)

	_, err := DiscoverModels(context.Background(), proxy.URL, "sk-test", options)

	if err == nil || !strings.Contains(err.Error(), "/v1/models returned 404") {
		t.Fatalf("err = %v, want it to contain %q", err, "/v1/models returned 404")
	}
	lines := stderr.Lines()
	if len(lines) != 1 {
		t.Fatalf("stderr lines = %q, want 1", lines)
	}
	for _, want := range []string{"health-mixed-mode", "explicitly incompatible deployment modes"} {
		if !strings.Contains(lines[0], want) {
			t.Errorf("message %q lacks %q", lines[0], want)
		}
	}
}

func TestFallbacks_ResponseModeUsesTheCorrelatedDetailModelNameInsteadOfTheHealthBackendModel(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info?litellm_model_id=uuid-redirect": fbData(`{"model_name":"different-route","litellm_params":{"model":"moonshot/kimi-k3",` +
			`"allowed_openai_params":["reasoning_effort"]},"model_info":{"id":"uuid-redirect","mode":"chat","supports_reasoning":true}}`),
		"/model/info": fbStatus(404),
		"/v1/models":  fbStatus(404),
		"/health":     fbHealth(`{"model":"authoritative-route","model_id":"uuid-redirect"}`),
	})

	fbMatch(t, result.Models[0], `{"id":"different-route","reasoning":true}`)
	fbEqual(t, result.Models[0].ThinkingLevelMap,
		`{"off":null,"minimal":null,"low":"low","medium":null,"high":"high","xhigh":null,"max":null}`)
}

func TestFallbacks_ResponseModeUsesCatalogProtocolForHealthEntriesWithoutDeploymentDetail(t *testing.T) {
	for _, tc := range []struct {
		route string
		api   ai.API
	}{
		{"openai/gpt-5.5", ai.APIOpenAIResponses},
		{"unknown/health-route", ai.APIOpenAICompletions},
	} {
		t.Run(tc.route, func(t *testing.T) {
			result, _ := fbDiscover(t, map[string]http.HandlerFunc{
				"/model/info": fbStatus(404),
				"/v1/models":  fbStatus(404),
				"/health":     fbHealth(`{"model":"` + tc.route + `"}`),
			})

			if result.Source != types.SourceHealth {
				t.Errorf("source = %s", result.Source)
			}
			model := result.Models[0]
			fbMatch(t, model, fmt.Sprintf(`{"id":%q,"api":%q}`, tc.route, tc.api))
			if model.Reasoning {
				fbEqual(t, model.ThinkingLevelMap, fbNoReasoningLevels)
			} else {
				fbLacks(t, model, "thinkingLevelMap")
			}
		})
	}
}

// ---- discoverModels fallback to /v1/models ----

func TestFallbacks_ModelsListKeepsUnqualifiedFallbackIDsBoundedInsteadOfScanningEveryProviderCatalog(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbStatus(403),
		"/v1/models":  fbData(`{"id":"gpt-4o"}`, `{"id":"kimi-k2.6"}`, `{"id":"grok-4.5"}`),
	})

	names := map[string]string{}
	for _, model := range result.Models {
		names[model.ID] = model.Name
	}
	for _, id := range []string{"gpt-4o", "kimi-k2.6", "grok-4.5"} {
		if names[id] != id+" (no metadata)" {
			t.Errorf("name of %s = %q, want %q", id, names[id], id+" (no metadata)")
		}
	}
}

func TestFallbacks_ModelsListFallsBackWhenModelInfoReturnsAnAuthOrMissingStatus(t *testing.T) {
	for _, status := range []int{401, 403, 404} {
		t.Run(fmt.Sprint(status), func(t *testing.T) {
			result, _ := fbDiscover(t, map[string]http.HandlerFunc{
				"/model/info": func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(status) },
				"/v1/models":  fbData(`{"id":"openai/gpt-4o"}`, `{"id":"anthropic/claude-3-5-sonnet"}`),
			})

			if result.Source != types.SourceModelsList {
				t.Errorf("source = %s", result.Source)
			}
			fbExpectStrings(t, slices.Sorted(slices.Values(fbIDs(result.Models))), "anthropic/claude-3-5-sonnet", "openai/gpt-4o")
			index := slices.Index(fbIDs(result.Models), "anthropic/claude-3-5-sonnet")
			anthropic := result.Models[index]
			if anthropic.Name != "anthropic/claude-3-5-sonnet (no metadata)" {
				t.Errorf("name = %q", anthropic.Name)
			}
			fbEqual(t, anthropic.Compat, `{"supportsStore":false,"cacheControlFormat":"anthropic"}`)
		})
	}
}

func TestFallbacks_ModelsListUsesPiCatalogMetadataForTheFallback(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbStatus(403),
		"/v1/models":  fbData(`{"id":"gpt-5.5","object":"model","owned_by":"openai"}`),
	})

	fbMatch(t, result.Models[0], `{"id":"gpt-5.5","name":"GPT-5.5","contextWindow":272000,"api":"openai-responses"}`)
}

func TestFallbacks_ModelsListKeepsAnOpaqueFallbackIDOnChat(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbStatus(403),
		"/v1/models":  fbData(`{"id":"opaque-fallback-route"}`),
	})

	fbMatch(t, result.Models[0], `{"id":"opaque-fallback-route","name":"opaque-fallback-route (no metadata)","api":"openai-completions"}`)
}

func TestFallbacks_ModelsListKeepsAnOpaqueFallbackIDOnChatCompletionsWithoutABackendFamily(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(401) },
		"/v1/models":  fbData(`{"id":"gpt-production","object":"model","owned_by":"openai"}`),
	})

	fbMatch(t, result.Models[0], `{"id":"gpt-production","name":"gpt-production (no metadata)","api":"openai-completions"}`)
	fbLacks(t, result.Models[0], "litellmBackendFamily")
}

func TestFallbacks_ModelsListEnrichesABareFable5FallbackModelFromThePiCatalogWithoutInferringACarrier(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbStatus(403),
		"/v1/models":  fbData(`{"id":"fable-5","object":"model","owned_by":"openai"}`),
	})

	fbMatch(t, result, `{"source":"models_list","models":[{"id":"fable-5","name":"Claude Fable 5","reasoning":true}]}`)
	fbEqual(t, result.Models[0].ThinkingLevelMap, fbNoReasoningLevels)
	fbNotMatch(t, result.Models[0].Compat, `{"supportsReasoningEffort":true}`)
}

func TestFallbacks_ModelsListEnrichesABareOpus5FallbackModelFromThePiCatalogWithoutInferringACarrier(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbStatus(403),
		"/v1/models":  fbData(`{"id":"opus-5","object":"model","owned_by":"openai"}`),
	})

	fbMatch(t, result, `{"source":"models_list","models":[{"id":"opus-5","name":"Claude Opus 5","reasoning":true}]}`)
	fbEqual(t, result.Models[0].ThinkingLevelMap, fbNoReasoningLevels)
	fbNotMatch(t, result.Models[0].Compat, `{"supportsReasoningEffort":true}`)
}

func TestFallbacks_ModelsListThrowsWhenModelInfoReturnsANon401403404Error(t *testing.T) {
	base := mockEndpoints(t, map[string]http.HandlerFunc{"/model/info": fbStatus(500)})
	options, _ := testOptions(t)

	_, err := DiscoverModels(context.Background(), base, "sk-test", options)

	if err == nil || !strings.Contains(err.Error(), "500") {
		t.Errorf("err = %v, want it to mention 500", err)
	}
}

func TestFallbacks_ModelsListUsesPiCatalogProtocolForFallbackEntriesWhileDenyingReasoningCarriers(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbStatus(404),
		"/v1/models":  fbData(`{"id":"openai/gpt-5.5","owned_by":"openai"}`),
	})

	fbMatch(t, result.Models[0], `{"api":"openai-responses","reasoning":true,"thinkingLevelMap":`+fbNoReasoningLevels+`}`)
	fbNotMatch(t, result.Models[0].Compat, `{"supportsReasoningEffort":true}`)
}

func TestFallbacks_ModelsListDeniesPiDefaultChatLevelsWhenCatalogMetadataHasNoLevelOrCarrierEvidence(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbStatus(403),
		"/v1/models":  fbData(`{"id":"claude-haiku-4-5","object":"model","owned_by":"anthropic"}`),
	})

	fbMatch(t, result, `{"source":"models_list","models":[{"id":"claude-haiku-4-5","reasoning":true,`+
		`"thinkingLevelMap":`+fbNoReasoningLevels+`}]}`)
}

// ---- discoverModels fallback to /health ----

func TestFallbacks_HealthFallbackBoundsDetailConcurrencyWhilePreservingEndpointOrderAndCompletionProgress(t *testing.T) {
	const total = 11
	entries := make([]string, total)
	ids := make([]string, total)
	for i := range total {
		entries[i] = fmt.Sprintf(`{"model":"model-%d","model_id":"uuid-%d"}`, i+1, i+1)
		ids[i] = fmt.Sprintf("model-%d", i+1)
	}
	var (
		mu        sync.Mutex
		pending   = map[string]chan struct{}{}
		active    int
		maxActive int
		progress  []string
	)
	stop := make(chan struct{})
	t.Cleanup(func() { close(stop) })
	detail := func(w http.ResponseWriter, r *http.Request) {
		deploymentID := r.URL.Query().Get("litellm_model_id")
		release := make(chan struct{})
		mu.Lock()
		active++
		maxActive = max(maxActive, active)
		pending[deploymentID] = release
		mu.Unlock()
		select {
		case <-release:
		case <-stop:
		case <-r.Context().Done():
		}
		mu.Lock()
		active--
		mu.Unlock()
		fbData(fmt.Sprintf(`{"model_name":"model-%s","model_info":{"mode":"chat"}}`,
			strings.TrimPrefix(deploymentID, "uuid-")))(w, r)
	}
	base := mockEndpoints(t, map[string]http.HandlerFunc{
		"/model/info":                         fbStatus(404),
		"/v1/models":                          fbStatus(404),
		"/health":                             fbHealth(entries...),
		"/model/info?litellm_model_id=uuid-1": detail, "/model/info?litellm_model_id=uuid-2": detail,
		"/model/info?litellm_model_id=uuid-3": detail, "/model/info?litellm_model_id=uuid-4": detail,
		"/model/info?litellm_model_id=uuid-5": detail, "/model/info?litellm_model_id=uuid-6": detail,
		"/model/info?litellm_model_id=uuid-7": detail, "/model/info?litellm_model_id=uuid-8": detail,
		"/model/info?litellm_model_id=uuid-9": detail, "/model/info?litellm_model_id=uuid-10": detail,
		"/model/info?litellm_model_id=uuid-11": detail,
	})
	options, _ := testOptions(t)
	options.OnProgress = func(message string) {
		mu.Lock()
		defer mu.Unlock()
		progress = append(progress, message)
	}
	waitFor := func(what string, condition func() bool) {
		t.Helper()
		deadline := time.Now().Add(5 * time.Second)
		for {
			mu.Lock()
			ok := condition()
			mu.Unlock()
			if ok {
				return
			}
			if time.Now().After(deadline) {
				t.Fatalf("timed out waiting for %s", what)
			}
			time.Sleep(2 * time.Millisecond)
		}
	}
	release := func(deploymentID string) {
		t.Helper()
		mu.Lock()
		channel, ok := pending[deploymentID]
		delete(pending, deploymentID)
		mu.Unlock()
		if !ok {
			t.Fatalf("%s is not pending", deploymentID)
		}
		close(channel)
	}
	has := func(deploymentID string) func() bool {
		return func() bool { _, ok := pending[deploymentID]; return ok }
	}
	type outcome struct {
		result *types.DiscoveryResult
		err    error
	}
	done := make(chan outcome, 1)
	go func() {
		result, err := DiscoverModels(context.Background(), base, "sk-test", options)
		done <- outcome{result, err}
	}()

	waitFor("the first eight detail requests", func() bool {
		keys := make([]string, 0, len(pending))
		for key := range pending {
			keys = append(keys, key)
		}
		slices.Sort(keys)
		want := []string{"uuid-1", "uuid-2", "uuid-3", "uuid-4", "uuid-5", "uuid-6", "uuid-7", "uuid-8"}
		return slices.Equal(keys, want)
	})
	mu.Lock()
	if maxActive != 8 {
		t.Errorf("maxActive = %d, want 8", maxActive)
	}
	mu.Unlock()

	release("uuid-8")
	waitFor("uuid-9", has("uuid-9"))
	release("uuid-7")
	waitFor("uuid-10", has("uuid-10"))
	release("uuid-6")
	waitFor("uuid-11", has("uuid-11"))
	mu.Lock()
	if slices.Contains(progress, "Fetched 10/11 models...") {
		t.Error("progress reported 10/11 too early")
	}
	if maxActive != 8 {
		t.Errorf("maxActive = %d, want 8", maxActive)
	}
	mu.Unlock()

	for _, deploymentID := range []string{"uuid-11", "uuid-10", "uuid-9", "uuid-5", "uuid-4", "uuid-3", "uuid-2", "uuid-1"} {
		release(deploymentID)
	}
	var finished outcome
	select {
	case finished = <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("discovery did not finish")
	}
	if finished.err != nil {
		t.Fatal(finished.err)
	}

	fbExpectStrings(t, fbIDs(finished.result.Models), ids...)
	mu.Lock()
	defer mu.Unlock()
	for _, want := range []string{"Fetched 10/11 models...", "Fetched 11/11 models..."} {
		if !slices.Contains(progress, want) {
			t.Errorf("progress %q lacks %q", progress, want)
		}
	}
}

func TestFallbacks_HealthFallbackUsesHealthAndPerEndpointModelInfoWhenModelListingIsUnavailable(t *testing.T) {
	result, proxy, _ := fbDiscoverProxy(t, map[string]http.HandlerFunc{
		"/model/info": fbStatus(404),
		"/v1/models":  fbStatus(404),
		"/health": fbHealth(`{"model":"vertex/claude-sonnet","model_id":"uuid-1"}`,
			`{"model":"openai/gpt-4o-mini","model_id":"uuid-2"}`),
		"/model/info?litellm_model_id=uuid-1": fbData(`{"model_name":"vertex/claude-sonnet","model_info":{"mode":"chat",` +
			`"max_input_tokens":200000,"supports_vision":true,"input_cost_per_token":0.000003,"output_cost_per_token":0.000015}}`),
		"/model/info?litellm_model_id=uuid-2": fbData(`{"model_name":"openai/gpt-4o-mini","model_info":{"mode":"chat",` +
			`"max_input_tokens":128000,"max_output_tokens":16384}}`),
	})

	requests := proxy.Requests()
	if len(requests) != 5 {
		t.Fatalf("requests = %v", requests)
	}
	fbExpectStrings(t, requests[:3], "/model/info", "/v1/models", "/health")
	// The two detail fetches run on concurrent workers, so only their set is fixed.
	fbExpectStrings(t, slices.Sorted(slices.Values(requests[3:])),
		"/model/info?litellm_model_id=uuid-1", "/model/info?litellm_model_id=uuid-2")
	if result.Source != types.SourceHealth {
		t.Errorf("source = %s", result.Source)
	}
	fbExpectStrings(t, fbIDs(result.Models), "vertex/claude-sonnet", "openai/gpt-4o-mini")
	fbMatch(t, result.Models[0], `{"input":["text","image"],"contextWindow":200000,`+
		`"compat":{"supportsStore":false,"cacheControlFormat":"anthropic"}}`)
}

func TestFallbacks_HealthFallbackUsesHealthyEndpointModelNamesWhenEntriesIncludeNoModelIDs(t *testing.T) {
	result, proxy, _ := fbDiscoverProxy(t, map[string]http.HandlerFunc{
		"/model/info": fbStatus(404),
		"/v1/models":  fbStatus(404),
		"/health": fbHealth(`{"model":"azure/gpt-35-turbo","api_base":"https://azure.example.com"}`,
			`{"model":"anthropic/claude-3-5-sonnet","api_base":"https://anthropic.example.com"}`,
			`{"model":"openai/gpt-5.5","api_base":"https://openai.example.com"}`),
	})

	fbExpectStrings(t, proxy.Requests(), "/model/info", "/v1/models", "/health")
	if result.Source != types.SourceHealth {
		t.Errorf("source = %s", result.Source)
	}
	fbExpectStrings(t, fbIDs(result.Models), "azure/gpt-35-turbo", "anthropic/claude-3-5-sonnet", "openai/gpt-5.5")
	// Neither route resolves in the Pi catalog, so both are evidence-free and must say so rather than
	// presenting default limits and zero cost as fact.
	fbMatch(t, result.Models[1], `{"name":"anthropic/claude-3-5-sonnet (no metadata)","contextWindow":128000,"maxTokens":16384,`+
		`"compat":{"supportsStore":false,"cacheControlFormat":"anthropic"}}`)
	if name := result.Models[0].Name; name != "azure/gpt-35-turbo (no metadata)" {
		t.Errorf("name = %q", name)
	}
}

func TestFallbacks_HealthFallbackUsesBoundedPiMetadataForHealthOnlyRoutes(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbStatus(404),
		"/v1/models":  fbStatus(404),
		"/health":     fbHealth(`{"model":"totally-unknown-route"}`, `{"model":"anthropic/claude-opus-4-7"}`),
	})

	if result.Source != types.SourceHealth {
		t.Errorf("source = %s", result.Source)
	}
	unresolved, resolved := result.Models[0], result.Models[1]
	fbMatch(t, unresolved, `{"id":"totally-unknown-route","name":"totally-unknown-route (no metadata)",`+
		`"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0},"contextWindow":128000,"maxTokens":16384}`)
	if !strings.Contains(unresolved.Name, " (no metadata)") {
		t.Errorf("name = %q", unresolved.Name)
	}
	catalog := fbPiModel(t, "anthropic", "claude-opus-4-7")
	fbMatch(t, resolved, fmt.Sprintf(`{"name":%q,"api":"openai-completions","cost":%s}`,
		catalog.DisplayName, fbJSONString(t, fbPiCost(catalog))))
}

func TestFallbacks_HealthFallbackKeepsMixedDetailedAndBareHealthRowsConservative(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbStatus(404),
		"/v1/models":  fbStatus(404),
		"/health":     fbHealth(`{"model":"openai/gpt-5.5","model_id":"detail"}`, `{"model":"openai/gpt-5.5"}`),
		"/model/info?litellm_model_id=detail": fbData(
			`{"model_name":"openai/gpt-5.5","litellm_params":{"model":"private/unknown"}}`),
	})

	fbMatch(t, result.Models[0], `{"name":"openai/gpt-5.5 (incomplete metadata)","reasoning":false,"contextWindow":128000,`+
		`"maxTokens":16384,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0}}`)
}

func TestFallbacks_HealthFallbackMarksAnUnresolvedHealthRouteReachedThroughPerEndpointModelInfo(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info?litellm_model_id=uuid-1": fbData(`{"model_info":{"mode":"chat"}}`),
		"/model/info":                         fbStatus(404),
		"/v1/models":                          fbStatus(404),
		"/health":                             fbHealth(`{"model":"vertex/claude-sonnet","model_id":"uuid-1"}`),
	})

	if result.Source != types.SourceHealth {
		t.Errorf("source = %s", result.Source)
	}
	if name := result.Models[0].Name; name != "vertex/claude-sonnet (incomplete metadata)" {
		t.Errorf("name = %q", name)
	}
	fbLacks(t, result.Models[0], "thinkingLevelMap")
}

func TestFallbacks_HealthFallbackGroupsHealthyDeploymentsByRouteBeforePublication(t *testing.T) {
	for _, endpointIDs := range [][]string{{"uuid-roomy", "uuid-cramped"}, {"uuid-cramped", "uuid-roomy"}} {
		t.Run(strings.Join(endpointIDs, ","), func(t *testing.T) {
			entries := make([]string, 0, len(endpointIDs))
			for _, id := range endpointIDs {
				entries = append(entries, `{"model":"shared-health-route","model_id":"`+id+`"}`)
			}
			result, _ := fbDiscover(t, map[string]http.HandlerFunc{
				"/model/info?litellm_model_id=uuid-roomy": fbData(`{"model_name":"shared-health-route","litellm_params":{"model":"openai/gpt-4o"},` +
					`"model_info":{"id":"roomy","mode":"responses","supports_reasoning":true,"supports_vision":true,` +
					`"max_input_tokens":128000,"max_output_tokens":16384,"input_cost_per_token":0.000005,` +
					`"output_cost_per_token":0.000015,"cache_read_input_token_cost":0.0000025,"cache_creation_input_token_cost":0}}`),
				"/model/info?litellm_model_id=uuid-cramped": fbData(`{"model_name":"shared-health-route","litellm_params":{"model":"internal/unknown"},` +
					`"model_info":{"id":"cramped","mode":"chat","supports_reasoning":false,"supports_vision":false,` +
					`"max_input_tokens":64000,"max_output_tokens":8192}}`),
				"/model/info": fbStatus(404),
				"/v1/models":  fbStatus(404),
				"/health":     fbHealth(entries...),
			})

			if result.Source != types.SourceHealth || len(result.Models) != 1 {
				t.Fatalf("source = %s, models = %v", result.Source, fbIDs(result.Models))
			}
			fbMatch(t, result.Models[0], `{"id":"shared-health-route","name":"shared-health-route (incomplete metadata)",`+
				`"api":"openai-completions","reasoning":false,"input":["text"],"contextWindow":64000,"maxTokens":8192,`+
				`"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0}}`)
			fbLacks(t, result.Models[0], "thinkingLevelMap")
		})
	}
}

func TestFallbacks_HealthFallbackGroupsHealthBackendsThatCorrelateToOneDetailModelName(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info?litellm_model_id=bedrock-a": fbData(`{"model_name":"claude-opus-5","litellm_params":{"model":"bedrock/us.anthropic.claude-opus-5-a"},` +
			`"model_info":{"id":"bedrock-a","mode":"chat"}}`),
		"/model/info?litellm_model_id=bedrock-b": fbData(`{"model_name":"claude-opus-5","litellm_params":{"model":"bedrock/eu.anthropic.claude-opus-5-b"},` +
			`"model_info":{"id":"bedrock-b","mode":"chat"}}`),
		"/model/info": fbStatus(404),
		"/v1/models":  fbStatus(404),
		"/health": fbHealth(`{"model":"bedrock/us.anthropic.claude-opus-5","model_id":"bedrock-a"}`,
			`{"model":"bedrock/eu.anthropic.claude-opus-5","model_id":"bedrock-b"}`),
	})

	if result.Source != types.SourceHealth || len(result.Models) != 1 {
		t.Fatalf("source = %s, models = %v", result.Source, fbIDs(result.Models))
	}
	if result.Models[0].ID != "claude-opus-5" {
		t.Errorf("id = %s", result.Models[0].ID)
	}
}

// ---- native Messages discovery ----

func TestFallbacks_NativeMessagesTreatsNativeReasoningFlagsAsBooleanEvidence(t *testing.T) {
	for _, tc := range []struct {
		name     string
		flags    []string // JSON literal per row; "" omits the property (undefined)
		low, max bool
	}{
		{"[null]", []string{"null"}, true, true},
		{"[true, undefined]", []string{"true", ""}, true, false},
		{"[true, null]", []string{"true", "null"}, true, false},
		{"[null, undefined]", []string{"null", ""}, true, true},
		{"[false, undefined]", []string{"false", ""}, false, false},
		{"[true, true]", []string{"true", "true"}, true, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			rows := make([]string, len(tc.flags))
			for index, flag := range tc.flags {
				flags := ""
				if flag != "" {
					flags = fmt.Sprintf(`,"supports_low_reasoning_effort":%s,"supports_max_reasoning_effort":%s`, flag, flag)
				}
				rows[index] = fmt.Sprintf(`{"model_name":"native-flag-evidence","litellm_params":{"model":"anthropic/claude-sonnet-4-6"},`+
					`"model_info":{"id":"%d","mode":"chat","litellm_provider":"anthropic"%s}}`, index, flags)
			}
			result, _ := fbDiscover(t, map[string]http.HandlerFunc{"/model/info": fbData(rows...)})

			model := result.Models[0]
			if model.API != ai.APIAnthropicMessages {
				t.Fatalf("api = %s, want anthropic-messages", model.API)
			}
			levels := fbSupportedLevels(model)
			if got := slices.Contains(levels, ai.ThinkingLow); got != tc.low {
				t.Errorf("levels %v include low = %v, want %v", levels, got, tc.low)
			}
			if got := slices.Contains(levels, ai.ThinkingMax); got != tc.max {
				t.Errorf("levels %v include max = %v, want %v", levels, got, tc.max)
			}
		})
	}
}

func TestFallbacks_NativeMessagesHonorsDeniedDefaultNativeReasoningLevels(t *testing.T) {
	for _, backend := range []string{"claude-sonnet-4-6", "claude-opus-4-5"} {
		t.Run(backend, func(t *testing.T) {
			result, _ := fbDiscover(t, map[string]http.HandlerFunc{
				"/model/info": fbData(`{"model_name":"native-levels","litellm_params":{"model":"anthropic/` + backend + `"},` +
					`"model_info":{"mode":"chat","litellm_provider":"anthropic","supports_none_reasoning_effort":false,` +
					`"supports_low_reasoning_effort":false}}`),
			})

			model := result.Models[0]
			if model.API != ai.APIAnthropicMessages {
				t.Fatalf("api = %s, want anthropic-messages", model.API)
			}
			levels := fbSupportedLevels(model)
			if slices.Contains(levels, ai.ThinkingOff) || slices.Contains(levels, ai.ThinkingLow) || !slices.Contains(levels, ai.ThinkingMedium) {
				t.Errorf("levels = %v, want no off/low and medium", levels)
			}
		})
	}
}

func TestFallbacks_NativeMessagesRetainsNativeReasoningDenialsAcrossDifferentClaudeGenerations(t *testing.T) {
	rows := make([]string, 0, 2)
	for _, backend := range []string{"claude-sonnet-4-6", "claude-fable-5"} {
		rows = append(rows, `{"model_name":"mixed-native-levels","litellm_params":{"model":"anthropic/`+backend+`"},`+
			`"model_info":{"id":"`+backend+`","mode":"chat","litellm_provider":"anthropic","supports_reasoning":true}}`)
	}
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{"/model/info": fbData(rows...)})

	model := result.Models[0]
	if model.API != ai.APIAnthropicMessages {
		t.Fatalf("api = %s, want anthropic-messages", model.API)
	}
	levels := fbSupportedLevels(model)
	if slices.Contains(levels, ai.ThinkingOff) || !slices.Contains(levels, ai.ThinkingHigh) {
		t.Errorf("levels = %v, want no off and high", levels)
	}
}

func TestFallbacks_NativeMessagesUsesAnthropicCatalogMetadataForTheStandardVertexClaudeAdapter(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(`{"model_name":"vertex-claude","litellm_params":{"model":"vertex_ai/claude-opus-4-5"},` +
			`"model_info":{"mode":"chat","litellm_provider":"vertex_ai-anthropic_models"}}`),
	})

	if len(result.Models) != 1 {
		t.Fatalf("models = %v, want 1", fbIDs(result.Models))
	}
	fbContaining(t, result.Models[0], `{"id":"vertex-claude","name":"vertex-claude","reasoning":true,"contextWindow":200000,`+
		`"maxTokens":64000,"cost":{"input":5,"output":25,"cacheRead":0.5,"cacheWrite":6.25},"api":"anthropic-messages"}`)
}

func TestFallbacks_NativeMessagesUsesAnthropicCatalogMetadataForTheVertexAIAdapterAlias(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(`{"model_name":"vertex-claude","litellm_params":{"model":"vertex_ai/claude-opus-4-5"},` +
			`"model_info":{"mode":"chat","litellm_provider":"vertex_ai"}}`),
	})

	fbMatch(t, result.Models[0], `{"id":"vertex-claude","name":"vertex-claude","reasoning":true,"contextWindow":200000,`+
		`"maxTokens":64000,"cost":{"input":5,"output":25,"cacheRead":0.5,"cacheWrite":6.25},"api":"anthropic-messages"}`)
}

// fbResolved resolves catalog evidence for a row and requires that some evidence exists, as
// `expect(resolved).not.toHaveProperty(...)` requires a defined value.
func fbResolved(t *testing.T, row string) *modelgroups.CatalogResolution {
	t.Helper()
	resolved := ResolveModelInfoCatalog(entryFrom(t, row), nil)
	if resolved == nil {
		t.Fatalf("no catalog resolution for %s", row)
	}
	return resolved
}

func TestFallbacks_NativeMessagesWithholdsMessagesWhenTheVertexAnthropicAdapterConflictsWithTheRoutedProvider(t *testing.T) {
	resolved := fbResolved(t, `{"model_name":"contradictory-adapter","litellm_params":{"model":"openai/gpt-4o"},`+
		`"model_info":{"mode":"chat","litellm_provider":"vertex_ai-anthropic_models"}}`)

	if resolved.MessagesCompat != nil {
		t.Errorf("messagesCompat = %+v, want none", resolved.MessagesCompat)
	}
}

func TestFallbacks_NativeMessagesWithholdsMessagesWhenARecognizedAdapterConflictsWithTheRoutedProvider(t *testing.T) {
	resolved := fbResolved(t, `{"model_name":"contradictory-adapter","litellm_params":{"model":"anthropic/claude-sonnet-4-6"},`+
		`"model_info":{"mode":"chat","litellm_provider":"openai"}}`)

	if resolved.MessagesCompat != nil {
		t.Errorf("messagesCompat = %+v, want none", resolved.MessagesCompat)
	}
}

func TestFallbacks_NativeMessagesWithholdsMessagesPolicyForConflictingClaudeGenerations(t *testing.T) {
	resolved := fbResolved(t, `{"model_name":"contradictory-claude-generations","litellm_params":{"model":"anthropic/claude-opus-4-7"},`+
		`"model_info":{"mode":"chat","litellm_provider":"anthropic","base_model":"anthropic/claude-opus-4-5"}}`)

	if resolved.MessagesCompat != nil {
		t.Errorf("messagesCompat = %+v, want none", resolved.MessagesCompat)
	}
}

func TestFallbacks_NativeMessagesWithholdsMessagesWhenAnyDeclaredClaudeBackendLacksACompatiblePolicy(t *testing.T) {
	resolved := fbResolved(t, `{"model_name":"partially-resolved-claude-generations","litellm_params":{"model":"anthropic/claude-opus-4-7"},`+
		`"model_info":{"mode":"chat","litellm_provider":"anthropic","base_model":"anthropic/claude-future-9-9"}}`)

	if resolved.Provider != "anthropic" || resolved.CatalogModelID != "anthropic/claude-future-9-9" {
		t.Errorf("resolved = %+v", resolved)
	}
	if resolved.Cost != nil || resolved.Reasoning != nil || resolved.ContextWindow != nil || resolved.MaxTokens != nil ||
		resolved.MessagesCompat != nil {
		t.Errorf("resolved = %+v, want no cost, reasoning, limits or messagesCompat", resolved)
	}
}

func TestFallbacks_NativeMessagesDowngradesHealthDerivedMessagesWhenAnUnreadableDetailNameUsesTheRouteFallback(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbStatus(404),
		"/v1/models":  fbStatus(404),
		"/health":     fbHealth(`{"model":"team-claude","model_id":"messages-uuid"}`),
		"/model/info?litellm_model_id=messages-uuid": fbData(`{"model_name":7,"model_info":{"mode":"chat","litellm_provider":"anthropic"},` +
			`"litellm_params":{"model":"anthropic/claude-opus-5"}}`),
	})

	if len(result.Models) != 1 {
		t.Fatalf("models = %v, want 1", fbIDs(result.Models))
	}
	fbContaining(t, result.Models[0], `{"id":"team-claude","api":"openai-completions","compat":{"supportsStore":false,`+
		`"supportsReasoningEffort":false,"cacheControlFormat":"anthropic"}}`)
}

func TestFallbacks_NativeMessagesRechecksChatReasoningLevelsWhenHealthDiscoveryDowngradesMessages(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbStatus(404),
		"/v1/models":  fbStatus(404),
		"/health":     fbHealth(`{"model":"claude-route","model_id":"claude-id"}`),
		"/model/info?litellm_model_id=claude-id": fbData(`{"model_name":"claude-route","litellm_params":{"model":"anthropic/claude-opus-5"},` +
			`"model_info":{"mode":"chat","litellm_provider":"anthropic","supports_reasoning":true,` +
			`"supports_low_reasoning_effort":true,"supported_openai_params":["reasoning_effort"]}}`),
	})

	model := result.Models[0]
	fbMatch(t, model, `{"api":"openai-completions","compat":{"supportsReasoningEffort":true}}`)
	levels := fbSupportedLevels(model)
	if !slices.Contains(levels, ai.ThinkingLow) || slices.Contains(levels, ai.ThinkingXHigh) || slices.Contains(levels, ai.ThinkingMax) {
		t.Errorf("levels = %v, want low without xhigh or max", levels)
	}
}

func TestFallbacks_NativeMessagesNeverSelectsNativeMessagesFromHealthEvenWithCompleteMatchingDetail(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info?litellm_model_id=uuid-claude": fbData(`{"model_name":"claude-route","model_info":{"id":"uuid-claude","mode":"chat",` +
			`"litellm_provider":"anthropic","supported_openai_params":[],"supports_reasoning":true,"supports_high_reasoning_effort":true},` +
			`"litellm_params":{"model":"anthropic/claude-sonnet-4-6"}}`),
		"/model/info": fbStatus(404),
		"/v1/models":  fbStatus(404),
		"/health":     fbHealth(`{"model":"claude-route","model_id":"uuid-claude"}`),
	})

	if result.Source != types.SourceHealth {
		t.Errorf("source = %s", result.Source)
	}
	fbMatch(t, result.Models[0], `{"id":"claude-route","api":"openai-completions","compat":{"supportsStore":false,`+
		`"cacheControlFormat":"anthropic"},"thinkingLevelMap":`+fbNoReasoningLevels+`}`)
	if levels := fbSupportedLevels(result.Models[0]); len(levels) != 0 {
		t.Errorf("levels = %v, want none", levels)
	}
}

func TestFallbacks_NativeMessagesUsesClaudeBaseModelEvidenceWhenEveryDeclaredAdapterTargetResolvesTheSamePolicy(t *testing.T) {
	resolved := fbResolved(t, `{"model_name":"qualified-route","litellm_params":{"model":"bedrock/us.anthropic.claude-sonnet-4-6-v1:0"},`+
		`"model_info":{"mode":"chat","litellm_provider":"bedrock","base_model":"bedrock/us.anthropic.claude-sonnet-4-6-v1:0"}}`)

	if resolved.MessagesCompat == nil {
		t.Error("messagesCompat is missing")
	}
}

// The shape LiteLLM v1.100 reports for Bedrock-served Claude: its model map has no endpoint list for the
// model, so `get_model_info` fills `supported_endpoints` with null.
func TestFallbacks_NativeMessagesSelectsNativeMessagesForBedrockClaudeReportedWithNullSupportedEndpoints(t *testing.T) {
	result, _ := fbDiscover(t, map[string]http.HandlerFunc{
		"/model/info": fbData(`{"model_name":"claude-sonnet-4-6","litellm_params":{"model":"bedrock/us.anthropic.claude-sonnet-4-6-v1:0"},` +
			`"model_info":{"mode":"chat","litellm_provider":"bedrock_converse","base_model":"us.anthropic.claude-sonnet-4-6-v1:0",` +
			`"supported_endpoints":null}}`),
	})

	if api := result.Models[0].API; api != ai.APIAnthropicMessages {
		t.Errorf("api = %s, want anthropic-messages", api)
	}
}

// ---- Moonshot transport suppression ----

func TestFallbacks_MoonshotKeepsRequestSideReasoningSuppressionTransportScoped(t *testing.T) {
	for _, tc := range []struct {
		name, entry string
		suppress    bool
	}{
		{"Azure-hosted Kimi", `{"model_name":"kimi-k3","litellm_params":{"model":"azure/FW-Kimi-K3"},"model_info":{"mode":"chat",` +
			`"base_model":"fireworks/accounts/fireworks/models/kimi-k3","litellm_provider":"azure"}}`, false},
		{"Bedrock-hosted Kimi", `{"model_name":"moonshotai-kimi-k2-5","litellm_params":{"model":"bedrock/moonshotai.kimi-k2.5"},` +
			`"model_info":{"mode":"chat","base_model":"moonshotai.kimi-k2.5","litellm_provider":"bedrock_converse"}}`, false},
		{"opaque Moonshot route", `{"model_name":"k3-prod","litellm_params":{"model":"moonshot/kimi-k2.5"},"model_info":{"mode":"chat"}}`, true},
		{"conflicting Moonshot provider and Azure backend", `{"model_name":"kimi-prod","litellm_params":{"custom_llm_provider":"moonshot",` +
			`"model":"azure_ai/FW-Kimi-K3"},"model_info":{"mode":"chat"}}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			result, _ := fbDiscover(t, map[string]http.HandlerFunc{"/model/info": fbData(tc.entry)})

			policy := result.Models[0].LiteLLMPolicy
			if got := policy != nil && policy.SuppressReasoningVisibility; got != tc.suppress {
				t.Errorf("suppressReasoningVisibility = %v, want %v", got, tc.suppress)
			}
		})
	}
}
