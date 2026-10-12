package discover

// Ports the second half of the "discoverModels via /model/info" describe block of tests/discover.test.ts
// (from "withholds a group whose duplicate deployment id carries conflicting backends" to the end of the
// block). Helpers carry the modelInfoB prefix so they cannot collide with other test files.

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"slices"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

// modelInfoBProbe answers the proxy-version probe with no header: the Vitest mocks make that fetch
// throw, and both mean "version unknown".
const modelInfoBProbe = "/v1/responses/resp_version_probe"

func modelInfoBEndpoints(t *testing.T, routes map[string]http.HandlerFunc) string {
	t.Helper()
	if _, ok := routes[modelInfoBProbe]; !ok {
		routes[modelInfoBProbe] = jsonResponse(404, map[string]any{})
	}
	return mockEndpoints(t, routes)
}

// modelInfoBServeAll is `vi.spyOn(fetch).mockResolvedValue(...)`: every URL gets the same reply.
func modelInfoBServeAll(t *testing.T, status int, body any) string {
	t.Helper()
	handler := jsonResponse(status, body)
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	return server.URL
}

func modelInfoBRows(t *testing.T, rows ...string) string {
	t.Helper()
	return modelInfoBEndpoints(t, map[string]http.HandlerFunc{"/model/info": jsonResponse(200, modelInfoBody(rows...))})
}

// modelInfoBDiscover discovers rows served from /model/info with offline options.
func modelInfoBDiscover(t *testing.T, rows ...string) (*types.DiscoveryResult, *reports) {
	t.Helper()
	options, collected := testOptions(t)
	return discover(t, modelInfoBRows(t, rows...), options), collected
}

func modelInfoBFirst(t *testing.T, result *types.DiscoveryResult) types.DiscoveredModel {
	t.Helper()
	if len(result.Models) == 0 {
		t.Fatal("no models discovered")
	}
	return result.Models[0]
}

func modelInfoBFind(t *testing.T, result *types.DiscoveryResult, id string) types.DiscoveredModel {
	t.Helper()
	for _, model := range result.Models {
		if model.ID == id {
			return model
		}
	}
	t.Fatalf("model %q not discovered; have %v", id, modelInfoBIDs(result.Models))
	return types.DiscoveredModel{}
}

func modelInfoBIDs(models []types.DiscoveredModel) []string {
	ids := make([]string, 0, len(models))
	for _, model := range models {
		ids = append(ids, model.ID)
	}
	return ids
}

// modelInfoBLevels builds a ThinkingLevelMap from off, minimal, low, medium, high, xhigh, max values;
// "" is an explicit null.
func modelInfoBLevels(values ...string) ai.ThinkingLevelMap {
	names := []ai.ThinkingLevel{ai.ThinkingOff, ai.ThinkingMinimal, ai.ThinkingLow, ai.ThinkingMedium, ai.ThinkingHigh, ai.ThinkingXHigh, ai.ThinkingMax}
	levels := ai.ThinkingLevelMap{}
	for i, value := range values {
		if value == "" {
			levels[names[i]] = nil
			continue
		}
		levels[names[i]] = &value
	}
	return levels
}

func modelInfoBNoLevels() ai.ThinkingLevelMap { return modelInfoBLevels("", "", "", "", "", "", "") }

func modelInfoBLevelsEqual(t *testing.T, got, want ai.ThinkingLevelMap) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Errorf("thinkingLevelMap = %s, want %s", modelInfoBJSON(got), modelInfoBJSON(want))
	}
}

func modelInfoBJSON(value any) string {
	encoded, _ := json.Marshal(value)
	return string(encoded)
}

// modelInfoBMatch is toMatchObject: every key of want must be present in got with a matching value;
// arrays must match element for element.
func modelInfoBMatch(t *testing.T, got any, want string) {
	t.Helper()
	var gotValue, wantValue any
	if err := json.Unmarshal([]byte(modelInfoBJSON(got)), &gotValue); err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal([]byte(want), &wantValue); err != nil {
		t.Fatalf("want %s: %v", want, err)
	}
	if problem := modelInfoBSubset("$", gotValue, wantValue); problem != "" {
		t.Errorf("%s\n got: %s\nwant: %s", problem, modelInfoBJSON(got), want)
	}
}

func modelInfoBSubset(path string, got, want any) string {
	switch wantTyped := want.(type) {
	case map[string]any:
		gotMap, ok := got.(map[string]any)
		if !ok {
			return path + " is not an object"
		}
		for key, wantField := range wantTyped {
			gotField, present := gotMap[key]
			if !present {
				return path + "." + key + " is missing"
			}
			if problem := modelInfoBSubset(path+"."+key, gotField, wantField); problem != "" {
				return problem
			}
		}
		return ""
	case []any:
		gotSlice, ok := got.([]any)
		if !ok || len(gotSlice) != len(wantTyped) {
			return path + " differs in length or type"
		}
		for i := range wantTyped {
			if problem := modelInfoBSubset(fmt.Sprintf("%s[%d]", path, i), gotSlice[i], wantTyped[i]); problem != "" {
				return problem
			}
		}
		return ""
	}
	if !reflect.DeepEqual(got, want) {
		return fmt.Sprintf("%s = %v, want %v", path, got, want)
	}
	return ""
}

// modelInfoBHasKey is `toHaveProperty` on the serialized value.
func modelInfoBHasKey(value any, key string) bool {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal([]byte(modelInfoBJSON(value)), &fields); err != nil {
		return false
	}
	_, ok := fields[key]
	return ok
}

func modelInfoBBool(p *bool, want bool) bool { return p != nil && *p == want }

func modelInfoBPolicyEquals(t *testing.T, got *types.LiteLLMModelPolicy, want types.LiteLLMModelPolicy) {
	t.Helper()
	if got == nil || *got != want {
		t.Errorf("litellmPolicy = %s, want %s", modelInfoBJSON(got), modelInfoBJSON(want))
	}
}

func modelInfoBSuppresses(model types.DiscoveredModel) bool {
	return model.LiteLLMPolicy != nil && model.LiteLLMPolicy.SuppressReasoningVisibility
}

func modelInfoBIncomplete(t *testing.T, model types.DiscoveredModel, id string) {
	t.Helper()
	modelInfoBMatch(t, model, fmt.Sprintf(`{"id":%q,"name":%q,"reasoning":false,"input":["text"],`+
		`"contextWindow":128000,"maxTokens":16384,"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0}}`,
		id, id+" (incomplete metadata)"))
	if model.ThinkingLevelMap != nil {
		t.Errorf("thinkingLevelMap = %s, want none", modelInfoBJSON(model.ThinkingLevelMap))
	}
}

func modelInfoBChatCompat() string {
	return `{"supportsStore":false,"supportsDeveloperRole":false,"supportsReasoningEffort":false,` +
		`"supportsStrictMode":false,"maxTokensField":"max_tokens"}`
}

func modelInfoBCatalogModel(t *testing.T, provider, id string) *ai.GeneratedModel {
	t.Helper()
	model := findCatalogModelInProvider(provider, []string{id})
	if model == nil {
		t.Fatalf("Pi catalog no longer lists %s/%s", provider, id)
	}
	return model
}

func modelInfoBMatchCatalog(t *testing.T, got types.DiscoveredModel, catalog *ai.GeneratedModel) {
	t.Helper()
	wantCost := [4]float64{catalog.InputCostPerMTokens, catalog.OutputCostPerMTokens, catalog.CacheReadCost, catalog.CacheWriteCost}
	gotCost := [4]float64{got.Cost.Input, got.Cost.Output, got.Cost.CacheRead, got.Cost.CacheWrite}
	if got.Reasoning != catalog.Reasoning || !slices.Equal(got.Input, catalog.Capabilities) ||
		got.ContextWindow != catalog.ContextWindow || got.MaxTokens != catalog.MaxOutputTokens || gotCost != wantCost {
		t.Errorf("model = %s, catalog = %+v", modelInfoBJSON(got), catalog)
	}
}

func modelInfoBStderrJoined(collected *reports) string { return strings.Join(collected.Lines(), " ") }

// modelInfoBModelsDev points discovery at a models.dev cache document (the Vitest tests serve the same
// document from the mocked models.dev endpoint).
func modelInfoBModelsDev(t *testing.T, document string) Options {
	t.Helper()
	options, _ := testOptions(t)
	options.ModelsDevCachePath = writeModelsDevCache(t, document)
	return options
}

func TestModelInfoB_WithholdsAGroupWhoseDuplicateDeploymentIDCarriesConflictingBackends(t *testing.T) {
	result, _ := modelInfoBDiscover(t,
		`{"model_name":"openai/gpt-5.5","model_info":{"id":"same","mode":"chat"}}`,
		`{"model_name":"openai/gpt-5.5","model_info":{"id":"same","mode":"chat","max_input_tokens":64000},"litellm_params":{"model":"internal/mystery"}}`)
	model := modelInfoBFirst(t, result)
	modelInfoBMatch(t, model, `{"id":"openai/gpt-5.5","name":"openai/gpt-5.5 (incomplete metadata)","reasoning":false,"input":["text"],`+
		`"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0},"maxTokens":16384}`)
	if model.ThinkingLevelMap != nil {
		t.Errorf("thinkingLevelMap = %s, want none", modelInfoBJSON(model.ThinkingLevelMap))
	}
}

func modelInfoBConflictingRows(route string) []string {
	return []string{
		fmt.Sprintf(`{"model_name":%q,"model_info":{"id":"%s-a","mode":"chat"},"litellm_params":{"model":"openai/gpt-4o"}}`, route, route),
		fmt.Sprintf(`{"model_name":%q,"model_info":{"id":"%s-b","mode":"chat"},"litellm_params":{"model":"anthropic/claude-sonnet-4-6"}}`, route, route),
	}
}

func TestModelInfoB_ReportsConflictingDeploymentProviderIdentityOnceWithBoundedDetail(t *testing.T) {
	routes := []string{"diagnostic-1", "diagnostic-2", "diagnostic-3", "diagnostic-4"}
	var rows []string
	for _, route := range routes {
		rows = append(rows, modelInfoBConflictingRows(route)...)
	}
	_, collected := modelInfoBDiscover(t, rows...)

	diagnostics := collected.Lines()
	if len(diagnostics) != 1 {
		t.Fatalf("diagnostics = %q, want one", diagnostics)
	}
	for _, want := range []string{
		"4 route group(s) have missing or conflicting deployment provider evidence",
		"diagnostic-1, diagnostic-2, diagnostic-3 (+1 more)",
	} {
		if !strings.Contains(diagnostics[0], want) {
			t.Errorf("diagnostic %q lacks %q", diagnostics[0], want)
		}
	}
	// Bounded: a count, at most three route ids, and no deployment ids or params.
	for _, forbidden := range []string{"diagnostic-4", "diagnostic-1-a"} {
		if strings.Contains(diagnostics[0], forbidden) {
			t.Errorf("diagnostic %q contains %q", diagnostics[0], forbidden)
		}
	}
}

func TestModelInfoB_ReportsEachAmbiguousRouteOncePerProcessNotOncePerDiscovery(t *testing.T) {
	routes := []string{"once-a", "once-b", "once-c"}
	options, collected := testOptions(t)
	run := func(routes []string) {
		var rows []string
		for _, route := range routes {
			rows = append(rows, modelInfoBConflictingRows(route)...)
		}
		discover(t, modelInfoBRows(t, rows...), options)
	}

	run(routes[:2])
	if lines := collected.Lines(); len(lines) != 1 || !strings.Contains(lines[0], "2 route group(s)") {
		t.Fatalf("first discovery diagnostics = %q", lines)
	}

	// A background refresh of the same misconfiguration must not repeat itself.
	run(routes[:2])
	if lines := collected.Lines(); len(lines) != 1 {
		t.Fatalf("repeated discovery diagnostics = %q", lines)
	}

	// A newly ambiguous route is still worth reporting, and only that one.
	run(routes)
	lines := collected.Lines()
	if len(lines) != 2 {
		t.Fatalf("diagnostics = %q, want two", lines)
	}
	second := lines[1]
	if !strings.Contains(second, "1 route group(s)") || !strings.Contains(second, routes[2]) || strings.Contains(second, routes[0]) {
		t.Errorf("second diagnostic = %q", second)
	}
}

func TestModelInfoB_ReportsARouteWhoseDeploymentsSupplyPartialProviderEvidence(t *testing.T) {
	// Withholding also happens when one deployment resolves a provider and another supplies none, so the
	// wording must not claim a conflict is the only cause.
	route := "partial-evidence-route"
	result, collected := modelInfoBDiscover(t,
		fmt.Sprintf(`{"model_name":%q,"model_info":{"id":"a","mode":"chat"},"litellm_params":{"model":"openai/gpt-4o"}}`, route),
		fmt.Sprintf(`{"model_name":%q,"model_info":{"id":"b","mode":"chat"}}`, route))

	if got := modelInfoBFirst(t, result).Name; got != route+" (incomplete metadata)" {
		t.Errorf("name = %q", got)
	}
	diagnostics := collected.Lines()
	if len(diagnostics) != 1 || !strings.Contains(diagnostics[0], "missing or conflicting") || !strings.Contains(diagnostics[0], route) {
		t.Errorf("diagnostics = %q", diagnostics)
	}
}

func TestModelInfoB_WithholdsMixedModeRoutesAndReportsOneBoundedDiagnostic(t *testing.T) {
	routes := []string{"mixed-mode-1", "mixed-mode-2", "mixed-mode-3", "mixed-mode-4"}
	var rows []string
	for _, route := range routes {
		rows = append(rows,
			fmt.Sprintf(`{"model_name":%q,"litellm_params":{"model":"openai/gpt-5.5"},"model_info":{"id":"%s-chat","mode":"chat"}}`, route, route),
			fmt.Sprintf(`{"model_name":%q,"model_info":{"id":"%s-embed","mode":"embedding"}}`, route, route))
	}
	result, collected := modelInfoBDiscover(t, rows...)

	if len(result.Models) != 0 {
		t.Errorf("models = %v, want none", modelInfoBIDs(result.Models))
	}
	lines := collected.Lines()
	if len(lines) != 1 {
		t.Fatalf("diagnostics = %q, want one", lines)
	}
	for _, want := range []string{
		"4 route group(s) mix chat-style and explicitly incompatible deployment modes",
		"mixed-mode-1, mixed-mode-2, mixed-mode-3 (+1 more)",
	} {
		if !strings.Contains(lines[0], want) {
			t.Errorf("diagnostic %q lacks %q", lines[0], want)
		}
	}
	if strings.Contains(lines[0], "mixed-mode-4") {
		t.Errorf("diagnostic %q names the fourth route", lines[0])
	}
}

func modelInfoBAgreedAndUnknownRows() []string {
	return []string{
		`{"model_name":"agreed","model_info":{"id":"a","mode":"chat"},"litellm_params":{"model":"openai/gpt-4o"}}`,
		`{"model_name":"agreed","model_info":{"id":"b","mode":"chat"},"litellm_params":{"model":"openai/gpt-4o"}}`,
		`{"model_name":"unknown","model_info":{"id":"c","mode":"chat"},"litellm_params":{"model":"internal/x"}}`,
		`{"model_name":"unknown","model_info":{"id":"d","mode":"chat"},"litellm_params":{"model":"internal/y"}}`,
	}
}

func TestModelInfoB_ReportsGroupsWhoseResolvedFallbackIdentitiesDisagree(t *testing.T) {
	_, collected := modelInfoBDiscover(t, modelInfoBAgreedAndUnknownRows()...)
	lines := collected.Lines()
	if len(lines) != 1 || !strings.Contains(lines[0], "unknown") {
		t.Errorf("diagnostics = %q", lines)
	}
}

func TestModelInfoB_KeepsKimiCompatibilityOnNonMoonshotRoutes(t *testing.T) {
	options, _ := testOptions(t)
	base := modelInfoBServeAll(t, 200, json.RawMessage(modelInfoBody(
		`{"model_name":"kimi-k3","litellm_params":{"model":"azure_ai/FW-Kimi-K3"},"model_info":{"mode":"chat"}}`)))
	model := modelInfoBFirst(t, discover(t, base, options))

	want := *BuildCompat("kimi-k3")
	want.RequiresReasoningContentOnAssistantMessages = boolPtr(true)
	modelInfoBMatch(t, model.Compat, modelInfoBJSON(want))
}

func TestModelInfoB_DerivesReasoningVisibilityFromMoonshotRoutingEvidence(t *testing.T) {
	for _, tc := range []struct {
		name, entry string
		suppress    bool
	}{
		{"Azure-hosted Kimi", `{"model_name":"kimi-k3","litellm_params":{"model":"azure/FW-Kimi-K3"},` +
			`"model_info":{"mode":"chat","base_model":"fireworks/accounts/fireworks/models/kimi-k3","litellm_provider":"azure"}}`, false},
		{"Bedrock-hosted Kimi", `{"model_name":"moonshotai-kimi-k2-5","litellm_params":{"model":"bedrock/moonshotai.kimi-k2.5"},` +
			`"model_info":{"mode":"chat","base_model":"moonshotai.kimi-k2.5","litellm_provider":"bedrock_converse"}}`, false},
		{"opaque Moonshot alias", `{"model_name":"k3-prod","litellm_params":{"model":"moonshot/kimi-k2.5"},"model_info":{"mode":"chat"}}`, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			options, _ := testOptions(t)
			model := modelInfoBFirst(t, discover(t, modelInfoBServeAll(t, 200, modelInfoBody(tc.entry)), options))
			if got := modelInfoBSuppresses(model); got != tc.suppress {
				t.Errorf("suppressReasoningVisibility = %v, want %v", got, tc.suppress)
			}
			if model.LiteLLMPolicy == nil || !model.LiteLLMPolicy.NormalizeThinkTags {
				t.Errorf("normalizeThinkTags = %s, want true", modelInfoBJSON(model.LiteLLMPolicy))
			}
		})
	}
}

func TestModelInfoB_RequiresMoonshotTransportForSuppression(t *testing.T) {
	for _, tc := range []struct {
		name      string
		routes    []string // "" means a row with no litellm_params
		suppress  bool
		normalize bool
	}{
		{"Moonshot deployments", []string{"moonshot/kimi-k3", "moonshot/kimi-k3"}, true, true},
		{"mixed deployments", []string{"moonshot/kimi-k3", "azure_ai/FW-Kimi-K3"}, false, true},
		{"reversed mixed deployments", []string{"azure_ai/FW-Kimi-K3", "moonshot/kimi-k3"}, false, true},
		{"incomplete deployment metadata", []string{"moonshot/kimi-k3", ""}, false, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var rows []string
			for _, route := range tc.routes {
				params := ""
				if route != "" {
					params = fmt.Sprintf(`"litellm_params":{"model":%q},`, route)
				}
				rows = append(rows, fmt.Sprintf(`{"model_name":"kimi-prod",%s"model_info":{"mode":"chat"}}`, params))
			}
			options, _ := testOptions(t)
			model := modelInfoBFirst(t, discover(t, modelInfoBServeAll(t, 200, modelInfoBody(rows...)), options))
			if got := modelInfoBSuppresses(model); got != tc.suppress {
				t.Errorf("suppressReasoningVisibility = %v, want %v", got, tc.suppress)
			}
			if got := model.LiteLLMPolicy != nil && model.LiteLLMPolicy.NormalizeThinkTags; got != tc.normalize {
				t.Errorf("normalizeThinkTags = %v, want %v", got, tc.normalize)
			}
		})
	}
}

func TestModelInfoB_WithholdsIncompatibleSiblingWhenReducingSuppression(t *testing.T) {
	for _, tc := range []struct {
		name  string
		first [2]string // model, mode
		other [2]string
	}{
		{"withholds a non-Moonshot incompatible sibling", [2]string{"moonshot/kimi-k3", "chat"}, [2]string{"openai/text-embedding-3-small", "embedding"}},
		{"withholds a Moonshot incompatible sibling", [2]string{"azure_ai/FW-Kimi-K3", "chat"}, [2]string{"moonshot/kimi-k3", "embedding"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var rows []string
			for i, deployment := range [][2]string{tc.first, tc.other} {
				rows = append(rows, fmt.Sprintf(`{"model_name":"kimi-incompatible-route","litellm_params":{"model":%q},`+
					`"model_info":{"id":"deployment-%d","mode":%q}}`, deployment[0], i, deployment[1]))
			}
			options, collected := testOptions(t)
			result := discover(t, modelInfoBServeAll(t, 200, modelInfoBody(rows...)), options)
			if len(result.Models) != 0 {
				t.Errorf("models = %v, want none", modelInfoBIDs(result.Models))
			}
			if lines := collected.Lines(); len(lines) != 1 {
				t.Errorf("diagnostics = %q, want one", lines)
			}
		})
	}
}

func TestModelInfoB_KeepsThinkTagNormalizationIndependentOfHostingTransport(t *testing.T) {
	for _, tc := range []struct {
		name, entry string
		suppress    bool
	}{
		{"Azure-hosted Kimi", `{"model_name":"kimi-k3","litellm_params":{"model":"azure/FW-Kimi-K3"},` +
			`"model_info":{"mode":"chat","base_model":"fireworks/accounts/fireworks/models/kimi-k3","litellm_provider":"azure"}}`, false},
		{"Bedrock-hosted Kimi", `{"model_name":"moonshotai-kimi-k2-5","litellm_params":{"model":"bedrock/moonshotai.kimi-k2.5"},` +
			`"model_info":{"mode":"chat","base_model":"moonshotai.kimi-k2.5","litellm_provider":"bedrock_converse"}}`, false},
		{"Moonshot-hosted opaque alias", `{"model_name":"k3-prod","litellm_params":{"model":"moonshot/kimi-k2.5"},"model_info":{"mode":"chat"}}`, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			options, _ := testOptions(t)
			model := modelInfoBFirst(t, discover(t, modelInfoBServeAll(t, 200, modelInfoBody(tc.entry)), options))
			if got := modelInfoBSuppresses(model); got != tc.suppress {
				t.Errorf("suppressReasoningVisibility = %v, want %v", got, tc.suppress)
			}
			if model.LiteLLMPolicy == nil || !model.LiteLLMPolicy.NormalizeThinkTags {
				t.Errorf("normalizeThinkTags = %s, want true", modelInfoBJSON(model.LiteLLMPolicy))
			}
		})
	}
}

func TestModelInfoB_DoesNotSuppressAnAliasRoutedToAForcedThinkingMoonshotModel(t *testing.T) {
	options, _ := testOptions(t)
	base := modelInfoBServeAll(t, 200, modelInfoBody(
		`{"model_name":"k3-prod","litellm_params":{"model":"moonshot/kimi-k2-thinking"},"model_info":{"mode":"chat"}}`))
	if modelInfoBSuppresses(modelInfoBFirst(t, discover(t, base, options))) {
		t.Error("suppressReasoningVisibility = true, want false")
	}
}

func TestModelInfoB_ChecksEveryDeclaredMoonshotRoutingSignal(t *testing.T) {
	for _, tc := range []struct {
		name, params string
		suppress     bool
	}{
		{"provider-only metadata", `{"custom_llm_provider":"moonshot"}`, true},
		{"Moonshot provider and Azure-hosted Kimi backend", `{"custom_llm_provider":"moonshot","model":"azure_ai/FW-Kimi-K3"}`, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			options, _ := testOptions(t)
			base := modelInfoBServeAll(t, 200, modelInfoBody(
				fmt.Sprintf(`{"model_name":"kimi-prod","litellm_params":%s,"model_info":{"mode":"chat"}}`, tc.params)))
			model := modelInfoBFirst(t, discover(t, base, options))
			if got := modelInfoBSuppresses(model); got != tc.suppress {
				t.Errorf("suppressReasoningVisibility = %v, want %v", got, tc.suppress)
			}
			if model.LiteLLMPolicy == nil || !model.LiteLLMPolicy.NormalizeThinkTags {
				t.Errorf("normalizeThinkTags = %s, want true", modelInfoBJSON(model.LiteLLMPolicy))
			}
		})
	}
}

func TestModelInfoB_KeepsRouteEvidenceIsolatedBetweenDiscoveries(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		model := "azure/gpt-4o"
		if strings.HasPrefix(r.URL.Path, "/moonshot/") {
			model = "moonshot/kimi-k3"
		}
		jsonResponse(200, modelInfoBody(
			fmt.Sprintf(`{"model_name":"kimi-prod","litellm_params":{"model":%q},"model_info":{"mode":"chat"}}`, model)))(w, r)
	}))
	t.Cleanup(server.Close)
	options, _ := testOptions(t)

	moonshot := discover(t, server.URL+"/moonshot", options)
	azure := discover(t, server.URL+"/azure", options)

	if !modelInfoBSuppresses(modelInfoBFirst(t, moonshot)) {
		t.Error("moonshot suppressReasoningVisibility = false, want true")
	}
	if modelInfoBSuppresses(modelInfoBFirst(t, azure)) {
		t.Error("azure suppressReasoningVisibility = true, want false")
	}
}

func TestModelInfoB_DoesNotReuseRouteEvidenceAfterMetadataFallback(t *testing.T) {
	var fallback atomic.Bool
	base := mockEndpoints(t, map[string]http.HandlerFunc{
		"/model/info": func(w http.ResponseWriter, r *http.Request) {
			if fallback.Load() {
				jsonResponse(403, map[string]any{})(w, r)
				return
			}
			jsonResponse(200, modelInfoBody(
				`{"model_name":"kimi-prod","litellm_params":{"model":"moonshot/kimi-k3"},"model_info":{"mode":"chat"}}`))(w, r)
		},
		"/v1/models":    jsonResponse(200, map[string]any{"data": []any{map[string]any{"id": "kimi-prod"}}}),
		modelInfoBProbe: jsonResponse(404, map[string]any{}),
	})
	options, _ := testOptions(t)

	discover(t, base, options)
	fallback.Store(true)
	result := discover(t, base, options)

	if modelInfoBSuppresses(modelInfoBFirst(t, result)) {
		t.Error("suppressReasoningVisibility = true, want false")
	}
}

func TestModelInfoB_UsesModelsDevReasoningOptionsForAnAzureGPT5Deployment(t *testing.T) {
	options := modelInfoBModelsDev(t, `{"azure":{"models":{`+
		`"gpt-5":{"reasoning_options":{"type":"effort","values":["minimal","low","medium","high"]}},`+
		`"gpt-5-adapter":{"reasoning_options":{"type":"effort","values":["low","high"]}}}}}`)
	base := modelInfoBRows(t, `{"model_name":"azure-gpt-5","litellm_params":{"model":"azure/gpt-5","allowed_openai_params":["reasoning_effort"]},`+
		`"model_info":{"id":"one","mode":"chat","litellm_provider":"azure","supports_reasoning":true}}`)

	modelInfoBLevelsEqual(t, modelInfoBFirst(t, discover(t, base, options)).ThinkingLevelMap,
		modelInfoBLevels("", "minimal", "low", "medium", "high", "", ""))
}

func TestModelInfoB_UsesTheAdapterAsThePublicCatalogLookupProviderWhenBackendIdentityHasNone(t *testing.T) {
	options := modelInfoBModelsDev(t, `{"azure":{"models":{"gpt-5-adapter":{"reasoning_options":{"type":"effort","values":["low","high"]}}}}}`)
	base := modelInfoBRows(t, `{"model_name":"azure-gpt-5","litellm_params":{"model":"gpt-5-adapter","allowed_openai_params":["reasoning_effort"]},`+
		`"model_info":{"id":"one","mode":"chat","litellm_provider":"azure","supports_reasoning":true}}`)

	modelInfoBLevelsEqual(t, modelInfoBFirst(t, discover(t, base, options)).ThinkingLevelMap,
		modelInfoBLevels("", "", "low", "", "high", "", ""))
}

// models.dev serves ChatGPT routes from OpenAI's key, but a field it omits must still come from the
// subscription's Codex catalog, not OpenAI API pricing.
func TestModelInfoB_KeepsCodexCatalogPricingForAPartialModelsDevChatGPTRecord(t *testing.T) {
	options := modelInfoBModelsDev(t, `{"openai":{"models":{"gpt-5.6-sol":{"reasoning_options":[{"type":"effort","values":["none","low","high"]}]}}}}`)
	base := modelInfoBRows(t, `{"model_name":"gpt-5.6-sol","litellm_params":{"model":"chatgpt/gpt-5.6-sol"},`+
		`"model_info":{"mode":"responses","litellm_provider":"chatgpt","supported_openai_params":["reasoning_effort"]}}`)

	model := modelInfoBFirst(t, discover(t, base, options))

	codex := modelInfoBCatalogModel(t, "openai-codex", "gpt-5.6-sol")
	if codex.InputCostPerMTokens <= 0 {
		t.Fatalf("Codex input price = %v, want > 0", codex.InputCostPerMTokens)
	}
	want := [4]float64{codex.InputCostPerMTokens, codex.OutputCostPerMTokens, codex.CacheReadCost, codex.CacheWriteCost}
	got := [4]float64{model.Cost.Input, model.Cost.Output, model.Cost.CacheRead, model.Cost.CacheWrite}
	if got != want {
		t.Errorf("cost = %v, want %v", got, want)
	}
}

func TestModelInfoB_UsesCustomProviderAuthorityForPublicReasoningEffortsOverAGenericAdapter(t *testing.T) {
	for _, customProvider := range []string{"azure", "azure_ai"} {
		t.Run(customProvider, func(t *testing.T) {
			options := modelInfoBModelsDev(t, `{"azure":{"models":{"gpt-5-adapter":{"reasoning_options":{"type":"effort","values":["low","high"]}}}},`+
				`"openai":{"models":{"gpt-5-adapter":{"reasoning_options":{"type":"effort","values":["minimal","high"]}}}}}`)
			base := modelInfoBRows(t, fmt.Sprintf(`{"model_name":"custom-azure-route","litellm_params":{"model":"gpt-5-adapter",`+
				`"custom_llm_provider":%q,"allowed_openai_params":["reasoning_effort"]},`+
				`"model_info":{"mode":"chat","litellm_provider":"openai","supports_reasoning":true}}`, customProvider))

			model := modelInfoBFirst(t, discover(t, base, options))
			if model.API != ai.APIOpenAICompletions {
				t.Errorf("api = %q", model.API)
			}
			modelInfoBLevelsEqual(t, model.ThinkingLevelMap, modelInfoBLevels("", "", "low", "", "high", "", ""))
		})
	}
}

func TestModelInfoB_IgnoresMalformedAcceptedParameterArraysWithoutDroppingHealthyModels(t *testing.T) {
	result, _ := modelInfoBDiscover(t,
		`{"model_name":"malformed","litellm_params":{"model":"internal/malformed","allowed_openai_params":7},"model_info":{"id":"bad","mode":"chat","supported_openai_params":{}}}`,
		`{"model_name":"healthy-route","litellm_params":{"model":"openai/gpt-4o"},"model_info":{"id":"good","mode":"chat"}}`)
	if !slices.Contains(modelInfoBIDs(result.Models), "healthy-route") {
		t.Errorf("ids = %v, want healthy-route", modelInfoBIDs(result.Models))
	}
}

func TestModelInfoB_UsesCustomLLMProviderAuthorityForCatalogMetadataAndTheChatCarrier(t *testing.T) {
	options, _ := testOptions(t)
	base := modelInfoBServeAll(t, 200, modelInfoBody(`{"model_name":"private/kimi-route","litellm_params":{"model":"kimi-k2.6",`+
		`"custom_llm_provider":"moonshot","allowed_openai_params":["thinking"]},"model_info":{"mode":"chat","supports_reasoning":true}}`))

	model := modelInfoBFirst(t, discover(t, base, options))

	modelInfoBMatch(t, model, `{"id":"private/kimi-route","name":"private/kimi-route","input":["text","image"],`+
		`"cost":{"input":0.95,"output":4,"cacheRead":0.16,"cacheWrite":0},"contextWindow":262144,"maxTokens":262144,"reasoning":true,`+
		`"thinkingLevelMap":{"off":"off","minimal":null,"low":null,"medium":null,"high":"high","xhigh":null,"max":null},`+
		`"compat":{"thinkingFormat":"deepseek","supportsReasoningEffort":false}}`)
}

func TestModelInfoB_PrefersCustomLLMProviderOverAGenericAdapterForCatalogLookup(t *testing.T) {
	options, _ := testOptions(t)
	base := modelInfoBServeAll(t, 200, modelInfoBody(
		`{"model_name":"kimi-route","litellm_params":{"model":"kimi-k2.6","custom_llm_provider":"moonshot"},"model_info":{"litellm_provider":"openai","mode":"chat"}}`,
		`{"model_name":"moonshot-control","litellm_params":{"model":"kimi-k2.6"},"model_info":{"litellm_provider":"moonshot","mode":"chat"}}`))
	result := discover(t, base, options)

	for _, id := range []string{"kimi-route", "moonshot-control"} {
		modelInfoBMatch(t, modelInfoBFind(t, result, id), fmt.Sprintf(`{"id":%q,"name":%q,"contextWindow":262144,"maxTokens":262144,`+
			`"cost":{"input":0.95,"output":4,"cacheRead":0.16,"cacheWrite":0}}`, id, id))
	}
}

func TestModelInfoB_DerivesGeminiNormalizationFromAdapterEvidenceForAnOpaqueRoute(t *testing.T) {
	result, _ := modelInfoBDiscover(t, `{"model_name":"internal/prod-route","litellm_params":{"model":"internal/opaque"},`+
		`"model_info":{"id":"a","mode":"chat","litellm_provider":"gemini"}}`)
	if policy := modelInfoBFirst(t, result).LiteLLMPolicy; policy == nil || !policy.NormalizeGeminiReasoningEffort {
		t.Errorf("litellmPolicy = %s, want normalizeGeminiReasoningEffort", modelInfoBJSON(policy))
	}
}

func TestModelInfoB_MarksChatRoutesOverGPT55PlusDeploymentsToDropReasoningWithTools(t *testing.T) {
	result, _ := modelInfoBDiscover(t,
		`{"model_name":"team-sol","litellm_params":{"model":"azure_ai/gpt-6-sol"},"model_info":{"id":"a","mode":"chat","base_model":"azure/gpt-6-sol"}}`,
		`{"model_name":"team-terra","litellm_params":{"model":"azure_ai/gpt-5.6-terra"},"model_info":{"id":"b","mode":"chat"}}`)

	for _, id := range []string{"team-sol", "team-terra"} {
		model := modelInfoBFind(t, result, id)
		if model.API != ai.APIOpenAICompletions {
			t.Errorf("%s api = %q", id, model.API)
		}
		if model.LiteLLMPolicy == nil || !model.LiteLLMPolicy.DropToolReasoning {
			t.Errorf("%s dropToolReasoning = %s, want true", id, modelInfoBJSON(model.LiteLLMPolicy))
		}
	}
	// Only GPT-6 needs an explicit off effort; GPT-5.6 accepts an omitted one.
	if policy := modelInfoBFind(t, result, "team-sol").LiteLLMPolicy; policy == nil || !policy.ExplicitToolReasoningOff {
		t.Errorf("team-sol explicitToolReasoningOff = %s, want true", modelInfoBJSON(policy))
	}
	if policy := modelInfoBFind(t, result, "team-terra").LiteLLMPolicy; policy != nil && policy.ExplicitToolReasoningOff {
		t.Errorf("team-terra explicitToolReasoningOff = true, want unset")
	}
}

func TestModelInfoB_KeepsReasoningWithToolsForOlderGPTBackendsAndGPT6ResponsesRoutes(t *testing.T) {
	result, _ := modelInfoBDiscover(t,
		`{"model_name":"gpt-6-sol","litellm_params":{"model":"azure_ai/gpt-5.1"},"model_info":{"id":"a","mode":"chat"}}`,
		`{"model_name":"responses-sol","litellm_params":{"model":"azure_ai/gpt-6-sol"},"model_info":{"id":"b","mode":"responses"}}`)

	if api := modelInfoBFind(t, result, "responses-sol").API; api != ai.APIOpenAIResponses {
		t.Errorf("responses-sol api = %q", api)
	}
	for _, id := range []string{"gpt-6-sol", "responses-sol"} {
		if policy := modelInfoBFind(t, result, id).LiteLLMPolicy; policy != nil && policy.DropToolReasoning {
			t.Errorf("%s dropToolReasoning = true, want unset", id)
		}
	}
}

func TestModelInfoB_CarriesTheToolReasoningDropToWildcardChildrenOverGPT55PlusDeployments(t *testing.T) {
	base := modelInfoBEndpoints(t, map[string]http.HandlerFunc{
		"/model/info": jsonResponse(200, modelInfoBody(
			`{"model_name":"azure_ai/*","litellm_params":{"model":"azure_ai/*"},"model_info":{"mode":"chat"}}`)),
		"/v1/models": jsonResponse(200, map[string]any{"data": []any{
			map[string]any{"id": "azure_ai/gpt-6-sol", "object": "model", "owned_by": "openai"},
			map[string]any{"id": "azure_ai/gpt-5.1", "object": "model", "owned_by": "openai"},
		}}),
	})
	options, _ := testOptions(t)
	result := discover(t, base, options)

	sol := modelInfoBFind(t, result, "azure_ai/gpt-6-sol")
	if sol.API != ai.APIOpenAICompletions {
		t.Errorf("sol api = %q", sol.API)
	}
	if sol.LiteLLMPolicy == nil || !sol.LiteLLMPolicy.DropToolReasoning || !sol.LiteLLMPolicy.ExplicitToolReasoningOff {
		t.Errorf("sol litellmPolicy = %s, want dropToolReasoning and explicitToolReasoningOff", modelInfoBJSON(sol.LiteLLMPolicy))
	}
	if policy := modelInfoBFind(t, result, "azure_ai/gpt-5.1").LiteLLMPolicy; policy != nil && policy.DropToolReasoning {
		t.Error("gpt-5.1 dropToolReasoning = true, want unset")
	}
}

func TestModelInfoB_WithholdsGeminiNormalizationWhenDeploymentFamilyEvidenceIsMixed(t *testing.T) {
	result, _ := modelInfoBDiscover(t,
		`{"model_name":"gemini-looking-route","litellm_params":{"model":"gemini/gemini-3.1-pro-preview"},"model_info":{"id":"a","mode":"chat","litellm_provider":"gemini"}}`,
		`{"model_name":"gemini-looking-route","litellm_params":{"model":"openai/gpt-4o"},"model_info":{"id":"b","mode":"chat","litellm_provider":"openai"}}`)
	if policy := modelInfoBFirst(t, result).LiteLLMPolicy; policy != nil {
		t.Errorf("litellmPolicy = %s, want none", modelInfoBJSON(policy))
	}
}

func TestModelInfoB_DoesNotUseAQualifiedSingletonPublicRouteAsCatalogAuthority(t *testing.T) {
	result, _ := modelInfoBDiscover(t, `{"model_name":"openai/gpt-5.5","model_info":{"id":"one","mode":"chat"}}`)
	modelInfoBIncomplete(t, modelInfoBFirst(t, result), "openai/gpt-5.5")
}

func TestModelInfoB_WithholdsRouteTextCatalogMetadataForASingleton(t *testing.T) {
	for _, tc := range []struct{ name, row string }{
		{"opaque backend model", `{"model_name":"openai/gpt-5.5","litellm_params":{"model":"internal/mystery"},"model_info":{"id":"only","mode":"chat"}}`},
		{"opaque base model", `{"model_name":"openai/gpt-5.5","model_info":{"base_model":"internal/mystery","id":"only","mode":"chat"}}`},
		{"unresolved adapter", `{"model_name":"openai/gpt-5.5","model_info":{"litellm_provider":"custom_proxy","id":"only","mode":"chat"}}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			result, _ := modelInfoBDiscover(t, tc.row)
			modelInfoBIncomplete(t, modelInfoBFirst(t, result), "openai/gpt-5.5")
		})
	}
}

func TestModelInfoB_KeepsKimiGenerationControlsOnTheTransport(t *testing.T) {
	for _, adapter := range []string{"openai", "custom_openai", "openai_like", "text-completion-openai", "azure", "azure_ai"} {
		t.Run(adapter, func(t *testing.T) {
			result, _ := modelInfoBDiscover(t, fmt.Sprintf(`{"model_name":"kimi-route",`+
				`"litellm_params":{"model":"openai/kimi-k2.5","allowed_openai_params":["thinking"]},`+
				`"model_info":{"mode":"chat","litellm_provider":%q,"supports_reasoning":true}}`, adapter))
			model := modelInfoBFirst(t, result)
			modelInfoBMatch(t, model, `{"compat":{"thinkingFormat":"deepseek","supportsReasoningEffort":false},`+
				`"litellmPolicy":{"normalizeStrictToolMessages":true,"normalizeThinkTags":true,"suppressReasoningVisibility":false}}`)
		})
	}
}

func TestModelInfoB_IgnoresFamilyAndGenerationWordsInUnknownBackendProvider(t *testing.T) {
	for _, model := range []string{"kimi-proxy/gpt-4-turbo", "kimi-k2.5-proxy/gpt-4-turbo"} {
		t.Run(model, func(t *testing.T) {
			result, _ := modelInfoBDiscover(t, fmt.Sprintf(`{"model_name":"opaque","litellm_params":{"model":%q},`+
				`"model_info":{"mode":"chat","supported_endpoints":["/v1/chat/completions"],"supports_reasoning":true,"supported_openai_params":["thinking"]}}`, model))
			if len(result.Models) != 1 {
				t.Fatalf("models = %v, want one", modelInfoBIDs(result.Models))
			}
			got := result.Models[0]
			if got.API != ai.APIOpenAICompletions || got.LiteLLMBackendFamily != types.FamilyOpenAI {
				t.Errorf("api = %q, family = %q", got.API, got.LiteLLMBackendFamily)
			}
			modelInfoBMatch(t, got.Compat, `{"supportsStore":false}`)
			if got.Compat == nil || !reflect.DeepEqual(*got.Compat, ai.ModelCompat{SupportsStore: boolPtr(false)}) {
				t.Errorf("compat = %s, want only supportsStore=false", modelInfoBJSON(got.Compat))
			}
			modelInfoBLevelsEqual(t, got.ThinkingLevelMap, modelInfoBNoLevels())
			if got.LiteLLMPolicy != nil {
				t.Errorf("litellmPolicy = %s, want none", modelInfoBJSON(got.LiteLLMPolicy))
			}
		})
	}
}

func TestModelInfoB_PublishesKimiCompatibilityWithoutMoonshotRequestParametersThroughAnOpenAITransport(t *testing.T) {
	options, _ := testOptions(t)
	base := modelInfoBServeAll(t, 200, modelInfoBody(`{"model_name":"kimi-through-openai",`+
		`"litellm_params":{"model":"openai/kimi-k2.5","allowed_openai_params":["thinking"]},`+
		`"model_info":{"id":"kimi-openai","mode":"chat","litellm_provider":"openai","supports_reasoning":true}}`))

	modelInfoBMatch(t, modelInfoBFirst(t, discover(t, base, options)), `{"id":"kimi-through-openai","reasoning":true,`+
		`"compat":`+modelInfoBChatCompat()+`,`+
		`"litellmPolicy":{"normalizeStrictToolMessages":true,"normalizeThinkTags":true,"suppressReasoningVisibility":false}}`)
}

func TestModelInfoB_AppliesKimiPolicyFromAMoonshotCustomProviderWithAnOpaqueModel(t *testing.T) {
	options, _ := testOptions(t)
	base := modelInfoBServeAll(t, 200, modelInfoBody(`{"model_name":"opaque-kimi-route",`+
		`"litellm_params":{"model":"opaque-model","custom_llm_provider":"moonshot"},"model_info":{"id":"kimi-custom-provider","mode":"chat"}}`))

	modelInfoBPolicyEquals(t, modelInfoBFirst(t, discover(t, base, options)).LiteLLMPolicy, types.LiteLLMModelPolicy{
		NormalizeStrictToolMessages: true, NormalizeThinkTags: true, SuppressReasoningVisibility: true,
	})
}

func TestModelInfoB_KeepsUnknownCatalogMetadataUnresolvedThroughTheAdapter(t *testing.T) {
	for _, adapter := range []string{"openai", "custom_openai", "openai_like", "text-completion-openai", "azure", "azure_ai"} {
		t.Run(adapter, func(t *testing.T) {
			resolution := ResolveModelInfoCatalog(entryFrom(t, fmt.Sprintf(`{"model_name":"opaque-route",`+
				`"litellm_params":{"model":"internal/model"},"model_info":{"mode":"chat","litellm_provider":%q}}`, adapter)), nil)
			if resolution == nil || resolution.Provider != "internal" || resolution.CatalogModelID != "internal/model" {
				t.Errorf("resolution = %+v", resolution)
			}
		})
	}
}

func TestModelInfoB_KeepsProviderIdentityFromTheBackendCandidateThatResolves(t *testing.T) {
	for _, tc := range []struct{ entry, provider string }{
		{`{"model_name":"mixed-evidence","litellm_params":{"model":"internal/claude-magic"},"model_info":{"mode":"chat","base_model":"openai/gpt-4o"}}`, "openai"},
		{`{"model_name":"aliased","litellm_params":{"model":"anthropic/opus-4-7"},"model_info":{"mode":"chat"}}`, "anthropic"},
	} {
		resolution := ResolveModelInfoCatalog(entryFrom(t, tc.entry), nil)
		if resolution == nil || resolution.Provider != tc.provider {
			t.Errorf("%s: resolution = %+v, want provider %q", tc.entry, resolution, tc.provider)
		}
	}
}

func TestModelInfoB_RetainsCatalogAuthorityIndependentlyOfFamilyDisagreement(t *testing.T) {
	for _, tc := range []struct{ name, entry, provider string }{
		{"routing model conflicts with base model",
			`{"model_name":"conflicting-models","litellm_params":{"model":"openai/gpt-4o"},"model_info":{"mode":"chat","base_model":"anthropic/claude-sonnet-4-6"}}`, "anthropic"},
		{"adapter conflicts with the qualified model",
			`{"model_name":"conflicting-adapter","litellm_params":{"model":"openai/gpt-4o"},"model_info":{"mode":"chat","litellm_provider":"anthropic"}}`, "openai"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resolution := ResolveModelInfoCatalog(entryFrom(t, tc.entry), nil)
			if resolution == nil || resolution.Provider != tc.provider {
				t.Errorf("resolution = %+v, want provider %q", resolution, tc.provider)
			}
		})
	}
}

func TestModelInfoB_RetainsTheQualifiedCatalogIdentityDespiteADifferentAdapterFamily(t *testing.T) {
	resolution := ResolveModelInfoCatalog(entryFrom(t, `{"model_name":"specific-adapter-conflict",`+
		`"litellm_params":{"model":"openai/kimi-k2.5"},"model_info":{"mode":"chat","litellm_provider":"anthropic"}}`), nil)
	if resolution == nil || resolution.Provider != "openai" || resolution.CatalogModelID != "openai/kimi-k2.5" {
		t.Errorf("resolution = %+v", resolution)
	}
}

func TestModelInfoB_UsesBaseModelCatalogAuthorityWhenConfiguredModelsDifferInOrder(t *testing.T) {
	for _, tc := range []struct{ name, routing, base string }{
		{"routing then base", "openai/gpt-4o", "openai/o3"},
		{"base then routing", "openai/o3", "openai/gpt-4o"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			resolution := ResolveModelInfoCatalog(entryFrom(t, fmt.Sprintf(`{"model_name":"conflicting-openai-models",`+
				`"litellm_params":{"model":%q},"model_info":{"mode":"chat","litellm_provider":"openai","base_model":%q}}`, tc.routing, tc.base)), nil)
			want := strings.TrimPrefix(tc.base, "openai/")
			if resolution == nil || resolution.Provider != "openai" || resolution.CatalogModelID != want {
				t.Errorf("resolution = %+v, want openai %q", resolution, want)
			}
		})
	}
}

func TestModelInfoB_DoesNotClassifyK27WithoutAnExactCodeSuffix(t *testing.T) {
	for _, model := range []string{"moonshot/kimi-k2.7", "moonshot/kimi-k2.7-instruct", "moonshot/kimi-k2.7-codec", "moonshot/kimi-k2.7-highspeedy"} {
		t.Run(model, func(t *testing.T) {
			result, _ := modelInfoBDiscover(t, fmt.Sprintf(`{"model_name":"unknown-k2.7-variant",`+
				`"litellm_params":{"model":%q,"allowed_openai_params":["thinking"]},"model_info":{"mode":"chat","supports_reasoning":true}}`, model))
			got := modelInfoBFirst(t, result)
			modelInfoBLevelsEqual(t, got.ThinkingLevelMap, modelInfoBNoLevels())
			if modelInfoBHasKey(got.Compat, "thinkingFormat") {
				t.Errorf("compat = %s, want no thinkingFormat", modelInfoBJSON(got.Compat))
			}
		})
	}
}

func TestModelInfoB_ClassifiesTheSupportedK27CodeSuffix(t *testing.T) {
	for _, model := range []string{"moonshot/kimi-k2.7-code", "moonshot/kimi-k2.7_highspeed", "moonshot/kimi-k2.7.code"} {
		t.Run(model, func(t *testing.T) {
			result, _ := modelInfoBDiscover(t, fmt.Sprintf(`{"model_name":"known-k2.7-variant",`+
				`"litellm_params":{"model":%q,"allowed_openai_params":["thinking"]},"model_info":{"mode":"chat","supports_reasoning":true}}`, model))
			got := modelInfoBFirst(t, result)
			modelInfoBMatch(t, got, `{"thinkingLevelMap":{"off":null,"minimal":null,"low":null,"medium":null,"high":"high","xhigh":null,"max":null},`+
				`"compat":{"thinkingFormat":"deepseek","supportsReasoningEffort":false}}`)
		})
	}
}

func TestModelInfoB_PublishesBaseModelCatalogMetadataWhenConfiguredModelsDifferWithinOneDeployment(t *testing.T) {
	rows := []struct{ row, base string }{
		{`{"model_name":"one-conflicted-openai-deployment","litellm_params":{"model":"openai/gpt-4o"},` +
			`"model_info":{"id":"one","mode":"chat","litellm_provider":"openai","base_model":"openai/o3"}}`, "o3"},
		{`{"model_name":"one-conflicted-openai-deployment","litellm_params":{"model":"openai/o3"},` +
			`"model_info":{"id":"one","mode":"chat","litellm_provider":"openai","base_model":"openai/gpt-4o"}}`, "gpt-4o"},
	}
	for _, tc := range rows {
		t.Run(tc.base, func(t *testing.T) {
			result, _ := modelInfoBDiscover(t, tc.row)
			model := modelInfoBFirst(t, result)
			catalog := modelInfoBCatalogModel(t, "openai", tc.base)
			if model.Name != "one-conflicted-openai-deployment" {
				t.Errorf("name = %q", model.Name)
			}
			modelInfoBMatchCatalog(t, model, catalog)
			if catalog.Reasoning {
				modelInfoBLevelsEqual(t, model.ThinkingLevelMap, modelInfoBNoLevels())
			} else if model.ThinkingLevelMap != nil {
				t.Errorf("thinkingLevelMap = %s, want none", modelInfoBJSON(model.ThinkingLevelMap))
			}
		})
	}
}

func TestModelInfoB_UsesBaseModelMetadataWhileWithholdingConflictingProviderFamilyPolicy(t *testing.T) {
	result, _ := modelInfoBDiscover(t, `{"model_name":"one-conflicted-deployment","litellm_params":{"model":"openai/gpt-4o"},`+
		`"model_info":{"id":"one","mode":"chat","base_model":"anthropic/claude-sonnet-4-6"}}`)
	model := modelInfoBFirst(t, result)

	catalog := modelInfoBCatalogModel(t, "anthropic", "claude-sonnet-4-6")
	if model.Name != "one-conflicted-deployment" {
		t.Errorf("name = %q", model.Name)
	}
	modelInfoBMatchCatalog(t, model, catalog)
	modelInfoBLevelsEqual(t, model.ThinkingLevelMap, modelInfoBNoLevels())
	if model.LiteLLMPolicy != nil {
		t.Errorf("litellmPolicy = %s, want none", modelInfoBJSON(model.LiteLLMPolicy))
	}
}

func TestModelInfoB_DoesNotRestoreRouteNameKimiPolicyAfterAnIntraRowAuthorityConflict(t *testing.T) {
	result, collected := modelInfoBDiscover(t, `{"model_name":"kimi-k2.6-vanity","litellm_params":{"model":"openai/gpt-4o"},`+
		`"model_info":{"id":"one","mode":"chat","base_model":"anthropic/claude-sonnet-4-6"}}`)
	model := modelInfoBFirst(t, result)

	modelInfoBMatch(t, model, `{"id":"kimi-k2.6-vanity","reasoning":true}`)
	modelInfoBLevelsEqual(t, model.ThinkingLevelMap, modelInfoBNoLevels())
	if modelInfoBHasKey(model.Compat, "requiresReasoningContentOnAssistantMessages") {
		t.Errorf("compat = %s, want no requiresReasoningContentOnAssistantMessages", modelInfoBJSON(model.Compat))
	}
	if model.LiteLLMPolicy != nil {
		t.Errorf("litellmPolicy = %s, want none", modelInfoBJSON(model.LiteLLMPolicy))
	}
	if strings.Contains(modelInfoBStderrJoined(collected), "strict tool-message repair is withheld") {
		t.Errorf("diagnostics = %q", collected.Lines())
	}
}

func TestModelInfoB_RetainsPublicEffortEvidenceWhileWithholdingConflictingFamilyPolicy(t *testing.T) {
	options := modelInfoBModelsDev(t, `{"openai":{"models":{"kimi-k2.5":{"reasoning_options":{"type":"effort","values":["low","medium","high"]}}}}}`)
	base := modelInfoBRows(t, `{"model_name":"conflicting-family-hit","litellm_params":{"model":"openai/private-gpt","allowed_openai_params":["reasoning_effort"]},`+
		`"model_info":{"id":"one","mode":"chat","litellm_provider":"openai","base_model":"openai/kimi-k2.5","supports_reasoning":true}}`)

	model := modelInfoBFirst(t, discover(t, base, options))

	modelInfoBMatch(t, model, `{"id":"conflicting-family-hit","reasoning":true,"compat":{"supportsStore":false}}`)
	if modelInfoBHasKey(model.Compat, "cacheControlFormat") {
		t.Errorf("compat = %s, want no cacheControlFormat", modelInfoBJSON(model.Compat))
	}
	modelInfoBLevelsEqual(t, model.ThinkingLevelMap, modelInfoBLevels("", "", "low", "medium", "high", "", ""))
	if model.LiteLLMPolicy != nil {
		t.Errorf("litellmPolicy = %s, want none", modelInfoBJSON(model.LiteLLMPolicy))
	}
}

func TestModelInfoB_AppliesAnthropicCacheCompatibilityToAFableBackedAlias(t *testing.T) {
	result, _ := modelInfoBDiscover(t,
		`{"model_name":"team-writer","litellm_params":{"model":"bedrock/fable-5"},"model_info":{"id":"one","mode":"chat"}}`)
	modelInfoBMatch(t, modelInfoBFirst(t, result), `{"id":"team-writer","compat":{"supportsStore":false,"cacheControlFormat":"anthropic"}}`)
}

func TestModelInfoB_AppliesAnthropicCacheCompatibilityToAnEvidenceFreeFableRoute(t *testing.T) {
	result, _ := modelInfoBDiscover(t, `{"model_name":"team-fable-5","model_info":{"id":"one","mode":"chat"}}`)
	modelInfoBMatch(t, modelInfoBFirst(t, result), `{"id":"team-fable-5","compat":{"supportsStore":false,"cacheControlFormat":"anthropic"}}`)
}

func TestModelInfoB_ClassifiesACodexDeploymentIdentityAsOpenAI(t *testing.T) {
	resolution := ResolveModelInfoCatalog(entryFrom(t, `{"model_name":"codex-route",`+
		`"litellm_params":{"model":"azure/codex-mini"},"model_info":{"mode":"chat"}}`), nil)
	if resolution == nil || resolution.Provider != "openai" || resolution.CatalogModelID != "codex-mini" {
		t.Errorf("resolution = %+v", resolution)
	}
}

func TestModelInfoB_DoesNotEnrichAnUnqualifiedRouteFromAnUnrelatedProviderCatalog(t *testing.T) {
	result, _ := modelInfoBDiscover(t, `{"model_name":"gpt-4o","model_info":{"mode":"chat"}}`)
	modelInfoBIncomplete(t, modelInfoBFirst(t, result), "gpt-4o")
}

func TestModelInfoB_DiagnosesConflictingProviderEvidenceWithinOneDeployment(t *testing.T) {
	result, collected := modelInfoBDiscover(t, `{"model_name":"conflicting-authority",`+
		`"litellm_params":{"model":"openai/gpt-4o","custom_llm_provider":"anthropic"},"model_info":{"id":"one","mode":"chat"}}`)

	modelInfoBMatch(t, modelInfoBFirst(t, result), `{"name":"conflicting-authority (incomplete metadata)","reasoning":false}`)
	if !strings.Contains(modelInfoBStderrJoined(collected), "conflicting-authority") {
		t.Errorf("diagnostics = %q", collected.Lines())
	}
}

func TestModelInfoB_AppliesCompleteMoonshotCompatToAVanityRouteFromUnanimousDeploymentEvidence(t *testing.T) {
	result, _ := modelInfoBDiscover(t, `{"model_name":"internal/prod-chat","litellm_params":{"model":"MoOnShOt/KiMi-K2.6"},"model_info":{"id":"one","mode":"chat"}}`)
	model := modelInfoBFirst(t, result)

	want := ai.ModelCompat{
		SupportsStore: boolPtr(false), SupportsDeveloperRole: boolPtr(false), SupportsReasoningEffort: boolPtr(false),
		SupportsStrictMode: boolPtr(false), MaxTokensField: "max_tokens",
	}
	if model.Compat == nil || !reflect.DeepEqual(*model.Compat, want) {
		t.Errorf("compat = %s, want %s", modelInfoBJSON(model.Compat), modelInfoBJSON(want))
	}
	if model.LiteLLMPolicy == nil || !model.LiteLLMPolicy.NormalizeStrictToolMessages {
		t.Errorf("litellmPolicy = %s, want normalizeStrictToolMessages", modelInfoBJSON(model.LiteLLMPolicy))
	}
}

func TestModelInfoB_AppliesKimiCompatToAFireworksAccountScopedPathRoutedByCustomLLMProvider(t *testing.T) {
	result, collected := modelInfoBDiscover(t, `{"model_name":"fireworks/kimi-k2p6",`+
		`"litellm_params":{"model":"accounts/fireworks/models/kimi-k2p6","custom_llm_provider":"fireworks_ai"},"model_info":{"id":"one","mode":"chat"}}`)

	modelInfoBMatch(t, modelInfoBFirst(t, result).Compat, `{"maxTokensField":"max_tokens","supportsStrictMode":false}`)
	if strings.Contains(modelInfoBStderrJoined(collected), "conflicting deployment family evidence") {
		t.Errorf("diagnostics = %q", collected.Lines())
	}
}

func TestModelInfoB_KeepsStrictRepairButWithholdsVisibilitySuppressionForMixedKimiThinkingModes(t *testing.T) {
	result, _ := modelInfoBDiscover(t,
		`{"model_name":"mixed-kimi-mode-route","litellm_params":{"model":"moonshot/kimi-k2.6"},"model_info":{"id":"normal","mode":"chat"}}`,
		`{"model_name":"mixed-kimi-mode-route","litellm_params":{"model":"moonshot/kimi-k2-thinking"},"model_info":{"id":"thinking","mode":"chat"}}`)

	modelInfoBPolicyEquals(t, modelInfoBFirst(t, result).LiteLLMPolicy, types.LiteLLMModelPolicy{NormalizeStrictToolMessages: true})
}

func TestModelInfoB_WithholdsStrictToolRepairForPartialMoonshotEvidenceAndReportsAVanityRoute(t *testing.T) {
	result, collected := modelInfoBDiscover(t,
		`{"model_name":"internal/mixed-tool-route","litellm_params":{"model":"moonshot/kimi-k2.6"},"model_info":{"id":"a","mode":"chat"}}`,
		`{"model_name":"internal/mixed-tool-route","litellm_params":{"model":"internal/opaque"},"model_info":{"id":"b","mode":"chat"}}`)

	policy := modelInfoBFirst(t, result).LiteLLMPolicy
	if policy == nil || policy.NormalizeStrictToolMessages || policy.NormalizeThinkTags {
		t.Errorf("litellmPolicy = %s, want strict repair and think tags withheld", modelInfoBJSON(policy))
	}
	withheld := 0
	for _, line := range collected.Lines() {
		if strings.Contains(line, "strict tool-message repair is withheld") {
			withheld++
		}
	}
	if withheld != 1 {
		t.Errorf("withheld diagnostics = %d, want 1 (%q)", withheld, collected.Lines())
	}
	if !strings.Contains(modelInfoBStderrJoined(collected), "internal/mixed-tool-route") {
		t.Errorf("diagnostics = %q", collected.Lines())
	}
}

// Issue #184: LiteLLM omits supported_openai_params for a deployment its model map does not describe,
// so an explicit supports_reasoning is the operator's opt-in.
func TestModelInfoB_KeepsPisStandardLevelsForAnOffMapDeploymentThatOptsIntoReasoning(t *testing.T) {
	options, _ := testOptions(t)
	base := modelInfoBServeAll(t, 200, modelInfoBody(`{"model_name":"opaque-chat-reasoner","litellm_params":{"model":"internal/reasoner"},`+
		`"model_info":{"id":"one","mode":"chat","supports_reasoning":true}}`))
	discovered := modelInfoBFirst(t, discover(t, base, options))

	if discovered.Compat == nil || !modelInfoBBool(discovered.Compat.SupportsReasoningEffort, true) {
		t.Errorf("compat = %s, want supportsReasoningEffort", modelInfoBJSON(discovered.Compat))
	}
	model := &ai.Model{ProviderMeta: ai.ProviderMetadata{Reasoning: discovered.Reasoning}, ThinkingLevelMap: discovered.ThinkingLevelMap}
	want := []ai.ThinkingLevel{ai.ThinkingOff, ai.ThinkingMinimal, ai.ThinkingLow, ai.ThinkingMedium, ai.ThinkingHigh}
	if got := ai.GetSupportedThinkingLevels(model); !slices.Equal(got, want) {
		t.Errorf("supported levels = %v, want %v", got, want)
	}
}

func TestModelInfoB_DeniesChatReasoningLevelsWithoutAnAcceptedCarrier(t *testing.T) {
	result, _ := modelInfoBDiscover(t, `{"model_name":"opaque-chat-reasoner","litellm_params":{"model":"internal/reasoner"},`+
		`"model_info":{"id":"one","mode":"chat","supported_openai_params":["temperature"],"supports_reasoning":true}}`)
	model := modelInfoBFirst(t, result)
	modelInfoBMatch(t, model, `{"api":"openai-completions","reasoning":true}`)
	modelInfoBLevelsEqual(t, model.ThinkingLevelMap, modelInfoBNoLevels())
}

func TestModelInfoB_DeniesResponsesReasoningLevelsWhenTheOnlyAcceptedControlIsChatThinking(t *testing.T) {
	result, _ := modelInfoBDiscover(t, `{"model_name":"thinking-only-responses",`+
		`"litellm_params":{"model":"moonshot/kimi-k2.6","allowed_openai_params":["thinking"]},`+
		`"model_info":{"id":"one","mode":"responses","supports_reasoning":true}}`)
	model := modelInfoBFirst(t, result)
	modelInfoBMatch(t, model, `{"api":"openai-responses","reasoning":true}`)
	modelInfoBLevelsEqual(t, model.ThinkingLevelMap, modelInfoBNoLevels())
	if model.LiteLLMResponsesReasoningControl {
		t.Error("litellmResponsesReasoningControl = true, want unset")
	}
}

func TestModelInfoB_DeniesResponsesReasoningLevelsUnlessEveryDeploymentAcceptsReasoningEffort(t *testing.T) {
	effort := `{"model_name":"mixed-control-responses","litellm_params":{"model":"moonshot/kimi-k2.6","allowed_openai_params":["reasoning_effort","thinking"]},` +
		`"model_info":{"id":"effort","mode":"responses","supports_reasoning":true}}`
	thinking := `{"model_name":"mixed-control-responses","litellm_params":{"model":"moonshot/kimi-k2.6","allowed_openai_params":["thinking"]},` +
		`"model_info":{"id":"thinking","mode":"responses","supports_reasoning":true}}`
	for _, rows := range [][]string{{effort, thinking}, {thinking, effort}} {
		result, _ := modelInfoBDiscover(t, rows...)
		model := modelInfoBFirst(t, result)
		modelInfoBMatch(t, model, `{"api":"openai-responses","reasoning":true}`)
		modelInfoBLevelsEqual(t, model.ThinkingLevelMap, modelInfoBNoLevels())
		if model.LiteLLMResponsesReasoningControl {
			t.Error("litellmResponsesReasoningControl = true, want unset")
		}
	}
}

func TestModelInfoB_PreservesAcceptedResponsesReasoningControlThroughCachedEnrichment(t *testing.T) {
	result, _ := modelInfoBDiscover(t, `{"model_name":"effort-responses",`+
		`"litellm_params":{"model":"moonshot/kimi-k3","allowed_openai_params":["reasoning_effort"]},`+
		`"model_info":{"id":"one","mode":"responses","supports_reasoning":true}}`)
	model := modelInfoBFirst(t, result)

	if model.API != ai.APIOpenAIResponses || !model.LiteLLMResponsesReasoningControl {
		t.Errorf("api = %q, litellmResponsesReasoningControl = %v", model.API, model.LiteLLMResponsesReasoningControl)
	}
	modelInfoBMatch(t, model.ThinkingLevelMap, `{"low":"low","high":"high","max":null}`)
	modelInfoBLevelsEqual(t, EnrichCachedModel(model).ThinkingLevelMap, model.ThinkingLevelMap)
}

func TestModelInfoB_KeepsAResponsesModeDeploymentOnResponsesWhenDiscoveredThroughHealth(t *testing.T) {
	base := mockEndpoints(t, map[string]http.HandlerFunc{
		"/model/info?litellm_model_id=uuid-1": jsonResponse(200, modelInfoBody(
			`{"model_name":"responses-route","litellm_params":{"model":"openai/gpt-5.5"},"model_info":{"id":"uuid-1","mode":"responses"}}`)),
		"/model/info": jsonResponse(403, map[string]any{}),
		"/v1/models":  jsonResponse(404, map[string]any{}),
		"/health": jsonResponse(200, map[string]any{"healthy_endpoints": []any{
			map[string]any{"model": "openai/gpt-5.5", "model_id": "uuid-1"}}}),
		modelInfoBProbe: jsonResponse(404, map[string]any{}),
	})
	options, _ := testOptions(t)
	result := discover(t, base, options)

	if result.Source != types.SourceHealth {
		t.Errorf("source = %q", result.Source)
	}
	modelInfoBMatch(t, modelInfoBFirst(t, result), `{"id":"responses-route","api":"openai-responses"}`)
}

func TestModelInfoB_FallsBackToTheHealthRouteNameWhenADeploymentRowsOwnNameIsUnreadable(t *testing.T) {
	// The detail row's `model_name` is unusable, but `/health` named the route, so the model must survive
	// under that name rather than being discarded.
	base := mockEndpoints(t, map[string]http.HandlerFunc{
		"/model/info?litellm_model_id=uuid-1": jsonResponse(200, modelInfoBody(`{"model_name":7,"model_info":{"mode":"chat"}}`)),
		"/model/info":                         jsonResponse(403, map[string]any{}),
		"/v1/models":                          jsonResponse(404, map[string]any{}),
		"/health": jsonResponse(200, map[string]any{"healthy_endpoints": []any{
			map[string]any{"model": "named-route", "model_id": "uuid-1"}}}),
	})
	options, _ := testOptions(t)
	result := discover(t, base, options)

	if result.Source != types.SourceHealth {
		t.Errorf("source = %q", result.Source)
	}
	if ids := modelInfoBIDs(result.Models); !reflect.DeepEqual(ids, []string{"named-route"}) {
		t.Errorf("ids = %v", ids)
	}
}

func TestModelInfoB_PreservesRouteOnlyKimiThinkTagPolicyWithoutEnablingRequestControls(t *testing.T) {
	result, _ := modelInfoBDiscover(t, `{"model_name":"kimi-k2.6","model_info":{"id":"only","mode":"chat"}}`)
	model := modelInfoBFirst(t, result)

	modelInfoBMatch(t, model, `{"litellmPolicy":{"normalizeStrictToolMessages":false,"normalizeThinkTags":true,"suppressReasoningVisibility":false}}`)
	if model.Reasoning {
		t.Error("reasoning = true, want false")
	}
	if model.ThinkingLevelMap != nil {
		t.Errorf("thinkingLevelMap = %s, want none", modelInfoBJSON(model.ThinkingLevelMap))
	}
	modelInfoBMatch(t, model.Compat, `{"supportsReasoningEffort":false}`)
}

func TestModelInfoB_DoesNotUseAPublicRouteNameAsEvidenceForConflictingDuplicateDeploymentIDs(t *testing.T) {
	// One deployment id with two disagreeing backends is not a single deployment, so the
	// catalog-resolvable route name must not enrich the group.
	result, _ := modelInfoBDiscover(t,
		`{"model_name":"openai/gpt-5.5","model_info":{"id":"same","mode":"chat"}}`,
		`{"model_name":"openai/gpt-5.5","model_info":{"id":"same","mode":"chat","max_input_tokens":64000},"litellm_params":{"model":"internal/mystery"}}`)
	model := modelInfoBFirst(t, result)

	modelInfoBMatch(t, model, `{"id":"openai/gpt-5.5","name":"openai/gpt-5.5 (incomplete metadata)","reasoning":false,"input":["text"],`+
		`"cost":{"input":0,"output":0,"cacheRead":0,"cacheWrite":0},"maxTokens":16384}`)
	if model.ThinkingLevelMap != nil {
		t.Errorf("thinkingLevelMap = %s, want none", modelInfoBJSON(model.ThinkingLevelMap))
	}
}

func TestModelInfoB_WithholdsAndDiagnosesAMixedChatStyleAndIncompatibleRoute(t *testing.T) {
	for _, tc := range []struct{ name, first, second string }{
		{"chat first", "chat", "embedding"},
		{"embedding first", "embedding", "chat"},
		{"Responses first", "responses", "embedding"},
		{"embedding before Responses", "embedding", "responses"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			route := "mixed-route-" + strings.ReplaceAll(tc.name, " ", "-")
			result, collected := modelInfoBDiscover(t,
				fmt.Sprintf(`{"model_name":%q,"model_info":{"id":"first","mode":%q}}`, route, tc.first),
				fmt.Sprintf(`{"model_name":%q,"model_info":{"id":"second","mode":%q}}`, route, tc.second))

			if len(result.Models) != 0 {
				t.Errorf("models = %v, want none", modelInfoBIDs(result.Models))
			}
			lines := collected.Lines()
			if len(lines) != 1 {
				t.Fatalf("diagnostics = %q, want one", lines)
			}
			for _, want := range []string{"1 route group(s) mix chat-style and explicitly incompatible deployment modes", route} {
				if !strings.Contains(lines[0], want) {
					t.Errorf("diagnostic %q lacks %q", lines[0], want)
				}
			}
		})
	}
}

func TestModelInfoB_BoundsIncompatibleModeDiagnosticsAndReportsEachRouteOnce(t *testing.T) {
	var rows []string
	for _, route := range []string{"bounded-mode-a", "bounded-mode-b", "bounded-mode-c", "bounded-mode-d"} {
		rows = append(rows,
			fmt.Sprintf(`{"model_name":%q,"model_info":{"id":"%s-chat","mode":"chat"}}`, route, route),
			fmt.Sprintf(`{"model_name":%q,"model_info":{"id":"%s-embedding","mode":"embedding"}}`, route, route))
	}
	options, collected := testOptions(t)
	run := func() { discover(t, modelInfoBRows(t, rows...), options) }

	run()
	lines := collected.Lines()
	if len(lines) != 1 {
		t.Fatalf("diagnostics = %q, want one", lines)
	}
	if !strings.Contains(lines[0], "bounded-mode-a, bounded-mode-b, bounded-mode-c (+1 more)") || strings.Contains(lines[0], "bounded-mode-d") {
		t.Errorf("diagnostic = %q", lines[0])
	}

	run()
	if lines := collected.Lines(); len(lines) != 1 {
		t.Errorf("diagnostics after repeat = %q, want one", lines)
	}
}

func TestModelInfoB_StaysSilentWhenProviderIdentityIsUnanimousOrWhollyUnknown(t *testing.T) {
	// The Vitest case serves the same rows as the "resolved fallback identities disagree" case in the same
	// process and expects no write: the "unknown" route was already reported there. Seed that state with
	// a discovery whose reports are discarded, then expect the repeat to stay silent.
	options, collected := testOptions(t)
	seeding := options
	seeding.Report = nil
	rows := modelInfoBAgreedAndUnknownRows()
	discover(t, modelInfoBRows(t, rows...), seeding)

	discover(t, modelInfoBRows(t, rows...), options)

	if lines := collected.Lines(); len(lines) != 0 {
		t.Errorf("diagnostics = %q, want none", lines)
	}
}
