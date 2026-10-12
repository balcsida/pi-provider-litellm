package discover

// Ports the "context window fallback diagnostic", "discoverModels timeout", "catalog provider
// candidates" and (first ten cases of) "discoverModels via /model/info" describe blocks of
// tests/discover.test.ts.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/modelgroups"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

func discover(t *testing.T, base string, options Options) *types.DiscoveryResult {
	t.Helper()
	result, err := DiscoverModels(context.Background(), base, "sk-test", options)
	if err != nil {
		t.Fatalf("DiscoverModels: %v", err)
	}
	return result
}

func contextWindows(models []types.DiscoveredModel) map[string]int {
	windows := map[string]int{}
	for _, model := range models {
		windows[model.ID] = model.ContextWindow
	}
	return windows
}

func TestContextWindowFallbackDiagnostic(t *testing.T) {
	// Reported routes are remembered per process; testOptions starts each case from a clean set.
	verbose := func(t *testing.T) {
		t.Setenv("LITELLM_VERBOSE_DISCOVERY", "1")
		t.Setenv("LITELLM_DEFAULT_CONTEXT_WINDOW", "")
	}

	for _, limit := range []string{"undefined", "null"} {
		t.Run("reports a defaulted window for limit "+limit+" with an escaped route", func(t *testing.T) {
			verbose(t)
			route, _ := json.Marshal("private\"\n\x1b[31m")
			info := `"mode":"chat"`
			if limit == "null" {
				info += `,"max_input_tokens":null`
			}
			base := mockEndpoints(t, map[string]http.HandlerFunc{
				"/model/info": jsonResponse(200, modelInfoBody(fmt.Sprintf(`{"model_name":%s,"model_info":{%s}}`, route, info))),
			})
			options, stderr := testOptions(t)

			result := discover(t, base, options)

			if got := result.Models[0]; got.ContextWindow != 128000 || got.MaxTokens != 16384 {
				t.Fatalf("model = %+v", got)
			}
			lines := stderr.Lines()
			if len(lines) != 1 {
				t.Fatalf("stderr lines = %q, want 1", lines)
			}
			message := lines[0]
			for _, want := range []string{`"private\"\n\u001b[31m"`, "1 route(s) default contextWindow to 128000", "model_info.max_input_tokens"} {
				if !strings.Contains(message, want) {
					t.Errorf("message %q lacks %q", message, want)
				}
			}
			if strings.Contains(strings.TrimRight(message, "\n"), "\n") || strings.Contains(message, "\x1b") {
				t.Errorf("message %q spans lines or carries an escape", message)
			}
		})
	}

	for _, verboseValue := range []string{"undefined", "0", "true"} {
		t.Run("stays quiet with LITELLM_VERBOSE_DISCOVERY="+verboseValue, func(t *testing.T) {
			verbose(t)
			if verboseValue == "undefined" {
				if err := os.Unsetenv("LITELLM_VERBOSE_DISCOVERY"); err != nil {
					t.Fatal(err)
				}
			} else {
				t.Setenv("LITELLM_VERBOSE_DISCOVERY", verboseValue)
			}
			base := mockEndpoints(t, map[string]http.HandlerFunc{
				"/model/info": jsonResponse(200, modelInfoBody(`{"model_name":"private-route","model_info":{"mode":"chat"}}`)),
			})
			options, stderr := testOptions(t)

			result := discover(t, base, options)

			if result.Models[0].ContextWindow != 128000 || len(stderr.Lines()) != 0 {
				t.Errorf("contextWindow = %d, stderr = %q", result.Models[0].ContextWindow, stderr.Lines())
			}
		})
	}

	for _, tc := range []struct{ source, row string }{
		{"explicit", `{"model_name":"private-route","model_info":{"mode":"chat","max_input_tokens":128000}}`},
		{"catalog", `{"model_name":"private-route","litellm_params":{"model":"openai/gpt-4o"},"model_info":{"mode":"chat"}}`},
	} {
		t.Run("does not mistake a "+tc.source+" window equal to the default for a fallback", func(t *testing.T) {
			verbose(t)
			base := mockEndpoints(t, map[string]http.HandlerFunc{"/model/info": jsonResponse(200, modelInfoBody(tc.row))})
			options, stderr := testOptions(t)

			result := discover(t, base, options)

			if result.Models[0].ContextWindow != 128000 || len(stderr.Lines()) != 0 {
				t.Errorf("contextWindow = %d, stderr = %q", result.Models[0].ContextWindow, stderr.Lines())
			}
		})
	}

	for _, tc := range []struct {
		knownLimit, expected int
		warns                bool
	}{
		{64000, 64000, false},
		{1048576, 922000, true},
	} {
		t.Run(fmt.Sprintf("reports only a winning configured fallback beside limit %d", tc.knownLimit), func(t *testing.T) {
			verbose(t)
			t.Setenv("LITELLM_DEFAULT_CONTEXT_WINDOW", "922000")
			rows := []string{
				fmt.Sprintf(`{"model_name":"private-route","model_info":{"mode":"chat","max_input_tokens":%d}}`, tc.knownLimit),
				`{"model_name":"private-route","model_info":{"mode":"chat"}}`,
			}
			for _, data := range [][]string{rows, {rows[1], rows[0]}} {
				base := mockEndpoints(t, map[string]http.HandlerFunc{"/model/info": jsonResponse(200, modelInfoBody(data...))})
				options, stderr := testOptions(t)

				result := discover(t, base, options)

				if result.Models[0].ContextWindow != tc.expected {
					t.Errorf("contextWindow = %d, want %d", result.Models[0].ContextWindow, tc.expected)
				}
				lines := stderr.Lines()
				if tc.warns {
					if len(lines) != 1 || !strings.Contains(lines[0], "default contextWindow to 922000") {
						t.Errorf("stderr = %q, want one 922000 warning", lines)
					}
				} else if len(lines) != 0 {
					t.Errorf("stderr = %q, want none", lines)
				}
			}
		})
	}

	t.Run("reports each published wildcard child once, excluding exact and tighter measured routes", func(t *testing.T) {
		verbose(t)
		ids := []string{"private/child", "private/child", "private/exact", "private/small-child"}
		list := make([]map[string]string, len(ids))
		for i, id := range ids {
			list[i] = map[string]string{"id": id}
		}
		base := mockEndpoints(t, map[string]http.HandlerFunc{
			"/model/info": jsonResponse(200, modelInfoBody(
				`{"model_name":"private/*","model_info":{"mode":"chat"}}`,
				`{"model_name":"private/small*","model_info":{"mode":"chat","max_input_tokens":64000}}`,
				`{"model_name":"private/exact","model_info":{"mode":"chat","max_input_tokens":128000}}`,
			)),
			"/v1/models": jsonResponse(200, map[string]any{"data": list}),
		})
		options, stderr := testOptions(t)

		result := discover(t, base, options)

		want := map[string]int{"private/child": 128000, "private/exact": 128000, "private/small-child": 64000}
		if got := contextWindows(result.Models); !reflect.DeepEqual(got, want) || len(result.Models) != len(want) {
			t.Errorf("models = %v, want %v", got, want)
		}
		lines := stderr.Lines()
		pattern := regexp.MustCompile(`^LiteLLM discovery: 1 route\(s\) default contextWindow to 128000; .*: "private/child"\n$`)
		if len(lines) != 1 || !pattern.MatchString(lines[0]) {
			t.Errorf("stderr = %q", lines)
		}
	})

	for _, mode := range []string{"chat", "embedding"} {
		t.Run("does not report unpublished "+mode+" routes", func(t *testing.T) {
			verbose(t)
			base := mockEndpoints(t, map[string]http.HandlerFunc{
				"/model/info": jsonResponse(200, modelInfoBody(fmt.Sprintf(`{"model_name":"private/*","model_info":{"mode":%q}}`, mode))),
				"/v1/models":  jsonResponse(200, map[string]any{"data": []any{}}),
			})
			options, stderr := testOptions(t)

			result := discover(t, base, options)

			if len(result.Models) != 0 || strings.Contains(strings.Join(stderr.Lines(), "\n"), "default contextWindow") {
				t.Errorf("models = %v, stderr = %q", result.Models, stderr.Lines())
			}
		})
	}

	t.Run("reports many defaulted routes on one bounded line, once per process", func(t *testing.T) {
		verbose(t)
		rows := make([]string, 50)
		for i := range rows {
			rows[i] = fmt.Sprintf(`{"model_name":"route-%d","model_info":{"mode":"chat"}}`, i)
		}
		base := mockEndpoints(t, map[string]http.HandlerFunc{"/model/info": jsonResponse(200, modelInfoBody(rows...))})
		options, stderr := testOptions(t)

		discover(t, base, options)
		discover(t, base, options)

		lines := stderr.Lines()
		pattern := regexp.MustCompile(`^LiteLLM discovery: 50 route\(s\) default contextWindow to 128000; .* \(\+47 more\)\n$`)
		if len(lines) != 1 || !pattern.MatchString(lines[0]) {
			t.Errorf("stderr = %q", lines)
		}
	})
}

func TestDiscoverModelsTimeout(t *testing.T) {
	t.Run("aborts the fetch after timeoutMs", func(t *testing.T) {
		base := mockEndpoints(t, map[string]http.HandlerFunc{
			"/model/info": func(_ http.ResponseWriter, r *http.Request) { <-r.Context().Done() },
		})
		options, _ := testOptions(t)
		timeout := 30 * time.Millisecond
		options.Timeout = &timeout

		start := time.Now()
		_, err := DiscoverModels(context.Background(), base, "sk-test", options)

		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("err = %v, want a deadline error", err)
		}
		if elapsed := time.Since(start); elapsed >= 500*time.Millisecond {
			t.Errorf("took %v, want under 500ms", elapsed)
		}
	})
}

func TestCatalogProviderCandidates(t *testing.T) {
	for _, id := range []string{
		"claude-opus-5", "claude-sonnet-5", "claude-fable-5", "claude-opus-4-5-20251101", "claude-sonnet-4-5-20250929",
		"claude-haiku-4-5-20251001", "opus-4-7", "sonnet-4-6", "haiku-4-5", "opus-4.7", "fable-5", "opus-5",
	} {
		t.Run("resolves the bare Anthropic backend id "+id+" from the shared lookup rule", func(t *testing.T) {
			base := mockEndpoints(t, map[string]http.HandlerFunc{
				"/model/info": jsonResponse(200, modelInfoBody(
					fmt.Sprintf(`{"model_name":"route-%s","litellm_params":{"model":%q},"model_info":{"mode":"chat"}}`, id, id))),
			})
			options, _ := testOptions(t)

			result := discover(t, base, options)

			if model := result.Models[0]; strings.Contains(model.Name, "metadata") || model.Cost.Input <= 0 {
				t.Errorf("name = %q, input cost = %v", model.Name, model.Cost.Input)
			}
		})
	}

	for _, id := range []string{"claudia-x", "opusclip-2", "haiku"} {
		t.Run("does not treat "+id+" as an Anthropic alias", func(t *testing.T) {
			base := mockEndpoints(t, map[string]http.HandlerFunc{
				"/model/info": jsonResponse(200, modelInfoBody(fmt.Sprintf(`{"model_name":%q,"model_info":{"mode":"chat"}}`, id))),
			})
			options, _ := testOptions(t)

			result := discover(t, base, options)

			if model := result.Models[0]; model.Name != id+" (incomplete metadata)" || !reflect.DeepEqual(model.Cost, ai.ModelCost{}) {
				t.Errorf("name = %q, cost = %+v", model.Name, model.Cost)
			}
		})
	}

	for _, tc := range []struct {
		name, backend, adapter string
		context                int
	}{
		{"decorated mixed-case Kimi", "ToGeThEr/MoOnShOtAi/KiMi-K3@prod", "together_ai", 1_048_576},
		{"mixed-case Claude", "AnThRoPiC/ClAuDe-SoNnEt-4-6", "", 1_000_000},
		{"decorated Claude", "BeDrOcK/US.AnThRoPiC.ClAuDe-SoNnEt-4-6-V1:0", "bedrock", 1_000_000},
	} {
		t.Run("retains catalog metadata for a "+tc.name+" provider-qualified backend", func(t *testing.T) {
			info := `"mode":"chat"`
			if tc.adapter != "" {
				info += fmt.Sprintf(`,"litellm_provider":%q`, tc.adapter)
			}
			base := mockEndpoints(t, map[string]http.HandlerFunc{
				"/model/info": jsonResponse(200, modelInfoBody(
					fmt.Sprintf(`{"model_name":"private/catalog-route","litellm_params":{"model":%q},"model_info":{%s}}`, tc.backend, info))),
			})
			// Options{}: the models.dev lookup is allowed, but the guard refuses it like an unstubbed fetch.
			options, _ := testOptions(t)
			options.ModelsDev = nil

			model := discover(t, base, options).Models[0]

			if model.ID != "private/catalog-route" || model.Name != "private/catalog-route" ||
				model.ContextWindow != tc.context || model.MaxTokens <= 0 || model.Cost.Input <= 0 {
				t.Errorf("model = %+v", model)
			}
		})
	}

	t.Run("does not strip arbitrary suffixes while normalizing provider-qualified ids", func(t *testing.T) {
		resolved := ResolveModelInfoCatalog(entryFrom(t, `{"model_name":"private/catalog-route",`+
			`"litellm_params":{"model":"anthropic/Claude-Sonnet-4-6-preview"},"model_info":{"mode":"chat"}}`), nil)

		want := &modelgroups.CatalogResolution{Provider: "anthropic", CatalogModelID: "anthropic/Claude-Sonnet-4-6-preview"}
		if !reflect.DeepEqual(resolved, want) {
			t.Errorf("resolved = %+v, want %+v", resolved, want)
		}
	})
}

// writeModelsDevCache seeds a models.dev cache file and returns its path.
func writeModelsDevCache(t *testing.T, catalog string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "models-dev.json")
	if err := os.WriteFile(path, []byte(`{"fetchedAt":1,"catalog":`+catalog+`}`), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func firstModel(t *testing.T, base string, options Options) types.DiscoveredModel {
	t.Helper()
	return discover(t, base, options).Models[0]
}

func TestDiscoverModelsViaModelInfo(t *testing.T) {
	for _, tc := range []struct {
		mode   string
		api    ai.API
		params string
	}{
		{"chat", ai.APIOpenAICompletions, `{"model":"chatgpt/gpt-5.6-sol"}`},
		{"responses", ai.APIOpenAIResponses, `{"model":"gpt-5.6-sol","custom_llm_provider":"chatgpt"}`},
	} {
		t.Run("enriches a ChatGPT subscription alias while retaining "+tc.mode+" routing", func(t *testing.T) {
			endpoint := "/v1/chat/completions"
			if tc.api == ai.APIOpenAIResponses {
				endpoint = "/v1/responses"
			}
			row := fmt.Sprintf(`{"model_name":"high","litellm_params":%s,"model_info":{"mode":%q,"supported_endpoints":[%q]}}`,
				tc.params, tc.mode, endpoint)
			// The synchronous fallback must work even when public-catalog loading exceeds its budget.
			resolved := ResolveModelInfoCatalog(entryFrom(t, row), nil)
			if resolved == nil || resolved.Provider != "chatgpt" || resolved.CatalogModelID != "gpt-5.6-sol" ||
				resolved.ContextWindow == nil || *resolved.ContextWindow != 272_000 ||
				resolved.MaxTokens == nil || *resolved.MaxTokens != 128_000 {
				t.Fatalf("resolved = %+v", resolved)
			}
			base := mockEndpoints(t, map[string]http.HandlerFunc{"/model/info": jsonResponse(200, modelInfoBody(row))})
			options, _ := testOptions(t)

			model := firstModel(t, base, options)

			if model.ID != "high" || model.API != tc.api || model.ContextWindow != 272_000 || model.MaxTokens != 128_000 || !model.Reasoning {
				t.Errorf("model = %+v", model)
			}
		})
	}

	t.Run("enriches the backend identity from models.dev before synchronous group reduction", func(t *testing.T) {
		cachePath := writeModelsDevCache(t, `{"fireworks-ai":{"models":{"accounts/fireworks/models/kimi-k3":{`+
			`"modalities":{"input":["text","image"]},"limit":{"context":131072,"output":16384},`+
			`"cost":{"input":0.4,"output":2,"cache_read":0.2,"cache_write":0.8}}}}}`)
		base := mockEndpoints(t, map[string]http.HandlerFunc{
			"/model/info": jsonResponse(200, modelInfoBody(`{"model_name":"kimi-k3",`+
				`"litellm_params":{"model":"azure_ai/FW-Kimi-K3","custom_llm_provider":"azure"},`+
				`"model_info":{"id":"deployment-a","mode":"chat","litellm_provider":"azure",`+
				`"base_model":"fireworks_ai/accounts/fireworks/models/kimi-k3","max_input_tokens":262144,"input_cost_per_token":0.000001}}`)),
		})
		options, _ := testOptions(t)
		options.ModelsDevCachePath = cachePath

		model := firstModel(t, base, options)

		wantCost := ai.ModelCost{Input: 1, Output: 2, CacheRead: 0.2, CacheWrite: 0.8}
		if model.ID != "kimi-k3" || !reflect.DeepEqual(model.Input, []string{"text", "image"}) ||
			model.ContextWindow != 131_072 || model.MaxTokens != 16_384 || !reflect.DeepEqual(model.Cost, wantCost) {
			t.Errorf("model = %+v", model)
		}
	})

	t.Run("fills partial models.dev cost from Pi catalog pricing", func(t *testing.T) {
		cachePath := writeModelsDevCache(t, `{"openai":{"models":{"gpt-5.5":{"cost":{"input":7}}}}}`)
		base := mockEndpoints(t, map[string]http.HandlerFunc{
			"/model/info": jsonResponse(200, modelInfoBody(
				`{"model_name":"gpt-route","litellm_params":{"model":"openai/gpt-5.5"},"model_info":{"mode":"chat"}}`)),
		})
		options, _ := testOptions(t)
		options.ModelsDevCachePath = cachePath

		model := firstModel(t, base, options)

		// toMatchObject ignores the catalog's price tiers, so only the four rates are compared.
		wantCost := ai.ModelCost{Input: 7, Output: 30, CacheRead: 0.5, CacheWrite: 0}
		if cost := model.Cost; model.Name != "gpt-route" ||
			cost.Input != wantCost.Input || cost.Output != wantCost.Output || cost.CacheRead != wantCost.CacheRead || cost.CacheWrite != wantCost.CacheWrite {
			t.Errorf("name = %q, cost = %+v", model.Name, model.Cost)
		}
	})

	t.Run("rejects a negative models.dev input price and marks the reduced model incomplete", func(t *testing.T) {
		cachePath := writeModelsDevCache(t, `{"private":{"models":{"priced-model":{"modalities":{"input":["text"]},`+
			`"limit":{"context":64000,"output":8000},"cost":{"input":-1,"output":2,"cache_read":0,"cache_write":0}}}}}`)
		base := mockEndpoints(t, map[string]http.HandlerFunc{
			"/model/info": jsonResponse(200, modelInfoBody(`{"model_name":"private-route","litellm_params":{"model":"private/priced-model"},`+
				`"model_info":{"mode":"chat","supports_reasoning":false,"supports_vision":false}}`)),
		})
		options, _ := testOptions(t)
		options.ModelsDevCachePath = cachePath

		model := firstModel(t, base, options)

		wantCost := ai.ModelCost{Input: 0, Output: 2, CacheRead: 0, CacheWrite: 0}
		if model.Name != "private-route (incomplete metadata)" || !reflect.DeepEqual(model.Cost, wantCost) {
			t.Errorf("name = %q, cost = %+v", model.Name, model.Cost)
		}
	})

	t.Run("keeps partial models.dev pricing incomplete without a Pi catalog fallback", func(t *testing.T) {
		cachePath := writeModelsDevCache(t, `{"private":{"models":{"priced-model":{"modalities":{"input":["text"]},`+
			`"limit":{"context":64000,"output":8000},"cost":{"input":7}}}}}`)
		base := mockEndpoints(t, map[string]http.HandlerFunc{
			"/model/info": jsonResponse(200, modelInfoBody(`{"model_name":"private-route","litellm_params":{"model":"private/priced-model"},`+
				`"model_info":{"mode":"chat","supports_reasoning":false}}`)),
		})
		options, _ := testOptions(t)
		options.ModelsDevCachePath = cachePath

		model := firstModel(t, base, options)

		wantCost := ai.ModelCost{Input: 7, Output: 0, CacheRead: 0, CacheWrite: 0}
		if model.Name != "private-route (incomplete metadata)" || !reflect.DeepEqual(model.Cost, wantCost) {
			t.Errorf("name = %q, cost = %+v", model.Name, model.Cost)
		}
	})

	for _, reasoning := range []string{"undefined", "false"} {
		t.Run("combines explicit reasoning="+reasoning+" with models.dev effort evidence", func(t *testing.T) {
			cachePath := writeModelsDevCache(t, `{"private":{"models":{"reasoner":{"reasoning_options":{"type":"effort","values":["low","high"]}}}}}`)
			info := `"mode":"chat"`
			if reasoning == "false" {
				info += `,"supports_reasoning":false`
			}
			base := mockEndpoints(t, map[string]http.HandlerFunc{
				"/model/info": jsonResponse(200, modelInfoBody(
					fmt.Sprintf(`{"model_name":"private-route","litellm_params":{"model":"private/reasoner"},"model_info":{%s}}`, info))),
			})
			options, _ := testOptions(t)
			options.ModelsDevCachePath = cachePath

			model := firstModel(t, base, options)

			if model.Reasoning != (reasoning != "false") || model.Name != "private-route (incomplete metadata)" {
				t.Fatalf("reasoning = %v, name = %q", model.Reasoning, model.Name)
			}
			if reasoning == "false" {
				if model.ThinkingLevelMap != nil {
					t.Errorf("levels = %v, want none", model.ThinkingLevelMap)
				}
			} else if !reflect.DeepEqual(model.ThinkingLevelMap, modelgroups.NoTransmissibleLevels()) {
				t.Errorf("levels = %v, want every level denied", model.ThinkingLevelMap)
			}
		})
	}

	// modelsDevRoute stands in for https://models.dev/api.json: it counts requests and answers each one
	// with release's document once release is closed.
	type modelsDevRoute struct {
		requests atomic.Int32
		release  chan struct{}
		body     string
	}
	newModelsDevRoute := func(t *testing.T, body string) (*modelsDevRoute, func(*http.Request) (*http.Response, error)) {
		route := &modelsDevRoute{release: make(chan struct{}), body: body}
		t.Cleanup(func() {
			select {
			case <-route.release:
			default:
				close(route.release)
			}
		})
		return route, func(r *http.Request) (*http.Response, error) {
			route.requests.Add(1)
			select {
			case <-route.release:
				var document any
				if err := json.Unmarshal([]byte(route.body), &document); err != nil {
					return nil, err
				}
				return jsonHTTPResponse(200, document), nil
			case <-r.Context().Done():
				return nil, r.Context().Err()
			}
		}
	}
	waitForRequests := func(t *testing.T, route *modelsDevRoute, want int32) {
		t.Helper()
		deadline := time.Now().Add(5 * time.Second)
		for route.requests.Load() != want {
			if time.Now().After(deadline) {
				t.Fatalf("models.dev requests = %d, want %d", route.requests.Load(), want)
			}
			time.Sleep(5 * time.Millisecond)
		}
	}
	type outcome struct {
		result *types.DiscoveryResult
		err    error
	}
	start := func(ctx context.Context, base string, options Options) <-chan outcome {
		done := make(chan outcome, 1)
		go func() {
			result, err := DiscoverModels(ctx, base, "sk-test", options)
			done <- outcome{result, err}
		}()
		return done
	}
	await := func(t *testing.T, done <-chan outcome) outcome {
		t.Helper()
		select {
		case result := <-done:
			return result
		case <-time.After(10 * time.Second):
			t.Fatal("discovery did not finish")
			return outcome{}
		}
	}

	t.Run("lets initial public-catalog callers abort independently", func(t *testing.T) {
		route, modelsDev := newModelsDevRoute(t,
			`{"openai":{"models":{"gpt-5.5":{"limit":{"context":1050000},"cost":{"input":5,"output":30}}}}}`)
		base := mockEndpoints(t, map[string]http.HandlerFunc{
			"/model/info": jsonResponse(200, modelInfoBody(
				`{"model_name":"gpt-route","litellm_params":{"model":"openai/gpt-5.5"},"model_info":{"mode":"chat"}}`)),
		})
		options, _ := testOptions(t)
		options.ModelsDev = nil
		options.HTTPClient = networkGuard(modelsDev)
		options.ModelsDevCachePath = filepath.Join(t.TempDir(), "models-dev.json")

		firstCtx, abortFirst := context.WithCancelCause(context.Background())
		defer abortFirst(nil)
		first := start(firstCtx, base, options)
		second := start(context.Background(), base, options)

		waitForRequests(t, route, 1)
		abortFirst(errors.New("first aborted"))
		if got := await(t, first); got.err == nil || got.err.Error() != "first aborted" {
			t.Fatalf("first err = %v, want %q", got.err, "first aborted")
		}
		close(route.release)

		got := await(t, second)
		if got.err != nil || got.result.Models[0].ID != "gpt-route" || got.result.Models[0].ContextWindow != 1_050_000 {
			t.Fatalf("second = %+v, %v", got.result, got.err)
		}
		if route.requests.Load() != 1 {
			t.Errorf("models.dev requests = %d, want 1", route.requests.Load())
		}
	})

	t.Run("returns the activation seed before its caller signal aborts when models.dev hangs", func(t *testing.T) {
		_, modelsDev := newModelsDevRoute(t, `{}`)
		base := mockEndpoints(t, map[string]http.HandlerFunc{
			"/model/info": jsonResponse(200, modelInfoBody(
				`{"model_name":"private-route","litellm_params":{"model":"private/catalog-model"},"model_info":{"mode":"chat"}}`)),
		})
		options, _ := testOptions(t)
		options.ModelsDev = nil
		options.HTTPClient = networkGuard(modelsDev)
		options.ModelsDevCachePath = filepath.Join(t.TempDir(), "models-dev.json")
		timeout := 5 * time.Second
		options.Timeout = &timeout
		ctx, abort := context.WithCancelCause(context.Background())
		timer := time.AfterFunc(3*time.Second, func() { abort(errors.New("seed aborted")) })
		defer timer.Stop()
		defer abort(nil)

		model := firstModelWithContext(t, ctx, base, options)

		if model.ID != "private-route" || model.Name != "private-route (incomplete metadata)" || model.ContextWindow != 128_000 {
			t.Errorf("model = %+v", model)
		}
		if ctx.Err() != nil {
			t.Error("the caller signal aborted before discovery returned")
		}
	})

	t.Run("lets each caller stop waiting for best-effort enrichment at its own budget", func(t *testing.T) {
		route, modelsDev := newModelsDevRoute(t,
			`{"private":{"models":{"catalog-model":{"limit":{"context":1050000},"cost":{"input":5,"output":30}}}}}`)
		routes := map[string]http.HandlerFunc{
			"/model/info": jsonResponse(200, modelInfoBody(
				`{"model_name":"private-route","litellm_params":{"model":"private/catalog-model"},"model_info":{"mode":"chat"}}`)),
		}
		shortBase, longBase := mockEndpoints(t, routes), mockEndpoints(t, routes)
		cachePath := filepath.Join(t.TempDir(), "models-dev.json")
		configure := func(timeout time.Duration) Options {
			options, _ := testOptions(t)
			options.ModelsDev = nil
			options.HTTPClient = networkGuard(modelsDev)
			options.ModelsDevCachePath = cachePath
			options.Timeout = &timeout
			return options
		}

		short := start(context.Background(), shortBase, configure(30*time.Millisecond))
		long := start(context.Background(), longBase, configure(30*time.Second))

		waitForRequests(t, route, 1)
		shortResult := await(t, short)
		if shortResult.err != nil {
			t.Fatal(shortResult.err)
		}
		if model := shortResult.result.Models[0]; model.ID != "private-route" ||
			model.Name != "private-route (incomplete metadata)" || model.ContextWindow != 128_000 {
			t.Errorf("short model = %+v", model)
		}
		close(route.release)

		longResult := await(t, long)
		if longResult.err != nil || longResult.result.Models[0].ContextWindow != 1_050_000 {
			t.Fatalf("long = %+v, %v", longResult.result, longResult.err)
		}
		if route.requests.Load() != 1 {
			t.Errorf("models.dev requests = %d, want 1", route.requests.Load())
		}
	})

	t.Run("keeps models with a null mode", func(t *testing.T) {
		base := mockEndpoints(t, map[string]http.HandlerFunc{
			"/model/info": jsonResponse(200, modelInfoBody(`{"model_name":"local/model","model_info":{"mode":null}}`)),
		})
		options, _ := testOptions(t)

		result := discover(t, base, options)

		var ids []string
		for _, model := range result.Models {
			ids = append(ids, model.ID)
		}
		if !reflect.DeepEqual(ids, []string{"local/model"}) {
			t.Errorf("ids = %v", ids)
		}
	})
}

func firstModelWithContext(t *testing.T, ctx context.Context, base string, options Options) types.DiscoveredModel {
	t.Helper()
	result, err := DiscoverModels(ctx, base, "sk-test", options)
	if err != nil {
		t.Fatalf("DiscoverModels: %v", err)
	}
	return result.Models[0]
}
