package discover

// Shared helpers for the discover tests ported from tests/discover.test.ts. A test that needs a LiteLLM
// proxy follows the Vitest shape:
//
//	base := mockEndpoints(t, map[string]http.HandlerFunc{
//		"/model/info": jsonResponse(200, map[string]any{"data": []any{entry}}),
//	})
//	opts, stderr := testOptions(t)
//	result, err := DiscoverModels(context.Background(), base, "sk-test", opts)
//
// mockEndpoints is the Go form of the Vitest `mockEndpoints`: only the listed path suffixes are served
// (matched against the request URI, query included, longest suffix first) and any other request fails
// the test and drops the connection, so a stray fetch can never be satisfied by another endpoint's
// payload. Use newMockProxy when the test also needs the requests that were made.
//
// testOptions returns Options that cannot reach the network: ModelsDev is false (no models.dev fetch),
// and HTTPClient refuses every host except loopback. Tests that exercise the public catalog replace
// HTTPClient with networkGuard. The returned *reports collects what the TypeScript writes to
// process.stderr. The "already reported" diagnostic sets are process-wide, as in the TypeScript module,
// so testOptions also clears them (the Vitest `vi.resetModules()`); a test that runs several discoveries
// and wants a fresh process between them calls resetReportedRoutes itself.
//
// Build /model/info rows with entryFrom (a JSON literal decoded through the production lenient
// decoder, so mistyped fields behave as in the proxy) or fixtureJSON for the shared fixtures.

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/fixtures"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

// mockProxy is an httptest server serving only the routes it was given.
type mockProxy struct {
	// URL is the base URL to hand to DiscoverModels.
	URL      string
	mu       sync.Mutex
	requests []string
}

// Requests returns the request URIs (path and query) seen so far, in arrival order.
func (p *mockProxy) Requests() []string {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]string(nil), p.requests...)
}

// newMockProxy starts a server answering the path suffixes in routes; the server closes with the test.
func newMockProxy(t *testing.T, routes map[string]http.HandlerFunc) *mockProxy {
	t.Helper()
	proxy := &mockProxy{}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		uri := r.URL.RequestURI()
		proxy.mu.Lock()
		proxy.requests = append(proxy.requests, uri)
		proxy.mu.Unlock()
		best := ""
		for suffix := range routes {
			if strings.HasSuffix(uri, suffix) && len(suffix) > len(best) {
				best = suffix
			}
		}
		if best == "" {
			t.Errorf("unexpected URL: %s", uri)
			// Abort the connection so the client sees a network failure, like the rejected fetch.
			panic(http.ErrAbortHandler)
		}
		routes[best](w, r)
	}))
	t.Cleanup(server.Close)
	proxy.URL = server.URL
	return proxy
}

// mockEndpoints is the Vitest `mockEndpoints`: it returns the base URL of a server serving only routes.
func mockEndpoints(t *testing.T, routes map[string]http.HandlerFunc) string {
	t.Helper()
	return newMockProxy(t, routes).URL
}

// jsonResponse is the Vitest `jsonResponse`: a handler replying with status and body marshalled as JSON.
// A json.RawMessage or []byte body is sent as is, which lets a test serve malformed documents.
func jsonResponse(status int, body any) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		var encoded []byte
		switch value := body.(type) {
		case json.RawMessage:
			encoded = value
		case []byte:
			encoded = value
		default:
			var err error
			if encoded, err = json.Marshal(body); err != nil {
				http.Error(w, err.Error(), http.StatusInternalServerError)
				return
			}
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		_, _ = w.Write(encoded)
	}
}

// jsonHTTPResponse is jsonResponse for an http.RoundTripper.
func jsonHTTPResponse(status int, body any) *http.Response {
	encoded, _ := json.Marshal(body)
	return &http.Response{
		StatusCode: status,
		Header:     http.Header{"Content-Type": {"application/json"}},
		Body:       io.NopCloser(bytes.NewReader(encoded)),
	}
}

// roundTripFunc adapts a function to http.RoundTripper.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// networkGuard returns a client that serves loopback hosts normally, hands https://models.dev requests
// to modelsDev (nil refuses them), and fails every other request loudly like the Vitest `unstubbed
// fetch`.
func networkGuard(modelsDev func(*http.Request) (*http.Response, error)) *http.Client {
	return &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		switch host := r.URL.Hostname(); {
		case host == "127.0.0.1" || host == "localhost" || host == "::1":
			return http.DefaultTransport.RoundTrip(r)
		case host == "models.dev" && modelsDev != nil:
			return modelsDev(r)
		}
		return nil, errors.New("unstubbed fetch: " + r.URL.String())
	})}
}

// reports collects the diagnostics DiscoverModels writes to stderr.
type reports struct {
	mu    sync.Mutex
	lines []string
}

func (r *reports) add(message string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.lines = append(r.lines, message)
}

// Lines returns every reported message, trailing newline included, like process.stderr.write calls.
func (r *reports) Lines() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]string(nil), r.lines...)
}

// resetReportedRoutes forgets which routes were already announced (the per-process diagnostic sets).
func resetReportedRoutes() {
	for _, set := range []*reportedSet{
		&reportedConflictingFamilyRoutes,
		&reportedWithheldRepairRoutes,
		&reportedDefaultedContextRoutes,
		&reportedAmbiguousRoutes,
		&reportedIncompatibleModeRoutes,
	} {
		set.reset()
	}
}

// testOptions returns offline Options (see the file comment) and the collector of their diagnostics.
func testOptions(t *testing.T) (Options, *reports) {
	t.Helper()
	resetReportedRoutes()
	t.Cleanup(resetReportedRoutes)
	collected := &reports{}
	modelsDev := false
	options := Options{HTTPClient: networkGuard(nil), Report: collected.add}
	options.ModelsDev = &modelsDev
	return options, collected
}

// fixtureJSON decodes a shared fixture, e.g. fixtureJSON[types.ModelInfoResponse](t, "proxy/prod-model-info-2026-09-04.json").
func fixtureJSON[T any](t *testing.T, rel string) T {
	t.Helper()
	var value T
	path := fixtures.Path(t, rel)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &value); err != nil {
		t.Fatalf("fixture %s: %v", rel, err)
	}
	return value
}

// entryFrom decodes a /model/info row written as a JSON literal.
func entryFrom(t testing.TB, raw string) types.ModelInfoEntry {
	t.Helper()
	var entry types.ModelInfoEntry
	if err := json.Unmarshal([]byte(raw), &entry); err != nil {
		t.Fatalf("entry %s: %v", raw, err)
	}
	return entry
}

// modelInfoBody wraps rows (JSON literals) in the `{ "data": [...] }` envelope of /model/info.
func modelInfoBody(rows ...string) json.RawMessage {
	return json.RawMessage(`{"data":[` + strings.Join(rows, ",") + `]}`)
}
