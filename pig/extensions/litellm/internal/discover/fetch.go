// Ports src/discover.ts (options, fetchJson, diagnostics) and probeProxyVersion from src/proxy-version.ts
package discover

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/modelgroups"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/proxyversion"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

// Options is DiscoveryOptions plus the discoverModels extras. The AbortSignal is the ctx parameter of
// every function.
type Options struct {
	types.DiscoveryOptions
	// OnProgress receives progress messages; Silent suppresses them.
	OnProgress func(string)
	Silent     bool
	// Report receives every message the TypeScript writes to process.stderr, including its trailing
	// newline. nil discards them.
	Report func(string)
	// HTTPClient nil means a default client. It also serves the public-catalog fetch.
	HTTPClient *http.Client
}

// timeout is `options.timeoutMs ?? DEFAULT_TIMEOUT_MS`.
func (o Options) timeout() time.Duration {
	if o.Timeout != nil {
		return *o.Timeout
	}
	return defaultTimeout
}

func (o Options) client() *http.Client {
	if o.HTTPClient != nil {
		return o.HTTPClient
	}
	return http.DefaultClient
}

func (o Options) report(message string) {
	if o.Report != nil {
		o.Report(message)
	}
}

func (o Options) progress(message string) {
	if !o.Silent && o.OnProgress != nil {
		o.OnProgress(message)
	}
}

// get issues the authenticated GET every discovery request uses. The caller must call cancel once it
// is done with the response, because the timeout also bounds the body read.
func (o Options) get(parent context.Context, rawURL, apiKey string) (*http.Response, context.CancelFunc, error) {
	ctx, cancel := context.WithTimeout(parent, o.timeout())
	request, err := http.NewRequestWithContext(ctx, http.MethodGet, rawURL, nil)
	if err != nil {
		cancel()
		return nil, nil, errors.New("LiteLLM request could not be built")
	}
	for name, value := range o.Headers {
		request.Header.Set(name, value)
	}
	request.Header.Set("Authorization", "Bearer "+apiKey)
	request.Header.Set("Accept", "application/json")
	response, err := o.client().Do(request)
	if err != nil {
		cancel()
		// A *url.Error embeds the request URL, which may carry credentials.
		var urlError *url.Error
		if errors.As(err, &urlError) {
			err = urlError.Err
		}
		// The caller's own cancellation reports its cause, as AbortSignal rejects with signal.reason.
		if parent.Err() != nil {
			err = context.Cause(parent)
		}
		return nil, nil, err
	}
	return response, cancel, nil
}

// maxJSONDepth bounds nesting before decoding; encoding/json refuses beyond 10,000 and would fail the
// whole response for one pathological value.
const maxJSONDepth = 256

// capJSONDepth replaces every subtree nested deeper than limit with null. It is a single string-aware
// pass and returns body itself, unallocated, when nothing is replaced.
func capJSONDepth(body []byte, limit int) []byte {
	var out []byte
	copied := 0 // body[:copied] is already in out
	depth := 0
	for i := 0; i < len(body); i++ {
		switch body[i] {
		case '"':
			for i++; i < len(body) && body[i] != '"'; i++ {
				if body[i] == '\\' {
					i++
				}
			}
		case '{', '[':
			if depth < limit {
				depth++
				continue
			}
			end := matchingClose(body, i)
			if end < 0 {
				return body
			}
			if out == nil {
				out = make([]byte, 0, len(body))
			}
			out = append(append(out, body[copied:i]...), "null"...)
			copied = end + 1
			i = end
		case '}', ']':
			depth--
		}
	}
	if out == nil {
		return body
	}
	return append(out, body[copied:]...)
}

// matchingClose returns the index of the bracket closing the one at start, or -1 when unterminated.
func matchingClose(body []byte, start int) int {
	depth := 0
	for i := start; i < len(body); i++ {
		switch body[i] {
		case '"':
			for i++; i < len(body) && body[i] != '"'; i++ {
				if body[i] == '\\' {
					i++
				}
			}
		case '{', '[':
			depth++
		case '}', ']':
			depth--
			if depth == 0 {
				return i
			}
		}
	}
	return -1
}

// FetchResult is `{ ok: true; data } | { ok: false; status }`.
type FetchResult[T any] struct {
	OK     bool
	Data   T
	Status int
}

// FetchJSON is fetchJson: a non-2xx reply is a result, a transport or decoding failure is an error.
func FetchJSON[T any](ctx context.Context, rawURL, apiKey string, options Options) (FetchResult[T], error) {
	var result FetchResult[T]
	response, cancel, err := options.get(ctx, rawURL, apiKey)
	if err != nil {
		return result, err
	}
	defer cancel()
	defer response.Body.Close()
	if response.StatusCode < 200 || response.StatusCode > 299 {
		result.Status = response.StatusCode
		return result, nil
	}
	body, err := io.ReadAll(response.Body)
	if err != nil {
		return result, err
	}
	body = capJSONDepth(body, maxJSONDepth)
	if err := json.Unmarshal(body, &result.Data); err != nil {
		// Like response.json() followed by `data.data ?? []`: only a body that is not JSON at all fails;
		// valid JSON of another shape yields no data.
		if !json.Valid(body) {
			return result, errors.New("LiteLLM response is not valid JSON")
		}
		var zero T
		result.Data = zero
	}
	result.OK = true
	return result, nil
}

// ProbeProxyVersion is probeProxyVersion. Best effort by design: the header rides on an error path
// LiteLLM does not document, so a missing header, an unreachable proxy or a timeout all mean "unknown"
// (nil) and never fail discovery. The reply body is never read, and the raw header is never reported.
func ProbeProxyVersion(ctx context.Context, base, apiKey string, options Options) *proxyversion.Version {
	response, cancel, err := options.get(ctx, base+proxyversion.ProbePath, apiKey)
	if err != nil {
		return nil
	}
	defer cancel()
	response.Body.Close()
	return proxyversion.FromHeader(response.Header)
}

const diagnosticRouteSample = 3

// reportedSet remembers reported routes, so a persistent misconfiguration is announced once rather
// than on every background refresh and every `/model` open. Keyed by route rather than a single flag
// so a newly ambiguous route is still reported.
type reportedSet struct {
	mu     sync.Mutex
	routes map[string]bool
}

func (s *reportedSet) claim(routes []string) []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.routes == nil {
		s.routes = map[string]bool{}
	}
	var unreported []string
	for _, route := range routes {
		if !s.routes[route] {
			s.routes[route] = true
			unreported = append(unreported, route)
		}
	}
	return unreported
}

func (s *reportedSet) reset() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.routes = nil
}

var (
	reportedConflictingFamilyRoutes reportedSet
	reportedWithheldRepairRoutes    reportedSet
	reportedDefaultedContextRoutes  reportedSet
	reportedAmbiguousRoutes         reportedSet
	reportedIncompatibleModeRoutes  reportedSet
)

func reportBoundedRoutes(set *reportedSet, routes []string, describe func(count int) string, options Options) {
	unreported := set.claim(routes)
	if len(unreported) == 0 {
		return
	}
	hidden := len(unreported) - diagnosticRouteSample
	sample := strings.Join(unreported[:min(len(unreported), diagnosticRouteSample)], ", ")
	more := ""
	if hidden > 0 {
		more = fmt.Sprintf(" (+%d more)", hidden)
	}
	options.report(fmt.Sprintf("%s: %s%s\n", describe(len(unreported)), sample, more))
}

// routeCollector gathers route ids for one diagnostic; a nil collector discards them.
type routeCollector struct{ routes []string }

func (c *routeCollector) add(route string) {
	if c != nil {
		c.routes = append(c.routes, route)
	}
}

func (c *routeCollector) list() []string {
	if c == nil {
		return nil
	}
	return c.routes
}

// reportAmbiguousCatalogAuthority: withholding catalog authority can be invisible in the model name
// when the router supplies complete prices; limits and other catalog-derived metadata may still use
// conservative defaults. Report that degradation regardless of LITELLM_VERBOSE_DISCOVERY, carrying
// only a count and bounded public route ids.
func reportAmbiguousCatalogAuthority(routes []string, options Options) {
	reportBoundedRoutes(&reportedAmbiguousRoutes, unique(routes), func(count int) string {
		return fmt.Sprintf("LiteLLM discovery: %d route group(s) have missing or conflicting deployment provider evidence; "+
			"catalog limits, pricing, and reasoning metadata are withheld", count)
	}, options)
}

func reportIncompatibleDeploymentModes(routes []string, options Options) {
	reportBoundedRoutes(&reportedIncompatibleModeRoutes, unique(routes), func(count int) string {
		return fmt.Sprintf("LiteLLM discovery: %d route group(s) mix chat-style and explicitly incompatible deployment modes; "+
			"the routes are withheld because not every deployment can accept chat requests", count)
	}, options)
}

func reportConflictingFamilyEvidence(routes []string, options Options) {
	reportBoundedRoutes(&reportedConflictingFamilyRoutes, unique(routes), func(count int) string {
		return fmt.Sprintf("LiteLLM discovery: %d route group(s) have conflicting deployment family evidence; "+
			"family-specific compatibility and request policy are withheld", count)
	}, options)
}

func reportWithheldToolRepair(routes []string, options Options) {
	reportBoundedRoutes(&reportedWithheldRepairRoutes, unique(routes), func(count int) string {
		return fmt.Sprintf("LiteLLM discovery: %d route group(s) look Moonshot-backed but not every deployment evidences it; "+
			"strict tool-message repair is withheld because it rewrites outbound messages and is unproven for a "+
			"deployment that has not identified its backend. Moonshot tool calls on these routes may fail until every "+
			"deployment declares its backend", count)
	}, options)
}

func reportWithheldToolRepairForModels(models []types.DiscoveredModel, options Options) {
	var routes []string
	for _, model := range models {
		policy := model.LiteLLMPolicy
		if policy != nil && !policy.NormalizeStrictToolMessages && !policy.NormalizeGeminiReasoningEffort {
			routes = append(routes, model.ID)
		}
	}
	reportWithheldToolRepair(routes, options)
}

func reportDefaultedContext(routes []string, options Options) {
	reportBoundedRoutes(&reportedDefaultedContextRoutes, routes, func(count int) string {
		return fmt.Sprintf("LiteLLM discovery: %d route(s) default contextWindow to %d; "+
			"set model_info.max_input_tokens on every deployment in each route", count, modelgroups.DefaultContextWindow())
	}, options)
}

func unique(values []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, value := range values {
		if !seen[value] {
			seen[value] = true
			out = append(out, value)
		}
	}
	return out
}

// jsonString is JSON.stringify for a string.
func jsonString(value string) string {
	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	encoder.SetEscapeHTML(false)
	_ = encoder.Encode(value)
	return strings.TrimSuffix(buffer.String(), "\n")
}

func verboseDiscovery() bool { return os.Getenv("LITELLM_VERBOSE_DISCOVERY") == "1" }
