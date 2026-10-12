package litellm

// Ports tests/budget.test.ts and the "budget footer" and "/litellm-budget" cases of tests/budget-status.test.ts.
// The footer and command cases drive budgetController through a fake host, clock and timer, because sdk.Context
// cannot be faked; the "through the extension" cases are ported in extension_test.go over a recording registrar.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"reflect"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/fixtures"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

// ---- proxy and fixtures ----

var (
	budgetReset = time.Date(2026, 11, 1, 0, 0, 0, 0, time.UTC).UnixMilli()
	budgetNow   = time.Date(2026, 10, 7, 12, 0, 0, 0, time.UTC).UnixMilli()
)

const (
	budgetDay = int64(86_400_000)
	// Shaped like an OIDC id_token: three dot-separated parts, which is how LiteLLM tells a JWT from a key.
	budgetJWT = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJVMSJ9.c2ln"
)

func budgetAuth() types.LiteLLMRuntimeAuth {
	return types.LiteLLMRuntimeAuth{BaseURL: "https://proxy.example.com", APIKey: "sk-test", Headers: map[string]string{"x-gateway": "g1"}}
}

func budgetFixture(t *testing.T, name string) any {
	t.Helper()
	raw, err := os.ReadFile(fixtures.Path(t, "budget/"+name+".json"))
	if err != nil {
		t.Fatal(err)
	}
	var value any
	if err := json.Unmarshal(raw, &value); err != nil {
		t.Fatal(err)
	}
	return value
}

func budgetObject(t *testing.T, name string) map[string]any {
	t.Helper()
	return budgetFixture(t, name).(map[string]any)
}

// budgetAbsent removes a field in budgetKeyWith.
type budgetAbsent struct{}

// budgetKeyWith is keyWith: the fixture with fields of its "info" replaced.
func budgetKeyWith(t *testing.T, name string, info map[string]any) map[string]any {
	t.Helper()
	body := budgetObject(t, name)
	fields := body["info"].(map[string]any)
	for key, value := range info {
		if _, remove := value.(budgetAbsent); remove {
			delete(fields, key)
		} else {
			fields[key] = value
		}
	}
	return body
}

// budgetTeamWithOrg is the team fixture charged to organization "O".
func budgetTeamWithOrg(t *testing.T) map[string]any {
	body := budgetObject(t, "team-info")
	body["team_info"].(map[string]any)["organization_id"] = "O"
	return body
}

type budgetReply struct {
	status int
	body   any
	raw    string
	err    error
}

func budgetOK(body any) budgetReply { return budgetReply{status: 200, body: body} }

func budgetCode(code int) budgetReply {
	return budgetReply{status: code, body: map[string]any{"error": "proxy says: secret detail"}}
}

func budgetJSON(body any) func() budgetReply { return func() budgetReply { return budgetOK(body) } }
func budgetStatusReply(code int) func() budgetReply {
	return func() budgetReply { return budgetCode(code) }
}

func (r budgetReply) response(request *http.Request) (*http.Response, error) {
	if r.err != nil {
		return nil, r.err
	}
	raw, contentType := r.raw, "text/html"
	if raw == "" {
		encoded, _ := json.Marshal(r.body)
		raw, contentType = string(encoded), "application/json"
	}
	return &http.Response{
		StatusCode: r.status, Status: fmt.Sprintf("%d %s", r.status, http.StatusText(r.status)),
		Header: http.Header{"Content-Type": {contentType}}, Body: io.NopCloser(strings.NewReader(raw)), Request: request,
	}, nil
}

type budgetRoundTrip func(*http.Request) (*http.Response, error)

func (f budgetRoundTrip) RoundTrip(request *http.Request) (*http.Response, error) { return f(request) }

// installBudgetTransport replaces http.DefaultTransport, which discover.FetchJSON's default client uses, so no
// test touches the network. Tests that use it never run in parallel.
func installBudgetTransport(t *testing.T, transport budgetRoundTrip) {
	t.Helper()
	previous := http.DefaultTransport
	http.DefaultTransport = transport
	t.Cleanup(func() { http.DefaultTransport = previous })
}

type budgetRecorder struct {
	mu    sync.Mutex
	seen  []string
	hosts []string
}

func (r *budgetRecorder) record(request *http.Request) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.seen = append(r.seen, request.URL.RequestURI())
	r.hosts = append(r.hosts, request.URL.Host+request.URL.Path)
}

func (r *budgetRecorder) paths() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.seen)
}

func (r *budgetRecorder) sorted() []string {
	paths := r.paths()
	slices.Sort(paths)
	return paths
}

func (r *budgetRecorder) count(path string) int {
	return strings.Count("\x00"+strings.Join(r.paths(), "\x00")+"\x00", "\x00"+path+"\x00")
}

func (r *budgetRecorder) reset() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.seen, r.hosts = nil, nil
}

type budgetExpect struct{ token, gateway string }

var budgetDefaultExpect = budgetExpect{"sk-test", "g1"}

// mockBudgetProxy maps path+query to a fresh reply and records the requests, like mockProxy in budget-helpers.ts.
func mockBudgetProxy(t *testing.T, routes map[string]func() budgetReply, expected ...budgetExpect) *budgetRecorder {
	t.Helper()
	want := budgetDefaultExpect
	if len(expected) > 0 {
		want = expected[0]
	}
	recorder := &budgetRecorder{}
	installBudgetTransport(t, func(request *http.Request) (*http.Response, error) {
		recorder.record(request)
		if got := request.Header.Get("Authorization"); got != "Bearer "+want.token {
			t.Errorf("authorization for %s = %q, want token %q", request.URL.RequestURI(), got, want.token)
		}
		if got := request.Header.Get("x-gateway"); got != want.gateway {
			t.Errorf("x-gateway for %s = %q, want %q", request.URL.RequestURI(), got, want.gateway)
		}
		route, found := routes[request.URL.RequestURI()]
		if !found {
			return budgetCode(404).response(request)
		}
		return route().response(request)
	})
	return recorder
}

// ---- pollBudget ----

func budgetPollWith(t *testing.T, auth types.LiteLLMRuntimeAuth, denied map[budgetEndpoint]int, previous budgetLevels) budgetPoll {
	t.Helper()
	if denied == nil {
		denied = map[budgetEndpoint]int{}
	}
	return pollBudget(context.Background(), auth, denied, previous, 5000)
}

func budgetPollOnce(t *testing.T) budgetPoll { return budgetPollWith(t, budgetAuth(), nil, nil) }

func budgetPollJWT(t *testing.T, previous budgetLevels) budgetPoll {
	auth := budgetAuth()
	auth.APIKey = budgetJWT
	return budgetPollWith(t, auth, nil, previous)
}

func budgetLevelOf(poll budgetPoll, name budgetLevelName) (budgetLevel, bool) {
	if !poll.OK {
		return budgetLevel{}, false
	}
	level, ok := poll.Levels[name]
	return level, ok
}

func budgetWantPoll(t *testing.T, got, want budgetPoll) {
	t.Helper()
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("poll = %+v, want %+v", got, want)
	}
}

func budgetWantSeen(t *testing.T, got []string, want ...string) {
	t.Helper()
	if !slices.Equal(got, want) {
		t.Fatalf("requests = %v, want %v", got, want)
	}
}

func budgetPersonalRoutes(t *testing.T) map[string]func() budgetReply {
	return map[string]func() budgetReply{
		"/key/info":     budgetJSON(budgetObject(t, "key-info-personal")),
		"/v2/user/info": budgetJSON(budgetObject(t, "user-info-v2")),
	}
}

func budgetTeamKeyRoutes(t *testing.T) map[string]func() budgetReply {
	return map[string]func() budgetReply{
		"/key/info":     budgetJSON(budgetObject(t, "key-info-team-user")),
		"/v2/user/info": budgetJSON(budgetObject(t, "user-info-v2")),
	}
}

var (
	budgetUserLevel   = budgetLevel{Spend: 0.343, MaxBudget: 100, ResetAt: budgetReset}
	budgetTeamLevel   = budgetLevel{Spend: 0.294, MaxBudget: 1000, ResetAt: budgetReset}
	budgetMemberLevel = budgetLevel{Spend: 0.147, MaxBudget: 50}
	budgetOrgLevel    = budgetLevel{Spend: 9210, MaxBudget: 50000}
)

func TestPollBudget(t *testing.T) {
	t.Run("reads key, user, team and member budgets for a key with a user and a team", func(t *testing.T) {
		routes := budgetTeamKeyRoutes(t)
		routes["/team/info?team_id=T"] = budgetJSON(budgetObject(t, "team-info"))
		seen := mockBudgetProxy(t, routes)
		denied := map[budgetEndpoint]int{}
		budgetWantPoll(t, budgetPollWith(t, budgetAuth(), denied, nil), budgetPoll{OK: true, KeyPolled: true,
			Levels: budgetLevels{budgetUser: budgetUserLevel, budgetTeam: budgetTeamLevel, budgetMember: budgetMemberLevel}})
		budgetWantSeen(t, seen.sorted(), "/key/info", "/team/info?team_id=T", "/v2/user/info")
		if len(denied) != 0 {
			t.Fatalf("denied = %v", denied)
		}
	})

	t.Run("reads the key budget and its reset time", func(t *testing.T) {
		seen := mockBudgetProxy(t, budgetPersonalRoutes(t))
		keyReset := time.Date(2026, 10, 8, 0, 0, 0, 0, time.UTC).UnixMilli()
		budgetWantPoll(t, budgetPollOnce(t), budgetPoll{OK: true, KeyPolled: true, Levels: budgetLevels{
			budgetKey: {Spend: 0.147, MaxBudget: 10, ResetAt: keyReset}, budgetUser: budgetUserLevel}})
		budgetWantSeen(t, seen.sorted(), "/key/info", "/v2/user/info")
	})

	t.Run("uses the team default member budget when the caller has no own entry", func(t *testing.T) {
		for _, userID := range []string{"U9", "U2"} {
			routes := budgetTeamKeyRoutes(t)
			routes["/key/info"] = budgetJSON(budgetKeyWith(t, "key-info-team-user", map[string]any{"user_id": userID}))
			routes["/team/info?team_id=T"] = budgetJSON(budgetObject(t, "team-info"))
			mockBudgetProxy(t, routes)
			member, _ := budgetLevelOf(budgetPollOnce(t), budgetMember)
			if want := (budgetLevel{Spend: 0, MaxBudget: 25, ResetAt: budgetReset}); member != want {
				t.Fatalf("%s member = %+v, want %+v", userID, member, want)
			}
		}
	})

	t.Run("skips user and member for a team key without a user", func(t *testing.T) {
		seen := mockBudgetProxy(t, map[string]func() budgetReply{
			"/key/info":            budgetJSON(budgetObject(t, "key-info-team-only")),
			"/team/info?team_id=T": budgetJSON(budgetObject(t, "team-info")),
		})
		got := budgetPollOnce(t)
		budgetWantSeen(t, seen.sorted(), "/key/info", "/team/info?team_id=T")
		budgetWantPoll(t, got, budgetPoll{OK: true, KeyPolled: true, Levels: budgetLevels{budgetTeam: budgetTeamLevel}})
	})

	t.Run("falls back to /user/info when /v2/user/info answers 404 and remembers it", func(t *testing.T) {
		seen := mockBudgetProxy(t, map[string]func() budgetReply{
			"/key/info":     budgetJSON(budgetKeyWith(t, "key-info-team-user", map[string]any{"team_id": nil})),
			"/v2/user/info": budgetStatusReply(404),
			"/user/info":    budgetJSON(budgetObject(t, "user-info-v1")),
		})
		denied := map[budgetEndpoint]int{}
		want := budgetPoll{OK: true, KeyPolled: true, Levels: budgetLevels{budgetUser: budgetUserLevel}}
		budgetWantPoll(t, budgetPollWith(t, budgetAuth(), denied, nil), want)
		if denied[endpointUserV2] != 404 {
			t.Fatalf("denied = %v", denied)
		}
		seen.reset()
		budgetWantPoll(t, budgetPollWith(t, budgetAuth(), denied, nil), want)
		if seen.count("/user/info") == 0 || seen.count("/v2/user/info") != 0 {
			t.Fatalf("requests = %v", seen.paths())
		}
	})

	t.Run("does not fall back to /user/info on other 4xx", func(t *testing.T) {
		seen := mockBudgetProxy(t, map[string]func() budgetReply{
			"/key/info":     budgetJSON(budgetKeyWith(t, "key-info-team-user", map[string]any{"team_id": nil})),
			"/v2/user/info": budgetStatusReply(403),
		})
		denied := map[budgetEndpoint]int{}
		budgetPollWith(t, budgetAuth(), denied, nil)
		if seen.count("/user/info") != 0 || denied[endpointUserV2] != 403 {
			t.Fatalf("requests = %v, denied = %v", seen.paths(), denied)
		}
	})

	t.Run("remembers 4xx endpoints and keeps 429 and 5xx retryable", func(t *testing.T) {
		routes := func(team func() budgetReply) map[string]func() budgetReply {
			r := budgetTeamKeyRoutes(t)
			r["/team/info?team_id=T"] = team
			return r
		}
		seen := mockBudgetProxy(t, routes(budgetStatusReply(403)))
		denied := map[budgetEndpoint]int{}
		budgetPollWith(t, budgetAuth(), denied, nil)
		if denied[endpointTeam] != 403 {
			t.Fatalf("denied = %v", denied)
		}
		seen.reset()
		budgetPollWith(t, budgetAuth(), denied, nil)
		if seen.count("/team/info?team_id=T") != 0 {
			t.Fatalf("a remembered 403 was retried: %v", seen.paths())
		}
		for _, code := range []int{503, 429} {
			mockBudgetProxy(t, routes(budgetStatusReply(code)))
			retry := map[budgetEndpoint]int{}
			previous := budgetLevels{budgetTeam: budgetTeamLevel, budgetMember: budgetMemberLevel}
			result := budgetPollWith(t, budgetAuth(), retry, previous)
			if result.Levels[budgetTeam] != budgetTeamLevel || result.Levels[budgetMember] != budgetMemberLevel || len(retry) != 0 {
				t.Fatalf("code %d: %+v, denied %v", code, result, retry)
			}
		}
	})

	t.Run("still tries the user when /key/info answers 4xx, and skips team and org", func(t *testing.T) {
		// A virtual key's user may be in a team the key is not charged to, such as the master key's default user.
		seen := mockBudgetProxy(t, map[string]func() budgetReply{
			"/key/info":     budgetStatusReply(403),
			"/v2/user/info": budgetJSON(budgetObject(t, "user-info-v2")),
		})
		denied := map[budgetEndpoint]int{}
		result := budgetPollWith(t, budgetAuth(), denied, nil)
		budgetWantSeen(t, seen.sorted(), "/key/info", "/v2/user/info")
		if !result.OK || result.KeyPolled || denied[endpointKey] != 403 {
			t.Fatalf("result = %+v, denied = %v", result, denied)
		}
	})

	t.Run("remembers a 500 like a 4xx and still reads the user level", func(t *testing.T) {
		seen := mockBudgetProxy(t, map[string]func() budgetReply{
			"/key/info":     budgetStatusReply(500),
			"/v2/user/info": budgetJSON(budgetObject(t, "user-info-v2")),
		})
		denied := map[budgetEndpoint]int{}
		budgetWantPoll(t, budgetPollWith(t, budgetAuth(), denied, nil), budgetPoll{OK: true, Levels: budgetLevels{budgetUser: budgetUserLevel}})
		if denied[endpointKey] != 500 {
			t.Fatalf("denied = %v", denied)
		}
		seen.reset()
		budgetPollWith(t, budgetAuth(), denied, nil)
		budgetWantSeen(t, seen.paths(), "/v2/user/info")
	})

	t.Run("takes team, member and org from a JWT user's only team, as a JWT has no key row", func(t *testing.T) {
		routes := map[string]func() budgetReply{
			"/key/info":            budgetStatusReply(500),
			"/v2/user/info":        budgetJSON(budgetObject(t, "user-info-v2")),
			"/team/info?team_id=T": budgetJSON(budgetTeamWithOrg(t)),
			"/organization/list":   budgetJSON(budgetFixture(t, "organization-list")),
		}
		jwt := budgetExpect{budgetJWT, "g1"}
		mockBudgetProxy(t, routes, jwt)
		levels := budgetLevels{budgetUser: budgetUserLevel, budgetTeam: budgetTeamLevel, budgetMember: budgetMemberLevel, budgetOrg: budgetOrgLevel}
		budgetWantPoll(t, budgetPollJWT(t, nil), budgetPoll{OK: true, Levels: levels})

		// A temporary user failure keeps the levels its team list led to.
		routes["/v2/user/info"] = budgetStatusReply(503)
		mockBudgetProxy(t, routes, jwt)
		budgetWantPoll(t, budgetPollJWT(t, levels), budgetPoll{OK: true, Levels: levels})
	})

	t.Run("reads a JWT user's only team from /user/info, and none when the user has several", func(t *testing.T) {
		jwt := budgetExpect{budgetJWT, "g1"}
		mockBudgetProxy(t, map[string]func() budgetReply{
			"/key/info":            budgetStatusReply(404),
			"/v2/user/info":        budgetStatusReply(404),
			"/user/info":           budgetJSON(budgetObject(t, "user-info-v1")),
			"/team/info?team_id=T": budgetJSON(budgetObject(t, "team-info")),
		}, jwt)
		if member, _ := budgetLevelOf(budgetPollJWT(t, nil), budgetMember); member != budgetMemberLevel {
			t.Fatalf("member = %+v", member)
		}

		several := budgetObject(t, "user-info-v2")
		several["teams"] = []any{"T", "T2"}
		seen := mockBudgetProxy(t, map[string]func() budgetReply{
			"/key/info":     budgetStatusReply(500),
			"/v2/user/info": budgetJSON(several),
		}, jwt)
		result := budgetPollJWT(t, nil)
		if len(result.Levels) != 1 || result.Levels[budgetUser] != budgetUserLevel {
			t.Fatalf("levels = %+v", result.Levels)
		}
		for _, path := range seen.paths() {
			if strings.HasPrefix(path, "/team/info") {
				t.Fatalf("polled a team: %v", seen.paths())
			}
		}
	})

	t.Run("ends the poll on a transient /key/info failure", func(t *testing.T) {
		for _, c := range []struct {
			reply  budgetReply
			reason string
		}{
			{budgetCode(503), "HTTP 503"},
			{budgetReply{err: errors.New("offline")}, "request failed"},
			{budgetReply{status: 200, raw: "<html>login</html>"}, "request failed"},
			{budgetOK(map[string]any{}), "malformed response"},
		} {
			seen := mockBudgetProxy(t, map[string]func() budgetReply{"/key/info": func() budgetReply { return c.reply }})
			budgetWantPoll(t, budgetPollOnce(t), budgetPoll{Reason: c.reason})
			budgetWantSeen(t, seen.paths(), "/key/info")
		}
	})

	t.Run("reads the org budget for the key's org, else the team's org", func(t *testing.T) {
		orgs := budgetFixture(t, "organization-list")
		routes := map[string]func() budgetReply{
			"/key/info":          budgetJSON(budgetKeyWith(t, "key-info-team-user", map[string]any{"organization_id": "O", "team_id": nil})),
			"/v2/user/info":      budgetJSON(budgetObject(t, "user-info-v2")),
			"/organization/list": budgetJSON(orgs),
		}
		mockBudgetProxy(t, routes)
		if org, _ := budgetLevelOf(budgetPollOnce(t), budgetOrg); org != budgetOrgLevel {
			t.Fatalf("key org = %+v", org)
		}

		mockBudgetProxy(t, map[string]func() budgetReply{
			"/key/info":            budgetJSON(budgetObject(t, "key-info-team-user")),
			"/v2/user/info":        budgetJSON(budgetObject(t, "user-info-v2")),
			"/team/info?team_id=T": budgetJSON(budgetTeamWithOrg(t)),
			"/organization/list":   budgetJSON(orgs),
		})
		if org, _ := budgetLevelOf(budgetPollOnce(t), budgetOrg); org != budgetOrgLevel {
			t.Fatalf("team org = %+v", org)
		}

		routes["/organization/list"] = budgetJSON(orgs.([]any)[:1])
		mockBudgetProxy(t, routes)
		if _, ok := budgetLevelOf(budgetPollOnce(t), budgetOrg); ok {
			t.Fatal("org of another organization was read")
		}

		seen := mockBudgetProxy(t, map[string]func() budgetReply{
			"/key/info":            budgetJSON(budgetObject(t, "key-info-team-only")),
			"/team/info?team_id=T": budgetJSON(budgetTeamWithOrg(t)),
		})
		budgetPollOnce(t)
		if seen.count("/organization/list") != 0 {
			t.Fatal("polled the org without a user")
		}
	})

	t.Run("keeps the team's org budget when /team/info fails temporarily", func(t *testing.T) {
		routes := budgetTeamKeyRoutes(t)
		routes["/organization/list"] = budgetJSON(budgetFixture(t, "organization-list"))
		routes["/team/info?team_id=T"] = budgetJSON(budgetTeamWithOrg(t))
		mockBudgetProxy(t, routes)
		first := budgetPollOnce(t)
		if first.Levels[budgetOrg] != budgetOrgLevel {
			t.Fatalf("first = %+v", first)
		}
		previous := first.Levels

		routes["/team/info?team_id=T"] = budgetStatusReply(503)
		mockBudgetProxy(t, routes)
		if second := budgetPollWith(t, budgetAuth(), nil, previous); !reflect.DeepEqual(second.Levels, previous) {
			t.Fatalf("second = %+v, want %+v", second.Levels, previous)
		}

		// The key names its own org, and that org is now denied: nothing stale is kept.
		routes["/key/info"] = budgetJSON(budgetKeyWith(t, "key-info-team-user", map[string]any{"organization_id": "O"}))
		routes["/organization/list"] = budgetStatusReply(403)
		mockBudgetProxy(t, routes)
		if _, ok := budgetLevelOf(budgetPollWith(t, budgetAuth(), nil, previous), budgetOrg); ok {
			t.Fatal("a denied org kept its old figure")
		}
	})

	t.Run("treats missing, null, zero and non-numeric limits as unset", func(t *testing.T) {
		for _, limit := range []any{float64(0), "10", nil} {
			body := budgetKeyWith(t, "key-info-personal", map[string]any{"max_budget": limit, "user_id": nil})
			mockBudgetProxy(t, map[string]func() budgetReply{"/key/info": budgetJSON(body)})
			if _, ok := budgetLevelOf(budgetPollOnce(t), budgetKey); ok {
				t.Fatalf("limit %v produced a key level", limit)
			}
		}
		body := budgetKeyWith(t, "key-info-personal", map[string]any{"spend": budgetAbsent{}, "user_id": nil})
		mockBudgetProxy(t, map[string]func() budgetReply{"/key/info": budgetJSON(body)})
		if key, _ := budgetLevelOf(budgetPollOnce(t), budgetKey); key.Spend != 0 || key.MaxBudget != 10 {
			t.Fatalf("key = %+v", key)
		}
	})

	t.Run("encodes the team id", func(t *testing.T) {
		seen := mockBudgetProxy(t, map[string]func() budgetReply{
			"/key/info": budgetJSON(budgetKeyWith(t, "key-info-team-only", map[string]any{"team_id": "a b/c"}))})
		budgetPollOnce(t)
		if seen.count("/team/info?team_id=a%20b%2Fc") != 1 {
			t.Fatalf("requests = %v", seen.paths())
		}
	})

	t.Run("sends nothing to a non-loopback http root unless insecure http is allowed", func(t *testing.T) {
		seen := mockBudgetProxy(t, map[string]func() budgetReply{"/key/info": budgetJSON(budgetObject(t, "key-info-team-only"))})
		insecure := budgetAuth()
		insecure.BaseURL = "http://proxy.example.com"
		budgetWantPoll(t, budgetPollWith(t, insecure, nil, nil), budgetPoll{Reason: "request failed"})
		budgetWantSeen(t, seen.paths())
		insecure.AllowInsecureHTTP = true
		if !budgetPollWith(t, insecure, nil, nil).OK || seen.count("/key/info") != 1 {
			t.Fatalf("requests = %v", seen.paths())
		}
	})

	t.Run("drops a /v1 suffix from the proxy root", func(t *testing.T) {
		seen := mockBudgetProxy(t, map[string]func() budgetReply{"/key/info": budgetJSON(budgetObject(t, "key-info-personal"))})
		auth := budgetAuth()
		auth.BaseURL = "https://proxy.example.com/v1/"
		budgetPollWith(t, auth, nil, nil)
		if seen.count("/key/info") != 1 || seen.count("/v1/key/info") != 0 {
			t.Fatalf("requests = %v", seen.paths())
		}
	})
}

// ---- formatting ----

type budgetPlainTheme struct{}

func (budgetPlainTheme) Fg(_, text string) string { return text }

type budgetMarkTheme struct{}

func (budgetMarkTheme) Fg(color, text string) string {
	return "<" + color + ">" + text + "</" + color + ">"
}

func TestMergeKeyHeaders(t *testing.T) {
	key := &budgetLevel{Spend: 5, MaxBudget: 10, ResetAt: 1}
	h := func(pairs ...string) map[string]string {
		headers := map[string]string{}
		for i := 0; i < len(pairs); i += 2 {
			headers[pairs[i]] = pairs[i+1]
		}
		return headers
	}
	t.Run("raises a polled key spend but never lowers it", func(t *testing.T) {
		got := mergeKeyHeaders(key, true, h("x-litellm-key-spend", "6", "x-litellm-key-max-budget", "10.0"))
		if want := (budgetLevel{Spend: 6, MaxBudget: 10, ResetAt: 1}); *got != want {
			t.Fatalf("got %+v", got)
		}
		if got := mergeKeyHeaders(key, true, h("x-litellm-key-spend", "0.0")); *got != *key {
			t.Fatalf("lowered: %+v", got)
		}
	})
	t.Run("takes a numeric header limit and ignores a missing, empty, or non-numeric one", func(t *testing.T) {
		if got := mergeKeyHeaders(key, true, h("x-litellm-key-spend", "5", "x-litellm-key-max-budget", "12")); got.MaxBudget != 12 {
			t.Fatalf("limit = %v", got.MaxBudget)
		}
		for _, headers := range []map[string]string{h("x-litellm-key-spend", "5"),
			h("x-litellm-key-spend", "5", "x-litellm-key-max-budget", ""), h("x-litellm-key-spend", "5", "x-litellm-key-max-budget", "None")} {
			if got := mergeKeyHeaders(key, true, headers); got.MaxBudget != 10 {
				t.Fatalf("%v: limit = %v", headers, got.MaxBudget)
			}
		}
	})
	t.Run("lets headers alone set the key until a poll succeeds", func(t *testing.T) {
		got := mergeKeyHeaders(nil, false, h("x-litellm-key-spend", "0.144", "x-litellm-key-max-budget", "10.0"))
		if want := (budgetLevel{Spend: 0.144, MaxBudget: 10}); got == nil || *got != want {
			t.Fatalf("got %+v", got)
		}
		if got := mergeKeyHeaders(nil, false, h("x-litellm-key-spend", "0.144")); got != nil {
			t.Fatalf("got %+v", got)
		}
	})
	t.Run("creates a key level when a polled key gains a limit", func(t *testing.T) {
		got := mergeKeyHeaders(nil, true, h("x-litellm-key-spend", "1", "x-litellm-key-max-budget", "10"))
		if want := (budgetLevel{Spend: 1, MaxBudget: 10}); got == nil || *got != want {
			t.Fatalf("got %+v", got)
		}
	})
	t.Run("ignores responses without a key-spend header", func(t *testing.T) {
		if got := mergeKeyHeaders(key, true, h("x-litellm-key-max-budget", "99")); got != key {
			t.Fatalf("got %+v", got)
		}
	})
}

func TestBudgetFormatting(t *testing.T) {
	t.Run("formats compact amounts", func(t *testing.T) {
		for value, want := range map[float64]string{3.1: "$3.10", 40: "$40", 0.042: "$0.04", 412.3: "$412", 1000: "$1k",
			9210: "$9.2k", 50000: "$50k", 1_200_000: "$1.2M", 1250: "$1.3k", 999.5: "$1000"} {
			if got := formatCompactAmount(value); got != want {
				t.Errorf("formatCompactAmount(%v) = %q, want %q", value, got, want)
			}
		}
	})
	t.Run("rounds cents in the footer like the breakdown", func(t *testing.T) {
		if got := formatCompactAmount(1.815); got != "$1.82" {
			t.Fatalf("got %q", got)
		}
		details := formatBudgetDetails("litellm", budgetLevels{budgetKey: {Spend: 1.815, MaxBudget: 10}}, nil, budgetNow)
		if !strings.Contains(details, "$1.82") {
			t.Fatalf("details = %q", details)
		}
	})
	t.Run("formats relative reset times", func(t *testing.T) {
		for _, c := range []struct {
			resetAt int64
			want    string
		}{
			{budgetNow + budgetDay*25/2, "12d"}, {budgetNow + 11*3_600_000/2, "5h"}, {budgetNow + 81*60_000/2, "40m"},
			{budgetNow + 30_000, "1m"}, {budgetNow - 1, ""}, {0, ""},
		} {
			if got := formatRelativeReset(c.resetAt, budgetNow); got != c.want {
				t.Errorf("formatRelativeReset(+%d) = %q, want %q", c.resetAt-budgetNow, got, c.want)
			}
		}
	})
	t.Run("shows every set level in order", func(t *testing.T) {
		levels := budgetLevels{budgetOrg: {Spend: 9210, MaxBudget: 50000}, budgetKey: {Spend: 3.1, MaxBudget: 10},
			budgetMember: {Spend: 20, MaxBudget: 50}, budgetUser: {Spend: 40, MaxBudget: 100}, budgetTeam: {Spend: 412.3, MaxBudget: 1000}}
		want := "LiteLLM key $3.10/$10 · user $40/$100 · team $412/$1k · member $20/$50 · org $9.2k/$50k"
		if got := formatBudgetStatus(levels, displayAll, "LiteLLM", budgetPlainTheme{}, budgetNow); got != want {
			t.Fatalf("got %q", got)
		}
	})
	t.Run("shows only the level with the least money left in tightest mode", func(t *testing.T) {
		levels := budgetLevels{budgetUser: {Spend: 40, MaxBudget: 1000}, budgetTeam: {Spend: 412.3, MaxBudget: 1000, ResetAt: budgetNow + budgetDay*25/2}}
		want := "LiteLLM team $412/$1k (41%) · resets 12d"
		if got := formatBudgetStatus(levels, displayTightest, "LiteLLM", budgetPlainTheme{}, budgetNow); got != want {
			t.Fatalf("got %q", got)
		}
	})
	t.Run("colours segments by use and picks an exceeded level as tightest", func(t *testing.T) {
		one := func(spend float64) string {
			return formatBudgetStatus(budgetLevels{budgetKey: {Spend: spend, MaxBudget: 10}}, displayAll, "P", budgetMarkTheme{}, budgetNow)
		}
		if got, want := one(7.9), "<dim>P</dim> <dim>key $7.90/$10</dim>"; got != want {
			t.Fatalf("got %q", got)
		}
		if got := one(8); !strings.Contains(got, "<warning>key $8/$10</warning>") {
			t.Fatalf("got %q", got)
		}
		if got := one(10.2); !strings.Contains(got, "<error>key $10.20/$10</error>") {
			t.Fatalf("got %q", got)
		}
		levels := budgetLevels{budgetKey: {Spend: 10.2, MaxBudget: 10}, budgetTeam: {Spend: 1, MaxBudget: 1000}}
		if got, want := formatBudgetStatus(levels, displayTightest, "P", budgetMarkTheme{}, budgetNow), "<dim>P</dim> <error>key $10.20/$10 (102%)</error>"; got != want {
			t.Fatalf("got %q", got)
		}
		levels[budgetUser] = budgetLevel{Spend: 0, MaxBudget: 5}
		if got := formatBudgetStatus(levels, displayAll, "P", budgetMarkTheme{}, budgetNow); !strings.Contains(got, "</error><dim> · </dim><dim>user") {
			t.Fatalf("got %q", got)
		}
	})
	t.Run("breaks tightest ties in level order", func(t *testing.T) {
		levels := budgetLevels{budgetKey: {Spend: 5, MaxBudget: 10}, budgetTeam: {Spend: 995, MaxBudget: 1000}}
		if got := formatBudgetStatus(levels, displayTightest, "P", budgetPlainTheme{}, budgetNow); !strings.HasPrefix(got, "P key ") {
			t.Fatalf("got %q", got)
		}
	})
	t.Run("returns nothing when no level is set", func(t *testing.T) {
		if got := formatBudgetStatus(budgetLevels{}, displayAll, "P", budgetPlainTheme{}, budgetNow); got != "" {
			t.Fatalf("got %q", got)
		}
	})
}

func TestFormatBudgetDetails(t *testing.T) {
	t.Run("formats the command breakdown", func(t *testing.T) {
		levels := budgetLevels{
			budgetKey:    {Spend: 3.1, MaxBudget: 10, ResetAt: budgetNow + 25*3_600_000/2},
			budgetUser:   {Spend: 40, MaxBudget: 100, ResetAt: budgetNow + budgetDay*49/2},
			budgetTeam:   {Spend: 412.3, MaxBudget: 1000, ResetAt: budgetNow + budgetDay*49/2},
			budgetMember: {Spend: 20, MaxBudget: 50},
		}
		want := strings.Join([]string{
			`LiteLLM ("litellm") budget`,
			"  key     $3.10 of $10 (31%), resets in 12h",
			"  user    $40.00 of $100 (40%), resets in 24d",
			"  team    $412.30 of $1,000 (41%), resets in 24d",
			"  member  $20.00 of $50 (40%)",
			"  Not readable with this credential: org (401)",
		}, "\n")
		if got := formatBudgetDetails("litellm", levels, map[budgetEndpoint]int{endpointOrg: 401}, budgetNow); got != want {
			t.Fatalf("got\n%s", got)
		}
	})
	t.Run("lists unreadable user endpoints once, and not a successful fallback", func(t *testing.T) {
		text := func(denied map[budgetEndpoint]int) string {
			return formatBudgetDetails("p", budgetLevels{}, denied, budgetNow)
		}
		if got := text(map[budgetEndpoint]int{endpointUserV2: 403}); !strings.Contains(got, "user (403)") {
			t.Fatalf("got %q", got)
		}
		if got := text(map[budgetEndpoint]int{endpointUserV2: 404, endpointUser: 404}); !strings.Contains(got, "user (404)") {
			t.Fatalf("got %q", got)
		}
		if got := text(map[budgetEndpoint]int{endpointUserV2: 404}); strings.Contains(got, "Not readable") {
			t.Fatalf("got %q", got)
		}
	})
	t.Run("says when no budgets are set", func(t *testing.T) {
		if got, want := formatBudgetDetails("litellm", budgetLevels{}, nil, budgetNow), "LiteLLM (\"litellm\") budget\n  no budgets set"; got != want {
			t.Fatalf("got %q", got)
		}
	})
	t.Run("does not say no budgets are set when some levels could not be read", func(t *testing.T) {
		denied := map[budgetEndpoint]int{endpointKey: 500, endpointUserV2: 500}
		want := "LiteLLM (\"litellm\") budget\n  Not readable with this credential: key (500), user (500)"
		if got := formatBudgetDetails("litellm", budgetLevels{}, denied, budgetNow); got != want {
			t.Fatalf("got %q", got)
		}
	})
}

func TestBudgetDisplaySetting(t *testing.T) {
	for _, c := range []struct {
		name    string
		input   any
		display budgetDisplay
		warning string
	}{
		{"unset", nil, displayAll, ""},
		{"empty", map[string]any{}, displayAll, ""},
		{"tightest", map[string]any{"display": "tightest"}, displayTightest, ""},
		{"unknown", map[string]any{"display": "compact"}, displayAll, `LiteLLM budget: unknown display "compact"; using "all".`},
		{"number", map[string]any{"display": float64(5)}, displayAll, `LiteLLM budget: unknown display 5; using "all".`},
		{"null", map[string]any{"display": nil}, displayAll, `LiteLLM budget: unknown display null; using "all".`},
	} {
		t.Run(c.name, func(t *testing.T) {
			display, warning := budgetDisplaySetting(c.input)
			if display != c.display || warning != c.warning {
				t.Fatalf("got %q, %q", display, warning)
			}
		})
	}
}

// ---- controller: fake host, clock and timers ----

type budgetNotice struct{ message, level string }

type fakeBudgetHost struct {
	mu       sync.Mutex
	ui       bool
	provider string
	auths    map[string]*types.LiteLLMRuntimeAuth
	authErr  error
	statuses []string
	notices  []budgetNotice
	// onCall runs at the start of every SetStatus and Notify, before the host's own lock is taken.
	onCall func()
}

func (h *fakeBudgetHost) HasUI() bool { return h.ui }
func (h *fakeBudgetHost) ModelProvider() string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.provider
}
func (h *fakeBudgetHost) setProvider(name string) { h.mu.Lock(); h.provider = name; h.mu.Unlock() }
func (h *fakeBudgetHost) setAuth(name string, auth *types.LiteLLMRuntimeAuth) {
	h.mu.Lock()
	h.auths[name] = auth
	h.mu.Unlock()
}
func (h *fakeBudgetHost) SetStatus(key, text string) {
	if h.onCall != nil {
		h.onCall()
	}
	if key != budgetStatusKey {
		panic("unexpected status key " + key)
	}
	h.mu.Lock()
	h.statuses = append(h.statuses, text)
	h.mu.Unlock()
}
func (h *fakeBudgetHost) Notify(message, level string) {
	if h.onCall != nil {
		h.onCall()
	}
	h.mu.Lock()
	h.notices = append(h.notices, budgetNotice{message, level})
	h.mu.Unlock()
}
func (h *fakeBudgetHost) Fg(_, text string) string { return text }
func (h *fakeBudgetHost) ResolveAuth(name string) (*types.LiteLLMRuntimeAuth, error) {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.auths[name], h.authErr
}

// texts is the non-empty statuses; an empty one clears the footer.
func (h *fakeBudgetHost) texts() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	var out []string
	for _, status := range h.statuses {
		if status != "" {
			out = append(out, status)
		}
	}
	return out
}

func (h *fakeBudgetHost) lastStatus() (string, bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	if len(h.statuses) == 0 {
		return "", false
	}
	return h.statuses[len(h.statuses)-1], true
}

func (h *fakeBudgetHost) allStatuses() []string {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.statuses)
}

func (h *fakeBudgetHost) notifications() []budgetNotice {
	h.mu.Lock()
	defer h.mu.Unlock()
	return slices.Clone(h.notices)
}

type fakeBudgetTimer struct {
	due            time.Duration
	f              func()
	stopped, fired bool
}

// fakeBudgetClock keeps timers on a monotonic axis of their own, so a test can skew the wall clock against them.
type fakeBudgetClock struct {
	mu     sync.Mutex
	start  time.Time
	mono   time.Duration
	skew   time.Duration
	timers []*fakeBudgetTimer
}

func (k *fakeBudgetClock) now() time.Time {
	k.mu.Lock()
	defer k.mu.Unlock()
	return k.start.Add(k.mono + k.skew)
}

func (k *fakeBudgetClock) after(d time.Duration, f func()) func() bool {
	k.mu.Lock()
	defer k.mu.Unlock()
	timer := &fakeBudgetTimer{due: k.mono + d, f: f}
	k.timers = append(k.timers, timer)
	return func() bool {
		k.mu.Lock()
		defer k.mu.Unlock()
		pending := !timer.stopped && !timer.fired
		timer.stopped = true
		return pending
	}
}

func (k *fakeBudgetClock) pending() int {
	k.mu.Lock()
	defer k.mu.Unlock()
	n := 0
	for _, timer := range k.timers {
		if !timer.stopped && !timer.fired {
			n++
		}
	}
	return n
}

type budgetRig struct {
	t     *testing.T
	c     *budgetController
	host  *fakeBudgetHost
	clock *fakeBudgetClock
	t0    time.Time
}

func newBudgetRig(t *testing.T, mutate ...func(*budgetOptions)) *budgetRig {
	t.Helper()
	clock := &fakeBudgetClock{start: time.Date(2026, 10, 7, 12, 0, 0, 0, time.UTC)}
	auth := budgetAuth()
	host := &fakeBudgetHost{ui: true, provider: "litellm", auths: map[string]*types.LiteLLMRuntimeAuth{"litellm": &auth}}
	options := budgetOptions{
		providers: []budgetProvider{{"litellm", "LiteLLM"}}, display: displayAll,
		timeoutMs: func() int { return 5000 }, disabledReason: func() string { return "" }, hostOffline: func() bool { return false },
		missingCredentials: func(name string) string { return "no credentials for " + name + "." },
		now:                clock.now, after: clock.after,
	}
	for _, change := range mutate {
		change(&options)
	}
	rig := &budgetRig{t: t, host: host, clock: clock, t0: clock.start}
	rig.c = newBudgetController(options)
	t.Cleanup(rig.settle)
	return rig
}

func (r *budgetRig) settle() { r.c.runs.Wait() }

// advance is vi.advanceTimersByTimeAsync: it fires due timers in order, letting each poll finish.
func (r *budgetRig) advance(d time.Duration) {
	r.t.Helper()
	k := r.clock
	k.mu.Lock()
	target := k.mono + d
	for {
		var next *fakeBudgetTimer
		for _, timer := range k.timers {
			if !timer.stopped && !timer.fired && timer.due <= target && (next == nil || timer.due < next.due) {
				next = timer
			}
		}
		if next == nil {
			break
		}
		k.mono, next.fired = next.due, true
		k.mu.Unlock()
		next.f()
		r.settle()
		k.mu.Lock()
	}
	k.mono = target
	k.mu.Unlock()
	r.settle()
}

func (r *budgetRig) elapsed() time.Duration { return r.clock.now().Sub(r.t0) }

func (r *budgetRig) sessionStart() { r.c.sessionStart(r.host); r.settle() }
func (r *budgetRig) turnEnd()      { r.c.turnEnd(r.host) }
func (r *budgetRig) selectProvider(name string) {
	r.host.setProvider(name)
	r.c.providerSelected(r.host, name)
	r.settle()
}
func (r *budgetRig) command(args string) { r.c.command(r.host, args); r.settle() }

func (r *budgetRig) wantLast(want string) {
	r.t.Helper()
	if got, ok := r.host.lastStatus(); !ok || got != want {
		r.t.Fatalf("last status = %q (set %v), want %q; all %q", got, ok, want, r.host.allStatuses())
	}
}

func budgetKeyPolls(r *budgetRecorder) int { return r.count("/key/info") }

func TestBudgetFooter(t *testing.T) {
	t.Run("shows the active provider's budgets after session start", func(t *testing.T) {
		mockBudgetProxy(t, budgetPersonalRoutes(t))
		rig := newBudgetRig(t)
		rig.sessionStart()
		rig.wantLast("LiteLLM key $0.15/$10 · user $0.34/$100")
	})

	t.Run("does nothing for another provider's model", func(t *testing.T) {
		seen := mockBudgetProxy(t, budgetPersonalRoutes(t))
		rig := newBudgetRig(t)
		rig.host.provider = "openai"
		rig.sessionStart()
		rig.turnEnd()
		rig.c.responseHeaders(rig.host, map[string]any{"x-litellm-key-spend": "1", "x-litellm-key-max-budget": "2"})
		rig.advance(120 * time.Second)
		budgetWantSeen(t, seen.paths())
		if got := rig.host.allStatuses(); len(got) != 0 {
			t.Fatalf("statuses = %q", got)
		}
	})

	t.Run("does nothing without UI", func(t *testing.T) {
		seen := mockBudgetProxy(t, budgetPersonalRoutes(t))
		rig := newBudgetRig(t)
		rig.host.ui = false
		rig.sessionStart()
		rig.turnEnd()
		rig.advance(120 * time.Second)
		budgetWantSeen(t, seen.paths())
		if got := rig.host.allStatuses(); len(got) != 0 {
			t.Fatalf("statuses = %q", got)
		}
	})

	for _, c := range []struct {
		name   string
		reason string
		host   bool
	}{{"LITELLM_OFFLINE=1", "LITELLM_OFFLINE=1", false}, {"LITELLM_DISCOVERY_TIMEOUT_MS=0", "LITELLM_DISCOVERY_TIMEOUT_MS=0", false}, {"host offline", "", true}} {
		t.Run("does not poll automatically under "+c.name+", but applies headers", func(t *testing.T) {
			seen := mockBudgetProxy(t, budgetPersonalRoutes(t))
			rig := newBudgetRig(t, func(o *budgetOptions) {
				o.disabledReason = func() string { return c.reason }
				o.hostOffline = func() bool { return c.host }
			})
			rig.sessionStart()
			rig.turnEnd()
			rig.advance(120 * time.Second)
			budgetWantSeen(t, seen.paths())
			rig.c.responseHeaders(rig.host, map[string]any{"x-litellm-key-spend": "0.144", "x-litellm-key-max-budget": "10.0"})
			rig.wantLast("LiteLLM key $0.14/$10")
		})
	}

	t.Run("polls 15 s after a turn and at most once a minute", func(t *testing.T) {
		seen := mockBudgetProxy(t, budgetPersonalRoutes(t))
		rig := newBudgetRig(t)
		rig.sessionStart()
		for _, step := range []struct {
			advance time.Duration
			turn    bool
			polls   int
		}{
			{0, false, 1}, {time.Second, true, 1}, {58 * time.Second, false, 1}, {time.Second, false, 2},
			{140 * time.Second, true, 2}, {14 * time.Second, false, 2}, {time.Second, false, 3},
		} {
			rig.advance(step.advance)
			if step.turn {
				rig.turnEnd()
			}
			if got := budgetKeyPolls(seen); got != step.polls {
				t.Fatalf("at %v after +%v: %d key polls, want %d", rig.elapsed(), step.advance, got, step.polls)
			}
		}
	})

	t.Run("polls again after the last turn of a run that ends within the settle delay of a poll", func(t *testing.T) {
		seen := mockBudgetProxy(t, budgetPersonalRoutes(t))
		rig := newBudgetRig(t)
		rig.sessionStart()
		check := func(want int) {
			t.Helper()
			if got := budgetKeyPolls(seen); got != want {
				t.Fatalf("at %v: %d key polls, want %d", rig.elapsed(), got, want)
			}
		}
		check(1)
		rig.advance(50 * time.Second)
		rig.turnEnd()
		rig.advance(5 * time.Second)
		rig.turnEnd()
		rig.advance(9999 * time.Millisecond)
		check(1)
		rig.advance(time.Millisecond)
		check(2)
		rig.advance(59999 * time.Millisecond)
		check(2)
		rig.advance(time.Millisecond)
		check(3)
		rig.advance(175 * time.Second)
		check(3)
	})

	t.Run("does not poll again when the timer fires a few milliseconds early", func(t *testing.T) {
		seen := mockBudgetProxy(t, budgetPersonalRoutes(t))
		rig := newBudgetRig(t)
		rig.sessionStart()
		rig.advance(100 * time.Second)
		rig.turnEnd()
		// Timers keep their own schedule: lag the wall clock so the timer fires before now() reaches its target.
		rig.clock.mu.Lock()
		rig.clock.skew -= 5 * time.Millisecond
		rig.clock.mu.Unlock()
		rig.advance(15 * time.Second)
		if got := budgetKeyPolls(seen); got != 2 {
			t.Fatalf("key polls = %d, want 2", got)
		}
		rig.advance(120 * time.Second)
		if got := budgetKeyPolls(seen); got != 2 {
			t.Fatalf("key polls = %d, want 2", got)
		}
	})

	twoProviders := func(o *budgetOptions) {
		o.providers = []budgetProvider{{"litellm", "LiteLLM"}, {"team", "Team GW"}}
	}
	twoHosts := func(t *testing.T, rig *budgetRig, personal func(host string) budgetReply) *budgetRecorder {
		team := budgetAuth()
		team.BaseURL = "https://team.example.com"
		rig.host.setAuth("team", &team)
		recorder := &budgetRecorder{}
		installBudgetTransport(t, func(request *http.Request) (*http.Response, error) {
			recorder.record(request)
			reply := budgetCode(404)
			if request.URL.Path == "/key/info" {
				reply = personal(request.URL.Host)
			}
			return reply.response(request)
		})
		return recorder
	}
	hostCount := func(r *budgetRecorder, hostPath string) int {
		r.mu.Lock()
		defer r.mu.Unlock()
		n := 0
		for _, h := range r.hosts {
			if h == hostPath {
				n++
			}
		}
		return n
	}

	t.Run("shows a provider's last result on model_select and polls only when stale", func(t *testing.T) {
		rig := newBudgetRig(t, twoProviders)
		seen := twoHosts(t, rig, func(string) budgetReply { return budgetOK(budgetObject(t, "key-info-personal")) })
		rig.sessionStart()
		if hostCount(seen, "proxy.example.com/key/info") != 1 {
			t.Fatalf("hosts = %v", seen.hosts)
		}
		rig.selectProvider("team")
		if hostCount(seen, "team.example.com/key/info") != 1 {
			t.Fatalf("hosts = %v", seen.hosts)
		}
		before := len(seen.hosts)
		rig.selectProvider("litellm")
		if len(seen.hosts) != before {
			t.Fatalf("a fresh provider was polled again: %v", seen.hosts)
		}
		rig.wantLast("LiteLLM key $0.15/$10")
		rig.selectProvider("openai")
		rig.wantLast("")
	})

	t.Run("drops a pending turn poll when another provider is selected", func(t *testing.T) {
		rig := newBudgetRig(t, twoProviders)
		seen := twoHosts(t, rig, func(string) budgetReply { return budgetOK(budgetObject(t, "key-info-personal")) })
		rig.sessionStart()
		rig.advance(time.Second)
		rig.turnEnd()
		rig.advance(54 * time.Second)
		// The turn's poll was due at +60 s for litellm; selecting team polls team now instead.
		rig.selectProvider("team")
		rig.advance(120 * time.Second)
		var polled []string
		for _, h := range seen.hosts {
			if strings.HasSuffix(h, "/key/info") {
				polled = append(polled, strings.TrimSuffix(h, "/key/info"))
			}
		}
		if want := []string{"proxy.example.com", "team.example.com"}; !slices.Equal(polled, want) {
			t.Fatalf("polled %v, want %v", polled, want)
		}
	})

	t.Run("still polls after a turn when its provider is selected again", func(t *testing.T) {
		rig := newBudgetRig(t, twoProviders)
		var mu sync.Mutex
		var polled []string
		twoHosts(t, rig, func(host string) budgetReply {
			mu.Lock()
			polled = append(polled, fmt.Sprintf("%s %v", host, rig.elapsed()))
			mu.Unlock()
			return budgetOK(budgetObject(t, "key-info-personal"))
		})
		rig.sessionStart()
		rig.advance(time.Second)
		rig.turnEnd()
		rig.advance(9 * time.Second)
		rig.selectProvider("team")
		rig.advance(10 * time.Second)
		// Back within a minute of its last poll: litellm's turn still gets its poll, when it was due.
		rig.selectProvider("litellm")
		rig.advance(100 * time.Second)
		want := []string{"proxy.example.com 0s", "team.example.com 10s", "proxy.example.com 1m0s"}
		if !slices.Equal(polled, want) {
			t.Fatalf("polled %v, want %v", polled, want)
		}
	})

	t.Run("keeps each provider's budgets under its own name", func(t *testing.T) {
		rig := newBudgetRig(t, twoProviders)
		twoHosts(t, rig, func(host string) budgetReply {
			body := budgetObject(t, "key-info-personal")
			if host == "team.example.com" {
				info := body["info"].(map[string]any)
				info["spend"], info["max_budget"] = float64(7), float64(20)
			}
			return budgetOK(body)
		})
		rig.sessionStart()
		rig.wantLast("LiteLLM key $0.15/$10")
		rig.selectProvider("team")
		rig.wantLast("Team GW key $7/$20")
		for _, text := range rig.host.texts() {
			if strings.HasPrefix(text, "LiteLLM") && strings.Contains(text, "$20") || strings.HasPrefix(text, "Team GW") && strings.Contains(text, "$10") {
				t.Fatalf("mixed budgets: %q", text)
			}
		}
	})

	t.Run("updates the key figure from response headers without a request", func(t *testing.T) {
		seen := mockBudgetProxy(t, budgetPersonalRoutes(t))
		rig := newBudgetRig(t)
		rig.sessionStart()
		count := len(seen.paths())
		rig.c.responseHeaders(rig.host, map[string]any{"X-Litellm-Key-Spend": "0.5"})
		rig.wantLast("LiteLLM key $0.50/$10 · user $0.34/$100")
		if len(seen.paths()) != count {
			t.Fatal("headers caused a request")
		}
	})

	t.Run("drops levels and 4xx memory when the credential changes", func(t *testing.T) {
		routes := budgetTeamKeyRoutes(t)
		routes["/team/info?team_id=T"] = budgetStatusReply(403)
		mockBudgetProxy(t, routes, budgetExpect{"sk-a", "g1"})
		rig := newBudgetRig(t)
		a := budgetAuth()
		a.APIKey = "sk-a"
		rig.host.setAuth("litellm", &a)
		rig.sessionStart()
		rig.wantLast("LiteLLM user $0.34/$100")

		// Key B can read the team, which key A's remembered 403 would have hidden.
		routes["/team/info?team_id=T"] = budgetJSON(budgetObject(t, "team-info"))
		seen := mockBudgetProxy(t, routes, budgetExpect{"sk-b", "g1"})
		b := budgetAuth()
		b.APIKey = "sk-b"
		rig.host.setAuth("litellm", &b)
		rig.advance(61 * time.Second)
		rig.selectProvider("litellm")
		if seen.count("/team/info?team_id=T") != 1 {
			t.Fatalf("requests = %v", seen.paths())
		}
		want := []string{"LiteLLM user $0.34/$100", "", "LiteLLM user $0.34/$100 · team $0.29/$1k · member $0.15/$50"}
		if got := rig.host.allStatuses(); !slices.Equal(got, want) {
			t.Fatalf("statuses = %q, want %q", got, want)
		}

		rig.host.setAuth("litellm", nil)
		rig.advance(61 * time.Second)
		rig.selectProvider("litellm")
		rig.wantLast("")
	})

	t.Run("treats changed custom headers as a new credential", func(t *testing.T) {
		routes := budgetTeamKeyRoutes(t)
		routes["/team/info?team_id=T"] = budgetStatusReply(403)
		mockBudgetProxy(t, routes)
		rig := newBudgetRig(t)
		rig.sessionStart()
		rig.wantLast("LiteLLM user $0.34/$100")

		// Same key, new gateway token in the custom headers: the old token's 403 must not hide the team.
		routes["/team/info?team_id=T"] = budgetJSON(budgetObject(t, "team-info"))
		seen := mockBudgetProxy(t, routes, budgetExpect{"sk-test", "g2"})
		changed := budgetAuth()
		changed.Headers = map[string]string{"x-gateway": "g2"}
		rig.host.setAuth("litellm", &changed)
		rig.advance(61 * time.Second)
		rig.selectProvider("litellm")
		if seen.count("/team/info?team_id=T") != 1 {
			t.Fatalf("requests = %v", seen.paths())
		}
	})

	t.Run("never shows proxy-supplied text", func(t *testing.T) {
		key := budgetObject(t, "key-info-team-user")
		key["info"].(map[string]any)["organization_id"] = "O"
		mockBudgetProxy(t, map[string]func() budgetReply{
			"/key/info":            budgetJSON(key),
			"/v2/user/info":        budgetJSON(budgetObject(t, "user-info-v2")),
			"/team/info?team_id=T": budgetJSON(budgetObject(t, "team-info")),
			"/organization/list":   budgetStatusReply(403),
		})
		rig := newBudgetRig(t)
		rig.sessionStart()
		if len(rig.host.texts()) == 0 {
			t.Fatal("no footer")
		}
		rig.command("")
		output := strings.Join(rig.host.texts(), "\n")
		for _, notice := range rig.host.notifications() {
			output += "\n" + notice.message
		}
		if !strings.Contains(output, "org (403)") {
			t.Fatalf("output = %q", output)
		}
		for _, bad := range []string{"\x1b", "proxy text", "T\x1b", "secret detail", "@example.com"} {
			if strings.Contains(output, bad) {
				t.Fatalf("output shows %q: %q", bad, output)
			}
		}
	})

	t.Run("clears the timer and status on session_shutdown and ignores late results", func(t *testing.T) {
		previous := shutdownDrainBudget
		shutdownDrainBudget = 10 * time.Millisecond
		t.Cleanup(func() { shutdownDrainBudget = previous })
		release := make(chan struct{})
		recorder := &budgetRecorder{}
		installBudgetTransport(t, func(request *http.Request) (*http.Response, error) {
			recorder.record(request)
			if request.URL.Path != "/key/info" {
				return budgetCode(404).response(request)
			}
			// Held without looking at the request context, so the late result reaches the generation guard.
			<-release
			return budgetOK(budgetObject(t, "key-info-personal")).response(request)
		})
		rig := newBudgetRig(t)
		rig.c.sessionStart(rig.host)
		rig.turnEnd()
		if rig.clock.pending() != 1 {
			t.Fatalf("pending timers = %d", rig.clock.pending())
		}
		rig.c.shutdown(rig.host)
		if rig.clock.pending() != 0 {
			t.Fatal("shutdown left a timer")
		}
		close(release)
		rig.settle()
		rig.advance(120 * time.Second)
		if got := rig.host.allStatuses(); len(got) != 0 {
			t.Fatalf("statuses = %q", got)
		}
		if got := budgetKeyPolls(recorder); got != 1 {
			t.Fatalf("key polls = %d, want 1", got)
		}
	})

	t.Run("cancels an in-flight poll on session_shutdown", func(t *testing.T) {
		installBudgetTransport(t, func(request *http.Request) (*http.Response, error) {
			<-request.Context().Done()
			return nil, request.Context().Err()
		})
		rig := newBudgetRig(t)
		rig.c.sessionStart(rig.host)
		finished := make(chan struct{})
		go func() { rig.c.shutdown(rig.host); close(finished) }()
		select {
		case <-finished:
		case <-time.After(5 * time.Second):
			t.Fatal("shutdown did not cancel the poll")
		}
	})

	t.Run("clears a shown footer on session_shutdown", func(t *testing.T) {
		mockBudgetProxy(t, budgetPersonalRoutes(t))
		rig := newBudgetRig(t)
		rig.sessionStart()
		if len(rig.host.texts()) != 1 {
			t.Fatalf("statuses = %q", rig.host.allStatuses())
		}
		rig.c.shutdown(rig.host)
		rig.wantLast("")
	})

	t.Run("hides the footer when no level has a limit", func(t *testing.T) {
		team := budgetObject(t, "team-info")
		info := team["team_info"].(map[string]any)
		info["max_budget"], info["team_member_budget_table"] = nil, nil
		team["team_memberships"] = []any{}
		user := budgetObject(t, "user-info-v2")
		user["max_budget"] = nil
		seen := mockBudgetProxy(t, map[string]func() budgetReply{
			"/key/info": budgetJSON(budgetObject(t, "key-info-team-user")), "/v2/user/info": budgetJSON(user),
			"/team/info?team_id=T": budgetJSON(team),
		})
		rig := newBudgetRig(t)
		rig.sessionStart()
		if budgetKeyPolls(seen) != 1 || len(rig.host.texts()) != 0 {
			t.Fatalf("key polls %d, statuses %q", budgetKeyPolls(seen), rig.host.allStatuses())
		}
	})

	t.Run("warns once about an unknown display setting", func(t *testing.T) {
		warning := `LiteLLM budget: unknown display "x"; using "all".`
		withWarning := func(o *budgetOptions) { o.displayWarning = warning }
		rig := newBudgetRig(t, withWarning)
		rig.host.provider = "openai"
		rig.sessionStart()
		rig.sessionStart()
		if got := rig.host.notifications(); !reflect.DeepEqual(got, []budgetNotice{{warning, "warning"}}) {
			t.Fatalf("notices = %v", got)
		}

		var diagnostics []string
		previous := reportDiagnostic
		reportDiagnostic = func(message string) { diagnostics = append(diagnostics, message) }
		t.Cleanup(func() { reportDiagnostic = previous })
		quiet := newBudgetRig(t, withWarning)
		quiet.host.ui = false
		quiet.sessionStart()
		if !slices.Equal(diagnostics, []string{warning}) || len(quiet.host.notifications()) != 0 {
			t.Fatalf("diagnostics = %q", diagnostics)
		}
	})
}

func TestBudgetCommand(t *testing.T) {
	t.Run("polls now, forgets 4xx endpoints, and shows the breakdown", func(t *testing.T) {
		var teamOK bool
		var mu sync.Mutex
		routes := budgetTeamKeyRoutes(t)
		routes["/team/info?team_id=T"] = func() budgetReply {
			mu.Lock()
			defer mu.Unlock()
			if teamOK {
				return budgetOK(budgetObject(t, "team-info"))
			}
			return budgetCode(403)
		}
		seen := mockBudgetProxy(t, routes)
		rig := newBudgetRig(t)
		rig.command("")
		if got := rig.host.notifications()[0].message; !strings.Contains(got, "Not readable with this credential: team (403)") {
			t.Fatalf("message = %q", got)
		}
		mu.Lock()
		teamOK = true
		mu.Unlock()
		count := seen.count("/team/info?team_id=T")
		rig.command("litellm")
		if seen.count("/team/info?team_id=T") != count+1 {
			t.Fatalf("requests = %v", seen.paths())
		}
		notice := rig.host.notifications()[1]
		want := formatBudgetDetails("litellm", budgetLevels{budgetUser: budgetUserLevel, budgetTeam: budgetTeamLevel, budgetMember: budgetMemberLevel},
			map[budgetEndpoint]int{}, rig.clock.now().UnixMilli())
		if notice.level != "info" || notice.message != want {
			t.Fatalf("notice = %+v, want %q", notice, want)
		}
	})

	t.Run("forgets 4xx endpoints and polls again instead of joining an in-flight poll", func(t *testing.T) {
		routes := budgetTeamKeyRoutes(t)
		routes["/team/info?team_id=T"] = budgetStatusReply(403)
		seen := mockBudgetProxy(t, routes)
		rig := newBudgetRig(t)
		rig.command("")
		inner := http.DefaultTransport
		reached, release := make(chan struct{}), make(chan struct{})
		var once sync.Once
		installBudgetTransport(t, func(request *http.Request) (*http.Response, error) {
			if request.URL.Path == "/key/info" {
				once.Do(func() { close(reached); <-release })
			}
			return inner.RoundTrip(request)
		})
		seen.reset()
		rig.c.sessionStart(rig.host)
		<-reached
		finished := make(chan struct{})
		go func() { rig.c.command(rig.host, ""); close(finished) }()
		close(release)
		<-finished
		rig.settle()
		var relevant []string
		for _, path := range seen.paths() {
			if strings.HasPrefix(path, "/key/info") || strings.HasPrefix(path, "/team/info") {
				relevant = append(relevant, path)
			}
		}
		budgetWantSeen(t, relevant, "/key/info", "/key/info", "/team/info?team_id=T")
	})

	t.Run("reports unknown providers, missing credentials, and failed polls", func(t *testing.T) {
		rig := newBudgetRig(t)
		rig.command(" nope ")
		rig.host.setAuth("litellm", nil)
		rig.command("")
		auth := budgetAuth()
		rig.host.setAuth("litellm", &auth)
		mockBudgetProxy(t, map[string]func() budgetReply{"/key/info": budgetStatusReply(503)})
		rig.command("")
		want := []budgetNotice{
			{`LiteLLM: unknown provider "nope"; configured: "litellm".`, "warning"},
			{`LiteLLM ("litellm"): budget check skipped; no credentials for litellm.`, "warning"},
			{`LiteLLM ("litellm"): budget check failed (HTTP 503).`, "warning"},
		}
		if got := rig.host.notifications(); !reflect.DeepEqual(got, want) {
			t.Fatalf("notices = %v", got)
		}
	})

	t.Run("keeps a header-only key level through polls that cannot read /key/info", func(t *testing.T) {
		mockBudgetProxy(t, map[string]func() budgetReply{
			"/key/info": budgetStatusReply(403), "/v2/user/info": budgetJSON(budgetObject(t, "user-info-v2"))})
		rig := newBudgetRig(t)
		rig.sessionStart()
		rig.wantLast("LiteLLM user $0.34/$100")
		rig.c.responseHeaders(rig.host, map[string]any{"x-litellm-key-spend": "0.5", "x-litellm-key-max-budget": "10"})
		both := "LiteLLM key $0.50/$10 · user $0.34/$100"
		rig.wantLast(both)
		rig.command("")
		rig.wantLast(both)
		message := rig.host.notifications()[0].message
		for _, want := range []string{"  key     $0.50 of $10 (5%)", "  Not readable with this credential: key (403)"} {
			if !strings.Contains(message, want) {
				t.Fatalf("message = %q, missing %q", message, want)
			}
		}
	})

	t.Run("treats a throwing credential lookup as missing credentials", func(t *testing.T) {
		rig := newBudgetRig(t)
		rig.host.authErr = errors.New("placeholder root")
		rig.command("")
		want := []budgetNotice{{`LiteLLM ("litellm"): budget check skipped; no credentials for litellm.`, "warning"}}
		if got := rig.host.notifications(); !reflect.DeepEqual(got, want) {
			t.Fatalf("notices = %v", got)
		}
	})

	t.Run("skips the command under LITELLM_OFFLINE or a zero timeout but not under PI_OFFLINE", func(t *testing.T) {
		seen := mockBudgetProxy(t, budgetPersonalRoutes(t))
		rig := newBudgetRig(t, func(o *budgetOptions) { o.disabledReason = func() string { return "LITELLM_OFFLINE=1" } })
		rig.command("")
		want := []budgetNotice{{"LiteLLM: budget check skipped (LITELLM_OFFLINE=1).", "warning"}}
		if got := rig.host.notifications(); !reflect.DeepEqual(got, want) {
			t.Fatalf("notices = %v", got)
		}
		budgetWantSeen(t, seen.paths())

		offline := newBudgetRig(t, func(o *budgetOptions) { o.hostOffline = func() bool { return true } })
		offline.command("")
		if seen.count("/key/info") != 1 {
			t.Fatalf("requests = %v", seen.paths())
		}
	})

	t.Run("writes the breakdown to the diagnostic stream without UI", func(t *testing.T) {
		mockBudgetProxy(t, budgetPersonalRoutes(t))
		var diagnostics []string
		previous := reportDiagnostic
		reportDiagnostic = func(message string) { diagnostics = append(diagnostics, message) }
		t.Cleanup(func() { reportDiagnostic = previous })
		rig := newBudgetRig(t)
		rig.host.ui = false
		rig.command("")
		if len(diagnostics) != 1 || !strings.HasPrefix(diagnostics[0], "LiteLLM (\"litellm\") budget\n") || len(rig.host.notifications()) != 0 {
			t.Fatalf("diagnostics = %q", diagnostics)
		}
	})

	t.Run("completes provider names", func(t *testing.T) {
		rig := newBudgetRig(t)
		if got := rig.c.completions("li"); len(got) != 1 || got[0].Value != "litellm" || got[0].Label != "litellm" {
			t.Fatalf("completions = %v", got)
		}
		if got := rig.c.completions("x"); got != nil {
			t.Fatalf("completions = %v", got)
		}
	})
}

func TestBudgetHostCallsRunOutsideTheControllerLock(t *testing.T) {
	// Every SetStatus and Notify must find the controller mutex free: a stuck host call cannot be allowed to
	// block the hooks that need it.
	withHost := func(t *testing.T, mutate ...func(*budgetOptions)) (*budgetRig, *[]string) {
		rig := newBudgetRig(t, mutate...)
		var held []string
		rig.host.onCall = func() {
			if rig.c.mu.TryLock() {
				rig.c.mu.Unlock()
				return
			}
			held = append(held, "host call under the controller lock")
		}
		return rig, &held
	}
	t.Run("status writes after a poll, a provider switch and response headers", func(t *testing.T) {
		mockBudgetProxy(t, budgetPersonalRoutes(t))
		rig, held := withHost(t)
		rig.sessionStart()
		rig.c.responseHeaders(rig.host, map[string]any{"x-litellm-key-spend": "9", "x-litellm-key-max-budget": "10"})
		rig.selectProvider("openai")
		if len(rig.host.allStatuses()) < 3 || len(*held) != 0 {
			t.Fatalf("statuses %q, violations %q", rig.host.allStatuses(), *held)
		}
	})
	t.Run("the display warning", func(t *testing.T) {
		rig, held := withHost(t, func(o *budgetOptions) { o.displayWarning = "careful" })
		rig.host.provider = "openai"
		rig.sessionStart()
		if len(rig.host.notifications()) != 1 || len(*held) != 0 {
			t.Fatalf("notices %v, violations %q", rig.host.notifications(), *held)
		}
	})
	t.Run("the footer cleared by session_shutdown", func(t *testing.T) {
		mockBudgetProxy(t, budgetPersonalRoutes(t))
		rig, held := withHost(t)
		rig.sessionStart()
		rig.c.shutdown(rig.host)
		if last, _ := rig.host.lastStatus(); last != "" || len(*held) != 0 {
			t.Fatalf("last status %q, violations %q", last, *held)
		}
	})
}

func TestBudgetShutdownDoesNotWaitForAStuckHost(t *testing.T) {
	mockBudgetProxy(t, budgetPersonalRoutes(t))
	previous := shutdownDrainBudget
	shutdownDrainBudget = 50 * time.Millisecond
	t.Cleanup(func() { shutdownDrainBudget = previous })
	rig := newBudgetRig(t)
	rig.sessionStart()
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	rig.host.onCall = func() { <-release }
	finished := make(chan struct{})
	go func() { rig.c.shutdown(rig.host); close(finished) }()
	select {
	case <-finished:
	case <-time.After(5 * time.Second):
		t.Fatal("shutdown waited for the host's SetStatus")
	}
}
