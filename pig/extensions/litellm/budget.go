package litellm

// Ports src/budget.ts: the footer budget status and /litellm-budget. The poll and format functions are
// pure; budgetController owns the per-provider state, the timers and the single background goroutine per poll.

import (
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"maps"
	"math"
	"math/big"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	sdk "github.com/MichaelKinsy/PiG/extensions/sdk"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/discover"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/protocols"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

type budgetLevelName string

const (
	budgetKey    budgetLevelName = "key"
	budgetUser   budgetLevelName = "user"
	budgetTeam   budgetLevelName = "team"
	budgetMember budgetLevelName = "member"
	budgetOrg    budgetLevelName = "org"
)

var budgetLevelNames = []budgetLevelName{budgetKey, budgetUser, budgetTeam, budgetMember, budgetOrg}

type budgetEndpoint string

const (
	endpointKey    budgetEndpoint = "key"
	endpointUserV2 budgetEndpoint = "userV2"
	endpointUser   budgetEndpoint = "user"
	endpointTeam   budgetEndpoint = "team"
	endpointOrg    budgetEndpoint = "org"
)

// budgetLevel is BudgetLevel. ResetAt is epoch milliseconds; 0 means no reset time.
type budgetLevel struct {
	Spend, MaxBudget float64
	ResetAt          int64
}

type budgetLevels map[budgetLevelName]budgetLevel

// budgetPoll is BudgetPoll: Reason is set exactly when OK is false.
type budgetPoll struct {
	OK        bool
	Levels    budgetLevels
	KeyPolled bool
	Reason    string
}

type budgetFetchStatus int

const (
	fetchOK budgetFetchStatus = iota
	fetchDenied
	fetchTransient
)

type budgetFetched struct {
	status budgetFetchStatus
	data   any
	reason string
}

func asObj(value any) map[string]any {
	object, _ := value.(map[string]any)
	return object
}

func asStr(value any) string {
	text, _ := value.(string)
	return text
}

// parseResetAt is Date.parse for the ISO timestamps LiteLLM emits; a zone-less one reads as UTC.
func parseResetAt(value any) int64 {
	text, ok := value.(string)
	if !ok {
		return 0
	}
	for _, layout := range []string{time.RFC3339Nano, "2006-01-02T15:04:05.999999999", "2006-01-02"} {
		if parsed, err := time.Parse(layout, text); err == nil {
			return parsed.UnixMilli()
		}
	}
	return 0
}

// newBudgetLevel exists only for a positive, finite limit; anything else means "no budget".
func newBudgetLevel(spend, maxBudget, resetAt any) (budgetLevel, bool) {
	limit, ok := maxBudget.(float64)
	if !ok || math.IsInf(limit, 0) || math.IsNaN(limit) || limit <= 0 {
		return budgetLevel{}, false
	}
	level := budgetLevel{MaxBudget: limit, ResetAt: parseResetAt(resetAt)}
	if used, ok := spend.(float64); ok && !math.IsInf(used, 0) && !math.IsNaN(used) && used >= 0 {
		level.Spend = used
	}
	return level, true
}

// keepBudgetLevels carries levels over from the last good poll when their endpoint failed transiently.
func keepBudgetLevels(previous budgetLevels, names ...budgetLevelName) budgetLevels {
	kept := budgetLevels{}
	for _, name := range names {
		if level, ok := previous[name]; ok {
			kept[name] = level
		}
	}
	return kept
}

// budgetPoller is one poll's request context. Its denied map is the caller's; user and team polls run
// concurrently, so access goes through the mutex.
type budgetPoller struct {
	ctx       context.Context
	auth      types.LiteLLMRuntimeAuth
	timeoutMs int

	mu     sync.Mutex
	denied map[budgetEndpoint]int
}

func (p *budgetPoller) deniedCode(endpoint budgetEndpoint) (int, bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	code, ok := p.denied[endpoint]
	return code, ok
}

func (p *budgetPoller) isDenied(endpoint budgetEndpoint) bool {
	_, ok := p.deniedCode(endpoint)
	return ok
}

func (p *budgetPoller) get(path string, endpoint budgetEndpoint) budgetFetched {
	transient := budgetFetched{status: fetchTransient, reason: "request failed"}
	base, err := protocols.NormalizeBaseURL(p.auth.BaseURL, p.auth.AllowInsecureHTTP)
	if err != nil {
		return transient
	}
	timeout := time.Duration(p.timeoutMs) * time.Millisecond
	result, err := discover.FetchJSON[any](p.ctx, base+path, p.auth.APIKey, discover.Options{DiscoveryOptions: types.DiscoveryOptions{
		Timeout: &timeout, Headers: p.auth.Headers, AllowInsecureHTTP: p.auth.AllowInsecureHTTP,
	}})
	if err != nil {
		return transient
	}
	if result.OK {
		return budgetFetched{status: fetchOK, data: result.Data}
	}
	// LiteLLM answers 500 when handling the request raises (no database, a credential it cannot look up, a
	// version bug), which repeats on every call, so it is remembered like a 4xx. 429 and 502-504 stay retryable.
	if (result.Status >= 400 && result.Status < 500 && result.Status != 429) || result.Status == 500 {
		p.mu.Lock()
		p.denied[endpoint] = result.Status
		p.mu.Unlock()
		return budgetFetched{status: fetchDenied}
	}
	return budgetFetched{status: fetchTransient, reason: fmt.Sprintf("HTTP %d", result.Status)}
}

type budgetUserPoll struct {
	levels    budgetLevels
	userID    string
	teamIDs   []string
	transient bool
}

func (p *budgetPoller) pollUser(previous budgetLevels) budgetUserPoll {
	keepUser := func() budgetUserPoll {
		return budgetUserPoll{levels: keepBudgetLevels(previous, budgetUser), transient: true}
	}
	none := budgetUserPoll{levels: budgetLevels{}}
	var fetched *budgetFetched
	nested := false
	if code, denied := p.deniedCode(endpointUserV2); !denied {
		result := p.get("/v2/user/info", endpointUserV2)
		fetched = &result
		if result.status == fetchDenied {
			if code, _ := p.deniedCode(endpointUserV2); code == 404 {
				fetched = nil
			}
		}
	} else if code != 404 {
		return none
	}
	if fetched == nil {
		if p.isDenied(endpointUser) {
			return none
		}
		nested = true
		result := p.get("/user/info", endpointUser)
		fetched = &result
	}
	if fetched.status == fetchDenied {
		return none
	}
	if fetched.status == fetchTransient {
		return keepUser()
	}
	data := asObj(fetched.data)
	body := data
	if nested {
		body = asObj(data["user_info"])
	}
	if body == nil {
		return keepUser()
	}
	poll := budgetUserPoll{levels: budgetLevels{}, userID: asStr(body["user_id"])}
	if user, ok := newBudgetLevel(body["spend"], body["max_budget"], body["budget_reset_at"]); ok {
		poll.levels[budgetUser] = user
	}
	// `/v2/user/info` lists team ids; `/user/info` lists team rows.
	teams, _ := data["teams"].([]any)
	for _, team := range teams {
		id := asStr(team)
		if id == "" {
			id = asStr(asObj(team)["team_id"])
		}
		if id != "" {
			poll.teamIDs = append(poll.teamIDs, id)
		}
	}
	return poll
}

type budgetTeamPoll struct {
	levels         budgetLevels
	organizationID string
	transient      bool
}

func (p *budgetPoller) pollTeam(teamID, userID string, previous budgetLevels) budgetTeamPoll {
	fetched := p.get("/team/info?team_id="+jsEncodeURIComponent(teamID), endpointTeam)
	if fetched.status == fetchDenied {
		return budgetTeamPoll{levels: budgetLevels{}}
	}
	var body map[string]any
	if fetched.status == fetchOK {
		body = asObj(fetched.data)
	}
	info := asObj(body["team_info"])
	if info == nil {
		return budgetTeamPoll{levels: keepBudgetLevels(previous, budgetTeam, budgetMember), transient: true}
	}
	levels := budgetLevels{}
	if team, ok := newBudgetLevel(info["spend"], info["max_budget"], info["budget_reset_at"]); ok {
		levels[budgetTeam] = team
	}
	if userID != "" {
		var own map[string]any
		memberships, _ := body["team_memberships"].([]any)
		for _, membership := range memberships {
			if candidate := asObj(membership); candidate != nil && candidate["user_id"] == userID {
				own = candidate
				break
			}
		}
		table := asObj(own["litellm_budget_table"])
		if table == nil {
			table = asObj(info["team_member_budget_table"])
		}
		if member, ok := newBudgetLevel(own["spend"], table["max_budget"], table["budget_reset_at"]); ok {
			levels[budgetMember] = member
		}
	}
	return budgetTeamPoll{levels: levels, organizationID: asStr(info["organization_id"])}
}

func (p *budgetPoller) pollOrg(organizationID string, previous budgetLevels) budgetLevels {
	fetched := p.get("/organization/list", endpointOrg)
	if fetched.status == fetchDenied {
		return budgetLevels{}
	}
	list, isList := fetched.data.([]any)
	if fetched.status == fetchTransient || !isList {
		return keepBudgetLevels(previous, budgetOrg)
	}
	var entry map[string]any
	for _, candidate := range list {
		if object := asObj(candidate); object != nil && object["organization_id"] == organizationID {
			entry = object
			break
		}
	}
	table := asObj(entry["litellm_budget_table"])
	if org, ok := newBudgetLevel(entry["spend"], table["max_budget"], table["budget_reset_at"]); ok {
		return budgetLevels{budgetOrg: org}
	}
	return budgetLevels{}
}

// jsEncodeURIComponent is encodeURIComponent: it leaves A-Z a-z 0-9 - _ . ! ~ * ' ( ) and escapes the rest.
func jsEncodeURIComponent(value string) string {
	var out strings.Builder
	for _, b := range []byte(value) {
		if b >= 'a' && b <= 'z' || b >= 'A' && b <= 'Z' || b >= '0' && b <= '9' || strings.IndexByte("-_.!~*'()", b) >= 0 {
			out.WriteByte(b)
		} else {
			fmt.Fprintf(&out, "%%%02X", b)
		}
	}
	return out.String()
}

// pollBudget is pollBudget. denied is read and updated in place; previous is only read.
func pollBudget(ctx context.Context, auth types.LiteLLMRuntimeAuth, denied map[budgetEndpoint]int, previous budgetLevels, timeoutMs int) budgetPoll {
	p := &budgetPoller{ctx: ctx, auth: auth, timeoutMs: timeoutMs, denied: denied}
	levels := budgetLevels{}
	var userID, teamID, organizationID string
	keyPolled := !p.isDenied(endpointKey)

	if keyPolled {
		fetched := p.get("/key/info", endpointKey)
		if fetched.status == fetchTransient {
			return budgetPoll{Reason: fetched.reason}
		}
		if fetched.status == fetchOK {
			info := asObj(asObj(fetched.data)["info"])
			if info == nil {
				return budgetPoll{Reason: "malformed response"}
			}
			if key, ok := newBudgetLevel(info["spend"], info["max_budget"], info["budget_reset_at"]); ok {
				levels[budgetKey] = key
			}
			userID, teamID, organizationID = asStr(info["user_id"]), asStr(info["team_id"]), asStr(info["organization_id"])
		}
	}
	keyKnown := keyPolled && !p.isDenied(endpointKey)

	userDone := make(chan budgetUserPoll, 1)
	if userID != "" || !keyKnown {
		go func() { userDone <- p.pollUser(previous) }()
	} else {
		userDone <- budgetUserPoll{levels: budgetLevels{}}
	}
	var user budgetUserPoll
	awaited := false
	awaitUser := func() budgetUserPoll {
		if !awaited {
			user, awaited = <-userDone, true
		}
		return user
	}
	// A JWT (LiteLLM's own test: three dot-separated parts), such as an OIDC login's id_token, has no key row, and LiteLLM
	// charges its user's team: the only one is certain, but with several it depends on the proxy's JWT settings. A
	// virtual key's user may be in a team the key is not charged to, such as the master key's default user.
	if !keyKnown && len(strings.Split(auth.APIKey, ".")) == 3 {
		polled := awaitUser()
		if polled.transient {
			maps.Copy(levels, keepBudgetLevels(previous, budgetTeam, budgetMember, budgetOrg))
		} else if len(polled.teamIDs) == 1 {
			teamID, userID = polled.teamIDs[0], polled.userID
		}
	}
	teamDone := make(chan budgetTeamPoll, 1)
	if teamID != "" && !p.isDenied(endpointTeam) {
		go func() { teamDone <- p.pollTeam(teamID, userID, previous) }()
	} else {
		teamDone <- budgetTeamPoll{levels: budgetLevels{}}
	}
	user = awaitUser()
	team := <-teamDone
	maps.Copy(levels, user.levels)
	maps.Copy(levels, team.levels)

	org := organizationID
	if org == "" {
		org = team.organizationID
	}
	if org != "" && userID != "" && !p.isDenied(endpointOrg) {
		maps.Copy(levels, p.pollOrg(org, previous))
	} else if org == "" && team.transient {
		// The team was the only source of the org id, so a temporary team failure keeps the last org figure too.
		maps.Copy(levels, keepBudgetLevels(previous, budgetOrg))
	}
	return budgetPoll{OK: true, Levels: levels, KeyPolled: keyKnown}
}

type budgetDisplay string

const (
	displayAll      budgetDisplay = "all"
	displayTightest budgetDisplay = "tightest"
)

// budgetTheme is BudgetTheme; sdk.UITheme satisfies it with the Pi color tokens "dim", "warning" and "error".
type budgetTheme interface {
	Fg(color, text string) string
}

// headerNumber counts a header when it is a non-blank string holding a finite number >= 0.
func headerNumber(headers map[string]string, name string) (float64, bool) {
	value, ok := headers[name]
	if !ok || strings.TrimSpace(value) == "" {
		return 0, false
	}
	parsed, err := strconv.ParseFloat(strings.TrimSpace(value), 64)
	if err != nil || math.IsInf(parsed, 0) || math.IsNaN(parsed) || parsed < 0 {
		return 0, false
	}
	return parsed, true
}

// mergeKeyHeaders lets x-litellm-key-* response headers only raise the key level, or set it alone while
// /key/info has not succeeded. Header names are lower case.
func mergeKeyHeaders(key *budgetLevel, keyPolled bool, headers map[string]string) *budgetLevel {
	spend, ok := headerNumber(headers, "x-litellm-key-spend")
	if !ok {
		return key
	}
	limit, hasLimit := headerNumber(headers, "x-litellm-key-max-budget")
	hasLimit = hasLimit && limit > 0
	if !keyPolled || key == nil {
		if !hasLimit {
			return nil
		}
		return &budgetLevel{Spend: spend, MaxBudget: limit}
	}
	merged := *key
	merged.Spend = math.Max(key.Spend, spend)
	if hasLimit {
		merged.MaxBudget = limit
	}
	return &merged
}

// money is toLocaleString("en-US") with maximumFractionDigits 2: it rounds the shortest decimal value half up
// (1.815 to 1.82), where a binary rounding would give 1.81.
func money(value float64, minDecimals int) string {
	whole, frac, _ := strings.Cut(strconv.FormatFloat(math.Abs(value), 'f', -1, 64), ".")
	if len(frac) > 2 {
		roundUp := frac[2] >= '5'
		frac = frac[:2]
		if roundUp {
			cents, _ := new(big.Int).SetString(whole+frac, 10)
			digits := cents.Add(cents, big.NewInt(1)).String()
			digits = strings.Repeat("0", max(0, 3-len(digits))) + digits
			whole, frac = digits[:len(digits)-2], digits[len(digits)-2:]
		}
	}
	frac = strings.TrimRight(frac, "0")
	frac += strings.Repeat("0", max(0, minDecimals-len(frac)))
	var grouped []string
	for len(whole) > 3 {
		grouped = append([]string{whole[len(whole)-3:]}, grouped...)
		whole = whole[:len(whole)-3]
	}
	text := "$" + strings.Join(append([]string{whole}, grouped...), ",")
	if frac != "" {
		text += "." + frac
	}
	return text
}

// toFixedOne is Number.prototype.toFixed(1) for 1 <= value < 10: it rounds the exact binary value, ties up.
func toFixedOne(value float64) string {
	whole, frac, _ := strings.Cut(strconv.FormatFloat(value, 'f', 60, 64), ".")
	tenths, _ := strconv.Atoi(whole + frac[:1])
	if frac[1] >= '5' {
		tenths++
	}
	return fmt.Sprintf("%d.%d", tenths/10, tenths%10)
}

func jsRoundInt(value float64) string {
	return strconv.FormatFloat(math.Floor(value+0.5), 'f', 0, 64)
}

func formatCompactAmount(value float64) string {
	if value < 100 {
		decimals := 2
		if value == math.Trunc(value) {
			decimals = 0
		}
		return money(value, decimals)
	}
	if value < 1000 {
		return "$" + jsRoundInt(value)
	}
	scaled, suffix := value/1000, "k"
	if value >= 1_000_000 {
		scaled, suffix = value/1_000_000, "M"
	}
	if scaled < 10 {
		return "$" + strings.TrimSuffix(toFixedOne(scaled), ".0") + suffix
	}
	return "$" + jsRoundInt(scaled) + suffix
}

// formatRelativeReset is "" for no (or a past) reset time. Times are epoch milliseconds.
func formatRelativeReset(resetAt, nowMs int64) string {
	if resetAt == 0 || resetAt <= nowMs {
		return ""
	}
	ms := resetAt - nowMs
	switch {
	case ms >= 86_400_000:
		return fmt.Sprintf("%dd", ms/86_400_000)
	case ms >= 3_600_000:
		return fmt.Sprintf("%dh", ms/3_600_000)
	}
	return fmt.Sprintf("%dm", max(1, ms/60_000))
}

func (l budgetLevel) percent() int { return int(math.Floor(l.Spend / l.MaxBudget * 100)) }

func (l budgetLevel) colour() string {
	switch ratio := l.Spend / l.MaxBudget; {
	case ratio >= 1:
		return "error"
	case ratio >= 0.8:
		return "warning"
	}
	return "dim"
}

// formatBudgetStatus is the footer text, or "" when no level is set.
func formatBudgetStatus(levels budgetLevels, display budgetDisplay, prefix string, theme budgetTheme, nowMs int64) string {
	type named struct {
		name  budgetLevelName
		level budgetLevel
	}
	var set []named
	for _, name := range budgetLevelNames {
		if level, ok := levels[name]; ok {
			set = append(set, named{name, level})
		}
	}
	if len(set) == 0 {
		return ""
	}
	head := theme.Fg("dim", prefix) + " "
	text := func(n named) string {
		return fmt.Sprintf("%s %s/%s", n.name, formatCompactAmount(n.level.Spend), formatCompactAmount(n.level.MaxBudget))
	}
	if display == displayAll {
		segments := make([]string, len(set))
		for i, n := range set {
			segments[i] = theme.Fg(n.level.colour(), text(n))
		}
		return head + strings.Join(segments, theme.Fg("dim", " · "))
	}
	// Strict "<" keeps the first level in order on ties.
	tight := set[0]
	for _, n := range set[1:] {
		if n.level.MaxBudget-n.level.Spend < tight.level.MaxBudget-tight.level.Spend {
			tight = n
		}
	}
	out := head + theme.Fg(tight.level.colour(), fmt.Sprintf("%s (%d%%)", text(tight), tight.level.percent()))
	if reset := formatRelativeReset(tight.level.ResetAt, nowMs); reset != "" {
		out += theme.Fg("dim", " · resets "+reset)
	}
	return out
}

func formatBudgetDetails(providerName string, levels budgetLevels, denied map[budgetEndpoint]int, nowMs int64) string {
	lines := []string{fmt.Sprintf("LiteLLM (\"%s\") budget", providerName)}
	for _, name := range budgetLevelNames {
		level, ok := levels[name]
		if !ok {
			continue
		}
		reset := ""
		if relative := formatRelativeReset(level.ResetAt, nowMs); relative != "" {
			reset = ", resets in " + relative
		}
		limitDecimals := 2
		if level.MaxBudget == math.Trunc(level.MaxBudget) {
			limitDecimals = 0
		}
		lines = append(lines, fmt.Sprintf("  %-8s%s of %s (%d%%)%s", name, money(level.Spend, 2), money(level.MaxBudget, limitDecimals), level.percent(), reset))
	}
	userCode, userDenied := denied[endpointUserV2]
	if !userDenied || userCode == 404 {
		userCode, userDenied = denied[endpointUser]
	}
	var unreadable []string
	for _, entry := range []struct {
		name   string
		code   int
		denied bool
	}{
		{"key", denied[endpointKey], hasKey(denied, endpointKey)},
		{"user", userCode, userDenied},
		{"team", denied[endpointTeam], hasKey(denied, endpointTeam)},
		{"org", denied[endpointOrg], hasKey(denied, endpointOrg)},
	} {
		if entry.denied {
			unreadable = append(unreadable, fmt.Sprintf("%s (%d)", entry.name, entry.code))
		}
	}
	if len(lines) == 1 && len(unreadable) == 0 {
		lines = append(lines, "  no budgets set")
	}
	if len(unreadable) > 0 {
		lines = append(lines, "  Not readable with this credential: "+strings.Join(unreadable, ", "))
	}
	return strings.Join(lines, "\n")
}

func hasKey(denied map[budgetEndpoint]int, endpoint budgetEndpoint) bool {
	_, ok := denied[endpoint]
	return ok
}

// budgetDisplaySetting reads `budget.display`; warning is "" when the value is valid.
func budgetDisplaySetting(budgetSettings any) (display budgetDisplay, warning string) {
	value, present := asObj(budgetSettings)["display"]
	if !present || value == "all" || value == "tightest" {
		if text, ok := value.(string); ok {
			return budgetDisplay(text), ""
		}
		return displayAll, ""
	}
	encoded, _ := json.Marshal(value)
	return displayAll, fmt.Sprintf("LiteLLM budget: unknown display %s; using \"all\".", encoded)
}

const (
	budgetStatusKey = "litellm-budget"
	pollSettleDelay = 15 * time.Second
	pollMinInterval = 60 * time.Second
)

// shutdownDrainBudget bounds how long session_shutdown waits for cancelled polls; tests shorten it.
var shutdownDrainBudget = time.Second

// budgetHost is the slice of sdk.Context the controller uses, so tests can drive it without a host.
type budgetHost interface {
	HasUI() bool
	ModelProvider() string
	SetStatus(key, text string)
	Notify(message, level string)
	Fg(color, text string) string
	// ResolveAuth is getRuntimeAuth for a configured provider; nil auth means no credentials.
	ResolveAuth(name string) (*types.LiteLLMRuntimeAuth, error)
}

type sdkBudgetHost struct {
	sdk.Context
	state *extensionState
}

func (h sdkBudgetHost) Fg(color, text string) string { return h.Context.UITheme().Fg(color, text) }

func (h sdkBudgetHost) ResolveAuth(name string) (*types.LiteLLMRuntimeAuth, error) {
	for _, definition := range h.state.definitions {
		if definition.Name == name {
			return h.state.runtimeAuth(h.Context, definition)
		}
	}
	return nil, nil
}

type budgetProvider struct{ name, displayName string }

type budgetOptions struct {
	providers          []budgetProvider
	display            budgetDisplay
	displayWarning     string
	timeoutMs          func() int
	disabledReason     func() string
	hostOffline        func() bool
	missingCredentials func(name string) string
	now                func() time.Time
	// after runs f once after d and returns a stop function; it is time.AfterFunc unless a test replaces it.
	after func(d time.Duration, f func()) (stop func() bool)
}

type budgetOutcomeKind int

const (
	outcomeOK budgetOutcomeKind = iota
	outcomeNoAuth
	outcomeFailed
)

type budgetOutcome struct {
	kind   budgetOutcomeKind
	reason string
}

// budgetFlight is a poll in progress; outcome is valid once done is closed.
type budgetFlight struct {
	done    chan struct{}
	outcome budgetOutcome
}

type budgetProviderState struct {
	digest     string
	levels     budgetLevels
	denied     map[budgetEndpoint]int
	keyPolled  bool
	lastPollAt time.Time
	lastTurnAt time.Time
	inFlight   *budgetFlight
}

// budgetController is setupLiteLLMBudget's closure state. Every method with a Locked suffix expects mu held;
// HTTP and credential lookups run outside it.
type budgetController struct {
	options     budgetOptions
	displayName map[string]string
	// Keyed like the MCP registration identity, because custom headers can carry gateway credentials.
	identityKey [32]byte

	mu             sync.Mutex
	states         map[string]*budgetProviderState
	shown          string
	activeProvider string
	latestHost     budgetHost
	stopTimer      func() bool
	timerSeq       int
	generation     int
	warned         bool
	runCtx         context.Context
	cancelRuns     context.CancelFunc
	runs           sync.WaitGroup
}

func newBudgetController(options budgetOptions) *budgetController {
	if options.now == nil {
		options.now = time.Now
	}
	if options.after == nil {
		options.after = func(d time.Duration, f func()) func() bool { return time.AfterFunc(d, f).Stop }
	}
	c := &budgetController{options: options, displayName: map[string]string{}, states: map[string]*budgetProviderState{}}
	for _, provider := range options.providers {
		c.displayName[provider.name] = provider.displayName
	}
	_, _ = rand.Read(c.identityKey[:])
	return c
}

func (c *budgetController) stateFor(name string) *budgetProviderState {
	state, ok := c.states[name]
	if !ok {
		state = &budgetProviderState{levels: budgetLevels{}, denied: map[budgetEndpoint]int{}}
		c.states[name] = state
	}
	return state
}

func (state *budgetProviderState) forget() {
	state.levels = budgetLevels{}
	clear(state.denied)
	state.keyPolled = false
}

func (c *budgetController) gated() bool {
	return c.options.disabledReason() != "" || c.options.hostOffline()
}

// renderLocked draws only the active provider's text, and only when it changed.
func (c *budgetController) renderLocked(host budgetHost, name string) {
	if !host.HasUI() || c.activeProvider != name {
		return
	}
	text := formatBudgetStatus(c.stateFor(name).levels, c.options.display, c.displayName[name], host, c.options.now().UnixMilli())
	if text == c.shown {
		return
	}
	host.SetStatus(budgetStatusKey, text)
	c.shown = text
}

func (c *budgetController) runContextLocked() context.Context {
	if c.runCtx == nil {
		c.runCtx, c.cancelRuns = context.WithCancel(context.Background())
	}
	return c.runCtx
}

func (c *budgetController) digest(auth types.LiteLLMRuntimeAuth) string {
	names := slices.Sorted(maps.Keys(auth.Headers))
	headers := make([][2]string, len(names))
	for i, name := range names {
		headers[i] = [2]string{name, auth.Headers[name]}
	}
	identity, _ := json.Marshal([]any{auth.BaseURL, auth.APIKey, headers})
	mac := hmac.New(sha256.New, c.identityKey[:])
	mac.Write(identity)
	return hex.EncodeToString(mac.Sum(nil))
}

func (c *budgetController) run(ctx context.Context, started int, host budgetHost, name string, state *budgetProviderState) budgetOutcome {
	auth, err := host.ResolveAuth(name)
	if err != nil {
		auth = nil
	}
	c.mu.Lock()
	if auth == nil {
		state.forget()
		state.digest = ""
		if started == c.generation {
			c.renderLocked(host, name)
		}
		c.mu.Unlock()
		return budgetOutcome{kind: outcomeNoAuth}
	}
	digest := c.digest(*auth)
	if state.digest != "" && state.digest != digest {
		state.forget()
		if started == c.generation {
			c.renderLocked(host, name)
		}
	}
	state.digest = digest
	denied, previous := maps.Clone(state.denied), maps.Clone(state.levels)
	c.mu.Unlock()

	result := pollBudget(ctx, *auth, denied, previous, c.options.timeoutMs())

	c.mu.Lock()
	defer c.mu.Unlock()
	state.denied = denied
	if result.OK {
		levels := maps.Clone(result.Levels)
		if key, ok := state.levels[budgetKey]; ok && !result.KeyPolled {
			levels[budgetKey] = key
		}
		state.levels, state.keyPolled = levels, result.KeyPolled
	}
	if started == c.generation {
		c.renderLocked(host, name)
	}
	if result.OK {
		return budgetOutcome{kind: outcomeOK}
	}
	return budgetOutcome{kind: outcomeFailed, reason: result.Reason}
}

// pollLocked joins a poll in flight or starts one off the handler. Its goroutine is owned by c.runs, ends
// with the run context that session_shutdown cancels, and records nothing but the outcome.
func (c *budgetController) pollLocked(host budgetHost, name string) *budgetFlight {
	state := c.stateFor(name)
	if state.inFlight != nil {
		return state.inFlight
	}
	flight := &budgetFlight{done: make(chan struct{})}
	state.inFlight = flight
	// Stamped here, not in the goroutine, so a reschedule right after this call sees it.
	state.lastPollAt = c.options.now()
	ctx, started := c.runContextLocked(), c.generation
	c.runs.Add(1)
	go func() {
		defer c.runs.Done()
		outcome := c.run(ctx, started, host, name, state)
		c.mu.Lock()
		state.inFlight = nil
		flight.outcome = outcome
		c.mu.Unlock()
		close(flight.done)
	}()
	return flight
}

func (c *budgetController) trackLocked(host budgetHost, provider string) string {
	c.latestHost = host
	c.activeProvider = ""
	if _, ok := c.displayName[provider]; ok {
		c.activeProvider = provider
	}
	return c.activeProvider
}

func (c *budgetController) clearTimerLocked() {
	if c.stopTimer != nil {
		c.stopTimer()
		c.stopTimer = nil
	}
	c.timerSeq++
}

func (c *budgetController) warnOnce(host budgetHost) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.warned || c.options.displayWarning == "" {
		return
	}
	c.warned = true
	if host.HasUI() {
		host.Notify(c.options.displayWarning, "warning")
	} else {
		reportDiagnostic(c.options.displayWarning)
	}
}

func (c *budgetController) schedulePollLocked(name string) {
	state := c.stateFor(name)
	at := state.lastTurnAt.Add(pollSettleDelay)
	if !state.lastPollAt.IsZero() {
		if next := state.lastPollAt.Add(pollMinInterval); next.After(at) {
			at = next
		}
	}
	c.timerSeq++
	seq := c.timerSeq
	c.stopTimer = c.options.after(max(0, at.Sub(c.options.now())), func() { c.timerFired(seq, name, at) })
}

func (c *budgetController) timerFired(seq int, name string, planned time.Time) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if seq != c.timerSeq {
		return
	}
	c.stopTimer = nil
	if c.latestHost == nil || c.activeProvider != name || c.gated() {
		return
	}
	c.pollLocked(c.latestHost, name)
	// A turn that ended within the settle delay may not have its spend written yet: poll once more. Compared
	// with the planned time, not the clock, because a timer can fire a few milliseconds early.
	if c.stateFor(name).lastTurnAt.Add(pollSettleDelay).After(planned) {
		c.schedulePollLocked(name)
	}
}

func (c *budgetController) sessionStart(host budgetHost) {
	c.warnOnce(host)
	if !host.HasUI() {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if name := c.trackLocked(host, host.ModelProvider()); name != "" && !c.gated() {
		c.pollLocked(host, name)
	}
}

// providerSelected is the model_select handler, run when the active provider changes (see setupBudget).
func (c *budgetController) providerSelected(host budgetHost) {
	if !host.HasUI() {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	previous := c.activeProvider
	name := c.trackLocked(host, host.ModelProvider())
	// A pending turn poll belongs to the provider its turns ran on; a newly selected one is polled below if stale.
	if name != previous {
		c.clearTimerLocked()
	}
	if name == "" {
		if c.shown != "" {
			host.SetStatus(budgetStatusKey, "")
			c.shown = ""
		}
		return
	}
	c.renderLocked(host, name)
	state := c.stateFor(name)
	if c.gated() {
		return
	}
	if state.lastPollAt.IsZero() || c.options.now().Sub(state.lastPollAt) >= pollMinInterval {
		c.pollLocked(host, name)
	}
	// A turn whose poll was dropped when this provider was switched away from still needs one.
	missed := !state.lastTurnAt.IsZero() && state.lastTurnAt.Add(pollSettleDelay).After(state.lastPollAt)
	if missed && c.stopTimer == nil {
		c.schedulePollLocked(name)
	}
}

func (c *budgetController) beforeAgentStart(host budgetHost) {
	c.mu.Lock()
	changed := host.ModelProvider() != c.activeProvider
	c.mu.Unlock()
	if changed {
		c.providerSelected(host)
	}
}

func (c *budgetController) turnEnd(host budgetHost) {
	if !host.HasUI() {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	name := c.trackLocked(host, host.ModelProvider())
	if name == "" || c.gated() {
		return
	}
	c.stateFor(name).lastTurnAt = c.options.now()
	if c.stopTimer == nil {
		c.schedulePollLocked(name)
	}
}

func (c *budgetController) responseHeaders(host budgetHost, headers map[string]any) {
	if !host.HasUI() {
		return
	}
	name := host.ModelProvider()
	if _, ok := c.displayName[name]; !ok {
		return
	}
	lower := map[string]string{}
	for header, value := range headers {
		if text, ok := value.(string); ok {
			lower[strings.ToLower(header)] = text
		}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	state := c.stateFor(name)
	var key *budgetLevel
	if level, ok := state.levels[budgetKey]; ok {
		key = &level
	}
	if merged := mergeKeyHeaders(key, state.keyPolled, lower); merged != nil {
		state.levels[budgetKey] = *merged
	} else {
		delete(state.levels, budgetKey)
	}
	c.renderLocked(host, name)
}

func (c *budgetController) shutdown(host budgetHost) {
	c.mu.Lock()
	c.clearTimerLocked()
	c.generation++
	if c.cancelRuns != nil {
		c.cancelRuns()
		c.runCtx, c.cancelRuns = nil, nil
	}
	if c.shown != "" && host.HasUI() {
		host.SetStatus(budgetStatusKey, "")
	}
	c.shown, c.activeProvider, c.latestHost = "", "", nil
	c.mu.Unlock()
	// Drain the cancelled polls, but never let a stuck host call hold up the shutdown.
	drained := make(chan struct{})
	go func() { c.runs.Wait(); close(drained) }()
	select {
	case <-drained:
	case <-time.After(shutdownDrainBudget):
	}
}

func (c *budgetController) completions(prefix string) []sdk.AutocompleteItem {
	var items []sdk.AutocompleteItem
	for _, provider := range c.options.providers {
		if strings.HasPrefix(provider.name, prefix) {
			items = append(items, sdk.AutocompleteItem{Value: provider.name, Label: provider.name})
		}
	}
	return items
}

func (c *budgetController) notify(host budgetHost, message, level string) {
	if host.HasUI() {
		host.Notify(message, level)
		return
	}
	reportDiagnostic(message)
}

// command is the /litellm-budget handler. It ignores the host-offline gate: only LITELLM_OFFLINE and a zero timeout skip it.
func (c *budgetController) command(host budgetHost, args string) {
	if reason := c.options.disabledReason(); reason != "" {
		c.notify(host, fmt.Sprintf("LiteLLM: budget check skipped (%s).", reason), "warning")
		return
	}
	requested := strings.TrimSpace(args)
	var selected []string
	for _, provider := range c.options.providers {
		if requested == "" || provider.name == requested {
			selected = append(selected, provider.name)
		}
	}
	if len(selected) == 0 {
		configured := make([]string, len(c.options.providers))
		for i, provider := range c.options.providers {
			configured[i] = jsonString(provider.name)
		}
		c.notify(host, fmt.Sprintf("LiteLLM: unknown provider %s; configured: %s.", jsonString(requested), strings.Join(configured, ", ")), "warning")
		return
	}
	for _, name := range selected {
		label := "LiteLLM (" + jsonString(name) + ")"
		c.mu.Lock()
		pending := c.stateFor(name).inFlight
		c.mu.Unlock()
		if pending != nil {
			<-pending.done
		}
		c.mu.Lock()
		clear(c.stateFor(name).denied)
		flight := c.pollLocked(host, name)
		c.mu.Unlock()
		<-flight.done
		switch outcome := flight.outcome; outcome.kind {
		case outcomeNoAuth:
			c.notify(host, fmt.Sprintf("%s: budget check skipped; %s", label, c.options.missingCredentials(name)), "warning")
		case outcomeFailed:
			c.notify(host, fmt.Sprintf("%s: budget check failed (%s).", label, outcome.reason), "warning")
		default:
			c.mu.Lock()
			state := c.stateFor(name)
			details := formatBudgetDetails(name, state.levels, state.denied, c.options.now().UnixMilli())
			c.mu.Unlock()
			c.notify(host, details, "info")
		}
	}
}

// setupBudget is setupLiteLLMBudget. The factory calls it as `setupBudget(e, state)`.
//
// PiG does not fire model_select yet, so the active provider is tracked from before_agent_start (which runs
// the model_select logic when the provider differs from the tracked one) and turn_end, through
// ctx.ModelProvider(). A provider switch while idle therefore shows on the next prompt, not at once.
func setupBudget(e *sdk.Extension, s *extensionState) {
	if !isFeatureEnabled(s.settings, "budget") {
		return
	}
	var budgetSettings any
	if s.settings != nil {
		budgetSettings = s.settings.Values["budget"]
	}
	display, warning := budgetDisplaySetting(budgetSettings)
	options := budgetOptions{
		display:        display,
		displayWarning: warning,
		timeoutMs:      discoveryTimeoutMs,
		disabledReason: discoveryDisabledReason,
		hostOffline:    isHostOffline,
		missingCredentials: func(name string) string {
			for _, definition := range s.definitions {
				if definition.Name == name {
					return missingCredentials(definition)
				}
			}
			return ""
		},
	}
	for _, definition := range s.definitions {
		options.providers = append(options.providers, budgetProvider{definition.Name, definition.DisplayName})
	}
	c := newBudgetController(options)
	host := func(ctx sdk.Context) budgetHost { return sdkBudgetHost{Context: ctx, state: s} }

	e.OnSessionStart(func(ctx sdk.Context, _ map[string]any) (any, error) {
		c.sessionStart(host(ctx))
		return nil, nil
	})
	e.OnEvent(sdk.EventBeforeAgentStart, func(ctx sdk.Context, _ map[string]any) (any, error) {
		c.beforeAgentStart(host(ctx))
		return nil, nil
	})
	e.OnEvent(sdk.EventTurnEnd, func(ctx sdk.Context, _ map[string]any) (any, error) {
		c.turnEnd(host(ctx))
		return nil, nil
	})
	e.OnEvent(sdk.EventAfterProviderResponse, func(ctx sdk.Context, data map[string]any) (any, error) {
		headers, _ := data["headers"].(map[string]any)
		c.responseHeaders(host(ctx), headers)
		return nil, nil
	})
	e.OnSessionShutdown(func(ctx sdk.Context, _ map[string]any) (any, error) {
		c.shutdown(host(ctx))
		return nil, nil
	})
	e.RegisterCommand("litellm-budget", sdk.CommandOptions{
		Description:            "Show LiteLLM key, user, team, member, and organization budgets",
		GetArgumentCompletions: func(prefix string) ([]sdk.AutocompleteItem, error) { return c.completions(prefix), nil },
		Handler: func(ctx sdk.Context, args string) error {
			c.command(host(ctx), args)
			return nil
		},
	})
}
