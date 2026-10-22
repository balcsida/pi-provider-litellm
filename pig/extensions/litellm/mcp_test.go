package litellm

// Ports tests/mcp-server.test.ts ("LiteLLM MCP server registration") as unit tests of mcpSync.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	sdk "github.com/MichaelKinsy/PiG/extensions/sdk"

	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/catalog"
)

// fakeRegistrar records the servers the way the host holds them.
type fakeRegistrar struct {
	servers   map[string]sdk.McpServerConfig
	registers int
	refuse    func(sdk.McpServerConfig) error
}

func newFakeRegistrar() *fakeRegistrar {
	return &fakeRegistrar{servers: map[string]sdk.McpServerConfig{}}
}

func (f *fakeRegistrar) RegisterMcpServer(name string, config sdk.McpServerConfig) error {
	f.registers++
	if f.refuse != nil {
		if err := f.refuse(config); err != nil {
			return err
		}
	}
	f.servers[name] = config
	return nil
}

func (f *fakeRegistrar) UnregisterMcpServer(name string) { delete(f.servers, name) }

// mcpFixture writes the settings, builds the state and returns the sync for them.
func mcpFixture(t *testing.T, settings string) (dir string, state *extensionState, m *mcpSync) {
	t.Helper()
	dir = hermeticAgentDir(t)
	if settings == "" {
		settings = `{"litellm":{}}`
	}
	raw := writeFileIn(t, dir, "settings.json", settings)
	data, _ := os.ReadFile(raw)
	state, _ = startupIn(t, dir)
	state.settings = readGlobalLiteLLMSettings()
	state.definitions = getProviderDefinitions(state.settings)
	m, err := newMcpSync(state, data)
	if err != nil {
		t.Fatal(err)
	}
	return dir, state, m
}

func defaultEnv(t *testing.T, root string) {
	t.Helper()
	t.Setenv(envBaseURL, root)
	t.Setenv(envAPIKey, "sk-default")
}

func mcpJSON(t *testing.T, f *fakeRegistrar) string {
	t.Helper()
	data, err := json.Marshal(f.servers)
	if err != nil {
		t.Fatal(err)
	}
	return string(data)
}

func TestMcpRegistersEachProvidersEndpointAuthenticatedByTheProviderItself(t *testing.T) {
	_, _, m := mcpFixture(t, `{"litellm":{"providers":{"team":{"baseUrl":"https://team.example.com/root","apiKey":"sk-team","headers":{"x-team":"blue"}}}}}`)
	defaultEnv(t, "https://proxy.example.com")
	f := newFakeRegistrar()
	m.syncAll(f)
	want := `{"litellm":{"type":"http","url":"https://proxy.example.com/mcp","auth":{"provider":"litellm"}},` +
		`"team":{"type":"http","url":"https://team.example.com/root/mcp","headers":{"x-team":"blue"},"auth":{"provider":"team"}}}`
	if got := mcpJSON(t, f); got != want {
		t.Fatalf("servers = %s, want %s", got, want)
	}
	if strings.Contains(mcpJSON(t, f), "sk-") {
		t.Fatal("a key reached the registration")
	}
}

func TestMcpPassesTheExposureSettingsUnchanged(t *testing.T) {
	_, _, m := mcpFixture(t, `{"litellm":{"mcp":{"exposure":"deferred","toolExposure":{"limble-get_*":"direct","*delete*":"hidden"}}}}`)
	defaultEnv(t, "https://proxy.example.com")
	f := newFakeRegistrar()
	m.syncAll(f)
	config := f.servers["litellm"]
	if config.Exposure != "deferred" || strings.Join(config.ToolExposure.Keys(), ",") != "limble-get_*,*delete*" {
		t.Fatalf("config = %+v", config)
	}
	if exposure, _ := config.ToolExposure.Get("*delete*"); exposure != "hidden" {
		t.Fatalf("*delete* = %q", exposure)
	}
}

func TestMcpRegistersNothingWhenDisabled(t *testing.T) {
	hermeticAgentDir(t)
	if isFeatureEnabled(&litellmSettings{Values: map[string]any{"mcp": map[string]any{"enabled": false}}}, "mcp") {
		t.Fatal("mcp.enabled=false keeps MCP on")
	}
}

func TestMcpRegistersNothingWhenNetworkAccessIsDisabled(t *testing.T) {
	for _, tc := range []struct{ name, value string }{
		{envOffline, "1"}, {envTimeout, "0"}, {envPIOffline, "1"}, {envPIOffline, "0"}, {envPIOffline, ""},
		{envPIGOffline, "1"},
	} {
		t.Run(tc.name+"="+tc.value, func(t *testing.T) {
			_, _, m := mcpFixture(t, "")
			defaultEnv(t, "https://proxy.example.com")
			t.Setenv(tc.name, tc.value)
			f := newFakeRegistrar()
			m.syncAll(f)
			if len(f.servers) != 0 {
				t.Fatalf("servers = %s", mcpJSON(t, f))
			}
		})
	}
}

func TestMcpFollowsALoginThatSuppliesTheProxyRootAndALogoutThatRemovesIt(t *testing.T) {
	dir, _, m := mcpFixture(t, "")
	f := newFakeRegistrar()
	m.syncAll(f)
	if len(f.servers) != 0 {
		t.Fatalf("servers = %s", mcpJSON(t, f))
	}
	writeFileIn(t, dir, "auth.json", `{"litellm":{"type":"api_key","key":"sk-login","env":{"LITELLM_BASE_URL":"https://login.example.com"}}}`)
	m.syncAll(f)
	if got, want := mcpJSON(t, f), `{"litellm":{"type":"http","url":"https://login.example.com/mcp","auth":{"provider":"litellm"}}}`; got != want {
		t.Fatalf("servers = %s, want %s", got, want)
	}
	if err := os.Remove(filepath.Join(dir, "auth.json")); err != nil {
		t.Fatal(err)
	}
	m.syncAll(f)
	if len(f.servers) != 0 {
		t.Fatalf("servers after logout = %s", mcpJSON(t, f))
	}
}

func TestMcpRegistersAgainOnlyWhenTheRegistrationChanges(t *testing.T) {
	_, _, m := mcpFixture(t, "")
	defaultEnv(t, "https://proxy.example.com")
	f := newFakeRegistrar()
	m.syncAll(f)
	m.syncAll(f)
	m.syncAll(f)
	if f.registers != 1 {
		t.Fatalf("registers = %d, want 1", f.registers)
	}
	t.Setenv(envBaseURL, "https://moved.example.com")
	m.syncAll(f)
	if f.registers != 2 || f.servers["litellm"].URL != "https://moved.example.com/mcp" {
		t.Fatalf("registers = %d, url = %s", f.registers, f.servers["litellm"].URL)
	}
}

func TestMcpReportsARefusedRegistrationOnceWithoutRetryingEveryTurn(t *testing.T) {
	_, _, m := mcpFixture(t, `{"litellm":{"mcp":{"exposure":"everywhere"}}}`)
	defaultEnv(t, "https://proxy.example.com")
	var messages []string
	old := reportDiagnostic
	reportDiagnostic = func(message string) { messages = append(messages, message) }
	t.Cleanup(func() { reportDiagnostic = old })
	f := newFakeRegistrar()
	f.refuse = func(config sdk.McpServerConfig) error {
		return errors.New(`server "litellm": exposure must be one of "codemode", "deferred", "direct", "hidden"`)
	}
	m.syncAll(f)
	m.syncAll(f)
	if f.registers != 1 {
		t.Fatalf("registers = %d, want 1", f.registers)
	}
	want := `LiteLLM MCP ("litellm"): server "litellm": exposure must be one of "codemode", "deferred", "direct", "hidden"`
	if len(messages) != 1 || messages[0] != want {
		t.Fatalf("messages = %q", messages)
	}
}

func TestMcpHoldsNoticesUntilTheSessionUICanShowThem(t *testing.T) {
	hooks, state, host := wire(t, "")
	defaultEnv(t, "https://proxy.example.com")
	diagnostics := collectDiagnostics(t)
	previous := stderrIsTerminal
	stderrIsTerminal = func() bool { return true }
	t.Cleanup(func() { stderrIsTerminal = previous })
	host.refuse = func(sdk.McpServerConfig) error { return errors.New("refused") }
	m, err := newMcpSync(state, nil)
	if err != nil {
		t.Fatal(err)
	}

	// The terminal is starting: the refusal is held, not written over the TUI.
	m.syncAll(host)
	if len(*diagnostics) != 0 || len(host.notifications()) != 0 || len(state.pendingMessages) != 1 {
		t.Fatalf("diagnostics %q, notices %v, pending %v", *diagnostics, host.notifications(), state.pendingMessages)
	}
	// session_start hands it to the session's UI, once.
	hooks.only(t, sdk.EventSessionStart, 0, host, nil)
	want := budgetNotice{`LiteLLM MCP ("litellm"): refused`, "warning"}
	if got := host.notifications(); len(got) != 1 || got[0] != want || len(state.pendingMessages) != 0 || len(*diagnostics) != 0 {
		t.Fatalf("notices %v, pending %v, diagnostics %q", got, state.pendingMessages, *diagnostics)
	}
}

func TestMcpKeepsResolvedHeaderValuesLiteral(t *testing.T) {
	_, _, m := mcpFixture(t, "")
	defaultEnv(t, "https://proxy.example.com")
	t.Setenv(envHeaders, `{"x-price":"$$HOME","x-token":"!echo pwned","x-plain":"tok"}`)
	f := newFakeRegistrar()
	m.syncAll(f)
	headers := f.servers["litellm"].Headers
	for name, want := range map[string]string{"x-price": "$$HOME", "x-token": "$!echo pwned", "x-plain": "tok"} {
		if got, _ := headers.Get(name); got != want {
			t.Errorf("%s = %q, want %q", name, got, want)
		}
	}
}

func TestMcpDisconnectsWhenALoginStartsBeforeItCanStoreANewCredential(t *testing.T) {
	dir, _, m := mcpFixture(t, "")
	defaultEnv(t, "https://old.example.com")
	f := newFakeRegistrar()
	m.syncAll(f)
	if _, ok := f.servers["litellm"]; !ok {
		t.Fatal("not registered")
	}
	m.drop(f, m.state.definitions[0])
	if _, ok := f.servers["litellm"]; ok {
		t.Fatal("still registered while the login prompts")
	}
	writeFileIn(t, dir, "auth.json", `{"litellm":{"type":"api_key","key":"sk-new","env":{"LITELLM_BASE_URL":"https://new.example.com"}}}`)
	m.syncAll(f)
	if f.servers["litellm"].URL != "https://new.example.com/mcp" {
		t.Fatalf("url = %s", f.servers["litellm"].URL)
	}
}

func TestMcpWithdrawsThePreviousServerWhenTheHostRefusesItsReplacement(t *testing.T) {
	_, _, m := mcpFixture(t, "")
	defaultEnv(t, "https://proxy.example.com")
	old := reportDiagnostic
	reportDiagnostic = func(string) {}
	t.Cleanup(func() { reportDiagnostic = old })
	f := newFakeRegistrar()
	f.refuse = func(config sdk.McpServerConfig) error {
		if strings.HasPrefix(config.URL, "https://moved.") {
			return errors.New("refused")
		}
		return nil
	}
	m.syncAll(f)
	if _, ok := f.servers["litellm"]; !ok {
		t.Fatal("not registered")
	}
	t.Setenv(envBaseURL, "https://moved.example.com")
	m.syncAll(f)
	m.syncAll(f)
	if _, ok := f.servers["litellm"]; ok {
		t.Fatal("previous server kept after refusal")
	}
	if f.registers != 2 {
		t.Fatalf("registers = %d, want 2", f.registers)
	}
}

func TestMcpFirstSyncRunsWhenTheSessionStartsNotWhileTheExtensionLoads(t *testing.T) {
	hooks, _, host := wire(t, "")
	defaultEnv(t, "https://proxy.example.com")
	// setupHooks registered through a registrar that has no RegisterMcpServer, so the factory cannot queue one;
	// nothing reaches the host until an event delivers a context.
	if host.registers != 0 {
		t.Fatalf("registers while loading = %d", host.registers)
	}
	hooks.emit(t, sdk.EventSessionStart, host, nil)
	if host.registers != 1 || host.servers["litellm"].URL != "https://proxy.example.com/mcp" {
		t.Fatalf("registers = %d, servers = %s", host.registers, mcpJSON(t, host.fakeRegistrar))
	}
	hooks.emit(t, sdk.EventBeforeAgentStart, host, map[string]any{"systemPrompt": fakePrompt("")})
	if host.registers != 1 {
		t.Fatalf("the turn registered again: %d", host.registers)
	}
	t.Setenv(envBaseURL, "https://moved.example.com")
	hooks.emit(t, sdk.EventBeforeAgentStart, host, map[string]any{"systemPrompt": fakePrompt("")})
	if host.registers != 2 || host.servers["litellm"].URL != "https://moved.example.com/mcp" {
		t.Fatalf("registers = %d, servers = %s", host.registers, mcpJSON(t, host.fakeRegistrar))
	}
}

func TestMcpAccessCheckStartsWhileTheExtensionLoadsAndSessionStartUsesItsAnswer(t *testing.T) {
	// wire, with the proxy configured and answering before the extension loads.
	dir := hermeticAgentDir(t)
	writeFileIn(t, dir, "settings.json", `{"litellm":{}}`)
	defaultEnv(t, "https://proxy.example.com")
	sent := make(chan struct{}, 1)
	checks := mcpProxy(t, func(r *http.Request) (*http.Response, error) {
		sent <- struct{}{}
		return noServersGranted(r)
	})
	parsed := readGlobalLiteLLMSettings()
	hooks := &recordingHooks{}
	setupHooks(hooks, newExtensionState(parsed, getProviderDefinitions(parsed), newPolicyStore(filepath.Join(dir, policiesFilename))))
	host := newFakeHookHost()
	select {
	case <-sent:
	case <-time.After(5 * time.Second):
		t.Fatal("no access check while the extension loads")
	}
	hooks.emit(t, sdk.EventSessionStart, host, nil)
	// session_start waited for the check started while loading, so its request is recorded.
	if len(*checks) != 1 {
		t.Fatalf("checks = %+v, want the one sent while loading", *checks)
	}
	want := `{"litellm":{"type":"http","enabled":false,"url":"https://proxy.example.com/mcp","auth":{"provider":"litellm"}}}`
	if got := mcpJSON(t, host.fakeRegistrar); got != want {
		t.Fatalf("servers = %s, want %s", got, want)
	}
}

func TestMcpSessionShutdownCancelsAnUnusedAccessCheckStartedWhileLoadingAndWaitsForIt(t *testing.T) {
	// wire, with the proxy configured before the extension loads.
	dir := hermeticAgentDir(t)
	writeFileIn(t, dir, "settings.json", `{"litellm":{}}`)
	defaultEnv(t, "https://proxy.example.com")
	first := make(chan struct{}, 1)
	first <- struct{}{}
	sent := make(chan struct{})
	ended := make(chan error, 1)
	checks := mcpProxy(t, func(r *http.Request) (*http.Response, error) {
		select {
		case <-first:
		default:
			return noServersGranted(r)
		}
		// The check started while loading is still unanswered when the session ends.
		close(sent)
		<-r.Context().Done()
		ended <- r.Context().Err()
		return nil, r.Context().Err()
	})
	parsed := readGlobalLiteLLMSettings()
	hooks := &recordingHooks{}
	setupHooks(hooks, newExtensionState(parsed, getProviderDefinitions(parsed), newPolicyStore(filepath.Join(dir, policiesFilename))))
	host := newFakeHookHost()
	select {
	case <-sent:
	case <-time.After(5 * time.Second):
		t.Fatal("no access check while the extension loads")
	}
	hooks.emit(t, sdk.EventSessionShutdown, host, nil)
	// Its own budget had seconds left: shutdown cancelled it and waited for it to end.
	select {
	case err := <-ended:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("the check ended with %v, want it cancelled", err)
		}
	default:
		t.Fatal("session_shutdown returned while the access check was still running")
	}
	// The next session asks the proxy itself rather than taking the cancelled check for an answer.
	hooks.emit(t, sdk.EventSessionStart, host, nil)
	if len(*checks) != 2 {
		t.Fatalf("checks = %+v, want the cancelled one and a new one", *checks)
	}
	want := `{"litellm":{"type":"http","enabled":false,"url":"https://proxy.example.com/mcp","auth":{"provider":"litellm"}}}`
	if got := mcpJSON(t, host.fakeRegistrar); got != want {
		t.Fatalf("servers = %s, want %s", got, want)
	}
}

func TestMcpReportsABogusExposureAndKeepsTheExtensionLoaded(t *testing.T) {
	hooks, _, host := wire(t, `{"litellm":{"mcp":{"exposure":"bogus"}}}`)
	defaultEnv(t, "https://proxy.example.com")
	diagnostics := collectDiagnostics(t)
	// The host validates the config like PiG does and answers with an error instead of failing a load.
	host.refuse = func(config sdk.McpServerConfig) error {
		switch config.Exposure {
		case "", sdk.McpExposureCodemode, sdk.McpExposureDeferred, sdk.McpExposureDirect, sdk.McpExposureHidden:
			return nil
		}
		return fmt.Errorf("server \"litellm\": exposure must be one of codemode, deferred, direct, hidden")
	}
	hooks.emit(t, sdk.EventSessionStart, host, nil)
	hooks.emit(t, sdk.EventBeforeAgentStart, host, map[string]any{"systemPrompt": fakePrompt("")})
	hooks.emit(t, sdk.EventBeforeAgentStart, host, map[string]any{"systemPrompt": fakePrompt("")})
	want := budgetNotice{`LiteLLM MCP ("litellm"): server "litellm": exposure must be one of codemode, deferred, direct, hidden`, "warning"}
	if got := host.notifications(); len(got) != 1 || got[0] != want {
		t.Fatalf("notices = %v, want one %v", got, want)
	}
	if host.registers != 1 || len(host.servers) != 0 || len(*diagnostics) != 0 {
		t.Fatalf("registers %d, servers %s, diagnostics %q", host.registers, mcpJSON(t, host.fakeRegistrar), *diagnostics)
	}
}

func TestMcpRegistersNothingWhenMCPIsDisabledAndKeepsSkills(t *testing.T) {
	hooks, _, host := wire(t, `{"litellm":{"skills":{"enabled":true},"mcp":{"enabled":false}}}`)
	defaultEnv(t, "https://proxy.example.com")
	hooks.emit(t, sdk.EventSessionStart, host, nil)
	hooks.emit(t, sdk.EventBeforeAgentStart, host, map[string]any{"systemPrompt": fakePrompt("")})
	if host.registers != 0 || len(host.servers) != 0 {
		t.Fatalf("registers %d, servers %s", host.registers, mcpJSON(t, host.fakeRegistrar))
	}
	if !slices.Contains(hooks.tools, "litellm_skill_list") {
		t.Fatalf("tools = %v", hooks.tools)
	}
}

func TestMcpLoginStartDropsTheServerThroughTheSessionHost(t *testing.T) {
	hooks, state, host := wire(t, "")
	defaultEnv(t, "https://proxy.example.com")
	hooks.emit(t, sdk.EventSessionStart, host, nil)
	if _, ok := host.servers["litellm"]; !ok {
		t.Fatal("not registered")
	}
	state.mu.Lock()
	callbacks := append([]func(providerDefinition){}, state.onLoginStart...)
	state.mu.Unlock()
	for _, callback := range callbacks {
		callback(state.definitions[0])
	}
	if _, ok := host.servers["litellm"]; ok {
		t.Fatal("still registered while the login prompts")
	}
}

// roundTripFunc adapts a function to http.RoundTripper.
type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// mcpCheck is one request the access check sent.
type mcpCheck struct {
	method, url, body string
	header            http.Header
	deadline          time.Time
}

// mcpProxy answers the access checks in-process, as the TypeScript fetch mock does, and records them.
func mcpProxy(t *testing.T, answer func(*http.Request) (*http.Response, error)) *[]mcpCheck {
	t.Helper()
	checks := &[]mcpCheck{}
	var mu sync.Mutex
	previous := mcpTransport
	mcpTransport = roundTripFunc(func(r *http.Request) (*http.Response, error) {
		var body []byte
		if r.Body != nil {
			body, _ = io.ReadAll(r.Body)
		}
		deadline, _ := r.Context().Deadline()
		mu.Lock()
		*checks = append(*checks, mcpCheck{r.Method, r.URL.String(), string(body), r.Header.Clone(), deadline})
		mu.Unlock()
		return answer(r)
	})
	// Registered last so it runs first: a detached DELETE must not outlive the transport it uses.
	t.Cleanup(func() { mcpTransport = previous })
	t.Cleanup(func() { drainDetached(t) })
	return checks
}

// drainDetached waits for the requests the extension detaches, such as the check's cleanup DELETE.
func drainDetached(t *testing.T) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := catalog.Drain(ctx); err != nil {
		t.Fatalf("detached requests did not finish: %v", err)
	}
}

func mcpAnswer(status int, headers map[string]string) (*http.Response, error) {
	response := &http.Response{StatusCode: status, Header: http.Header{}, Body: io.NopCloser(strings.NewReader("{}"))}
	for name, value := range headers {
		response.Header.Set(name, value)
	}
	return response, nil
}

// noServersGranted is LiteLLM 1.102.0's answer to `initialize` for a key with no MCP servers granted.
func noServersGranted(*http.Request) (*http.Response, error) { return mcpAnswer(403, nil) }

func TestMcpRegistersTheServerDisabledWithoutANoticeWhenTheProxyRefusesTheKeyAtInitialize(t *testing.T) {
	_, state, m := mcpFixture(t, "")
	checks := mcpProxy(t, noServersGranted)
	defaultEnv(t, "https://proxy.example.com")
	diagnostics := collectDiagnostics(t)
	f := newFakeRegistrar()
	m.syncAll(f)
	m.syncAll(f)
	want := `{"litellm":{"type":"http","enabled":false,"url":"https://proxy.example.com/mcp","auth":{"provider":"litellm"}}}`
	if got := mcpJSON(t, f); got != want {
		t.Fatalf("servers = %s, want %s", got, want)
	}
	if len(*checks) != 1 || len(*diagnostics) != 0 || len(state.pendingMessages) != 0 {
		t.Fatalf("checks = %d, diagnostics = %q, pending = %v", len(*checks), *diagnostics, state.pendingMessages)
	}
}

func TestMcpChecksWithTheCredentialAndHeadersTheHostSendsAndEndsTheSessionTheCheckOpened(t *testing.T) {
	_, _, m := mcpFixture(t, "")
	checks := mcpProxy(t, func(r *http.Request) (*http.Response, error) {
		if r.Method != http.MethodPost {
			return mcpAnswer(200, nil)
		}
		// Long enough for a cleanup on its own clock to show a later deadline.
		time.Sleep(5 * time.Millisecond)
		return mcpAnswer(200, map[string]string{"mcp-session-id": "check-session"})
	})
	defaultEnv(t, "https://proxy.example.com")
	t.Setenv(envHeaders, `{"x-mcp-client":"pi"}`)
	f := newFakeRegistrar()
	m.syncAll(f)
	want := `{"litellm":{"type":"http","url":"https://proxy.example.com/mcp","headers":{"x-mcp-client":"pi"},"auth":{"provider":"litellm"}}}`
	if got := mcpJSON(t, f); got != want {
		t.Fatalf("servers = %s, want %s", got, want)
	}
	drainDetached(t)
	if len(*checks) != 2 {
		t.Fatalf("checks = %+v", *checks)
	}
	check, closing := (*checks)[0], (*checks)[1]
	if check.method != http.MethodPost || check.url != "https://proxy.example.com/mcp" ||
		check.header.Get("Authorization") != "Bearer sk-default" || check.header.Get("x-mcp-client") != "pi" ||
		!strings.Contains(check.body, `"method":"initialize"`) {
		t.Fatalf("check = %+v", check)
	}
	if closing.method != http.MethodDelete || closing.header.Get("mcp-session-id") != "check-session" ||
		closing.header.Get("Authorization") != "Bearer sk-default" {
		t.Fatalf("close = %+v", closing)
	}
	// The cleanup runs on its own budget, so a slow initialize cannot starve it.
	if !closing.deadline.After(check.deadline) {
		t.Fatalf("close deadline %v is not after the check's %v", closing.deadline, check.deadline)
	}
}

func TestMcpSyncDoesNotWaitForTheCleanupDeleteOfTheCheckSession(t *testing.T) {
	_, _, m := mcpFixture(t, "")
	release := make(chan struct{})
	deleting := make(chan struct{})
	checks := mcpProxy(t, func(r *http.Request) (*http.Response, error) {
		if r.Method == http.MethodDelete {
			close(deleting)
			<-release
			return mcpAnswer(200, nil)
		}
		return mcpAnswer(200, map[string]string{"mcp-session-id": "check-session"})
	})
	released := false
	releaseDelete := func() {
		if !released {
			released = true
			close(release)
		}
	}
	t.Cleanup(releaseDelete) // runs before mcpProxy's drain, so a failed assertion cannot leave the DELETE blocked
	defaultEnv(t, "https://proxy.example.com")
	f := newFakeRegistrar()
	synced := make(chan struct{})
	go func() {
		m.syncAll(f)
		close(synced)
	}()
	select {
	case <-synced:
	case <-time.After(3 * time.Second):
		releaseDelete()
		t.Fatal("syncAll waited for the blocked DELETE")
	}
	if _, ok := f.servers["litellm"]; !ok {
		t.Fatalf("servers = %s", mcpJSON(t, f))
	}
	select {
	case <-deleting:
	case <-time.After(3 * time.Second):
		t.Fatal("the detached DELETE was never sent")
	}
	releaseDelete()
	drainDetached(t)
	if len(*checks) != 2 {
		t.Fatalf("checks = %+v", *checks)
	}
}

func TestMcpRegistersTheServerEnabledWhenTheCheckGetsAnotherStatus(t *testing.T) {
	for _, status := range []int{401, 404, 500} {
		t.Run(fmt.Sprint(status), func(t *testing.T) {
			_, _, m := mcpFixture(t, "")
			mcpProxy(t, func(*http.Request) (*http.Response, error) { return mcpAnswer(status, nil) })
			defaultEnv(t, "https://proxy.example.com")
			f := newFakeRegistrar()
			m.syncAll(f)
			if config, ok := f.servers["litellm"]; !ok || config.Enabled != nil {
				t.Fatalf("servers = %s", mcpJSON(t, f))
			}
		})
	}
}

func TestMcpSkipsTheCheckForAKeyHelperAndRegistersEnabled(t *testing.T) {
	_, _, m := mcpFixture(t, `{"litellm":{"apiKey":"!echo sk-helper"}}`)
	checks := mcpProxy(t, noServersGranted)
	t.Setenv(envBaseURL, "https://proxy.example.com")
	f := newFakeRegistrar()
	m.syncAll(f)
	if config, ok := f.servers["litellm"]; !ok || config.Enabled != nil || len(*checks) != 0 {
		t.Fatalf("servers = %s, checks = %d", mcpJSON(t, f), len(*checks))
	}
}

func TestMcpGivesUpACheckTheProxyDoesNotAnswerWithinTheDiscoveryBudget(t *testing.T) {
	_, _, m := mcpFixture(t, "")
	t.Setenv(envTimeout, "50")
	mcpProxy(t, func(r *http.Request) (*http.Response, error) {
		<-r.Context().Done()
		return nil, r.Context().Err()
	})
	defaultEnv(t, "https://proxy.example.com")
	f := newFakeRegistrar()
	m.syncAll(f)
	if config, ok := f.servers["litellm"]; !ok || config.Enabled != nil {
		t.Fatalf("servers = %s", mcpJSON(t, f))
	}
}

func TestMcpDropsACheckThatALoginOvertakesAndChecksAgainOnTheNextTurn(t *testing.T) {
	_, state, m := mcpFixture(t, "")
	f := newFakeRegistrar()
	overtaken := false
	checks := mcpProxy(t, func(*http.Request) (*http.Response, error) {
		if !overtaken {
			overtaken = true
			// A login starts while the check is in flight.
			m.drop(f, state.definitions[0])
		}
		return mcpAnswer(200, nil)
	})
	defaultEnv(t, "https://proxy.example.com")
	m.syncAll(f)
	if len(f.servers) != 0 || len(*checks) != 1 {
		t.Fatalf("servers = %s, checks = %d", mcpJSON(t, f), len(*checks))
	}
	m.syncAll(f)
	if _, ok := f.servers["litellm"]; !ok || len(*checks) != 2 {
		t.Fatalf("servers = %s, checks = %d", mcpJSON(t, f), len(*checks))
	}
}
