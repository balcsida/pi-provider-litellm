package litellm

// Ports tests/mcp-server.test.ts ("LiteLLM MCP server registration") as unit tests of mcpSync.

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	sdk "github.com/MichaelKinsy/PiG/extensions/sdk"
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
	_, state, m := mcpFixture(t, "")
	defaultEnv(t, "https://proxy.example.com")
	var messages []string
	old := reportDiagnostic
	reportDiagnostic = func(message string) { messages = append(messages, message) }
	t.Cleanup(func() { reportDiagnostic = old })
	// notifyMcp's buffering is covered with the session state; without a terminal a refusal goes straight to
	// the diagnostic stream, once, and nothing is left pending.
	f := newFakeRegistrar()
	f.refuse = func(sdk.McpServerConfig) error { return errors.New("refused") }
	m.syncAll(f)
	if len(messages) != 1 || messages[0] != `LiteLLM MCP ("litellm"): refused` || len(state.pendingMessages) != 0 {
		t.Fatalf("messages = %q, pending = %v", messages, state.pendingMessages)
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
