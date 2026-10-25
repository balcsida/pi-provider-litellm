package litellm

// Ports src/index.ts: the LiteLLM MCP registration (mcpServerConfig, dropMcpServer, syncMcpServer).
// LiteLLM serves MCP over streamable HTTP at `/mcp`, so each provider registers it as a PiG MCP server and
// the host's client does the rest. `auth.provider` makes the host send the provider's current credential on
// every request, so token refreshes and key helpers apply without registering again; only a new proxy root
// or new headers change the registration.

import (
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	sdk "github.com/MichaelKinsy/PiG/extensions/sdk"

	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/catalog"
)

// mcpTransport is the transport of the access check; nil means http.DefaultTransport. Tests inject one that
// answers in-process.
var mcpTransport http.RoundTripper

// mcpRegistrar is the part of *sdk.Extension and sdk.Context that registers MCP servers.
type mcpRegistrar interface {
	RegisterMcpServer(name string, config sdk.McpServerConfig) error
	UnregisterMcpServer(name string)
}

// mcpAttempt is the last config attempted for a provider, and whether the host accepted it.
type mcpAttempt struct {
	identity   string
	registered bool
}

// mcpSync registers and withdraws the MCP servers of the configured providers.
type mcpSync struct {
	state        *extensionState
	exposure     sdk.McpExposure
	toolExposure *sdk.OrderedExposures
	// settingsErr is an exposure setting that cannot be carried to the host; it is reported like a host refusal.
	settingsErr error

	mu       sync.Mutex
	attempts map[string]*mcpAttempt
	last     mcpRegistrar
	// early holds each provider's access check started while the extension loads, for the first sync to use.
	early map[string]*earlyCheck
	// Headers can carry credentials, so a config is remembered as a keyed digest.
	salt []byte
}

func newMcpSync(state *extensionState, settingsRaw []byte) (*mcpSync, error) {
	salt := make([]byte, 32)
	if _, err := rand.Read(salt); err != nil {
		return nil, err
	}
	m := &mcpSync{state: state, attempts: map[string]*mcpAttempt{}, early: map[string]*earlyCheck{}, salt: salt}
	// The host validates exposure and toolExposure; an unusable value is reported when registering. The
	// settings file is decoded again here because toolExposure patterns are ordered and the decoded map is not.
	var top map[string]json.RawMessage
	if json.Unmarshal(settingsRaw, &top) != nil {
		return m, nil
	}
	var block map[string]json.RawMessage
	if json.Unmarshal(top[settingsKey], &block) != nil {
		return m, nil
	}
	var mcp struct {
		Exposure     sdk.McpExposure       `json:"exposure"`
		ToolExposure *sdk.OrderedExposures `json:"toolExposure"`
	}
	if raw, ok := block["mcp"]; ok && json.Unmarshal(raw, &mcp) != nil {
		var object map[string]json.RawMessage
		if json.Unmarshal(raw, &object) == nil {
			m.settingsErr = fmt.Errorf("mcp exposure settings must be strings and an object of strings")
		}
	}
	m.exposure, m.toolExposure = mcp.Exposure, mcp.ToolExposure
	return m, nil
}

// config is mcpServerConfig: nil when MCP is off, offline, or the provider has no usable proxy root.
func (m *mcpSync) config(definition providerDefinition) *sdk.McpServerConfig {
	if definition.EnableMCP != nil && !*definition.EnableMCP {
		return nil
	}
	if discoveryDisabledReason() != "" || isHostOffline() {
		return nil
	}
	stored := readStoredCredential(definition.Name, filepath.Join(agentDir(), "auth.json"))
	root, err := credentialRoot(definition, credentialInfoFromStored(stored))
	if err != nil {
		return nil
	}
	config := &sdk.McpServerConfig{
		Type:         "http",
		URL:          root + "/mcp",
		Auth:         &sdk.McpAuthConfig{Provider: definition.Name},
		Exposure:     m.exposure,
		ToolExposure: m.toolExposure,
	}
	if headers := resolveHeaders(definition); len(headers) > 0 {
		names := make([]string, 0, len(headers))
		for name := range headers {
			names = append(names, name)
		}
		sort.Strings(names)
		pairs := make([]string, 0, 2*len(names))
		for _, name := range names {
			pairs = append(pairs, name, piLiteral(headers[name]))
		}
		config.Headers = sdk.NewOrderedStrings(pairs...)
	}
	return config
}

func (m *mcpSync) identity(config *sdk.McpServerConfig) string {
	if config == nil {
		return ""
	}
	data, _ := json.Marshal(config)
	mac := hmac.New(sha256.New, m.salt)
	mac.Write(data)
	return hex.EncodeToString(mac.Sum(nil))
}

// drop is dropMcpServer.
func (m *mcpSync) drop(registrar mcpRegistrar, definition providerDefinition) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.dropLocked(registrar, definition)
}

func (m *mcpSync) dropLocked(registrar mcpRegistrar, definition providerDefinition) {
	attempt := m.attempts[definition.Name]
	delete(m.attempts, definition.Name)
	if attempt != nil && attempt.registered {
		registrar.UnregisterMcpServer(definition.Name)
	}
}

// mcpRefusal is what the access check learned: why the server is off, or mcpNotRefused.
type mcpRefusal int

const (
	mcpNotRefused mcpRefusal = iota
	mcpNoServers
	mcpOutOfBudget
)

// mcpRefusalBodyLimit bounds how much of a refusal's body is read.
const mcpRefusalBodyLimit = 64 << 10

// mcpAccessRefused is mcpAccessRefused. LiteLLM 1.102.0 and later answer `initialize` with 403 when the key
// has no MCP servers granted, and a host warns at every start about a server that fails to connect. Asking
// first lets such a server be registered disabled: the host neither connects to it nor warns, and the user
// can enable it once servers are granted. Ending ctx ends the check, which then counts as not refused.
func mcpAccessRefused(ctx context.Context, definition providerDefinition, url string) mcpRefusal {
	stored := readStoredCredential(definition.Name, filepath.Join(agentDir(), "auth.json"))
	// The check must send the credential the host will send. Like seeding, it runs no key helper and mints
	// no ADC token, and resolving without them would fall through to another key.
	if (stored != nil && stored.Type == "api_key" && strings.HasPrefix(stored.Key, "!")) ||
		strings.HasPrefix(definition.APIKeyConfig, "!") ||
		usesGcloudToken(definition) {
		return mcpNotRefused
	}
	budget := time.Duration(seedTimeoutMsBudget()) * time.Millisecond
	ctx, cancel := context.WithTimeout(ctx, budget)
	defer cancel()
	var apiKey, root string
	if stored != nil && stored.Type == "oauth" {
		resolved, err := credentialRoot(definition, credentialInfoFromStored(stored))
		if err != nil {
			return mcpNotRefused
		}
		apiKey, root = stored.Access, resolved
	} else {
		auth, err := resolveAPIKeyAuth(ctx, definition, processEnv, credentialFromStored(stored), false)
		if err != nil || auth == nil {
			return mcpNotRefused
		}
		apiKey, root = auth.Auth.APIKey, auth.Auth.BaseURL
	}
	// The credential goes only to its own root.
	if apiKey == "" || root+"/mcp" != url {
		return mcpNotRefused
	}
	// The credential and headers the host's client sends, so the proxy answers as it will answer the host.
	headers := http.Header{}
	for name, value := range resolveHeaders(definition) {
		headers.Set(name, value)
	}
	headers.Set("Authorization", "Bearer "+apiKey)
	client := &http.Client{
		Transport:     mcpTransport,
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
	// The proxy's answer does not depend on the client it is told about.
	body := `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"pig","version":"0"}}}`
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(body))
	if err != nil {
		return mcpNotRefused
	}
	request.Header = headers.Clone()
	request.Header.Set("Accept", "application/json, text/event-stream")
	request.Header.Set("Content-Type", "application/json")
	response, err := client.Do(request)
	if err != nil {
		return mcpNotRefused
	}
	// Only a refusal is read: a granted session can answer with an open event stream.
	var refusal []byte
	if response.StatusCode < 200 || response.StatusCode > 299 {
		refusal, _ = io.ReadAll(io.LimitReader(response.Body, mcpRefusalBodyLimit))
	}
	response.Body.Close()
	// The host opens its own session; end the one this check opened. Like the TypeScript `void fetch`, on its
	// own budget and not waited for, so the answer costs one round trip; Drain waits for it at shutdown. Not
	// on ctx: a check cancelled at shutdown must still close the session the proxy opened for it.
	if session := response.Header.Get("mcp-session-id"); session != "" {
		closeHeaders := headers.Clone()
		closeHeaders.Set("mcp-session-id", session)
		catalog.Go(func() {
			closeCtx, cancelClose := context.WithTimeout(context.Background(), budget)
			defer cancelClose()
			closing, err := http.NewRequestWithContext(closeCtx, http.MethodDelete, url, nil)
			if err != nil {
				return
			}
			closing.Header = closeHeaders
			if closed, err := client.Do(closing); err == nil {
				closed.Body.Close()
			}
		})
	}
	// The budget refusal's status is configurable on the proxy, so its text is checked first.
	if isBudgetExceeded(string(refusal)) {
		return mcpOutOfBudget
	}
	if response.StatusCode == http.StatusForbidden {
		return mcpNoServers
	}
	return mcpNotRefused
}

// sync is syncMcpServer.
func (m *mcpSync) sync(registrar mcpRegistrar, definition providerDefinition) {
	config := m.config(definition)
	identity := m.identity(config)
	m.mu.Lock()
	if previous, ok := m.attempts[definition.Name]; (ok && identity == previous.identity) || (!ok && config == nil) {
		m.mu.Unlock()
		return
	}
	// Close the old connection before opening one to a new root, and leave nothing behind when the host
	// refuses the new config.
	m.dropLocked(registrar, definition)
	if config == nil {
		m.mu.Unlock()
		return
	}
	// Claimed before the access check, so the next turn does not repeat it, and a login that starts
	// meanwhile drops the claim before this registration can land.
	attempt := &mcpAttempt{identity: identity}
	m.attempts[definition.Name] = attempt
	m.mu.Unlock()
	refused := mcpNotRefused
	if m.settingsErr == nil {
		refused = m.accessRefused(definition, config.URL, identity)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.attempts[definition.Name] != attempt {
		return
	}
	if refused != mcpNotRefused {
		config.Enabled = new(false)
	}
	err := m.settingsErr
	if err == nil {
		err = registrar.RegisterMcpServer(definition.Name, *config)
	}
	if err != nil {
		// The claim stays, so the refusal is reported once rather than on every turn.
		m.state.notifyMcp(fmt.Sprintf("LiteLLM MCP (%s): %s", jsonString(definition.Name), err.Error()), "warning")
		return
	}
	attempt.registered = true
	if refused == mcpOutOfBudget {
		m.state.notifyMcp(fmt.Sprintf("LiteLLM MCP (%s): out of budget, so its MCP server is off for this session. "+
			"Turn it on in /mcp once the budget resets.", jsonString(definition.Name)), "warning")
	}
}

// earlyCheck is an access check running ahead of the first sync; refused is set before done closes.
type earlyCheck struct {
	identity string
	done     chan struct{}
	refused  mcpRefusal
}

// startChecks starts the access checks while the extension loads, as the TypeScript overlaps them with
// seeding (setupHooks runs before it), so the first session_start does not wait for the proxy's round trip.
// Session shutdown cancels the checks no sync has used, then drains them.
func (m *mcpSync) startChecks() {
	if m.settingsErr != nil {
		return
	}
	ctx, cancel := context.WithCancel(context.Background())
	for _, definition := range m.state.definitions {
		config := m.config(definition)
		if config == nil {
			continue
		}
		check := &earlyCheck{identity: m.identity(config), done: make(chan struct{})}
		m.early[definition.Name] = check
		catalog.Go(func() {
			check.refused = mcpAccessRefused(ctx, definition, config.URL)
			close(check.done)
		})
	}
	m.state.mu.Lock()
	m.state.onShutdown = append(m.state.onShutdown, func() {
		// A later sync asks the proxy itself rather than take a cancelled check for an answer.
		m.mu.Lock()
		clear(m.early)
		m.mu.Unlock()
		cancel()
	})
	m.state.mu.Unlock()
}

// accessRefused answers from the early check when it was for this config, and asks the proxy otherwise.
func (m *mcpSync) accessRefused(definition providerDefinition, url, identity string) mcpRefusal {
	m.mu.Lock()
	check := m.early[definition.Name]
	delete(m.early, definition.Name)
	m.mu.Unlock()
	if check != nil && check.identity == identity {
		<-check.done
		return check.refused
	}
	return mcpAccessRefused(context.Background(), definition, url)
}

// lastRegistrar is the registrar of the latest sync, nil before the first one.
func (m *mcpSync) lastRegistrar() mcpRegistrar {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.last
}

func (m *mcpSync) syncAll(registrar mcpRegistrar) {
	m.mu.Lock()
	m.last = registrar
	m.mu.Unlock()
	for _, definition := range m.state.definitions {
		m.sync(registrar, definition)
	}
}

func setupMCP(e hookRegistrar, state *extensionState) {
	if !isFeatureEnabled(state.settings, "mcp") {
		return
	}
	raw, _ := os.ReadFile(filepath.Join(agentDir(), "settings.json"))
	raw = stripBOM(raw)
	m, err := newMcpSync(state, raw)
	if err != nil {
		reportDiagnostic("LiteLLM MCP: " + err.Error())
		return
	}
	m.startChecks()
	// A login drops the server before it can store a new credential, since the host reads the credential per request.
	// The drop reaches the host through the context of the session that last synced.
	state.mu.Lock()
	state.onLoginStart = append(state.onLoginStart, func(definition providerDefinition) {
		if registrar := m.lastRegistrar(); registrar != nil {
			m.drop(registrar, definition)
		}
	})
	state.mu.Unlock()
	// The factory's registrations are queued and validated while the extension loads, where one the host
	// refuses would fail the whole load. A server registered once the session has started is applied at once,
	// the host's error comes back to be reported, and it connects through mcp_servers_change. Each turn then
	// picks up a login that changed the proxy root, and a logout that removed it.
	sync := func(h hookHost, _ map[string]any) (any, error) {
		m.syncAll(h)
		return nil, nil
	}
	e.OnEvent(sdk.EventSessionStart, sync)
	e.OnEvent(sdk.EventBeforeAgentStart, sync)
}
