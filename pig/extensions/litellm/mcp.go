package litellm

// Ports src/index.ts: the LiteLLM MCP registration (mcpServerConfig, dropMcpServer, syncMcpServer).
// LiteLLM serves MCP over streamable HTTP at `/mcp`, so each provider registers it as a PiG MCP server and
// the host's client does the rest. `auth.provider` makes the host send the provider's current credential on
// every request, so token refreshes and key helpers apply without registering again; only a new proxy root
// or new headers change the registration.

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"

	sdk "github.com/MichaelKinsy/PiG/extensions/sdk"
)

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
	attempts map[string]mcpAttempt
	// Headers can carry credentials, so a config is remembered as a keyed digest.
	salt []byte
}

func newMcpSync(state *extensionState, settingsRaw []byte) (*mcpSync, error) {
	salt := make([]byte, 32)
	if _, err := rand.Read(salt); err != nil {
		return nil, err
	}
	m := &mcpSync{state: state, attempts: map[string]mcpAttempt{}, salt: salt}
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
	if attempt.registered {
		registrar.UnregisterMcpServer(definition.Name)
	}
}

// sync is syncMcpServer.
func (m *mcpSync) sync(registrar mcpRegistrar, definition providerDefinition) {
	config := m.config(definition)
	identity := m.identity(config)
	m.mu.Lock()
	defer m.mu.Unlock()
	if previous, ok := m.attempts[definition.Name]; (ok && identity == previous.identity) || (!ok && config == nil) {
		return
	}
	// Close the old connection before opening one to a new root, and leave nothing behind when the host
	// refuses the new config.
	m.dropLocked(registrar, definition)
	if config == nil {
		return
	}
	err := m.settingsErr
	if err == nil {
		err = registrar.RegisterMcpServer(definition.Name, *config)
	}
	if err != nil {
		// Remembered so the refusal is reported once rather than on every turn.
		m.attempts[definition.Name] = mcpAttempt{identity: identity}
		m.state.notifyMcp(fmt.Sprintf("LiteLLM MCP (%s): %s", jsonString(definition.Name), err.Error()), "warning")
		return
	}
	m.attempts[definition.Name] = mcpAttempt{identity: identity, registered: true}
}

func (m *mcpSync) syncAll(registrar mcpRegistrar) {
	for _, definition := range m.state.definitions {
		m.sync(registrar, definition)
	}
}

func setupMCP(e *sdk.Extension, state *extensionState) {
	if !isFeatureEnabled(state.settings, "mcp") {
		return
	}
	raw, _ := os.ReadFile(filepath.Join(agentDir(), "settings.json"))
	m, err := newMcpSync(state, raw)
	if err != nil {
		reportDiagnostic("LiteLLM MCP: " + err.Error())
		return
	}
	// A login drops the server before it can store a new credential, since the host reads the credential per request.
	state.mu.Lock()
	state.onLoginStart = append(state.onLoginStart, func(definition providerDefinition) { m.drop(e, definition) })
	state.mu.Unlock()
	// Servers registered while the extension loads connect when the session starts. Each turn picks up a
	// login that changed the proxy root, and a logout that removed it.
	m.syncAll(e)
	e.OnEvent(sdk.EventBeforeAgentStart, func(ctx sdk.Context, _ map[string]any) (any, error) {
		m.syncAll(ctx)
		return nil, nil
	})
}
