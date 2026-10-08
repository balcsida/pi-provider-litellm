package litellm

// Ports src/index.ts (constants, settings, provider definitions, config values, helper commands, headers,
// credential resolution and environment helpers).

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"

	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/protocols"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

const (
	providerName                = "litellm"
	settingsKey                 = "litellm"
	envBaseURL                  = "LITELLM_BASE_URL"
	envAPIKey                   = "LITELLM_API_KEY"
	envDisplayName              = "LITELLM_DISPLAY_NAME"
	envAPIKeyHelper             = "LITELLM_API_KEY_HELPER"
	envHeaders                  = "LITELLM_HEADERS"
	envTimeout                  = "LITELLM_DISCOVERY_TIMEOUT_MS"
	envCLIJWTExpirationHours    = "LITELLM_CLI_JWT_EXPIRATION_HOURS"
	envOffline                  = "LITELLM_OFFLINE"
	envVerboseDiscovery         = "LITELLM_VERBOSE_DISCOVERY"
	envModelsDev                = "LITELLM_MODELS_DEV"
	envPIOffline                = "PI_OFFLINE"
	envPIGOffline               = "PIG_OFFLINE"
	modelsDevCacheFilename      = "litellm-models-dev.json"
	defaultTimeoutMs            = 5000
	seedTimeoutMs               = 3000
	defaultCLIJWTExpirationHour = 24
	tokenRefreshLeadMs          = 5 * 60 * 1000
	permanentTokenExpiresAt     = int64(1<<53 - 1)
	apiKeyHelperTimeout         = 10 * time.Second
	gcloudADCSource             = "gcloud ADC"
)

// now is replaced by tests that pin the clock.
var now = time.Now

// reportDiagnostic receives the one-line diagnostics the TypeScript extension writes to stderr. The
// message carries no trailing newline. The extension may redirect it; the default writes to os.Stderr.
var reportDiagnostic = func(message string) {
	_, _ = os.Stderr.WriteString(message + "\n")
}

// providerDefinition is ProviderDefinition. Headers is nil, a string template, or a decoded JSON object.
type providerDefinition struct {
	Name               string
	DisplayName        string
	BaseURL            string
	APIKeyConfig       string
	Headers            any
	UseDefaultEnv      bool
	UseGcloudTokenAuth bool
	EnableOAuth        bool
	AllowInsecureHTTP  bool
	// OIDC is the raw `oidc` setting, validated at login so a bad value never breaks startup.
	OIDC any
}

// litellmSettings is the global `litellm` settings block. Go maps lose key order, so the provider
// order of settings.json is kept beside the decoded values.
type litellmSettings struct {
	Values        map[string]any
	ProviderOrder []string
}

// agentDir is getAgentDir for PiG; it resolves like PiG's own extension host.
func agentDir() string {
	home, _ := os.UserHomeDir()
	expand := func(path string) string {
		if path == "~" {
			return home
		}
		if rest, ok := strings.CutPrefix(path, "~/"); ok {
			return filepath.Join(home, rest)
		}
		return path
	}
	if os.Getenv("PIG_USE_PI_DIRS") == "1" {
		if dir := os.Getenv("PI_CODING_AGENT_DIR"); dir != "" {
			return expand(dir)
		}
		return filepath.Join(home, ".pi", "agent")
	}
	if dir := os.Getenv("PIG_CODING_AGENT_DIR"); dir != "" {
		return expand(dir)
	}
	if root := os.Getenv("PIG_HOME"); root != "" {
		return filepath.Join(expand(root), "agent")
	}
	if root := os.Getenv("XDG_CONFIG_HOME"); root != "" {
		return filepath.Join(expand(root), "pig", "agent")
	}
	return filepath.Join(home, ".pig", "agent")
}

// readGlobalLiteLLMSettings reads the `litellm` block of <agentDir>/settings.json; nil when absent or invalid.
func readGlobalLiteLLMSettings() *litellmSettings {
	raw, err := os.ReadFile(filepath.Join(agentDir(), "settings.json"))
	if err != nil {
		return nil
	}
	var top map[string]json.RawMessage
	if json.Unmarshal(raw, &top) != nil {
		return nil
	}
	block, ok := top[settingsKey]
	if !ok {
		return nil
	}
	var values map[string]any
	if json.Unmarshal(block, &values) != nil || values == nil {
		return nil
	}
	settings := &litellmSettings{Values: values}
	var fields map[string]json.RawMessage
	if json.Unmarshal(block, &fields) == nil {
		settings.ProviderOrder = objectKeys(fields["providers"])
	}
	return settings
}

// objectKeys returns the keys of a JSON object in document order.
func objectKeys(raw json.RawMessage) []string {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	if token, err := decoder.Token(); err != nil || token != json.Delim('{') {
		return nil
	}
	var keys []string
	for decoder.More() {
		token, err := decoder.Token()
		key, isString := token.(string)
		if err != nil || !isString {
			return keys
		}
		keys = append(keys, key)
		var skipped json.RawMessage
		if decoder.Decode(&skipped) != nil {
			return keys
		}
	}
	return keys
}

// cleanConfig trims a configured value; the literal "undefined" counts as unset.
func cleanConfig(raw string) string {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "undefined" {
		return ""
	}
	return trimmed
}

func stringSetting(value any) string {
	if text, ok := value.(string); ok {
		return cleanConfig(text)
	}
	return ""
}

func normalizeCommand(raw string) string {
	trimmed := cleanConfig(raw)
	if trimmed == "" {
		return ""
	}
	if strings.HasPrefix(trimmed, "!") {
		return trimmed
	}
	return "!" + trimmed
}

func apiKeyHelperCommand() string {
	return normalizeCommand(os.Getenv(envAPIKeyHelper))
}

// parseAPIKeyCommand splits a helper command into argv. Shell syntax is rejected; nothing runs through a shell.
func parseAPIKeyCommand(commandConfig string) ([]string, error) {
	command := strings.TrimPrefix(commandConfig, "!")
	if strings.ContainsAny(command, "\n\r;&|<>`$(){}*?~%^![]") {
		return nil, errors.New("LiteLLM API key helper shell syntax is not supported")
	}
	runes := []rune(command)
	var parts []string
	var current []rune
	var quote rune
	for index := 0; index < len(runes); index++ {
		character := runes[index]
		var next rune
		if index+1 < len(runes) {
			next = runes[index+1]
		}
		switch {
		case character == '\\' && quote != '\'' && next != 0 && (next == '\\' || next == quote || unicode.IsSpace(next)):
			current = append(current, next)
			index++
		case (character == '\'' || character == '"') && (quote == 0 || quote == character):
			if quote == 0 {
				quote = character
			} else {
				quote = 0
			}
		case quote == 0 && unicode.IsSpace(character):
			if len(current) > 0 {
				parts = append(parts, string(current))
				current = nil
			}
		default:
			current = append(current, character)
		}
	}
	if quote != 0 {
		return nil, fmt.Errorf("LiteLLM API key helper command has an unterminated quote: %s", command)
	}
	if len(current) > 0 {
		parts = append(parts, string(current))
	}
	if len(parts) == 0 {
		return nil, errors.New("LiteLLM API key helper command is empty")
	}
	if lower := strings.ToLower(parts[0]); strings.HasSuffix(lower, ".cmd") || strings.HasSuffix(lower, ".bat") {
		return nil, errors.New("LiteLLM API key helper shell scripts are not supported; use an executable")
	}
	return parts, nil
}

// executeAPIKeyCommand runs the helper without a shell and returns its trimmed stdout. Stderr is never echoed.
func executeAPIKeyCommand(ctx context.Context, commandConfig string) (string, error) {
	argv, err := parseAPIKeyCommand(commandConfig)
	if err != nil {
		return "", err
	}
	ctx, cancel := context.WithTimeout(ctx, apiKeyHelperTimeout)
	defer cancel()
	var stdout bytes.Buffer
	command := exec.CommandContext(ctx, argv[0], argv[1:]...)
	command.Stdout = &stdout
	if err := command.Run(); err != nil {
		return "", fmt.Errorf("LiteLLM API key helper failed: %s", argv[0])
	}
	output := strings.TrimSpace(stdout.String())
	if output == "" {
		return "", fmt.Errorf("LiteLLM API key helper produced no output: %s", argv[0])
	}
	return output, nil
}

// tokenExpiresAt is the refresh deadline in epoch milliseconds for a JWT, or opaqueFallback for any other key.
func tokenExpiresAt(apiKey string, opaqueFallback int64) int64 {
	parts := strings.Split(apiKey, ".")
	if len(parts) < 2 || parts[1] == "" {
		return opaqueFallback
	}
	payload, err := base64.RawURLEncoding.DecodeString(strings.TrimRight(parts[1], "="))
	if err != nil {
		return opaqueFallback
	}
	var claims struct {
		Exp *float64 `json:"exp"`
	}
	if json.Unmarshal(payload, &claims) != nil || claims.Exp == nil {
		return opaqueFallback
	}
	return max(now().UnixMilli(), int64(*claims.Exp*1000)-tokenRefreshLeadMs)
}

// configuredCLIJWTExpiresAt is the epoch-millisecond expiry of a CLI SSO token with no server lifetime.
func configuredCLIJWTExpiresAt() int64 {
	hours, err := strconv.ParseFloat(strings.TrimSpace(os.Getenv(envCLIJWTExpirationHours)), 64)
	if err != nil || math.IsNaN(hours) || math.IsInf(hours, 0) || hours <= 0 {
		hours = defaultCLIJWTExpirationHour
	}
	return now().UnixMilli() + int64(hours*60*60*1000)
}

// envLookup reads one environment value; ok is false when it is unset.
type envLookup func(name string) (string, bool)

func processEnv(name string) (string, bool) { return os.LookupEnv(name) }

func isEnvNameStart(c byte) bool {
	return c == '_' || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
}

// resolveTemplateConfigValue expands $NAME and ${NAME} ($$ is a literal $, $! a literal !). ok is false
// when a referenced variable is unset.
func resolveTemplateConfigValue(config string, env envLookup) (string, bool) {
	var resolved strings.Builder
	for index := 0; index < len(config); {
		dollar := strings.IndexByte(config[index:], '$')
		if dollar == -1 {
			resolved.WriteString(config[index:])
			return resolved.String(), true
		}
		dollar += index
		resolved.WriteString(config[index:dollar])
		var next byte
		if dollar+1 < len(config) {
			next = config[dollar+1]
		}
		switch {
		case next == '$' || next == '!':
			resolved.WriteByte(next)
			index = dollar + 2
		case next == '{':
			end := strings.IndexByte(config[dollar+2:], '}')
			if end == -1 {
				resolved.WriteByte('$')
				index = dollar + 1
				continue
			}
			value, ok := env(config[dollar+2 : dollar+2+end])
			if !ok {
				return "", false
			}
			resolved.WriteString(value)
			index = dollar + 2 + end + 1
		case next != 0 && isEnvNameStart(next):
			end := dollar + 1
			for end < len(config) && (isEnvNameStart(config[end]) || (config[end] >= '0' && config[end] <= '9')) {
				end++
			}
			value, ok := env(config[dollar+1 : end])
			if !ok {
				return "", false
			}
			resolved.WriteString(value)
			index = end
		default:
			resolved.WriteByte('$')
			index = dollar + 1
		}
	}
	return resolved.String(), true
}

// resolveConfigValue resolves a `!command` (only when executeCommands) or a template against the process environment.
func resolveConfigValue(ctx context.Context, config string, executeCommands bool) (string, bool, error) {
	if strings.HasPrefix(config, "!") {
		if !executeCommands {
			return "", false, nil
		}
		value, err := executeAPIKeyCommand(ctx, config)
		return value, err == nil, err
	}
	value, ok := resolveTemplateConfigValue(config, processEnv)
	return value, ok, nil
}

var (
	warnedUnresolvedAPIKeysMu sync.Mutex
	warnedUnresolvedAPIKeys   = map[string]bool{}
)

func resetWarnedUnresolvedAPIKeys() {
	warnedUnresolvedAPIKeysMu.Lock()
	warnedUnresolvedAPIKeys = map[string]bool{}
	warnedUnresolvedAPIKeysMu.Unlock()
}

func warnUnresolvedAPIKeyConfig(name, config string) {
	key := name + " " + config
	warnedUnresolvedAPIKeysMu.Lock()
	seen := warnedUnresolvedAPIKeys[key]
	warnedUnresolvedAPIKeys[key] = true
	warnedUnresolvedAPIKeysMu.Unlock()
	if seen {
		return
	}
	reportDiagnostic(fmt.Sprintf("LiteLLM (%s): configured apiKey did not resolve (unset environment variable?); use $$ for a literal $.", name))
}

// parseHeaderRecord resolves a decoded JSON header object; nil when no header survives.
func parseHeaderRecord(value any) map[string]string {
	record, ok := value.(map[string]any)
	if !ok || record == nil {
		return nil
	}
	headers := map[string]string{}
	for key, raw := range record {
		if strings.TrimSpace(key) == "" {
			continue
		}
		var resolved string
		switch typed := raw.(type) {
		case string:
			resolved, _ = resolveTemplateConfigValue(typed, processEnv)
		case float64:
			resolved = strconv.FormatFloat(typed, 'f', -1, 64)
		case bool:
			resolved = strconv.FormatBool(typed)
		default:
			reportDiagnostic(fmt.Sprintf("LiteLLM: ignoring non-primitive header value for %q.", key))
			continue
		}
		if resolved != "" {
			headers[key] = resolved
		}
	}
	if len(headers) == 0 {
		return nil
	}
	return headers
}

func parseCustomHeaders(raw string) map[string]string {
	trimmed := cleanConfig(raw)
	if trimmed == "" {
		return nil
	}
	var parsed any
	if err := json.Unmarshal([]byte(trimmed), &parsed); err != nil {
		reportDiagnostic(fmt.Sprintf("LiteLLM: failed to parse custom headers (%s).", err.Error()))
		return nil
	}
	return parseHeaderRecord(parsed)
}

// resolveHeadersFrom resolves the definition's headers; a string template reads variables through env.
func resolveHeadersFrom(definition providerDefinition, env envLookup) map[string]string {
	if template, ok := definition.Headers.(string); ok {
		resolved, _ := resolveTemplateConfigValue(template, env)
		return parseCustomHeaders(resolved)
	}
	return parseHeaderRecord(definition.Headers)
}

func resolveHeaders(definition providerDefinition) map[string]string {
	return resolveHeadersFrom(definition, processEnv)
}

// resolveCredentials resolves the key and base URL of a definition from settings, helpers, environment and
// Google ADC. With executeHelpers false no `!command` and no ADC exchange runs.
func resolveCredentials(ctx context.Context, definition providerDefinition, executeHelpers bool) (types.ResolvedCredentials, error) {
	configuredBase := cleanConfig(definition.BaseURL)
	if configuredBase == "" && definition.UseDefaultEnv {
		configuredBase = cleanConfig(os.Getenv(envBaseURL))
	}
	var envKey, envHelperCommand string
	if definition.UseDefaultEnv {
		envKey = cleanConfig(os.Getenv(envAPIKey))
		envHelperCommand = apiKeyHelperCommand()
	}
	var gcloudKey string
	if executeHelpers && definition.UseGcloudTokenAuth && isGcloudTokenAuthEnabled() {
		gcloudKey = strings.TrimSpace(getGcloudToken(ctx))
	}
	// Resolved lazily so a `!command` key is not executed when a higher-precedence credential already won.
	var configuredKey string
	if gcloudKey == "" && definition.APIKeyConfig != "" {
		value, ok, err := resolveConfigValue(ctx, definition.APIKeyConfig, executeHelpers)
		if err != nil {
			return types.ResolvedCredentials{}, err
		}
		if ok {
			configuredKey = value
		} else if !strings.HasPrefix(definition.APIKeyConfig, "!") {
			warnUnresolvedAPIKeyConfig(definition.Name, definition.APIKeyConfig)
		}
	}
	var helperKey string
	if gcloudKey == "" && configuredKey == "" && executeHelpers && envHelperCommand != "" {
		value, err := executeAPIKeyCommand(ctx, envHelperCommand)
		if err != nil {
			return types.ResolvedCredentials{}, err
		}
		helperKey = value
	}
	apiKey := firstNonEmpty(gcloudKey, configuredKey, helperKey, envKey)

	var apiKeyConfig string
	switch {
	case configuredKey != "" && definition.APIKeyConfig != "":
		apiKeyConfig = definition.APIKeyConfig
	case !executeHelpers && strings.HasPrefix(definition.APIKeyConfig, "!"):
		apiKeyConfig = definition.APIKeyConfig
	case helperKey != "" && envHelperCommand != "":
		apiKeyConfig = envHelperCommand
	case !executeHelpers && envHelperCommand != "":
		apiKeyConfig = envHelperCommand
	case envKey != "":
		apiKeyConfig = "$" + envAPIKey
	}
	resolved := types.ResolvedCredentials{APIKey: apiKey, APIKeyConfig: apiKeyConfig, APIKeyFromGcloudADC: gcloudKey != ""}
	if configuredBase != "" {
		root, err := protocols.NormalizeBaseURL(configuredBase, definition.AllowInsecureHTTP)
		if err != nil {
			return types.ResolvedCredentials{}, err
		}
		resolved.BaseURL = root
	}
	return resolved, nil
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}

// discoveryTimeoutMs is LITELLM_DISCOVERY_TIMEOUT_MS: the leading integer, or the default when unset, invalid or negative.
func discoveryTimeoutMs() int {
	raw, ok := os.LookupEnv(envTimeout)
	if !ok {
		return defaultTimeoutMs
	}
	text := strings.TrimLeft(raw, " \t\r\n")
	end := 0
	if end < len(text) && (text[end] == '+' || text[end] == '-') {
		end++
	}
	for end < len(text) && text[end] >= '0' && text[end] <= '9' {
		end++
	}
	parsed, err := strconv.Atoi(text[:end])
	if err != nil || parsed < 0 {
		return defaultTimeoutMs
	}
	return parsed
}

// seedTimeoutMsBudget is the absolute budget for activation-time discovery (getSeedTimeoutMs).
func seedTimeoutMsBudget() int {
	return min(discoveryTimeoutMs(), seedTimeoutMs)
}

func isOffline() bool {
	return os.Getenv(envOffline) == "1"
}

// isHostOffline: the host disables all model network access when PI_OFFLINE (or PiG's PIG_OFFLINE) is set.
func isHostOffline() bool {
	_, pi := os.LookupEnv(envPIOffline)
	_, pig := os.LookupEnv(envPIGOffline)
	return pi || pig
}

func isVerboseDiscovery() bool {
	return os.Getenv(envVerboseDiscovery) == "1"
}

// modelsDevDiscoveryOptions: models.dev is an opt-in escape hatch (LITELLM_MODELS_DEV=1).
func modelsDevDiscoveryOptions() (modelsDev bool, cachePath string) {
	if os.Getenv(envModelsDev) != "1" {
		return false, ""
	}
	return !isHostOffline(), filepath.Join(agentDir(), modelsDevCacheFilename)
}

// piLiteral escapes a resolved value for the host's own `$VAR` and leading `!command` resolution.
func piLiteral(value string) string {
	escaped := strings.ReplaceAll(value, "$", "$$")
	if strings.HasPrefix(escaped, "!") {
		return "$" + escaped
	}
	return escaped
}

// normalizeProviderSettings returns the provider's raw settings, or nil when absent, not an object, or disabled.
func normalizeProviderSettings(raw any) map[string]any {
	record, ok := raw.(map[string]any)
	if !ok || record == nil || record["enabled"] == false {
		return nil
	}
	return record
}

// isFeatureEnabled reports whether the "skills", "mcp" or "budget" block is not disabled.
func isFeatureEnabled(settings *litellmSettings, feature string) bool {
	if settings == nil {
		return true
	}
	block, ok := settings.Values[feature].(map[string]any)
	return !ok || block["enabled"] != false
}

// getProviderDefinitions builds the default `litellm` definition first, then each alias in settings order.
func getProviderDefinitions(settings *litellmSettings) []providerDefinition {
	var providers map[string]any
	var order []string
	if settings != nil {
		providers, _ = settings.Values["providers"].(map[string]any)
		order = settings.ProviderOrder
	}
	if len(order) == 0 && len(providers) > 0 {
		for name := range providers {
			order = append(order, name)
		}
	}
	defaultDisplayName := firstNonEmpty(cleanConfig(os.Getenv(envDisplayName)), "LiteLLM")

	makeDefinition := func(name string, raw map[string]any, isDefault bool) providerDefinition {
		displayName := stringSetting(raw["displayName"])
		if displayName == "" {
			displayName = name
			if isDefault {
				displayName = defaultDisplayName
			}
		}
		headers := raw["headers"]
		if headers == nil && isDefault {
			headers = "$" + envHeaders
		}
		definition := providerDefinition{
			Name:               name,
			DisplayName:        displayName,
			BaseURL:            stringSetting(raw["baseUrl"]),
			APIKeyConfig:       stringSetting(raw["apiKey"]),
			Headers:            headers,
			UseDefaultEnv:      isDefault,
			UseGcloudTokenAuth: isDefault,
			EnableOAuth:        isDefault,
			AllowInsecureHTTP:  raw["allowInsecureHttp"] == true,
		}
		if isDefault {
			definition.OIDC = raw["oidc"]
		}
		return definition
	}

	definitions := []providerDefinition{makeDefinition(providerName, normalizeProviderSettings(providers[providerName]), true)}
	for _, name := range order {
		if name == providerName {
			continue
		}
		if normalized := normalizeProviderSettings(providers[name]); normalized != nil {
			definitions = append(definitions, makeDefinition(name, normalized, false))
		}
	}
	return definitions
}

// storedCredential is an auth.json entry. types.AuthFileEntry has no api_key `env`, so it is added here.
type storedCredential struct {
	types.AuthFileEntry
	Env          map[string]string `json:"env,omitempty"`
	LegacyAPIKey string            `json:"apiKey,omitempty"`
}

// readStoredCredential reads one provider's entry from auth.json; nil when missing or unreadable.
func readStoredCredential(provider, path string) *storedCredential {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var entries map[string]json.RawMessage
	if json.Unmarshal(bytes.TrimPrefix(raw, []byte("\xef\xbb\xbf")), &entries) != nil {
		return nil
	}
	entry, ok := entries[provider]
	if !ok {
		return nil
	}
	var credential storedCredential
	if json.Unmarshal(entry, &credential) != nil {
		return nil
	}
	if credential.Type == "api" {
		credential.Type = "api_key"
	}
	if credential.Key == "" {
		credential.Key = credential.LegacyAPIKey
	}
	return &credential
}

// knownBaseURL offers the proxy URL already known for a re-login: settings, environment, or the previous login.
func knownBaseURL(definition providerDefinition) (url, source string, ok bool) {
	stored := readStoredCredential(definition.Name, filepath.Join(agentDir(), "auth.json"))
	var envURL, previous string
	if definition.UseDefaultEnv {
		envURL = cleanConfig(os.Getenv(envBaseURL))
	}
	if stored != nil {
		if stored.Type == "oauth" {
			previous = cleanConfig(stored.BaseURL)
		} else {
			previous = cleanConfig(stored.Env[envBaseURL])
		}
	}
	candidates := [][2]string{
		{cleanConfig(definition.BaseURL), "configured"},
		{envURL, "$" + envBaseURL},
		{previous, "previous login"},
	}
	for _, candidate := range candidates {
		if candidate[0] == "" {
			continue
		}
		// A URL we could not use anyway is not worth offering; try the next candidate.
		if root, err := protocols.NormalizeBaseURL(candidate[0], definition.AllowInsecureHTTP); err == nil {
			return root, candidate[1], true
		}
	}
	return "", "", false
}
