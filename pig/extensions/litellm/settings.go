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
	"regexp"
	"slices"
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
	envProviders                = "LITELLM_PROVIDERS"
	envProvidersJSON            = "LITELLM_PROVIDERS_JSON"
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
	// HasOIDC is set when the key is present, so a JSON null is rejected rather than read as absent.
	HasOIDC bool
	// EnvPrefix is the LITELLM_PROVIDER_<TOKEN> prefix whose scoped variables back this definition.
	EnvPrefix string
	// EnableMCP is nil when nothing set it; MCP is skipped only for an explicit false.
	EnableMCP *bool
}

var reservedProviderNames = map[string]bool{"mcp": true, "skills": true, "codemode": true, "tool_search": true, "litellm": true}

var (
	providerNamePattern   = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]{0,63}$`)
	providerEnvVarPattern = regexp.MustCompile(`^LITELLM_PROVIDER_([A-Z0-9_]+?)_(BASE_URL|API_KEY_HELPER|API_KEY|HEADERS|DISPLAY_NAME|NAME|ALLOW_INSECURE_HTTP|USE_GCLOUD_AUTH|ENABLE_OAUTH|ENABLE_MCP|MCP_ENABLED)$`)
)

func isValidProviderName(name string) bool {
	return providerNamePattern.MatchString(name) && !reservedProviderNames[name]
}

// parseCanonicalProviderList splits LITELLM_PROVIDERS on commas and whitespace.
func parseCanonicalProviderList(raw string) []string {
	return strings.FieldsFunc(raw, func(r rune) bool { return r == ',' || unicode.IsSpace(r) })
}

// parseProvidersJson parses LITELLM_PROVIDERS_JSON into its object-valued entries, keeping the key order.
// It returns nil providers when the value is unset, invalid, not an object, or has no object entry.
func parseProvidersJson(raw string) (order []string, providers map[string]map[string]any) {
	trimmed := strings.TrimSpace(raw)
	var parsed map[string]any
	if trimmed == "" || json.Unmarshal([]byte(trimmed), &parsed) != nil || parsed == nil {
		return nil, nil
	}
	providers = map[string]map[string]any{}
	for _, key := range objectKeys(json.RawMessage(trimmed)) {
		if value, ok := parsed[key].(map[string]any); ok && value != nil {
			if _, seen := providers[key]; !seen {
				order = append(order, key)
			}
			providers[key] = value
		}
	}
	if len(providers) == 0 {
		return nil, nil
	}
	return order, providers
}

func parseBooleanSetting(value any) bool {
	switch typed := value.(type) {
	case bool:
		return typed
	case string:
		trimmed := strings.ToLower(strings.TrimSpace(typed))
		return trimmed == "1" || trimmed == "true"
	}
	return false
}

// parsedProviderEnv is ParsedProviderEnv: the scoped variables of one LITELLM_PROVIDER_<TOKEN> prefix. The
// boolean fields are nil until their variable is set.
type parsedProviderEnv struct {
	Token              string
	EnvPrefix          string
	Name               string
	DisplayName        string
	BaseURL            string
	APIKey             string
	APIKeyHelper       string
	Headers            string
	AllowInsecureHTTP  *bool
	UseGcloudTokenAuth *bool
	EnableOAuth        *bool
	EnableMCP          *bool
}

// parseProviderEnvVars scans environ ("NAME=value" entries) for LITELLM_PROVIDER_<TOKEN>_<FIELD> variables.
// tokens lists the tokens in first-seen order, which step 4 of getProviderDefinitions depends on.
func parseProviderEnvVars(environ []string) (tokens []string, byToken map[string]*parsedProviderEnv) {
	byToken = map[string]*parsedProviderEnv{}
	for _, entry := range environ {
		key, val, _ := strings.Cut(entry, "=")
		match := providerEnvVarPattern.FindStringSubmatch(key)
		if match == nil {
			continue
		}
		token, field := match[1], match[2]
		parsed := byToken[token]
		if parsed == nil {
			parsed = &parsedProviderEnv{Token: token, EnvPrefix: "LITELLM_PROVIDER_" + token}
			byToken[token] = parsed
			tokens = append(tokens, token)
		}
		flag := func() *bool { return new(parseBooleanSetting(val)) }
		switch field {
		case "BASE_URL":
			parsed.BaseURL = cleanConfig(val)
		case "API_KEY":
			parsed.APIKey = cleanConfig(val)
		case "API_KEY_HELPER":
			parsed.APIKeyHelper = cleanConfig(val)
		case "HEADERS":
			parsed.Headers = cleanConfig(val)
		case "DISPLAY_NAME":
			parsed.DisplayName = cleanConfig(val)
		case "NAME":
			parsed.Name = cleanConfig(val)
		case "ALLOW_INSECURE_HTTP":
			parsed.AllowInsecureHTTP = flag()
		case "USE_GCLOUD_AUTH":
			parsed.UseGcloudTokenAuth = flag()
		case "ENABLE_OAUTH":
			parsed.EnableOAuth = flag()
		case "ENABLE_MCP", "MCP_ENABLED":
			parsed.EnableMCP = flag()
		}
	}
	return tokens, byToken
}

// usesGcloudToken is the ADC condition: the default provider also needs the LITELLM_GCLOUD_TOKEN_AUTH switch.
func usesGcloudToken(definition providerDefinition) bool {
	if definition.UseDefaultEnv {
		return definition.UseGcloudTokenAuth && isGcloudTokenAuthEnabled()
	}
	return definition.UseGcloudTokenAuth
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

// stripBOM removes a leading UTF-8 byte order mark, as PiG's own settings loader does.
func stripBOM(raw []byte) []byte {
	return bytes.TrimPrefix(raw, []byte("\xef\xbb\xbf"))
}

// stripJSONComments removes // line comments and trailing commas, leaving string literals untouched.
// Copied from PiG v0.4.1 internal/codingagent/model_registry.go (that package is internal): the extension
// must accept exactly the models.json files PiG's own reader accepts.
func stripJSONComments(input string) string {
	var out []byte
	i := 0
	for i < len(input) {
		c := input[i]
		switch {
		case c == '"':
			start := i
			i++
			for i < len(input) {
				ch := input[i]
				if ch == '\\' {
					i += 2
					continue
				}
				i++
				if ch == '"' {
					break
				}
			}
			out = append(out, input[start:min(i, len(input))]...)
		case c == '/' && i+1 < len(input) && input[i+1] == '/':
			for i < len(input) && input[i] != '\n' {
				i++
			}
		case c == ',':
			j := i + 1
			for j < len(input) && (input[j] == ' ' || input[j] == '\t' || input[j] == '\r' || input[j] == '\n') {
				j++
			}
			if j < len(input) && (input[j] == '}' || input[j] == ']') {
				i++
			} else {
				out = append(out, c)
				i++
			}
		default:
			out = append(out, c)
			i++
		}
	}
	return string(out)
}

// readGlobalLiteLLMSettings reads the `litellm` block of <agentDir>/settings.json and merges the providers of
// <agentDir>/models.json over it; nil when neither contributes.
func readGlobalLiteLLMSettings() *litellmSettings {
	return mergeProviderSettings(readSettingsJSONLiteLLM(), readModelsProviderSettings())
}

// readSettingsJSONLiteLLM reads the `litellm` block of <agentDir>/settings.json; nil when absent or invalid.
func readSettingsJSONLiteLLM() *litellmSettings {
	raw, err := os.ReadFile(filepath.Join(agentDir(), "settings.json"))
	if err != nil {
		return nil
	}
	var top map[string]json.RawMessage
	if json.Unmarshal(stripBOM(raw), &top) != nil {
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

// readModelsProviderSettings reads only the providers of <agentDir>/models.json (global agent dir, never a
// project copy): aliases from `litellm.providers`, then the native `providers.litellm` slot, which wins per
// field. The file is JSONC, as PiG reads it. The result holds just `providers`; nil when there is none.
func readModelsProviderSettings() *litellmSettings {
	raw, err := os.ReadFile(filepath.Join(agentDir(), "models.json"))
	if err != nil {
		return nil
	}
	var top map[string]json.RawMessage
	if json.Unmarshal([]byte(stripJSONComments(string(stripBOM(raw)))), &top) != nil {
		return nil
	}
	providers := map[string]any{}
	var order []string
	var litellmBlock map[string]json.RawMessage
	if json.Unmarshal(top[settingsKey], &litellmBlock) == nil {
		var aliases map[string]any
		if json.Unmarshal(litellmBlock["providers"], &aliases) == nil {
			for _, name := range objectKeys(litellmBlock["providers"]) {
				if _, seen := providers[name]; !seen {
					order = append(order, name)
				}
				providers[name] = aliases[name]
			}
		}
	}
	var nativeProviders map[string]any
	if json.Unmarshal(top["providers"], &nativeProviders) == nil {
		if native, ok := nativeProviders[providerName].(map[string]any); ok && native != nil {
			if existing, ok := providers[providerName].(map[string]any); ok && existing != nil {
				merged := make(map[string]any, len(existing)+len(native))
				for key, value := range existing {
					merged[key] = value
				}
				for key, value := range native {
					merged[key] = value
				}
				native = merged
			} else if _, seen := providers[providerName]; !seen {
				order = append(order, providerName)
			}
			providers[providerName] = native
		}
	}
	if len(providers) == 0 {
		return nil
	}
	return &litellmSettings{Values: map[string]any{"providers": providers}, ProviderOrder: order}
}

// readModelLimitOverrides reads Pi `modelOverrides` limits from <agentDir>/models.json (global agent dir, never
// a project copy), keyed by provider id then model id. Only finite numbers > 0 count; anything else is ignored.
func readModelLimitOverrides() map[string]map[string]types.LimitOverrides {
	result := map[string]map[string]types.LimitOverrides{}
	raw, err := os.ReadFile(filepath.Join(agentDir(), "models.json"))
	if err != nil {
		return result
	}
	var top struct {
		Providers map[string]any `json:"providers"`
	}
	if json.Unmarshal([]byte(stripJSONComments(string(stripBOM(raw)))), &top) != nil {
		return result
	}
	limit := func(value any) *float64 {
		if number, ok := value.(float64); ok && number > 0 && !math.IsInf(number, 0) {
			return &number
		}
		return nil
	}
	for providerID, provider := range top.Providers {
		block, _ := provider.(map[string]any)
		overrides, _ := block["modelOverrides"].(map[string]any)
		for modelID, value := range overrides {
			override, _ := value.(map[string]any)
			entry := types.LimitOverrides{ContextWindow: limit(override["contextWindow"]), MaxTokens: limit(override["maxTokens"])}
			if entry.ContextWindow == nil && entry.MaxTokens == nil {
				continue
			}
			if result[providerID] == nil {
				result[providerID] = map[string]types.LimitOverrides{}
			}
			result[providerID][modelID] = entry
		}
	}
	return result
}

// mergeProviderSettings merges a provider block that may live in either file per field, models.json winning.
// Non-provider settings.json keys pass through untouched, which keeps `skills`/`mcp`/`budget` working. The
// provider order is settings.json's, then the names only models.json contributes.
func mergeProviderSettings(settings, models *litellmSettings) *litellmSettings {
	var modelsProviders map[string]any
	if models != nil {
		modelsProviders, _ = models.Values["providers"].(map[string]any)
	}
	if modelsProviders == nil {
		return settings
	}
	merged := &litellmSettings{Values: map[string]any{}}
	var settingsProviders map[string]any
	if settings != nil {
		for key, value := range settings.Values {
			merged.Values[key] = value
		}
		settingsProviders, _ = settings.Values["providers"].(map[string]any)
		if settingsProviders != nil {
			merged.ProviderOrder = slices.Clone(settings.ProviderOrder)
		}
	}
	providers := make(map[string]any, len(settingsProviders)+len(modelsProviders))
	for name, value := range settingsProviders {
		providers[name] = value
	}
	for _, name := range models.ProviderOrder {
		raw := modelsProviders[name]
		base, baseIsObject := settingsProviders[name].(map[string]any)
		override, overrideIsObject := raw.(map[string]any)
		if baseIsObject && base != nil && overrideIsObject && override != nil {
			combined := make(map[string]any, len(base)+len(override))
			for key, value := range base {
				combined[key] = value
			}
			for key, value := range override {
				combined[key] = value
			}
			raw = combined
		}
		if _, seen := settingsProviders[name]; !seen {
			merged.ProviderOrder = append(merged.ProviderOrder, name)
		}
		providers[name] = raw
	}
	merged.Values["providers"] = providers
	return merged
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

// resolveHeadersFrom resolves the definition's headers; a string template reads variables through env. Its own
// headers win, then the scoped <prefix>_HEADERS, then LITELLM_HEADERS for the default provider; the variable's
// value is itself the template.
func resolveHeadersFrom(definition providerDefinition, env envLookup) map[string]string {
	raw := definition.Headers
	if raw == nil && definition.EnvPrefix != "" {
		if value, ok := env(definition.EnvPrefix + "_HEADERS"); ok {
			raw = value
		}
	}
	if raw == nil && definition.UseDefaultEnv {
		if value, ok := env(envHeaders); ok {
			raw = value
		}
	}
	if template, ok := raw.(string); ok {
		resolved, _ := resolveTemplateConfigValue(template, env)
		return parseCustomHeaders(resolved)
	}
	return parseHeaderRecord(raw)
}

func resolveHeaders(definition providerDefinition) map[string]string {
	return resolveHeadersFrom(definition, processEnv)
}

// resolveCredentials resolves the key and base URL of a definition from settings, helpers, environment and
// Google ADC. With executeHelpers false no `!command` and no ADC exchange runs.
func resolveCredentials(ctx context.Context, definition providerDefinition, executeHelpers bool) (types.ResolvedCredentials, error) {
	configuredBase := cleanConfig(definition.BaseURL)
	if configuredBase == "" && definition.EnvPrefix != "" {
		configuredBase = cleanConfig(os.Getenv(definition.EnvPrefix + "_BASE_URL"))
	}
	if configuredBase == "" && definition.UseDefaultEnv {
		configuredBase = cleanConfig(os.Getenv(envBaseURL))
	}
	var gcloudKey string
	if executeHelpers && usesGcloudToken(definition) {
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

	var helperKey, helperCommand, envKey, envKeySource string
	// fromEnv reads the helper, then the key, of one variable family.
	fromEnv := func(helperName, keyName string) error {
		helperCommand = normalizeCommand(os.Getenv(helperName))
		if helperCommand != "" && executeHelpers {
			value, err := executeAPIKeyCommand(ctx, helperCommand)
			if err != nil {
				return err
			}
			helperKey = value
		}
		if helperCommand == "" {
			if envKey = cleanConfig(os.Getenv(keyName)); envKey != "" {
				envKeySource = "$" + keyName
			}
		}
		return nil
	}
	if gcloudKey == "" && configuredKey == "" && definition.EnvPrefix != "" {
		if err := fromEnv(definition.EnvPrefix+"_API_KEY_HELPER", definition.EnvPrefix+"_API_KEY"); err != nil {
			return types.ResolvedCredentials{}, err
		}
	}
	if gcloudKey == "" && configuredKey == "" && helperKey == "" && helperCommand == "" && envKey == "" && definition.UseDefaultEnv {
		if err := fromEnv(envAPIKeyHelper, envAPIKey); err != nil {
			return types.ResolvedCredentials{}, err
		}
	}
	apiKey := firstNonEmpty(gcloudKey, configuredKey, helperKey, envKey)

	var apiKeyConfig string
	switch {
	case configuredKey != "" && definition.APIKeyConfig != "":
		apiKeyConfig = definition.APIKeyConfig
	case !executeHelpers && strings.HasPrefix(definition.APIKeyConfig, "!"):
		apiKeyConfig = definition.APIKeyConfig
	case helperKey != "" && helperCommand != "":
		apiKeyConfig = helperCommand
	case !executeHelpers && helperCommand != "":
		apiKeyConfig = helperCommand
	case envKey != "":
		apiKeyConfig = envKeySource
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

// normalizeProviderSettings returns a copy of the provider's raw settings with enableMcp derived from
// enableMcp, else mcp.enabled, else mcp itself; nil when absent, not an object, or disabled. A present JSON
// null counts as present, so it reads as false.
func normalizeProviderSettings(raw any) map[string]any {
	record, ok := raw.(map[string]any)
	if !ok || record == nil || record["enabled"] == false {
		return nil
	}
	normalized := make(map[string]any, len(record)+1)
	for key, value := range record {
		normalized[key] = value
	}
	if value, ok := record["enableMcp"]; ok {
		normalized["enableMcp"] = parseBooleanSetting(value)
	} else if mcp, ok := record["mcp"].(map[string]any); ok && mcp != nil {
		if value, ok := mcp["enabled"]; ok {
			normalized["enableMcp"] = parseBooleanSetting(value)
		}
	} else if value, ok := record["mcp"]; ok {
		normalized["enableMcp"] = parseBooleanSetting(value)
	}
	return normalized
}

// isFeatureEnabled reports whether the "skills", "mcp" or "budget" block is not disabled.
func isFeatureEnabled(settings *litellmSettings, feature string) bool {
	if settings == nil {
		return true
	}
	block, ok := settings.Values[feature].(map[string]any)
	return !ok || block["enabled"] != false
}

// intermediateProvider is IntermediateProvider; the boolean fields are nil until something sets them.
type intermediateProvider struct {
	Name               string
	DisplayName        string
	BaseURL            string
	APIKeyConfig       string
	Headers            any
	AllowInsecureHTTP  *bool
	UseGcloudTokenAuth *bool
	EnableOAuth        *bool
	EnvPrefix          string
	EnableMCP          *bool
}

// secondaryProviders is a Map: it keeps insertion order, and a re-set keeps the original position.
type secondaryProviders struct {
	order   []string
	entries map[string]*intermediateProvider
}

func (p *secondaryProviders) set(name string, entry *intermediateProvider) {
	if _, ok := p.entries[name]; !ok {
		p.order = append(p.order, name)
	}
	p.entries[name] = entry
}

func (p *secondaryProviders) delete(name string) {
	if _, ok := p.entries[name]; !ok {
		return
	}
	delete(p.entries, name)
	p.order = slices.DeleteFunc(p.order, func(item string) bool { return item == name })
}

// firstBool is `a ?? b` for tri-state booleans.
func firstBool(a, b *bool) *bool {
	if a != nil {
		return a
	}
	return b
}

// settingBool is a settings value that is a JSON boolean, else nil.
func settingBool(value any) *bool {
	if typed, ok := value.(bool); ok {
		return &typed
	}
	return nil
}

// tokenForProvider is the environment token a provider id maps to.
func tokenForProvider(id string) string {
	return strings.ReplaceAll(strings.ToUpper(id), "-", "_")
}

// getProviderDefinitions builds the default `litellm` definition first, then the aliases: those from
// LITELLM_PROVIDERS_JSON, LITELLM_PROVIDERS, the LITELLM_PROVIDER_<TOKEN>_* variables, and settings.json, in
// insertion order. Settings override the environment, and an alias disabled in either place is dropped.
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
	defaultSettings := normalizeProviderSettings(providers[providerName])
	defaultDisplayName := firstNonEmpty(cleanConfig(os.Getenv(envDisplayName)), "LiteLLM")
	tokens, parsedEnv := parseProviderEnvVars(os.Environ())
	primaryToken := strings.ToUpper(providerName)

	primaryHeaders := defaultSettings["headers"]
	if primaryHeaders == nil {
		primaryHeaders = "$" + envHeaders
	}
	primary := providerDefinition{
		Name:               providerName,
		DisplayName:        firstNonEmpty(stringSetting(defaultSettings["displayName"]), defaultDisplayName),
		BaseURL:            stringSetting(defaultSettings["baseUrl"]),
		APIKeyConfig:       stringSetting(defaultSettings["apiKey"]),
		Headers:            primaryHeaders,
		UseDefaultEnv:      true,
		UseGcloudTokenAuth: true,
		EnableOAuth:        true,
		AllowInsecureHTTP:  defaultSettings["allowInsecureHttp"] == true,
		OIDC:               defaultSettings["oidc"],
		EnableMCP:          settingBool(defaultSettings["enableMcp"]),
	}
	_, primary.HasOIDC = defaultSettings["oidc"]
	if primary.EnableMCP == nil && parsedEnv[primaryToken] != nil {
		primary.EnableMCP = parsedEnv[primaryToken].EnableMCP
	}

	secondary := &secondaryProviders{entries: map[string]*intermediateProvider{}}
	disabled := map[string]bool{}

	// 1. Incorporate LITELLM_PROVIDERS_JSON
	jsonOrder, jsonProviders := parseProvidersJson(os.Getenv(envProvidersJSON))
	for _, name := range jsonOrder {
		if !isValidProviderName(name) {
			continue
		}
		raw := jsonProviders[name]
		if raw["enabled"] == false {
			disabled[name] = true
			continue
		}
		normalized := normalizeProviderSettings(raw)
		if normalized == nil {
			continue
		}
		secondary.set(name, &intermediateProvider{
			Name:               name,
			DisplayName:        stringSetting(normalized["displayName"]),
			BaseURL:            stringSetting(normalized["baseUrl"]),
			APIKeyConfig:       stringSetting(normalized["apiKey"]),
			Headers:            normalized["headers"],
			AllowInsecureHTTP:  new(parseBooleanSetting(normalized["allowInsecureHttp"])),
			UseGcloudTokenAuth: new(parseBooleanSetting(normalized["useGcloudTokenAuth"])),
			EnableOAuth:        new(parseBooleanSetting(normalized["enableOAuth"])),
			EnableMCP:          settingBool(normalized["enableMcp"]),
		})
	}

	// 2. The scanned tokens; the default provider's is never an alias
	claimedTokens := map[string]bool{primaryToken: true}

	// 3. Declared ids claim their env token before step 4 derives new ids: LITELLM_PROVIDERS first, then JSON keys,
	// then settings.json aliases that have scoped variables (step 5 still applies their settings on top)
	tokenOwners := map[string]string{primaryToken: providerName}
	candidates := append(parseCanonicalProviderList(os.Getenv(envProviders)), secondary.order...)
	for _, name := range order {
		if normalizeProviderSettings(providers[name]) != nil && parsedEnv[tokenForProvider(name)] != nil {
			candidates = append(candidates, name)
		}
	}
	var declared []string
	seen := map[string]bool{}
	for _, id := range candidates {
		if !seen[id] {
			seen[id] = true
			declared = append(declared, id)
		}
	}
	for _, canonicalID := range declared {
		if !isValidProviderName(canonicalID) || disabled[canonicalID] {
			continue
		}
		token := tokenForProvider(canonicalID)
		owner, owned := tokenOwners[token]
		hasCollision := owned && owner != canonicalID
		if !hasCollision && !owned {
			tokenOwners[token] = canonicalID
		}
		var envData *parsedProviderEnv
		if !hasCollision {
			envData = parsedEnv[token]
		}
		if envData != nil {
			claimedTokens[token] = true
		}
		secondary.set(canonicalID, mergeEnvProvider(canonicalID, envData, secondary.entries[canonicalID]))
	}

	// 4. Incorporate remaining unlisted prefix-scanned tokens
	for _, token := range tokens {
		envData := parsedEnv[token]
		if claimedTokens[token] {
			continue
		}
		providerID := envData.Name
		if providerID == "" {
			providerID = strings.ReplaceAll(strings.ToLower(token), "_", "-")
		}
		if !isValidProviderName(providerID) || disabled[providerID] || disabled[strings.ToLower(token)] {
			continue
		}
		tokenOwners[token] = providerID
		secondary.set(providerID, mergeEnvProvider(providerID, envData, secondary.entries[providerID]))
	}

	// 5. Merge with settings.json (disk settings override env vars; enabled === false excludes)
	for _, name := range order {
		if name == providerName || name == "" || reservedProviderNames[name] {
			continue
		}
		raw := normalizeProviderSettings(providers[name])
		if raw == nil {
			secondary.delete(name)
			continue
		}
		existing := secondary.entries[name]
		token := tokenForProvider(name)
		var matchingEnvPrefix string
		if existing != nil {
			matchingEnvPrefix = existing.EnvPrefix
		} else {
			owner, owned := tokenOwners[token]
			if (!owned || owner == name) && !claimedTokens[token] && parsedEnv[token] != nil {
				matchingEnvPrefix = "LITELLM_PROVIDER_" + token
				tokenOwners[token] = name
				claimedTokens[token] = true
			}
		}
		if existing == nil {
			existing = &intermediateProvider{}
		}
		secondary.set(name, &intermediateProvider{
			Name:               name,
			DisplayName:        firstNonEmpty(stringSetting(raw["displayName"]), existing.DisplayName),
			BaseURL:            firstNonEmpty(stringSetting(raw["baseUrl"]), existing.BaseURL),
			APIKeyConfig:       firstNonEmpty(stringSetting(raw["apiKey"]), existing.APIKeyConfig),
			Headers:            firstAny(raw["headers"], existing.Headers),
			AllowInsecureHTTP:  firstBool(settingBool(raw["allowInsecureHttp"]), existing.AllowInsecureHTTP),
			UseGcloudTokenAuth: firstBool(settingBool(raw["useGcloudTokenAuth"]), existing.UseGcloudTokenAuth),
			EnableOAuth:        firstBool(settingBool(raw["enableOAuth"]), existing.EnableOAuth),
			EnvPrefix:          matchingEnvPrefix,
			EnableMCP:          firstBool(settingBool(raw["enableMcp"]), existing.EnableMCP),
		})
	}

	definitions := []providerDefinition{primary}
	for _, name := range secondary.order {
		entry := secondary.entries[name]
		definitions = append(definitions, providerDefinition{
			Name:               name,
			DisplayName:        firstNonEmpty(entry.DisplayName, name),
			BaseURL:            entry.BaseURL,
			APIKeyConfig:       entry.APIKeyConfig,
			Headers:            entry.Headers,
			UseGcloudTokenAuth: entry.UseGcloudTokenAuth != nil && *entry.UseGcloudTokenAuth,
			EnableOAuth:        entry.EnableOAuth != nil && *entry.EnableOAuth,
			AllowInsecureHTTP:  entry.AllowInsecureHTTP != nil && *entry.AllowInsecureHTTP,
			EnvPrefix:          entry.EnvPrefix,
			EnableMCP:          entry.EnableMCP,
		})
	}
	return definitions
}

// mergeEnvProvider layers a token's scanned variables over an earlier definition of the same id (steps 3 and 4).
// envData and existing may be nil.
func mergeEnvProvider(id string, envData *parsedProviderEnv, existing *intermediateProvider) *intermediateProvider {
	if envData == nil {
		envData = &parsedProviderEnv{}
	}
	if existing == nil {
		existing = &intermediateProvider{}
	}
	return &intermediateProvider{
		Name:               id,
		DisplayName:        firstNonEmpty(envData.DisplayName, existing.DisplayName),
		BaseURL:            existing.BaseURL,
		APIKeyConfig:       existing.APIKeyConfig,
		Headers:            existing.Headers,
		AllowInsecureHTTP:  firstBool(envData.AllowInsecureHTTP, existing.AllowInsecureHTTP),
		UseGcloudTokenAuth: firstBool(envData.UseGcloudTokenAuth, existing.UseGcloudTokenAuth),
		EnableOAuth:        firstBool(envData.EnableOAuth, existing.EnableOAuth),
		EnvPrefix:          firstNonEmpty(envData.EnvPrefix, existing.EnvPrefix),
		EnableMCP:          firstBool(envData.EnableMCP, existing.EnableMCP),
	}
}

// firstAny is `a ?? b`: a JSON null counts as absent.
func firstAny(a, b any) any {
	if a != nil {
		return a
	}
	return b
}

// storedCredential is an auth.json entry. types.AuthFileEntry has neither the api_key `env` nor the direct OIDC
// login's identity provider, which the next login to the same proxy offers again, so they are added here.
type storedCredential struct {
	types.AuthFileEntry
	Env          map[string]string `json:"env,omitempty"`
	LegacyAPIKey string            `json:"apiKey,omitempty"`
	Flow         string            `json:"flow,omitempty"`
	Issuer       string            `json:"issuer,omitempty"`
	ClientID     string            `json:"clientId,omitempty"`
	Scope        string            `json:"scope,omitempty"`
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
	var scopedURL, envURL, previous string
	if definition.EnvPrefix != "" {
		scopedURL = cleanConfig(os.Getenv(definition.EnvPrefix + "_BASE_URL"))
	}
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
		{scopedURL, "$" + definition.EnvPrefix + "_BASE_URL"},
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
