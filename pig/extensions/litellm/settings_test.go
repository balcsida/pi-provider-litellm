package litellm

import (
	"context"
	"encoding/base64"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"
)

var hermeticVars = []string{
	envBaseURL, envAPIKey, envAPIKeyHelper, envHeaders, envDisplayName, envTimeout, envOffline,
	envVerboseDiscovery, envModelsDev, envCLIJWTExpirationHours, envGcloudTokenAuth, envGoogleCredentials,
	envPIOffline, envPIGOffline, "PIG_USE_PI_DIRS", "PIG_HOME", "XDG_CONFIG_HOME", "PI_CODING_AGENT_DIR",
}

// hermeticAgentDir clears every variable the extension reads and points the agent dir at a temp dir.
func hermeticAgentDir(t *testing.T) string {
	t.Helper()
	for _, name := range hermeticVars {
		t.Setenv(name, "")
		os.Unsetenv(name)
	}
	dir := t.TempDir()
	t.Setenv("PIG_CODING_AGENT_DIR", dir)
	t.Setenv("HOME", t.TempDir())
	resetGcloudTokenCache()
	return dir
}

func writeFileIn(t *testing.T, dir, name, content string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

// writeHelper writes a /bin/sh helper that prints the next of outputs on each run and counts runs.
func writeHelper(t *testing.T, dir string, outputs []string, fileName string) (path string, runs func() int) {
	t.Helper()
	counter := filepath.Join(dir, "helper-count")
	var cases strings.Builder
	for index, output := range outputs {
		fmt.Fprintf(&cases, "  %d) printf '%%s\\n' '%s' ;;\n", index+1, output)
	}
	script := fmt.Sprintf("#!/bin/sh\nn=$(cat '%[1]s' 2>/dev/null || echo 0)\nn=$((n+1))\necho $n > '%[1]s'\ncase $n in\n%[2]sesac\n", counter, cases.String())
	path = writeFileIn(t, dir, fileName, script)
	if err := os.Chmod(path, 0o700); err != nil {
		t.Fatal(err)
	}
	return path, func() int {
		raw, err := os.ReadFile(counter)
		if err != nil {
			return 0
		}
		count, _ := strconv.Atoi(strings.TrimSpace(string(raw)))
		return count
	}
}

func TestAgentDir(t *testing.T) {
	hermeticAgentDir(t)
	os.Unsetenv("PIG_CODING_AGENT_DIR")
	home := t.TempDir()
	t.Setenv("HOME", home)
	check := func(want string) {
		t.Helper()
		if got := agentDir(); got != want {
			t.Fatalf("agentDir = %q, want %q", got, want)
		}
	}
	check(filepath.Join(home, ".pig", "agent"))
	t.Setenv("XDG_CONFIG_HOME", "/xdg")
	check("/xdg/pig/agent")
	t.Setenv("PIG_HOME", "~/pighome")
	check(filepath.Join(home, "pighome", "agent"))
	t.Setenv("PIG_CODING_AGENT_DIR", "/explicit")
	check("/explicit")
	t.Setenv("PIG_USE_PI_DIRS", "1")
	check(filepath.Join(home, ".pi", "agent"))
	t.Setenv("PI_CODING_AGENT_DIR", "/pi")
	check("/pi")
}

func TestProviderDefinitions(t *testing.T) {
	loadDefinitions := func(t *testing.T, settingsJSON string) []providerDefinition {
		dir := hermeticAgentDir(t)
		if settingsJSON != "" {
			writeFileIn(t, dir, "settings.json", settingsJSON)
		}
		return getProviderDefinitions(readGlobalLiteLLMSettings())
	}

	t.Run("sets the default provider display name from LITELLM_DISPLAY_NAME", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv(envDisplayName, "Acme Gateway")
		if got := getProviderDefinitions(nil)[0].DisplayName; got != "Acme Gateway" {
			t.Fatalf("display name = %q", got)
		}
	})

	t.Run("prefers settings.json displayName over LITELLM_DISPLAY_NAME", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		t.Setenv(envDisplayName, "Env Gateway")
		writeFileIn(t, dir, "settings.json", `{"litellm":{"providers":{"litellm":{"displayName":"Config Gateway"}}}}`)
		if got := getProviderDefinitions(readGlobalLiteLLMSettings())[0].DisplayName; got != "Config Gateway" {
			t.Fatalf("display name = %q", got)
		}
	})

	for _, value := range []string{"", "   ", "undefined"} {
		t.Run(fmt.Sprintf("falls back to LiteLLM when LITELLM_DISPLAY_NAME is %q", value), func(t *testing.T) {
			hermeticAgentDir(t)
			t.Setenv(envDisplayName, value)
			if got := getProviderDefinitions(nil)[0].DisplayName; got != "LiteLLM" {
				t.Fatalf("display name = %q", got)
			}
		})
	}

	t.Run("does not apply LITELLM_DISPLAY_NAME to alias providers, and keeps settings order", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		t.Setenv(envDisplayName, "Custom Gateway")
		writeFileIn(t, dir, "settings.json", `{"litellm":{"providers":{"zeta":{},"litellm-alias":{},"off":{"enabled":false},"custom-named":{"displayName":"Custom Name"}}}}`)
		definitions := getProviderDefinitions(readGlobalLiteLLMSettings())
		var names, display []string
		for _, definition := range definitions {
			names = append(names, definition.Name)
			display = append(display, definition.DisplayName)
		}
		if !reflect.DeepEqual(names, []string{"litellm", "zeta", "litellm-alias", "custom-named"}) ||
			!reflect.DeepEqual(display, []string{"Custom Gateway", "zeta", "litellm-alias", "Custom Name"}) {
			t.Fatalf("names=%v display=%v", names, display)
		}
	})

	t.Run("only the default provider takes env, ADC, OAuth, oidc and the headers variable", func(t *testing.T) {
		definitions := loadDefinitions(t, `{"litellm":{"providers":{"litellm":{"oidc":{"issuer":"x"}},"alias":{"oidc":{"issuer":"y"}}}}}`)
		primary, alias := definitions[0], definitions[1]
		if !primary.UseDefaultEnv || !primary.UseGcloudTokenAuth || !primary.EnableOAuth || primary.OIDC == nil || primary.Headers != "$"+envHeaders {
			t.Fatalf("default = %+v", primary)
		}
		if alias.UseDefaultEnv || alias.UseGcloudTokenAuth || alias.EnableOAuth || alias.OIDC != nil || alias.Headers != nil {
			t.Fatalf("alias = %+v", alias)
		}
	})

	t.Run("reads baseUrl, apiKey and allowInsecureHttp, treating the literal undefined as unset", func(t *testing.T) {
		definitions := loadDefinitions(t, `{"litellm":{"providers":{"litellm":{"baseUrl":" http://host.docker.internal ","apiKey":"undefined","allowInsecureHttp":true}}}}`)
		got := definitions[0]
		if got.BaseURL != "http://host.docker.internal" || got.APIKeyConfig != "" || !got.AllowInsecureHTTP {
			t.Fatalf("definition = %+v", got)
		}
	})

	t.Run("ignores a missing, invalid or non-object settings file", func(t *testing.T) {
		for _, content := range []string{"", "not json", `{"litellm":[]}`, `{"litellm":null}`} {
			if len(loadDefinitions(t, content)) != 1 {
				t.Fatalf("content %q must yield only the default definition", content)
			}
		}
	})
}

func TestIsFeatureEnabled(t *testing.T) {
	dir := hermeticAgentDir(t)
	writeFileIn(t, dir, "settings.json", `{"litellm":{"skills":{"enabled":false},"mcp":{},"budget":true}}`)
	settings := readGlobalLiteLLMSettings()
	for feature, want := range map[string]bool{"skills": false, "mcp": true, "budget": true} {
		if got := isFeatureEnabled(settings, feature); got != want {
			t.Errorf("%s: got %v, want %v", feature, got, want)
		}
	}
	if !isFeatureEnabled(nil, "skills") {
		t.Error("nil settings enable every feature")
	}
}

func TestEnvironmentHelpers(t *testing.T) {
	hermeticAgentDir(t)
	t.Run("discovery timeout", func(t *testing.T) {
		for raw, want := range map[string]int{"": defaultTimeoutMs, "0": 0, "250": 250, "250ms": 250, "-1": defaultTimeoutMs, "abc": defaultTimeoutMs, "15000": 15000} {
			t.Setenv(envTimeout, raw)
			if got := discoveryTimeoutMs(); got != want {
				t.Errorf("%q: got %d, want %d", raw, got, want)
			}
		}
		os.Unsetenv(envTimeout)
		if discoveryTimeoutMs() != defaultTimeoutMs {
			t.Error("unset must use the default")
		}
		t.Setenv(envTimeout, "15000")
		if seedTimeoutMsBudget() != seedTimeoutMs {
			t.Error("seed budget must be capped")
		}
	})
	t.Run("offline and verbose flags", func(t *testing.T) {
		t.Setenv(envOffline, "true")
		if isOffline() {
			t.Error("only LITELLM_OFFLINE=1 is offline")
		}
		t.Setenv(envOffline, "1")
		t.Setenv(envVerboseDiscovery, "1")
		if !isOffline() || !isVerboseDiscovery() {
			t.Error("flags must be on")
		}
	})
	t.Run("host offline honours PI_OFFLINE and PIG_OFFLINE", func(t *testing.T) {
		os.Unsetenv(envPIOffline)
		os.Unsetenv(envPIGOffline)
		if isHostOffline() {
			t.Fatal("unset must be online")
		}
		t.Setenv(envPIGOffline, "1")
		if !isHostOffline() {
			t.Fatal("PIG_OFFLINE must be offline")
		}
	})
	t.Run("models.dev is opt-in", func(t *testing.T) {
		os.Unsetenv(envPIGOffline)
		if on, path := modelsDevDiscoveryOptions(); on || path != "" {
			t.Fatalf("default = %v %q", on, path)
		}
		t.Setenv(envModelsDev, "1")
		on, path := modelsDevDiscoveryOptions()
		if !on || filepath.Base(path) != modelsDevCacheFilename || filepath.Dir(path) != agentDir() {
			t.Fatalf("on=%v path=%q", on, path)
		}
		t.Setenv(envPIOffline, "1")
		if on, _ := modelsDevDiscoveryOptions(); on {
			t.Fatal("host offline must suppress models.dev refresh")
		}
	})
	t.Run("CLI JWT expiry", func(t *testing.T) {
		clock := time.Date(2026, 6, 10, 12, 0, 0, 0, time.UTC)
		previous := now
		now = func() time.Time { return clock }
		t.Cleanup(func() { now = previous })
		for raw, hours := range map[string]float64{"": 24, "abc": 24, "-3": 24, "0": 24, "2": 2, "0.5": 0.5} {
			t.Setenv(envCLIJWTExpirationHours, raw)
			if got, want := configuredCLIJWTExpiresAt(), clock.UnixMilli()+int64(hours*3600*1000); got != want {
				t.Errorf("%q: got %d, want %d", raw, got, want)
			}
		}
	})
}

func TestPiLiteral(t *testing.T) {
	for input, want := range map[string]string{"plain": "plain", "a$b": "a$$b", "!cmd": "$!cmd", "$!x": "$$!x"} {
		if got := piLiteral(input); got != want {
			t.Errorf("piLiteral(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestTokenExpiresAt(t *testing.T) {
	clock := time.Date(2026, 6, 10, 12, 0, 0, 0, time.UTC)
	previous := now
	now = func() time.Time { return clock }
	t.Cleanup(func() { now = previous })
	jwt := func(payload string) string {
		return "h." + base64.RawURLEncoding.EncodeToString([]byte(payload)) + ".s"
	}
	exp := clock.Add(time.Hour).Unix()
	cases := map[string]int64{
		"sk-opaque":                         permanentTokenExpiresAt,
		"a.":                                permanentTokenExpiresAt,
		"a.!!!.c":                           permanentTokenExpiresAt,
		jwt(`{"sub":"x"}`):                  permanentTokenExpiresAt,
		jwt(`{"exp":"soon"}`):               permanentTokenExpiresAt,
		jwt(fmt.Sprintf(`{"exp":%d}`, exp)): exp*1000 - tokenRefreshLeadMs,
		jwt(`{"exp":1}`):                    clock.UnixMilli(),
	}
	for key, want := range cases {
		if got := tokenExpiresAt(key, permanentTokenExpiresAt); got != want {
			t.Errorf("tokenExpiresAt(%q) = %d, want %d", key, got, want)
		}
	}
	if got := tokenExpiresAt("sk-opaque", 0); got != 0 {
		t.Errorf("opaque fallback = %d", got)
	}
}

func TestResolveTemplateConfigValue(t *testing.T) {
	env := func(name string) (string, bool) {
		value, ok := map[string]string{"A": "alpha", "B_2": "beta", "EMPTY": ""}[name]
		return value, ok
	}
	cases := []struct {
		config, want string
		ok           bool
	}{
		{"plain", "plain", true},
		{"$A", "alpha", true},
		{"${A}-${B_2}", "alpha-beta", true},
		{"$A.x", "alpha.x", true},
		{"$$A", "$A", true},
		{"$!cmd", "!cmd", true},
		{"cost $5", "cost $5", true},
		{"trailing $", "trailing $", true},
		{"${unterminated", "${unterminated", true},
		{"$EMPTY", "", true},
		{"$MISSING", "", false},
		{"${MISSING}", "", false},
	}
	for _, tc := range cases {
		got, ok := resolveTemplateConfigValue(tc.config, env)
		if got != tc.want || ok != tc.ok {
			t.Errorf("resolveTemplateConfigValue(%q) = (%q, %v), want (%q, %v)", tc.config, got, ok, tc.want, tc.ok)
		}
	}
}

func TestParseAPIKeyCommand(t *testing.T) {
	good := map[string][]string{
		"!helper arg":                 {"helper", "arg"},
		"helper   a\tb":               {"helper", "a", "b"},
		`!"/path with space/h" 'x y'`: {"/path with space/h", "x y"},
		`/p/token\helper.sh`:          {`/p/token\helper.sh`},
		`h a\ b`:                      {"h", "a b"},
		`h "a\"b"`:                    {"h", `a"b`},
		`h 'a\b'`:                     {"h", `a\b`},
	}
	for input, want := range good {
		got, err := parseAPIKeyCommand(input)
		if err != nil || !reflect.DeepEqual(got, want) {
			t.Errorf("parseAPIKeyCommand(%q) = %q, %v; want %q", input, got, err, want)
		}
	}
	for _, input := range []string{"a; b", "a && b", "a | b", "a > f", "echo `x`", "echo $(x)", "echo ${X}", "a\nb", "ls *", "ls ~", "a [b]", "echo %x%", "a ^ b", "a !b"} {
		if _, err := parseAPIKeyCommand(input); err == nil || !strings.Contains(err.Error(), "shell syntax is not supported") {
			t.Errorf("parseAPIKeyCommand(%q) error = %v", input, err)
		}
	}
	for input, want := range map[string]string{
		`h "open`:   "unterminated quote",
		"!   ":      "empty",
		"run.cmd x": "shell scripts are not supported",
		"RUN.BAT":   "shell scripts are not supported",
	} {
		if _, err := parseAPIKeyCommand(input); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("parseAPIKeyCommand(%q) error = %v, want %q", input, err, want)
		}
	}
}

func TestExecuteAPIKeyCommand(t *testing.T) {
	dir := t.TempDir()
	ctx := context.Background()
	helper, runs := writeHelper(t, dir, []string{"first", "second"}, "helper.sh")

	if got, err := executeAPIKeyCommand(ctx, "!"+helper); err != nil || got != "first" {
		t.Fatalf("got %q, %v", got, err)
	}
	quoted, _ := writeHelper(t, t.TempDir(), []string{"backslash-key"}, `token\helper.sh`)
	if got, err := executeAPIKeyCommand(ctx, `"`+quoted+`"`); err != nil || got != "backslash-key" {
		t.Fatalf("backslash path: got %q, %v", got, err)
	}
	if runs() != 1 {
		t.Fatalf("runs = %d", runs())
	}
	if _, err := executeAPIKeyCommand(ctx, "printf safe-key; printf injected-key"); err == nil || !strings.Contains(err.Error(), "shell syntax is not supported") {
		t.Fatalf("shell syntax error = %v", err)
	}
	if got, err := executeAPIKeyCommand(ctx, "echo -n   padded  "); err != nil || got != "padded" {
		t.Fatalf("echo: got %q, %v", got, err)
	}
	if _, err := executeAPIKeyCommand(ctx, "true"); err == nil || !strings.Contains(err.Error(), "produced no output") {
		t.Fatalf("empty output error = %v", err)
	}
	failing := writeFileIn(t, dir, "fail.sh", "#!/bin/sh\necho secret-on-stderr >&2\nexit 3\n")
	os.Chmod(failing, 0o700)
	if _, err := executeAPIKeyCommand(ctx, failing); err == nil || strings.Contains(err.Error(), "secret-on-stderr") {
		t.Fatalf("failure must not echo stderr: %v", err)
	}
}

func TestParseHeaders(t *testing.T) {
	hermeticAgentDir(t)
	t.Setenv("TENANT", "acme")

	t.Run("drops non-primitive header values instead of stringifying them", func(t *testing.T) {
		diagnostics := captureDiagnostics(t)
		dir := hermeticAgentDir(t)
		writeFileIn(t, dir, "settings.json", `{"litellm":{"providers":{"alias":{"headers":{"x-obj":{"team":"a"},"x-null":null,"x-num":30,"x-bool":false,"x-env":"$TENANT","x-gone":"$UNSET_HEADER_VAR"," ":"blank"}}}}}`)
		t.Setenv("TENANT", "acme")
		alias := getProviderDefinitions(readGlobalLiteLLMSettings())[1]
		want := map[string]string{"x-num": "30", "x-bool": "false", "x-env": "acme"}
		if got := resolveHeaders(alias); !reflect.DeepEqual(got, want) {
			t.Fatalf("headers = %v", got)
		}
		if got := diagnostics(); !strings.Contains(got, "x-obj") || !strings.Contains(got, "x-null") {
			t.Fatalf("diagnostics = %q", got)
		}
	})

	t.Run("LITELLM_HEADERS JSON feeds the default provider", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv(envHeaders, `{"x-tenant":"t1","x-n":2.5}`)
		got := resolveHeaders(getProviderDefinitions(nil)[0])
		if !reflect.DeepEqual(got, map[string]string{"x-tenant": "t1", "x-n": "2.5"}) {
			t.Fatalf("headers = %v", got)
		}
	})

	t.Run("invalid LITELLM_HEADERS is reported and ignored", func(t *testing.T) {
		hermeticAgentDir(t)
		diagnostics := captureDiagnostics(t)
		t.Setenv(envHeaders, "{nope")
		if got := resolveHeaders(getProviderDefinitions(nil)[0]); got != nil {
			t.Fatalf("headers = %v", got)
		}
		if !strings.Contains(diagnostics(), "failed to parse custom headers") {
			t.Fatalf("diagnostics = %q", diagnostics())
		}
	})

	t.Run("unset, blank and array headers yield none", func(t *testing.T) {
		hermeticAgentDir(t)
		for _, raw := range []string{"", "  ", "undefined", "[]", "{}", "null", `"x"`} {
			t.Setenv(envHeaders, raw)
			if got := resolveHeaders(getProviderDefinitions(nil)[0]); got != nil {
				t.Errorf("%q: headers = %v", raw, got)
			}
		}
	})
}

func TestResolveCredentials(t *testing.T) {
	ctx := context.Background()

	t.Run("treats literal undefined env values as unset", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv(envBaseURL, "undefined")
		t.Setenv(envAPIKey, "undefined")
		got, err := resolveCredentials(ctx, getProviderDefinitions(nil)[0], true)
		if err != nil || got.BaseURL != "" || got.APIKey != "" {
			t.Fatalf("got %+v, %v", got, err)
		}
	})

	t.Run("environment key and base URL for the default provider only", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv(envBaseURL, "https://proxy.example.com/v1/")
		t.Setenv(envAPIKey, "env-key")
		definitions := getProviderDefinitions(nil)
		got, err := resolveCredentials(ctx, definitions[0], true)
		if err != nil || got.BaseURL != "https://proxy.example.com" || got.APIKey != "env-key" || got.APIKeyConfig != "$"+envAPIKey {
			t.Fatalf("got %+v, %v", got, err)
		}
		alias := providerDefinition{Name: "alias"}
		if got, _ := resolveCredentials(ctx, alias, true); got.APIKey != "" || got.BaseURL != "" {
			t.Fatalf("alias leaked the default env: %+v", got)
		}
	})

	t.Run("insecure HTTP only when explicitly allowed", func(t *testing.T) {
		hermeticAgentDir(t)
		definition := providerDefinition{Name: "alias", BaseURL: "http://host.docker.internal", APIKeyConfig: "sk-local"}
		if _, err := resolveCredentials(ctx, definition, true); err == nil {
			t.Fatal("http to a non-loopback host must be rejected")
		}
		definition.AllowInsecureHTTP = true
		got, err := resolveCredentials(ctx, definition, true)
		if err != nil || got.BaseURL != "http://host.docker.internal" || got.APIKey != "sk-local" {
			t.Fatalf("got %+v, %v", got, err)
		}
	})

	t.Run("precedence: configured key, then helper, then environment key", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		helper, runs := writeHelper(t, dir, []string{"helper-key", "again"}, "h.sh")
		t.Setenv(envAPIKeyHelper, helper)
		t.Setenv(envAPIKey, "env-key")
		t.Setenv("CUSTOM_KEY", "configured-key")
		definition := providerDefinition{Name: providerName, UseDefaultEnv: true, APIKeyConfig: "$CUSTOM_KEY"}

		got, _ := resolveCredentials(ctx, definition, true)
		if got.APIKey != "configured-key" || got.APIKeyConfig != "$CUSTOM_KEY" || runs() != 0 {
			t.Fatalf("configured key must win without running the helper: %+v runs=%d", got, runs())
		}
		definition.APIKeyConfig = ""
		got, _ = resolveCredentials(ctx, definition, true)
		if got.APIKey != "helper-key" || got.APIKeyConfig != "!"+helper || runs() != 1 {
			t.Fatalf("helper must beat the env key: %+v runs=%d", got, runs())
		}
	})

	t.Run("executeHelpers false never runs a command and reports it as the config", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		helper, runs := writeHelper(t, dir, []string{"x"}, "h.sh")
		t.Setenv(envAPIKeyHelper, helper)
		definition := providerDefinition{Name: providerName, UseDefaultEnv: true}
		got, err := resolveCredentials(ctx, definition, false)
		if err != nil || got.APIKey != "" || got.APIKeyConfig != "!"+helper || runs() != 0 {
			t.Fatalf("env helper: %+v, %v, runs=%d", got, err, runs())
		}
		definition = providerDefinition{Name: "alias", APIKeyConfig: "!" + helper}
		got, err = resolveCredentials(ctx, definition, false)
		if err != nil || got.APIKey != "" || got.APIKeyConfig != "!"+helper || runs() != 0 {
			t.Fatalf("configured helper: %+v, %v, runs=%d", got, err, runs())
		}
	})

	t.Run("an unresolved configured key warns once", func(t *testing.T) {
		hermeticAgentDir(t)
		resetWarnedUnresolvedAPIKeys()
		t.Cleanup(resetWarnedUnresolvedAPIKeys)
		diagnostics := captureDiagnostics(t)
		definition := providerDefinition{Name: "warn-once-provider", APIKeyConfig: "$DEFINITELY_UNSET_KEY_VAR"}
		for range 2 {
			if got, _ := resolveCredentials(ctx, definition, true); got.APIKey != "" {
				t.Fatalf("got %+v", got)
			}
		}
		if got := diagnostics(); strings.Count(got, "did not resolve") != 1 || !strings.Contains(got, "warn-once-provider") {
			t.Fatalf("diagnostics = %q", got)
		}
	})

	t.Run("a failing configured helper is an error", func(t *testing.T) {
		hermeticAgentDir(t)
		definition := providerDefinition{Name: "alias", APIKeyConfig: "!false"}
		if _, err := resolveCredentials(ctx, definition, true); err == nil {
			t.Fatal("expected helper failure")
		}
	})
}

func TestReadStoredCredential(t *testing.T) {
	dir := t.TempDir()
	path := writeFileIn(t, dir, "auth.json", `{
	  "litellm": {"type":"oauth","access":"sk-sso","refresh":"r","expires":123,"baseUrl":"https://oauth.example.com"},
	  "keyed": {"type":"api_key","key":"sk-1","env":{"LITELLM_BASE_URL":"https://env.example.com"}},
	  "legacy": {"type":"api","apiKey":"sk-old"},
	  "broken": 7
	}`)
	oauth := readStoredCredential("litellm", path)
	if oauth == nil || oauth.Access != "sk-sso" || oauth.BaseURL != "https://oauth.example.com" || oauth.Expires == nil || *oauth.Expires != 123 {
		t.Fatalf("oauth = %+v", oauth)
	}
	if keyed := readStoredCredential("keyed", path); keyed == nil || keyed.Key != "sk-1" || keyed.Env[envBaseURL] != "https://env.example.com" {
		t.Fatalf("keyed = %+v", keyed)
	}
	if legacy := readStoredCredential("legacy", path); legacy == nil || legacy.Type != "api_key" || legacy.Key != "sk-old" {
		t.Fatalf("legacy = %+v", legacy)
	}
	for _, name := range []string{"missing", "broken"} {
		if readStoredCredential(name, path) != nil {
			t.Errorf("%s must be nil", name)
		}
	}
	if readStoredCredential("litellm", filepath.Join(dir, "absent.json")) != nil {
		t.Error("missing file must yield nil")
	}
}

func TestKnownBaseURL(t *testing.T) {
	definition := providerDefinition{Name: providerName, UseDefaultEnv: true}
	t.Run("names where the offered URL came from", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		writeFileIn(t, dir, "auth.json", `{"litellm":{"type":"oauth","access":"a","baseUrl":"https://old.example.com/v1/"}}`)
		if url, source, ok := knownBaseURL(definition); !ok || url != "https://old.example.com" || source != "previous login" {
			t.Fatalf("previous login: %q %q %v", url, source, ok)
		}
		t.Setenv(envBaseURL, "https://env.example.com")
		if url, source, _ := knownBaseURL(definition); url != "https://env.example.com" || source != "$LITELLM_BASE_URL" {
			t.Fatalf("env: %q %q", url, source)
		}
		definition := definition
		definition.BaseURL = "https://configured.example.com"
		if url, source, _ := knownBaseURL(definition); url != "https://configured.example.com" || source != "configured" {
			t.Fatalf("configured: %q %q", url, source)
		}
	})
	t.Run("ignores a stored base URL that is no longer usable", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		writeFileIn(t, dir, "auth.json", `{"litellm":{"type":"oauth","access":"a","baseUrl":"http://insecure.example.com"}}`)
		if _, _, ok := knownBaseURL(definition); ok {
			t.Fatal("an insecure stored URL must not be offered")
		}
	})
	t.Run("an api_key login offers its env base URL", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		writeFileIn(t, dir, "auth.json", `{"litellm":{"type":"api_key","key":"k","env":{"LITELLM_BASE_URL":"http://127.0.0.1:4000"}}}`)
		if url, source, ok := knownBaseURL(definition); !ok || url != "http://127.0.0.1:4000" || source != "previous login" {
			t.Fatalf("got %q %q %v", url, source, ok)
		}
	})
	t.Run("nothing configured", func(t *testing.T) {
		hermeticAgentDir(t)
		if _, _, ok := knownBaseURL(definition); ok {
			t.Fatal("expected nothing")
		}
	})
}
