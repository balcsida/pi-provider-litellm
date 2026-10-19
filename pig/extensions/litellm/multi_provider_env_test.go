package litellm

// Ports tests/multi-provider-env.test.ts. The Vitest cases that load the extension through createPi() build the
// provider with extensionState.newProvider and call its auth methods directly, which is what the harness does.

import (
	"context"
	"encoding/json"
	"net/http"
	"os"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/MichaelKinsy/PiG/ai"
	sdk "github.com/MichaelKinsy/PiG/extensions/sdk"
)

// definitionNames lists the names of the definitions, in order.
func definitionNames(definitions []providerDefinition) []string {
	names := make([]string, 0, len(definitions))
	for _, definition := range definitions {
		names = append(names, definition.Name)
	}
	return names
}

// findDefinition returns the named definition, or nil.
func findDefinition(definitions []providerDefinition, name string) *providerDefinition {
	for index := range definitions {
		if definitions[index].Name == name {
			return &definitions[index]
		}
	}
	return nil
}

func mustFindDefinition(t *testing.T, definitions []providerDefinition, name string) providerDefinition {
	t.Helper()
	definition := findDefinition(definitions, name)
	if definition == nil {
		t.Fatalf("no %q in %v", name, definitionNames(definitions))
	}
	return *definition
}

func expectNames(t *testing.T, definitions []providerDefinition, want ...string) {
	t.Helper()
	if got := definitionNames(definitions); !slices.Equal(got, want) {
		t.Fatalf("names = %v, want %v", got, want)
	}
}

func expectEnableMCP(t *testing.T, definition providerDefinition, want *bool) {
	t.Helper()
	if (definition.EnableMCP == nil) != (want == nil) || (want != nil && *definition.EnableMCP != *want) {
		t.Errorf("%s: EnableMCP = %v, want %v", definition.Name, derefBool(definition.EnableMCP), derefBool(want))
	}
}

func derefBool(value *bool) any {
	if value == nil {
		return nil
	}
	return *value
}

// loadProvider builds the native provider of the named definition, as the extension factory does.
func loadProvider(t *testing.T, dir, name string) (*ai.ModelsProvider, *extensionState) {
	t.Helper()
	state, _ := startupIn(t, dir)
	definition := mustFindDefinition(t, state.definitions, name)
	provider, err := state.newProvider(definition, nil)
	if err != nil {
		t.Fatal(err)
	}
	return provider, state
}

// processAuthInput reads the process environment, as the Vitest helpers do with process.env.
func processAuthInput() ai.APIKeyAuthInput { return ai.APIKeyAuthInput{} }

func TestIsValidProviderName(t *testing.T) {
	t.Run("accepts valid lowercase alphanumeric and hyphen/underscore provider names", func(t *testing.T) {
		for _, name := range []string{"corp", "team-a", "team_b", "dev-1_prod", "a", strings.Repeat("a", 64)} {
			if !isValidProviderName(name) {
				t.Errorf("%q rejected", name)
			}
		}
	})

	t.Run("rejects invalid provider names", func(t *testing.T) {
		for _, name := range []string{"", "TEAM", "teamA", "-team", "_team", "team name", "team@corp", strings.Repeat("a", 65)} {
			if isValidProviderName(name) {
				t.Errorf("%q accepted", name)
			}
		}
	})

	t.Run("rejects reserved tool and command names", func(t *testing.T) {
		for _, name := range []string{"mcp", "skills", "codemode", "tool_search", "litellm"} {
			if isValidProviderName(name) {
				t.Errorf("%q accepted", name)
			}
		}
	})
}

func TestParseCanonicalProviderList(t *testing.T) {
	t.Run("parses comma- and whitespace-separated provider IDs", func(t *testing.T) {
		for raw, want := range map[string][]string{
			"corp-east, dev":                {"corp-east", "dev"},
			"corp-east dev team_b":          {"corp-east", "dev", "team_b"},
			"  team-a,   team_b , team-c  ": {"team-a", "team_b", "team-c"},
		} {
			if got := parseCanonicalProviderList(raw); !slices.Equal(got, want) {
				t.Errorf("%q: got %v, want %v", raw, got, want)
			}
		}
	})

	t.Run("handles empty or undefined input", func(t *testing.T) {
		for _, raw := range []string{"", "   ", ", , ,"} {
			if got := parseCanonicalProviderList(raw); len(got) != 0 {
				t.Errorf("%q: got %v", raw, got)
			}
		}
	})
}

func TestParseProvidersJson(t *testing.T) {
	t.Run("parses valid JSON provider settings map", func(t *testing.T) {
		order, result := parseProvidersJson(`{"team-a":{"baseUrl":"https://team-a.example.com","apiKey":"sk-team-a","displayName":"Team A"},"team-b":{"baseUrl":"https://team-b.example.com","allowInsecureHttp":true}}`)
		if result == nil || !slices.Equal(order, []string{"team-a", "team-b"}) {
			t.Fatalf("order = %v, result = %v", order, result)
		}
		wantA := map[string]any{"baseUrl": "https://team-a.example.com", "apiKey": "sk-team-a", "displayName": "Team A"}
		wantB := map[string]any{"baseUrl": "https://team-b.example.com", "allowInsecureHttp": true}
		if !reflect.DeepEqual(result["team-a"], wantA) || !reflect.DeepEqual(result["team-b"], wantB) {
			t.Fatalf("result = %v", result)
		}
	})

	t.Run("returns undefined for invalid or non-object JSON", func(t *testing.T) {
		for _, raw := range []string{"", "{invalid json", "123", `"string"`, `["array"]`, "null", `{"a":1}`} {
			if order, result := parseProvidersJson(raw); result != nil || order != nil {
				t.Errorf("%q: got %v, %v", raw, order, result)
			}
		}
	})
}

func TestParseProviderEnvVars(t *testing.T) {
	t.Run("scans and extracts provider fields by prefix token", func(t *testing.T) {
		tokens, parsed := parseProviderEnvVars([]string{
			"LITELLM_PROVIDER_CORP_EAST_BASE_URL=https://corp-east.example.com",
			"LITELLM_PROVIDER_CORP_EAST_API_KEY=sk-corp-east",
			"LITELLM_PROVIDER_CORP_EAST_DISPLAY_NAME=Corporate East",
			"LITELLM_PROVIDER_CORP_EAST_ALLOW_INSECURE_HTTP=1",
			"LITELLM_PROVIDER_CORP_EAST_USE_GCLOUD_AUTH=true",
			"LITELLM_PROVIDER_CORP_EAST_ENABLE_OAUTH=1",
			"LITELLM_PROVIDER_DEV_BASE_URL=https://dev.example.com",
			"LITELLM_PROVIDER_DEV_API_KEY_HELPER=dev-key-helper",
			"LITELLM_BASE_URL=https://primary.example.com",
			"OTHER_VAR=ignored",
		})
		if !slices.Equal(tokens, []string{"CORP_EAST", "DEV"}) {
			t.Fatalf("tokens = %v", tokens)
		}
		wantCorpEast := &parsedProviderEnv{
			Token: "CORP_EAST", EnvPrefix: "LITELLM_PROVIDER_CORP_EAST", BaseURL: "https://corp-east.example.com",
			APIKey: "sk-corp-east", DisplayName: "Corporate East",
			AllowInsecureHTTP: new(true), UseGcloudTokenAuth: new(true), EnableOAuth: new(true),
		}
		if !reflect.DeepEqual(parsed["CORP_EAST"], wantCorpEast) {
			t.Errorf("CORP_EAST = %+v", parsed["CORP_EAST"])
		}
		wantDev := &parsedProviderEnv{Token: "DEV", EnvPrefix: "LITELLM_PROVIDER_DEV", BaseURL: "https://dev.example.com", APIKeyHelper: "dev-key-helper"}
		if !reflect.DeepEqual(parsed["DEV"], wantDev) {
			t.Errorf("DEV = %+v", parsed["DEV"])
		}
		if parsed["LITELLM"] != nil || parsed["OTHER"] != nil {
			t.Errorf("unexpected tokens in %v", parsed)
		}
	})

	t.Run("maps boolean fields correctly", func(t *testing.T) {
		_, parsed := parseProviderEnvVars([]string{
			"LITELLM_PROVIDER_TEST_ALLOW_INSECURE_HTTP=true",
			"LITELLM_PROVIDER_TEST_USE_GCLOUD_AUTH=1",
			"LITELLM_PROVIDER_TEST_ENABLE_OAUTH=0",
		})
		test := parsed["TEST"]
		if test.AllowInsecureHTTP == nil || !*test.AllowInsecureHTTP ||
			test.UseGcloudTokenAuth == nil || !*test.UseGcloudTokenAuth ||
			test.EnableOAuth == nil || *test.EnableOAuth {
			t.Fatalf("TEST = %+v", test)
		}
	})

	t.Run("parses ENABLE_MCP and MCP_ENABLED boolean variations", func(t *testing.T) {
		_, parsed := parseProviderEnvVars([]string{
			"LITELLM_PROVIDER_P1_ENABLE_MCP=0",
			"LITELLM_PROVIDER_P2_ENABLE_MCP=false",
			"LITELLM_PROVIDER_P3_MCP_ENABLED=0",
			"LITELLM_PROVIDER_P4_MCP_ENABLED=false",
			"LITELLM_PROVIDER_P5_ENABLE_MCP=1",
			"LITELLM_PROVIDER_P6_MCP_ENABLED=true",
		})
		for token, want := range map[string]bool{"P1": false, "P2": false, "P3": false, "P4": false, "P5": true, "P6": true} {
			if got := parsed[token].EnableMCP; got == nil || *got != want {
				t.Errorf("%s: EnableMCP = %v, want %v", token, derefBool(got), want)
			}
		}
	})

	t.Run("handles tokens containing reserved substrings like NAME or API", func(t *testing.T) {
		_, parsed := parseProviderEnvVars([]string{
			"LITELLM_PROVIDER_MY_NAME_NAME=custom-name-override",
			"LITELLM_PROVIDER_MY_NAME_BASE_URL=https://myname.example.com",
			"LITELLM_PROVIDER_API_CORP_API_KEY_HELPER=helper-script",
			"LITELLM_PROVIDER_API_CORP_BASE_URL=https://apicorp.example.com",
		})
		wantMyName := &parsedProviderEnv{Token: "MY_NAME", EnvPrefix: "LITELLM_PROVIDER_MY_NAME", Name: "custom-name-override", BaseURL: "https://myname.example.com"}
		if !reflect.DeepEqual(parsed["MY_NAME"], wantMyName) {
			t.Errorf("MY_NAME = %+v", parsed["MY_NAME"])
		}
		wantAPICorp := &parsedProviderEnv{Token: "API_CORP", EnvPrefix: "LITELLM_PROVIDER_API_CORP", APIKeyHelper: "helper-script", BaseURL: "https://apicorp.example.com"}
		if !reflect.DeepEqual(parsed["API_CORP"], wantAPICorp) {
			t.Errorf("API_CORP = %+v", parsed["API_CORP"])
		}
	})

	t.Run("does not parse OIDC env vars for secondary providers", func(t *testing.T) {
		_, parsed := parseProviderEnvVars([]string{
			"LITELLM_PROVIDER_TEAM_A_BASE_URL=https://team-a.example.com",
			`LITELLM_PROVIDER_TEAM_A_OIDC={"clientId": "abc", "issuer": "https://auth.example.com"}`,
		})
		want := &parsedProviderEnv{Token: "TEAM_A", EnvPrefix: "LITELLM_PROVIDER_TEAM_A", BaseURL: "https://team-a.example.com"}
		if !reflect.DeepEqual(parsed["TEAM_A"], want) {
			t.Errorf("TEAM_A = %+v", parsed["TEAM_A"])
		}
	})
}

func TestGetProviderDefinitionsWithEnvironmentVariables(t *testing.T) {
	definitionsFor := func(settingsJSON string) []providerDefinition {
		return getProviderDefinitions(settingsFromJSON(t, settingsJSON))
	}

	t.Run("always includes primary litellm provider with useDefaultEnv: true", func(t *testing.T) {
		hermeticAgentDir(t)
		definitions := getProviderDefinitions(nil)
		if len(definitions) != 1 {
			t.Fatalf("definitions = %v", definitionNames(definitions))
		}
		d := definitions[0]
		if d.Name != "litellm" || d.DisplayName != "LiteLLM" || !d.UseDefaultEnv || !d.UseGcloudTokenAuth || !d.EnableOAuth {
			t.Fatalf("definition = %+v", d)
		}
	})

	t.Run("discovers single and multiple providers via prefix-scanned env vars", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv("LITELLM_PROVIDER_CORP_BASE_URL", "https://corp.example.com")
		t.Setenv("LITELLM_PROVIDER_CORP_API_KEY", "sk-corp")
		t.Setenv("LITELLM_PROVIDER_DEV_BASE_URL", "https://dev.example.com")
		t.Setenv("LITELLM_PROVIDER_DEV_API_KEY", "sk-dev")
		definitions := getProviderDefinitions(nil)
		expectNames(t, definitions, "litellm", "corp", "dev")
		corp := mustFindDefinition(t, definitions, "corp")
		if corp.DisplayName != "corp" || corp.EnvPrefix != "LITELLM_PROVIDER_CORP" || corp.UseDefaultEnv {
			t.Fatalf("corp = %+v", corp)
		}
	})

	t.Run("normalizes uppercase tokens with underscores to lowercase hyphens by default", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv("LITELLM_PROVIDER_TEAM_A_BASE_URL", "https://team-a.example.com")
		t.Setenv("LITELLM_PROVIDER_TEAM_A_API_KEY", "sk-team-a")
		teamA := mustFindDefinition(t, getProviderDefinitions(nil), "team-a")
		if teamA.EnvPrefix != "LITELLM_PROVIDER_TEAM_A" {
			t.Fatalf("EnvPrefix = %q", teamA.EnvPrefix)
		}
	})

	t.Run("allows overriding provider ID via _NAME variable", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv("LITELLM_PROVIDER_TEAM_A_NAME", "custom_team")
		t.Setenv("LITELLM_PROVIDER_TEAM_A_BASE_URL", "https://team-a.example.com")
		definitions := getProviderDefinitions(nil)
		if findDefinition(definitions, "team-a") != nil {
			t.Fatal("team-a must not exist")
		}
		if got := mustFindDefinition(t, definitions, "custom_team").EnvPrefix; got != "LITELLM_PROVIDER_TEAM_A" {
			t.Fatalf("EnvPrefix = %q", got)
		}
	})

	t.Run("preserves canonical list IDs specified in LITELLM_PROVIDERS", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv("LITELLM_PROVIDERS", "team-a,team_b")
		t.Setenv("LITELLM_PROVIDER_TEAM_A_BASE_URL", "https://team-a.example.com")
		t.Setenv("LITELLM_PROVIDER_TEAM_B_BASE_URL", "https://team-b.example.com")
		definitions := getProviderDefinitions(nil)
		expectNames(t, definitions, "litellm", "team-a", "team_b")
		if got := mustFindDefinition(t, definitions, "team_b").EnvPrefix; got != "LITELLM_PROVIDER_TEAM_B" {
			t.Fatalf("EnvPrefix = %q", got)
		}
	})

	t.Run("parses providers from LITELLM_PROVIDERS_JSON", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv("LITELLM_PROVIDERS_JSON", `{"json-provider":{"baseUrl":"https://json.example.com","apiKey":"sk-json","displayName":"JSON Provider"}}`)
		p := mustFindDefinition(t, getProviderDefinitions(nil), "json-provider")
		if p.DisplayName != "JSON Provider" || p.BaseURL != "https://json.example.com" || p.APIKeyConfig != "sk-json" || p.UseDefaultEnv {
			t.Fatalf("json-provider = %+v", p)
		}
	})

	t.Run("attaches scoped env vars to a LITELLM_PROVIDERS_JSON id instead of deriving a second provider", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv("LITELLM_PROVIDERS_JSON", `{"team_a":{"baseUrl":"https://team.example.com"}}`)
		t.Setenv("LITELLM_PROVIDER_TEAM_A_API_KEY", "sk-team-a")
		definitions := getProviderDefinitions(nil)
		expectNames(t, definitions, "litellm", "team_a")
		teamA := mustFindDefinition(t, definitions, "team_a")
		if teamA.EnvPrefix != "LITELLM_PROVIDER_TEAM_A" {
			t.Fatalf("EnvPrefix = %q", teamA.EnvPrefix)
		}
		creds, err := resolveCredentials(context.Background(), teamA, true)
		if err != nil || creds.BaseURL != "https://team.example.com" || creds.APIKey != "sk-team-a" {
			t.Fatalf("creds = %+v, %v", creds, err)
		}
	})

	t.Run("gives a token shared by LITELLM_PROVIDERS and LITELLM_PROVIDERS_JSON ids to the LITELLM_PROVIDERS id", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv("LITELLM_PROVIDERS", "team-a")
		t.Setenv("LITELLM_PROVIDERS_JSON", `{"team_a":{"baseUrl":"https://json.example.com"}}`)
		t.Setenv("LITELLM_PROVIDER_TEAM_A_API_KEY", "sk-team-a")
		definitions := getProviderDefinitions(nil)
		expectNames(t, definitions, "litellm", "team_a", "team-a")
		if got := mustFindDefinition(t, definitions, "team-a").EnvPrefix; got != "LITELLM_PROVIDER_TEAM_A" {
			t.Fatalf("team-a EnvPrefix = %q", got)
		}
		if got := mustFindDefinition(t, definitions, "team_a").EnvPrefix; got != "" {
			t.Fatalf("team_a EnvPrefix = %q", got)
		}
	})

	t.Run("attaches scoped env vars to a settings.json alias instead of deriving a second provider", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv("LITELLM_PROVIDER_TEAM_A_API_KEY", "sk-team-a")
		t.Setenv("LITELLM_PROVIDER_TEAM_A_ENABLE_MCP", "0")
		definitions := definitionsFor(`{"providers":{"team_a":{"baseUrl":"https://team.example.com"}}}`)
		expectNames(t, definitions, "litellm", "team_a")
		teamA := mustFindDefinition(t, definitions, "team_a")
		if teamA.EnvPrefix != "LITELLM_PROVIDER_TEAM_A" || teamA.EnableMCP == nil || *teamA.EnableMCP {
			t.Fatalf("team_a = %+v", teamA)
		}
		creds, err := resolveCredentials(context.Background(), teamA, true)
		if err != nil || creds.BaseURL != "https://team.example.com" || creds.APIKey != "sk-team-a" {
			t.Fatalf("creds = %+v, %v", creds, err)
		}
	})

	t.Run("merges settings.json with env vars: disk settings override matching fields", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv("LITELLM_PROVIDER_CORP_BASE_URL", "https://env.example.com")
		t.Setenv("LITELLM_PROVIDER_CORP_API_KEY", "sk-env")
		corp := mustFindDefinition(t, definitionsFor(`{"providers":{"corp":{"baseUrl":"https://disk.example.com"}}}`), "corp")
		if corp.BaseURL != "https://disk.example.com" || corp.EnvPrefix != "LITELLM_PROVIDER_CORP" {
			t.Fatalf("corp = %+v", corp)
		}
	})

	t.Run("combines non-overlapping disk settings and env vars additively", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv("LITELLM_PROVIDER_CORP_BASE_URL", "https://corp.example.com")
		expectNames(t, definitionsFor(`{"providers":{"diskonly":{"baseUrl":"https://diskonly.example.com"}}}`), "litellm", "corp", "diskonly")
	})

	t.Run("excludes providers when disk settings specify enabled: false", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv("LITELLM_PROVIDER_DISABLED_BASE_URL", "https://disabled.example.com")
		expectNames(t, definitionsFor(`{"providers":{"disabled":{"enabled":false}}}`), "litellm")
	})

	t.Run("excludes providers when LITELLM_PROVIDERS_JSON specifies enabled: false", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv("LITELLM_PROVIDERS_JSON", `{"disabled":{"baseUrl":"https://disabled.example.com","enabled":false}}`)
		expectNames(t, getProviderDefinitions(nil), "litellm")
	})

	t.Run("rejects invalid or reserved provider names from env and settings", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv("LITELLM_PROVIDER_MCP_BASE_URL", "https://mcp.example.com")
		t.Setenv("LITELLM_PROVIDER_SKILLS_BASE_URL", "https://skills.example.com")
		t.Setenv("LITELLM_PROVIDER_INVALID_NAME", "Invalid Provider!")
		t.Setenv("LITELLM_PROVIDER_INVALID_BASE_URL", "https://invalid.example.com")
		expectNames(t, definitionsFor(`{"providers":{"tool_search":{"baseUrl":"https://tool.example.com"},"skills":{"baseUrl":"https://skills.example.com"},"mcp":{"baseUrl":"https://mcp.example.com"},"":{"baseUrl":"https://empty.example.com"}}}`), "litellm")
	})

	t.Run("preserves settings.json aliases like TeamA and team.corp for backwards compatibility", func(t *testing.T) {
		hermeticAgentDir(t)
		definitions := definitionsFor(`{"providers":{"TeamA":{"baseUrl":"https://teama.example.com","apiKey":"sk-teama"},"team.corp":{"baseUrl":"https://teamcorp.example.com","displayName":"Team Corp"}}}`)
		expectNames(t, definitions, "litellm", "TeamA", "team.corp")
		teamA := mustFindDefinition(t, definitions, "TeamA")
		if teamA.DisplayName != "TeamA" || teamA.BaseURL != "https://teama.example.com" || teamA.APIKeyConfig != "sk-teama" {
			t.Fatalf("TeamA = %+v", teamA)
		}
		teamCorp := mustFindDefinition(t, definitions, "team.corp")
		if teamCorp.DisplayName != "Team Corp" || teamCorp.BaseURL != "https://teamcorp.example.com" {
			t.Fatalf("team.corp = %+v", teamCorp)
		}
	})

	t.Run("ensures disabled providers in LITELLM_PROVIDERS_JSON are not re-added by subsequent env steps", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv("LITELLM_PROVIDERS_JSON", `{"team-a":{"baseUrl":"https://json-teama.example.com","enabled":false},"team-b":{"baseUrl":"https://json-teamb.example.com","enabled":false}}`)
		// Attempt to re-add via the canonical list LITELLM_PROVIDERS
		t.Setenv("LITELLM_PROVIDERS", "team-a,team-c")
		t.Setenv("LITELLM_PROVIDER_TEAM_A_BASE_URL", "https://env-teama.example.com")
		// Attempt to re-add via prefix scanning
		t.Setenv("LITELLM_PROVIDER_TEAM_B_BASE_URL", "https://env-teamb.example.com")
		t.Setenv("LITELLM_PROVIDER_TEAM_C_BASE_URL", "https://env-teamc.example.com")
		expectNames(t, getProviderDefinitions(nil), "litellm", "team-c")
	})

	t.Run("prevents distinct canonical IDs colliding on token (e.g. team-a vs team_a) from sharing prefix", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv("LITELLM_PROVIDERS", "team-a,team_a")
		t.Setenv("LITELLM_PROVIDER_TEAM_A_BASE_URL", "https://teama.example.com")
		t.Setenv("LITELLM_PROVIDER_TEAM_A_API_KEY", "sk-team-a")
		definitions := getProviderDefinitions(nil)
		expectNames(t, definitions, "litellm", "team-a", "team_a")

		teamHyphen := mustFindDefinition(t, definitions, "team-a")
		if teamHyphen.EnvPrefix != "LITELLM_PROVIDER_TEAM_A" {
			t.Fatalf("team-a EnvPrefix = %q", teamHyphen.EnvPrefix)
		}
		teamUnderscore := mustFindDefinition(t, definitions, "team_a")
		if teamUnderscore.EnvPrefix != "" {
			t.Fatalf("team_a EnvPrefix = %q", teamUnderscore.EnvPrefix)
		}

		hyphenCreds, err := resolveCredentials(context.Background(), teamHyphen, true)
		if err != nil || hyphenCreds.BaseURL != "https://teama.example.com" || hyphenCreds.APIKey != "sk-team-a" {
			t.Fatalf("team-a creds = %+v, %v", hyphenCreds, err)
		}
		underscoreCreds, err := resolveCredentials(context.Background(), teamUnderscore, true)
		if err != nil || underscoreCreds.BaseURL != "" || underscoreCreds.APIKey != "" {
			t.Fatalf("team_a creds = %+v, %v", underscoreCreds, err)
		}
	})

	t.Run("retains undefined envPrefix for team_a when settings.json defines team_a and LITELLM_PROVIDERS has team-a,team_a", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv("LITELLM_PROVIDERS", "team-a,team_a")
		t.Setenv("LITELLM_PROVIDER_TEAM_A_BASE_URL", "https://teama.example.com")
		t.Setenv("LITELLM_PROVIDER_TEAM_A_API_KEY", "sk-team-a")
		definitions := definitionsFor(`{"providers":{"team_a":{"displayName":"Team A from Settings"}}}`)

		if got := mustFindDefinition(t, definitions, "team-a").EnvPrefix; got != "LITELLM_PROVIDER_TEAM_A" {
			t.Fatalf("team-a EnvPrefix = %q", got)
		}
		teamUnderscore := mustFindDefinition(t, definitions, "team_a")
		if teamUnderscore.EnvPrefix != "" || teamUnderscore.DisplayName != "Team A from Settings" {
			t.Fatalf("team_a = %+v", teamUnderscore)
		}
		creds, err := resolveCredentials(context.Background(), teamUnderscore, true)
		if err != nil || creds.BaseURL != "" || creds.APIKey != "" {
			t.Fatalf("team_a creds = %+v, %v", creds, err)
		}
	})

	t.Run("ensures secondary providers always have oidc: undefined even if configured in settings or JSON", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv("LITELLM_PROVIDERS_JSON", `{"team-json":{"baseUrl":"https://team-json.example.com","oidc":{"clientId":"xyz","issuer":"https://auth.example.com"}}}`)
		t.Setenv("LITELLM_PROVIDER_TEAM_ENV_BASE_URL", "https://team-env.example.com")
		definitions := definitionsFor(`{"providers":{"litellm":{"oidc":{"clientId":"primary","issuer":"https://auth-primary.example.com"}},"team-disk":{"baseUrl":"https://team-disk.example.com","oidc":{"clientId":"disk","issuer":"https://auth.example.com"}}}}`)

		primary := mustFindDefinition(t, definitions, "litellm")
		want := map[string]any{"clientId": "primary", "issuer": "https://auth-primary.example.com"}
		if !primary.HasOIDC || !reflect.DeepEqual(primary.OIDC, want) {
			t.Fatalf("primary oidc = %v (present %v)", primary.OIDC, primary.HasOIDC)
		}
		for _, name := range []string{"team-json", "team-env", "team-disk"} {
			if definition := mustFindDefinition(t, definitions, name); definition.OIDC != nil || definition.HasOIDC {
				t.Errorf("%s oidc = %v (present %v)", name, definition.OIDC, definition.HasOIDC)
			}
		}
	})

	t.Run("does not spawn an alias provider from LITELLM_PROVIDER_LITELLM_* variables", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv("LITELLM_PROVIDER_LITELLM_NAME", "team")
		t.Setenv("LITELLM_PROVIDER_LITELLM_ENABLE_MCP", "0")
		definitions := getProviderDefinitions(nil)
		expectNames(t, definitions, "litellm")
		expectEnableMCP(t, definitions[0], new(false))
		if findDefinition(definitions, "team") != nil {
			t.Fatal("team must not exist")
		}
	})

	t.Run("sets enableMcp on ProviderDefinition from environment variables", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv("LITELLM_PROVIDER_DISABLED_MCP_BASE_URL", "https://disabled-mcp.example.com")
		t.Setenv("LITELLM_PROVIDER_DISABLED_MCP_ENABLE_MCP", "false")
		t.Setenv("LITELLM_PROVIDER_ENABLED_MCP_BASE_URL", "https://enabled-mcp.example.com")
		t.Setenv("LITELLM_PROVIDER_ENABLED_MCP_MCP_ENABLED", "1")
		t.Setenv("LITELLM_PROVIDER_DEFAULT_MCP_BASE_URL", "https://default-mcp.example.com")
		t.Setenv("LITELLM_PROVIDER_LITELLM_ENABLE_MCP", "0")
		definitions := getProviderDefinitions(nil)
		expectEnableMCP(t, mustFindDefinition(t, definitions, "litellm"), new(false))
		expectEnableMCP(t, mustFindDefinition(t, definitions, "disabled-mcp"), new(false))
		expectEnableMCP(t, mustFindDefinition(t, definitions, "enabled-mcp"), new(true))
		expectEnableMCP(t, mustFindDefinition(t, definitions, "default-mcp"), nil)
	})

	t.Run("configures or overrides enableMcp via settings.json mcp.enabled or enableMcp", func(t *testing.T) {
		hermeticAgentDir(t)
		t.Setenv("LITELLM_PROVIDER_CORP_BASE_URL", "https://corp.example.com")
		t.Setenv("LITELLM_PROVIDER_CORP_ENABLE_MCP", "1")
		t.Setenv("LITELLM_PROVIDER_TEAM_A_BASE_URL", "https://team-a.example.com")
		t.Setenv("LITELLM_PROVIDER_TEAM_A_ENABLE_MCP", "0")
		definitions := definitionsFor(`{"providers":{"litellm":{"enableMcp":false},"corp":{"mcp":{"enabled":false}},"team-a":{"enableMcp":true},"team-b":{"baseUrl":"https://team-b.example.com","mcp":false}}}`)
		expectEnableMCP(t, mustFindDefinition(t, definitions, "litellm"), new(false))
		expectEnableMCP(t, mustFindDefinition(t, definitions, "corp"), new(false))
		expectEnableMCP(t, mustFindDefinition(t, definitions, "team-a"), new(true))
		expectEnableMCP(t, mustFindDefinition(t, definitions, "team-b"), new(false))
	})
}

// settingsFromJSON decodes a litellm settings block the way readGlobalLiteLLMSettings does.
func settingsFromJSON(t *testing.T, block string) *litellmSettings {
	t.Helper()
	if block == "" {
		return nil
	}
	writeFileIn(t, os.Getenv("PIG_CODING_AGENT_DIR"), "settings.json", `{"litellm":`+block+`}`)
	return readGlobalLiteLLMSettings()
}

func TestPerProviderCredentialAndEndpointResolution(t *testing.T) {
	t.Run("uses scoped _BASE_URL and _API_KEY for secondary providers", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		t.Setenv("LITELLM_PROVIDER_CORP_BASE_URL", "https://corp.example.com")
		t.Setenv("LITELLM_PROVIDER_CORP_API_KEY", "sk-corp")
		t.Setenv(envTimeout, "0")
		corp, _ := loadProvider(t, dir, "corp")

		check, err := corp.Auth.APIKey.Check(context.Background(), processAuthInput())
		if err != nil || check == nil || check.Type != ai.CredentialAPIKey || check.Source != "$LITELLM_PROVIDER_CORP_API_KEY" {
			t.Fatalf("check = %+v, %v", check, err)
		}
		result, err := corp.Auth.APIKey.Resolve(context.Background(), processAuthInput())
		if err != nil || result == nil {
			t.Fatalf("resolve = %+v, %v", result, err)
		}
		if result.Auth.APIKey != "sk-corp" || result.Auth.Headers != nil || result.Auth.BaseURL != "https://corp.example.com" ||
			!reflect.DeepEqual(result.Env, map[string]string{"LITELLM_BASE_URL": "https://corp.example.com"}) ||
			result.Source != "$LITELLM_PROVIDER_CORP_API_KEY" {
			t.Fatalf("result = %+v", result)
		}
	})

	t.Run("does NOT leak or fallback to global LITELLM_BASE_URL and LITELLM_API_KEY for secondary providers", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		t.Setenv("LITELLM_BASE_URL", "https://primary.example.com")
		t.Setenv("LITELLM_API_KEY", "sk-primary")
		t.Setenv("LITELLM_PROVIDERS", "corp")
		t.Setenv(envTimeout, "0")
		corp, _ := loadProvider(t, dir, "corp")

		// Corp has no scoped BASE_URL or API_KEY, so check and resolve must return nothing.
		if check, err := corp.Auth.APIKey.Check(context.Background(), processAuthInput()); err != nil || check != nil {
			t.Fatalf("check = %+v, %v", check, err)
		}
		if result, err := corp.Auth.APIKey.Resolve(context.Background(), processAuthInput()); err != nil || result != nil {
			t.Fatalf("resolve = %+v, %v", result, err)
		}

		// Primary provider still resolves its global credentials.
		litellm, _ := loadProvider(t, dir, "litellm")
		result, err := litellm.Auth.APIKey.Resolve(context.Background(), processAuthInput())
		if err != nil || result == nil || result.Auth.APIKey != "sk-primary" || result.Auth.BaseURL != "https://primary.example.com" || result.Source != "LITELLM_API_KEY" {
			t.Fatalf("litellm = %+v, %v", result, err)
		}
	})

	t.Run("lazily executes scoped _API_KEY_HELPER command for secondary providers", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		t.Setenv("LITELLM_PROVIDER_CORP_BASE_URL", "https://corp.example.com")
		t.Setenv("LITELLM_PROVIDER_CORP_API_KEY_HELPER", "echo sk-corp-helper")
		t.Setenv(envTimeout, "0")
		corp, _ := loadProvider(t, dir, "corp")

		check, err := corp.Auth.APIKey.Check(context.Background(), processAuthInput())
		if err != nil || check == nil || check.Source != "$LITELLM_PROVIDER_CORP_API_KEY_HELPER" {
			t.Fatalf("check = %+v, %v", check, err)
		}
		result, err := corp.Auth.APIKey.Resolve(context.Background(), processAuthInput())
		if err != nil || result == nil {
			t.Fatalf("resolve = %+v, %v", result, err)
		}
		if result.Auth.APIKey != "sk-corp-helper" || result.Auth.Headers != nil || result.Auth.BaseURL != "https://corp.example.com" ||
			!reflect.DeepEqual(result.Env, map[string]string{"LITELLM_BASE_URL": "https://corp.example.com"}) ||
			result.Source != "$LITELLM_PROVIDER_CORP_API_KEY_HELPER" {
			t.Fatalf("result = %+v", result)
		}
	})

	t.Run("resolves and passes scoped _HEADERS JSON for secondary providers", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		t.Setenv("LITELLM_PROVIDER_CORP_BASE_URL", "https://corp.example.com")
		t.Setenv("LITELLM_PROVIDER_CORP_API_KEY", "sk-corp")
		t.Setenv("LITELLM_PROVIDER_CORP_HEADERS", `{"X-Custom-Tenant":"tenant-corp","X-Secret-Env":"$DYNAMIC_SECRET"}`)
		t.Setenv("DYNAMIC_SECRET", "secret-123")
		t.Setenv(envTimeout, "0")
		corp, _ := loadProvider(t, dir, "corp")

		result, err := corp.Auth.APIKey.Resolve(context.Background(), processAuthInput())
		if err != nil || result == nil {
			t.Fatalf("resolve = %+v, %v", result, err)
		}
		got := map[string]string{}
		for name, value := range result.Auth.Headers {
			got[name] = *value
		}
		if want := map[string]string{"X-Custom-Tenant": "tenant-corp", "X-Secret-Env": "secret-123"}; !reflect.DeepEqual(got, want) {
			t.Fatalf("headers = %v", got)
		}
	})

	t.Run("enables Google ADC when _USE_GCLOUD_AUTH is set to 1 for secondary providers", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		gcloudTest(t)
		tokenServer(t, http.StatusOK, "ya29.corp-minted")
		t.Setenv(envGoogleCredentials, writeADCFile(t, authorizedUser(nil)))
		t.Setenv("LITELLM_PROVIDER_CORP_BASE_URL", "https://corp.example.com")
		t.Setenv("LITELLM_PROVIDER_CORP_USE_GCLOUD_AUTH", "1")
		t.Setenv(envTimeout, "0")
		corp, _ := loadProvider(t, dir, "corp")

		check, err := corp.Auth.APIKey.Check(context.Background(), processAuthInput())
		if err != nil || check == nil || check.Type != ai.CredentialAPIKey || check.Source != "gcloud ADC" {
			t.Fatalf("check = %+v, %v", check, err)
		}
		result, err := corp.Auth.APIKey.Resolve(context.Background(), processAuthInput())
		if err != nil || result == nil {
			t.Fatalf("resolve = %+v, %v", result, err)
		}
		if result.Auth.APIKey != "ya29.corp-minted" || result.Auth.Headers != nil || result.Auth.BaseURL != "https://corp.example.com" ||
			!reflect.DeepEqual(result.Env, map[string]string{"LITELLM_BASE_URL": "https://corp.example.com"}) ||
			result.Source != "gcloud ADC" {
			t.Fatalf("result = %+v", result)
		}
	})

	t.Run("does NOT use Google ADC for secondary providers when _USE_GCLOUD_AUTH is not set", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		gcloudTest(t)
		t.Setenv(envGoogleCredentials, writeADCFile(t, authorizedUser(nil)))
		t.Setenv("LITELLM_PROVIDER_CORP_BASE_URL", "https://corp.example.com")
		t.Setenv(envTimeout, "0")
		corp, _ := loadProvider(t, dir, "corp")

		if check, err := corp.Auth.APIKey.Check(context.Background(), processAuthInput()); err != nil || check != nil {
			t.Fatalf("check = %+v, %v", check, err)
		}
		if result, err := corp.Auth.APIKey.Resolve(context.Background(), processAuthInput()); err != nil || result != nil {
			t.Fatalf("resolve = %+v, %v", result, err)
		}
	})

	t.Run("enables OAuth descriptor on provider auth when _ENABLE_OAUTH is set to 1", func(t *testing.T) {
		dir := oauthTest(t)
		t.Setenv("LITELLM_PROVIDER_CORP_BASE_URL", "https://corp.example.com")
		t.Setenv("LITELLM_PROVIDER_CORP_ENABLE_OAUTH", "1")
		t.Setenv("LITELLM_PROVIDER_DEV_BASE_URL", "https://dev.example.com")
		t.Setenv("LITELLM_PROVIDER_DEV_ENABLE_OAUTH", "0")
		t.Setenv(envTimeout, "0")

		corp, _ := loadProvider(t, dir, "corp")
		if corp.Auth.OAuth == nil || corp.Auth.OAuth.Name != "LiteLLM SSO" {
			t.Fatalf("corp oauth = %+v", corp.Auth.OAuth)
		}
		dev, _ := loadProvider(t, dir, "dev")
		if dev.Auth.OAuth != nil {
			t.Fatalf("dev oauth = %+v", dev.Auth.OAuth)
		}
	})

	t.Run("resolveCredentials directly", func(t *testing.T) {
		corpDefinition := providerDefinition{Name: "corp", DisplayName: "corp", EnvPrefix: "LITELLM_PROVIDER_CORP"}

		t.Run("resolves scoped baseUrl and apiKey for secondary provider definition", func(t *testing.T) {
			hermeticAgentDir(t)
			t.Setenv("LITELLM_PROVIDER_CORP_BASE_URL", "https://corp.example.com")
			t.Setenv("LITELLM_PROVIDER_CORP_API_KEY", "sk-corp")
			creds, err := resolveCredentials(context.Background(), corpDefinition, true)
			if err != nil || creds.BaseURL != "https://corp.example.com" || creds.APIKey != "sk-corp" ||
				creds.APIKeyConfig != "$LITELLM_PROVIDER_CORP_API_KEY" || creds.APIKeyFromGcloudADC {
				t.Fatalf("creds = %+v, %v", creds, err)
			}
		})

		t.Run("does not fall back to global LITELLM_BASE_URL and LITELLM_API_KEY", func(t *testing.T) {
			hermeticAgentDir(t)
			t.Setenv("LITELLM_BASE_URL", "https://primary.example.com")
			t.Setenv("LITELLM_API_KEY", "sk-primary")
			creds, err := resolveCredentials(context.Background(), corpDefinition, true)
			if err != nil || creds.BaseURL != "" || creds.APIKey != "" || creds.APIKeyConfig != "" || creds.APIKeyFromGcloudADC {
				t.Fatalf("creds = %+v, %v", creds, err)
			}
		})

		t.Run("resolves scoped API key helper command", func(t *testing.T) {
			hermeticAgentDir(t)
			t.Setenv("LITELLM_PROVIDER_CORP_BASE_URL", "https://corp.example.com")
			t.Setenv("LITELLM_PROVIDER_CORP_API_KEY_HELPER", "echo sk-helper-out")
			creds, err := resolveCredentials(context.Background(), corpDefinition, true)
			if err != nil || creds.BaseURL != "https://corp.example.com" || creds.APIKey != "sk-helper-out" ||
				creds.APIKeyConfig != "!echo sk-helper-out" || creds.APIKeyFromGcloudADC {
				t.Fatalf("creds = %+v, %v", creds, err)
			}
		})
	})

	t.Run("interactive login for secondary providers", func(t *testing.T) {
		t.Run("provides apiKey.login on secondary providers", func(t *testing.T) {
			dir := oauthTest(t)
			t.Setenv("LITELLM_PROVIDER_CORP_BASE_URL", "https://corp.example.com")
			t.Setenv(envTimeout, "0")
			corp, _ := loadProvider(t, dir, "corp")
			if corp.Auth.APIKey.Login == nil {
				t.Fatal("no API-key login")
			}
		})

		t.Run("offers the provider-scoped base URL in knownBaseUrl for /login <alias>", func(t *testing.T) {
			dir := oauthTest(t)
			t.Setenv("LITELLM_PROVIDER_CORP_BASE_URL", "https://corp.example.com")
			t.Setenv(envTimeout, "0")
			corp, _ := loadProvider(t, dir, "corp")

			var offered []ai.AuthSelectOption
			script := &scriptedAuth{answer: func(prompt ai.AuthPrompt) (string, error) {
				if selection, ok := prompt.(ai.AuthSelectPrompt); ok {
					offered = selection.Options
					return selection.Options[0].ID, nil
				}
				return "sk-corp-key", nil
			}}
			credential, err := corp.Auth.APIKey.Login(context.Background(), script.interaction())
			if err != nil {
				t.Fatal(err)
			}
			want := []ai.AuthSelectOption{
				{ID: "https://corp.example.com", Label: "https://corp.example.com ($LITELLM_PROVIDER_CORP_BASE_URL)"},
				{ID: "enter-a-different-url", Label: "Enter a different URL…"},
			}
			if !reflect.DeepEqual(offered, want) {
				t.Fatalf("offered = %+v", offered)
			}
			if credential.Type != ai.CredentialAPIKey || credential.Key != "sk-corp-key" ||
				!reflect.DeepEqual(credential.Env, map[string]string{"LITELLM_BASE_URL": "https://corp.example.com"}) {
				t.Fatalf("credential = %+v", credential)
			}
		})

		t.Run("completes login for a secondary provider with custom URL and API key", func(t *testing.T) {
			dir := oauthTest(t)
			t.Setenv("LITELLM_PROVIDER_CORP_BASE_URL", "https://corp.example.com")
			t.Setenv(envTimeout, "0")
			corp, _ := loadProvider(t, dir, "corp")

			script := &scriptedAuth{answer: func(prompt ai.AuthPrompt) (string, error) {
				switch typed := prompt.(type) {
				case ai.AuthSelectPrompt:
					return "enter-a-different-url", nil
				case ai.AuthTextPrompt:
					if typed.Placeholder != "" {
						return "https://custom-corp.example.com", nil
					}
				}
				return "sk-custom-corp", nil
			}}
			credential, err := corp.Auth.APIKey.Login(context.Background(), script.interaction())
			if err != nil {
				t.Fatal(err)
			}
			if credential.Type != ai.CredentialAPIKey || credential.Key != "sk-custom-corp" ||
				!reflect.DeepEqual(credential.Env, map[string]string{"LITELLM_BASE_URL": "https://custom-corp.example.com"}) {
				t.Fatalf("credential = %+v", credential)
			}
		})

		// corpMcpFixture wires the hooks with an alias `corp` that has a stored credential, and starts the session.
		corpMcpFixture := func(t *testing.T, corpSettings string) (*recordingHooks, *extensionState, *fakeHookHost, *ai.ModelsProvider) {
			t.Helper()
			hooks, state, host := wire(t, `{"litellm":{"providers":{"corp":`+corpSettings+`}}}`)
			writeFileIn(t, agentDir(), "auth.json", `{"corp":{"type":"api_key","key":"sk-stored","env":{"LITELLM_BASE_URL":"https://corp.example.com"}}}`)
			hooks.emit(t, sdk.EventSessionStart, host, nil)
			provider, err := state.newProvider(mustFindDefinition(t, state.definitions, "corp"), nil)
			if err != nil {
				t.Fatal(err)
			}
			return hooks, state, host, provider
		}
		// promptRecorder answers the login like the Vitest prompt and records whether corp is registered each time.
		promptRecorder := func(host *fakeHookHost) (*scriptedAuth, *[]bool) {
			var registered []bool
			return &scriptedAuth{answer: func(prompt ai.AuthPrompt) (string, error) {
				_, isRegistered := host.servers["corp"]
				registered = append(registered, isRegistered)
				if selection, ok := prompt.(ai.AuthSelectPrompt); ok {
					return selection.Options[0].ID, nil
				}
				return "sk-new-key", nil
			}}, &registered
		}

		t.Run("drops existing MCP registration for <alias> on login start", func(t *testing.T) {
			oauthTest(t)
			hooks, _, host, corp := corpMcpFixture(t, `{"baseUrl":"https://corp.example.com"}`)
			if _, ok := host.servers["corp"]; !ok {
				t.Fatal("corp is not registered")
			}
			script, registered := promptRecorder(host)
			if _, err := corp.Auth.APIKey.Login(context.Background(), script.interaction()); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(*registered, []bool{false, false}) {
				t.Fatalf("registered when prompted = %v", *registered)
			}
			if _, ok := host.servers["corp"]; ok {
				t.Fatal("corp is still registered after login")
			}
			hooks.emit(t, sdk.EventBeforeAgentStart, host, map[string]any{"systemPrompt": fakePrompt("")})
			if _, ok := host.servers["corp"]; !ok {
				t.Fatal("corp is not registered again on the next turn")
			}
		})

		t.Run("does not call syncMcpServer in complete() and does not re-enable disabled MCP", func(t *testing.T) {
			oauthTest(t)
			hooks, _, host, corp := corpMcpFixture(t, `{"baseUrl":"https://corp.example.com","enableMcp":false}`)
			if _, ok := host.servers["corp"]; ok {
				t.Fatal("corp is registered")
			}
			script, _ := promptRecorder(host)
			if _, err := corp.Auth.APIKey.Login(context.Background(), script.interaction()); err != nil {
				t.Fatal(err)
			}
			if _, ok := host.servers["corp"]; ok {
				t.Fatal("corp is registered after login")
			}
			hooks.emit(t, sdk.EventBeforeAgentStart, host, map[string]any{"systemPrompt": fakePrompt("")})
			if _, ok := host.servers["corp"]; ok {
				t.Fatal("corp is registered on the next turn")
			}
		})
	})
}

func TestPerProviderMCPRegistrationToggle(t *testing.T) {
	// startSession wires the hooks for the settings, applies env to the definitions, and starts the session.
	startSession := func(t *testing.T, settings string, env map[string]string) (*recordingHooks, *extensionState, *fakeHookHost) {
		t.Helper()
		hooks, state, host := wire(t, settings)
		for name, value := range env {
			t.Setenv(name, value)
		}
		state.definitions = getProviderDefinitions(state.settings)
		hooks.emit(t, sdk.EventSessionStart, host, nil)
		return hooks, state, host
	}
	registered := func(host *fakeHookHost, name string) bool {
		_, ok := host.servers[name]
		return ok
	}

	t.Run("does not register MCP server for a provider when enableMcp is false", func(t *testing.T) {
		hermeticAgentDir(t)
		dir := agentDir()
		auth := map[string]any{}
		for name, root := range map[string]string{
			"litellm":                "https://proxy.example.com",
			"team-enabled":           "https://enabled.example.com",
			"team-disabled-env":      "https://disabled-env.example.com",
			"team-disabled-settings": "https://disabled-settings.example.com",
		} {
			auth[name] = map[string]any{"type": "api_key", "key": "sk-" + name, "env": map[string]string{"LITELLM_BASE_URL": root}}
		}
		raw, _ := json.Marshal(auth)
		writeFileIn(t, dir, "auth.json", string(raw))

		_, _, host := startSession(t, `{"litellm":{"providers":{
			"team-enabled":{"baseUrl":"https://enabled.example.com"},
			"team-disabled-env":{"baseUrl":"https://disabled-env.example.com"},
			"team-disabled-settings":{"baseUrl":"https://disabled-settings.example.com","mcp":{"enabled":false}}}}}`,
			map[string]string{
				"LITELLM_BASE_URL": "https://proxy.example.com", "LITELLM_API_KEY": "sk-litellm",
				"LITELLM_PROVIDER_TEAM_DISABLED_ENV_ENABLE_MCP": "false",
			})
		if !registered(host, "litellm") || !registered(host, "team-enabled") ||
			registered(host, "team-disabled-env") || registered(host, "team-disabled-settings") {
			t.Fatalf("servers = %s", mcpJSON(t, host.fakeRegistrar))
		}
	})

	t.Run("does not register MCP server for primary litellm provider when disabled via env or settings", func(t *testing.T) {
		hermeticAgentDir(t)
		writeFileIn(t, agentDir(), "auth.json", `{"litellm":{"type":"api_key","key":"sk-litellm","env":{"LITELLM_BASE_URL":"https://proxy.example.com"}}}`)
		_, _, host := startSession(t, `{"litellm":{"providers":{"litellm":{"enableMcp":false}}}}`,
			map[string]string{"LITELLM_BASE_URL": "https://proxy.example.com", "LITELLM_API_KEY": "sk-litellm"})
		if registered(host, "litellm") {
			t.Fatalf("servers = %s", mcpJSON(t, host.fakeRegistrar))
		}
	})

	t.Run("does not register MCP server for primary litellm provider when disabled via env var", func(t *testing.T) {
		hermeticAgentDir(t)
		writeFileIn(t, agentDir(), "auth.json", `{"litellm":{"type":"api_key","key":"sk-litellm","env":{"LITELLM_BASE_URL":"https://proxy.example.com"}}}`)
		_, _, host := startSession(t, "", map[string]string{
			"LITELLM_BASE_URL": "https://proxy.example.com", "LITELLM_API_KEY": "sk-litellm", "LITELLM_PROVIDER_LITELLM_ENABLE_MCP": "0",
		})
		if registered(host, "litellm") {
			t.Fatalf("servers = %s", mcpJSON(t, host.fakeRegistrar))
		}
	})

	t.Run("skips MCP access check in mcpAccessRefused for an alias provider with useGcloudTokenAuth: true", func(t *testing.T) {
		hermeticAgentDir(t)
		checks := mcpProxy(t, noServersGranted)
		_, _, host := startSession(t, "", map[string]string{
			"LITELLM_PROVIDER_CORP_BASE_URL": "https://corp.example.com", "LITELLM_PROVIDER_CORP_API_KEY": "sk-fallback",
			"LITELLM_PROVIDER_CORP_USE_GCLOUD_AUTH": "1",
		})
		if len(*checks) != 0 {
			t.Fatalf("access checks = %d", len(*checks))
		}
		want := `{"type":"http","url":"https://corp.example.com/mcp","auth":{"provider":"corp"}}`
		if got, _ := json.Marshal(host.servers["corp"]); string(got) != want {
			t.Fatalf("corp = %s, want %s", got, want)
		}
	})
}
