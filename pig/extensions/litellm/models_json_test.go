package litellm

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/MichaelKinsy/PiG/ai"
)

// TestModelsJSONProviderConfig ports describe("models.json provider config") from tests/index.test.ts. The case
// that drives Pi's real ModelRuntime has no Go harness; the live smoke in pig/PORTING.md covers it.
func TestModelsJSONProviderConfig(t *testing.T) {
	writeModels := func(t *testing.T, dir string, value any) {
		t.Helper()
		raw, err := json.Marshal(value)
		if err != nil {
			t.Fatal(err)
		}
		writeFileIn(t, dir, "models.json", string(raw))
	}
	type object = map[string]any
	// resolveKey resolves a definition's key and root against the given environment and stored credential.
	resolveKey := func(t *testing.T, definition providerDefinition, env map[string]string, credential *ai.Credential) (apiKey, baseURL string) {
		t.Helper()
		got := mustResolve(t, definition, env, credential)
		return got.Auth.APIKey, got.Auth.BaseURL
	}
	wantAuth := func(t *testing.T, definition providerDefinition, env map[string]string, credential *ai.Credential, wantKey, wantBase string) {
		t.Helper()
		if key, base := resolveKey(t, definition, env, credential); key != wantKey || (wantBase != "" && base != wantBase) {
			t.Fatalf("auth = %q %q, want %q %q", key, base, wantKey, wantBase)
		}
	}
	wantRoot := func(t *testing.T, definition providerDefinition, want string) {
		t.Helper()
		credentials, err := resolveCredentials(t.Context(), definition, false)
		if err != nil || credentials.BaseURL != want {
			t.Fatalf("base URL = %q, %v, want %q", credentials.BaseURL, err, want)
		}
	}

	t.Run("registers the default provider from models.json providers.litellm", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		writeModels(t, dir, object{"providers": object{"litellm": object{"baseUrl": "https://models.example.com", "apiKey": "sk-models"}}})
		definition := defaultDefinition(t)
		wantRoot(t, definition, "https://models.example.com")
		wantAuth(t, definition, nil, nil, "sk-models", "https://models.example.com")
	})

	t.Run("registers aliases from models.json litellm.providers", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		writeModels(t, dir, object{
			"providers": object{},
			"litellm": object{"providers": object{"litellm-anthropic": object{
				"baseUrl": "https://litellm-anthropic.example.com", "apiKey": "$LITELLM_ANTHROPIC_API_KEY",
			}}},
		})
		definitions := getProviderDefinitions(readGlobalLiteLLMSettings())
		if len(definitions) != 2 || definitions[1].Name != "litellm-anthropic" {
			t.Fatalf("definitions = %+v", definitions)
		}
		wantAuth(t, definitions[1], map[string]string{"LITELLM_ANTHROPIC_API_KEY": "sk-anthropic"}, nil, "sk-anthropic", "")
	})

	t.Run("accepts the default provider through models.json litellm.providers.litellm", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		writeModels(t, dir, object{
			"providers": object{},
			"litellm":   object{"providers": object{"litellm": object{"baseUrl": "https://nested.example.com", "apiKey": "sk-nested"}}},
		})
		definition := defaultDefinition(t)
		wantRoot(t, definition, "https://nested.example.com")
		wantAuth(t, definition, nil, nil, "sk-nested", "")
	})

	t.Run("lets the native providers.litellm slot win per field over litellm.providers.litellm", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		writeModels(t, dir, object{
			"providers": object{"litellm": object{"baseUrl": "https://native.example.com"}},
			"litellm":   object{"providers": object{"litellm": object{"baseUrl": "https://nested.example.com", "apiKey": "sk-nested"}}},
		})
		definition := defaultDefinition(t)
		wantRoot(t, definition, "https://native.example.com")
		wantAuth(t, definition, nil, nil, "sk-nested", "")
	})

	t.Run("merges models.json over settings.json per field", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		writeFileIn(t, dir, "settings.json", `{"litellm":{"providers":{"litellm":{"baseUrl":"https://settings.example.com","displayName":"Settings Gateway","apiKey":"sk-settings"}}}}`)
		// models.json sets only baseUrl, so the apiKey and displayName settings.json sets must survive.
		writeModels(t, dir, object{"providers": object{"litellm": object{"baseUrl": "https://models.example.com"}}})
		definition := defaultDefinition(t)
		if definition.DisplayName != "Settings Gateway" {
			t.Fatalf("display name = %q", definition.DisplayName)
		}
		wantRoot(t, definition, "https://models.example.com")
		wantAuth(t, definition, nil, nil, "sk-settings", "")
	})

	t.Run("resolves a $ENV apiKey from models.json", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		writeModels(t, dir, object{"providers": object{"litellm": object{"baseUrl": "https://models.example.com", "apiKey": "$MODELS_JSON_KEY"}}})
		wantAuth(t, defaultDefinition(t), map[string]string{"MODELS_JSON_KEY": "sk-from-env"}, nil, "sk-from-env", "")
	})

	t.Run("keeps a stored credential ahead of a models.json apiKey", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		writeModels(t, dir, object{"providers": object{"litellm": object{"baseUrl": "https://models.example.com", "apiKey": "sk-models"}}})
		credential := &ai.Credential{Type: ai.CredentialAPIKey, Key: "sk-stored"}
		wantAuth(t, defaultDefinition(t), nil, credential, "sk-stored", "")
	})

	t.Run("keeps skills/mcp/budget flags in settings.json when models.json supplies providers", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		writeFileIn(t, dir, "settings.json", `{"litellm":{"skills":{"enabled":false}}}`)
		writeModels(t, dir, object{"providers": object{"litellm": object{"baseUrl": "https://models.example.com", "apiKey": "sk-models"}}})
		settings := readGlobalLiteLLMSettings()
		if isFeatureEnabled(settings, "skills") {
			t.Fatal("skills should stay disabled by settings.json")
		}
		wantRoot(t, getProviderDefinitions(settings)[0], "https://models.example.com")
	})

	t.Run("reads no skills/mcp/budget block from models.json", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		writeModels(t, dir, object{
			"providers": object{"litellm": object{"baseUrl": "https://models.example.com"}},
			"litellm":   object{"skills": object{"enabled": false}, "mcp": object{"enabled": false}, "budget": object{"enabled": false}},
		})
		settings := readGlobalLiteLLMSettings()
		for _, feature := range []string{"skills", "mcp", "budget"} {
			if !isFeatureEnabled(settings, feature) {
				t.Fatalf("%s switched off by models.json", feature)
			}
		}
	})

	t.Run("falls back to settings.json when models.json is malformed", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		writeFileIn(t, dir, "models.json", "{ not json")
		writeFileIn(t, dir, "settings.json", `{"litellm":{"providers":{"litellm":{"baseUrl":"https://settings.example.com","apiKey":"sk-settings"}}}}`)
		definition := defaultDefinition(t)
		wantRoot(t, definition, "https://settings.example.com")
		wantAuth(t, definition, nil, nil, "sk-settings", "")
	})

	t.Run("ignores a models.json that carries no litellm provider block", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		// `providers` is present but holds no litellm entry, so nothing is contributed to the config.
		writeModels(t, dir, object{"providers": object{"openai": object{"apiKey": "sk-openai"}}})
		t.Setenv(envBaseURL, "https://env.example.com")
		t.Setenv(envAPIKey, "sk-env")
		if readModelsProviderSettings() != nil {
			t.Fatal("models.json contributed settings")
		}
		definition := defaultDefinition(t)
		wantRoot(t, definition, "https://env.example.com")
		wantAuth(t, definition, map[string]string{envAPIKey: "sk-env", envBaseURL: "https://env.example.com"}, nil, "sk-env", "")
	})

	t.Run("parses a JSONC models.json with comments and trailing commas", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		// Pi strips only `//` comments, so a `/* */` block comment is deliberately absent here.
		writeFileIn(t, dir, "models.json", `{
  // Pi reads this file as JSONC, not strict JSON.
  "providers": {
    // the default provider
    "litellm": {
      "baseUrl": "https://models.example.com",
      "apiKey": "sk-models",
      "compat": { "supportsStrictTools": false, },
    },
  },
}`)
		definition := defaultDefinition(t)
		wantRoot(t, definition, "https://models.example.com")
		wantAuth(t, definition, nil, nil, "sk-models", "")
	})

	t.Run("parses a models.json that starts with a byte order mark", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		writeFileIn(t, dir, "models.json", "\uFEFF"+`{"providers":{"litellm":{"baseUrl":"https://models.example.com","apiKey":"sk-bom"}}}`)
		definition := defaultDefinition(t)
		wantRoot(t, definition, "https://models.example.com")
		wantAuth(t, definition, nil, nil, "sk-bom", "")
	})

	t.Run("keeps a // inside a models.json string literal", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		writeModels(t, dir, object{"providers": object{"litellm": object{"baseUrl": "https://models.example.com", "apiKey": "sk-models", "displayName": "Gateway // prod"}}})
		if got := defaultDefinition(t).DisplayName; got != "Gateway // prod" {
			t.Fatalf("display name = %q", got)
		}
	})

	t.Run("parses a settings.json that starts with a byte order mark", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		writeFileIn(t, dir, "settings.json", "\uFEFF"+`{"litellm":{"providers":{"litellm":{"baseUrl":"https://settings.example.com"}}}}`)
		wantRoot(t, defaultDefinition(t), "https://settings.example.com")
	})

	t.Run("orders providers as settings.json then models.json only names", func(t *testing.T) {
		dir := hermeticAgentDir(t)
		writeFileIn(t, dir, "settings.json", `{"litellm":{"providers":{"zeta":{},"litellm":{},"alpha":{}}}}`)
		writeFileIn(t, dir, "models.json", `{"providers":{"litellm":{"baseUrl":"https://m.example.com"}},"litellm":{"providers":{"beta":{},"alpha":{"displayName":"A"},"gamma":{}}}}`)
		settings := readGlobalLiteLLMSettings()
		if want := []string{"zeta", "litellm", "alpha", "beta", "gamma"}; !reflect.DeepEqual(settings.ProviderOrder, want) {
			t.Fatalf("order = %v, want %v", settings.ProviderOrder, want)
		}
		if only := readModelsProviderSettings(); !reflect.DeepEqual(only.ProviderOrder, []string{"beta", "alpha", "gamma", "litellm"}) {
			t.Fatalf("models order = %v", only.ProviderOrder)
		}
	})
}
