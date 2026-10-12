// Ports tests/backend-identity.test.ts
package backend

import (
	"fmt"
	"testing"

	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

func row(model string, params *types.ModelInfoParams) types.ModelInfoEntry {
	params.Model = model
	return types.ModelInfoEntry{ModelName: "route", LiteLLMParams: params}
}

func TestResolveBackendIdentity(t *testing.T) {
	t.Run("uses base_model before the configured model and route name", func(t *testing.T) {
		got := ResolveIdentity(types.ModelInfoEntry{
			ModelName:     "public-alias",
			LiteLLMParams: &types.ModelInfoParams{Model: "azure/gpt-5"},
			ModelInfo:     &types.ModelInfoDetails{BaseModel: "fireworks_ai/accounts/fireworks/models/kimi-k3", LiteLLMProvider: "azure"},
		})
		if got == nil || got.Provider != "fireworks_ai" || got.ModelID != "accounts/fireworks/models/kimi-k3" ||
			got.QualifiedID != "fireworks_ai/accounts/fireworks/models/kimi-k3" {
			t.Fatalf("got %+v", got)
		}
	})
	t.Run("falls back through configured model to route name", func(t *testing.T) {
		got := ResolveIdentity(row("bedrock/claude-sonnet-4-6", &types.ModelInfoParams{}))
		if got == nil || got.Provider != "bedrock" || got.ModelID != "claude-sonnet-4-6" {
			t.Fatalf("got %+v", got)
		}
		got = ResolveIdentity(types.ModelInfoEntry{ModelName: "gpt-5"})
		if got == nil || got.ModelID != "gpt-5" || got.Family != FamilyOpenAI {
			t.Fatalf("got %+v", got)
		}
	})
	t.Run("never treats a generic adapter as backend identity", func(t *testing.T) {
		got := ResolveIdentity(types.ModelInfoEntry{ModelName: "opaque", ModelInfo: &types.ModelInfoDetails{LiteLLMProvider: "azure"}})
		if got == nil || *got != (Identity{ModelID: "opaque", QualifiedID: "opaque"}) {
			t.Fatalf("got %+v", got)
		}
	})
	t.Run("takes custom_llm_provider as provider evidence for an unprefixed model", func(t *testing.T) {
		cases := []struct {
			model, provider string
			want            Identity
		}{
			{"kimi-k3", "moonshot", Identity{"moonshot", "kimi-k3", "moonshot/kimi-k3", FamilyKimi}},
			{"opaque", "moonshot", Identity{"moonshot", "opaque", "moonshot/opaque", FamilyKimi}},
			// openai is any OpenAI-compatible server; the provider name proves no family.
			{"my-deploy", "openai", Identity{"openai", "my-deploy", "openai/my-deploy", ""}},
		}
		for _, c := range cases {
			got := ResolveIdentity(row(c.model, &types.ModelInfoParams{CustomLLMProvider: c.provider}))
			if got == nil || *got != c.want {
				t.Errorf("%s: got %+v want %+v", c.model, got, c.want)
			}
		}
	})
	t.Run("derives the family from custom_llm_provider for an opaque model", func(t *testing.T) {
		for provider, family := range map[string]Family{
			"anthropic": FamilyClaude, "deepseek": FamilyDeepSeek, "gemini": FamilyGemini,
			"moonshot": FamilyKimi, "moonshotai": FamilyKimi,
		} {
			got := ResolveIdentity(row("opaque", &types.ModelInfoParams{CustomLLMProvider: provider}))
			want := Identity{provider, "opaque", provider + "/opaque", family}
			if got == nil || *got != want {
				t.Errorf("%s: got %+v want %+v", provider, got, want)
			}
		}
	})
	t.Run("withholds identity when custom_llm_provider contradicts the model prefix", func(t *testing.T) {
		if got := ResolveIdentity(row("openai/gpt-4o", &types.ModelInfoParams{CustomLLMProvider: "anthropic"})); got != nil {
			t.Errorf("got %+v", got)
		}
		for _, m := range []string{"vertex_ai/claude-sonnet-4", "bedrock/anthropic.claude-3", "groq/llama-3.3-70b"} {
			if got := ResolveIdentity(row(m, &types.ModelInfoParams{CustomLLMProvider: "sagemaker"})); got != nil {
				t.Errorf("%s: got %+v", m, got)
			}
		}
		// A generic adapter is transport, so it cannot contradict the prefix.
		got := ResolveIdentity(row("azure_ai/FW-Kimi-K3", &types.ModelInfoParams{CustomLLMProvider: "azure"}))
		if got == nil || got.ModelID != "FW-Kimi-K3" || got.Family != FamilyKimi {
			t.Errorf("got %+v", got)
		}
		got = ResolveIdentity(row("fireworks_ai/accounts/fireworks/models/kimi-k3", &types.ModelInfoParams{CustomLLMProvider: "azure"}))
		if got == nil || got.Provider != "fireworks_ai" || got.Family != FamilyKimi {
			t.Errorf("got %+v", got)
		}
	})
	t.Run("treats a known model-path first segment as path when custom_llm_provider routes it", func(t *testing.T) {
		got := ResolveIdentity(types.ModelInfoEntry{
			ModelName:     "fireworks/kimi-k2p6",
			LiteLLMParams: &types.ModelInfoParams{Model: "accounts/fireworks/models/kimi-k2p6", CustomLLMProvider: "fireworks_ai"},
		})
		want := Identity{"fireworks_ai", "accounts/fireworks/models/kimi-k2p6", "fireworks_ai/accounts/fireworks/models/kimi-k2p6", FamilyKimi}
		if got == nil || *got != want {
			t.Fatalf("got %+v", got)
		}
	})
	t.Run("recognizes the settled OpenAI-family spelling", func(t *testing.T) {
		for _, id := range []string{"openai/gpt-5", "gpt-4.1", "gpt-5.3-codex", "codex-mini-latest", "o1", "o3-mini", "o4-mini", "o5-mini"} {
			if !IsOpenAIBackend(id) {
				t.Errorf("%s should be OpenAI", id)
			}
		}
		for _, id := range []string{"azure/kimi-k3", "deepseek-v4", "glm-5"} {
			if IsOpenAIBackend(id) {
				t.Errorf("%s should not be OpenAI", id)
			}
		}
		if DiscoveryVersion != 6 {
			t.Errorf("DiscoveryVersion = %d", DiscoveryVersion)
		}
	})
	t.Run("takes the family from the model id, never from an adapter prefix", func(t *testing.T) {
		family := func(model string) Family {
			got := ResolveIdentity(row(model, &types.ModelInfoParams{}))
			if got == nil {
				return ""
			}
			return got.Family
		}
		for _, m := range []string{"openai/qwen3.8-27B-a", "openai/LongCat-2.0", "openai_like/qwen3", "custom_openai/llama-4", "openai/production", "openai/custom:o1-clone"} {
			if f := family(m); f != "" {
				t.Errorf("%s: family %q", m, f)
			}
		}
		for _, m := range []string{"openai/gpt-5.5", "openai/o3", "openai/gpt-oss-120b", "openai/ft:gpt-4o-mini-2024-07-18:org::abc123", "openai/ft:o4-mini-2025-04-16:org::abc123", "chatgpt/gpt-5.6-sol", "azure/gpt-5"} {
			if f := family(m); f != FamilyOpenAI {
				t.Errorf("%s: family %q", m, f)
			}
		}
		if f := family("openai/kimi-k2.5"); f != FamilyKimi {
			t.Errorf("kimi: family %q", f)
		}
	})
	t.Run("classifies codex-mini-latest as an OpenAI backend", func(t *testing.T) {
		if got := ResolveIdentity(row("codex-mini-latest", &types.ModelInfoParams{})); got == nil || got.Family != FamilyOpenAI {
			t.Errorf("got %+v", got)
		}
	})
	t.Run("recognizes Fable as a Claude-family model", func(t *testing.T) {
		if got := ResolveIdentity(types.ModelInfoEntry{ModelName: "fable-5"}); got == nil || got.Family != FamilyClaude {
			t.Errorf("got %+v", got)
		}
	})
}

func TestResolveCatalogProvider(t *testing.T) {
	for _, reported := range []string{"azure", "azure_ai", "custom_openai", "openai", "openai_like", "text-completion-openai"} {
		t.Run(fmt.Sprintf("prefers the configured adapter over generic %s metadata", reported), func(t *testing.T) {
			got := ResolveCatalogProvider(types.ModelInfoEntry{
				ModelInfo:     &types.ModelInfoDetails{BaseModel: "openai/gpt-5.2", LiteLLMProvider: reported},
				LiteLLMParams: &types.ModelInfoParams{CustomLLMProvider: "azure"},
			})
			if got != "azure" {
				t.Errorf("got %q", got)
			}
		})
	}
	for _, c := range [][3]string{
		{"fireworks_ai/accounts/fireworks/models/kimi-k3", "azure", "fireworks_ai"},
		{"anthropic/claude-sonnet-4-6", "custom_openai", "anthropic"},
	} {
		got := ResolveCatalogProvider(types.ModelInfoEntry{
			ModelInfo:     &types.ModelInfoDetails{BaseModel: c[0], LiteLLMProvider: c[1]},
			LiteLLMParams: &types.ModelInfoParams{CustomLLMProvider: c[1]},
		})
		if got != c[2] {
			t.Errorf("%s through %s: got %q", c[0], c[1], got)
		}
	}
	t.Run("preserves a resolved provider over non-generic reported metadata", func(t *testing.T) {
		got := ResolveCatalogProvider(types.ModelInfoEntry{
			ModelInfo:     &types.ModelInfoDetails{BaseModel: "openai/gpt-5.2", LiteLLMProvider: "anthropic"},
			LiteLLMParams: &types.ModelInfoParams{CustomLLMProvider: "azure"},
		})
		if got != "openai" {
			t.Errorf("got %q", got)
		}
	})
	t.Run("falls back to reported then configured provider without an identity", func(t *testing.T) {
		got := ResolveCatalogProvider(types.ModelInfoEntry{ModelInfo: &types.ModelInfoDetails{LiteLLMProvider: " groq "}, LiteLLMParams: &types.ModelInfoParams{CustomLLMProvider: "azure"}})
		if got != "groq" {
			t.Errorf("got %q", got)
		}
		got = ResolveCatalogProvider(types.ModelInfoEntry{ModelInfo: &types.ModelInfoDetails{LiteLLMProvider: "undefined"}, LiteLLMParams: &types.ModelInfoParams{CustomLLMProvider: " azure "}})
		if got != "azure" {
			t.Errorf("got %q", got)
		}
	})
}

func TestRoutesOnlyThrough(t *testing.T) {
	anthropic := map[string]bool{"anthropic": true}
	t.Run("requires every declared routing signal to name an allowed provider", func(t *testing.T) {
		if !RoutesOnlyThrough(row("Anthropic/claude-fable-5-1", &types.ModelInfoParams{}), anthropic) {
			t.Error("prefix")
		}
		if !RoutesOnlyThrough(row("claude-fable-5-1", &types.ModelInfoParams{CustomLLMProvider: " anthropic "}), anthropic) {
			t.Error("custom")
		}
		if RoutesOnlyThrough(row("anthropic/claude-fable-5-1", &types.ModelInfoParams{CustomLLMProvider: "bedrock"}), anthropic) {
			t.Error("conflict")
		}
	})
	t.Run("ignores lookup metadata and treats a missing routing signal as unproven", func(t *testing.T) {
		r := types.ModelInfoEntry{
			LiteLLMParams: &types.ModelInfoParams{Model: "us.anthropic.claude-fable-5-1"},
			ModelInfo:     &types.ModelInfoDetails{BaseModel: "anthropic/claude-fable-5-1", LiteLLMProvider: "anthropic"},
		}
		if RoutesOnlyThrough(r, anthropic) {
			t.Error("lookup metadata")
		}
		if RoutesOnlyThrough(types.ModelInfoEntry{LiteLLMParams: &types.ModelInfoParams{Model: "/claude-fable-5-1"}}, anthropic) {
			t.Error("leading slash")
		}
	})
}
