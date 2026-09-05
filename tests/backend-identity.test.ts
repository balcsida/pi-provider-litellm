import { describe, expect, it } from "vitest";
import { isOpenAIBackend, LITELLM_DISCOVERY_VERSION, resolveBackendIdentity } from "../src/backend-identity.js";

describe("resolveBackendIdentity", () => {
  it("uses base_model before the configured model and route name", () => {
    expect(
      resolveBackendIdentity({
        model_name: "public-alias",
        litellm_params: { model: "azure/gpt-5" },
        model_info: { base_model: "fireworks_ai/accounts/fireworks/models/kimi-k3", litellm_provider: "azure" },
      }),
    ).toMatchObject({
      provider: "fireworks_ai",
      modelId: "accounts/fireworks/models/kimi-k3",
      qualifiedId: "fireworks_ai/accounts/fireworks/models/kimi-k3",
    });
  });

  it("falls back through configured model to route name", () => {
    expect(
      resolveBackendIdentity({ model_name: "route", litellm_params: { model: "bedrock/claude-sonnet-4-6" } }),
    ).toMatchObject({ provider: "bedrock", modelId: "claude-sonnet-4-6" });
    expect(resolveBackendIdentity({ model_name: "gpt-5" })).toMatchObject({ modelId: "gpt-5", family: "openai" });
  });

  it("never treats a generic adapter as backend identity", () => {
    expect(resolveBackendIdentity({ model_name: "opaque", model_info: { litellm_provider: "azure" } })).toEqual({
      modelId: "opaque",
      qualifiedId: "opaque",
    });
  });

  it("takes custom_llm_provider as provider evidence for an unprefixed model", () => {
    expect(
      resolveBackendIdentity({
        model_name: "route",
        litellm_params: { model: "kimi-k3", custom_llm_provider: "moonshot" },
      }),
    ).toEqual({ provider: "moonshot", modelId: "kimi-k3", qualifiedId: "moonshot/kimi-k3", family: "kimi" });
    expect(
      resolveBackendIdentity({
        model_name: "route",
        litellm_params: { model: "opaque", custom_llm_provider: "moonshot" },
      }),
    ).toEqual({ provider: "moonshot", modelId: "opaque", qualifiedId: "moonshot/opaque", family: "kimi" });
    // `openai` is any OpenAI-compatible server in LiteLLM; the provider name proves no family.
    expect(
      resolveBackendIdentity({
        model_name: "route",
        litellm_params: { model: "my-deploy", custom_llm_provider: "openai" },
      }),
    ).toEqual({ provider: "openai", modelId: "my-deploy", qualifiedId: "openai/my-deploy" });
  });

  it.each([
    ["anthropic", "claude"],
    ["deepseek", "deepseek"],
    ["gemini", "gemini"],
    ["moonshot", "kimi"],
    ["moonshotai", "kimi"],
  ])("derives the %s family from custom_llm_provider for an opaque model", (provider, family) => {
    expect(
      resolveBackendIdentity({
        model_name: "route",
        litellm_params: { model: "opaque", custom_llm_provider: provider },
      }),
    ).toEqual({ provider, modelId: "opaque", qualifiedId: `${provider}/opaque`, family });
  });

  it("withholds identity when custom_llm_provider contradicts the model prefix", () => {
    expect(
      resolveBackendIdentity({
        model_name: "route",
        litellm_params: { model: "openai/gpt-4o", custom_llm_provider: "anthropic" },
      }),
    ).toBeUndefined();
    // A generic adapter is transport, so it cannot contradict the prefix.
    expect(
      resolveBackendIdentity({
        model_name: "route",
        litellm_params: { model: "azure_ai/FW-Kimi-K3", custom_llm_provider: "azure" },
      }),
    ).toMatchObject({ modelId: "FW-Kimi-K3", family: "kimi" });
    expect(
      resolveBackendIdentity({
        model_name: "route",
        litellm_params: { model: "fireworks_ai/accounts/fireworks/models/kimi-k3", custom_llm_provider: "azure" },
      }),
    ).toMatchObject({ provider: "fireworks_ai", family: "kimi" });
  });

  it("recognizes the settled OpenAI-family spelling", () => {
    for (const id of ["openai/gpt-5", "gpt-4.1", "gpt-5.3-codex", "o1", "o3-mini", "o4-mini", "o5-mini"]) {
      expect(isOpenAIBackend(id)).toBe(true);
    }
    for (const id of ["codex-mini-latest", "azure/kimi-k3", "deepseek-v4", "glm-5"]) {
      expect(isOpenAIBackend(id)).toBe(false);
    }
    expect(LITELLM_DISCOVERY_VERSION).toBe(2);
  });
});
