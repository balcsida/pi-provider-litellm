import { describe, expect, it } from "vitest";
import {
  isOpenAIBackend,
  LITELLM_DISCOVERY_VERSION,
  resolveBackendIdentity,
  resolveCatalogProvider,
  routesOnlyThrough,
} from "../src/backend-identity.js";

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
    // Routing providers conflict too, not just family vendors; so do prefixes we don't recognize.
    for (const model of ["vertex_ai/claude-sonnet-4", "bedrock/anthropic.claude-3", "groq/llama-3.3-70b"]) {
      expect(
        resolveBackendIdentity({ model_name: "route", litellm_params: { model, custom_llm_provider: "sagemaker" } }),
      ).toBeUndefined();
    }
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

  it("treats a known model-path first segment as path when custom_llm_provider routes it", () => {
    // LiteLLM prepends custom_llm_provider when the first segment differs, so this is
    // `fireworks_ai/accounts/fireworks/models/kimi-k2p6`, not a provider named "accounts".
    expect(
      resolveBackendIdentity({
        model_name: "fireworks/kimi-k2p6",
        litellm_params: { model: "accounts/fireworks/models/kimi-k2p6", custom_llm_provider: "fireworks_ai" },
      }),
    ).toEqual({
      provider: "fireworks_ai",
      modelId: "accounts/fireworks/models/kimi-k2p6",
      qualifiedId: "fireworks_ai/accounts/fireworks/models/kimi-k2p6",
      family: "kimi",
    });
  });

  it("recognizes the settled OpenAI-family spelling", () => {
    for (const id of [
      "openai/gpt-5",
      "gpt-4.1",
      "gpt-5.3-codex",
      "codex-mini-latest",
      "o1",
      "o3-mini",
      "o4-mini",
      "o5-mini",
    ]) {
      expect(isOpenAIBackend(id)).toBe(true);
    }
    for (const id of ["azure/kimi-k3", "deepseek-v4", "glm-5"]) {
      expect(isOpenAIBackend(id)).toBe(false);
    }
    expect(LITELLM_DISCOVERY_VERSION).toBe(5);
  });

  it("takes the family from the model id, never from an adapter prefix", () => {
    const family = (model: string) =>
      resolveBackendIdentity({ model_name: "route", litellm_params: { model } })?.family;
    for (const model of [
      "openai/qwen3.8-27B-a",
      "openai/LongCat-2.0",
      "openai_like/qwen3",
      "custom_openai/llama-4",
      "openai/production",
    ]) {
      expect(family(model)).toBeUndefined();
    }
    for (const model of [
      "openai/gpt-5.5",
      "openai/o3",
      "openai/gpt-oss-120b",
      "openai/ft:gpt-4o-mini-2024-07-18:org::abc123",
      "openai/ft:o4-mini-2025-04-16:org::abc123",
      "chatgpt/gpt-5.6-sol",
      "azure/gpt-5",
    ]) {
      expect(family(model)).toBe("openai");
    }
    expect(family("openai/custom:o1-clone")).toBeUndefined();
    expect(family("openai/kimi-k2.5")).toBe("kimi");
  });

  it("classifies codex-mini-latest as an OpenAI backend", () => {
    expect(
      resolveBackendIdentity({ model_name: "route", litellm_params: { model: "codex-mini-latest" } })?.family,
    ).toBe("openai");
  });

  it("recognizes Fable as a Claude-family model", () => {
    expect(resolveBackendIdentity({ model_name: "fable-5" })).toMatchObject({ family: "claude" });
  });
});

describe("resolveCatalogProvider", () => {
  it.each(["azure", "azure_ai", "custom_openai", "openai", "openai_like", "text-completion-openai"])(
    "prefers the configured adapter over generic %s metadata",
    (reported) => {
      expect(
        resolveCatalogProvider({
          model_info: { base_model: "openai/gpt-5.2", litellm_provider: reported },
          litellm_params: { custom_llm_provider: "azure" },
        }),
      ).toBe("azure");
    },
  );

  it.each([
    ["fireworks_ai/accounts/fireworks/models/kimi-k3", "azure", "fireworks_ai"],
    ["anthropic/claude-sonnet-4-6", "custom_openai", "anthropic"],
  ])("preserves specific backend %s authority through %s", (model, adapter, provider) => {
    expect(
      resolveCatalogProvider({
        model_info: { base_model: model, litellm_provider: adapter },
        litellm_params: { custom_llm_provider: adapter },
      }),
    ).toBe(provider);
  });

  it("preserves a resolved provider over non-generic reported metadata", () => {
    expect(
      resolveCatalogProvider({
        model_info: { base_model: "openai/gpt-5.2", litellm_provider: "anthropic" },
        litellm_params: { custom_llm_provider: "azure" },
      }),
    ).toBe("openai");
  });

  it("falls back to reported then configured provider without an identity", () => {
    expect(
      resolveCatalogProvider({
        model_info: { litellm_provider: " groq " },
        litellm_params: { custom_llm_provider: "azure" },
      }),
    ).toBe("groq");
    expect(
      resolveCatalogProvider({
        model_info: { litellm_provider: "undefined" },
        litellm_params: { custom_llm_provider: " azure " },
      }),
    ).toBe("azure");
  });
});

describe("routesOnlyThrough", () => {
  const anthropic = new Set(["anthropic"]);

  it("requires every declared routing signal to name an allowed provider", () => {
    expect(routesOnlyThrough({ litellm_params: { model: "Anthropic/claude-fable-5-1" } }, anthropic)).toBe(true);
    expect(
      routesOnlyThrough(
        { litellm_params: { model: "claude-fable-5-1", custom_llm_provider: " anthropic " } },
        anthropic,
      ),
    ).toBe(true);
    expect(
      routesOnlyThrough(
        { litellm_params: { model: "anthropic/claude-fable-5-1", custom_llm_provider: "bedrock" } },
        anthropic,
      ),
    ).toBe(false);
  });

  it("ignores lookup metadata and treats a missing routing signal as unproven", () => {
    const row = {
      litellm_params: { model: "us.anthropic.claude-fable-5-1" },
      model_info: { base_model: "anthropic/claude-fable-5-1", litellm_provider: "anthropic" },
    };
    expect(routesOnlyThrough(row, anthropic)).toBe(false);
    expect(routesOnlyThrough({ litellm_params: { model: "/claude-fable-5-1" } }, anthropic)).toBe(false);
  });
});
