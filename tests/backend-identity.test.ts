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

  it("withholds identity when a base_model or route-name prefix contradicts custom_llm_provider", () => {
    // LiteLLM routes only the configured model, so these prefixes name a backend, not a namespace.
    for (const model of ["openai/gpt-4o", "vertex_ai/claude-sonnet-4", "groq/llama-3.3-70b"]) {
      expect(
        resolveBackendIdentity({
          litellm_params: { model: "opaque", custom_llm_provider: "sagemaker" },
          model_info: { base_model: model },
        }),
      ).toBeUndefined();
      expect(
        resolveBackendIdentity({ model_name: model, litellm_params: { custom_llm_provider: "sagemaker" } }),
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

  it("reads a vendor namespace on the openai transport instead of withholding the identity", () => {
    // LiteLLM publishes this deployment with base_model `deepseek/deepseek-v4.1-flash` while
    // routing it through the OpenAI-compatible transport. The prefix names the vendor, so the
    // identity resolves; a routing provider that is not an OpenAI-compatible transport still
    // contradicts an unrecognised first segment.
    const row = {
      model_name: "commandcode/deepseek/deepseek-v4.1-flash",
      litellm_params: { model: "commandcode/deepseek/deepseek-v4.1-flash", custom_llm_provider: "openai" },
      model_info: { base_model: "deepseek/deepseek-v4.1-flash", litellm_provider: "openai" },
    };
    expect(resolveBackendIdentity(row)).toEqual({
      provider: "openai",
      modelId: "deepseek/deepseek-v4.1-flash",
      qualifiedId: "openai/deepseek/deepseek-v4.1-flash",
      family: "deepseek",
    });
    expect(resolveCatalogProvider(row)).toBe("openai");
    // Both transport spellings resolve the same way.
    expect(
      resolveBackendIdentity({
        ...row,
        litellm_params: { ...row.litellm_params, custom_llm_provider: "text-completion-openai" },
      }),
    ).toMatchObject({
      provider: "text-completion-openai",
      modelId: "deepseek/deepseek-v4.1-flash",
      family: "deepseek",
    });
    // A transport label in the differing segment names no vendor, so it must not reach the family
    // scan: "openai/qwen3" stays familyless instead of being read as OpenAI.
    expect(
      resolveBackendIdentity({
        ...row,
        model_info: { base_model: "openai/qwen3", litellm_provider: "text-completion-openai" },
        litellm_params: { model: "openai/qwen3", custom_llm_provider: "text-completion-openai" },
      }),
    ).toEqual({ provider: "text-completion-openai", modelId: "qwen3", qualifiedId: "text-completion-openai/qwen3" });
    expect(
      resolveBackendIdentity({
        ...row,
        litellm_params: { ...row.litellm_params, custom_llm_provider: "sagemaker" },
      }),
    ).toBeUndefined();
  });

  it.each([
    ["openai/gpt-4o", "openai"],
    ["deepseek/deepseek-v4.1-flash", "deepseek"],
    ["moonshotai/kimi-k2.5", "kimi"],
    ["anthropic/claude-sonnet-4-6", "claude"],
    ["google/gemini-3-pro", "gemini"],
    ["z-ai/glm-5.3", undefined],
    ["example-lab/opaque", undefined],
  ] as const)("resolves the OpenRouter model namespace %s without a routing prefix", (model, family) => {
    const row = {
      model_name: "public-alias",
      litellm_params: { model, custom_llm_provider: "openrouter" },
      model_info: { litellm_provider: "openai" },
    };
    const identity = resolveBackendIdentity(row);
    expect(identity).toEqual({
      provider: "openrouter",
      modelId: model,
      qualifiedId: `openrouter/${model}`,
      ...(family ? { family } : {}),
    });
    expect(identity).toEqual(
      resolveBackendIdentity({
        ...row,
        litellm_params: { ...row.litellm_params, model: `openrouter/${model}` },
      }),
    );
    expect(resolveCatalogProvider(row)).toBe("openrouter");
    expect(routesOnlyThrough(row, new Set(["anthropic", "openai", "azure", "azure_ai"]))).toBe(false);
    expect(row.litellm_params.model).toBe(model);
  });

  it.each([
    ["hosted_vllm", "Qwen/Qwen3-32B", undefined],
    ["hosted_vllm", "openai/gpt-oss-120b", "openai"],
    ["hosted_vllm", "moonshotai/Kimi-K2-Instruct", "kimi"],
    ["hosted_vllm", "deepseek-ai/DeepSeek-V3", "deepseek"],
    ["together_ai", "meta-llama/Llama-3.3-70B-Instruct-Turbo", undefined],
    ["sagemaker", "vertex_ai/claude-sonnet-4", "claude"],
    ["sagemaker", "groq/llama-3.3-70b", undefined],
    // A vendor provider name counts in the family scan, as in the prefixed form.
    ["moonshot", "openai/gpt-4o", "kimi"],
    ["anthropic", "openai/gpt-4o", "claude"],
  ] as const)("resolves %s model %s like its routing-prefixed form", (provider, model, family) => {
    // LiteLLM prepends a differing custom_llm_provider, so both spellings are one deployment.
    const row = { litellm_params: { model, custom_llm_provider: provider } };
    const identity = resolveBackendIdentity(row);
    expect(identity).toEqual({
      provider,
      modelId: model,
      qualifiedId: `${provider}/${model}`,
      ...(family ? { family } : {}),
    });
    expect(identity).toEqual(
      resolveBackendIdentity({ litellm_params: { model: `${provider}/${model}`, custom_llm_provider: provider } }),
    );
    expect(resolveCatalogProvider(row)).toBe(provider);
    expect(routesOnlyThrough(row, new Set(["anthropic"]))).toBe(false);
    expect(routesOnlyThrough(row, new Set(["openai", "azure", "azure_ai"]))).toBe(false);
  });

  it("normalizes the explicit OpenRouter provider without changing its model id", () => {
    expect(
      resolveBackendIdentity({
        litellm_params: { model: "DeepSeek/deepseek-v4.1-flash", custom_llm_provider: " OpEnRoUtEr " },
      }),
    ).toEqual({
      provider: "openrouter",
      modelId: "DeepSeek/deepseek-v4.1-flash",
      qualifiedId: "openrouter/DeepSeek/deepseek-v4.1-flash",
      family: "deepseek",
    });
  });

  it("keeps base_model authoritative over the configured OpenRouter model namespace", () => {
    const row = {
      litellm_params: { model: "moonshotai/kimi-k2.5", custom_llm_provider: "openrouter" },
      model_info: { base_model: "openrouter/openai/gpt-4o" },
    };
    expect(resolveBackendIdentity(row)).toEqual({
      provider: "openrouter",
      modelId: "openai/gpt-4o",
      qualifiedId: "openrouter/openai/gpt-4o",
      family: "openai",
    });
    expect(
      resolveBackendIdentity({ ...row, model_info: { base_model: "anthropic/claude-sonnet-4-6" } }),
    ).toBeUndefined();
  });

  it("does not authorize an OpenRouter model namespace from the public route alone", () => {
    expect(
      resolveBackendIdentity({
        model_name: "anthropic/claude-sonnet-4-6",
        litellm_params: { custom_llm_provider: "openrouter" },
      }),
    ).toBeUndefined();
  });

  it.each(["azure", "azure_ai", "custom_openai", "openai_like"])(
    "keeps stripping the generic %s adapter prefix under OpenRouter",
    (adapter) => {
      const identity = resolveBackendIdentity({
        litellm_params: { model: `${adapter}/gpt-5`, custom_llm_provider: "openrouter" },
      });
      expect(identity).toEqual(
        resolveBackendIdentity({ litellm_params: { model: "gpt-5", custom_llm_provider: "openrouter" } }),
      );
      expect(identity?.modelId).toBe("gpt-5");
    },
  );

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
    expect(LITELLM_DISCOVERY_VERSION).toBe(16);
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
