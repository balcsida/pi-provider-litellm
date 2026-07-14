import type { ProviderModelConfig } from "@earendil-works/pi-coding-agent";
import { describe, expect, it } from "vitest";
import { setupLiteLLMCostTracking } from "../src/cost.js";

type Handler = (event: any, ctx?: any) => any;

function createPi() {
  const handlers = new Map<string, Handler[]>();
  return {
    handlers,
    on(event: string, handler: Handler): void {
      handlers.set(event, [...(handlers.get(event) ?? []), handler]);
    },
  };
}

function model(id: string, cost: ProviderModelConfig["cost"]): ProviderModelConfig {
  return {
    id,
    name: id,
    reasoning: false,
    input: ["text"],
    contextWindow: 128_000,
    maxTokens: 4096,
    cost,
  };
}

describe("setupLiteLLMCostTracking", () => {
  it("keeps alias model costs separate from the default provider for the same model id", async () => {
    const pi = createPi();
    setupLiteLLMCostTracking(pi as any, [
      { provider: "litellm", models: [model("shared-model", { input: 3, output: 15, cacheRead: 0, cacheWrite: 0 })] },
      {
        provider: "litellm-anthropic",
        models: [model("shared-model", { input: 10, output: 20, cacheRead: 0, cacheWrite: 0 })],
      },
    ]);

    const endHandler = pi.handlers.get("message_end")?.[0];
    const defaultResult = await endHandler?.({
      message: { role: "assistant", provider: "litellm", model: "shared-model", usage: { input: 100, output: 50 } },
    });
    const aliasResult = await endHandler?.({
      message: {
        role: "assistant",
        provider: "litellm-anthropic",
        model: "shared-model",
        usage: { input: 100, output: 50 },
      },
    });

    expect(defaultResult.message.usage.cost.total).toBeCloseTo(0.00105, 10);
    expect(aliasResult.message.usage.cost.total).toBeCloseTo(0.002, 10);
  });

  it("applies LiteLLM response-cost headers to alias provider messages", async () => {
    const pi = createPi();
    setupLiteLLMCostTracking(pi as any, [{ provider: "litellm-anthropic", models: [] }]);

    const responseHandler = pi.handlers.get("after_provider_response")?.[0];
    responseHandler?.(
      { headers: { "x-litellm-response-cost": "0.42" } },
      { model: { provider: "litellm-anthropic", id: "claude-sonnet" } },
    );

    const endHandler = pi.handlers.get("message_end")?.[0];
    const result = await endHandler?.({
      message: {
        role: "assistant",
        provider: "litellm-anthropic",
        model: "claude-sonnet",
        usage: { input: 100, output: 50 },
      },
    });

    expect(result.message.usage.cost.total).toBe(0.42);
  });

  it("does not let one provider's headerless response clear another provider's pending cost", async () => {
    const pi = createPi();
    setupLiteLLMCostTracking(pi as any, [
      { provider: "litellm", models: [model("gpt-5", { input: 3, output: 15, cacheRead: 0, cacheWrite: 0 })] },
      {
        provider: "litellm-anthropic",
        models: [model("claude-sonnet", { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 })],
      },
    ]);

    const responseHandler = pi.handlers.get("after_provider_response")?.[0];
    // Default provider's response carries an accurate cost header.
    responseHandler?.(
      { headers: { "x-litellm-response-cost": "0.42" } },
      { model: { provider: "litellm", id: "gpt-5" } },
    );
    // Alias provider's response arrives before the default's message_end and has no cost header.
    responseHandler?.({ headers: {} }, { model: { provider: "litellm-anthropic", id: "claude-sonnet" } });

    const endHandler = pi.handlers.get("message_end")?.[0];
    const result = await endHandler?.({
      message: { role: "assistant", provider: "litellm", model: "gpt-5", usage: { input: 100, output: 50 } },
    });

    expect(result.message.usage.cost.total).toBe(0.42);
  });
});
