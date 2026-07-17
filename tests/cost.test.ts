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

describe("setupLiteLLMCostTracking", () => {
  it("preserves Pi's precomputed cost when LiteLLM omits its response-cost header", async () => {
    const pi = createPi();
    setupLiteLLMCostTracking(pi as any, ["litellm"]);
    const piCost = { input: 0.001, output: 0.001, cacheRead: 0, cacheWrite: 0, total: 0.002 };

    const result = await pi.handlers.get("message_end")?.[0]?.({
      message: {
        role: "assistant",
        provider: "litellm",
        model: "gpt-5",
        usage: { input: 100, output: 50, cost: piCost },
      },
    });

    expect(result).toBeUndefined();
  });

  it("applies LiteLLM response-cost headers to alias provider messages", async () => {
    const pi = createPi();
    setupLiteLLMCostTracking(pi as any, ["litellm-anthropic"]);

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
    setupLiteLLMCostTracking(pi as any, ["litellm", "litellm-anthropic"]);

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
