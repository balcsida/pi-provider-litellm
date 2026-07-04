import type { ExtensionAPI, ProviderModelConfig } from "@earendil-works/pi-coding-agent";

const PROVIDER_NAME = "litellm";

export interface ModelCostInfo {
  inputCostPerToken: number;
  outputCostPerToken: number;
  cacheReadCostPerToken: number;
  cacheWriteCostPerToken: number;
}

export function setupLiteLLMCostTracking(
  pi: ExtensionAPI,
  models: ProviderModelConfig[],
): (models: ProviderModelConfig[]) => void {
  const modelCosts = new Map<string, ModelCostInfo>();
  let lastResponseCost: number | null = null;

  // pi reports model costs per million tokens; cost lookups multiply by raw token counts.
  const updateCosts = (newModels: ProviderModelConfig[]): void => {
    modelCosts.clear();
    for (const model of newModels) {
      if (!model.cost) continue;
      modelCosts.set(model.id, {
        inputCostPerToken: (model.cost.input ?? 0) / 1_000_000,
        outputCostPerToken: (model.cost.output ?? 0) / 1_000_000,
        cacheReadCostPerToken: (model.cost.cacheRead ?? 0) / 1_000_000,
        cacheWriteCostPerToken: (model.cost.cacheWrite ?? 0) / 1_000_000,
      });
    }
  };

  updateCosts(models);

  pi.on("after_provider_response", (event, ctx) => {
    // Global Pi hook: responses from other providers must not feed LiteLLM cost state.
    if (ctx.model?.provider !== PROVIDER_NAME) return;
    const costHeader = event.headers?.["x-litellm-response-cost"] ?? event.headers?.["X-Litellm-Response-Cost"];
    if (costHeader) {
      const cost = Number.parseFloat(String(costHeader));
      if (!Number.isNaN(cost)) {
        lastResponseCost = cost;
        return;
      }
    }
    lastResponseCost = null;
  });

  pi.on("message_end", async (event) => {
    if (event.message.role !== "assistant" || event.message.provider !== PROVIDER_NAME) return;

    const usage = event.message.usage;
    if (!usage) return;

    let totalCost: number | null = null;
    if (lastResponseCost !== null) {
      totalCost = lastResponseCost;
      lastResponseCost = null;
    }

    if (totalCost === null) {
      const modelId = event.message.model;
      const costInfo = modelId ? modelCosts.get(modelId) : undefined;
      if (costInfo) {
        const inputCost = costInfo.inputCostPerToken * usage.input;
        const outputCost = costInfo.outputCostPerToken * usage.output;
        const cacheReadCost = costInfo.cacheReadCostPerToken * (usage.cacheRead ?? 0);
        const cacheWriteCost = costInfo.cacheWriteCostPerToken * (usage.cacheWrite ?? 0);
        totalCost = inputCost + outputCost + cacheReadCost + cacheWriteCost;
        return {
          message: {
            ...event.message,
            usage: {
              ...usage,
              cost: {
                input: inputCost,
                output: outputCost,
                cacheRead: cacheReadCost,
                cacheWrite: cacheWriteCost,
                total: totalCost,
              },
            },
          },
        };
      }
    }

    if (totalCost !== null) {
      const totalTokens = usage.input + usage.output + (usage.cacheRead ?? 0) + (usage.cacheWrite ?? 0);
      if (totalTokens > 0) {
        const inputFraction = usage.input / totalTokens;
        const outputFraction = usage.output / totalTokens;
        const cacheReadFraction = (usage.cacheRead ?? 0) / totalTokens;
        const cacheWriteFraction = (usage.cacheWrite ?? 0) / totalTokens;

        return {
          message: {
            ...event.message,
            usage: {
              ...usage,
              cost: {
                input: totalCost * inputFraction,
                output: totalCost * outputFraction,
                cacheRead: totalCost * cacheReadFraction,
                cacheWrite: totalCost * cacheWriteFraction,
                total: totalCost,
              },
            },
          },
        };
      }

      return {
        message: {
          ...event.message,
          usage: {
            ...usage,
            cost: {
              input: 0,
              output: 0,
              cacheRead: 0,
              cacheWrite: 0,
              total: totalCost,
            },
          },
        },
      };
    }

    return;
  });

  return updateCosts;
}
