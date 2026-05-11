import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { discoverModels } from "./discover.js";
import { resolveCredentials } from "./litellm.js";

export interface ModelCostInfo {
  inputCostPerToken: number;
  outputCostPerToken: number;
  cacheReadCostPerToken: number;
  cacheWriteCostPerToken: number;
}

export function setupLiteLLMCostTracking(pi: ExtensionAPI): void {
  const modelCosts = new Map<string, ModelCostInfo>();
  let lastResponseCost: number | null = null;

  pi.on("session_start", async (_event, ctx) => {
    try {
      const config = await resolveCredentials();
      if (!config.baseUrl || !config.apiKey) {
        ctx.ui?.notify?.("LiteLLM cost extension: no credentials configured", "warning");
        return;
      }

      const response = await fetch(`${config.baseUrl}/model/info`, {
        headers: {
          Authorization: `Bearer ${config.apiKey}`,
          Accept: "application/json",
        },
      });

      if (!response.ok) {
        ctx.ui?.notify?.(`LiteLLM cost extension: /model/info returned ${response.status}`, "warning");
        return;
      }

      const payload = (await response.json()) as {
        data: Array<{
          model_name: string;
          litellm_params?: { model?: string };
          model_info?: {
            input_cost_per_token?: number;
            output_cost_per_token?: number;
            cache_read_input_token_cost?: number;
            cache_creation_input_token_cost?: number;
          };
        }>;
      };

      for (const entry of payload.data) {
        const info = entry.model_info;
        if (!info) continue;

        const costInfo: ModelCostInfo = {
          inputCostPerToken: info.input_cost_per_token ?? 0,
          outputCostPerToken: info.output_cost_per_token ?? 0,
          cacheReadCostPerToken: info.cache_read_input_token_cost ?? 0,
          cacheWriteCostPerToken: info.cache_creation_input_token_cost ?? 0,
        };

        modelCosts.set(entry.model_name, costInfo);
        if (entry.litellm_params?.model) {
          modelCosts.set(entry.litellm_params.model, costInfo);
        }
      }

      if (modelCosts.size > 0) {
        ctx.ui?.notify?.(`LiteLLM cost: loaded pricing for ${modelCosts.size} model(s)`, "info");
      }
    } catch (error) {
      ctx.ui?.notify?.(
        `LiteLLM cost extension: could not fetch /model/info (${error instanceof Error ? error.message : String(error)})`,
        "warning",
      );
    }
  });

  pi.on("after_provider_response", (event) => {
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
    if (event.message.role !== "assistant") return;

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
}
