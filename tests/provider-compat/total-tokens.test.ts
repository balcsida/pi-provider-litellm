import type { Context } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import { createCompatibilityHarness, sseChunk } from "./helpers.js";

const user = (content: string) => ({ role: "user" as const, content, timestamp: 1 });

describe("native provider total token compatibility", () => {
  it("sets totalTokens to input + output + cacheRead + cacheWrite", async () => {
    const { models, model, respond } = await createCompatibilityHarness();
    const context: Context = { messages: [user("First")] };
    respond(
      sseChunk({ choices: [{ delta: { content: "one" }, finish_reason: null }] }),
      sseChunk({
        choices: [{ delta: {}, finish_reason: "stop" }],
        usage: {
          prompt_tokens: 13,
          completion_tokens: 5,
          prompt_tokens_details: { cached_tokens: 3, cache_write_tokens: 2 },
        },
      }),
    );
    const first = await models.streamSimple(model, context).result();

    expect(first.usage).toMatchObject({ input: 8, output: 5, cacheRead: 3, cacheWrite: 2, totalTokens: 18 });
    expect(first.usage.totalTokens).toBe(
      first.usage.input + first.usage.output + first.usage.cacheRead + first.usage.cacheWrite,
    );

    context.messages.push(first, user("Second"));
    respond(
      sseChunk({ choices: [{ delta: { content: "two" }, finish_reason: null }] }),
      sseChunk({
        choices: [{ delta: {}, finish_reason: "stop" }],
        usage: {
          prompt_tokens: 21,
          completion_tokens: 4,
          prompt_tokens_details: { cached_tokens: 7, cache_write_tokens: 1 },
        },
      }),
    );
    const second = await models.streamSimple(model, context).result();

    expect(second.usage).toMatchObject({ input: 13, output: 4, cacheRead: 7, cacheWrite: 1, totalTokens: 25 });
    expect(second.usage.totalTokens).toBe(
      second.usage.input + second.usage.output + second.usage.cacheRead + second.usage.cacheWrite,
    );
  });
});
