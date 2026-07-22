import { describe, expect, it } from "vitest";
import { createCompatibilityHarness, sseChunk } from "./helpers.js";

describe("native provider token compatibility", () => {
  it("reports the final available token statistics on abort", async () => {
    const { models, model, respond } = await createCompatibilityHarness();
    const controller = new AbortController();
    respond(
      sseChunk({ choices: [{ delta: { content: "partial" }, finish_reason: null }] }),
      sseChunk(
        {
          choices: [{ delta: {}, finish_reason: "stop" }],
          usage: { prompt_tokens: 11, completion_tokens: 5 },
        },
        true,
      ),
    );

    const stream = models.streamSimple(
      model,
      { messages: [{ role: "user", content: "Count", timestamp: 1 }] },
      { signal: controller.signal },
    );
    for await (const event of stream) {
      if (event.type === "text_delta") controller.abort();
    }
    const message = await stream.result();

    expect(message.stopReason).toBe("aborted");
    expect(message.usage).toMatchObject({
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    });
  });
});
