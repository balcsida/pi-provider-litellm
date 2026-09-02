import type { Context } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import { anthropicSseChunk, claudeRoute, createCompatibilityHarness, sseChunk } from "./helpers.js";

const user = (content: string) => ({ role: "user" as const, content, timestamp: 1 });

describe("native provider abort compatibility", () => {
  it("aborts mid-stream and completes a later request", async () => {
    const { models, model, respond } = await createCompatibilityHarness();
    const controller = new AbortController();
    respond(
      sseChunk({ choices: [{ delta: { content: "partial" }, finish_reason: null }] }),
      sseChunk({ choices: [{ delta: { content: " ignored" }, finish_reason: null }] }, true),
      sseChunk({ choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 2, completion_tokens: 2 } }),
    );

    const stream = models.streamSimple(model, { messages: [user("First")] }, { signal: controller.signal });
    for await (const event of stream) {
      if (event.type === "text_delta") controller.abort();
    }
    const aborted = await stream.result();

    expect(aborted.stopReason).toBe("aborted");
    expect(aborted.content).toEqual([{ type: "text", text: "partial" }]);

    respond(
      sseChunk({ choices: [{ delta: { content: "recovered" }, finish_reason: null }] }),
      sseChunk({ choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 1, completion_tokens: 1 } }),
    );
    const recovered = await models.streamSimple(model, { messages: [user("Second")] }).result();
    expect(recovered.stopReason).toBe("stop");
    expect(recovered.content).toEqual([{ type: "text", text: "recovered" }]);
  });

  it("aborts a Messages stream without leaking a later chunk", async () => {
    const { models, model, respond } = await createCompatibilityHarness(
      claudeRoute("anthropic", "anthropic/claude-sonnet-4-6"),
    );
    const controller = new AbortController();
    respond(
      anthropicSseChunk({
        type: "message_start",
        message: {
          id: "msg_abort",
          type: "message",
          role: "assistant",
          content: [],
          model: "team-claude",
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 1, output_tokens: 0 },
        },
      }),
      anthropicSseChunk({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
      anthropicSseChunk({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: "partial" } }),
      anthropicSseChunk(
        { type: "content_block_delta", index: 0, delta: { type: "text_delta", text: " ignored" } },
        true,
      ),
      anthropicSseChunk({ type: "message_stop" }),
    );

    const stream = models.streamSimple(model, { messages: [user("First")] }, { signal: controller.signal });
    for await (const event of stream) {
      if (event.type === "text_delta") controller.abort();
    }
    const aborted = await stream.result();

    expect(model.api).toBe("anthropic-messages");
    expect(aborted.stopReason).toBe("aborted");
    expect(aborted.content).toEqual([{ type: "text", text: "partial" }]);
  });

  it.each([
    ["chunked", false],
    ["raw", true],
  ] as const)("handles an already-aborted signal for a %s response", async (_name, raw) => {
    const { provider, model, respondRaw } = await createCompatibilityHarness();
    if (raw) respondRaw("data: [DONE]\n\n");
    const signal = AbortSignal.abort();

    const message = await provider
      .streamSimple(model, { messages: [user("Stop")] }, { apiKey: "sk-test", signal })
      .result();

    expect(message.stopReason).toBe("aborted");
    expect(message.content).toEqual([]);
    expect(message.usage.totalTokens).toBe(0);
  });

  it("keeps an empty aborted assistant in a successful follow-up context", async () => {
    const { models, model, requests, respond } = await createCompatibilityHarness();
    const context: Context = { messages: [user("Stop")] };
    const aborted = await models.streamSimple(model, context, { signal: AbortSignal.abort() }).result();
    context.messages.push(aborted, user("Continue"));
    respond(
      sseChunk({ choices: [{ delta: { content: "continued" }, finish_reason: null }] }),
      sseChunk({ choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 2, completion_tokens: 1 } }),
    );

    const message = await models.streamSimple(model, context).result();

    expect(message.content).toEqual([{ type: "text", text: "continued" }]);
    expect(requests.at(-1)?.messages).toEqual([
      expect.objectContaining({ role: "user", content: "Stop" }),
      expect.objectContaining({ role: "user", content: "Continue" }),
    ]);
  });
});
