import type { Context } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import { createCompatibilityHarness, RED_CIRCLE_PNG, sseChunk } from "./helpers.js";

const user = (content: string) => ({ role: "user" as const, content, timestamp: 1 });

describe("native provider stream compatibility", () => {
  it("completes two text turns with usage", async () => {
    const { provider, model, respond } = await createCompatibilityHarness();
    const context: Context = { messages: [user("First")] };
    respond(
      sseChunk({ choices: [{ delta: { content: "Hello" }, finish_reason: null }] }),
      sseChunk({ choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 3, completion_tokens: 2 } }),
    );

    const first = await provider.streamSimple(model, context).result();
    expect(first.content).toEqual([{ type: "text", text: "Hello" }]);
    expect(first.usage.input).toBeGreaterThan(0);
    expect(first.usage.output).toBeGreaterThan(0);

    context.messages.push(first, user("Second"));
    respond(
      sseChunk({ choices: [{ delta: { content: "Again" }, finish_reason: null }] }),
      sseChunk({ choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 7, completion_tokens: 1 } }),
    );
    expect((await provider.streamSimple(model, context).result()).content).toEqual([{ type: "text", text: "Again" }]);
  });

  it("emits text start, delta, and end events", async () => {
    const { provider, model, respond } = await createCompatibilityHarness();
    respond(
      sseChunk({ choices: [{ delta: { content: "Hello" }, finish_reason: null }] }),
      sseChunk({ choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 1, completion_tokens: 1 } }),
    );

    const types: string[] = [];
    for await (const event of provider.streamSimple(model, { messages: [user("Hi")] })) types.push(event.type);

    expect(types).toEqual(expect.arrayContaining(["text_start", "text_delta", "text_end"]));
  });

  it("assembles tool-call argument deltas", async () => {
    const { provider, model, respond } = await createCompatibilityHarness();
    respond(
      sseChunk({
        choices: [
          {
            delta: {
              tool_calls: [{ index: 0, id: "call_1", type: "function", function: { name: "add", arguments: '{"a":' } }],
            },
            finish_reason: null,
          },
        ],
      }),
      sseChunk({
        choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: "714}" } }] }, finish_reason: null }],
      }),
      sseChunk({
        choices: [{ delta: {}, finish_reason: "tool_calls" }],
        usage: { prompt_tokens: 2, completion_tokens: 3 },
      }),
    );

    const message = await provider
      .streamSimple(model, {
        messages: [user("Add")],
        tools: [
          { name: "add", description: "Add", parameters: { type: "object", properties: { a: { type: "number" } } } },
        ],
      })
      .result();

    expect(message.content).toContainEqual({ type: "toolCall", id: "call_1", name: "add", arguments: { a: 714 } });
  });

  it("handles thinking and a tool result across turns", async () => {
    const { provider, model, requests, respond } = await createCompatibilityHarness();
    const context: Context = { messages: [user("Calculate")], tools: [] };
    respond(
      sseChunk({ choices: [{ delta: { reasoning_content: "714" }, finish_reason: null }] }),
      sseChunk({
        choices: [
          {
            delta: {
              tool_calls: [{ index: 0, id: "call_2", type: "function", function: { name: "lookup", arguments: "{}" } }],
            },
            finish_reason: "tool_calls",
          },
        ],
        usage: { prompt_tokens: 2, completion_tokens: 2 },
      }),
    );
    const first = await provider.streamSimple(model, context, { reasoning: "high" }).result();
    expect(first.content).toContainEqual(expect.objectContaining({ type: "thinking", thinking: "714" }));

    context.messages.push(first, {
      role: "toolResult",
      toolCallId: "call_2",
      toolName: "lookup",
      content: [{ type: "text", text: "887" }],
      isError: false,
      timestamp: 2,
    });
    respond(
      sseChunk({ choices: [{ delta: { content: "887" }, finish_reason: null }] }),
      sseChunk({ choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 4, completion_tokens: 1 } }),
    );
    const second = await provider.streamSimple(model, context).result();

    expect(second.content).toContainEqual({ type: "text", text: "887" });
    expect(requests.at(-1)?.messages).toContainEqual(expect.objectContaining({ role: "tool", content: "887" }));
  });

  it("serializes image input", async () => {
    const { provider, model, requests, respond } = await createCompatibilityHarness();
    respond(
      sseChunk({
        choices: [{ delta: { content: "red" }, finish_reason: "stop" }],
        usage: { prompt_tokens: 2, completion_tokens: 1 },
      }),
    );

    await provider
      .streamSimple(model, {
        messages: [
          { role: "user", content: [{ type: "image", data: RED_CIRCLE_PNG, mimeType: "image/png" }], timestamp: 1 },
        ],
      })
      .result();

    expect(requests[0]?.messages[0]?.content).toContainEqual({
      type: "image_url",
      image_url: { url: `data:image/png;base64,${RED_CIRCLE_PNG}` },
    });
  });
});
