import { type Context, getSupportedThinkingLevels } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import { createCompatibilityHarness, RED_CIRCLE_PNG, sseChunk, successfulResponse } from "./helpers.js";

const user = (content: string) => ({ role: "user" as const, content, timestamp: 1 });

describe("native provider stream compatibility", () => {
  it("completes two text turns with usage", async () => {
    const { models, model, respond } = await createCompatibilityHarness();
    const context: Context = { messages: [user("First")] };
    respond(
      sseChunk({ choices: [{ delta: { content: "Hello" }, finish_reason: null }] }),
      sseChunk({ choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 3, completion_tokens: 2 } }),
    );

    const first = await models.streamSimple(model, context).result();
    expect(first.content).toEqual([{ type: "text", text: "Hello" }]);
    expect(first.usage.input).toBeGreaterThan(0);
    expect(first.usage.output).toBeGreaterThan(0);

    context.messages.push(first, user("Second"));
    respond(
      sseChunk({ choices: [{ delta: { content: "Again" }, finish_reason: null }] }),
      sseChunk({ choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 7, completion_tokens: 1 } }),
    );
    const second = await models.streamSimple(model, context).result();
    expect(second.content).toEqual([{ type: "text", text: "Again" }]);
    expect(second.usage.input).toBeGreaterThan(0);
    expect(second.usage.output).toBeGreaterThan(0);
  });

  it("emits text start, delta, and end events", async () => {
    const { models, model, respond } = await createCompatibilityHarness();
    respond(
      sseChunk({ choices: [{ delta: { content: "Hello" }, finish_reason: null }] }),
      sseChunk({ choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 1, completion_tokens: 1 } }),
    );

    const events: Array<{ type: string; delta?: string }> = [];
    for await (const event of models.streamSimple(model, { messages: [user("Hi")] })) {
      if (event.type === "text_delta") events.push({ type: event.type, delta: event.delta });
      else if (event.type === "text_start" || event.type === "text_end") events.push({ type: event.type });
    }

    expect(events).toEqual([{ type: "text_start" }, { type: "text_delta", delta: "Hello" }, { type: "text_end" }]);
  });

  it("assembles tool-call argument deltas", async () => {
    const { models, model, respond } = await createCompatibilityHarness();
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

    const message = await models
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
    const { models, model, requests, respond } = await createCompatibilityHarness();
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
    const first = await models.streamSimple(model, context, { reasoning: "high" }).result();
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
    const second = await models.streamSimple(model, context).result();

    expect(second.content).toContainEqual({ type: "text", text: "887" });
    expect(requests.at(-1)?.messages).toContainEqual(expect.objectContaining({ role: "tool", content: "887" }));
  });

  it("serializes Chat reasoning with tools using only Chat fields", async () => {
    const { models, model, requests, respond } = await createCompatibilityHarness({
      model_name: "gpt-5.5",
      model_info: { mode: "chat" },
    });
    respond(...successfulResponse("done"));

    await models
      .streamSimple(
        model,
        {
          messages: [user("Use a tool")],
          tools: [{ name: "noop", description: "No operation", parameters: { type: "object" } }],
        },
        { reasoning: "high" },
      )
      .result();

    expect(requests[0]).toMatchObject({
      reasoning_effort: "high",
      tools: [expect.objectContaining({ type: "function" })],
    });
    expect(requests[0]).not.toHaveProperty("reasoning");
    expect(requests[0]).not.toHaveProperty("include");
    expect(requests[0]).not.toHaveProperty("thinking");
  });

  it("serializes Responses reasoning with tools using only Responses fields", async () => {
    const { models, model, requests, respond } = await createCompatibilityHarness({
      model_name: "gpt-5.5",
      model_info: { mode: "responses" },
    });
    respond(
      sseChunk({ type: "response.created", response: { id: "resp_1" } }),
      sseChunk({
        type: "response.completed",
        response: {
          id: "resp_1",
          status: "completed",
          output: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            total_tokens: 2,
            input_tokens_details: { cached_tokens: 0 },
            output_tokens_details: { reasoning_tokens: 0 },
          },
        },
      }),
    );

    await models
      .streamSimple(
        model,
        {
          messages: [user("Use a tool")],
          tools: [{ name: "noop", description: "No operation", parameters: { type: "object" } }],
        },
        { reasoning: "high" },
      )
      .result();

    expect(requests[0]).toMatchObject({
      reasoning: { effort: "high", summary: "auto" },
      include: ["reasoning.encrypted_content"],
      tools: [expect.objectContaining({ type: "function" })],
    });
    expect(requests[0]).not.toHaveProperty("reasoning_effort");
    expect(requests[0]).not.toHaveProperty("thinking");
  });

  it("serializes boolean Kimi reasoning selections as LiteLLM thinking objects", async () => {
    const { models, model, requests, respond } = await createCompatibilityHarness({ model_name: "kimi-k2.6" });
    expect(getSupportedThinkingLevels(model)).toEqual(["off", "high"]);

    respond(...successfulResponse("enabled"));
    await models.streamSimple(model, { messages: [user("Think")] }, { reasoning: "high" }).result();

    expect(requests[0]).toMatchObject({
      thinking: { type: "enabled" },
    });
    expect(requests[0]).not.toHaveProperty("reasoning_effort");

    respond(...successfulResponse("disabled"));
    await models.streamSimple(model, { messages: [user("Do not think")] }).result();

    expect(requests[1]).toMatchObject({
      thinking: { type: "disabled" },
    });
    expect(requests[1]).not.toHaveProperty("reasoning_effort");
  });

  it("serializes catalog-resolved Kimi route aliases as boolean thinking", async () => {
    const { models, model, requests, respond } = await createCompatibilityHarness({
      model_name: "custom-route/kimi-k2.6",
      model_info: { mode: null },
    });
    expect(getSupportedThinkingLevels(model)).toEqual(["off", "high"]);

    respond(...successfulResponse("enabled"));
    await models.streamSimple(model, { messages: [user("Think")] }, { reasoning: "high" }).result();

    expect(requests[0]).toMatchObject({ thinking: { type: "enabled" } });
    expect(requests[0]).not.toHaveProperty("reasoning_effort");
  });

  it("serializes granular DeepSeek reasoning selections through LiteLLM-safe fields", async () => {
    const { models, model, requests, respond } = await createCompatibilityHarness({
      model_name: "deepseek/deepseek-v4-flash",
    });
    expect(getSupportedThinkingLevels(model)).toEqual(["off", "high", "max"]);

    respond(...successfulResponse("deepseek"));

    await models.streamSimple(model, { messages: [user("Think deeply")] }, { reasoning: "max" }).result();

    expect(requests[0]).toMatchObject({
      thinking: { type: "enabled" },
      reasoning_effort: "max",
    });
  });

  it("keeps unknown reasoning routes usable without speculative thinking controls", async () => {
    const { models, model, requests, respond } = await createCompatibilityHarness({
      model_name: "custom-reasoning-smoke",
    });
    expect(getSupportedThinkingLevels(model)).toEqual(["off"]);

    respond(...successfulResponse("custom"));
    await models.streamSimple(model, { messages: [user("Use the custom route")] }).result();

    expect(requests[0]).not.toHaveProperty("thinking");
    expect(requests[0]).not.toHaveProperty("reasoning_effort");
    expect(requests[0]).not.toHaveProperty("reasoning");
  });

  it("serializes image input", async () => {
    const { models, model, requests, respond } = await createCompatibilityHarness();
    respond(
      sseChunk({
        choices: [{ delta: { content: "red" }, finish_reason: "stop" }],
        usage: { prompt_tokens: 2, completion_tokens: 1 },
      }),
    );

    await models
      .streamSimple(model, {
        messages: [
          { role: "user", content: [{ type: "image", data: RED_CIRCLE_PNG, mimeType: "image/png" }], timestamp: 1 },
        ],
      })
      .result();

    expect(requests[0]?.messages?.[0]?.content).toContainEqual({
      type: "image_url",
      image_url: { url: `data:image/png;base64,${RED_CIRCLE_PNG}` },
    });
  });
});
