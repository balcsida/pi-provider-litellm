import { type Context, getSupportedThinkingLevels } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import { anthropicSseChunk, anthropicTextResponse, createCompatibilityHarness, user } from "./helpers.js";

const anthropicRoute = {
  model_name: "claude-opus-5",
  model_info: {
    mode: "chat",
    litellm_provider: "bedrock_converse",
    base_model: "anthropic.claude-opus-5",
    supports_reasoning: true,
    supports_max_reasoning_effort: true,
  },
};

describe("Anthropic Messages wire compatibility", () => {
  it("streams Anthropic SSE text from the exact Messages endpoint", async () => {
    const { models, model, requestUrls, requests, respond } = await createCompatibilityHarness(anthropicRoute);
    respond(...anthropicTextResponse("Hello from Messages"));

    const message = await models.streamSimple(model, { messages: [user("Hello")] }).result();

    expect(model.api).toBe("anthropic-messages");
    expect(requestUrls).toEqual(["https://litellm.example.com/v1/messages"]);
    expect(requestUrls[0]).not.toContain("/v1/v1/messages");
    expect(message.content).toEqual([{ type: "text", text: "Hello from Messages" }]);
    expect(message.usage).toMatchObject({ input: 2, output: 1 });
    expect(requests[0]).toMatchObject({
      model: "claude-opus-5",
      messages: [{ role: "user", content: expect.anything() }],
      stream: true,
    });
  });

  it("serializes max effort, tools, and cache controls with only Anthropic-native fields", async () => {
    const { models, model, requests, respond } = await createCompatibilityHarness(anthropicRoute);
    expect(getSupportedThinkingLevels(model)).toContain("max");
    respond(...anthropicTextResponse("ready"));

    await models
      .streamSimple(
        model,
        {
          systemPrompt: "Keep answers short.",
          messages: [user("Look up the answer")],
          tools: [
            {
              name: "lookup",
              description: "Look up a value",
              parameters: {
                type: "object",
                properties: { key: { type: "string" } },
                required: ["key"],
              },
            },
          ],
        },
        { reasoning: "max" },
      )
      .result();

    expect(requests[0]).toMatchObject({
      system: [expect.objectContaining({ type: "text", cache_control: { type: "ephemeral" } })],
      thinking: { type: "adaptive", display: "summarized" },
      output_config: { effort: "max" },
      tools: [
        expect.objectContaining({
          name: "lookup",
          input_schema: expect.objectContaining({ type: "object" }),
          cache_control: { type: "ephemeral" },
        }),
      ],
    });
    expect(requests[0]?.messages?.[0]?.content).toContainEqual(
      expect.objectContaining({ type: "text", cache_control: { type: "ephemeral" } }),
    );
    expect(requests[0]).not.toHaveProperty("reasoning_effort");
    expect(requests[0]).not.toHaveProperty("reasoning");
    expect(requests[0]).not.toHaveProperty("include");
  });

  it("replays native thinking, tool use, and tool results on the next turn", async () => {
    const { models, model, requests, respond } = await createCompatibilityHarness(anthropicRoute);
    const context: Context = {
      messages: [user("Look up the answer")],
      tools: [
        {
          name: "lookup",
          description: "Look up a value",
          parameters: { type: "object", properties: { key: { type: "string" } } },
        },
      ],
    };
    respond(
      anthropicSseChunk({
        type: "message_start",
        message: {
          id: "msg_tool",
          type: "message",
          role: "assistant",
          content: [],
          model: "claude-opus-5",
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 5, output_tokens: 0 },
        },
      }),
      anthropicSseChunk({
        type: "content_block_start",
        index: 0,
        content_block: { type: "thinking", thinking: "", signature: "" },
      }),
      anthropicSseChunk({
        type: "content_block_delta",
        index: 0,
        delta: { type: "thinking_delta", thinking: "I should use the tool." },
      }),
      anthropicSseChunk({
        type: "content_block_delta",
        index: 0,
        delta: { type: "signature_delta", signature: "signed-thinking" },
      }),
      anthropicSseChunk({ type: "content_block_stop", index: 0 }),
      anthropicSseChunk({
        type: "content_block_start",
        index: 1,
        content_block: { type: "tool_use", id: "tool_1", name: "lookup", input: {} },
      }),
      anthropicSseChunk({
        type: "content_block_delta",
        index: 1,
        delta: { type: "input_json_delta", partial_json: '{"key":"answer"}' },
      }),
      anthropicSseChunk({ type: "content_block_stop", index: 1 }),
      anthropicSseChunk({
        type: "message_delta",
        delta: { stop_reason: "tool_use", stop_sequence: null },
        usage: { output_tokens: 8, output_tokens_details: { thinking_tokens: 3 } },
      }),
      anthropicSseChunk({ type: "message_stop" }),
    );

    const first = await models.streamSimple(model, context, { reasoning: "max" }).result();
    expect(first.content).toEqual([
      { type: "thinking", thinking: "I should use the tool.", thinkingSignature: "signed-thinking" },
      { type: "toolCall", id: "tool_1", name: "lookup", arguments: { key: "answer" } },
    ]);
    expect(first.stopReason).toBe("toolUse");

    context.messages.push(first, {
      role: "toolResult",
      toolCallId: "tool_1",
      toolName: "lookup",
      content: [{ type: "text", text: "42" }],
      isError: false,
      timestamp: 3,
    });
    respond(...anthropicTextResponse("The answer is 42."));

    const second = await models.streamSimple(model, context, { reasoning: "max" }).result();

    expect(second.content).toEqual([{ type: "text", text: "The answer is 42." }]);
    expect(requests[1]?.messages).toEqual([
      expect.objectContaining({ role: "user" }),
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "I should use the tool.", signature: "signed-thinking" },
          { type: "tool_use", id: "tool_1", name: "lookup", input: { key: "answer" } },
        ],
      },
      {
        role: "user",
        content: [
          expect.objectContaining({
            type: "tool_result",
            tool_use_id: "tool_1",
            content: "42",
            is_error: false,
          }),
        ],
      },
    ]);
    expect(requests[1]).not.toHaveProperty("reasoning_effort");
  });
});
