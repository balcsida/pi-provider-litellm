import { describe, expect, it } from "vitest";
import { assistant, createCompatibilityHarness, user } from "./helpers.js";

describe("native cross-provider handoff compatibility", () => {
  it("replays foreign thinking, tool calls, and tool results into LiteLLM", async () => {
    const { models, model, foreignModel, requests } = await createCompatibilityHarness();
    const message = await models
      .streamSimple(model, {
        messages: [
          user("Start elsewhere"),
          assistant(
            foreignModel,
            [
              { type: "thinking", thinking: "look up 714", thinkingSignature: "reasoning_content" },
              {
                type: "toolCall",
                id: "foreign_call",
                name: "lookup",
                arguments: { value: 714 },
                thoughtSignature: "opaque",
              },
            ],
            "toolUse",
          ),
          {
            role: "toolResult",
            toolCallId: "foreign_call",
            toolName: "lookup",
            content: [{ type: "text", text: "887" }],
            isError: false,
            timestamp: 3,
          },
          user("Continue in LiteLLM"),
        ],
      })
      .result();

    expect(message.content).toEqual([{ type: "text", text: "LiteLLM continued" }]);
    expect(requests[0]?.messages).toEqual([
      { role: "user", content: "Start elsewhere" },
      expect.objectContaining({
        role: "assistant",
        content: "look up 714",
        tool_calls: [
          { id: "foreign_call", type: "function", function: { name: "lookup", arguments: '{"value":714}' } },
        ],
      }),
      { role: "tool", content: "887", tool_call_id: "foreign_call" },
      { role: "user", content: "Continue in LiteLLM" },
    ]);
  });

  it("replays a LiteLLM transcript into another OpenAI-compatible provider", async () => {
    const { models, model, foreignModel, foreignRequests } = await createCompatibilityHarness();
    const message = await models
      .streamSimple(foreignModel, {
        messages: [
          user("Start in LiteLLM"),
          assistant(
            model,
            [
              { type: "thinking", thinking: "call lookup", thinkingSignature: "reasoning_content" },
              { type: "toolCall", id: "litellm_call", name: "lookup", arguments: { value: 887 } },
            ],
            "toolUse",
          ),
          {
            role: "toolResult",
            toolCallId: "litellm_call",
            toolName: "lookup",
            content: [{ type: "text", text: "714" }],
            isError: false,
            timestamp: 3,
          },
          user("Continue elsewhere"),
        ],
      })
      .result();

    expect(message.content).toEqual([{ type: "text", text: "foreign continued" }]);
    expect(foreignRequests[0]?.messages).toEqual([
      { role: "user", content: "Start in LiteLLM" },
      expect.objectContaining({
        role: "assistant",
        content: "call lookup",
        tool_calls: [
          { id: "litellm_call", type: "function", function: { name: "lookup", arguments: '{"value":887}' } },
        ],
      }),
      { role: "tool", content: "714", tool_call_id: "litellm_call" },
      { role: "user", content: "Continue elsewhere" },
    ]);
  });

  it("does not treat a lookalike foreign origin as a foreign request", async () => {
    const { models, foreignModel, requests, foreignRequests } = await createCompatibilityHarness();
    foreignModel.baseUrl = "https://foreign.example.com.attacker.invalid";

    await models.streamSimple(foreignModel, { messages: [user("Continue elsewhere")] }).result();

    expect(requests).toHaveLength(1);
    expect(foreignRequests).toHaveLength(0);
  });
});
