import { expect, it } from "vitest";
import { assistant, createCompatibilityHarness, successfulResponse, user } from "./helpers.js";

it.each(["emoji 😀", "LinkedIn 🚀🎉", "broken \uD83D"])("serializes tool result %s", async (text) => {
  const { models, model, requests, respond } = await createCompatibilityHarness();
  respond(...successfulResponse("continued"));

  const message = await models
    .streamSimple(model, {
      messages: [
        user("Use the tool"),
        assistant(model, [{ type: "toolCall", id: "call_1", name: "lookup", arguments: {} }], "toolUse"),
        {
          role: "toolResult",
          toolCallId: "call_1",
          toolName: "lookup",
          content: [{ type: "text", text }],
          isError: false,
          timestamp: 3,
        },
        user("Continue"),
      ],
    })
    .result();

  expect(message.stopReason).toBe("stop");
  expect(requests[0]?.messages).toContainEqual(expect.objectContaining({ role: "tool" }));
});
