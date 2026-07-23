import { expect, it } from "vitest";
import { assistant, createCompatibilityHarness, successfulResponse, user } from "./helpers.js";

it("keeps orphan tool-call history well formed", async () => {
  const { models, model, requests, respond } = await createCompatibilityHarness();
  respond(...successfulResponse("continued"));

  const message = await models
    .streamSimple(model, {
      messages: [
        user("Use the tool"),
        assistant(model, [{ type: "toolCall", id: "call_1", name: "lookup", arguments: {} }], "toolUse"),
        user("Continue without it"),
      ],
    })
    .result();

  expect(message.stopReason).toBe("stop");
  expect(requests[0]?.messages).toEqual([
    { role: "user", content: "Use the tool" },
    {
      role: "assistant",
      content: null,
      tool_calls: [
        {
          id: "call_1",
          type: "function",
          function: { name: "lookup", arguments: "{}" },
        },
      ],
    },
    { role: "tool", content: "No result provided", tool_call_id: "call_1" },
    { role: "user", content: "Continue without it" },
  ]);
});
