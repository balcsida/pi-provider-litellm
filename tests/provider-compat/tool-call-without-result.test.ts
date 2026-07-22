import { expect, it } from "vitest";
import { assistant, createCompatibilityHarness, successfulResponse, user } from "./helpers.js";

it("drops an assistant tool call with no matching result", async () => {
  const { models, model, requests, respond } = await createCompatibilityHarness();
  respond(...successfulResponse("continued"));

  const message = await models
    .streamSimple(model, {
      messages: [
        user("Use the tool"),
        assistant(model, [{ type: "toolCall", id: "call_1", name: "lookup", arguments: {} }], "error"),
        user("Continue without it"),
      ],
    })
    .result();

  expect(message.stopReason).toBe("stop");
  expect(requests[0]?.messages).toEqual([
    expect.objectContaining({ role: "user", content: "Use the tool" }),
    expect.objectContaining({ role: "user", content: "Continue without it" }),
  ]);
});
