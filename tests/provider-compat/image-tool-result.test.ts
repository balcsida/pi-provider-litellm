import { describe, expect, it } from "vitest";
import { assistant, createCompatibilityHarness, RED_CIRCLE_PNG, user } from "./helpers.js";

describe("native provider image tool-result compatibility", () => {
  it("handles an image-only tool result", async () => {
    const { models, model, requests } = await createCompatibilityHarness();
    const message = await models
      .streamSimple(model, {
        messages: [
          user("Inspect the image"),
          assistant(model, [{ type: "toolCall", id: "call_image", name: "inspect", arguments: {} }], "toolUse"),
          {
            role: "toolResult",
            toolCallId: "call_image",
            toolName: "inspect",
            content: [{ type: "image", data: RED_CIRCLE_PNG, mimeType: "image/png" }],
            isError: false,
            timestamp: 3,
          },
        ],
      })
      .result();

    expect(message.content).toEqual([{ type: "text", text: "red circle" }]);
    expect(requests[0]?.messages).toContainEqual({
      role: "user",
      content: [
        { type: "text", text: "Attached image(s) from tool result:" },
        { type: "image_url", image_url: { url: `data:image/png;base64,${RED_CIRCLE_PNG}` } },
      ],
    });
  });

  it("handles a text-and-image tool result", async () => {
    const { models, model, requests } = await createCompatibilityHarness();
    const message = await models
      .streamSimple(model, {
        messages: [
          user("Measure the image"),
          assistant(model, [{ type: "toolCall", id: "call_measure", name: "measure", arguments: {} }], "toolUse"),
          {
            role: "toolResult",
            toolCallId: "call_measure",
            toolName: "measure",
            content: [
              { type: "text", text: "diameter: 2 px" },
              { type: "image", data: RED_CIRCLE_PNG, mimeType: "image/png" },
            ],
            isError: false,
            timestamp: 3,
          },
        ],
      })
      .result();

    expect(message.content).toEqual([{ type: "text", text: "diameter 2 px" }]);
    expect(requests[0]?.messages).toContainEqual({
      role: "tool",
      content: "diameter: 2 px",
      tool_call_id: "call_measure",
    });
    expect(requests[0]?.messages).toContainEqual({
      role: "user",
      content: [
        { type: "text", text: "Attached image(s) from tool result:" },
        { type: "image_url", image_url: { url: `data:image/png;base64,${RED_CIRCLE_PNG}` } },
      ],
    });
  });
});
