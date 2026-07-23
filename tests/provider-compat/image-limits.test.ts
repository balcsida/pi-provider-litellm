import { expect, it } from "vitest";
import { createCompatibilityHarness, RED_CIRCLE_PNG, SECOND_PIXEL_PNG, sseChunk } from "./helpers.js";

it("preserves multiple image inputs within the model contract", async () => {
  const { models, model, requests, respond } = await createCompatibilityHarness();
  respond(
    sseChunk({
      choices: [{ delta: { content: "two" }, finish_reason: "stop" }],
      usage: { prompt_tokens: 3, completion_tokens: 1 },
    }),
  );

  await models
    .streamSimple(model, {
      messages: [
        {
          role: "user",
          content: [
            { type: "image", data: RED_CIRCLE_PNG, mimeType: "image/png" },
            { type: "image", data: SECOND_PIXEL_PNG, mimeType: "image/png" },
          ],
          timestamp: 1,
        },
      ],
    })
    .result();

  expect(requests[0]?.messages[0]?.content).toEqual([
    { type: "image_url", image_url: { url: `data:image/png;base64,${RED_CIRCLE_PNG}` } },
    { type: "image_url", image_url: { url: `data:image/png;base64,${SECOND_PIXEL_PNG}` } },
  ]);
});
