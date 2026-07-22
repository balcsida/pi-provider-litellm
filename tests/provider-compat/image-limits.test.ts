import { expect, it } from "vitest";
import { createCompatibilityHarness, RED_CIRCLE_PNG, sseChunk } from "./helpers.js";

it("preserves multiple image inputs within the model contract", async () => {
  const { provider, model, requests, respond } = await createCompatibilityHarness();
  respond(
    sseChunk({
      choices: [{ delta: { content: "two" }, finish_reason: "stop" }],
      usage: { prompt_tokens: 3, completion_tokens: 1 },
    }),
  );

  await provider
    .streamSimple(model, {
      messages: [
        {
          role: "user",
          content: [
            { type: "image", data: RED_CIRCLE_PNG, mimeType: "image/png" },
            { type: "image", data: RED_CIRCLE_PNG, mimeType: "image/png" },
          ],
          timestamp: 1,
        },
      ],
    })
    .result();

  const content = requests[0]?.messages[0]?.content;
  expect(Array.isArray(content) ? content.filter((part) => part.type === "image_url") : []).toHaveLength(2);
});
