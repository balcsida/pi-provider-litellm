import { isContextOverflow } from "@earendil-works/pi-ai";
import { expect, it } from "vitest";
import { createCompatibilityHarness, user } from "./helpers.js";

it("normalizes LiteLLM context overflow for Pi detection", async () => {
  const { models, model } = await createCompatibilityHarness();

  const message = await models.streamSimple(model, { messages: [user("Overflow the context")] }).result();

  expect(message.stopReason).toBe("error");
  expect(message.errorMessage).toContain("maximum context length");
  expect(isContextOverflow(message)).toBe(true);
});
