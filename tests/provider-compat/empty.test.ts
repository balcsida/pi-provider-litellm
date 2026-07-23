import type { Context } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import { assistant, createCompatibilityHarness, successfulResponse, user } from "./helpers.js";

describe("native provider empty-content compatibility", () => {
  it.each([[], "", "   "])("handles empty user content %j", async (content) => {
    const { models, model, requests, respond } = await createCompatibilityHarness();
    respond(...successfulResponse("ok"));

    const message = await models
      .streamSimple(model, { messages: [{ role: "user", content, timestamp: 1 }] } as Context)
      .result();

    expect(message.stopReason).toBe("stop");
    expect(message.content).toEqual([{ type: "text", text: "ok" }]);
    expect(requests[0]?.messages).toEqual(Array.isArray(content) ? [] : [{ role: "user", content }]);
  });

  it("handles an empty assistant message between user turns", async () => {
    const { models, model, requests, respond } = await createCompatibilityHarness();
    respond(...successfulResponse("continued"));

    const message = await models
      .streamSimple(model, { messages: [user("First"), assistant(model, []), user("Continue")] })
      .result();

    expect(message.stopReason).toBe("stop");
    expect(requests[0]?.messages).toEqual([
      expect.objectContaining({ role: "user", content: "First" }),
      expect.objectContaining({ role: "user", content: "Continue" }),
    ]);
  });
});
