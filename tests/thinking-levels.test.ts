import { describe, expect, it } from "vitest";
import { intersectThinkingLevelMaps, THINKING_LEVEL_DEFINITIONS } from "../src/thinking-levels.js";

describe("intersectThinkingLevelMaps", () => {
  it("keeps only unanimously identical values and explicitly denies disagreement or absence", () => {
    expect(
      intersectThinkingLevelMaps([
        { off: "none", low: "low", high: "high", max: null },
        { off: "none", low: null, high: "high", xhigh: "xhigh" },
      ]),
    ).toEqual({ off: "none", low: null, high: "high", xhigh: null, max: null });
  });

  it("preserves the all-absent distinction and covers every supported thinking level", () => {
    expect(intersectThinkingLevelMaps([undefined, undefined])).toBeUndefined();
    expect(THINKING_LEVEL_DEFINITIONS.map(([level]) => level)).toEqual([
      "off",
      "minimal",
      "low",
      "medium",
      "high",
      "xhigh",
      "max",
    ]);
  });
});
