import { describe, expect, it } from "vitest";
import { parsePushTriggers } from "./workflow-triggers.js";

describe("parsePushTriggers", () => {
  it("parses block tag filters", () => {
    expect(
      parsePushTriggers(`name: Release
on:
  push:
    tags:
      - 'v*.*.*'
permissions:
  contents: write
`),
    ).toEqual({ branches: [], tags: ["v*.*.*"] });
  });

  it("parses inline branch and tag filters", () => {
    expect(
      parsePushTriggers(`on:
  push:
    branches: [main, release]
    tags: ['v*', "rc-*"]
jobs: {}
`),
    ).toEqual({ branches: ["main", "release"], tags: ["v*", "rc-*"] });
  });

  it("reports branch-triggered workflows instead of accepting an unrelated tags key", () => {
    expect(
      parsePushTriggers(`on:
  push:
    branches:
      - main
jobs:
  test:
    tags:
      - 'v*.*.*'
`),
    ).toEqual({ branches: ["main"], tags: [] });
  });
});
