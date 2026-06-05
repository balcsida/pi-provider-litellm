import { readFile } from "node:fs/promises";
import { describe, expect, it } from "vitest";

describe("package gallery metadata", () => {
  it("uses the gallery image URL expected by pi.dev", async () => {
    const { default: manifest } = await import("../package.json", {
      with: { type: "json" },
    });

    expect(manifest.pi.image).toBe(
      "https://raw.githubusercontent.com/balcsida/pi-provider-litellm/refs/heads/main/assets/pi_litellm_gallery.png",
    );
  });

  it("does not expose the npm badge as gallery media", async () => {
    const readme = await readFile("README.md", "utf8");

    expect(readme).not.toContain("https://img.shields.io/npm/v/pi-provider-litellm.svg");
  });
});

describe("pi package compatibility", () => {
  it("accepts every pi package version through peer dependencies", async () => {
    const { default: manifest } = await import("../package.json", {
      with: { type: "json" },
    });

    expect(manifest.peerDependencies["@earendil-works/pi-ai"]).toBe("*");
    expect(manifest.peerDependencies["@earendil-works/pi-coding-agent"]).toBe("*");
  });
});

describe("dependency security overrides", () => {
  it("keeps vulnerable transitive dependencies above alerted ranges", async () => {
    const lockfile = JSON.parse(await readFile("package-lock.json", "utf8")) as {
      packages?: Record<string, { version?: string }>;
    };

    expect([undefined, "6.0.1"]).toContain(lockfile.packages?.["node_modules/basic-ftp"]?.version);
    expect(lockfile.packages?.["node_modules/protobufjs"]?.version).toBe("8.2.1");
    expect(lockfile.packages?.["node_modules/fast-xml-builder"]?.version).toBe("1.2.0");
  });
});
