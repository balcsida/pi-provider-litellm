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

    const copiesOf = (name: string): Record<string, string> =>
      Object.fromEntries(
        Object.entries(lockfile.packages ?? {})
          .filter(([path]) => path === `node_modules/${name}` || path.endsWith(`/node_modules/${name}`))
          .map(([path, pkg]) => [path, pkg.version ?? "missing"]),
      );

    // basic-ftp left the dependency tree entirely; its override is vestigial.
    expect(Object.values(copiesOf("basic-ftp")).every((version) => version === "6.0.1")).toBe(true);
    const fastXmlBuilderCopies = Object.values(copiesOf("fast-xml-builder"));
    expect(fastXmlBuilderCopies).not.toHaveLength(0);
    expect(fastXmlBuilderCopies.every((version) => version === "1.2.0")).toBe(true);
    // pi-coding-agent publishes its own "overrides" field, so npm isolates its
    // subtree from root overrides (none of plain, @google/genai-scoped, or
    // pi-coding-agent-scoped overrides reach it; verified with npm 11.16).
    // Its nested protobufjs copy therefore stays on the latest 7.x until that
    // is fixed upstream. Any other copy or version drift must fail this test.
    expect(copiesOf("protobufjs")).toEqual({
      "node_modules/protobufjs": "8.7.1",
      "node_modules/@earendil-works/pi-coding-agent/node_modules/protobufjs": "7.6.4",
    });
  });
});
