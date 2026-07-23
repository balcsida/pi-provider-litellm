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
  it("requires the native Provider extension API", async () => {
    const { default: manifest } = await import("../package.json", {
      with: { type: "json" },
    });

    expect(manifest.peerDependencies["@earendil-works/pi-ai"]).toBe(">=0.81.0");
    expect(manifest.peerDependencies["@earendil-works/pi-coding-agent"]).toBe(">=0.81.0");
    expect(manifest.devDependencies["@earendil-works/pi-ai"]).toBe("^0.81.1");
    expect(manifest.devDependencies["@earendil-works/pi-coding-agent"]).toBe("^0.81.1");
  });

  it("documents native Provider model persistence", async () => {
    const readme = await readFile("README.md", "utf8");

    expect(readme).toContain("Pi 0.81.0+ is required");
    expect(readme).toContain("native Provider");
    expect(readme).toContain("run `/login`, choose `Sign in with an API key`, then choose `LiteLLM API key`");
    expect(readme).toContain("With `/login litellm`, choose `Sign in with an API key` directly");
    expect(readme).toContain("~/.pi/agent/models-store.json");
    expect(readme).toContain("Opening `/model` refreshes configured provider catalogs");
    expect(readme).not.toContain("/litellm-refresh");
    expect(readme).toContain("Legacy `litellm-models*.json` files are ignored and are not deleted");
    expect(readme).not.toContain("older than 24 hours");
    expect(readme).not.toContain("enter `2` for SSO");
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
    // Pi 0.81.1 no longer ships a nested protobufjs copy.
    expect(copiesOf("protobufjs")).toEqual({
      "node_modules/protobufjs": "8.7.1",
    });
  });
});
