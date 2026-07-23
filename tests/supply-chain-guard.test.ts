import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";

import { checkSupplyChain, parsePackageFiles } from "../scripts/supply-chain-guard.js";

describe("supply-chain guard", () => {
  it("parses npm 11 and npm 12 package metadata", () => {
    const packResult = { files: [{ path: "package.json" }] };

    expect(parsePackageFiles(JSON.stringify([packResult]))).toEqual(["package.json"]);
    expect(parsePackageFiles(JSON.stringify({ fixture: packResult }))).toEqual(["package.json"]);
  });

  it("rejects install hooks, optional runtime dependencies, Git specs, and packaged payloads", async () => {
    const fixture = await mkdtemp(join(tmpdir(), "pi-provider-litellm-guard-"));

    try {
      await writeFile(
        join(fixture, "package.json"),
        JSON.stringify(
          {
            name: "malicious-fixture",
            version: "1.0.0",
            files: ["payload.js"],
            scripts: {
              postinstall: "node payload.js",
            },
            optionalDependencies: {
              "telemetry-helper": "git+https://github.com/attacker/telemetry-helper.git#deadbeef",
            },
          },
          null,
          2,
        ),
      );
      await writeFile(
        join(fixture, "package-lock.json"),
        JSON.stringify(
          {
            name: "malicious-fixture",
            version: "1.0.0",
            lockfileVersion: 3,
            packages: {
              "": {
                name: "malicious-fixture",
                version: "1.0.0",
                optionalDependencies: {
                  "telemetry-helper": "git+https://github.com/attacker/telemetry-helper.git#deadbeef",
                },
              },
              "node_modules/telemetry-helper": {
                version: "1.0.0",
                resolved: "git+https://github.com/attacker/telemetry-helper.git#deadbeef",
              },
            },
          },
          null,
          2,
        ),
      );
      await writeFile(join(fixture, "payload.js"), "console.log('payload');\n");

      const result = await checkSupplyChain(fixture);

      expect(result.ok).toBe(false);
      expect(result.errors.join("\n")).toContain("postinstall");
      expect(result.errors.join("\n")).toContain("optionalDependencies.telemetry-helper");
      expect(result.errors.join("\n")).toContain("non-registry resolved URL");
      expect(result.errors.join("\n")).toContain("payload.js");
    } finally {
      await rm(fixture, { recursive: true, force: true });
    }
  });

  it("accepts this package policy while keeping the publish build gate", async () => {
    const result = await checkSupplyChain(process.cwd(), { checkPackageContents: false });

    expect(result.errors).toEqual([]);
    expect(result.ok).toBe(true);
  });

  it("accepts the intentional runtime dist files in the package", async () => {
    const fixture = await mkdtemp(join(tmpdir(), "pi-provider-litellm-allowed-"));

    try {
      await mkdir(join(fixture, "dist"), { recursive: true });
      await writeFile(
        join(fixture, "package.json"),
        JSON.stringify(
          {
            name: "allowed-fixture",
            version: "1.0.0",
            files: ["dist", "README.md", "LICENSE"],
          },
          null,
          2,
        ),
      );
      await writeFile(
        join(fixture, "package-lock.json"),
        JSON.stringify({
          name: "allowed-fixture",
          version: "1.0.0",
          lockfileVersion: 3,
          packages: { "": { name: "allowed-fixture", version: "1.0.0" } },
        }),
      );
      await writeFile(join(fixture, "README.md"), "# fixture\n");
      await writeFile(join(fixture, "LICENSE"), "MIT\n");
      for (const file of [
        "cache",
        "cost",
        "discover",
        "gcloud-token",
        "gcloud-token-cli",
        "index",
        "litellm",
        "mcp-tools",
        "provider",
        "skills",
        "types",
      ]) {
        await writeFile(join(fixture, "dist", `${file}.js`), "export {};\n");
        await writeFile(join(fixture, "dist", `${file}.d.ts`), "export {};\n");
      }

      const result = await checkSupplyChain(fixture);

      expect(result.errors).toEqual([]);
      expect(result.ok).toBe(true);
    } finally {
      await rm(fixture, { recursive: true, force: true });
    }
  });
});
