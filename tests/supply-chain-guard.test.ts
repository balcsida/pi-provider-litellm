import { mkdir, mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { allowedSourceModules, checkSupplyChain, parsePackageFiles } from "../scripts/supply-chain-guard.js";

async function writeFixturePackage(
  fixture: string,
  files: string[],
  contents: Array<{ path: string; body?: string }>,
): Promise<void> {
  await writeFile(
    join(fixture, "package.json"),
    JSON.stringify({ name: "allowed-fixture", version: "1.0.0", files }, null, 2),
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
  for (const { path, body } of contents) {
    await mkdir(join(fixture, dirname(path)), { recursive: true });
    await writeFile(join(fixture, path), body ?? "export {};\n");
  }
}

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

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
    const result = await checkSupplyChain(repoRoot, { checkPackageContents: false });

    expect(result.errors).toEqual([]);
    expect(result.ok).toBe(true);
  });

  it("authorizes exactly the source modules this branch ships", async () => {
    const shipped = (await readdir(join(repoRoot, "src")))
      .filter((file) => file.endsWith(".ts"))
      .map((file) => file.replace(/\.ts$/, ""))
      .sort();

    expect([...allowedSourceModules].sort()).toEqual(shipped);
  });

  it("accepts only the intentional source modules in the package", async () => {
    const fixture = await mkdtemp(join(tmpdir(), "pi-provider-litellm-allowed-"));

    try {
      await writeFixturePackage(
        fixture,
        ["src", "README.md", "LICENSE"],
        allowedSourceModules.map((module) => ({ path: `src/${module}.ts` })),
      );

      const result = await checkSupplyChain(fixture);

      expect(result.errors).toEqual([]);
      expect(result.ok).toBe(true);
    } finally {
      await rm(fixture, { recursive: true, force: true });
    }
  });

  it("rejects a published build directory", async () => {
    const fixture = await mkdtemp(join(tmpdir(), "pi-provider-litellm-dist-"));

    try {
      await writeFixturePackage(
        fixture,
        ["src", "dist", "README.md", "LICENSE"],
        [{ path: "src/index.ts" }, { path: "dist/index.js" }, { path: "dist/index.d.ts" }],
      );

      const result = await checkSupplyChain(fixture);

      expect(result.ok).toBe(false);
      expect(result.errors).toContain("npm package: unexpected published file dist/index.js");
      expect(result.errors).toContain("npm package: unexpected published file dist/index.d.ts");
    } finally {
      await rm(fixture, { recursive: true, force: true });
    }
  });

  it.for([
    ["an empty files list", [] as string[]],
    ["a misspelled files entry", ["scr", "README.md", "LICENSE"]],
  ] as const)("rejects %s that ships no source", async ([, files]) => {
    const fixture = await mkdtemp(join(tmpdir(), "pi-provider-litellm-nosrc-"));

    try {
      await writeFixturePackage(fixture, [...files], [{ path: "src/index.ts" }]);

      const result = await checkSupplyChain(fixture);

      expect(result.ok).toBe(false);
      expect(result.errors).toContain("npm package: required published file src/index.ts is missing");
    } finally {
      await rm(fixture, { recursive: true, force: true });
    }
  });

  it.for([
    "preinstall",
    "install",
    "postinstall",
    "predependencies",
    "dependencies",
    "postdependencies",
    "preprepare",
    "prepare",
    "postprepare",
    "prepack",
    "postpack",
    "prepublish",
    "publish",
    "postpublish",
  ] as const)("rejects an automatic %s lifecycle hook", async (hook) => {
    const fixture = await mkdtemp(join(tmpdir(), "pi-provider-litellm-hook-"));

    try {
      await writeFixturePackage(
        fixture,
        ["src", "README.md", "LICENSE"],
        allowedSourceModules.map((module) => ({ path: `src/${module}.ts` })),
      );
      const manifest = JSON.parse(await readFile(join(fixture, "package.json"), "utf8")) as Record<string, unknown>;
      manifest.scripts = { [hook]: "node inject.js" };
      await writeFile(join(fixture, "package.json"), JSON.stringify(manifest, null, 2));

      const result = await checkSupplyChain(fixture);

      expect(result.ok).toBe(false);
      expect(result.errors).toContain(
        `package.json: scripts.${hook} runs automatically during install, pack, or publish`,
      );
    } finally {
      await rm(fixture, { recursive: true, force: true });
    }
  });

  it("accepts prepublishOnly as an explicit verification command", async () => {
    const fixture = await mkdtemp(join(tmpdir(), "pi-provider-litellm-verify-"));

    try {
      await writeFixturePackage(
        fixture,
        ["src", "README.md", "LICENSE"],
        allowedSourceModules.map((module) => ({ path: `src/${module}.ts` })),
      );
      const manifest = JSON.parse(await readFile(join(fixture, "package.json"), "utf8")) as Record<string, unknown>;
      manifest.scripts = { prepublishOnly: "npm run check" };
      await writeFile(join(fixture, "package.json"), JSON.stringify(manifest, null, 2));

      const result = await checkSupplyChain(fixture);

      expect(result.errors).toEqual([]);
      expect(result.ok).toBe(true);
    } finally {
      await rm(fixture, { recursive: true, force: true });
    }
  });

  it("rejects a source module outside the allowlist", async () => {
    const fixture = await mkdtemp(join(tmpdir(), "pi-provider-litellm-unlisted-"));

    try {
      await writeFixturePackage(
        fixture,
        ["src", "README.md", "LICENSE"],
        [{ path: "src/index.ts" }, { path: "src/exfiltrate.ts" }],
      );

      const result = await checkSupplyChain(fixture);

      expect(result.ok).toBe(false);
      expect(result.errors).toContain("npm package: unexpected published file src/exfiltrate.ts");
    } finally {
      await rm(fixture, { recursive: true, force: true });
    }
  });
});
