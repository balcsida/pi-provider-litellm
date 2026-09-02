import { execFile } from "node:child_process";
import { cp, mkdir, mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { promisify } from "node:util";
import { describe, expect, it } from "vitest";
import { checkSupplyChain } from "../scripts/supply-chain-guard.js";
import { importSpecifiers, initImportSpecifiers } from "./import-specifiers.js";
import { hermeticChildEnv } from "./test-helpers.js";

const execFileAsync = promisify(execFile);
const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");

interface PackageManifest {
  main?: string;
  types?: string;
  exports?: unknown;
  files: string[];
  pi: { extensions: string[]; image: string };
}

interface LoadResult {
  errors: unknown[];
  extensions: unknown[];
}

async function loadExtension(
  entrypoint: string,
  cwd: string,
  loaderPath = resolve(repoRoot, "node_modules/@earendil-works/pi-coding-agent/dist/core/extensions/loader.js"),
): Promise<LoadResult> {
  let loader: unknown;
  try {
    loader = await import(pathToFileURL(loaderPath).href);
  } catch (error) {
    throw new Error(
      `Pi's internal extension loader is unavailable at ${loaderPath}; update this package test for the installed @earendil-works/pi-coding-agent layout`,
      { cause: error },
    );
  }

  const loadExtensions = (loader as { loadExtensions?: unknown }).loadExtensions;
  if (typeof loadExtensions !== "function") {
    throw new Error(`Pi's internal extension loader at ${loaderPath} does not export loadExtensions`);
  }

  return (loadExtensions as (paths: string[], cwd: string) => Promise<LoadResult>)([entrypoint], cwd);
}

function isPathOutsideRepo(path: string, root = repoRoot): boolean {
  const pathFromRepo = relative(root, path);
  return (
    pathFromRepo !== "" && (isAbsolute(pathFromRepo) || pathFromRepo === ".." || pathFromRepo.startsWith(`..${sep}`))
  );
}

function expectFixtureOutsideRepo(fixture: string): void {
  expect(isPathOutsideRepo(fixture), `${fixture} must be outside ${repoRoot}`).toBe(true);
}

async function runListModels(piBin: string, cwd: string, agentDir: string): Promise<string> {
  const { stdout, stderr } = await execFileAsync(piBin, ["-e", ".", "--list-models", "litellm"], {
    cwd,
    env: hermeticChildEnv({
      PI_CODING_AGENT_DIR: agentDir,
      LITELLM_HEADERS: "{bad json",
      LITELLM_OFFLINE: "1",
      LITELLM_BASE_URL: "https://proxy.invalid",
      LITELLM_API_KEY: "sk-not-a-real-key",
    }),
  });
  return `${stdout}${stderr}`;
}

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
    const readme = await readFile(join(repoRoot, "README.md"), "utf8");

    expect(readme).not.toContain("https://img.shields.io/npm/v/pi-provider-litellm.svg");
  });
});

describe("pi package compatibility", () => {
  it("declares a single source-only Pi entrypoint", async () => {
    const manifest = JSON.parse(await readFile(join(repoRoot, "package.json"), "utf8")) as PackageManifest;

    expect(manifest.pi.extensions).toEqual(["./src/index.ts"]);
    expect(manifest.files).toEqual(["src", "README.md", "LICENSE"]);
    expect(manifest).not.toHaveProperty("main");
    expect(manifest).not.toHaveProperty("types");
    expect(manifest).not.toHaveProperty("exports");
  });

  it("reports when Pi's internal extension loader path changes", async () => {
    const missingLoader = join(repoRoot, "node_modules/@earendil-works/pi-coding-agent/dist/missing-loader.js");

    await expect(loadExtension("unused.ts", repoRoot, missingLoader)).rejects.toThrow(
      `Pi's internal extension loader is unavailable at ${missingLoader}`,
    );
  });

  it("loads a production-only Git archive from its manifest entrypoint", async () => {
    const fixture = await mkdtemp(join(tmpdir(), "pi-provider-litellm-git-package-"));
    try {
      expectFixtureOutsideRepo(fixture);
      // Archive HEAD, not the working tree: a Git install materialises the committed
      // tree, `git stash create` writes objects into an object store shared with sibling
      // worktrees, and its output silently omits untracked files. Working-tree fidelity
      // is owned by the `npm pack` test below, which is filesystem-based.
      const archivePath = join(fixture, "git-package.tar");
      await execFileAsync("git", ["archive", "--format=tar", `--output=${archivePath}`, "HEAD"], { cwd: repoRoot });
      await execFileAsync("tar", ["-xf", archivePath, "-C", fixture]);
      await rm(archivePath);
      const manifest = JSON.parse(await readFile(join(fixture, "package.json"), "utf8")) as PackageManifest;
      const entrypoint = resolve(fixture, manifest.pi.extensions[0]);

      const result = await loadExtension(entrypoint, fixture);

      expect(result.errors).toEqual([]);
      expect(result.extensions).toHaveLength(1);
    } finally {
      await rm(fixture, { force: true, recursive: true });
    }
  }, 30_000);

  // Proves the packed tarball ships the right files and that its declared entrypoint
  // loads through Pi's real loader from both the extracted package root and a
  // node_modules-shaped path. Peer resolution is deliberately not asserted: the loader
  // is imported from this repository, so it resolves the extension's bare specifiers
  // through its own context rather than from the fixture. `npm install` here could not
  // prove peer resolution either, and cost a registry download of the peers' transitive
  // dependencies on every run.
  it("packs, installs, and loads the source package without TypeScript stripping in node_modules", async () => {
    const fixture = await mkdtemp(join(tmpdir(), "pi-provider-litellm-npm-package-"));
    try {
      expectFixtureOutsideRepo(fixture);
      const { stdout: tarballName } = await execFileAsync(
        "npm",
        ["pack", "--ignore-scripts", "--pack-destination", fixture],
        { cwd: repoRoot },
      );
      const tarballPath = resolve(fixture, tarballName.trim().split("\n").at(-1) ?? "");
      const { stdout: fileList } = await execFileAsync("tar", ["-tzf", tarballPath]);
      await execFileAsync("tar", ["-xzf", tarballPath, "-C", fixture]);

      const packageRoot = join(fixture, "package");
      const manifest = JSON.parse(await readFile(join(packageRoot, "package.json"), "utf8")) as PackageManifest;
      const entrypoint = resolve(packageRoot, manifest.pi.extensions[0]);
      const result = await loadExtension(entrypoint, packageRoot);

      const nodeModules = join(fixture, "node_modules");
      await mkdir(nodeModules, { recursive: true });
      await cp(packageRoot, join(nodeModules, "pi-provider-litellm"), { recursive: true });
      const installedEntrypoint = resolve(fixture, "node_modules/pi-provider-litellm", manifest.pi.extensions[0]);
      const installedResult = await loadExtension(installedEntrypoint, fixture);

      const guard = await checkSupplyChain(repoRoot);
      expect(guard.errors).toEqual([]);
      expect(fileList.trim().split("\n").sort()).toEqual(guard.packageFiles.map((file) => `package/${file}`).sort());
      expect(result.errors).toEqual([]);
      expect(result.extensions).toHaveLength(1);
      expect(installedResult.errors).toEqual([]);
      expect(installedResult.extensions).toHaveLength(1);
    } finally {
      await rm(fixture, { force: true, recursive: true });
    }
  }, 30_000);

  // The in-process loader tests hand `loadExtensions` an absolute file path, so they skip
  // the manifest resolution that `pi -e <dir>` and `pi install` actually perform. This
  // drives the real CLI against a Git-install-shaped tree instead. A malformed
  // LITELLM_HEADERS value is used as the load detector because the extension writes a
  // distinctive line while registering the provider -- `--list-models` alone is not
  // discriminating, since it also reports models from a cached store. The agent dir is
  // redirected at an empty directory so no globally installed copy can satisfy the probe.
  it("loads through the Pi CLI from a Git-install-shaped tree", async () => {
    const fixture = await mkdtemp(join(tmpdir(), "pi-provider-litellm-cli-"));
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-agent-"));
    const detector = "failed to parse custom headers";
    const piBin = resolve(repoRoot, "node_modules/.bin/pi");

    try {
      expectFixtureOutsideRepo(fixture);
      expectFixtureOutsideRepo(agentDir);
      const archivePath = join(fixture, "git-package.tar");
      await execFileAsync("git", ["archive", "--format=tar", `--output=${archivePath}`, "HEAD"], { cwd: repoRoot });
      await execFileAsync("tar", ["-xf", archivePath, "-C", fixture]);
      await rm(archivePath);

      expect(await runListModels(piBin, fixture, agentDir)).toContain(detector);

      // Negative control: a manifest entrypoint that does not exist is skipped silently by
      // the loader, so without this the assertion above could pass on a stale global copy.
      const manifestPath = join(fixture, "package.json");
      const manifest = JSON.parse(await readFile(manifestPath, "utf8")) as PackageManifest;
      manifest.pi.extensions = ["./src/does-not-exist.ts"];
      await writeFile(manifestPath, JSON.stringify(manifest, null, 2));

      expect(await runListModels(piBin, fixture, agentDir)).not.toContain(detector);
    } finally {
      await rm(fixture, { force: true, recursive: true });
      await rm(agentDir, { force: true, recursive: true });
    }
  }, 60_000);

  // The CI smoke asserts model ids in `--list-models` output, and Pi serves those from its own
  // models-store as well as from a loaded extension. This proves the assertion cannot be
  // satisfied by a prepopulated store alone: with a cached model present and the manifest
  // entrypoint broken, the extension does not load and its provider is absent.
  it("cannot satisfy the list-models assertion from a prepopulated store alone", async () => {
    const fixture = await mkdtemp(join(tmpdir(), "pi-provider-litellm-cache-"));
    const agentDir = await mkdtemp(join(tmpdir(), "pi-provider-litellm-cachedir-"));
    const cachedModelId = "cached-only-model";
    const piBin = resolve(repoRoot, "node_modules/.bin/pi");

    try {
      expectFixtureOutsideRepo(fixture);
      expectFixtureOutsideRepo(agentDir);
      const archivePath = join(fixture, "git-package.tar");
      await execFileAsync("git", ["archive", "--format=tar", `--output=${archivePath}`, "HEAD"], { cwd: repoRoot });
      await execFileAsync("tar", ["-xf", archivePath, "-C", fixture]);
      await rm(archivePath);

      await writeFile(
        join(agentDir, "models-store.json"),
        JSON.stringify({
          litellm: {
            checkedAt: Date.now(),
            models: [
              {
                id: cachedModelId,
                name: cachedModelId,
                provider: "litellm",
                api: "openai-completions",
                baseUrl: "https://litellm.example.com/v1",
                reasoning: false,
                input: ["text"],
                cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
                contextWindow: 128_000,
                maxTokens: 4096,
              },
            ],
          },
        }),
        "utf8",
      );

      const populatedOutput = await runListModels(piBin, fixture, agentDir);
      expect(populatedOutput).toContain(cachedModelId);

      const manifestPath = join(fixture, "package.json");
      const manifest = JSON.parse(await readFile(manifestPath, "utf8")) as PackageManifest;
      manifest.pi.extensions = ["./src/does-not-exist.ts"];
      await writeFile(manifestPath, JSON.stringify(manifest, null, 2));

      const brokenEntrypointOutput = await runListModels(piBin, fixture, agentDir);
      expect(brokenEntrypointOutput).not.toContain("failed to parse custom headers");
    } finally {
      await rm(fixture, { force: true, recursive: true });
      await rm(agentDir, { force: true, recursive: true });
    }
  }, 60_000);

  it("keeps source runtime imports loader-provided or built-in", async () => {
    await initImportSpecifiers();
    const sourceDir = join(repoRoot, "src");
    const sourceFiles = (await readdir(sourceDir, { recursive: true })).filter((file) => file.endsWith(".ts"));
    const imports = await Promise.all(
      sourceFiles.map(
        async (file) => [file, importSpecifiers(await readFile(join(sourceDir, file), "utf8"), file)] as const,
      ),
    );

    // A scanner bug that returned nothing would make the allowlist vacuously true, so
    // require every shipped module to yield at least one specifier. The oracle itself is
    // pinned by tests/import-specifiers.test.ts.
    expect(sourceFiles.length).toBeGreaterThan(0);
    for (const [file, specifiers] of imports) {
      expect(specifiers.length, `${file}: no module specifiers found`).toBeGreaterThan(0);
    }

    const allowed = new Set([
      "@earendil-works/pi-ai",
      "@earendil-works/pi-ai/compat",
      "@earendil-works/pi-ai/providers/all",
      "@earendil-works/pi-coding-agent",
    ]);
    for (const [file, specifiers] of imports) {
      for (const specifier of specifiers) {
        // UNVERIFIABLE_SPECIFIER and FORBIDDEN_RESOLVER land here too: a computed `import()`
        // cannot be shown to resolve to something the loader provides, and a CommonJS or
        // dynamic-evaluation resolver bypasses the loader entirely. Both fail by default.
        expect(
          specifier.startsWith("node:") || specifier.startsWith("./") || allowed.has(specifier),
          `${file}: ${specifier}`,
        ).toBe(true);
      }
    }
  });

  it("requires the native Provider extension API", async () => {
    const { default: manifest } = await import("../package.json", {
      with: { type: "json" },
    });

    expect(manifest.peerDependencies["@earendil-works/pi-ai"]).toBe(">=0.81.0");
    expect(manifest.peerDependencies["@earendil-works/pi-coding-agent"]).toBe(">=0.81.0");
    expect(manifest.peerDependenciesMeta).toEqual({
      "@earendil-works/pi-ai": { optional: true },
      "@earendil-works/pi-coding-agent": { optional: true },
    });
    expect(manifest.devDependencies["@earendil-works/pi-ai"]).toBe("^0.84.2");
    expect(manifest.devDependencies["@earendil-works/pi-coding-agent"]).toBe("^0.84.2");
  });

  it("documents native Provider model persistence and the extension-only package surface", async () => {
    const readme = await readFile(join(repoRoot, "README.md"), "utf8");

    expect(readme).toContain("Pi 0.81.0+ is required");
    expect(readme).toMatch(/source\s+entrypoint has been smoke-tested with Pi 0\.81\.0/);
    expect(readme).toMatch(/no\s+longer exposes that library import/);
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
    const lockfile = JSON.parse(await readFile(join(repoRoot, "package-lock.json"), "utf8")) as {
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
    expect(Object.values(copiesOf("brace-expansion"))).toEqual(["5.0.9"]);
    expect(Object.values(copiesOf("nanoid"))).toEqual(["3.3.18"]);
    expect(Object.values(copiesOf("undici"))).toEqual(["8.9.0"]);
    // Pi 0.84.2 still ships a nested protobufjs 7.x copy.
    expect(copiesOf("protobufjs")).toEqual({
      "node_modules/protobufjs": "8.7.1",
      "node_modules/@earendil-works/pi-coding-agent/node_modules/protobufjs": "7.6.5",
    });
  });
});
