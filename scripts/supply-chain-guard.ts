import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

// Hooks npm runs automatically during install, packing or publishing. Any of these can
// mutate the tarball or the consumer's machine without the guard observing it, because
// the guard inspects `npm pack --dry-run --ignore-scripts`. `prepare` has automatic
// pre/post siblings that also run on `npm ci`, and the release workflow runs `npm ci`
// before publishing, so a `postprepare` would execute with repository write access
// before the tarball is built. `prepublishOnly` is deliberately absent: it runs only on
// an explicit publish and is this package's verification command, not a mutation hook.
const automaticLifecycleScripts = new Set([
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
]);
const runtimeDependencySections = new Set([
  "dependencies",
  "optionalDependencies",
  "bundleDependencies",
  "bundledDependencies",
]);
const dependencySections = [
  "dependencies",
  "devDependencies",
  "optionalDependencies",
  "peerDependencies",
  "bundleDependencies",
  "bundledDependencies",
  "overrides",
];

const nonRegistrySpecPrefixes = [
  "git:",
  "git+",
  "github:",
  "gitlab:",
  "bitbucket:",
  "http:",
  "https:",
  "file:",
  "link:",
  "workspace:",
  "npm:",
  "../",
  "./",
];

// The published source modules, listed explicitly so adding one is a reviewed
// decision. `tests/supply-chain-guard.test.ts` asserts this list matches `src/`
// exactly, so an entry for a module this branch does not ship fails the suite.
export const allowedSourceModules = [
  "cost",
  "discover",
  "gcloud-token",
  "index",
  "litellm",
  "mcp-tools",
  "provider",
  "skills",
  "types",
];

const allowedPackageFiles = [
  /^package\.json$/,
  /^README\.md$/,
  /^LICENSE$/,
  new RegExp(`^src/(?:${allowedSourceModules.join("|")})\\.ts$`),
];

export interface SupplyChainGuardOptions {
  checkPackageContents?: boolean;
}

export interface SupplyChainGuardResult {
  ok: boolean;
  errors: string[];
  packageFiles: string[];
}

interface PackageManifest {
  scripts?: Record<string, string>;
  [key: string]: unknown;
}

interface PackageLock {
  packages?: Record<string, { resolved?: unknown }>;
}

interface PackResult {
  files?: Array<{ path?: string }>;
}

export function parsePackageFiles(stdout: string): string[] {
  const parsed = JSON.parse(stdout) as PackResult[] | Record<string, PackResult>;
  return (Array.isArray(parsed) ? parsed : Object.values(parsed)).flatMap(
    (result) => result.files?.map((file) => file.path).filter(Boolean) ?? [],
  ) as string[];
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

async function readJson<T>(path: string, errors: string[]): Promise<T | undefined> {
  try {
    return JSON.parse(await readFile(path, "utf8")) as T;
  } catch (error) {
    errors.push(`${path}: could not parse JSON (${error instanceof Error ? error.message : String(error)})`);
    return undefined;
  }
}

function hasNonRegistrySpec(spec: unknown): boolean {
  if (typeof spec !== "string") return false;

  const normalized = spec.trim().toLowerCase();
  return nonRegistrySpecPrefixes.some((prefix) => normalized.startsWith(prefix));
}

function checkDependencyValue(section: string, value: unknown, errors: string[], path = section): void {
  if (Array.isArray(value)) {
    if (runtimeDependencySections.has(section) && value.length > 0) {
      errors.push(`package.json: ${section} is not allowed in the published package (${value.join(", ")})`);
    }
    return;
  }

  if (!isRecord(value)) return;

  const entries = Object.entries(value);
  if (runtimeDependencySections.has(section) && entries.length > 0) {
    errors.push(
      `package.json: ${section} is not allowed in this runtime-dependency-free provider (${entries
        .map(([name]) => name)
        .join(", ")})`,
    );
  }

  for (const [name, spec] of entries) {
    const nextPath = `${path}.${name}`;
    if (hasNonRegistrySpec(spec)) {
      errors.push(`package.json: ${nextPath} uses a non-registry dependency spec (${String(spec)})`);
    } else if (isRecord(spec) || Array.isArray(spec)) {
      checkDependencyValue(section, spec, errors, nextPath);
    }
  }
}

function checkManifest(manifest: PackageManifest, errors: string[]): void {
  for (const [name] of Object.entries(manifest.scripts ?? {})) {
    if (automaticLifecycleScripts.has(name)) {
      errors.push(`package.json: scripts.${name} runs automatically during install, pack, or publish`);
    }
  }

  for (const section of dependencySections) {
    checkDependencyValue(section, manifest[section], errors);
  }
}

async function checkLockfile(root: string, errors: string[]): Promise<void> {
  const lockfilePath = join(root, "package-lock.json");
  if (!existsSync(lockfilePath)) {
    errors.push("package-lock.json: required for reproducible npm installs and audit");
    return;
  }

  const lockfile = await readJson<PackageLock>(lockfilePath, errors);
  for (const [packagePath, entry] of Object.entries(lockfile?.packages ?? {})) {
    if (typeof entry.resolved !== "string") continue;
    if (entry.resolved.startsWith("https://registry.npmjs.org/")) continue;

    errors.push(`package-lock.json: ${packagePath} has non-registry resolved URL (${entry.resolved})`);
  }
}

async function listPackageFiles(root: string, errors: string[]): Promise<string[]> {
  try {
    const { stdout } = await execFileAsync("npm", ["pack", "--dry-run", "--json", "--ignore-scripts"], {
      cwd: root,
      maxBuffer: 1024 * 1024 * 10,
    });
    return parsePackageFiles(stdout);
  } catch (error) {
    errors.push(
      `npm pack --dry-run --json --ignore-scripts failed (${error instanceof Error ? error.message : error})`,
    );
    return [];
  }
}

// The allowlist bounds what may ship; this bounds what must. Without it a `files` typo
// publishes a package with no source at all and the guard reports success, because it
// only ever inspects files that are present. Only the source modules are checked: npm
// includes package.json, README and LICENSE whatever `files` says, so asserting those
// could never fail.
function checkRequiredPackageFiles(files: string[], errors: string[]): void {
  const present = new Set(files);
  for (const module of allowedSourceModules) {
    if (!present.has(`src/${module}.ts`)) {
      errors.push(`npm package: required published file src/${module}.ts is missing`);
    }
  }
}

function checkPackageFiles(files: string[], errors: string[]): void {
  checkRequiredPackageFiles(files, errors);
  for (const file of files) {
    if (!allowedPackageFiles.some((pattern) => pattern.test(file))) {
      errors.push(`npm package: unexpected published file ${file}`);
    }

    const filename = basename(file);
    if (filename.endsWith(".sh") || filename.endsWith(".cjs") || filename.endsWith(".mjs")) {
      errors.push(`npm package: executable-style payload file is not expected (${file})`);
    }
  }
}

export async function checkSupplyChain(
  root = process.cwd(),
  options: SupplyChainGuardOptions = {},
): Promise<SupplyChainGuardResult> {
  const checkPackageContents = options.checkPackageContents ?? true;
  const resolvedRoot = resolve(root);
  const errors: string[] = [];
  const manifest = await readJson<PackageManifest>(join(resolvedRoot, "package.json"), errors);

  if (manifest) {
    checkManifest(manifest, errors);
  }
  await checkLockfile(resolvedRoot, errors);

  const packageFiles = checkPackageContents ? await listPackageFiles(resolvedRoot, errors) : [];
  if (checkPackageContents) {
    checkPackageFiles(packageFiles, errors);
  }

  return { ok: errors.length === 0, errors, packageFiles };
}

async function main(): Promise<void> {
  const result = await checkSupplyChain();
  if (!result.ok) {
    console.error("Supply-chain guard failed:");
    for (const error of result.errors) {
      console.error(`- ${error}`);
    }
    process.exitCode = 1;
    return;
  }

  console.log(`Supply-chain guard passed (${result.packageFiles.length} package files checked).`);
}

if (import.meta.url === pathToFileURL(process.argv[1] ?? "").href) {
  void main();
}
