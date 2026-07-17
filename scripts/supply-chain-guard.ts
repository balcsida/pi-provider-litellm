import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

const installLifecycleScripts = new Set(["preinstall", "install", "postinstall", "prepare"]);
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

const allowedPackageFiles = [
  /^package\.json$/,
  /^README\.md$/,
  /^LICENSE$/,
  /^dist\/(?:cache|cost|discover|gcloud-token|gcloud-token-cli|index|litellm|mcp-tools|skills|types)\.(?:js|d\.ts)$/,
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
    if (installLifecycleScripts.has(name)) {
      errors.push(`package.json: scripts.${name} runs during package installation`);
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

function checkPackageFiles(files: string[], errors: string[]): void {
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
