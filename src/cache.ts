import { scryptSync } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import type { CacheFile } from "./types.js";

const FINGERPRINT_SALT = "pi-provider-litellm-cache-fingerprint-v1";

export function fingerprint(apiKey: string): string {
  return scryptSync(apiKey, FINGERPRINT_SALT, 32).toString("hex");
}

export function isCacheValid(
  cache: CacheFile | null,
  baseUrl: string,
  apiKey: string,
  headersFingerprint?: string,
): boolean {
  if (!cache) return false;
  return (
    cache.baseUrl === baseUrl &&
    cache.apiKeyFingerprint === fingerprint(apiKey) &&
    cache.headersFingerprint === headersFingerprint
  );
}

export async function readCache(path: string): Promise<CacheFile | null> {
  let raw: string;
  try {
    raw = await readFile(path, "utf8");
  } catch {
    return null;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return null;
  }
  if (!isCacheFileShape(parsed)) return null;
  return parsed;
}

function isCacheFileShape(value: unknown): value is CacheFile {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    typeof v.baseUrl === "string" &&
    typeof v.apiKeyFingerprint === "string" &&
    typeof v.fetchedAt === "number" &&
    (v.source === "model_info" || v.source === "models_list" || v.source === "health") &&
    Array.isArray(v.models)
  );
}

export async function writeJsonAtomic(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const tmp = `${path}.${process.pid}.${Date.now()}.tmp`;
  await writeFile(tmp, JSON.stringify(value, null, 2), "utf8");
  await rename(tmp, path);
}
