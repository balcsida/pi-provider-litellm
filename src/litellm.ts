import { basename } from "node:path";

export function getSessionIdFromFile(sessionFile?: string): string | undefined {
  if (!sessionFile) return undefined;
  const filename = basename(sessionFile).replace(/\.jsonl$/i, "");
  if (!filename) return undefined;
  const uuidMatch = filename.match(/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$/i);
  return uuidMatch?.[1] ?? filename;
}
