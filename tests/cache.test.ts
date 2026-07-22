import { mkdtemp, readFile, readdir } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { writeJsonAtomic } from "../src/cache.js";

describe("writeJsonAtomic", () => {
  it("replaces JSON without leaving a temporary file", async () => {
    const dir = await mkdtemp(join(tmpdir(), "pi-litellm-cache-"));
    const path = join(dir, "models.json");

    await writeJsonAtomic(path, { version: 1 });
    await writeJsonAtomic(path, { version: 2 });

    expect(JSON.parse(await readFile(path, "utf8"))).toEqual({ version: 2 });
    expect(await readdir(dir)).toEqual(["models.json"]);
  });
});
