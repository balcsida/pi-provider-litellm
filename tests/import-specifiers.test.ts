import { beforeAll, describe, expect, it } from "vitest";
import {
  FORBIDDEN_RESOLVER,
  importSpecifiers,
  initImportSpecifiers,
  UNVERIFIABLE_SPECIFIER,
} from "./import-specifiers.js";

// The package-load contract in tests/package.test.ts trusts this oracle, and an assertion
// over an empty result would be vacuously true -- so an oracle returning nothing would make
// that contract pass for any source. These cases pin the oracle itself.
describe("importSpecifiers", () => {
  beforeAll(async () => {
    await initImportSpecifiers();
  });

  it.for([
    ['import "pkg";', ["pkg"]],
    ['import a from "pkg";', ["pkg"]],
    ['import { a } from "pkg";', ["pkg"]],
    ['import\n  { a }\n  from\n  "pkg";', ["pkg"]],
    ['import type { A } from "pkg";', ["pkg"]],
    ['export { a } from "pkg";', ["pkg"]],
    ['export * from "pkg";', ["pkg"]],
    ['export * as ns from "pkg";', ["pkg"]],
    ['await import("pkg");', ["pkg"]],
    ['import a from "./rel.js";', ["./rel.js"]],
    ['import { readFile } from "node:fs/promises";', ["node:fs/promises"]],
  ] as const)("extracts %s", ([source, expected]) => {
    expect(importSpecifiers(source)).toEqual([...expected]);
  });

  it.for([
    ["a line comment", '// import "pkg";\n'],
    ["a block comment", '/* import "pkg"; */'],
    ["a comment mentioning from", '// see from "pkg" for details\n'],
    ["a plain string", `const msg = "resolved from 'pkg'";`],
    ["a string with escaped quotes", 'const doc = "import x from \\"pkg\\"";'],
    ["a template literal", 'const msg = `alias from "pkg" is unknown`;'],
    ["an error message quoting a dynamic import", `throw new Error("use import('./x.js') instead");`],
  ] as const)("ignores %s", ([, source]) => {
    expect(importSpecifiers(source)).toEqual([]);
  });

  it.for([
    ["a bare identifier", "await import(name);"],
    ["a template with substitution", `await import(\`$${"{base}"}/mod.js\`);`],
    ["a plain template literal", "await import(`pkg`);"],
    ["a concatenation", 'await import("pk" + "g");'],
  ] as const)("reports %s as unverifiable", ([, source]) => {
    expect(importSpecifiers(source)).toContain(UNVERIFIABLE_SPECIFIER);
  });

  it.for([
    ["require", 'const m = require("pkg");'],
    ["createRequire", 'const r = createRequire(import.meta.url); r("pkg");'],
    ["eval", 'eval("import(\\"pkg\\")");'],
    ["new Function", 'new Function("return import(\\"pkg\\")");'],
    ["import.meta.resolve", 'import.meta.resolve("pkg");'],
    ["resolver after a regex containing quotes", 'const re = /[`"]+/; require("pkg");'],
    ["resolver after a regex containing comment markers", 'const re = /\\/\\/* not a comment/; eval("1");'],
    ["resolver in a template substitution", 'const value = `\u0024{require("pkg")}`;'],
    ["resolver after a nested brace in a template substitution", 'const value = `\u0024{({}), eval("1")}`;'],
    ["resolver after a conditional regex", 'if (ok) /safe/.test(value); require("pkg");'],
    ["resolver after division", 'const ratio = total / count; require("pkg");'],
  ] as const)("rejects %s as a forbidden resolver", ([, source]) => {
    expect(importSpecifiers(source)).toContain(FORBIDDEN_RESOLVER);
  });

  // Only executable text counts. Flagging prose would fail the package contract for a doc
  // comment or an error message, which is worse than the bypass it would be guarding.
  it.for([
    ["a line comment", "// we never call require() here\n"],
    ["a block comment", "/* eval() is forbidden in this package */"],
    ["a double-quoted message", 'const hint = "do not call require() directly";'],
    ["a single-quoted message", "const hint = 'createRequire() is not allowed';"],
    ["a template message", "const hint = `new Function() is rejected`;"],
    ["an error string", 'throw new Error("import.meta.resolve() is unsupported here");'],
    ["a regex literal", 'const pattern = /require\\("pkg"\\)|eval\\("1"\\)/;'],
    ["a conditional regex literal", 'if (ok) /require\\("pkg"\\)/.test(value);'],
    ["a template literal body", 'const value = `require("pkg")`;'],
    ["a string in a template substitution", 'const value = `\u0024{"require(\\"pkg\\")"}`;'],
    ["an identifier that merely contains the word", "const requiredFields = [1];\nconst evaluate = () => 1;"],
  ] as const)("does not read %s as a forbidden resolver", ([, source]) => {
    expect(importSpecifiers(source)).toEqual([]);
  });

  it("finds a lazy import inside a function that is never called", () => {
    const source = ["export async function unused() {", '  return await import("pkg");', "}"].join("\n");

    expect(importSpecifiers(source)).toEqual(["pkg"]);
  });

  // Asserted literally so renaming either constant cannot turn a rejected specifier into one
  // an allowlist would silently accept.
  it.for([UNVERIFIABLE_SPECIFIER, FORBIDDEN_RESOLVER] as const)("uses a sentinel no allowlist accepts: %s", (s) => {
    expect(s.startsWith("<")).toBe(true);
    expect(s.startsWith("node:")).toBe(false);
    expect(s.startsWith("./")).toBe(false);
  });
});
