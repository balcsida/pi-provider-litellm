import { SyntaxKind } from "@typescript/native-preview/unstable/ast";
import { createScanner } from "@typescript/native-preview/unstable/ast/scanner";
import { init, parse } from "es-module-lexer";

// Extracts module specifiers from source text so the package-load contract can assert that
// `src/` only imports things Pi's loader provides. es-module-lexer parses ESM imports, while
// TypeScript's scanner identifies executable CommonJS and dynamic-evaluation escape hatches
// without mistaking comments, strings, templates, or regex literals for calls.
export const UNVERIFIABLE_SPECIFIER = "<unverifiable-dynamic-specifier>";
export const FORBIDDEN_RESOLVER = "<forbidden-dynamic-resolver>";

let ready: Promise<void> | undefined;

// es-module-lexer compiles a WASM module on first use.
export async function initImportSpecifiers(): Promise<void> {
  ready ??= init;
  await ready;
}

const REGEX_PRECEDERS = new Set([
  SyntaxKind.OpenParenToken,
  SyntaxKind.OpenBracketToken,
  SyntaxKind.OpenBraceToken,
  SyntaxKind.CommaToken,
  SyntaxKind.SemicolonToken,
  SyntaxKind.ColonToken,
  SyntaxKind.QuestionToken,
  SyntaxKind.EqualsToken,
  SyntaxKind.EqualsGreaterThanToken,
  SyntaxKind.ReturnKeyword,
  SyntaxKind.ThrowKeyword,
  SyntaxKind.CaseKeyword,
  SyntaxKind.DeleteKeyword,
  SyntaxKind.TypeOfKeyword,
  SyntaxKind.VoidKeyword,
  SyntaxKind.YieldKeyword,
  SyntaxKind.AwaitKeyword,
]);

function tokenEndsExpression(kind: SyntaxKind): boolean {
  return (
    kind === SyntaxKind.Identifier ||
    kind === SyntaxKind.RequireKeyword ||
    kind === SyntaxKind.NumericLiteral ||
    kind === SyntaxKind.BigIntLiteral ||
    kind === SyntaxKind.StringLiteral ||
    kind === SyntaxKind.RegularExpressionLiteral ||
    kind === SyntaxKind.NoSubstitutionTemplateLiteral ||
    kind === SyntaxKind.TemplateTail ||
    kind === SyntaxKind.TrueKeyword ||
    kind === SyntaxKind.FalseKeyword ||
    kind === SyntaxKind.NullKeyword ||
    kind === SyntaxKind.ThisKeyword ||
    kind === SyntaxKind.SuperKeyword ||
    kind === SyntaxKind.CloseParenToken ||
    kind === SyntaxKind.CloseBracketToken ||
    kind === SyntaxKind.CloseBraceToken ||
    kind === SyntaxKind.PlusPlusToken ||
    kind === SyntaxKind.MinusMinusToken
  );
}

function scanTokens(source: string): Array<{ kind: SyntaxKind; text: string }> {
  const scanner = createScanner(true, undefined, source);
  const tokens: Array<{ kind: SyntaxKind; text: string }> = [];
  const templateBraceDepths: number[] = [];
  let previousKind: SyntaxKind | undefined;

  while (true) {
    let kind = scanner.scan();
    if (
      (kind === SyntaxKind.SlashToken || kind === SyntaxKind.SlashEqualsToken) &&
      (previousKind === undefined || REGEX_PRECEDERS.has(previousKind) || !tokenEndsExpression(previousKind))
    ) {
      kind = scanner.reScanSlashToken();
    }
    if (kind === SyntaxKind.EndOfFile) break;

    const text = scanner.getTokenText();
    if (kind === SyntaxKind.TemplateHead) templateBraceDepths.push(0);
    tokens.push({ kind, text });

    const templateIndex = templateBraceDepths.length - 1;
    if (kind === SyntaxKind.OpenBraceToken && templateIndex >= 0) {
      templateBraceDepths[templateIndex] += 1;
    } else if (kind === SyntaxKind.CloseBraceToken && templateIndex >= 0) {
      if (templateBraceDepths[templateIndex] > 0) {
        templateBraceDepths[templateIndex] -= 1;
      } else {
        kind = scanner.reScanTemplateToken(false);
        tokens.push({ kind, text: scanner.getTokenText() });
        if (kind === SyntaxKind.TemplateTail) templateBraceDepths.pop();
      }
    }
    previousKind = kind;
  }

  return tokens;
}

function containsForbiddenResolver(source: string): boolean {
  const tokens = scanTokens(source);

  for (let index = 0; index < tokens.length; index += 1) {
    const current = tokens[index];
    const next = tokens[index + 1];
    const previous = tokens[index - 1];

    if (
      next?.kind === SyntaxKind.OpenParenToken &&
      (current.kind === SyntaxKind.RequireKeyword || current.text === "createRequire" || current.text === "eval")
    ) {
      return true;
    }

    if (
      current.kind === SyntaxKind.NewKeyword &&
      next?.text === "Function" &&
      tokens[index + 2]?.kind === SyntaxKind.OpenParenToken
    ) {
      return true;
    }

    if (
      previous?.kind === SyntaxKind.DotToken &&
      current.text === "resolve" &&
      next?.kind === SyntaxKind.OpenParenToken &&
      tokens[index - 2]?.text === "meta" &&
      tokens[index - 3]?.kind === SyntaxKind.DotToken &&
      tokens[index - 4]?.kind === SyntaxKind.ImportKeyword
    ) {
      return true;
    }
  }

  return false;
}

export function importSpecifiers(source: string, filename = "source.ts"): string[] {
  const [imports] = parse(source, filename);
  const specifiers = imports.map((entry) => entry.n ?? UNVERIFIABLE_SPECIFIER);
  if (containsForbiddenResolver(source)) specifiers.push(FORBIDDEN_RESOLVER);
  return specifiers;
}
