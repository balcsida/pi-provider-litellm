import { createHash, createHmac, randomBytes } from "node:crypto";
import type { Static, TSchema } from "@earendil-works/pi-ai";
import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ExtensionContext, type ToolDefinition } from "@earendil-works/pi-coding-agent";
import { normalizeBaseUrl } from "./discover.js";
import type { LiteLLMMcpTool, LiteLLMRuntimeAuth } from "./types.js";

const LIST_TIMEOUT_MS = 10_000;
const CALL_TIMEOUT_MS = 30_000;
const MAX_DISCOVERY_BODY_BYTES = 5 * 1024 * 1024;
const MAX_DISCOVERY_ENTRIES = 10_000;
const MAX_CALL_BODY_BYTES = 5 * 1024 * 1024;
const MAX_REGISTERED_TOOLS = 512;
const MAX_DESCRIPTION_BYTES = 4 * 1024;
const MAX_SCHEMA_BYTES = 64 * 1024;
const MAX_SCHEMA_DEPTH = 16;
const MAX_RESULT_BYTES = 64 * 1024;
const MAX_TOOL_NAME_LENGTH = 64;
const TOOL_NAME_HASH_LENGTH = 10;
const MAX_LABEL_BYTES = 256;
const MAX_DETAIL_BYTES = 256;
const MAX_ERROR_WALK_DEPTH = 16;
const MAX_DIAGNOSTIC_SAMPLES = 5;
const MAX_DIAGNOSTIC_CAUSE_BYTES = 200;
const MAX_DIAGNOSTIC_LABEL_BYTES = 96;
const TRUNCATION_MARKER = "\n[truncated by pi-provider-litellm]";
const DESCRIPTION_TRUNCATION_MARKER = "… [truncated]";
const SHORT_TRUNCATION_MARKER = "…";

interface RawLiteLLMMcpTool {
  name?: unknown;
  description?: unknown;
  inputSchema?: unknown;
  input_schema?: unknown;
  server_id?: unknown;
  server_name?: unknown;
  mcp_info?: {
    server_id?: unknown;
    server_name?: unknown;
  };
}

interface PreparedTool {
  tool: LiteLLMMcpTool;
  name: string;
  parameters: TSchema;
  syntheticArgsEnvelope: boolean;
}

// Why a discovered MCP tool was not registered as the proxy described it.
// `schema-envelope` is a degradation, not a loss: the tool still registers, with the
// extension-owned envelope in place of a schema that could not be proven safe.
export type McpDropReason = "invalid-tool" | "duplicate-identity" | "invalid-schema" | "name-collision";
export type McpDegradeReason = "schema-envelope";
export type McpIncidentReason = McpDropReason | McpDegradeReason;

const DROP_REASON_TEXT: Readonly<Record<McpDropReason, string>> = {
  "invalid-tool": "a missing name or server identity",
  "duplicate-identity": "a repeated valid identity (the first valid occurrence of each is the one registered)",
  "invalid-schema": "an invalid or oversized input schema",
  "name-collision": "a colliding generated Pi name",
};

const DEGRADE_REASON_TEXT: Readonly<Record<McpDegradeReason, string>> = {
  "schema-envelope": "a schema that could not be proven free of proxy-supplied regexes or references",
};

const INCIDENT_REASON_TEXT: Readonly<Record<McpIncidentReason, string>> = {
  ...DROP_REASON_TEXT,
  ...DEGRADE_REASON_TEXT,
};

export interface McpPreparationReport {
  // Raw tool entries the proxy returned, before any normalization.
  discovered: number;
  // Tools that will be registered, whatever parameter shape they ended up with.
  prepared: number;
  // Of `prepared`, how many carry the extension-owned envelope rather than a proxy-supplied schema.
  // Includes both the schemaless path and hazard degradations; `degraded` counts only the latter.
  enveloped: number;
  // Tools discarded because the registered-tool cap was reached.
  overflow: number;
  // Bounded generated names for the overflow membership, used to detect catalog changes.
  overflowTools: string[];
  dropped: Array<{ reason: McpDropReason; tools: string[] }>;
  degraded: Array<{ reason: McpDegradeReason; tools: string[] }>;
}

// Bounded discovery result: `raw` is what the proxy returned, so losses can be reconciled.
export interface McpDiscovery {
  raw: number;
  tools: LiteLLMMcpTool[];
  // Entries dropped by normalization, as bounded safe labels.
  invalid: string[];
}

// Incident-aware suppression: an incident class is re-reported whenever its message changes
// (different count or different tools), so a stable proxy stays quiet across refreshes while a
// changed one is always surfaced. Keys come from a fixed set of classes, so this cannot grow
// without bound.
const lastEmittedIncident = new Map<string, string>();

// Reports an incident unless its identity is unchanged since the last report of that class.
// `identity` must capture everything an operator would want to hear about again — for tool
// incidents that is the full sorted membership, not just the bounded sample that gets printed.
function emitSafetyDiagnostic(incident: string, message: string, identity: string = message): void {
  if (lastEmittedIncident.get(incident) === identity) return;
  lastEmittedIncident.set(incident, identity);
  process.stderr.write(`LiteLLM MCP: ${message}\n`);
}

function clearIncident(incident: string): void {
  lastEmittedIncident.delete(incident);
}

function membershipIdentity(tools: readonly string[]): string {
  return createHash("sha256")
    .update([...tools].sort().join("\0"))
    .digest("hex")
    .slice(0, 16);
}

function plural(count: number, singular: string, pluralForm = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : pluralForm}`;
}

// Renders a bounded sample list. Names are already sanitized to `[a-z0-9_]`, so they are safe for stderr.
function sampleList(names: readonly string[]): string {
  const shown = names.slice(0, MAX_DIAGNOSTIC_SAMPLES);
  const remainder = names.length - shown.length;
  return remainder > 0 ? `${shown.join(", ")} (+${remainder} more)` : shown.join(", ");
}

// Reports a registration pass that Pi aborted.
//
// Pi's `registerTool` is a synchronous replace-by-name whose only throw comes from a staleness
// check that is never reset, so a throw means the pass is over, not that one tool was rejected.
// `cause` is Pi-authored and bounded; no proxy schema, description, or body reaches stderr.
export function reportMcpRegistrationFatal(registered: number, attempted: number, cause: unknown): void {
  const causeText = truncateUtf8(
    cause instanceof Error ? cause.message : String(cause),
    MAX_DIAGNOSTIC_CAUSE_BYTES,
    SHORT_TRUNCATION_MARKER,
  );
  emitSafetyDiagnostic(
    "registration-fatal",
    `registration stopped after ${registered} of ${plural(attempted, "MCP tool")}; ` +
      `no further attempts will be made by this extension instance (${causeText}).`,
  );
}

// A successful pass clears any fatal incident left by an earlier extension instance, so the same
// failure is reported if it later recurs.
export function reportMcpRegistrationSuccess(): void {
  clearIncident("registration-fatal");
}

// Reports a discovery that yielded no registrable tool, so the silence is explained. A pass that did
// register something clears the incident, so a later recurrence is reported rather than suppressed as
// an unchanged message.
export function reportMcpCatalogOutcome(raw: number, registered: number): void {
  if (registered > 0) {
    clearIncident("empty-catalog");
    return;
  }
  emitSafetyDiagnostic(
    "empty-catalog",
    `no MCP tools were registered from ${plural(raw, "raw entry", "raw entries")} returned by the proxy.`,
    `empty-catalog:${raw}`,
  );
}

// Random per process, never logged and never persisted. The catalog identity it feeds is only ever
// compared with other identities computed in the same process, so it does not need to be stable
// across runs — and a random salt means even a leaked identity cannot be dictionary-attacked back to
// the credential, which a bare digest of a short key would not prevent.
const credentialIdentitySalt = randomBytes(32);

// A non-reversible stand-in for the credential material behind an MCP catalog.
//
// Covers the headers as well as the key: `LITELLM_HEADERS` can carry its own authorization material,
// so fingerprinting the key alone would leave secrets in the identity. Any change to either yields a
// different fingerprint, which is what the catalog needs in order to notice a credential change.
export function credentialFingerprint(apiKey: string, headers?: Record<string, string>): string {
  const material = JSON.stringify([
    apiKey,
    Object.entries(headers ?? {}).sort(([left], [right]) => (left < right ? -1 : left > right ? 1 : 0)),
  ]);
  return createHmac("sha256", credentialIdentitySalt).update(material).digest("hex").slice(0, 32);
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : undefined;
}

function boundedSignal(timeoutMs: number, parent?: AbortSignal): AbortSignal {
  const timeout = AbortSignal.timeout(timeoutMs);
  return parent ? AbortSignal.any([parent, timeout]) : timeout;
}

function stringValue(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function byteLength(value: string): number {
  return Buffer.byteLength(value, "utf8");
}

function truncateUtf8(value: string, maxBytes: number, marker: string): string {
  if (byteLength(value) <= maxBytes) return value;
  const markerBytes = byteLength(marker);
  const source = Buffer.from(value, "utf8");
  let end = Math.max(0, maxBytes - markerBytes);
  while (end > 0 && (source[end] & 0xc0) === 0x80) end -= 1;
  return `${source.subarray(0, end).toString("utf8")}${marker}`;
}

async function readBoundedText(response: Response, limit: number, surface: string): Promise<string> {
  if (!response.body) return "";
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > limit) {
        // Build the diagnostic and the error before cancelling: `cancel()` rejects with the stream's
        // stored error if the body errored concurrently, which would otherwise discard both.
        emitSafetyDiagnostic(`${surface}-body-cap`, `${surface} response exceeded its ${limit}-byte limit.`);
        const capError = new Error(`${surface} response exceeds its ${limit}-byte limit`);
        await reader.cancel().catch(() => undefined);
        throw capError;
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  // A response that stayed within the cap clears the incident, so a later breach on this surface
  // is reported again instead of being suppressed as an unchanged message.
  clearIncident(`${surface}-body-cap`);
  return Buffer.concat(
    chunks.map((chunk) => Buffer.from(chunk)),
    size,
  ).toString("utf8");
}

function parseJson(text: string, surface: string): unknown {
  try {
    return JSON.parse(text) as unknown;
  } catch {
    throw new Error(`${surface} returned invalid JSON`);
  }
}

function skipJsonWhitespace(text: string, start: number): number {
  let index = start;
  while (index < text.length && /\s/.test(text[index] ?? "")) index += 1;
  return index;
}

function jsonStringEnd(text: string, start: number): number | undefined {
  if (text[start] !== '"') return undefined;
  for (let index = start + 1; index < text.length; index += 1) {
    if (text[index] === "\\") {
      index += 1;
      continue;
    }
    if (text[index] === '"') return index + 1;
  }
  return undefined;
}

// Advances over one JSON value without materializing it. Full syntax validation remains JSON.parse's
// job; this scanner only needs enough structure to locate a top-level `tools` array before parsing.
function jsonValueEnd(text: string, start: number): number {
  const first = text[start];
  if (first === '"') return jsonStringEnd(text, start) ?? text.length;
  if (first !== "[" && first !== "{") {
    let index = start;
    while (index < text.length && !/[\s,}\]]/.test(text[index] ?? "")) index += 1;
    return index;
  }

  let depth = 1;
  for (let index = start + 1; index < text.length; index += 1) {
    const character = text[index];
    if (character === '"') {
      index = (jsonStringEnd(text, index) ?? text.length) - 1;
      continue;
    }
    if (character === "[" || character === "{") depth += 1;
    else if (character === "]" || character === "}") {
      depth -= 1;
      if (depth === 0) return index + 1;
    }
  }
  return text.length;
}

function assertJsonArrayEntryLimit(text: string, start: number): void {
  let depth = 1;
  let expectingEntry = true;
  let entries = 0;
  const recordEntry = (): void => {
    entries += 1;
    expectingEntry = false;
    if (entries > MAX_DISCOVERY_ENTRIES) {
      throw new Error(`MCP discovery exceeds its ${MAX_DISCOVERY_ENTRIES}-entry limit`);
    }
  };

  for (let index = start + 1; index < text.length; index += 1) {
    const character = text[index];
    if (character === '"') {
      if (depth === 1 && expectingEntry) recordEntry();
      index = (jsonStringEnd(text, index) ?? text.length) - 1;
      continue;
    }
    if (character === "[" || character === "{") {
      if (depth === 1 && expectingEntry) recordEntry();
      depth += 1;
      continue;
    }
    if (character === "]" || character === "}") {
      depth -= 1;
      if (depth === 0) return;
      continue;
    }
    if (depth !== 1) continue;
    if (character === ",") {
      expectingEntry = true;
      continue;
    }
    if (/\s/.test(character ?? "") || !expectingEntry) continue;
    recordEntry();
  }
}

// Rejects an excessive catalog before JSON.parse can allocate an object for every raw entry. Both
// documented response shapes are handled, including duplicate or escaped top-level `tools` keys.
// The response byte cap bounds this lexical pass, and the entry cap makes the hostile-array path
// stop after the first 10,001 primitive values rather than traversing the complete payload.
function assertDiscoveryEntryLimit(text: string): void {
  let index = skipJsonWhitespace(text, 0);
  if (text[index] === "[") {
    assertJsonArrayEntryLimit(text, index);
    return;
  }
  if (text[index] !== "{") return;
  index += 1;

  while (index < text.length) {
    index = skipJsonWhitespace(text, index);
    if (text[index] === "}") return;
    const keyStart = index;
    const keyEnd = jsonStringEnd(text, keyStart);
    if (keyEnd === undefined) return;
    index = skipJsonWhitespace(text, keyEnd);
    if (text[index] !== ":") return;
    index = skipJsonWhitespace(text, index + 1);

    let key: unknown;
    try {
      key = JSON.parse(text.slice(keyStart, keyEnd));
    } catch {
      return;
    }
    if (key === "tools" && text[index] === "[") assertJsonArrayEntryLimit(text, index);
    index = skipJsonWhitespace(text, jsonValueEnd(text, index));
    if (text[index] === ",") {
      index += 1;
      continue;
    }
    return;
  }
}

function parseDiscoveryJson(text: string): unknown {
  assertDiscoveryEntryLimit(text);
  return parseJson(text, "MCP discovery");
}

function normalizeMcpTool(value: unknown): LiteLLMMcpTool | undefined {
  const raw = asRecord(value) as RawLiteLLMMcpTool | undefined;
  if (!raw) return undefined;
  const name = stringValue(raw.name);
  const serverName =
    stringValue(raw.mcp_info?.server_name) ??
    stringValue(raw.server_name) ??
    stringValue(raw.mcp_info?.server_id) ??
    stringValue(raw.server_id);
  if (!name || !serverName) return undefined;
  const hasInputSchema = Object.hasOwn(raw, "inputSchema");
  const hasSnakeInputSchema = Object.hasOwn(raw, "input_schema");
  const suppliedSchema = hasInputSchema ? raw.inputSchema : hasSnakeInputSchema ? raw.input_schema : undefined;
  const inputSchema = asRecord(suppliedSchema);
  return {
    name,
    server_name: serverName,
    server_id: stringValue(raw.mcp_info?.server_id) ?? stringValue(raw.server_id) ?? serverName,
    description: stringValue(raw.description) ?? name,
    input_schema: inputSchema ?? {},
    // Present but not an object: malformed, as distinct from absent.
    ...(inputSchema === undefined && (hasInputSchema || hasSnakeInputSchema) ? { input_schema_malformed: true } : {}),
  };
}

// A bounded, injection-safe label for an entry that normalization rejected. Such an entry may have
// no usable name at all, so fall back to a positional placeholder rather than echoing proxy text.
function invalidToolLabel(value: unknown, index: number): string {
  const raw = asRecord(value) as RawLiteLLMMcpTool | undefined;
  const name = raw ? stringValue(raw.name) : undefined;
  const server = raw
    ? (stringValue(raw.mcp_info?.server_name) ?? stringValue(raw.server_name) ?? stringValue(raw.server_id))
    : undefined;
  // Sanitizing the charset is not enough: a proxy can supply a megabyte-long name and this label is
  // printed to stderr. Bound it like every other untrusted string in this module.
  const bounded = (value: string): string => truncateUtf8(value, MAX_DIAGNOSTIC_LABEL_BYTES, SHORT_TRUNCATION_MARKER);
  if (name && server) return bounded(`${sanitizeName(server)}_${sanitizeName(name)}`);
  if (name) return bounded(`unknown_server_${sanitizeName(name)}`);
  if (server) return bounded(`${sanitizeName(server)}_unnamed`);
  return `entry_${index}`;
}

export async function discoverMcpTools(
  baseUrl: string,
  apiKey: string,
  headers?: Record<string, string>,
  onProgress?: (message: string) => void,
  parentSignal?: AbortSignal,
  allowInsecureHttp = false,
): Promise<McpDiscovery> {
  parentSignal?.throwIfAborted();
  onProgress?.("Discovering MCP tools from server...");
  const signal = boundedSignal(LIST_TIMEOUT_MS, parentSignal);
  try {
    onProgress?.("Querying MCP tools/list endpoint...");
    const response = await fetch(`${normalizeBaseUrl(baseUrl, allowInsecureHttp)}/mcp-rest/tools/list`, {
      headers: {
        ...headers,
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      signal,
    });
    if (!response.ok) {
      onProgress?.(`MCP tools discovery failed with status ${response.status}`);
      throw new Error(`HTTP ${response.status}`);
    }
    const body = parseDiscoveryJson(await readBoundedText(response, MAX_DISCOVERY_BODY_BYTES, "MCP discovery"));
    const bodyRecord = asRecord(body);
    // An unrecognized body shape is not the same as an empty catalog, and reporting it as "0 raw
    // entries" would send an operator looking at the wrong thing.
    const rawTools = Array.isArray(body) ? body : Array.isArray(bodyRecord?.tools) ? bodyRecord.tools : undefined;
    if (rawTools === undefined) throw new Error("MCP discovery returned an unexpected body shape");
    if (rawTools.length > MAX_DISCOVERY_ENTRIES) {
      throw new Error(`MCP discovery exceeds its ${MAX_DISCOVERY_ENTRIES}-entry limit`);
    }
    onProgress?.(`Found ${rawTools.length} raw MCP tools, normalizing...`);
    const tools: LiteLLMMcpTool[] = [];
    const invalid: string[] = [];
    // Every raw entry lands in exactly one bucket, so later counts reconcile back to `raw`.
    rawTools.forEach((raw, index) => {
      const tool = normalizeMcpTool(raw);
      if (tool) tools.push(tool);
      else invalid.push(invalidToolLabel(raw, index));
    });
    onProgress?.(`Normalized ${tools.length} of ${rawTools.length} raw MCP tools`);
    return { raw: rawTools.length, tools, invalid };
  } catch (error) {
    onProgress?.("MCP tools discovery encountered an error");
    if (parentSignal?.aborted) throw parentSignal.reason;
    throw error;
  }
}

function mcpErrorMessage(value: unknown, depth = 0): string | undefined {
  if (depth > MAX_ERROR_WALK_DEPTH) return undefined;
  if (typeof value === "string") return value;
  if (Array.isArray(value)) {
    for (const entry of value) {
      const message = mcpErrorMessage(entry, depth + 1);
      if (message !== undefined) return message;
    }
    return undefined;
  }
  const record = asRecord(value);
  if (!record) return undefined;
  return (
    stringValue(record.message) ??
    (record.error !== value ? mcpErrorMessage(record.error, depth + 1) : undefined) ??
    stringValue(record.text)
  );
}

function serializeResult(value: unknown): string {
  const serialized = JSON.stringify(value, null, 2) ?? "null";
  return truncateUtf8(serialized, MAX_RESULT_BYTES, TRUNCATION_MARKER);
}

function mcpCallError(serverId: string, toolName: string, detail: string): Error {
  return new Error(
    truncateUtf8(`Error calling ${toolName} on ${serverId}: ${detail}`, MAX_RESULT_BYTES, TRUNCATION_MARKER),
  );
}

export async function executeMcpTool(
  baseUrl: string,
  apiKey: string,
  serverId: string,
  toolName: string,
  args: Record<string, unknown>,
  headers?: Record<string, string>,
  signalOrAllowInsecure?: AbortSignal | boolean,
  allowInsecureHttp = false,
): Promise<string> {
  // `allowInsecureHttp` occupied this positional argument before cancellation support was added.
  // Accept both forms so existing direct callers keep working while tool definitions can pass both.
  const parentSignal = typeof signalOrAllowInsecure === "boolean" ? undefined : signalOrAllowInsecure;
  const insecureHttpAllowed = typeof signalOrAllowInsecure === "boolean" ? signalOrAllowInsecure : allowInsecureHttp;
  parentSignal?.throwIfAborted();
  const signal = boundedSignal(CALL_TIMEOUT_MS, parentSignal);
  let response: Response;
  try {
    response = await fetch(`${normalizeBaseUrl(baseUrl, insecureHttpAllowed)}/mcp-rest/tools/call`, {
      method: "POST",
      headers: {
        ...headers,
        Authorization: `Bearer ${apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ server_id: serverId, name: toolName, arguments: args }),
      signal,
    });
    if (!response.ok) throw mcpCallError(serverId, toolName, `HTTP ${response.status}`);
    const body = parseJson(await readBoundedText(response, MAX_CALL_BODY_BYTES, "MCP tool call"), "MCP tool call");
    const bodyRecord = asRecord(body);
    if (bodyRecord?.error != null) {
      throw mcpCallError(serverId, toolName, mcpErrorMessage(bodyRecord.error) ?? "MCP error");
    }
    const result = bodyRecord && "result" in bodyRecord ? bodyRecord.result : body;
    const resultRecord = asRecord(result);
    if (resultRecord?.isError === true || bodyRecord?.isError === true) {
      const message =
        mcpErrorMessage(resultRecord?.content ?? bodyRecord?.content ?? result) ?? serializeResult(result);
      throw mcpCallError(serverId, toolName, message);
    }
    return serializeResult(result);
  } catch (error) {
    if (parentSignal?.aborted) throw parentSignal.reason;
    throw error;
  }
}

function sanitizeName(name: string): string {
  const sanitized = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return sanitized || "tool";
}

function toolIdentity(tool: LiteLLMMcpTool): string {
  return JSON.stringify([tool.server_id ?? tool.server_name, tool.server_name, tool.name]);
}

// Derives a Pi tool name from the tool's identity alone.
//
// The hash is unconditional so the name is a pure function of `server_id`/`server_name`/`name`.
// A conditional hash would make a survivor's name depend on which *other* tools happened to be in
// the same catalog, so adding or removing a sibling would rename it — and because Pi cannot
// unregister, the old name would linger and the model would see one tool twice.
function buildPiToolName(tool: LiteLLMMcpTool): string {
  const hash = createHash("sha256").update(toolIdentity(tool)).digest("hex").slice(0, TOOL_NAME_HASH_LENGTH);
  const base = `mcp_${sanitizeName(tool.server_name)}_${sanitizeName(tool.name)}`;
  return `${base.slice(0, MAX_TOOL_NAME_LENGTH - hash.length - 1)}_${hash}`;
}

function schemaDepth(value: unknown, depth = 0): number {
  if (!value || typeof value !== "object") return depth;
  if (depth > MAX_SCHEMA_DEPTH) return depth;
  const values = Array.isArray(value) ? value : Object.values(value as Record<string, unknown>);
  return values.reduce((maximum, child) => Math.max(maximum, schemaDepth(child, depth + 1)), depth);
}

const DIRECT_SCHEMA_KEYWORDS = [
  "additionalProperties",
  "contains",
  "contentSchema",
  "else",
  "if",
  "not",
  "propertyNames",
  "then",
  "unevaluatedItems",
  "unevaluatedProperties",
] as const;
const SCHEMA_ARRAY_KEYWORDS = ["allOf", "anyOf", "oneOf", "prefixItems"] as const;
const SCHEMA_MAP_KEYWORDS = ["$defs", "definitions", "dependentSchemas", "patternProperties", "properties"] as const;

// TypeBox compiles a schema's `pattern` and the keys of `patternProperties` into a backtracking
// RegExp and executes them with no time limit, so a proxy-supplied expression is an unbounded CPU
// sink. A `$ref` resolves an arbitrary JSON pointer, which means a regex can be parked under any key
// at all — including data-only positions such as `default` or `examples`. Enumerating "positions
// where a subschema may appear" is therefore not a sound bound. We instead walk the entire supplied
// document, every key and every value, and refuse to vouch for anything we could not fully inspect.
const UNSAFE_REGEX_KEY = "pattern";
const UNSAFE_REGEX_MAP_KEY = "patternProperties";
const NONLOCAL_REF_KEYS = ["$dynamicRef", "$recursiveRef"] as const;
// These keywords change TypeBox's reference base or introduce aliases. Rather than duplicate its
// URI and dynamic-scope resolver, fail closed whenever an untrusted schema defines one.
const REF_SCOPE_KEYS = new Set(["$id", "$anchor", "$dynamicAnchor", "$recursiveAnchor"]);
const UNSAFE_POINTER_TOKENS = new Set(["__proto__", "constructor", "prototype"]);
const MAX_SCAN_NODES = 20_000;
const MAX_REF_HOPS = 32;
const POINTER_MISSING = Symbol("pointer-missing");
// Keywords whose immediate children are keyed by *name* rather than by keyword. Inside one of these
// a key like `patternProperties` is the name of a tool argument, not a constraint, so the keyword
// checks must not fire on it. This is the only context the scan needs; it does not restrict where
// the scan looks, only how a key is interpreted.
const NAME_KEYED_KEYWORDS = new Set([
  "properties",
  "patternProperties",
  "$defs",
  "definitions",
  "dependentSchemas",
  "dependencies",
]);

// Why a supplied schema could not be proven safe to hand to the validator.
export type SchemaHazard =
  | "regex"
  | "malformed-ref"
  | "nonlocal-ref"
  | "unresolvable-ref"
  | "resolver-scope"
  | "ref-cycle"
  | "budget"
  | "cycle";

// Resolves a JSON pointer fragment against the supplied document. Returns POINTER_MISSING rather
// than undefined so that a target whose value genuinely is `undefined` cannot be mistaken for a hit.
function resolvePointer(root: unknown, fragment: string): unknown {
  if (fragment === "") return root;
  // A plain `#name` is an $anchor reference, which cannot be resolved without an anchor index.
  if (!fragment.startsWith("/")) return POINTER_MISSING;
  let current: unknown = root;
  for (const rawToken of fragment.slice(1).split("/")) {
    const token = rawToken.replaceAll("~1", "/").replaceAll("~0", "~");
    // TypeBox's pointer lookup refuses these keys even when they are own properties.
    if (UNSAFE_POINTER_TOKENS.has(token)) return POINTER_MISSING;
    if (Array.isArray(current)) {
      // TypeBox indexes arrays with the pointer token itself. Accept only the canonical decimal
      // spellings that can name JSON array elements; Number("01") would incorrectly alias index 1.
      if (!/^(0|[1-9]\d*)$/.test(token)) return POINTER_MISSING;
      const index = Number(token);
      if (!Number.isSafeInteger(index) || index >= current.length) return POINTER_MISSING;
      current = current[index];
      continue;
    }
    const record = asRecord(current);
    if (!record || !Object.hasOwn(record, token)) return POINTER_MISSING;
    current = record[token];
  }
  return current;
}

// Follows a local `#` reference chain and reports whether the validator could actually use it.
//
// Covering the whole document proves no regex hides behind a pointer, but it does not prove the
// pointer names a subschema at all. TypeBox resolves the pointer and then treats whatever it finds
// as a schema, so `#/description` makes it throw on a string, a mutual `$defs` pair makes it recurse
// until the stack is gone, and a missing target silently resolves to `false` — a tool that can never
// validate any argument. Object identity cannot see any of these, because each pointer hop is a
// different object.
function findLocalRefHazard(
  root: Record<string, unknown>,
  ref: string,
  schemaPositionNodes: WeakSet<object>,
): SchemaHazard | undefined {
  const activePointers = new Set<string>();
  const checkedPointers = new Set<string>();
  // This is intentionally shared across the complete traversal from one root ref so a broad
  // combinator graph cannot evade the same bound by branching.
  let referenceCount = 0;

  const visitRef = (pointer: string): SchemaHazard | undefined => {
    if (activePointers.has(pointer)) return "ref-cycle";
    if (checkedPointers.has(pointer)) return undefined;
    referenceCount += 1;
    if (referenceCount > MAX_REF_HOPS) return "budget";
    // TypeBox percent-decodes URI fragments before resolving their JSON pointer. Keeping encoded
    // refs would require exactly mirroring that decoding (including failures), so refuse them.
    if (pointer.includes("%")) return "unresolvable-ref";
    const target = resolvePointer(root, pointer.slice(1));
    if (target === POINTER_MISSING) return "unresolvable-ref";
    // A boolean is always a complete schema, independent of where it appears in the document.
    if (typeof target === "boolean") return undefined;
    const record = asRecord(target);
    // Name-keyed containers are not schemas. Requiring an object target to have been visited where
    // keys are interpreted as schema keywords prevents a ref from turning one into a schema behind
    // the scan's back (for example, `#/dependencies`).
    if (!record || !schemaPositionNodes.has(record) || !hasValidSchemaShape(record)) return "unresolvable-ref";

    activePointers.add(pointer);
    const hazard = visitSameInstanceSchemas(record);
    activePointers.delete(pointer);
    if (!hazard) checkedPointers.add(pointer);
    return hazard;
  };

  // TypeBox evaluates these applicators against the same instance as their containing schema.
  // Following only those edges catches unproductive reference cycles without rejecting ordinary
  // recursive object/array schemas, whose property or item edge advances to a nested value.
  const visitSameInstanceSchemas = (schema: Record<string, unknown>): SchemaHazard | undefined => {
    if (typeof schema.$ref === "string") {
      const hazard = schema.$ref.startsWith("#") ? visitRef(schema.$ref) : "nonlocal-ref";
      if (hazard) return hazard;
    }
    for (const keyword of ["not", "if", "then", "else"] as const) {
      const child = asRecord(schema[keyword]);
      if (!child) continue;
      const hazard = visitSameInstanceSchemas(child);
      if (hazard) return hazard;
    }
    for (const keyword of ["allOf", "anyOf", "oneOf"] as const) {
      const schemas = schema[keyword];
      if (!Array.isArray(schemas)) continue;
      for (const child of schemas) {
        const record = asRecord(child);
        if (!record) continue;
        const hazard = visitSameInstanceSchemas(record);
        if (hazard) return hazard;
      }
    }
    for (const keyword of ["dependentSchemas", "dependencies"] as const) {
      const map = asRecord(schema[keyword]);
      if (!map) continue;
      for (const child of Object.values(map)) {
        const record = asRecord(child);
        if (!record) continue;
        const hazard = visitSameInstanceSchemas(record);
        if (hazard) return hazard;
      }
    }
    return undefined;
  };

  return visitRef(ref);
}

function schemaPositionNodes(root: Record<string, unknown>): WeakSet<object> {
  const positions = new WeakSet<object>();
  const seen = new WeakSet<object>();

  const visitSchema = (value: unknown): void => {
    if (typeof value === "boolean") return;
    const schema = asRecord(value);
    if (!schema || seen.has(schema)) return;
    seen.add(schema);
    positions.add(schema);

    for (const keyword of DIRECT_SCHEMA_KEYWORDS) visitSchema(schema[keyword]);
    for (const keyword of SCHEMA_MAP_KEYWORDS) {
      const map = asRecord(schema[keyword]);
      if (map) for (const child of Object.values(map)) visitSchema(child);
    }
    const dependencies = asRecord(schema.dependencies);
    if (dependencies) {
      for (const child of Object.values(dependencies)) {
        // Draft-07 dependencies may also be arrays of property names; only schema-valued entries
        // establish independently reachable schema positions.
        if (typeof child === "boolean" || asRecord(child)) visitSchema(child);
      }
    }
    for (const keyword of SCHEMA_ARRAY_KEYWORDS) {
      const schemas = schema[keyword];
      if (Array.isArray(schemas)) for (const child of schemas) visitSchema(child);
    }
    const items = schema.items;
    if (Array.isArray(items)) for (const child of items) visitSchema(child);
    else visitSchema(items);
  };

  visitSchema(root);
  return positions;
}

// Walks the whole untrusted schema graph looking for anything the validator could turn into a
// regex, plus references we cannot resolve within the document we were given.
//
// Bounded three ways so a hostile document cannot exhaust us while we inspect it: a depth cap, a
// node budget, and identity-based cycle detection. Hitting any bound returns a hazard rather than
// `undefined`, because an unfinished scan proves nothing.
//
// A local `#/...` ref is only accepted when the walk covers its target *and* the target resolves to
// a usable subschema, with the reference chain proven acyclic. Any other ref form is a hazard.
export function findSchemaHazard(root: Record<string, unknown>): SchemaHazard | undefined {
  const seenKeywordObjects = new WeakSet<object>();
  const seenNameMaps = new WeakSet<object>();
  const active = new WeakSet<object>();
  const localRefs: string[] = [];
  let budget = MAX_SCAN_NODES;

  const walk = (value: unknown, depth: number, keysAreNames: boolean): SchemaHazard | undefined => {
    if (budget-- <= 0) return "budget";
    if (depth > MAX_SCHEMA_DEPTH) return "budget";
    if (value === null || typeof value !== "object") return undefined;
    if (active.has(value)) return "cycle";
    // Shared identities need inspection once per interpretation context: a `pattern` key is an
    // argument name under `properties`, but the same object is hazardous if also used as a schema.
    const seen = keysAreNames ? seenNameMaps : seenKeywordObjects;
    if (seen.has(value)) return undefined;
    seen.add(value);
    active.add(value);

    if (Array.isArray(value)) {
      for (const child of value) {
        const hazard = walk(child, depth + 1, false);
        if (hazard) return hazard;
      }
      active.delete(value);
      return undefined;
    }

    for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
      if (!keysAreNames) {
        // `$id` changes the base against which fragment-only refs resolve, while the anchor
        // keywords introduce aliases and dynamic scope. TypeBox applies those semantics while
        // compiling, so the pointer-only safety resolver must not accept schemas that define them.
        if (REF_SCOPE_KEYS.has(key)) return "resolver-scope";
        // Any `pattern` or `patternProperties` at a keyword position, whatever its value type. This
        // TypeBox only compiles a string-valued `pattern`, so a non-string one is not a CPU risk —
        // but the expression text would still land in the schema Pi holds and forward to the
        // provider, so refusing to vouch for it keeps the stated invariant literally true and does
        // not depend on an upstream type guard staying as it is. A key that is an argument *name*
        // is excluded by `keysAreNames` before reaching here.
        if (key === UNSAFE_REGEX_KEY || key === UNSAFE_REGEX_MAP_KEY) return "regex";
        // A pointer outside the supplied document cannot be inspected at all. A local one is only
        // safe if it also resolves to something the validator can use as a schema.
        if (key === "$ref") {
          if (typeof child !== "string") return "malformed-ref";
          if (!child.startsWith("#")) return "nonlocal-ref";
          // TypeBox decodes percent escapes before pointer lookup. Reject encoded local refs rather
          // than risk validating a different target from the one TypeBox will compile.
          if (child.includes("%")) return "unresolvable-ref";
          // Resolve after the complete walk so acceptance cannot depend on whether the referenced
          // schema appeared before or after this `$ref` in object insertion order.
          localRefs.push(child);
        }
        if ((NONLOCAL_REF_KEYS as readonly string[]).includes(key)) {
          if (typeof child !== "string") return "malformed-ref";
          return "nonlocal-ref";
        }
      }
      const hazard = walk(child, depth + 1, !keysAreNames && NAME_KEYED_KEYWORDS.has(key));
      if (hazard) return hazard;
    }
    active.delete(value);
    return undefined;
  };

  const walkHazard = walk(root, 0, false);
  if (walkHazard) return walkHazard;
  const validPositions = schemaPositionNodes(root);
  for (const ref of localRefs) {
    const refHazard = findLocalRefHazard(root, ref, validPositions);
    if (refHazard) return refHazard;
  }
  return undefined;
}

function isValidSchema(value: unknown): boolean {
  if (typeof value === "boolean") return true;
  const schema = asRecord(value);
  return schema !== undefined && hasValidSchemaShape(schema);
}

function isValidSchemaMap(value: unknown): boolean {
  const schemas = asRecord(value);
  return schemas !== undefined && Object.values(schemas).every(isValidSchema);
}

function isValidSchemaArray(value: unknown): boolean {
  return Array.isArray(value) && value.every(isValidSchema);
}

function isStringArray(value: unknown): boolean {
  return Array.isArray(value) && value.every((item) => typeof item === "string");
}

function isValidDependencies(value: unknown): boolean {
  const dependencies = asRecord(value);
  return (
    dependencies !== undefined &&
    Object.values(dependencies).every((entry) => isValidSchema(entry) || isStringArray(entry))
  );
}

// Shape validation is a separate concern from the hazard scan above: it exists to avoid handing the
// validator a structurally broken document, and it deliberately only looks where a subschema may
// legally appear so that data positions are not misread as constraints.
function hasValidSchemaShape(record: Record<string, unknown>): boolean {
  if ("required" in record && !isStringArray(record.required)) return false;
  if ("dependencies" in record && !isValidDependencies(record.dependencies)) return false;
  if (SCHEMA_MAP_KEYWORDS.some((keyword) => keyword in record && !isValidSchemaMap(record[keyword]))) return false;
  if (SCHEMA_ARRAY_KEYWORDS.some((keyword) => keyword in record && !isValidSchemaArray(record[keyword]))) return false;
  if (DIRECT_SCHEMA_KEYWORDS.some((keyword) => keyword in record && !isValidSchema(record[keyword]))) return false;
  const items = record.items;
  return !("items" in record) || isValidSchema(items) || isValidSchemaArray(items);
}

// The extension-owned parameter shape. Nothing in it originates from the proxy.
function trustedEnvelope(): TSchema {
  return Type.Object({
    args: Type.Record(Type.String(), Type.Unknown(), { description: "Tool arguments as key-value pairs" }),
  });
}

type BuiltParameters = {
  parameters: TSchema;
  syntheticArgsEnvelope: boolean;
  // Set when the proxy supplied a usable schema that we replaced with the trusted envelope.
  degraded?: McpDegradeReason;
};

function buildParameters(tool: LiteLLMMcpTool): BuiltParameters | McpDropReason {
  // An `inputSchema` that was present but not a JSON object is a malformed tool, not a schemaless
  // one, and must not be silently upgraded to the permissive envelope.
  if (tool.input_schema_malformed) return "invalid-schema";
  const inputSchema = tool.input_schema;
  // Absent or empty schema: the proxy asked for no parameters, which the trusted envelope models
  // exactly. This is the normal schemaless path, not a degradation.
  if (Object.keys(inputSchema).length === 0) {
    return { parameters: trustedEnvelope(), syntheticArgsEnvelope: true };
  }
  if (inputSchema.type !== "object") return "invalid-schema";
  let serialized: string;
  try {
    serialized = JSON.stringify(inputSchema);
  } catch {
    return "invalid-schema";
  }
  if (byteLength(serialized) > MAX_SCHEMA_BYTES || schemaDepth(inputSchema) > MAX_SCHEMA_DEPTH) {
    return "invalid-schema";
  }
  if (!hasValidSchemaShape(inputSchema)) return "invalid-schema";
  // Structurally fine, but we will only forward it if the whole graph is provably free of
  // proxy-supplied regexes and unresolvable references. Otherwise the tool still registers, with
  // the trusted envelope standing in for its schema.
  if (findSchemaHazard(inputSchema) !== undefined) {
    return { parameters: trustedEnvelope(), syntheticArgsEnvelope: true, degraded: "schema-envelope" };
  }
  return { parameters: Type.Unsafe(inputSchema), syntheticArgsEnvelope: false };
}

export function prepareTools(discovery: McpDiscovery): {
  prepared: PreparedTool[];
  report: McpPreparationReport;
} {
  const drops = new Map<McpDropReason, string[]>();
  const degrades = new Map<McpDegradeReason, string[]>();
  const recordDrop = (reason: McpDropReason, label: string): void => {
    const labels = drops.get(reason);
    if (labels) labels.push(label);
    else drops.set(reason, [label]);
  };

  for (const label of discovery.invalid) recordDrop("invalid-tool", label);

  const accepted: Array<Omit<PreparedTool, "name"> & { degraded?: McpDegradeReason }> = [];
  const acceptedIdentities = new Set<string>();
  for (const tool of discovery.tools) {
    const built = buildParameters(tool);
    if (typeof built === "string") {
      recordDrop(built, buildPiToolName(tool));
      continue;
    }
    const identity = toolIdentity(tool);
    if (acceptedIdentities.has(identity)) {
      recordDrop("duplicate-identity", buildPiToolName(tool));
      continue;
    }
    acceptedIdentities.add(identity);
    accepted.push({
      tool,
      parameters: built.parameters,
      syntheticArgsEnvelope: built.syntheticArgsEnvelope,
      degraded: built.degraded,
    });
  }

  const candidates = accepted.slice(0, MAX_REGISTERED_TOOLS);
  const overflowTools = accepted.slice(MAX_REGISTERED_TOOLS).map(({ tool }) => buildPiToolName(tool));
  const finalNames = new Set<string>();
  const prepared: PreparedTool[] = [];
  for (const candidate of candidates) {
    // Membership-independent: the name depends only on this tool's identity.
    const name = buildPiToolName(candidate.tool);
    if (finalNames.has(name)) {
      recordDrop("name-collision", name);
      continue;
    }
    finalNames.add(name);
    const { degraded, ...preparedCandidate } = candidate;
    prepared.push({ ...preparedCandidate, name });
    if (degraded) {
      const labels = degrades.get(degraded);
      if (labels) labels.push(name);
      else degrades.set(degraded, [name]);
    }
  }

  return {
    prepared,
    report: {
      discovered: discovery.raw,
      prepared: prepared.length,
      // Counting only hazard degradations would tell an operator that schemaless tools retain a
      // proxy-specific argument contract; they have only the extension-owned args-envelope shape.
      enveloped: prepared.filter((tool) => tool.syntheticArgsEnvelope).length,
      overflow: overflowTools.length,
      overflowTools,
      dropped: [...drops.entries()].map(([reason, tools]) => ({ reason, tools })),
      degraded: [...degrades.entries()].map(([reason, tools]) => ({ reason, tools })),
    },
  };
}

function emitPreparationDiagnostics(report: McpPreparationReport): void {
  const seen = new Set<string>();
  const emitClass = (reason: McpIncidentReason, message: string, tools: string[]): void => {
    seen.add(reason);
    // Identity covers the full membership, so a change beyond the printed sample still re-reports.
    emitSafetyDiagnostic(reason, message, `${tools.length}:${membershipIdentity(tools)}`);
  };
  for (const { reason, tools } of report.dropped) {
    emitClass(
      reason,
      `dropped ${plural(tools.length, "MCP tool")} with ${DROP_REASON_TEXT[reason]}: ${sampleList(tools)}.`,
      tools,
    );
  }
  for (const { reason, tools } of report.degraded) {
    emitClass(
      reason,
      `kept ${plural(tools.length, "MCP tool")} but replaced ${tools.length === 1 ? "its" : "their"} schema with a ` +
        `safe args envelope, because of ${DEGRADE_REASON_TEXT[reason]}: ${sampleList(tools)}.`,
      tools,
    );
  }
  if (report.overflow > 0) {
    seen.add("tool-cap");
    emitSafetyDiagnostic(
      "tool-cap",
      `ignoring ${plural(report.overflow, "MCP tool")} beyond the ${MAX_REGISTERED_TOOLS}-tool limit.`,
      `${report.overflow}:${membershipIdentity(report.overflowTools)}`,
    );
  }
  // Clear classes that did not recur, so the same incident is reported again if it comes back
  // after a clean refresh instead of being suppressed as an unchanged identity.
  for (const reason of [...Object.keys(INCIDENT_REASON_TEXT), "tool-cap"]) {
    if (!seen.has(reason)) clearIncident(reason);
  }
}

export async function createMcpToolDefinitions(
  getAuth: (ctx?: ExtensionContext) => Promise<LiteLLMRuntimeAuth>,
  onProgress?: (message: string) => void,
  signal?: AbortSignal,
): Promise<{ definitions: ToolDefinition[]; report: McpPreparationReport }> {
  const discoveryAuth = await getAuth();
  const discovery = await discoverMcpTools(
    discoveryAuth.baseUrl,
    discoveryAuth.apiKey,
    discoveryAuth.headers,
    onProgress,
    signal,
    discoveryAuth.allowInsecureHttp,
  );
  const { prepared, report } = prepareTools(discovery);
  emitPreparationDiagnostics(report);
  const lostTotal = report.dropped.reduce((total, entry) => total + entry.tools.length, 0) + report.overflow;
  const lostDetail = [
    ...report.dropped.map((entry) => `${entry.reason}=${entry.tools.length}`),
    ...(report.overflow > 0 ? [`tool-cap=${report.overflow}`] : []),
  ].join(", ");
  // raw = prepared + lost, so an operator can always account for every entry the proxy returned.
  onProgress?.(
    `Prepared ${report.prepared} of ${report.discovered} raw MCP tools ` +
      `(${report.enveloped} kept with a safe args envelope); lost ${lostTotal}` +
      `${lostDetail ? ` (${lostDetail})` : ""}`,
  );

  const definitions = prepared.map(({ tool: mcpTool, name, parameters, syntheticArgsEnvelope }) => {
    const description = truncateUtf8(
      `${mcpTool.description} (via ${mcpTool.server_name} MCP server)`,
      MAX_DESCRIPTION_BYTES,
      DESCRIPTION_TRUNCATION_MARKER,
    );
    const promptSnippet = truncateUtf8(
      `${mcpTool.description} via ${mcpTool.server_name} MCP server`,
      MAX_DESCRIPTION_BYTES,
      DESCRIPTION_TRUNCATION_MARKER,
    );

    // `label` and the `details` fields below are proxy-supplied, so they are bounded like every other
    // untrusted string that reaches Pi's UI or the model.
    const label = truncateUtf8(`${mcpTool.server_name}: ${mcpTool.name}`, MAX_LABEL_BYTES, SHORT_TRUNCATION_MARKER);
    const boundedDetail = (value: string): string => truncateUtf8(value, MAX_DETAIL_BYTES, SHORT_TRUNCATION_MARKER);

    return defineTool({
      name,
      label,
      description,
      promptSnippet,
      executionMode: "parallel",
      parameters,
      async execute(_toolCallId, params: Static<typeof parameters>, toolSignal, _onUpdate, ctx) {
        const auth = await getAuth(ctx);
        const rawParams = params as Record<string, unknown>;
        const args = syntheticArgsEnvelope ? asRecord(rawParams.args) : rawParams;
        if (!args) throw new Error("Synthetic MCP tool arguments must contain an object-valued args property");
        const text = await executeMcpTool(
          auth.baseUrl,
          auth.apiKey,
          mcpTool.server_id ?? mcpTool.server_name,
          mcpTool.name,
          args,
          auth.headers,
          toolSignal,
          auth.allowInsecureHttp,
        );
        return {
          content: [{ type: "text", text }],
          details: {
            server: boundedDetail(mcpTool.server_name),
            serverId: boundedDetail(mcpTool.server_id ?? mcpTool.server_name),
            tool: boundedDetail(mcpTool.name),
          },
        };
      },
    });
  });

  return { definitions, report };
}
