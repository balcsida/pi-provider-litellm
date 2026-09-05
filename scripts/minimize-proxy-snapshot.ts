import { readFile, writeFile } from "node:fs/promises";
import { isRecord } from "../src/backend-identity.js";
import { acceptanceOracle, errorClass } from "./probe-proxy.js";

const API_BASE_PLACEHOLDER = "https://example.invalid";
const MODEL_PARAMS = ["model", "custom_llm_provider", "api_base", "api_version", "allowed_openai_params"] as const;
const MODEL_INFO = [
  "base_model",
  "litellm_provider",
  "mode",
  "supported_endpoints",
  "supported_openai_params",
  "input_cost_per_token",
  "output_cost_per_token",
  "cache_read_input_token_cost",
  "cache_creation_input_token_cost",
  "max_input_tokens",
  "max_output_tokens",
  "supports_reasoning",
  "supports_vision",
  "supports_minimal_reasoning_effort",
  "supports_low_reasoning_effort",
  "supports_xhigh_reasoning_effort",
  "supports_max_reasoning_effort",
] as const;
const FORBIDDEN_FIXTURE_FIELDS = new Set(["api_key", "extra_headers", "access_groups", "tenant"]);
const GROUP_FIELDS = [
  "model_group",
  "max_input_tokens",
  "max_output_tokens",
  "input_cost_per_token",
  "output_cost_per_token",
  "mode",
  "supports_vision",
  "supports_reasoning",
] as const;

type RecordValue = Record<string, unknown>;

function pick(value: unknown, fields: readonly string[]): RecordValue {
  if (!isRecord(value)) return {};
  return Object.fromEntries(fields.filter((field) => value[field] !== undefined).map((field) => [field, value[field]]));
}

export function minimizeModelInfo(value: unknown): { data: RecordValue[] } {
  const rows = isRecord(value) && Array.isArray(value.data) ? value.data : [];
  return {
    data: rows.filter(isRecord).map((row) => {
      const litellmParams = pick(row.litellm_params, MODEL_PARAMS);
      if (typeof litellmParams.api_base === "string") litellmParams.api_base = API_BASE_PLACEHOLDER;
      if (Array.isArray(litellmParams.allowed_openai_params)) {
        litellmParams.allowed_openai_params = litellmParams.allowed_openai_params.filter(
          (entry) => typeof entry === "string" && entry !== "extra_headers",
        );
      }
      const modelInfo = pick(row.model_info, MODEL_INFO);
      if (Array.isArray(modelInfo.supported_openai_params)) {
        modelInfo.supported_openai_params = modelInfo.supported_openai_params.filter(
          (entry) => typeof entry === "string" && entry !== "extra_headers",
        );
      }
      return {
        ...(typeof row.model_name === "string" ? { model_name: row.model_name } : {}),
        litellm_params: litellmParams,
        model_info: modelInfo,
      };
    }),
  };
}

export function minimizeModelGroups(value: unknown): { data: RecordValue[] } {
  const rows = isRecord(value) && Array.isArray(value.data) ? value.data : [];
  return {
    data: rows.filter(isRecord).map((row) => {
      const minimized = pick(row, GROUP_FIELDS);
      if (Array.isArray(row.supported_openai_params)) {
        minimized.supported_openai_params = row.supported_openai_params.filter(
          (entry) => typeof entry === "string" && entry !== "extra_headers",
        );
      }
      return minimized;
    }),
  };
}

export function assertSafeFixture(value: unknown): void {
  if (Array.isArray(value)) {
    for (const entry of value) assertSafeFixture(entry);
    return;
  }
  if (!isRecord(value)) return;
  for (const [key, entry] of Object.entries(value)) {
    if (FORBIDDEN_FIXTURE_FIELDS.has(key)) throw new Error(`Unsafe fixture field: ${key}`);
    if (key === "api_base" && entry !== API_BASE_PLACEHOLDER) {
      throw new Error(`Unsafe fixture api_base: ${String(entry)}`);
    }
    assertSafeFixture(entry);
  }
}

export function deriveAcceptanceOracle(value: unknown): unknown[] {
  if (!Array.isArray(value)) throw new Error("Probe results must be an array");
  return value.map((entry) => {
    if (!isRecord(entry) || typeof entry.status !== "number") throw new Error("Invalid probe result");
    const rawError = entry.error;
    const classified =
      (typeof entry.errorClass === "string" && entry.errorClass) || errorClass(rawError) || errorClass(entry);
    const accepted = acceptanceOracle({ status: entry.status, errorClass: classified });
    return {
      path: entry.path,
      model: entry.model,
      level: entry.level,
      status: entry.status,
      ...(classified ? { errorClass: classified } : {}),
      accepted,
    };
  });
}

async function main(args: string[]): Promise<void> {
  const [mode, input, output] = args;
  if (!mode || !input || !output) {
    throw new Error("Usage: minimize-proxy-snapshot <model-info|model-groups|acceptance> <input> <output>");
  }
  const parsed: unknown = JSON.parse(await readFile(input, "utf8"));
  const result =
    mode === "model-info"
      ? minimizeModelInfo(parsed)
      : mode === "model-groups"
        ? minimizeModelGroups(parsed)
        : mode === "acceptance"
          ? deriveAcceptanceOracle(parsed)
          : undefined;
  if (!result) throw new Error(`Unknown mode: ${mode}`);
  assertSafeFixture(result);
  await writeFile(output, `${JSON.stringify(result, null, 2)}\n`);
}

if (import.meta.main) {
  main(process.argv.slice(2)).catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
