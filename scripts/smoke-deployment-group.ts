export const GROUPED_MODEL_NAME = "grouped-vidaimock";
const EXPECTED_MODES = ["chat", "responses"] as const;
const DEFAULT_TIMEOUT_MS = 3000;

type JsonObject = Record<string, unknown>;

export type GroupedDeployment = JsonObject & {
  model_name: string;
  model_info: JsonObject & {
    id: string;
    mode: string;
    supported_openai_params: unknown[];
  };
  litellm_params: JsonObject & {
    allowed_openai_params: unknown[];
  };
};

function isObject(value: unknown): value is JsonObject {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function validateGroupedDeployments(payload: unknown): GroupedDeployment[] {
  if (!isObject(payload) || !Array.isArray(payload.data)) {
    throw new Error("/model/info response is missing data rows");
  }

  const rows = payload.data.filter((row): row is JsonObject => isObject(row) && row.model_name === GROUPED_MODEL_NAME);
  if (rows.length !== 2) {
    throw new Error(`expected two grouped deployments, received ${rows.length}`);
  }

  const deployments = rows.map((row) => {
    const modelInfo = row.model_info;
    const litellmParams = row.litellm_params;
    if (!isObject(modelInfo) || typeof modelInfo.id !== "string" || modelInfo.id.length === 0) {
      throw new Error("grouped deployment is missing model_info.id");
    }
    if (!Array.isArray(modelInfo.supported_openai_params)) {
      throw new Error("grouped deployment is missing supported_openai_params");
    }
    if (!isObject(litellmParams) || !Array.isArray(litellmParams.allowed_openai_params)) {
      throw new Error("grouped deployment is missing allowed_openai_params");
    }
    return row as GroupedDeployment;
  });

  const ids = new Set(deployments.map((row) => row.model_info.id));
  if (ids.size !== deployments.length) {
    throw new Error("grouped deployments must preserve unique model_info.id values");
  }

  const modes = new Set(deployments.map((row) => row.model_info.mode));
  if (modes.size !== EXPECTED_MODES.length || EXPECTED_MODES.some((mode) => !modes.has(mode))) {
    throw new Error("grouped deployments must preserve exactly chat and responses modes");
  }

  return deployments;
}

export function summarizeGroupedDeployments(deployments: GroupedDeployment[]) {
  return deployments.map((deployment) => ({
    id: deployment.model_info.id,
    mode: deployment.model_info.mode,
    supported_openai_params: deployment.model_info.supported_openai_params,
    allowed_openai_params: deployment.litellm_params.allowed_openai_params,
  }));
}

export async function captureGroupedDeployments(
  baseUrl: string,
  apiKey: string,
  timeoutMs = DEFAULT_TIMEOUT_MS,
): Promise<GroupedDeployment[]> {
  const response = await fetch(`${baseUrl.replace(/\/+$/, "")}/model/info`, {
    headers: { Authorization: `Bearer ${apiKey}` },
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!response.ok) throw new Error(`/model/info returned ${response.status}`);
  return validateGroupedDeployments(await response.json());
}

export async function captureGroupedDeploymentsFromEnv(
  env: NodeJS.ProcessEnv = process.env,
): Promise<GroupedDeployment[]> {
  const baseUrl = env.LITELLM_BASE_URL?.trim();
  const apiKey = env.LITELLM_API_KEY?.trim();
  if (!baseUrl || !apiKey) throw new Error("LITELLM_BASE_URL and LITELLM_API_KEY must be set");
  return captureGroupedDeployments(baseUrl, apiKey);
}

if (import.meta.main) {
  captureGroupedDeploymentsFromEnv()
    .then((rows) => console.log(JSON.stringify(summarizeGroupedDeployments(rows), null, 2)))
    .catch((error) => {
      console.error(error instanceof Error ? error.message : error);
      process.exitCode = 1;
    });
}
