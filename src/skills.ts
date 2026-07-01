import type { Static } from "@earendil-works/pi-ai";
import { Type } from "@earendil-works/pi-ai";
import { defineTool, type ToolDefinition } from "@earendil-works/pi-coding-agent";
import { normalizeBaseUrl } from "./discover.js";
import type { LiteLLMSkill } from "./types.js";

const SKILLS_CACHE_TTL_MS = 60_000;

let skillsCache:
  | {
      baseUrl: string;
      apiKey: string;
      fetchedAt: number;
      skills: LiteLLMSkill[];
    }
  | undefined;

function getSkillsFromBody(body: unknown): LiteLLMSkill[] {
  if (Array.isArray(body)) return body as LiteLLMSkill[];
  if (body && typeof body === "object") {
    const record = body as { data?: unknown; skills?: unknown };
    if (Array.isArray(record.data)) return record.data as LiteLLMSkill[];
    if (Array.isArray(record.skills)) return record.skills as LiteLLMSkill[];
  }
  return [];
}

export async function listSkills(
  baseUrl: string,
  apiKey: string,
  headers?: Record<string, string>,
): Promise<LiteLLMSkill[]> {
  const normalizedBaseUrl = normalizeBaseUrl(baseUrl);
  if (
    skillsCache &&
    skillsCache.baseUrl === normalizedBaseUrl &&
    skillsCache.apiKey === apiKey &&
    Date.now() - skillsCache.fetchedAt < SKILLS_CACHE_TTL_MS
  ) {
    return skillsCache.skills;
  }

  try {
    const response = await fetch(`${normalizedBaseUrl}/v1/skills`, {
      headers: { ...headers, Authorization: `Bearer ${apiKey}`, Accept: "application/json" },
      signal: AbortSignal.timeout(10_000),
    });
    if (!response.ok) return [];
    const skills = getSkillsFromBody(await response.json());
    skillsCache = { baseUrl: normalizedBaseUrl, apiKey, fetchedAt: Date.now(), skills };
    return skills;
  } catch {
    return [];
  }
}

export async function createSkill(
  baseUrl: string,
  apiKey: string,
  input: { name: string; description: string; code: string; inputSchema?: Record<string, unknown> },
  headers?: Record<string, string>,
): Promise<unknown> {
  const response = await fetch(`${normalizeBaseUrl(baseUrl)}/v1/skills`, {
    method: "POST",
    headers: {
      ...headers,
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      name: input.name,
      description: input.description,
      code: input.code,
      input_schema: input.inputSchema ?? { type: "object", properties: {} },
    }),
    signal: AbortSignal.timeout(10_000),
  });
  if (!response.ok) throw new Error(`LiteLLM skill create failed: HTTP ${response.status}`);
  skillsCache = undefined;
  return response.json().catch(() => ({}));
}

export async function deleteSkill(
  baseUrl: string,
  apiKey: string,
  skillId: string,
  headers?: Record<string, string>,
): Promise<void> {
  const response = await fetch(`${normalizeBaseUrl(baseUrl)}/v1/skills/${encodeURIComponent(skillId)}`, {
    method: "DELETE",
    headers: { ...headers, Authorization: `Bearer ${apiKey}` },
    signal: AbortSignal.timeout(10_000),
  });
  if (!response.ok && response.status !== 404) throw new Error(`LiteLLM skill delete failed: HTTP ${response.status}`);
  skillsCache = undefined;
}

function escapeXml(value: string): string {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
}

export function createSkillsPromptSection(skills: LiteLLMSkill[]): string | undefined {
  const enabled = skills.filter((skill) => skill.enabled !== false && skill.name);
  if (enabled.length === 0) return undefined;

  const body = enabled
    .map((skill) => {
      const description = skill.description?.trim() || "No description provided.";
      return `<skill name="${escapeXml(skill.name)}">\n${escapeXml(description)}\n</skill>`;
    })
    .join("\n");
  return `LiteLLM Skills Gateway provided these active skills. Use them as additional guidance when relevant.\n<litellm_skills>\n${body}\n</litellm_skills>`;
}

function formatSkills(skills: LiteLLMSkill[]): string {
  if (skills.length === 0) return "No LiteLLM skills are registered.";
  return skills
    .map((skill) => `- ${skill.name}${skill.enabled === false ? " (disabled)" : ""}: ${skill.description ?? ""}`)
    .join("\n");
}

const CreateSkillParams = Type.Object({
  name: Type.String({ description: "Skill name" }),
  description: Type.String({ description: "Skill description" }),
  code: Type.String({ description: "Skill implementation or prompt code" }),
  inputSchemaJson: Type.Optional(Type.String({ description: "Optional JSON Schema string for skill inputs" })),
});

const DeleteSkillParams = Type.Object({
  skillId: Type.String({ description: "LiteLLM skill id" }),
});

export function createSkillToolDefinitions(
  baseUrl: string,
  getApiKey: () => Promise<string>,
  headers?: Record<string, string>,
): ToolDefinition[] {
  return [
    defineTool({
      name: "litellm_skill_list",
      label: "LiteLLM Skills",
      description: "List skills registered on the LiteLLM proxy Skills Gateway.",
      promptSnippet: "List LiteLLM Skills Gateway skills",
      parameters: Type.Object({}),
      async execute() {
        const apiKey = await getApiKey();
        const skills = await listSkills(baseUrl, apiKey, headers);
        return { content: [{ type: "text", text: formatSkills(skills) }], details: { count: skills.length } };
      },
    }),
    defineTool({
      name: "litellm_skill_create",
      label: "Create LiteLLM Skill",
      description: "Create a skill on the LiteLLM proxy Skills Gateway.",
      parameters: CreateSkillParams,
      async execute(_toolCallId, params: Static<typeof CreateSkillParams>) {
        const apiKey = await getApiKey();
        const inputSchema = params.inputSchemaJson
          ? (JSON.parse(params.inputSchemaJson) as Record<string, unknown>)
          : undefined;
        const result = await createSkill(baseUrl, apiKey, { ...params, inputSchema }, headers);
        return { content: [{ type: "text", text: "LiteLLM skill created." }], details: { result } };
      },
    }),
    defineTool({
      name: "litellm_skill_delete",
      label: "Delete LiteLLM Skill",
      description: "Delete a skill from the LiteLLM proxy Skills Gateway.",
      parameters: DeleteSkillParams,
      async execute(_toolCallId, params: Static<typeof DeleteSkillParams>) {
        const apiKey = await getApiKey();
        await deleteSkill(baseUrl, apiKey, params.skillId, headers);
        return { content: [{ type: "text", text: `LiteLLM skill deleted: ${params.skillId}` }], details: {} };
      },
    }),
  ];
}

export function resetSkillsCache(): void {
  skillsCache = undefined;
}
