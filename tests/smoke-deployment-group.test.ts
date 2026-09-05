import { execFile } from "node:child_process";
import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  captureGroupedDeployments,
  GROUPED_MODEL_NAME,
  summarizeGroupedDeployments,
  validateGroupedDeployments,
} from "../scripts/smoke-deployment-group.js";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const execFileAsync = promisify(execFile);
const scriptPath = resolve(repoRoot, "scripts/smoke-deployment-group.ts");
const tsxPath = resolve(repoRoot, "node_modules/.bin/tsx");
const scriptErrorLinePrefixes = [
  "Error",
  "/model/info response is missing data rows",
  "/model/info returned ",
  "expected two grouped deployments",
  "grouped deployment is missing ",
  "grouped deployments must preserve ",
  "LITELLM_BASE_URL and LITELLM_API_KEY must be set",
];

function deployment(id: string, mode: string, modelName = GROUPED_MODEL_NAME) {
  return {
    model_name: modelName,
    model_info: {
      id,
      mode,
      supported_openai_params: ["temperature"],
    },
    litellm_params: {
      allowed_openai_params: ["reasoning_effort"],
    },
  };
}

afterEach(() => {
  vi.restoreAllMocks();
});

describe("validateGroupedDeployments", () => {
  it("uses the grouped model name configured by the smoke workflow", () => {
    const workflow = readFileSync(resolve(repoRoot, ".github/workflows/litellm-smoke.yml"), "utf8");
    const configuredNames = workflow.match(/^\s*- model_name:\s*(\S+)\s*$/gm) ?? [];

    expect(configuredNames.filter((line) => line.trim() === `- model_name: ${GROUPED_MODEL_NAME}`)).toHaveLength(2);
  });

  it("returns the two grouped deployments when their ids are unique and their modes are exactly chat and responses", () => {
    const chat = deployment("deployment-chat", "chat");
    const responses = deployment("deployment-responses", "responses");

    expect(
      validateGroupedDeployments({
        data: [deployment("unrelated-responses", "responses", "vidaimock-responses"), chat, responses],
      }),
    ).toEqual([chat, responses]);
  });

  it("rejects duplicate deployment ids", () => {
    expect(() =>
      validateGroupedDeployments({
        data: [deployment("duplicate", "chat"), deployment("duplicate", "responses")],
      }),
    ).toThrow("grouped deployments must preserve unique model_info.id values");
  });

  it("does not let an unrelated responses deployment satisfy the grouped mode requirement", () => {
    expect(() =>
      validateGroupedDeployments({
        data: [
          deployment("grouped-chat-1", "chat"),
          deployment("grouped-chat-2", "chat"),
          deployment("unrelated-responses", "responses", "vidaimock-responses"),
        ],
      }),
    ).toThrow("grouped deployments must preserve exactly chat and responses modes");
  });

  it.each([
    [
      "model_info.id",
      {
        model_name: GROUPED_MODEL_NAME,
        model_info: { mode: "chat", supported_openai_params: [] },
        litellm_params: { allowed_openai_params: [] },
      },
    ],
    [
      "supported_openai_params",
      {
        model_name: GROUPED_MODEL_NAME,
        model_info: { id: "chat", mode: "chat" },
        litellm_params: { allowed_openai_params: [] },
      },
    ],
    [
      "allowed_openai_params",
      {
        model_name: GROUPED_MODEL_NAME,
        model_info: { id: "chat", mode: "chat", supported_openai_params: [] },
        litellm_params: {},
      },
    ],
  ])("rejects a grouped deployment missing %s", (field, invalidChat) => {
    expect(() => validateGroupedDeployments({ data: [invalidChat, deployment("responses", "responses")] })).toThrow(
      `grouped deployment is missing ${field}`,
    );
  });
});

describe("summarizeGroupedDeployments", () => {
  it("reports only deployment ids, modes, and parameter names", () => {
    const chat = {
      ...deployment("chat-id", "chat"),
      litellm_params: {
        api_key: "sk-must-not-leak",
        model: "openai/gpt-4o-mini",
        allowed_openai_params: ["reasoning_effort"],
      },
    };

    const summary = summarizeGroupedDeployments(
      validateGroupedDeployments({ data: [chat, deployment("responses-id", "responses")] }),
    );

    expect(summary).toEqual([
      {
        id: "chat-id",
        mode: "chat",
        supported_openai_params: ["temperature"],
        allowed_openai_params: ["reasoning_effort"],
      },
      {
        id: "responses-id",
        mode: "responses",
        supported_openai_params: ["temperature"],
        allowed_openai_params: ["reasoning_effort"],
      },
    ]);
    expect(JSON.stringify(summary)).not.toContain("sk-must-not-leak");
    expect(JSON.stringify(summary)).not.toContain("openai/gpt-4o-mini");
  });
});

describe("captureGroupedDeployments", () => {
  it("fetches /model/info with authentication and validates the live payload", async () => {
    const rows = [deployment("chat", "chat"), deployment("responses", "responses")];
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(Response.json({ data: rows }));

    await expect(captureGroupedDeployments("http://127.0.0.1:4000/", "sk-smoke", 1250)).resolves.toEqual(rows);
    expect(fetchSpy).toHaveBeenCalledWith("http://127.0.0.1:4000/model/info", {
      headers: { Authorization: "Bearer sk-smoke" },
      signal: expect.any(AbortSignal),
    });
  });
});

describe("smoke-deployment-group script", () => {
  it("executes directly and prints only the redacted deployment summary", async () => {
    const secret = "sk-direct-execution-secret";
    const rows = [
      {
        ...deployment("chat", "chat"),
        litellm_params: {
          api_key: secret,
          model: "openai/private-chat-backend",
          allowed_openai_params: ["reasoning_effort"],
        },
      },
      deployment("responses", "responses"),
    ];
    const server = createServer((request, response) => {
      if (request.url !== "/model/info" || request.headers.authorization !== `Bearer ${secret}`) {
        response.writeHead(404).end();
        return;
      }
      response.writeHead(200, { "content-type": "application/json" }).end(JSON.stringify({ data: rows }));
    });
    server.listen(0, "127.0.0.1");
    await new Promise<void>((resolveListen) => server.once("listening", resolveListen));

    try {
      const address = server.address();
      if (!address || typeof address === "string") throw new Error("test server did not bind a TCP port");
      const { stdout, stderr } = await execFileAsync(tsxPath, [scriptPath], {
        cwd: repoRoot,
        env: {
          ...process.env,
          LITELLM_BASE_URL: `http://127.0.0.1:${address.port}`,
          LITELLM_API_KEY: secret,
        },
      });

      expect(JSON.parse(stdout)).toEqual(summarizeGroupedDeployments(rows));
      expect(stdout).not.toContain(secret);
      expect(stdout).not.toContain("private-chat-backend");
      expect(stderr).not.toContain(secret);
      expect(stderr).not.toContain("private-chat-backend");
      expect(
        stderr
          .split(/\r?\n/)
          .map((line) => line.trimStart())
          .filter((line) => scriptErrorLinePrefixes.some((prefix) => line.startsWith(prefix))),
      ).toEqual([]);
    } finally {
      await new Promise<void>((resolveClose, rejectClose) =>
        server.close((error) => (error ? rejectClose(error) : resolveClose())),
      );
    }
  });

  it("exits unsuccessfully with a concise error when required environment is missing", async () => {
    const secret = "sk-missing-environment-secret";
    const error = await execFileAsync(tsxPath, [scriptPath], {
      cwd: repoRoot,
      env: { ...process.env, LITELLM_BASE_URL: "", LITELLM_API_KEY: secret },
    }).catch((reason: unknown) => reason);

    expect(error).toMatchObject({
      code: 1,
      stdout: "",
      stderr: expect.stringContaining("LITELLM_BASE_URL and LITELLM_API_KEY must be set"),
    });
    expect((error as { stderr: string }).stderr).not.toContain(secret);
  });
});
