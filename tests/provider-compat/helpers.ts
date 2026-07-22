import type { Api, AuthContext, Model, Models, Provider } from "@earendil-works/pi-ai";
import { createModels, InMemoryCredentialStore, InMemoryModelsStore } from "@earendil-works/pi-ai";
import { afterEach, vi } from "vitest";

export const RED_CIRCLE_PNG =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2nLkAAAAASUVORK5CYII=";
export const SECOND_PIXEL_PNG =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

type Chunk = { data: unknown; waitForAbort: boolean };
type RequestBody = { messages: Array<{ role: string; content: unknown }>; [key: string]: unknown };

export function sseChunk(data: unknown, waitForAbort = false): Chunk {
  return { data, waitForAbort };
}

export async function createCompatibilityHarness(): Promise<{
  provider: Provider;
  models: Models;
  model: Model<Api>;
  requests: RequestBody[];
  respond: (...chunks: Chunk[]) => void;
}> {
  vi.doMock("@earendil-works/pi-coding-agent", () => ({
    defineTool: (tool: unknown) => tool,
    getAgentDir: () => "/tmp/pi-provider-litellm-compat",
  }));
  vi.stubEnv("LITELLM_BASE_URL", "https://litellm.example.com");
  vi.stubEnv("LITELLM_API_KEY", "sk-test");

  const requests: RequestBody[] = [];
  const responses: Chunk[][] = [];
  const respond = (...chunks: Chunk[]) => responses.push(chunks);
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const request = input instanceof Request ? input : undefined;
    const url = request?.url ?? String(input);
    if (url.endsWith("/model/info")) {
      return Response.json({
        data: [
          {
            model_name: "local-model",
            model_info: {
              mode: "chat",
              supports_reasoning: true,
              supports_vision: true,
              max_input_tokens: 4096,
              max_output_tokens: 1024,
            },
          },
        ],
      });
    }
    if (url.endsWith("/mcp-rest/tools/list")) return Response.json([]);
    if (!url.endsWith("/chat/completions")) throw new Error(`unexpected URL: ${url}`);

    requests.push((request ? await request.clone().json() : JSON.parse(String(init?.body))) as RequestBody);
    const chunks = responses.shift();
    if (!chunks) throw new Error("missing mock response");
    const signal = request?.signal ?? init?.signal;
    const body = new ReadableStream({
      async start(controller) {
        const encoder = new TextEncoder();
        for (const chunk of chunks) {
          if (chunk.waitForAbort && signal && !signal.aborted) {
            await new Promise<void>((resolve) => signal.addEventListener("abort", () => resolve(), { once: true }));
          }
          if (signal?.aborted) return controller.error(signal.reason);
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(chunk.data)}\n\n`));
        }
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
      },
    });
    return new Response(body, { headers: { "content-type": "text/event-stream" } });
  });

  const providers: Provider[] = [];
  const extension = (await import("../../src/index.js")).default;
  await extension({
    registerProvider: (provider: Provider) => providers.push(provider),
    registerCommand: () => undefined,
    registerTool: () => undefined,
    on: () => undefined,
  } as never);
  const provider = providers[0];
  if (!provider?.refreshModels) throw new Error("LiteLLM provider was not registered");
  const credential = {
    type: "api_key" as const,
    key: "sk-test",
    env: { LITELLM_BASE_URL: "https://litellm.example.com" },
  };
  const credentials = new InMemoryCredentialStore();
  await credentials.modify(provider.id, async () => credential);
  const authContext: AuthContext = {
    env: async (name) => process.env[name],
    fileExists: async () => false,
  };
  const models = createModels({ credentials, modelsStore: new InMemoryModelsStore(), authContext });
  models.setProvider(provider);
  const refresh = await models.refresh({ allowNetwork: true });
  const refreshError = refresh.errors.get(provider.id);
  if (refreshError) throw refreshError;
  const model = models.getModel(provider.id, "local-model");
  if (!model) throw new Error("LiteLLM model was not discovered");

  return { provider, models, model, requests, respond };
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
});
