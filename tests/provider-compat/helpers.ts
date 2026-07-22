import type {
  Api,
  Model,
  Provider,
  ProviderModelsStore,
  SimpleStreamOptions,
  StreamOptions,
} from "@earendil-works/pi-ai";
import { afterEach, vi } from "vitest";

export const RED_CIRCLE_PNG =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2nLkAAAAASUVORK5CYII=";

type Chunk = { data: unknown; delay: number };
type RequestBody = { messages: Array<{ role: string; content: unknown }>; [key: string]: unknown };

export function sseChunk(data: unknown, delay = 0): Chunk {
  return { data, delay };
}

export async function createCompatibilityHarness(): Promise<{
  provider: Provider;
  model: Model<Api>;
  requests: RequestBody[];
  respond: (...chunks: Chunk[]) => void;
}> {
  vi.resetModules();
  vi.doMock("@earendil-works/pi-coding-agent", () => ({
    defineTool: (tool: unknown) => tool,
    getAgentDir: () => "/tmp/pi-provider-litellm-compat",
  }));
  process.env.LITELLM_BASE_URL = "https://litellm.example.com";
  process.env.LITELLM_API_KEY = "sk-test";

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
          if (chunk.delay) await new Promise((resolve) => setTimeout(resolve, chunk.delay));
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
  const store: ProviderModelsStore = {
    read: async () => undefined,
    write: async () => undefined,
    delete: async () => undefined,
  };
  const credential = {
    type: "api_key" as const,
    key: "sk-test",
    env: { LITELLM_BASE_URL: "https://litellm.example.com" },
  };
  await provider.refreshModels({ allowNetwork: true, credential, store });
  const model = provider.getModels()[0];
  if (!model) throw new Error("LiteLLM model was not discovered");
  const auth = await provider.auth.apiKey?.resolve({
    credential,
    ctx: { env: async (name) => process.env[name], fileExists: async () => false },
  });
  if (!auth?.auth.apiKey) throw new Error("LiteLLM auth was not resolved");
  const requestModel = { ...model, baseUrl: auth.auth.baseUrl ?? model.baseUrl };
  const stream = provider.stream.bind(provider);
  const streamSimple = provider.streamSimple.bind(provider);
  provider.stream = ((_: Model<Api>, context: Parameters<Provider["stream"]>[1], options?: StreamOptions) =>
    stream(requestModel, context, {
      ...options,
      apiKey: auth.auth.apiKey,
      headers: auth.auth.headers,
    })) as Provider["stream"];
  provider.streamSimple = ((
    _: Model<Api>,
    context: Parameters<Provider["streamSimple"]>[1],
    options?: SimpleStreamOptions,
  ) =>
    streamSimple(requestModel, context, {
      ...options,
      apiKey: auth.auth.apiKey,
      headers: auth.auth.headers,
    })) as Provider["streamSimple"];

  return { provider, model, requests, respond };
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.resetModules();
  delete process.env.LITELLM_BASE_URL;
  delete process.env.LITELLM_API_KEY;
});
