import type { Api, AssistantMessage, AuthContext, Model, Models, Provider, StreamOptions } from "@earendil-works/pi-ai";
import { createModels, createProvider, InMemoryCredentialStore, InMemoryModelsStore } from "@earendil-works/pi-ai";
import { openAICompletionsApi } from "@earendil-works/pi-ai/api/openai-completions.lazy";
import { afterEach, vi } from "vitest";
import type { ModelInfoEntry } from "../../src/types.js";
import { useHermeticEnv } from "../test-helpers.js";

// Every provider-compat suite imports this module, and each drives real discovery through
// the extension, which reads LITELLM_* from the environment. Registering the hermetic
// hooks here gives all of them a known-empty environment; without it an ambient
// LITELLM_OFFLINE or LITELLM_BASE_URL changes what these tests exercise.
useHermeticEnv();

export const RED_CIRCLE_PNG =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2nLkAAAAASUVORK5CYII=";
export const SECOND_PIXEL_PNG =
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=";

type Chunk = { data: unknown; waitForAbort: boolean };
type RequestBody = { messages: Array<{ role: string; content: unknown }>; [key: string]: unknown };

export function sseChunk(data: unknown, waitForAbort = false): Chunk {
  return { data, waitForAbort };
}

export function user(content: string) {
  return { role: "user" as const, content, timestamp: 1 };
}

export function assistant(
  model: Model<Api>,
  content: AssistantMessage["content"],
  stopReason: AssistantMessage["stopReason"] = "stop",
): AssistantMessage {
  return {
    role: "assistant",
    content,
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason,
    timestamp: 2,
  };
}

export function successfulResponse(text: string): Chunk[] {
  return [
    sseChunk({ choices: [{ delta: { content: text }, finish_reason: null }] }),
    sseChunk({
      choices: [{ delta: {}, finish_reason: "stop" }],
      usage: { prompt_tokens: 1, completion_tokens: 1 },
    }),
  ];
}

// Responses-shaped equivalent. A `mode: responses` deployment is served on
// `/responses`, which speaks a different event vocabulary from chat completions,
// so a chat-shaped reply cannot stand in for it — without this the harness threw
// on the URL and no Responses request could be inspected at all.
export function successfulResponsesReply(text: string): Chunk[] {
  const message = {
    id: "msg_1",
    type: "message",
    role: "assistant",
    status: "completed",
    content: [{ type: "output_text", text, annotations: [] }],
  };
  return [
    sseChunk({ type: "response.created", response: { id: "resp_1", status: "in_progress", output: [] } }),
    sseChunk({ type: "response.output_item.added", output_index: 0, item: { ...message, content: [] } }),
    sseChunk({ type: "response.output_text.delta", output_index: 0, content_index: 0, delta: text }),
    sseChunk({ type: "response.output_item.done", output_index: 0, item: message }),
    sseChunk({
      type: "response.completed",
      response: {
        id: "resp_1",
        status: "completed",
        output: [message],
        usage: { input_tokens: 1, output_tokens: 1, input_tokens_details: { cached_tokens: 0 } },
      },
    }),
  ];
}

export async function createCompatibilityHarness(
  discoveryRows?: readonly ModelInfoEntry[],
  options: { modelsStore?: InMemoryModelsStore; allowNetwork?: boolean } = {},
): Promise<{
  provider: Provider;
  models: Models;
  model: Model<Api>;
  foreignModel: Model<Api>;
  requests: RequestBody[];
  foreignRequests: RequestBody[];
  respond: (...chunks: Chunk[]) => void;
}> {
  vi.doMock("@earendil-works/pi-coding-agent", () => ({
    defineTool: (tool: unknown) => tool,
    getAgentDir: () => "/tmp/pi-provider-litellm-compat",
  }));
  vi.stubEnv("LITELLM_BASE_URL", "https://litellm.example.com");
  vi.stubEnv("LITELLM_API_KEY", "sk-test");

  const requests: RequestBody[] = [];
  const foreignRequests: RequestBody[] = [];
  const responses: Chunk[][] = [];
  const respond = (...chunks: Chunk[]) => responses.push(chunks);
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
    const request = input instanceof Request ? input : undefined;
    const url = request?.url ?? String(input);
    const isForeignRequest = new URL(url).origin === "https://foreign.example.com";
    if (url.endsWith("/model/info")) {
      return Response.json({
        data: discoveryRows ?? [
          {
            model_name: "local-model",
            model_info: {
              mode: "chat",
              supports_reasoning: true,
              supports_vision: true,
              max_input_tokens: 4096,
              max_output_tokens: 1024,
              input_cost_per_token: 0.000001,
              output_cost_per_token: 0.000002,
              cache_read_input_token_cost: 0.000003,
              cache_creation_input_token_cost: 0.000004,
            },
          },
        ],
      });
    }
    if (url.endsWith("/mcp-rest/tools/list")) return Response.json([]);
    // Both request surfaces are served so a `mode: responses` deployment can be
    // exercised on the wire, not only through discovery.
    if (!url.endsWith("/chat/completions") && !url.endsWith("/responses")) {
      throw new Error(`unexpected URL: ${url}`);
    }

    const requestBody = (request ? await request.clone().json() : JSON.parse(String(init?.body))) as RequestBody;
    (isForeignRequest ? foreignRequests : requests).push(requestBody);
    const history = JSON.stringify(requestBody.messages ?? requestBody.input ?? []);
    if (history.includes("Overflow the context")) {
      return Response.json(
        {
          error: {
            message: "Requested token count exceeds the model's maximum context length of 4096 tokens",
            type: "invalid_request_error",
            param: "messages",
            code: "context_length_exceeded",
          },
        },
        { status: 400 },
      );
    }
    const chunks =
      responses.shift() ??
      (isForeignRequest && history.includes("Continue elsewhere")
        ? successfulResponse("foreign continued")
        : history.includes("Continue in LiteLLM")
          ? successfulResponse("LiteLLM continued")
          : history.includes("diameter: 2 px")
            ? successfulResponse("diameter 2 px")
            : history.includes("Inspect the image")
              ? successfulResponse("red circle")
              : undefined);
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
  const handlers = new Map<string, Array<(event: any, ctx: any) => unknown>>();
  const extension = (await import("../../src/index.js")).default;
  await extension({
    registerProvider: (provider: Provider) => providers.push(provider),
    registerCommand: () => undefined,
    registerTool: () => undefined,
    on: (event: string, handler: (event: unknown, ctx: unknown) => unknown) => {
      const registered = handlers.get(event) ?? [];
      registered.push(handler);
      handlers.set(event, registered);
    },
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
  const models = createModels({
    credentials,
    modelsStore: options.modelsStore ?? new InMemoryModelsStore(),
    authContext,
  });
  const beforeRequestHandlers = handlers.get("before_provider_request") ?? [];
  const composePayloadHook =
    (original: StreamOptions["onPayload"]): StreamOptions["onPayload"] =>
    async (payload, payloadModel) => {
      let current = (await original?.(payload, payloadModel)) ?? payload;
      for (const handler of beforeRequestHandlers) {
        const next = await handler({ type: "before_provider_request", payload: current }, { model: payloadModel });
        current = next ?? current;
      }
      return current;
    };
  const hookProvider: Provider = {
    ...provider,
    stream: (streamModel, context, options?: StreamOptions) =>
      provider.stream(streamModel, context, {
        ...options,
        onPayload: composePayloadHook(options?.onPayload),
      } as never),
    streamSimple: (streamModel, context, options) =>
      provider.streamSimple(streamModel, context, {
        ...options,
        onPayload: composePayloadHook(options?.onPayload),
      }),
  };
  models.setProvider(hookProvider);
  const foreignModel = {
    id: "foreign-model",
    name: "Foreign model",
    api: "openai-completions" as const,
    provider: "foreign",
    baseUrl: "https://foreign.example.com/v1",
    reasoning: true,
    input: ["text", "image"] as ("text" | "image")[],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 4096,
    maxTokens: 1024,
  };
  models.setProvider(
    createProvider({
      id: "foreign",
      auth: { apiKey: { name: "Foreign", resolve: async () => ({ auth: { apiKey: "foreign-test" } }) } },
      models: [foreignModel],
      api: openAICompletionsApi(),
    }),
  );
  const refresh = await models.refresh({ allowNetwork: options.allowNetwork ?? true });
  const refreshError = refresh.errors.get(provider.id);
  if (refreshError) throw refreshError;
  const discoveredId = discoveryRows?.[0]?.model_name ?? "local-model";
  const model = models.getModel(provider.id, discoveredId);
  if (!model) throw new Error("LiteLLM model was not discovered");

  return { provider, models, model, foreignModel, requests, foreignRequests, respond };
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
});
