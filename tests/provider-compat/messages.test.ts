import { type Context, getSupportedThinkingLevels, InMemoryModelsStore } from "@earendil-works/pi-ai";
import { describe, expect, it, vi } from "vitest";
import { anthropicSseChunk, anthropicTextResponse, claudeRoute, createCompatibilityHarness, user } from "./helpers.js";

const anthropicRoute = {
  model_name: "claude-opus-5",
  model_info: {
    mode: "chat",
    litellm_provider: "bedrock_converse",
    base_model: "bedrock/us.anthropic.claude-opus-5",
    supports_reasoning: true,
    supports_vision: true,
  },
  litellm_params: { model: "bedrock/us.anthropic.claude-opus-5" },
};

const budgetThinkingRoute = {
  model_name: "claude-sonnet-4-5",
  model_info: {
    mode: "chat",
    litellm_provider: "anthropic",
    supports_reasoning: true,
  },
  litellm_params: { model: "anthropic/claude-sonnet-4-5" },
};

// Decorated backend ids the Anthropic catalog does not list verbatim: a dated release,
// a Bedrock cross-region inference profile, and a Vertex serving suffix. All three are
// Opus 4.7, which requires adaptive thinking and rejects temperature.
const decoratedAdaptiveRoutes = [
  ["dated release", "anthropic", "anthropic/claude-opus-4-7-20260415"],
  ["bedrock inference profile", "bedrock", "bedrock/us.anthropic.claude-opus-4-7-v1:0"],
  ["bedrock converse path", "bedrock_converse", "bedrock/converse/us.anthropic.claude-opus-4-7-v1:0"],
  ["vertex serving suffix", "vertex_ai", "vertex_ai/claude-opus-4-7@20260101"],
] as const;

// One public alias load-balanced across two Claude generations, as happens mid-migration.
// Opus 4.7 requires adaptive thinking and rejects temperature; Opus 4.5 requires budget
// thinking and accepts it. No single Messages request satisfies both.
const mixedGenerationRoutes = [
  claudeRoute("anthropic", "anthropic/claude-opus-4-7", "a"),
  claudeRoute("anthropic", "anthropic/claude-opus-4-5", "b"),
];

const unknownBackendRoute = claudeRoute("bedrock", "bedrock/us.anthropic.claude-invented-9-9-v1:0");

describe("Anthropic Messages wire compatibility", () => {
  it("streams Anthropic SSE text from the exact Messages endpoint", async () => {
    const { models, model, requestHeaders, requestUrls, requests, respond } = await createCompatibilityHarness(
      anthropicRoute,
      { customHeaders: { "x-tenant": "team-a" }, sessionFile: "/tmp/pi-compat-session-wire.jsonl" },
    );
    respond(...anthropicTextResponse("Hello from Messages"));

    const message = await models.streamSimple(model, { messages: [user("Hello")] }).result();

    expect(model.api).toBe("anthropic-messages");
    expect(requestUrls).toEqual(["https://proxy.example.com/v1/messages"]);
    expect(requestUrls[0]).not.toContain("/v1/v1/messages");
    expect(message.content).toEqual([{ type: "text", text: "Hello from Messages" }]);
    expect(message.usage).toMatchObject({ input: 2, output: 1 });
    expect(requests[0]).toMatchObject({
      model: "claude-opus-5",
      messages: [{ role: "user", content: expect.anything() }],
      stream: true,
      max_tokens: expect.any(Number),
    });
    expect(requests[0]?.max_tokens).toBeGreaterThan(0);
    expect(requests[0]).not.toHaveProperty("max_completion_tokens");
    expect(requests[0]).not.toHaveProperty("litellm_session_id");
    expect(requestHeaders[0]?.get("x-api-key")).toBe("sk-test");
    expect(requestHeaders[0]?.get("authorization")).toBeNull();
    expect(requestHeaders[0]?.get("anthropic-version")).toBe("2023-06-01");
    expect(requestHeaders[0]?.get("anthropic-beta")).toBeNull();
    expect(requestHeaders[0]?.get("x-tenant")).toBe("team-a");
  });

  it("serializes images, tools, and cache controls with Anthropic-native fields", async () => {
    const { models, model, requests, respond } = await createCompatibilityHarness(anthropicRoute, {
      sessionFile: "/tmp/pi-compat-session-serialize.jsonl",
    });
    expect(getSupportedThinkingLevels(model)).toContain("high");
    respond(...anthropicTextResponse("ready"));

    await models
      .streamSimple(
        model,
        {
          systemPrompt: "Keep answers short.",
          messages: [
            {
              role: "user",
              content: [
                { type: "text", text: "Inspect this" },
                { type: "image", data: "aGVsbG8=", mimeType: "image/png" },
              ],
              timestamp: 1,
            },
          ],
          tools: [
            {
              name: "lookup",
              description: "Look up a value",
              parameters: {
                type: "object",
                properties: { key: { type: "string" } },
                required: ["key"],
              },
            },
          ],
        },
        { reasoning: "high" },
      )
      .result();

    expect(requests[0]).toMatchObject({
      system: [expect.objectContaining({ type: "text", cache_control: { type: "ephemeral" } })],
      thinking: { type: "adaptive" },
      output_config: { effort: "high" },
      tools: [
        expect.objectContaining({
          name: "lookup",
          input_schema: expect.objectContaining({ type: "object" }),
          cache_control: { type: "ephemeral" },
        }),
      ],
    });
    expect(requests[0]?.messages?.[0]?.content).toContainEqual(
      expect.objectContaining({ type: "image", source: expect.objectContaining({ media_type: "image/png" }) }),
    );
    expect(requests[0]).not.toHaveProperty("reasoning_effort");
    expect(requests[0]).not.toHaveProperty("include");
    expect(requests[0]).not.toHaveProperty("litellm_session_id");
  });

  it("ignores router OpenAI effort additions on native Messages", async () => {
    const route = {
      ...anthropicRoute,
      model_info: {
        ...anthropicRoute.model_info,
        supports_minimal_reasoning_effort: true,
        supports_xhigh_reasoning_effort: false,
      },
    };
    const { models, model, requests, respond } = await createCompatibilityHarness(route);
    respond(...anthropicTextResponse("ready"));

    expect(model.thinkingLevelMap).not.toHaveProperty("minimal");
    expect(getSupportedThinkingLevels(model)).not.toContain("xhigh");
    expect(getSupportedThinkingLevels(model)).toContain("max");

    await models.streamSimple(model, { messages: [user("Think carefully")] }, { reasoning: "max" }).result();

    expect(model.thinkingLevelMap).toEqual({ xhigh: null, max: "max" });
    expect(requests[0]).toMatchObject({ thinking: { type: "adaptive" }, output_config: { effort: "max" } });
  });

  it("preserves budget-based thinking for older native Claude models", async () => {
    const { models, model, requests, respond } = await createCompatibilityHarness(budgetThinkingRoute);
    respond(...anthropicTextResponse("ready"));

    await models.streamSimple(model, { messages: [user("Think carefully")] }, { reasoning: "high" }).result();

    expect(model).toMatchObject({ api: "anthropic-messages" });
    expect(model.compat).toEqual({ supportsStrictTools: true });
    expect(requests[0]).toMatchObject({ thinking: { type: "enabled", budget_tokens: expect.any(Number) } });
    expect(requests[0]).not.toHaveProperty("output_config");
  });

  it.each(decoratedAdaptiveRoutes)(
    "sends adaptive thinking for a %s backend the catalog does not list verbatim",
    async (_label, adapter, backend) => {
      const { models, model, requests, respond } = await createCompatibilityHarness(claudeRoute(adapter, backend));
      respond(...anthropicTextResponse("ready"));

      await models.streamSimple(model, { messages: [user("Think")] }, { reasoning: "high" }).result();

      expect(model.api).toBe("anthropic-messages");
      expect(model.compat).toEqual({
        forceAdaptiveThinking: true,
        supportsTemperature: false,
        supportsStrictTools: true,
      });
      expect(requests[0]).toMatchObject({ thinking: { type: "adaptive" }, output_config: { effort: "high" } });
      expect(requests[0]?.thinking).not.toHaveProperty("budget_tokens");
    },
  );

  it("reduces a mixed-generation Claude alias to Chat instead of a wrong Messages shape", async () => {
    const { models, model, requestUrls, requests, respondRaw } =
      await createCompatibilityHarness(mixedGenerationRoutes);
    respondRaw(
      'data: {"choices":[{"delta":{"content":"chat"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n' +
        "data: [DONE]\n\n",
    );

    await models.streamSimple(model, { messages: [user("Hello")] }, { reasoning: "high", temperature: 0.7 }).result();

    expect(model.api).toBe("openai-completions");
    // Prompt caching still reaches Claude through LiteLLM's translation on this path.
    expect(model.compat).toMatchObject({ cacheControlFormat: "anthropic" });
    expect(requestUrls).toEqual(["https://proxy.example.com/v1/chat/completions"]);
    expect(requests[0]).not.toHaveProperty("thinking");
    expect(requests[0]).not.toHaveProperty("output_config");
  });

  it("keeps a same-generation Claude alias on native Messages", async () => {
    const { model } = await createCompatibilityHarness([
      claudeRoute("anthropic", "anthropic/claude-opus-4-7", "a"),
      claudeRoute("anthropic", "anthropic/claude-opus-4-8", "b"),
    ]);

    expect(model.api).toBe("anthropic-messages");
    expect(model.compat).toEqual({
      forceAdaptiveThinking: true,
      supportsTemperature: false,
      supportsStrictTools: true,
    });
  });

  it("omits temperature for a decorated Opus backend that rejects it", async () => {
    const [, adapter, backend] = decoratedAdaptiveRoutes[1];
    const { models, model, requests, respond } = await createCompatibilityHarness(claudeRoute(adapter, backend));
    respond(...anthropicTextResponse("ready"));

    await models.streamSimple(model, { messages: [user("Hello")] }, { temperature: 0.7 }).result();

    expect(requests[0]).not.toHaveProperty("temperature");
  });

  it("keeps temperature for a native Claude backend that accepts it", async () => {
    const { models, model, requests, respond } = await createCompatibilityHarness(budgetThinkingRoute);
    respond(...anthropicTextResponse("ready"));

    await models.streamSimple(model, { messages: [user("Hello")] }, { temperature: 0.7 }).result();

    expect(requests[0]).toMatchObject({ temperature: 0.7 });
  });

  it("routes an unrecognized Claude backend through Chat rather than guessing", async () => {
    const { models, model, requestUrls, respondRaw } = await createCompatibilityHarness(unknownBackendRoute);
    respondRaw(
      'data: {"choices":[{"delta":{"content":"chat"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n' +
        "data: [DONE]\n\n",
    );

    await models.streamSimple(model, { messages: [user("Think")] }, { reasoning: "high" }).result();

    expect(model.api).toBe("openai-completions");
    expect(requestUrls).toEqual(["https://proxy.example.com/v1/chat/completions"]);
  });

  it("omits session grouping from Messages while real Chat serialization retains it", async () => {
    const sessionFile = "/tmp/pi-compat-session.jsonl";
    const messages = await createCompatibilityHarness(anthropicRoute, { sessionFile });
    messages.respond(...anthropicTextResponse("messages"));
    await messages.models.streamSimple(messages.model, { messages: [user("Hello")] }).result();
    expect(messages.requests[0]).not.toHaveProperty("litellm_session_id");

    vi.restoreAllMocks();
    vi.resetModules();
    const chat = await createCompatibilityHarness(undefined, { sessionFile });
    chat.respondRaw(
      'data: {"choices":[{"delta":{"content":"chat"},"finish_reason":null}]}\n\n' +
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n' +
        "data: [DONE]\n\n",
    );
    await chat.models.streamSimple(chat.model, { messages: [user("Hello")] }).result();
    expect(chat.requests[0]?.litellm_session_id).toBeTruthy();
  });

  it("rehydrates the same Messages API, URL, and body from the offline model store", async () => {
    const modelsStore = new InMemoryModelsStore();
    const online = await createCompatibilityHarness(anthropicRoute, { modelsStore });
    online.respond(...anthropicTextResponse("online"));
    await online.models.streamSimple(online.model, { messages: [user("Hello")] }, { reasoning: "high" }).result();

    vi.restoreAllMocks();
    vi.resetModules();
    const offline = await createCompatibilityHarness(anthropicRoute, { modelsStore, allowNetwork: false });
    offline.respond(...anthropicTextResponse("offline"));
    await offline.models.streamSimple(offline.model, { messages: [user("Hello")] }, { reasoning: "high" }).result();

    expect(offline.model.api).toBe("anthropic-messages");
    expect(offline.model).toEqual(online.model);
    expect(offline.requestUrls).toEqual(["https://proxy.example.com/v1/messages"]);
    expect(offline.requests[0]).toEqual(online.requests[0]);
    expect(offline.requests[0]).toMatchObject({
      thinking: { type: "adaptive" },
      output_config: { effort: "high" },
    });
  });

  it("reports a non-overflow Anthropic error envelope", async () => {
    const { models, model } = await createCompatibilityHarness(anthropicRoute);
    // Replace only the completion response after discovery with a realistic LiteLLM native envelope.
    vi.mocked(globalThis.fetch).mockImplementationOnce(async () =>
      Response.json(
        { type: "error", error: { type: "overloaded_error", message: "upstream overloaded" } },
        { status: 529 },
      ),
    );
    const message = await models.streamSimple(model, { messages: [user("Hello")] }).result();
    expect(message.stopReason).toBe("error");
    expect(message.errorMessage).toContain("upstream overloaded");
  });

  it("returns an error for malformed Anthropic SSE", async () => {
    const { models, model, respondRaw } = await createCompatibilityHarness(anthropicRoute);
    respondRaw('event: message_start\ndata: {"type":"message_start"\n\n');
    const message = await models.streamSimple(model, { messages: [user("Hello")] }).result();
    expect(message.stopReason).toBe("error");
    expect(message.errorMessage).toBeTruthy();
  });

  it("replays native thinking, tool use, and tool results on the next turn", async () => {
    const { models, model, requests, respond } = await createCompatibilityHarness(anthropicRoute);
    const context: Context = {
      messages: [user("Look up the answer")],
      tools: [
        {
          name: "lookup",
          description: "Look up a value",
          parameters: { type: "object", properties: { key: { type: "string" } } },
        },
      ],
    };
    respond(
      anthropicSseChunk({
        type: "message_start",
        message: {
          id: "msg_tool",
          type: "message",
          role: "assistant",
          content: [],
          model: "claude-opus-5",
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 5, output_tokens: 0 },
        },
      }),
      anthropicSseChunk({
        type: "content_block_start",
        index: 0,
        content_block: { type: "thinking", thinking: "", signature: "" },
      }),
      anthropicSseChunk({
        type: "content_block_delta",
        index: 0,
        delta: { type: "thinking_delta", thinking: "I should use the tool." },
      }),
      anthropicSseChunk({
        type: "content_block_delta",
        index: 0,
        delta: { type: "signature_delta", signature: "signed-thinking" },
      }),
      anthropicSseChunk({ type: "content_block_stop", index: 0 }),
      anthropicSseChunk({
        type: "content_block_start",
        index: 1,
        content_block: { type: "tool_use", id: "tool_1", name: "lookup", input: {} },
      }),
      anthropicSseChunk({
        type: "content_block_delta",
        index: 1,
        delta: { type: "input_json_delta", partial_json: '{"key":"answer"}' },
      }),
      anthropicSseChunk({ type: "content_block_stop", index: 1 }),
      anthropicSseChunk({
        type: "message_delta",
        delta: { stop_reason: "tool_use", stop_sequence: null },
        usage: { output_tokens: 8, output_tokens_details: { thinking_tokens: 3 } },
      }),
      anthropicSseChunk({ type: "message_stop" }),
    );

    const first = await models.streamSimple(model, context, { reasoning: "high" }).result();
    expect(first.content).toEqual([
      { type: "thinking", thinking: "I should use the tool.", thinkingSignature: "signed-thinking" },
      { type: "toolCall", id: "tool_1", name: "lookup", arguments: { key: "answer" } },
    ]);
    expect(first.stopReason).toBe("toolUse");

    context.messages.push(first, {
      role: "toolResult",
      toolCallId: "tool_1",
      toolName: "lookup",
      content: [{ type: "text", text: "42" }],
      isError: false,
      timestamp: 3,
    });
    respond(...anthropicTextResponse("The answer is 42."));

    const second = await models.streamSimple(model, context, { reasoning: "high" }).result();

    expect(second.content).toEqual([{ type: "text", text: "The answer is 42." }]);
    expect(requests[1]?.messages).toEqual([
      expect.objectContaining({ role: "user" }),
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "I should use the tool.", signature: "signed-thinking" },
          { type: "tool_use", id: "tool_1", name: "lookup", input: { key: "answer" } },
        ],
      },
      {
        role: "user",
        content: [
          expect.objectContaining({
            type: "tool_result",
            tool_use_id: "tool_1",
            content: "42",
            is_error: false,
          }),
        ],
      },
    ]);
    expect(requests[1]).not.toHaveProperty("reasoning_effort");
  });
});
