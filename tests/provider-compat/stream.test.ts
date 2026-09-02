import { type Context, getSupportedThinkingLevels, InMemoryModelsStore } from "@earendil-works/pi-ai";
import { describe, expect, it, vi } from "vitest";
import {
  createCompatibilityHarness,
  RED_CIRCLE_PNG,
  sseChunk,
  successfulResponse,
  successfulResponsesReply,
} from "./helpers.js";

const user = (content: string) => ({ role: "user" as const, content, timestamp: 1 });

const NO_LEVELS = {
  off: null,
  minimal: null,
  low: null,
  medium: null,
  high: null,
  xhigh: null,
  max: null,
};

describe("native provider stream compatibility", () => {
  it("completes two text turns with usage", async () => {
    const { models, model, requests, respond } = await createCompatibilityHarness();
    const context: Context = { messages: [user("First")] };
    respond(
      sseChunk({ choices: [{ delta: { content: "Hello" }, finish_reason: null }] }),
      sseChunk({ choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 3, completion_tokens: 2 } }),
    );
    respond(
      sseChunk({ choices: [{ delta: { content: "Again" }, finish_reason: null }] }),
      sseChunk({ choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 7, completion_tokens: 1 } }),
    );

    const first = await models.streamSimple(model, context).result();
    expect(first.content).toEqual([{ type: "text", text: "Hello" }]);
    expect(first.usage.input).toBeGreaterThan(0);
    expect(first.usage.output).toBeGreaterThan(0);

    context.messages.push(first, user("Second"));
    const second = await models.streamSimple(model, context).result();
    expect(second.stopReason).toBe("stop");
    expect(second.content).toEqual([{ type: "text", text: "Again" }]);
    expect(second.usage.input).toBeGreaterThan(0);
    expect(second.usage.output).toBeGreaterThan(0);
    expect(requests).toHaveLength(2);
  });

  it("emits text start, delta, and end events", async () => {
    const { models, model, respond } = await createCompatibilityHarness();
    respond(
      sseChunk({ choices: [{ delta: { content: "Hello" }, finish_reason: null }] }),
      sseChunk({ choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 1, completion_tokens: 1 } }),
    );

    const events: Array<{ type: string; delta?: string }> = [];
    for await (const event of models.streamSimple(model, { messages: [user("Hi")] })) {
      if (event.type === "text_delta") events.push({ type: event.type, delta: event.delta });
      else if (event.type === "text_start" || event.type === "text_end") events.push({ type: event.type });
    }

    expect(events).toEqual([{ type: "text_start" }, { type: "text_delta", delta: "Hello" }, { type: "text_end" }]);
  });

  it("assembles tool-call argument deltas", async () => {
    const { models, model, respond } = await createCompatibilityHarness();
    respond(
      sseChunk({
        choices: [
          {
            delta: {
              tool_calls: [{ index: 0, id: "call_1", type: "function", function: { name: "add", arguments: '{"a":' } }],
            },
            finish_reason: null,
          },
        ],
      }),
      sseChunk({
        choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: "714}" } }] }, finish_reason: null }],
      }),
      sseChunk({
        choices: [{ delta: {}, finish_reason: "tool_calls" }],
        usage: { prompt_tokens: 2, completion_tokens: 3 },
      }),
    );

    const message = await models
      .streamSimple(model, {
        messages: [user("Add")],
        tools: [
          { name: "add", description: "Add", parameters: { type: "object", properties: { a: { type: "number" } } } },
        ],
      })
      .result();

    expect(message.content).toContainEqual({ type: "toolCall", id: "call_1", name: "add", arguments: { a: 714 } });
  });

  it.each([
    {
      name: "Kimi K2.6 binary thinking",
      backend: "moonshot/kimi-k2.6",
      params: ["thinking"],
      reasoning: "high" as const,
      expected: {
        thinking: { type: "enabled" },
        include_reasoning: false,
        reasoning_content: false,
        merge_reasoning_content_in_choices: true,
      },
      absent: ["reasoning_effort"],
    },
    {
      name: "Kimi K3 effort",
      backend: "moonshot/kimi-k3",
      params: ["reasoning_effort"],
      reasoning: "max" as const,
      expected: {
        reasoning_effort: "max",
        include_reasoning: false,
        reasoning_content: false,
        merge_reasoning_content_in_choices: true,
      },
      absent: ["thinking"],
    },
    {
      name: "DeepSeek V4 native controls",
      backend: "deepseek/deepseek-v4",
      params: ["thinking", "reasoning_effort"],
      reasoning: "max" as const,
      expected: { thinking: { type: "enabled" }, reasoning_effort: "max" },
      absent: ["include_reasoning", "reasoning_content", "merge_reasoning_content_in_choices"],
    },
    {
      name: "Azure Foundry DeepSeek effort only",
      backend: "azure_ai/deepseek-v4",
      params: ["reasoning_effort"],
      reasoning: "high" as const,
      expected: { reasoning_effort: "high" },
      absent: ["thinking", "include_reasoning", "reasoning_content", "merge_reasoning_content_in_choices"],
    },
  ])("serializes $name from discovered policy", async ({ backend, params, reasoning, expected, absent }) => {
    const { models, model, requests, respond } = await createCompatibilityHarness([
      {
        model_name: "evidence-route",
        litellm_params: { model: backend, allowed_openai_params: params },
        model_info: { id: "deployment", mode: "chat" },
      },
    ]);
    respond(...successfulResponse("ok"));

    await models.streamSimple(model, { messages: [user("Think")] }, { reasoning }).result();

    expect(requests[0]).toMatchObject(expected);
    for (const field of absent) expect(requests[0]).not.toHaveProperty(field);
  });

  it.each([
    { name: "Kimi K3", backend: "moonshot/kimi-k3" },
    { name: "DeepSeek V4", backend: "deepseek/deepseek-v4-pro" },
  ])("sends no speculative controls for evidence-absent $name", async ({ backend }) => {
    const { models, model, requests, respond } = await createCompatibilityHarness([
      {
        model_name: "evidence-absent-route",
        litellm_params: { model: backend },
        model_info: { id: "deployment", mode: "chat", supports_reasoning: true },
      },
    ]);
    respond(...successfulResponse("ok"));

    await models.streamSimple(model, { messages: [user("Think")] }, { reasoning: "high" }).result();

    expect(model.reasoning).toBe(true);
    expect(getSupportedThinkingLevels(model)).toEqual([]);
    expect(requests).toHaveLength(1);
    expect(requests[0]).not.toHaveProperty("thinking");
    expect(requests[0]).not.toHaveProperty("reasoning_effort");
  });

  it("serializes disabled Kimi K2.6 as binary thinking off", async () => {
    const { models, model, requests, respond } = await createCompatibilityHarness([
      {
        model_name: "binary-route",
        litellm_params: { model: "moonshot/kimi-k2.6" },
        model_info: { id: "deployment", mode: "chat", supported_openai_params: ["thinking"] },
      },
    ]);
    respond(...successfulResponse("ok"));

    await models.streamSimple(model, { messages: [user("Think")] }).result();

    expect(requests[0]).toMatchObject({ thinking: { type: "disabled" } });
    expect(requests[0]).not.toHaveProperty("reasoning_effort");
  });

  it("transmits the Kimi K2.7 Code level it advertises and never invents off", async () => {
    const { models, model, requests, respond } = await createCompatibilityHarness([
      {
        model_name: "code-route",
        litellm_params: { model: "moonshot/kimi-k2.7-code" },
        model_info: { id: "deployment", mode: "chat", supported_openai_params: ["thinking"] },
      },
    ]);
    respond(...successfulResponse("ok"), ...successfulResponse("ok"));

    // `high` is the only offered level and it must reach the wire.
    expect(getSupportedThinkingLevels(model)).toEqual(["high"]);

    await models.streamSimple(model, { messages: [user("Think")] }, { reasoning: "high" }).result();

    expect(requests[0]).toMatchObject({
      thinking: { type: "enabled" },
      include_reasoning: false,
      reasoning_content: false,
      merge_reasoning_content_in_choices: true,
    });
    expect(requests[0]).not.toHaveProperty("reasoning_effort");

    // K2.7 Code cannot stop reasoning, so no disable request is fabricated.
    await models.streamSimple(model, { messages: [user("Think")] }).result();

    expect(requests[1]).not.toHaveProperty("thinking");
    expect(requests[1]).not.toHaveProperty("reasoning_effort");
  });

  it.each(["moonshot/kimi-k2.6", "deepseek/deepseek-v4"])(
    "does not translate thinking-only evidence into a Responses control for %s",
    async (backend) => {
      const { models, model, requests, respond } = await createCompatibilityHarness([
        {
          model_name: "thinking-only-responses",
          litellm_params: { model: backend, allowed_openai_params: ["thinking"] },
          model_info: { id: "deployment", mode: "responses", supports_reasoning: true },
        },
      ]);
      respond(...successfulResponsesReply("ok"));

      await models.streamSimple(model, { messages: [user("Think")] }, { reasoning: "high" }).result();

      expect(getSupportedThinkingLevels(model)).toEqual([]);
      expect(requests[0]).not.toHaveProperty("reasoning");
      expect(requests[0]).not.toHaveProperty("include");
    },
  );

  it("keeps route-only Kimi normalization policy without inventing generation controls", async () => {
    const { models, model, requests, respond } = await createCompatibilityHarness([
      { model_name: "kimi-k2.6", model_info: { id: "deployment", mode: "chat" } },
    ]);
    respond(...successfulResponse("ok"));

    await models.streamSimple(model, { messages: [user("Think")] }, { reasoning: "high" }).result();

    expect(model).toMatchObject({
      reasoning: false,
      litellmPolicy: { normalizeThinkTags: true, suppressReasoningVisibility: false },
    });
    expect(requests[0]).not.toHaveProperty("thinking");
    expect(requests[0]).not.toHaveProperty("reasoning_effort");
  });

  it.each([
    { name: "Kimi K2.7 Code", backend: "moonshot/kimi-k2.7-code" },
    { name: "Kimi K2.6", backend: "moonshot/kimi-k2.6" },
  ])("advertises no $name level it cannot transmit without control evidence", async ({ backend }) => {
    const { models, model, requests, respond } = await createCompatibilityHarness([
      {
        model_name: "evidence-absent-kimi",
        litellm_params: { model: backend },
        model_info: { id: "deployment", mode: "chat", supports_reasoning: true },
      },
    ]);
    respond(...successfulResponse("ok"));

    // An absent map would let pi-ai offer every standard level, none of which
    // this deployment can carry, so each level is denied explicitly instead.
    expect(model.reasoning).toBe(true);
    expect(getSupportedThinkingLevels(model)).toEqual([]);

    await models.streamSimple(model, { messages: [user("Think")] }, { reasoning: "high" }).result();

    expect(requests[0]).not.toHaveProperty("thinking");
    expect(requests[0]).not.toHaveProperty("reasoning_effort");
  });

  // Strict tool-message repair rewrites outbound messages, so it is applied only
  // when every routable deployment evidences the Moonshot need.
  it.each([
    {
      source: "route text only",
      rows: [{ model_name: "kimi-k3", model_info: { mode: "chat" } }],
      repaired: false,
    },
    {
      source: "a Moonshot deployment beside an unidentified one",
      rows: [
        { model_name: "mixed", litellm_params: { model: "moonshot/kimi-k2.6" }, model_info: { id: "a", mode: "chat" } },
        { model_name: "mixed", litellm_params: { model: "internal/opaque" }, model_info: { id: "b", mode: "chat" } },
      ],
      repaired: false,
    },
    {
      source: "every deployment evidencing Moonshot",
      rows: [
        {
          model_name: "unanimous",
          litellm_params: { model: "moonshot/kimi-k2.6" },
          model_info: { id: "a", mode: "chat" },
        },
        {
          model_name: "unanimous",
          litellm_params: { model: "moonshot/kimi-k3" },
          model_info: { id: "b", mode: "chat" },
        },
      ],
      repaired: true,
    },
  ])("repairs strict tool messages only with $source", async ({ rows, repaired }) => {
    const { models, model, requests, respond } = await createCompatibilityHarness(rows);
    respond(...successfulResponse("ok"));

    await models
      .streamSimple(model, {
        messages: [
          user("Call a tool"),
          {
            role: "assistant",
            content: [{ type: "toolCall", id: "call_1", name: "lookup", arguments: {} }],
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
            stopReason: "toolUse",
            timestamp: 2,
          },
          {
            role: "toolResult",
            toolCallId: "call_1",
            toolName: "lookup",
            content: [{ type: "text", text: "result" }],
            isError: false,
            timestamp: 3,
          },
        ],
      })
      .result();

    expect(model).toMatchObject({ litellmPolicy: { normalizeStrictToolMessages: repaired } });
    const assistant = requests[0]?.messages?.find((message) => message.role === "assistant");
    expect(assistant?.content).toBe(repaired ? "" : null);
  });

  it("withholds visibility suppression when a Kimi route mixes normal and always-thinking deployments", async () => {
    const { models, model, requests, respond } = await createCompatibilityHarness([
      {
        model_name: "mixed-kimi-mode-route",
        litellm_params: { model: "moonshot/kimi-k2.6" },
        model_info: { id: "normal", mode: "chat" },
      },
      {
        model_name: "mixed-kimi-mode-route",
        litellm_params: { model: "moonshot/kimi-k2-thinking" },
        model_info: { id: "thinking", mode: "chat" },
      },
    ]);
    respond(...successfulResponse("ok"));

    await models.streamSimple(model, { messages: [user("Think")] }).result();

    expect(model).toMatchObject({
      litellmPolicy: {
        normalizeStrictToolMessages: true,
        normalizeThinkTags: false,
        suppressReasoningVisibility: false,
      },
    });
    expect(requests[0]).not.toHaveProperty("include_reasoning");
    expect(requests[0]).not.toHaveProperty("reasoning_content");
    expect(requests[0]).not.toHaveProperty("merge_reasoning_content_in_choices");
  });

  it.each([
    { name: "Kimi K3", backend: "moonshot/kimi-k3", params: ["reasoning_effort"] },
    { name: "Kimi K2.7 Code", backend: "moonshot/kimi-k2.7-code", params: ["thinking"] },
    {
      name: "DeepSeek V4",
      backend: "deepseek/deepseek-v4",
      params: ["thinking", "reasoning_effort"],
    },
  ])("replays required reasoning content for $name", async ({ backend, params }) => {
    const { models, model, requests, respond } = await createCompatibilityHarness([
      {
        model_name: "replay-route",
        litellm_params: { model: backend },
        model_info: { mode: "chat", supported_openai_params: params },
      },
    ]);
    respond(...successfulResponse("ok"));

    await models
      .streamSimple(model, {
        messages: [
          user("First"),
          {
            role: "assistant",
            content: [{ type: "text", text: "Answer" }],
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
            stopReason: "stop",
            timestamp: 2,
          },
          user("Continue"),
        ],
      })
      .result();

    expect(requests[0]?.messages).toContainEqual(expect.objectContaining({ role: "assistant", reasoning_content: "" }));
  });

  it.each([
    {
      name: "Kimi K3",
      backend: "moonshot/kimi-k3",
      params: ["reasoning_effort"],
      expected: { reasoning_effort: "max" },
      absent: ["thinking"],
    },
    {
      name: "DeepSeek V4",
      backend: "deepseek/deepseek-v4",
      params: ["thinking", "reasoning_effort"],
      expected: { thinking: { type: "enabled" }, reasoning_effort: "max" },
      absent: [],
    },
  ])("rehydrates exact $name policy and wire behavior offline", async ({ backend, params, expected, absent }) => {
    const modelsStore = new InMemoryModelsStore();
    const rows = [
      {
        model_name: "cached-reasoning-route",
        litellm_params: { model: backend },
        model_info: { mode: "chat", supported_openai_params: params },
      },
    ];
    const online = await createCompatibilityHarness(rows, { modelsStore });
    online.respond(...successfulResponse("online"));
    const onlineMessage = await online.models
      .streamSimple(online.model, { messages: [user("Think")] }, { reasoning: "max" })
      .result();
    expect(onlineMessage.stopReason).toBe("stop");

    vi.restoreAllMocks();
    vi.resetModules();
    const offline = await createCompatibilityHarness(rows, { modelsStore, allowNetwork: false });
    offline.respond(...successfulResponse("offline"));
    const offlineMessage = await offline.models
      .streamSimple(offline.model, { messages: [user("Think")] }, { reasoning: "max" })
      .result();
    expect(offlineMessage.stopReason).toBe("stop");

    expect(offline.model).toEqual(online.model);
    expect(offline.requests[0]).toEqual(online.requests[0]);
    expect(offline.requests[0]).toMatchObject(expected);
    for (const field of absent) expect(offline.requests[0]).not.toHaveProperty(field);
  });

  it("handles thinking and a tool result across turns", async () => {
    const { models, model, requests, respond } = await createCompatibilityHarness();
    const context: Context = { messages: [user("Calculate")], tools: [] };
    respond(
      sseChunk({ choices: [{ delta: { reasoning_content: "714" }, finish_reason: null }] }),
      sseChunk({
        choices: [
          {
            delta: {
              tool_calls: [{ index: 0, id: "call_2", type: "function", function: { name: "lookup", arguments: "{}" } }],
            },
            finish_reason: "tool_calls",
          },
        ],
        usage: { prompt_tokens: 2, completion_tokens: 2 },
      }),
    );
    const first = await models.streamSimple(model, context, { reasoning: "high" }).result();
    expect(first.content).toContainEqual(expect.objectContaining({ type: "thinking", thinking: "714" }));

    context.messages.push(first, {
      role: "toolResult",
      toolCallId: "call_2",
      toolName: "lookup",
      content: [{ type: "text", text: "887" }],
      isError: false,
      timestamp: 2,
    });
    respond(
      sseChunk({ choices: [{ delta: { content: "887" }, finish_reason: null }] }),
      sseChunk({ choices: [{ delta: {}, finish_reason: "stop" }], usage: { prompt_tokens: 4, completion_tokens: 1 } }),
    );
    const second = await models.streamSimple(model, context).result();

    expect(second.stopReason).toBe("stop");
    expect(second.content).toContainEqual({ type: "text", text: "887" });
    expect(requests).toHaveLength(2);
    expect(requests.at(-1)?.messages).toContainEqual(expect.objectContaining({ role: "tool", content: "887" }));
  });

  it("serializes image input", async () => {
    const { models, model, requests, respond } = await createCompatibilityHarness();
    respond(
      sseChunk({
        choices: [{ delta: { content: "red" }, finish_reason: "stop" }],
        usage: { prompt_tokens: 2, completion_tokens: 1 },
      }),
    );

    await models
      .streamSimple(model, {
        messages: [
          { role: "user", content: [{ type: "image", data: RED_CIRCLE_PNG, mimeType: "image/png" }], timestamp: 1 },
        ],
      })
      .result();

    expect(requests[0]?.messages[0]?.content).toContainEqual({
      type: "image_url",
      image_url: { url: `data:image/png;base64,${RED_CIRCLE_PNG}` },
    });
  });
});

// The branch's central invariant: pi-ai must never be handed a thinking level it
// cannot serialize. Advertisement (thinkingLevelMap) and transmissibility
// (compat) used to travel on separate channels, so four different paths offered
// levels that reached the wire as nothing. This sweeps the semantic families and
// evidence shapes against real request bodies so the invariant is enforced
// mechanically rather than restated per model.
describe("advertised thinking levels are transmissible", () => {
  const BACKENDS = [
    "moonshot/kimi-k2.5",
    "moonshot/kimi-k2.6",
    "moonshot/kimi-k2.7-code",
    "moonshot/kimi-k2.7-code-highspeed",
    "moonshot/kimi-k3",
    // Kimi generations with no tabled contract: the live proxies expose no
    // reasoning controls for these, so they must fail closed rather than guess.
    "moonshot/kimi-k2-thinking",
    "moonshot/kimi-latest",
    "moonshot/kimi-k2-0905-preview",
    "deepseek/deepseek-v4",
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-r1",
    "azure_ai/deepseek-v4",
    // Non-Kimi families must keep working through the generic effort path.
    "openai/o3",
    "openai/gpt-5.5",
    "anthropic/claude-sonnet-4-6",
    "internal/opaque-backend",
  ];
  const EVIDENCE = [
    { name: "no declared params", params: undefined },
    { name: "unrelated params", params: ["temperature"] },
    { name: "thinking", params: ["thinking"] },
    { name: "reasoning_effort", params: ["reasoning_effort"] },
    { name: "both controls", params: ["thinking", "reasoning_effort"] },
  ];

  it.each(BACKENDS.flatMap((backend) => EVIDENCE.map((evidence) => ({ backend, evidence }))))(
    "$backend with $evidence.name never advertises an untransmittable level",
    async ({ backend, evidence }) => {
      const { models, model, requests, respond } = await createCompatibilityHarness([
        {
          model_name: "sweep-route",
          litellm_params: { model: backend, ...(evidence.params ? { allowed_openai_params: evidence.params } : {}) },
          model_info: { id: "d1", mode: "chat", supports_reasoning: true },
        },
      ]);

      const offered = getSupportedThinkingLevels(model).filter((level) => level !== "off");
      for (const level of offered) {
        respond(...successfulResponse("ok"));
        await models.streamSimple(model, { messages: [user("Think")] }, { reasoning: level }).result();
        const body = requests.at(-1) ?? {};
        const carried = Object.keys(body).filter((key) => /^(reasoning|reasoning_effort|thinking)$/.test(key));
        // A level the picker offers must put something on the wire.
        expect(carried, `${backend} / ${evidence.name} advertised "${level}" but sent nothing`).not.toEqual([]);
      }

      // And when nothing is transmissible the denial must be explicit: an absent
      // map means "every standard level supported" to pi-ai.
      if (offered.length === 0) {
        respond(...successfulResponse("ok"));
        await models.streamSimple(model, { messages: [user("Think")] }, { reasoning: "high" }).result();
        expect(requests.at(-1)).not.toHaveProperty("reasoning_effort");
        expect(requests.at(-1)).not.toHaveProperty("thinking");
        if (model.reasoning) expect(model.thinkingLevelMap).toEqual(NO_LEVELS);
      }
    },
  );
});

// Required property: for every model any discovery path can produce, on either
// API, every level the picker offers must serialize a control the API recognizes
// — or no level is offered. Verified against the real serializers, not a
// re-implementation of the rule under test.
describe("advertised levels serialize on both APIs", () => {
  // Values each API actually accepts. Chat carries a level through
  // `reasoning_effort` or a `thinking` payload; Responses carries `reasoning.effort`
  // and has no `off`/`max` effort.
  const RESPONSES_EFFORTS = new Set(["none", "minimal", "low", "medium", "high", "xhigh"]);

  const BACKENDS = [
    "moonshot/kimi-k2.5",
    "moonshot/kimi-k2.6",
    "moonshot/kimi-k2.7-code",
    "moonshot/kimi-k3",
    // Untabled and future Kimi shapes: negative cases that must fail closed.
    "moonshot/kimi-k2-thinking",
    "moonshot/kimi-latest",
    "moonshot/kimi-k4",
    "moonshot/kimi-k9.9-ultra",
    "moonshot/kimi-k2.8-code",
    "deepseek/deepseek-v4",
    "deepseek/deepseek-r1",
    "openai/o3",
    "openai/gpt-5.5",
    "anthropic/claude-sonnet-4-6",
    "internal/opaque-backend",
  ];
  const EVIDENCE = [undefined, ["temperature"], ["thinking"], ["reasoning_effort"], ["thinking", "reasoning_effort"]];

  it.each(
    BACKENDS.flatMap((backend) =>
      ["chat", "responses"].flatMap((mode) => EVIDENCE.map((params) => ({ backend, mode, params }))),
    ),
  )("$backend / $mode / $params", async ({ backend, mode, params }) => {
    const { models, model, requests, respond } = await createCompatibilityHarness([
      {
        model_name: "prop-route",
        litellm_params: { model: backend, ...(params ? { allowed_openai_params: params } : {}) },
        model_info: { id: "d1", mode, supports_reasoning: true },
      },
    ]);

    const reply = model.api === "openai-responses" ? successfulResponsesReply : successfulResponse;
    const offered = getSupportedThinkingLevels(model).filter((level) => level !== "off");
    for (const level of offered) {
      respond(...reply("ok"));
      await models.streamSimple(model, { messages: [user("Think")] }, { reasoning: level }).result();
      const body = (requests.at(-1) ?? {}) as Record<string, unknown>;
      const label = `${backend} / ${mode} / ${JSON.stringify(params)} level=${level}`;

      if (model.api === "openai-responses") {
        const reasoning = body.reasoning as { effort?: string } | undefined;
        expect(params?.includes("reasoning_effort"), `${label} lacks accepted Responses control evidence`).toBe(true);
        expect(reasoning?.effort, `${label} sent no reasoning.effort`).toBeDefined();
        // `off` and `max` are not Responses efforts; a Chat-shaped map leaking
        // through would emit exactly those.
        expect(RESPONSES_EFFORTS.has(String(reasoning?.effort)), `${label} effort=${reasoning?.effort}`).toBe(true);
      } else {
        const carried = ["reasoning_effort", "thinking", "reasoning"].filter((key) => body[key] !== undefined);
        expect(carried, `${label} advertised a level but sent nothing`).not.toEqual([]);
      }
    }

    if (offered.length === 0 && model.reasoning) {
      // The no-level conclusion must be explicit; an absent map means every
      // standard level upstream.
      expect(model.thinkingLevelMap).toEqual(NO_LEVELS);
    }
  });
});
