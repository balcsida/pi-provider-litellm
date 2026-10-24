import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type {
  AnyModel,
  ClassifierContext,
  ClassifierModel,
  Credential,
  ModelsPublication,
  ProviderAuth,
  RefreshModelsContext,
} from "@earendil-works/pi-ai";
import { afterAll, afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-classifiers-"));
vi.mock("@earendil-works/pi-coding-agent", () => ({
  getAgentDir: () => agentDir,
}));

import { LITELLM_DISCOVERY_VERSION } from "../src/backend-identity.js";
import {
  classify,
  isDecisionDeployment,
  LITELLM_DECISIONS_API,
  type LiteLLMClassifierModel,
  mapFromClassifierModelInfoGroup,
} from "../src/classifiers.js";
import { discoverModels } from "../src/discover.js";
import { defaultContextWindow } from "../src/model-groups.js";
import { readStoredModels } from "../src/models-store.js";
import { createLiteLLMProvider, toNativeClassifierModels, toNativeModels } from "../src/provider.js";
import type { DiscoveryResult, ModelInfoEntry } from "../src/types.js";

beforeEach(() => {
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    throw new Error(`unstubbed fetch: ${input instanceof URL ? input.toString() : String(input)}`);
  });
});

afterEach(() => {
  vi.restoreAllMocks();
});

afterAll(async () => {
  await rm(agentDir, { recursive: true, force: true });
});

function jsonResponse(status: number, body: unknown, headers: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json", ...headers },
  });
}

function decisionRow(route: string, overrides: NonNullable<ModelInfoEntry["model_info"]> = {}): ModelInfoEntry {
  return {
    model_name: route,
    litellm_params: { model: "azure_ai/Microsoft-Decision-1", custom_llm_provider: "azure_ai" },
    model_info: {
      mode: "evaluation",
      supported_endpoints: null,
      input_cost_per_token: 1e-6,
      output_cost_per_token: 4e-6,
      max_input_tokens: 65_536,
      ...overrides,
    },
  };
}

function chatRow(route: string, overrides: NonNullable<ModelInfoEntry["model_info"]> = {}): ModelInfoEntry {
  return {
    model_name: route,
    litellm_params: { model: "openai/gpt-6.1-sol", custom_llm_provider: "openai" },
    model_info: {
      mode: "chat",
      input_cost_per_token: 2e-6,
      output_cost_per_token: 1e-5,
      max_input_tokens: 1_050_000,
      ...overrides,
    },
  };
}

function classifierModel(overrides: Partial<LiteLLMClassifierModel> = {}): LiteLLMClassifierModel {
  return {
    ...toNativeClassifierModels("litellm", "https://proxy.example", [
      mapFromClassifierModelInfoGroup("decision-1", [decisionRow("decision-1")])!,
    ])[0]!,
    ...overrides,
  };
}

const context: ClassifierContext = {
  state: { text: "ship it" },
  questions: {
    safe: { type: "bool", instructions: "Is it safe?", criteria: { true: "safe", false: "unsafe" } },
    tone: { type: "choice", instructions: "Tone?", criteria: { calm: "calm", angry: "angry" } },
    quality: { type: "score", instructions: "Quality?", criteria: ["bad", "good"] },
  },
};

const wireAnswers = {
  safe: { type: "noul", noul: 0.99 },
  tone: { type: "choice", choice: "calm", confidence: 1, probabilities: { calm: 0.9, angry: 0.1 } },
  quality: { type: "score", score: 0.95, confidence: 0.86, legend: {}, probabilities: {} },
};

describe("decision route discovery", () => {
  it("recognizes decision deployments by mode or by declared endpoints", () => {
    const withMode = (mode: unknown) => decisionRow("r", { mode: mode as string });
    for (const mode of ["evaluation", "decisions", " Evaluation ", "DECISIONS"]) {
      expect(isDecisionDeployment(withMode(mode))).toBe(true);
    }
    for (const mode of ["chat", "image_generation", "", undefined, null, 1]) {
      expect(isDecisionDeployment(withMode(mode))).toBe(false);
    }
    for (const endpoint of ["/v1/systemone", "/v1/decisions"]) {
      expect(
        isDecisionDeployment(chatRow("r", { supported_endpoints: ["/v1/chat/completions", ` ${endpoint} `] })),
      ).toBe(true);
    }
    expect(isDecisionDeployment(decisionRow("r", { supported_endpoints: ["/v1/chat/completions"] }))).toBe(false);
  });

  it("maps a unanimous group to a classifier model with per-million prices", () => {
    expect(mapFromClassifierModelInfoGroup("decision-1", [decisionRow("decision-1")])).toEqual({
      type: "classifier",
      id: "decision-1",
      name: "decision-1",
      api: LITELLM_DECISIONS_API,
      input: ["text"],
      cost: { input: 1, output: 4, cacheRead: 0, cacheWrite: 0 },
      contextWindow: 65_536,
      litellmDiscoveryVersion: LITELLM_DISCOVERY_VERSION,
    });
  });

  it("takes the highest declared rate and reports 0 when any deployment omits one", () => {
    const priced = mapFromClassifierModelInfoGroup("d", [
      decisionRow("d"),
      decisionRow("d", { input_cost_per_token: 3e-6, output_cost_per_token: 9e-6 }),
    ]);
    expect(priced?.cost).toMatchObject({ input: 3, output: 9 });
    const partial = mapFromClassifierModelInfoGroup("d", [
      decisionRow("d"),
      decisionRow("d", { input_cost_per_token: undefined, output_cost_per_token: undefined }),
    ]);
    expect(partial?.cost).toMatchObject({ input: 0, output: 0 });
  });

  it("takes the smallest declared window and falls back to the default for an undeclared one", () => {
    const smallest = mapFromClassifierModelInfoGroup("d", [
      decisionRow("d"),
      decisionRow("d", { max_input_tokens: 8192 }),
    ]);
    expect(smallest?.contextWindow).toBe(8192);
    const undeclared = mapFromClassifierModelInfoGroup("d", [
      decisionRow("d", { max_input_tokens: 1_000_000 }),
      decisionRow("d", { max_input_tokens: undefined }),
    ]);
    expect(undeclared?.contextWindow).toBe(defaultContextWindow());
  });

  it("withholds mixed, mode-less, wildcard and empty groups", () => {
    expect(mapFromClassifierModelInfoGroup("mixed", [decisionRow("mixed"), chatRow("mixed")])).toBeUndefined();
    expect(mapFromClassifierModelInfoGroup("bare", [{ model_name: "bare", model_info: {} }])).toBeUndefined();
    expect(mapFromClassifierModelInfoGroup("d/*", [decisionRow("d/*")])).toBeUndefined();
    expect(mapFromClassifierModelInfoGroup("empty", [])).toBeUndefined();
  });

  it("publishes decision routes from /model/info as classifiers, never as chat", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [
            chatRow("gpt-6.1-sol"),
            decisionRow("decision-1"),
            chatRow("gpt-6-luna", { supported_endpoints: ["/v1/chat/completions", "/v1/decisions"] }),
          ],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const result = await discoverModels("https://proxy.example", "secret", { silent: true });
    expect(result.models.map((model) => model.id)).toEqual(["gpt-6.1-sol", "gpt-6-luna"]);
    expect(result.classifierModels?.map((model) => model.id)).toEqual(["decision-1", "gpt-6-luna"]);
  });

  it("leaves the discovery result unchanged when the proxy has no decision routes", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return jsonResponse(200, { data: [chatRow("gpt-6.1-sol")] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const result = await discoverModels("https://proxy.example", "secret", { silent: true });
    expect(result).not.toHaveProperty("classifierModels");
  });
});

describe("classify", () => {
  it("posts System One questions and normalizes the answers", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(
        jsonResponse(
          200,
          { model: "decision-1", answers: wireAnswers, usage: { input_tokens: 391, output_tokens: 67 } },
          { "x-litellm-response-cost": "0.0125" },
        ),
      );
    const onResponse = vi.fn();
    const result = await classify(classifierModel(), context, {
      apiKey: "secret",
      headers: { "x-team": "ai", "x-drop": null },
      onResponse,
    });

    expect(result.stopReason).toBe("stop");
    expect(result.answers).toEqual({
      safe: { type: "bool", probability: 0.99 },
      tone: { type: "choice", choice: "calm", probabilities: { calm: 0.9, angry: 0.1 }, confidence: 1 },
      quality: { type: "score", score: 0.95, confidence: 0.86 },
    });
    const [url, init] = fetchSpy.mock.calls[0]!;
    expect(url).toBe("https://proxy.example/v1/systemone");
    expect(JSON.parse(String(init?.body))).toEqual({
      model: "decision-1",
      state: { text: "ship it" },
      questions: {
        safe: { type: "noul", instructions: "Is it safe?", criteria: { true: "safe", false: "unsafe" } },
        tone: context.questions.tone,
        quality: context.questions.quality,
      },
    });
    expect(init?.headers).toMatchObject({ authorization: "Bearer secret", "x-team": "ai" });
    expect(init?.headers).not.toHaveProperty("x-drop");
    expect(onResponse).toHaveBeenCalledWith(expect.objectContaining({ status: 200 }), expect.anything());
    expect(result.usage).toMatchObject({ input: 391, output: 67, totalTokens: 458 });
    expect(result.usage?.cost.total).toBeCloseTo(0.0125, 9);
    expect(result.usage!.cost.input + result.usage!.cost.output).toBeCloseTo(0.0125, 9);
  });

  it("merges header overrides case-insensitively and keeps its own bearer", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { answers: wireAnswers }));
    await classify(
      classifierModel({ headers: { "X-Tenant": "old", "X-Drop": "gone", Authorization: "Bearer model" } }),
      context,
      { apiKey: "secret", headers: { "x-tenant": "new", "x-drop": null } },
    );
    const sent = new Headers(fetchSpy.mock.calls[0]![1]?.headers);
    expect(sent.get("x-tenant")).toBe("new");
    expect(sent.has("x-drop")).toBe(false);
    expect(sent.get("authorization")).toBe("Bearer secret");
  });

  it("estimates cost from discovered rates when the proxy sends no cost header", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, { answers: wireAnswers, usage: { input_tokens: 100, output_tokens: 10 } }),
    );
    const result = await classify(classifierModel(), context, { apiKey: "k" });
    // 100 x $1/M + 10 x $4/M
    expect(result.usage?.cost.total).toBeCloseTo(100 * 1e-6 + 10 * 4e-6, 12);
  });

  it("takes the proxy's cost even when the response carries no usage", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, { answers: wireAnswers }, { "x-litellm-response-cost": "0.002" }),
    );
    const result = await classify(classifierModel(), context, { apiKey: "k" });
    expect(result.usage).toMatchObject({ input: 0, output: 0, totalTokens: 0 });
    expect(result.usage?.cost.total).toBeCloseTo(0.002, 9);
  });

  it("lets onPayload replace the body and uses a caller-supplied fetch", async () => {
    const customFetch = vi.fn(async () => jsonResponse(200, { answers: wireAnswers }));
    const result = await classify(classifierModel(), context, {
      apiKey: "k",
      fetch: customFetch as unknown as typeof fetch,
      onPayload: (payload) => ({ ...(payload as object), extra: true }),
    });
    expect(result.stopReason).toBe("stop");
    expect(customFetch).toHaveBeenCalledOnce();
    expect(JSON.parse(String((customFetch.mock.calls[0] as unknown as [string, RequestInit])[1].body))).toMatchObject({
      extra: true,
    });
  });

  it.each([
    ["no API key", {}, context, /No API key/],
    [
      "an image in the context",
      { apiKey: "k" },
      { ...context, images: [{ type: "image" as const, data: "AAAA", mimeType: "image/png" }] },
      /text state only/,
    ],
  ])("returns an error result for %s without calling the proxy", async (_label, options, input, message) => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const result = await classify(classifierModel(), input, options);
    expect(result.stopReason).toBe("error");
    expect(result.answers).toEqual({});
    expect(result.errorMessage).toMatch(message);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("reports proxy errors with a bounded message and never rejects", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(400, { error: { message: `Bad decision request. ${"x".repeat(1000)}` } }),
    );
    const result = await classify(classifierModel(), context, { apiKey: "k" });
    expect(result.stopReason).toBe("error");
    expect(result.errorMessage).toMatch(/^LiteLLM decisions request failed \(400\): Bad decision request\./);
    expect(result.errorMessage!.length).toBeLessThan(400);
  });

  it("reports non-JSON error bodies and unparsable success bodies as errors", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(new Response("Service Unavailable", { status: 503 }));
    const failed = await classify(classifierModel(), context, { apiKey: "k" });
    expect(failed.stopReason).toBe("error");
    expect(failed.errorMessage).toBe("LiteLLM decisions request failed (503): Service Unavailable");

    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("<html>", { status: 200, headers: { "x-litellm-response-cost": "0.002" } }),
    );
    const garbled = await classify(classifierModel(), context, { apiKey: "k" });
    expect(garbled.stopReason).toBe("error");
    expect(garbled.answers).toEqual({});
    // The proxy charged for the request even though its reply could not be read.
    expect(garbled.usage?.cost.total).toBeCloseTo(0.002, 9);
  });

  it.each([
    ["a non-object body", [], /unexpected response/],
    ["a missing answer", { answers: { ...wireAnswers, tone: undefined } }, /did not return a choice answer for tone/],
    [
      "a wrong answer type",
      { answers: { ...wireAnswers, safe: wireAnswers.quality } },
      /did not return a bool answer for safe/,
    ],
    [
      "a non-finite probability",
      { answers: { ...wireAnswers, tone: { ...wireAnswers.tone, probabilities: { calm: "high" } } } },
      /invalid probabilities for tone/,
    ],
    [
      "a noul answer without a number",
      { answers: { ...wireAnswers, safe: { type: "noul" } } },
      /invalid noul for safe/,
    ],
  ])("returns an error result for %s", async (_label, body, message) => {
    const withUsage = Array.isArray(body) ? body : { ...body, usage: { input_tokens: 10, output_tokens: 2 } };
    vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, withUsage));
    const result = await classify(classifierModel(), context, { apiKey: "k" });
    expect(result.stopReason).toBe("error");
    expect(result.answers).toEqual({});
    expect(result.errorMessage).toMatch(message);
    // A malformed answer set was still billed.
    if (!Array.isArray(body)) expect(result.usage).toMatchObject({ input: 10, output: 2 });
  });

  it("marks a cancelled request as aborted", async () => {
    const controller = new AbortController();
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      controller.abort();
      throw new DOMException("aborted", "AbortError");
    });
    const result = await classify(classifierModel(), context, { apiKey: "k", signal: controller.signal });
    expect(result.stopReason).toBe("aborted");
  });

  it("fails instead of hanging when timeoutMs elapses", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(
      (_input, init) =>
        new Promise((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => reject(init.signal?.reason));
        }),
    );
    const result = await classify(classifierModel(), context, { apiKey: "k", timeoutMs: 10 });
    expect(result.stopReason).toBe("error");
    expect(result.errorMessage).toMatch(/timeout/i);
  });
});

describe("stored classifier models", () => {
  const store = (baseUrl: string) => ({
    litellm: { checkedAt: Date.now(), models: [{ ...classifierModel(), baseUrl }] },
  });

  it("restores a stored classifier model as it is and discovers again for a stale root", async () => {
    const options = { agentDir, provider: "litellm", baseUrl: "https://proxy.example" };
    await writeFile(join(agentDir, "models-store.json"), JSON.stringify(store("https://proxy.example/v1")));
    expect(await readStoredModels(options)).toEqual([classifierModel()]);
    await writeFile(join(agentDir, "models-store.json"), JSON.stringify(store("https://other.example/v1")));
    expect(await readStoredModels(options)).toBeUndefined();
  });
});

describe("LiteLLM provider classifier models", () => {
  const credential: Credential = { type: "api_key", key: "secret" };
  const auth: ProviderAuth = {
    apiKey: { name: "API key", resolve: async () => ({ auth: { apiKey: "secret" } }) },
  };
  const discovered = (): DiscoveryResult => ({
    source: "model_info",
    models: [],
    classifierModels: [mapFromClassifierModelInfoGroup("decision-1", [decisionRow("decision-1")])!],
  });

  function refreshContext(stored: readonly AnyModel[] | undefined, allowNetwork: boolean): RefreshModelsContext {
    return {
      stored: stored ? { models: stored as never, checkedAt: 1 } : undefined,
      allowNetwork,
      credential,
      signal: new AbortController().signal,
      publish: vi.fn(async (publication: ModelsPublication) => {
        publication.update?.();
        return true;
      }),
    };
  }

  function provider(root = "https://proxy.example", discover = vi.fn(async () => discovered())) {
    return createLiteLLMProvider({
      id: "litellm",
      name: "LiteLLM",
      baseUrl: "https://proxy.example/v1",
      auth,
      discover,
      resolveCredentialRoot: () => root,
    });
  }

  const classifiers = (models: readonly AnyModel[] | undefined) =>
    (models ?? []).filter((model): model is ClassifierModel<string> => model.type === "classifier");

  it("publishes discovered classifier models alongside chat models", async () => {
    const value = provider();
    await value.refreshModels?.(refreshContext(undefined, true));
    expect(value.getModels()).toEqual([]);
    expect(classifiers(value.getAllModels?.())).toEqual([
      expect.objectContaining({ id: "decision-1", provider: "litellm", baseUrl: "https://proxy.example/v1" }),
    ]);
  });

  it("keeps the chat and classifier seeds of a route published as both apart", async () => {
    const [chat] = toNativeModels("litellm", "https://proxy.example/v1", [
      {
        id: "gpt-6-luna",
        name: "gpt-6-luna",
        api: "openai-completions",
        reasoning: false,
        input: ["text"],
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
        contextWindow: 128_000,
        maxTokens: 4096,
        litellmDiscoveryVersion: LITELLM_DISCOVERY_VERSION,
      },
    ]);
    const discover = vi.fn(async () => discovered());
    const value = createLiteLLMProvider({
      id: "litellm",
      name: "LiteLLM",
      baseUrl: "https://proxy.example/v1",
      auth,
      discover,
      resolveCredentialRoot: () => "https://proxy.example",
      models: [chat, classifierModel({ id: "gpt-6-luna" })],
    });
    // The classifier seed shares the chat seed's id and base URL; a stale stored chat entry must still find the
    // fresh chat seed instead of being restored as legacy.
    const staleChat = { ...chat, litellmDiscoveryVersion: 1 } as unknown as AnyModel;
    await value.refreshModels?.(refreshContext([staleChat], false));
    expect(value.getModels()).toEqual([
      expect.objectContaining({ id: "gpt-6-luna", litellmDiscoveryVersion: LITELLM_DISCOVERY_VERSION }),
    ]);
    expect(discover).not.toHaveBeenCalled();
  });

  it("restores current stored classifier models offline and drops stale-version and foreign-api ones", async () => {
    const discover = vi.fn(async () => discovered());
    const value = provider(undefined, discover);
    const current = classifierModel({ id: "kept" });
    const stale = { ...classifierModel({ id: "stale" }), litellmDiscoveryVersion: 1 } as unknown as AnyModel;
    const foreign = { ...classifierModel({ id: "foreign" }), api: "other-decisions" } as AnyModel;
    await value.refreshModels?.(refreshContext([current, stale, foreign], false));
    expect(classifiers(value.getAllModels?.()).map((model) => model.id)).toEqual(["kept"]);
    expect(discover).not.toHaveBeenCalled();
  });

  it("hides cached classifier models whose root no longer matches the credential", () => {
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const value = provider("https://other.example");
    expect(value.filterAllModels?.([classifierModel()], credential) ?? []).toEqual([]);
    expect(stderr).toHaveBeenCalledWith(expect.stringContaining("stale LiteLLM model root"));
  });

  it("rebases a matching cached classifier model onto the credential root", () => {
    const value = provider("https://proxy.example/v1/");
    expect(value.filterAllModels?.([classifierModel()], credential)).toEqual([
      expect.objectContaining({ id: "decision-1", baseUrl: "https://proxy.example/v1" }),
    ]);
  });

  it("classifies through the credential's root and refuses a stale cached root", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(jsonResponse(200, { answers: wireAnswers }));
    const ok = await provider().classify!(classifierModel(), context, { apiKey: "k" });
    expect(ok.stopReason).toBe("stop");
    expect(fetchSpy.mock.calls[0]![0]).toBe("https://proxy.example/v1/systemone");

    fetchSpy.mockClear();
    const stale = await provider("https://other.example").classify!(classifierModel(), context, { apiKey: "k" });
    expect(stale.stopReason).toBe("error");
    expect(stale.errorMessage).toMatch(/stale LiteLLM model root/);
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});
