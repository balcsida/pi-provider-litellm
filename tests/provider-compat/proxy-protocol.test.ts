import { type Api, InMemoryModelsStore, type Model } from "@earendil-works/pi-ai";
import { describe, expect, it, vi } from "vitest";
import { createCompatibilityHarness, successfulResponsesReply, user } from "./helpers.js";

// A synthetic route served by an Azure deployment that LiteLLM exposes on /v1/responses.
const responsesRoute = {
  model_name: "team-responses",
  litellm_params: { model: "azure/gpt-5.5", custom_llm_provider: "azure" },
  model_info: { mode: "responses", supported_endpoints: ["/v1/responses"], supports_reasoning: true },
};

// The host can hand the provider a model whose API came from Pi's Azure catalog instead of discovery.
function withApi(model: Model<Api>, api: string): Model<Api> {
  return { ...model, api } as Model<Api>;
}

describe("Azure Responses API identifier on a LiteLLM model", () => {
  it("dispatches through the proxy's Responses endpoint with the proxy credential", async () => {
    const { models, model, requestUrls, requestHeaders, respond } = await createCompatibilityHarness(responsesRoute);
    respond(...successfulResponsesReply("hello"));

    const message = await models
      .streamSimple(withApi(model, "azure-openai-responses"), { messages: [user("Hello")] })
      .result();

    expect(message).toMatchObject({
      api: "openai-responses",
      stopReason: "stop",
      content: [{ type: "text", text: "hello" }],
    });
    expect(requestUrls).toEqual(["https://proxy.example.com/v1/responses"]);
    expect(requestHeaders[0]?.get("authorization")).toBe("Bearer sk-test");
    expect(requestHeaders[0]?.get("api-key")).toBeNull();
  });

  it("still rejects a protocol the proxy does not serve before any request", async () => {
    const { models, model, requestUrls } = await createCompatibilityHarness(responsesRoute);

    const message = await models
      .streamSimple(withApi(model, "google-generative-ai"), { messages: [user("Hello")] })
      .result();

    expect(message.stopReason).toBe("error");
    expect(message.errorMessage).toMatch(/declares unsupported protocol "google-generative-ai"/);
    expect(requestUrls).toEqual([]);
  });

  it("restores a cached Azure Responses entry onto the proxy's Responses protocol offline", async () => {
    const modelsStore = new InMemoryModelsStore();
    const online = await createCompatibilityHarness(responsesRoute, { modelsStore });
    const stored = await modelsStore.read("litellm");
    if (!stored) throw new Error("discovery was not stored");
    await modelsStore.write("litellm", {
      ...stored,
      models: stored.models.map((entry) => ({ ...entry, api: "azure-openai-responses" }) as typeof entry),
    });

    vi.restoreAllMocks();
    vi.resetModules();
    const offline = await createCompatibilityHarness(responsesRoute, { modelsStore, allowNetwork: false });
    offline.respond(...successfulResponsesReply("cached"));
    const message = await offline.models.streamSimple(offline.model, { messages: [user("Hello")] }).result();

    expect(offline.model).toEqual(online.model);
    expect(message).toMatchObject({ api: "openai-responses", stopReason: "stop" });
    expect(offline.requestUrls).toEqual(["https://proxy.example.com/v1/responses"]);
  });
});
