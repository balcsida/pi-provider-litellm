import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type {
  AnyModel,
  Credential,
  ImageModel,
  ModelsPublication,
  ProviderAuth,
  RefreshModelsContext,
} from "@earendil-works/pi-ai";
import { afterAll, afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const agentDir = await mkdtemp(join(tmpdir(), "pi-litellm-images-"));
vi.mock("@earendil-works/pi-coding-agent", () => ({
  getAgentDir: () => agentDir,
}));

import { LITELLM_DISCOVERY_VERSION } from "../src/backend-identity.js";
import { discoverModels } from "../src/discover.js";
import {
  generateImages,
  isImageGenerationMode,
  LITELLM_IMAGES_API,
  type LiteLLMImageModel,
  mapFromImageModelInfoGroup,
} from "../src/images.js";
import { createLiteLLMProvider, toNativeImageModels } from "../src/provider.js";
import type { DiscoveryResult, ModelInfoEntry } from "../src/types.js";

// Small but real magic-byte prefixes: "/9j/" is JPEG, "iVBORw0KGgo" is PNG.
const JPEG_B64 = "/9j/4AAQSkZJRgABAQAAAQABAAD";
const PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAE";

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

function imageRow(route: string, overrides: NonNullable<ModelInfoEntry["model_info"]> = {}): ModelInfoEntry {
  return {
    model_name: route,
    litellm_params: { model: "openrouter/google/gemini-nano-banana-2.1", custom_llm_provider: "openrouter" },
    model_info: {
      mode: "image_generation",
      input_cost_per_token: 1.5e-6,
      output_cost_per_token: 7.5e-6,
      output_cost_per_image_token: 3e-5,
      ...overrides,
    },
  };
}

function chatRow(route: string, mode = "chat"): ModelInfoEntry {
  return {
    model_name: route,
    litellm_params: { model: "openai/gpt-6.1-sol", custom_llm_provider: "openai" },
    model_info: { mode, input_cost_per_token: 2e-6, output_cost_per_token: 1e-5, max_input_tokens: 1_050_000 },
  };
}

function imageModel(overrides: Partial<LiteLLMImageModel> = {}): LiteLLMImageModel {
  return {
    ...toNativeImageModels("litellm", "https://proxy.example", [
      mapFromImageModelInfoGroup("nano-banana", [imageRow("nano-banana")])!,
    ])[0]!,
    ...overrides,
  };
}

describe("image route discovery", () => {
  it("recognizes only LiteLLM's image_generation mode", () => {
    expect(isImageGenerationMode("image_generation")).toBe(true);
    expect(isImageGenerationMode(" IMAGE_GENERATION ")).toBe(true);
    for (const mode of ["chat", "responses", "embedding", "image_edit", "", undefined, null, 1]) {
      expect(isImageGenerationMode(mode)).toBe(false);
    }
  });

  it("maps a unanimous image_generation group to an image model with per-million prices", () => {
    const model = mapFromImageModelInfoGroup("nano-banana", [imageRow("nano-banana")]);
    expect(model).toEqual({
      type: "image",
      id: "nano-banana",
      name: "nano-banana",
      api: LITELLM_IMAGES_API,
      input: ["text"],
      output: ["image"],
      // The image-token rate wins over the text output rate: images dominate the bill.
      cost: { input: 1.5, output: 30, cacheRead: 0, cacheWrite: 0 },
      litellmDiscoveryVersion: LITELLM_DISCOVERY_VERSION,
    });
  });

  it("takes the highest declared rate and reports 0 when any deployment omits one", () => {
    const priced = mapFromImageModelInfoGroup("img", [
      imageRow("img"),
      imageRow("img", { input_cost_per_token: 4e-6, output_cost_per_image_token: 4e-5 }),
    ]);
    expect(priced?.cost).toMatchObject({ input: 4, output: 40 });
    const partial = mapFromImageModelInfoGroup("img", [
      imageRow("img"),
      imageRow("img", {
        input_cost_per_token: undefined,
        output_cost_per_image_token: undefined,
        output_cost_per_token: undefined,
      }),
    ]);
    expect(partial?.cost).toMatchObject({ input: 0, output: 0 });
  });

  it("withholds mixed-mode, mode-less and wildcard groups", () => {
    expect(mapFromImageModelInfoGroup("mixed", [imageRow("mixed"), chatRow("mixed")])).toBeUndefined();
    expect(mapFromImageModelInfoGroup("bare", [{ model_name: "bare", model_info: {} }])).toBeUndefined();
    expect(mapFromImageModelInfoGroup("images/*", [imageRow("images/*")])).toBeUndefined();
    expect(mapFromImageModelInfoGroup("empty", [])).toBeUndefined();
  });

  it("publishes image routes from /model/info next to chat routes, never as chat", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) {
        return jsonResponse(200, {
          data: [chatRow("gpt-6.1-sol"), imageRow("nano-banana"), imageRow("mixed"), chatRow("mixed")],
        });
      }
      throw new Error(`unexpected URL: ${url}`);
    });
    const result = await discoverModels("https://proxy.example", "secret", { silent: true });
    expect(result.models.map((model) => model.id)).toEqual(["gpt-6.1-sol"]);
    expect(result.imageModels?.map((model) => model.id)).toEqual(["nano-banana"]);
  });

  it("leaves the discovery result unchanged when the proxy has no image routes", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      const url = String(input);
      if (url.endsWith("/model/info")) return jsonResponse(200, { data: [chatRow("gpt-6.1-sol")] });
      throw new Error(`unexpected URL: ${url}`);
    });
    const result = await discoverModels("https://proxy.example", "secret", { silent: true });
    expect(result).not.toHaveProperty("imageModels");
  });
});

describe("generateImages", () => {
  it("posts the text prompt to /images/generations and returns base64 image blocks", async () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(
        200,
        {
          data: [{ b64_json: JPEG_B64, revised_prompt: null, url: null }],
          usage: { input_tokens: 5, output_tokens: 1120, total_tokens: 1125 },
        },
        { "x-litellm-response-cost": "0.034935" },
      ),
    );
    const onResponse = vi.fn();
    const result = await generateImages(
      imageModel(),
      {
        input: [
          { type: "text", text: "a red apple" },
          { type: "text", text: "on a white table" },
        ],
      },
      { apiKey: "secret", headers: { "x-team": "ai", "x-drop": null }, onResponse },
    );

    expect(result.stopReason).toBe("stop");
    expect(result.output).toEqual([{ type: "image", data: JPEG_B64, mimeType: "image/jpeg" }]);
    const [url, init] = fetchSpy.mock.calls[0]!;
    expect(url).toBe("https://proxy.example/v1/images/generations");
    expect(JSON.parse(String(init?.body))).toEqual({
      model: "nano-banana",
      prompt: "a red apple\n\non a white table",
      n: 1,
    });
    expect(init?.headers).toMatchObject({ authorization: "Bearer secret", "x-team": "ai" });
    expect(init?.headers).not.toHaveProperty("x-drop");
    expect(onResponse).toHaveBeenCalledWith(expect.objectContaining({ status: 200 }), expect.anything());
    // LiteLLM's response-cost header is authoritative over the token-rate estimate.
    expect(result.usage).toMatchObject({ input: 5, output: 1120, totalTokens: 1125 });
    expect(result.usage?.cost.total).toBeCloseTo(0.034935, 9);
    expect(result.usage!.cost.input + result.usage!.cost.output).toBeCloseTo(0.034935, 9);
  });

  it("estimates cost from discovered rates when the proxy sends no cost header", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(200, { data: [{ b64_json: PNG_B64 }], usage: { input_tokens: 10, output_tokens: 1000 } }),
    );
    const result = await generateImages(imageModel(), { input: [{ type: "text", text: "cube" }] }, { apiKey: "k" });
    expect(result.output[0]).toMatchObject({ mimeType: "image/png" });
    // 10 x $1.5/M + 1000 x $30/M
    expect(result.usage?.cost.total).toBeCloseTo(10 * 1.5e-6 + 1000 * 3e-5, 12);
  });

  it("honours a declared output_format and a caller-supplied fetch", async () => {
    const customFetch = vi.fn(async () => jsonResponse(200, { data: [{ b64_json: "AAAA" }], output_format: "webp" }));
    const result = await generateImages(
      imageModel(),
      { input: [{ type: "text", text: "cube" }] },
      { apiKey: "k", fetch: customFetch as unknown as typeof fetch },
    );
    expect(customFetch).toHaveBeenCalledOnce();
    expect(result.output[0]).toMatchObject({ mimeType: "image/webp" });
  });

  it("lets onPayload replace the request body", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(200, { data: [{ b64_json: PNG_B64 }] }));
    await generateImages(
      imageModel(),
      { input: [{ type: "text", text: "cube" }] },
      { apiKey: "k", onPayload: (payload) => ({ ...(payload as object), size: "1024x1024" }) },
    );
    expect(JSON.parse(String(fetchSpy.mock.calls[0]![1]?.body))).toMatchObject({ size: "1024x1024" });
  });

  it.each([
    ["no API key", {}, [{ type: "text" as const, text: "x" }], /No API key/],
    ["an empty prompt", { apiKey: "k" }, [{ type: "text" as const, text: "  " }], /non-empty text prompt/],
    [
      "a reference image",
      { apiKey: "k" },
      [
        { type: "text" as const, text: "edit this" },
        { type: "image" as const, data: PNG_B64, mimeType: "image/png" },
      ],
      /text prompt only/,
    ],
  ])("returns an error result for %s without calling the proxy", async (_label, options, input, message) => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const result = await generateImages(imageModel(), { input }, options);
    expect(result.stopReason).toBe("error");
    expect(result.errorMessage).toMatch(message);
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("reports proxy errors with a bounded message and never rejects", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      jsonResponse(400, { error: { message: `LLM Provider NOT provided. ${"x".repeat(1000)}` } }),
    );
    const result = await generateImages(imageModel(), { input: [{ type: "text", text: "x" }] }, { apiKey: "k" });
    expect(result.stopReason).toBe("error");
    expect(result.errorMessage).toMatch(/^LiteLLM images request failed \(400\): LLM Provider NOT provided\./);
    expect(result.errorMessage!.length).toBeLessThan(400);
  });

  it("does not fetch backend image URLs", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(200, { data: [{ url: "https://cdn.example/image.png" }] }));
    const result = await generateImages(imageModel(), { input: [{ type: "text", text: "x" }] }, { apiKey: "k" });
    expect(result.stopReason).toBe("error");
    expect(result.errorMessage).toMatch(/image URLs instead of base64/);
    expect(fetchSpy).toHaveBeenCalledOnce();
  });

  it("marks a cancelled request as aborted", async () => {
    const controller = new AbortController();
    vi.spyOn(globalThis, "fetch").mockImplementation(async () => {
      controller.abort();
      throw new DOMException("aborted", "AbortError");
    });
    const result = await generateImages(
      imageModel(),
      { input: [{ type: "text", text: "x" }] },
      { apiKey: "k", signal: controller.signal },
    );
    expect(result.stopReason).toBe("aborted");
  });
});

describe("LiteLLM provider image models", () => {
  const credential: Credential = { type: "api_key", key: "secret" };
  const auth: ProviderAuth = {
    apiKey: { name: "API key", resolve: async () => ({ auth: { apiKey: "secret" } }) },
  };
  const discovered = (): DiscoveryResult => ({
    source: "model_info",
    models: [],
    imageModels: [mapFromImageModelInfoGroup("nano-banana", [imageRow("nano-banana")])!],
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

  const images = (models: readonly AnyModel[] | undefined) =>
    (models ?? []).filter((model): model is ImageModel<string> => model.type === "image");

  it("publishes discovered image models alongside chat models", async () => {
    const value = provider();
    await value.refreshModels?.(refreshContext(undefined, true));
    expect(value.getModels()).toEqual([]);
    expect(images(value.getAllModels?.())).toEqual([
      expect.objectContaining({ id: "nano-banana", provider: "litellm", baseUrl: "https://proxy.example/v1" }),
    ]);
  });

  it("restores current stored image models offline and drops stale-version ones", async () => {
    const discover = vi.fn(async () => discovered());
    const value = provider(undefined, discover);
    const current = imageModel({ id: "kept" });
    const stale = { ...imageModel({ id: "stale" }), litellmDiscoveryVersion: 1 } as unknown as AnyModel;
    const foreign = { ...imageModel({ id: "foreign" }), api: "openrouter-images" } as AnyModel;
    await value.refreshModels?.(refreshContext([current, stale, foreign], false));
    expect(images(value.getAllModels?.()).map((model) => model.id)).toEqual(["kept"]);
    expect(discover).not.toHaveBeenCalled();
  });

  it("hides cached image models whose root no longer matches the credential", () => {
    const stderr = vi.spyOn(process.stderr, "write").mockImplementation(() => true);
    const value = provider("https://other.example");
    const filtered = value.filterAllModels?.([imageModel()], credential) ?? [];
    expect(filtered).toEqual([]);
    expect(stderr).toHaveBeenCalledWith(expect.stringContaining("stale LiteLLM model root"));
  });

  it("rebases a matching cached image model onto the credential root", () => {
    const value = provider("https://proxy.example/v1/");
    expect(value.filterAllModels?.([imageModel()], credential)).toEqual([
      expect.objectContaining({ id: "nano-banana", baseUrl: "https://proxy.example/v1" }),
    ]);
  });

  it("generates through the credential's root and refuses a stale cached root", async () => {
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockResolvedValue(jsonResponse(200, { data: [{ b64_json: PNG_B64 }] }));
    const ok = await provider().generateImages!(
      imageModel(),
      { input: [{ type: "text", text: "x" }] },
      { apiKey: "k" },
    );
    expect(ok.stopReason).toBe("stop");
    expect(fetchSpy.mock.calls[0]![0]).toBe("https://proxy.example/v1/images/generations");

    fetchSpy.mockClear();
    const stale = await provider("https://other.example").generateImages!(
      imageModel(),
      { input: [{ type: "text", text: "x" }] },
      { apiKey: "k" },
    );
    expect(stale.stopReason).toBe("error");
    expect(stale.errorMessage).toMatch(/stale LiteLLM model root/);
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});
