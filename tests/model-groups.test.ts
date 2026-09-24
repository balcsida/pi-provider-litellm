import { afterEach, describe, expect, it, vi } from "vitest";
import {
  type CatalogResolution,
  type CatalogResolver,
  closeSerializerPolicy,
  conservativeCostTiers,
  meetVendorCompat,
  NO_TRANSMISSIBLE_LEVELS,
  reduceModelGroup,
  stableJson,
  toResponsesLevels,
} from "../src/model-groups.js";
import { intersectThinkingLevelMaps } from "../src/thinking-levels.js";
import type { ModelInfoEntry } from "../src/types.js";

type ModelCost = NonNullable<ReturnType<typeof reduceModelGroup>>["cost"];

const catalog = new Map<string, CatalogResolution>([
  [
    "openai/gpt-4o",
    {
      provider: "openai",
      reasoning: false,
      vision: true,
      contextWindow: 128_000,
      maxTokens: 16_384,
      cost: { input: 5, output: 15, cacheRead: 2.5, cacheWrite: 0 },
    },
  ],
  [
    "anthropic/claude-sonnet-4-6",
    {
      provider: "anthropic",
      reasoning: true,
      vision: true,
      contextWindow: 200_000,
      maxTokens: 64_000,
      cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
    },
  ],
  [
    "bedrock/anthropic.claude-sonnet-4-6",
    {
      provider: "amazon-bedrock",
      reasoning: true,
      vision: true,
      contextWindow: 200_000,
      maxTokens: 64_000,
      cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
    },
  ],
]);

const resolveCatalog = (entry: ModelInfoEntry) => {
  const backend = entry.litellm_params?.model ?? entry.model_info?.base_model;
  return backend ? catalog.get(backend) : undefined;
};

function row(overrides: Partial<ModelInfoEntry> = {}): ModelInfoEntry {
  const { model_info, litellm_params, ...entry } = overrides;
  return {
    model_name: "route",
    ...entry,
    model_info: {
      id: "deployment-a",
      mode: "chat",
      // Described by LiteLLM's model map without a carrier; an absent list is an operator opt-in.
      supported_openai_params: [],
      supports_reasoning: true,
      supports_vision: true,
      max_input_tokens: 200_000,
      max_output_tokens: 32_000,
      input_cost_per_token: 0.000003,
      output_cost_per_token: 0.000015,
      cache_read_input_token_cost: 0.0000003,
      cache_creation_input_token_cost: 0.00000375,
      ...model_info,
    },
    litellm_params: { model: "anthropic/claude-sonnet-4-6", ...litellm_params },
  };
}

// A deployment LiteLLM did not price, so the catalog supplies the schedule.
const CATALOG_PRICED = {
  input_cost_per_token: undefined,
  output_cost_per_token: undefined,
  cache_read_input_token_cost: undefined,
  cache_creation_input_token_cost: undefined,
};

function permutations<T>(values: readonly T[]): T[][] {
  if (values.length < 2) return [[...values]];
  return values.flatMap((value, index) =>
    permutations([...values.slice(0, index), ...values.slice(index + 1)]).map((rest) => [value, ...rest]),
  );
}

const NO_LEVELS = {
  off: null,
  minimal: null,
  low: null,
  medium: null,
  high: null,
  xhigh: null,
  max: null,
};

describe("toResponsesLevels", () => {
  it.each([
    {
      name: "an absent map",
      levels: undefined,
      expected: {
        off: "none",
        minimal: "minimal",
        low: "low",
        medium: "medium",
        high: "high",
        xhigh: null,
        max: null,
      },
    },
    {
      name: "a partial Chat map",
      levels: { low: "high" },
      expected: {
        off: "none",
        minimal: "minimal",
        low: "high",
        medium: "medium",
        high: "high",
        xhigh: null,
        max: null,
      },
    },
    {
      name: "explicit extended levels",
      levels: { off: null, xhigh: "xhigh", max: "max" },
      expected: {
        off: null,
        minimal: "minimal",
        low: "low",
        medium: "medium",
        high: "high",
        xhigh: "xhigh",
        max: "max",
      },
    },
    {
      name: "a Chat-only value with no Responses spelling",
      levels: { high: "off" },
      expected: {
        off: "none",
        minimal: "minimal",
        low: "low",
        medium: "medium",
        high: null,
        xhigh: null,
        max: null,
      },
    },
  ])("never widens Responses beyond Chat for $name", ({ levels, expected }) => {
    expect(toResponsesLevels(levels)).toEqual(expected);
  });
});

describe("closeSerializerPolicy", () => {
  it("retains only Responses compatibility when closing a Responses policy", () => {
    const policy = closeSerializerPolicy({
      api: "openai-responses",
      reasoning: true,
      vendorCompat: {
        supportsDeveloperRole: false,
        supportsStrictMode: true,
        sessionAffinityFormat: "openai",
        supportsLongCacheRetention: false,
        supportsOpenAIGrammarTools: true,
        supportsAdditionalTools: true,
        supportsToolSearch: true,
        supportsExplicitPromptCacheMode: true,
        thinkingFormat: "deepseek",
        supportsReasoningEffort: false,
      },
      denyLevels: true,
    });
    expect(policy.compat).toEqual({
      supportsDeveloperRole: false,
      supportsStrictMode: true,
      sessionAffinityFormat: "openai",
      supportsLongCacheRetention: false,
      supportsOpenAIGrammarTools: true,
      supportsAdditionalTools: true,
      supportsToolSearch: true,
      supportsExplicitPromptCacheMode: true,
    });
    expect(policy.thinkingLevelMap).toEqual(NO_LEVELS);
    expect(
      closeSerializerPolicy({
        api: "openai-responses",
        reasoning: false,
        vendorCompat: { supportsReasoningEffort: false },
        denyLevels: true,
      }).compat,
    ).toBeUndefined();
  });

  it("keeps Chat and Responses closed when vendor compatibility denies reasoning effort", () => {
    const input = {
      reasoning: true,
      vendorCompat: { supportsReasoningEffort: false } as const,
      catalogLevels: { off: "off", low: "low", high: "high" },
    };

    const chat = closeSerializerPolicy({ ...input, api: "openai-completions" });
    const responses = closeSerializerPolicy({ ...input, api: "openai-responses" });

    expect(chat.thinkingLevelMap).toEqual({
      off: null,
      minimal: null,
      low: null,
      medium: null,
      high: null,
      xhigh: null,
      max: null,
    });
    expect(responses.thinkingLevelMap).toEqual({
      off: null,
      minimal: null,
      low: null,
      medium: null,
      high: null,
      xhigh: null,
      max: null,
    });
  });

  it.each(["openai-completions", "openai-responses"] as const)(
    "makes denyLevels explicitly deny selectable levels for %s",
    (api) => {
      expect(
        closeSerializerPolicy({
          api,
          reasoning: true,
          vendorCompat: { supportsStore: false, supportsReasoningEffort: true },
          catalogLevels: { low: "low", high: "high" },
          acceptsResponsesReasoningControl: true,
          denyLevels: true,
        }),
      ).toEqual({
        reasoning: true,
        thinkingLevelMap: NO_LEVELS,
        compat: api === "openai-responses" ? undefined : { supportsStore: false, supportsReasoningEffort: false },
      });
    },
  );

  it("denies Responses levels until reasoning_effort acceptance is evidenced", () => {
    const input = {
      api: "openai-responses" as const,
      reasoning: true,
      vendorCompat: { supportsStore: false } as const,
      semanticLevels: { off: "off", high: "high", max: "max" },
    };

    expect(closeSerializerPolicy(input).thinkingLevelMap).toEqual(NO_LEVELS);
    expect(closeSerializerPolicy({ ...input, acceptsResponsesReasoningControl: true }).thinkingLevelMap).toEqual({
      off: "none",
      minimal: "minimal",
      low: "low",
      medium: "medium",
      high: "high",
      xhigh: null,
      max: "max",
    });
  });

  it("denies implicit Chat levels until a carrier is evidenced", () => {
    const input = {
      api: "openai-completions" as const,
      reasoning: true,
      vendorCompat: { supportsStore: false } as const,
    };

    expect(closeSerializerPolicy({ ...input, requireChatCarrier: true }).thinkingLevelMap).toEqual(NO_LEVELS);
    expect(
      closeSerializerPolicy({
        ...input,
        vendorCompat: { supportsStore: false, supportsReasoningEffort: true },
      }),
    ).toEqual({
      reasoning: true,
      compat: { supportsStore: false, supportsReasoningEffort: true },
    });
  });
});

describe("meetVendorCompat", () => {
  it("keeps Moonshot restrictions but withholds shape changes from an unidentified sibling", () => {
    expect(
      meetVendorCompat([
        {
          supportsStore: false,
          supportsDeveloperRole: false,
          supportsReasoningEffort: false,
          supportsStrictMode: false,
          maxTokensField: "max_tokens",
        },
        undefined,
      ]),
    ).toEqual({
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsStrictMode: false,
    });
  });

  it("retains the complete Moonshot block only when every deployment agrees", () => {
    const moonshot = {
      supportsStore: false,
      supportsDeveloperRole: false,
      supportsReasoningEffort: false,
      supportsStrictMode: false,
      maxTokensField: "max_tokens" as const,
    };

    expect(meetVendorCompat([moonshot, moonshot])).toEqual(moonshot);
  });
});

describe("reduceModelGroup", () => {
  it("is permutation invariant for heterogeneous deployment evidence", () => {
    const deployments = [
      row({ model_info: { id: "deployment-a", mode: "responses", max_input_tokens: 150_000 } }),
      row({
        model_info: { id: "deployment-b", mode: "chat", max_output_tokens: 16_000 },
        litellm_params: { model: "openai/gpt-4o" },
      }),
      row({
        model_info: { id: "deployment-c", mode: null },
        litellm_params: { model: "internal/unknown" },
      }),
      row({
        model_info: { id: undefined, mode: "chat", output_cost_per_token: 0.00002 },
        litellm_params: { model: "internal/unknown" },
      }),
    ];
    const expected = {
      id: "route",
      api: "openai-completions",
      reasoning: true,
      acceptsResponsesReasoningControl: false,
      vision: true,
      contextWindow: 150_000,
      maxTokens: 16_000,
      cost: { input: 3, output: 20, cacheRead: 0.3, cacheWrite: 3.75 },
      hasCompleteCost: true,
      hasCompleteMetadata: true,
      catalogAuthorityAmbiguous: true,
      deploymentFamilies: [undefined, undefined, undefined, undefined],
      normalizeThinkTags: false,
      suppressReasoningVisibility: false,
      acceptedOpenAIParams: [],
      reasoningPolicy: { reasoning: false },
    };

    for (const order of permutations(deployments)) {
      expect(reduceModelGroup(order, resolveCatalog)).toEqual(expected);
    }
  });

  it("deduplicates exact rows and reduces conflicting duplicate ids conservatively", () => {
    const repeated = row();
    const conflicting = row({ model_info: { id: "deployment-a", mode: "chat", max_input_tokens: 8_000 } });
    const anonymous = row({ model_info: { id: undefined, mode: "chat" } });

    expect(reduceModelGroup([repeated, repeated], resolveCatalog)).toEqual(
      reduceModelGroup([repeated], resolveCatalog),
    );
    // Conflicting variants of one deployment id both stay in the reduction.
    const expected = reduceModelGroup([repeated, conflicting], resolveCatalog);
    expect(expected).toMatchObject({ contextWindow: 8_000 });
    expect(reduceModelGroup([conflicting, repeated], resolveCatalog)).toEqual(expected);

    // Exact id-less repeats remain plural: equal content is not enough evidence
    // that two rows describe the same deployment.
    let calls = 0;
    reduceModelGroup([anonymous, anonymous], () => {
      calls++;
      return undefined;
    });
    expect(calls).toBe(2);
  });

  it("selects Responses only when every deployment explicitly reports it", () => {
    const responses = row({ model_info: { id: "responses", mode: "responses" } });
    const response = row({ model_info: { id: "response", mode: "response" } });
    const chat = row({ model_info: { id: "chat", mode: "chat" } });
    const unknown = row({ model_info: { id: "unknown", mode: null } });

    expect(reduceModelGroup([responses, response], resolveCatalog)?.api).toBe("openai-responses");
    expect(reduceModelGroup([responses, chat], resolveCatalog)?.api).toBe("openai-completions");
    expect(reduceModelGroup([responses, unknown], resolveCatalog)?.api).toBe("openai-completions");
  });

  it("requires every Responses deployment to accept reasoning_effort", () => {
    const accepted = (id: string, params: string[] | undefined) =>
      row({
        model_info: { id, mode: "responses", supported_openai_params: params },
        litellm_params: { model: `internal/${id}` },
      });

    for (const order of permutations([accepted("effort", ["reasoning_effort"]), accepted("thinking", ["thinking"])])) {
      expect(reduceModelGroup(order, resolveCatalog)).toMatchObject({
        api: "openai-responses",
        acceptsResponsesReasoningControl: false,
      });
    }
    expect(
      reduceModelGroup(
        [accepted("a", ["reasoning_effort", "thinking"]), accepted("b", ["reasoning_effort"])],
        resolveCatalog,
      ),
    ).toMatchObject({ api: "openai-responses", acceptsResponsesReasoningControl: true });
  });

  it.each([
    ["chat first", "chat", "embedding"],
    ["embedding first", "embedding", "chat"],
    ["Responses first", "responses", "embedding"],
    ["embedding before Responses", "embedding", "responses"],
  ])("rejects a mixed chat-style and unsupported group with $0", (_case, firstMode, secondMode) => {
    const deployment = (id: string, mode: string) =>
      row({
        model_info: { id, mode },
        litellm_params: { model: `internal/${id}` },
      });

    expect(
      reduceModelGroup([deployment("first", firstMode), deployment("second", secondMode)], resolveCatalog),
    ).toBeUndefined();
  });

  it.each(["chat", "response", "responses"])("retains a pure %s group", (mode) => {
    const deployments = ["first", "second"].map((id) =>
      row({
        model_info: { id, mode },
        litellm_params: { model: `internal/${id}` },
      }),
    );

    expect(reduceModelGroup(deployments, resolveCatalog)?.api).toBe(
      mode === "chat" ? "openai-completions" : "openai-responses",
    );
  });

  it("ignores limits that are not finite positive token counts", () => {
    const good = row({ model_info: { id: "good", mode: "chat", max_input_tokens: 64_000, max_output_tokens: 8_000 } });
    for (const invalid of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      const broken = row({
        model_info: { id: "broken", mode: "chat", max_input_tokens: invalid, max_output_tokens: invalid },
      });
      // The catalog resolves for both rows, so an unusable router limit falls back
      // to catalog evidence instead of clamping the group to zero.
      expect(reduceModelGroup([good, broken], resolveCatalog)).toMatchObject({
        contextWindow: 64_000,
        maxTokens: 8_000,
      });
    }

    const unknownBackend = row({
      model_info: { id: "broken", mode: "chat", max_input_tokens: 0, max_output_tokens: 0 },
      litellm_params: { model: "internal/unknown" },
    });
    expect(reduceModelGroup([unknownBackend], resolveCatalog)).toMatchObject({
      contextWindow: 128_000,
      maxTokens: 16_384,
    });
  });

  afterEach(() => vi.unstubAllEnvs());

  it.each([
    ["922000", 922_000],
    ["", 128_000],
    ["not-a-number", 128_000],
    ["0", 128_000],
    ["-1", 128_000],
    ["1.5", 128_000],
    ["922000junk", 128_000],
  ])("uses LITELLM_DEFAULT_CONTEXT_WINDOW=%j as the fallback window", (configured, expected) => {
    vi.stubEnv("LITELLM_DEFAULT_CONTEXT_WINDOW", configured);
    const noLimits = row({ model_info: { id: "only", mode: "chat", max_input_tokens: undefined } });

    // Only the missing limit is filled; a reported window still wins.
    expect(reduceModelGroup([noLimits], () => undefined)).toMatchObject({ contextWindow: expected });
    const reported = row({ model_info: { id: "only", mode: "chat", max_input_tokens: 8_000 } });
    expect(reduceModelGroup([reported], () => undefined)).toMatchObject({ contextWindow: 8_000 });
  });

  it.each([
    ["zero", 0],
    ["negative", -1],
    ["not a number", Number.NaN],
    ["infinite", Number.POSITIVE_INFINITY],
  ])("ignores a %s catalog limit and falls back to the conservative default", (_case, invalid) => {
    // The same validation must apply to catalog-supplied limits, not only to the
    // router-reported ones, or a bad catalog value would clamp the whole group.
    const noRouterLimits = row({
      model_info: { id: "only", mode: "chat", max_input_tokens: undefined, max_output_tokens: undefined },
    });
    const brokenCatalog: CatalogResolver = () => ({
      provider: "anthropic",
      reasoning: true,
      vision: true,
      contextWindow: invalid,
      maxTokens: invalid,
      cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
    });

    const result = reduceModelGroup([noRouterLimits], brokenCatalog);

    expect(result).toMatchObject({ contextWindow: 128_000, maxTokens: 16_384 });
    // Authority is not discarded wholesale; only the unusable limits are.
    expect(result).toMatchObject({ catalogProvider: "anthropic", reasoning: true });
  });

  it("treats an unreadable mode as unknown rather than as evidence of a non-chat deployment", () => {
    // An unreadable `mode` must not relax the group. Dropping the row the way a
    // genuinely non-chat deployment is dropped would discard its limits and let the
    // group report a larger context window than any deployment can serve.
    const roomy = row({ model_info: { id: "roomy", mode: "chat", max_input_tokens: 200_000 } });
    const cramped = { id: "cramped", max_input_tokens: 8_000 };
    const unreadable = row({ model_info: { ...cramped, mode: 7 as unknown as string } });
    const embedding = row({ model_info: { ...cramped, mode: "embedding" } });

    // Unreadable: still a deployment, so its tighter limit clamps the group.
    expect(reduceModelGroup([roomy, unreadable], resolveCatalog)).toMatchObject({
      contextWindow: 8_000,
      api: "openai-completions",
    });
    // Genuinely non-chat: rejects the entire mixed route.
    expect(reduceModelGroup([roomy, embedding], resolveCatalog)).toBeUndefined();

    // A lone unreadable row is surfaced conservatively rather than silently hidden.
    expect(reduceModelGroup([unreadable], resolveCatalog)).toMatchObject({
      contextWindow: 8_000,
      api: "openai-completions",
    });
  });

  it("does not read an unreadable capability flag as true", () => {
    // `"false"` and `"no"` are truthy, so coercion would advertise a capability no
    // deployment claimed. A group guarantee must never be relaxed by a bad wire type.
    const lying = row({
      model_info: {
        id: "lying",
        mode: "chat",
        supports_vision: "no" as unknown as boolean,
        supports_reasoning: "false" as unknown as boolean,
      },
      litellm_params: { model: "internal/unknown" },
    });

    expect(reduceModelGroup([lying], resolveCatalog)).toMatchObject({ vision: false, reasoning: false });
  });

  it("drops a group when every deployment is non-chat", () => {
    expect(
      reduceModelGroup(
        [
          row({ model_info: { id: "embed-a", mode: "embedding" } }),
          row({ model_info: { id: "embed-b", mode: "embedding" } }),
        ],
        resolveCatalog,
      ),
    ).toBeUndefined();
  });

  it.each([
    [[true, true], true],
    [[true, false], false],
    [[true, undefined], false],
  ] as const)("reduces capability guarantees %j to %s", (values, expected) => {
    const deployments = values.map((value, index) =>
      row({
        model_info: { id: `deployment-${index}`, mode: "chat", supports_vision: value },
        ...(value === undefined ? { litellm_params: { model: "internal/unknown" } } : {}),
      }),
    );
    expect(reduceModelGroup(deployments, resolveCatalog)?.vision).toBe(expected);
  });

  it("uses the smaller valid router and catalog limit", () => {
    const router = row({
      model_info: { id: "router", mode: "chat", max_input_tokens: 300_000, max_output_tokens: 100_000 },
    });
    const catalog: CatalogResolver = () => ({
      provider: "openai",
      contextWindow: 200_000,
      maxTokens: 64_000,
    });

    expect(reduceModelGroup([router], catalog)).toMatchObject({ contextWindow: 200_000, maxTokens: 64_000 });
  });

  it("resolves deployment limits before taking the safe group minimum", () => {
    const explicit = row({
      model_info: { id: "explicit", mode: "chat", max_input_tokens: 100_000, max_output_tokens: 8_000 },
    });
    const fromCatalog = row({
      model_info: { id: "catalog", mode: "chat", max_input_tokens: undefined, max_output_tokens: undefined },
    });
    const unknown = row({
      model_info: { id: "unknown", mode: "chat", max_input_tokens: undefined, max_output_tokens: undefined },
      litellm_params: { model: "internal/unknown" },
    });

    expect(reduceModelGroup([explicit, fromCatalog], resolveCatalog)).toMatchObject({
      contextWindow: 100_000,
      maxTokens: 8_000,
    });
    expect(reduceModelGroup([explicit, unknown], resolveCatalog)).toMatchObject({
      contextWindow: 100_000,
      maxTokens: 8_000,
    });
    expect(reduceModelGroup([unknown], resolveCatalog)).toMatchObject({
      contextWindow: 128_000,
      maxTokens: 16_384,
    });
  });

  it("uses the maximum complete display price and marks incomplete price evidence unknown", () => {
    const cheaper = row({
      model_info: {
        id: "cheap",
        mode: "chat",
        input_cost_per_token: 0,
        output_cost_per_token: 0.00001,
        cache_read_input_token_cost: 0.0000002,
        cache_creation_input_token_cost: 0.000003,
      },
    });
    const pricier = row({
      model_info: {
        id: "pricey",
        mode: "chat",
        input_cost_per_token: 0.000004,
        output_cost_per_token: 0.00002,
        cache_read_input_token_cost: 0.0000004,
        cache_creation_input_token_cost: 0.000004,
      },
    });
    const complete = reduceModelGroup([cheaper, pricier], resolveCatalog);
    expect(complete).toMatchObject({
      hasCompleteCost: true,
      cost: { input: 4, output: 20, cacheWrite: 4 },
    });
    expect(complete?.cost.cacheRead).toBeCloseTo(0.4);

    const incomplete = row({
      model_info: {
        id: "incomplete",
        mode: "chat",
        input_cost_per_token: 0.000004,
        output_cost_per_token: undefined,
      },
      litellm_params: { model: "internal/unknown" },
    });
    expect(reduceModelGroup([cheaper, incomplete], resolveCatalog)).toMatchObject({
      hasCompleteCost: false,
      cost: { input: 4, output: 0, cacheRead: 0.3, cacheWrite: 3.75 },
    });
  });

  it("rejects negative explicit prices as unresolved", () => {
    const negative = row({
      model_info: {
        id: "negative",
        mode: "chat",
        input_cost_per_token: -0.000001,
        output_cost_per_token: 0.000002,
      },
      litellm_params: { model: "internal/unknown" },
    });

    expect(reduceModelGroup([negative], resolveCatalog)).toMatchObject({
      hasCompleteCost: false,
      cost: { input: 0, output: 2 },
    });
  });

  it("retains proven display prices and zeroes only unresolved fields without catalog authority", () => {
    // Characterization of the existing per-field cost block. No backend resolves,
    // so cache pricing is genuinely unknown rather than free: input and output
    // survive at their proven values, the unresolved cache fields read zero, and
    // `hasCompleteCost` stays false so the model can be marked incomplete.
    const priced = (id: string, input: number, output: number) =>
      row({
        model_info: {
          id,
          mode: "chat",
          input_cost_per_token: input,
          output_cost_per_token: output,
          cache_read_input_token_cost: undefined,
          cache_creation_input_token_cost: undefined,
        },
        litellm_params: { model: "internal/unknown" },
      });

    const singleton = reduceModelGroup([priced("only", 0.000003, 0.000015)], resolveCatalog);
    expect(singleton).toMatchObject({
      hasCompleteCost: false,
      cost: { input: 3, output: 15, cacheRead: 0, cacheWrite: 0 },
    });
    expect(singleton).not.toHaveProperty("catalogProvider");
    expect(singleton?.cost.tiers).toBeUndefined();

    // Proven fields still reduce to the maximum across a group.
    expect(
      reduceModelGroup([priced("a", 0.000003, 0.000015), priced("b", 0.000004, 0.000015)], resolveCatalog),
    ).toMatchObject({
      hasCompleteCost: false,
      cost: { input: 4, output: 15, cacheRead: 0, cacheWrite: 0 },
    });
  });

  it("applies public effort levels and LiteLLM overrides", () => {
    const result = reduceModelGroup(
      [
        row({
          model_info: {
            id: "reasoner",
            mode: "chat",
            supported_openai_params: ["reasoning_effort"],
            supports_minimal_reasoning_effort: false,
            supports_xhigh_reasoning_effort: true,
          },
        }),
      ],
      () => ({
        provider: "openai",
        reasoning: true,
        effortLevels: ["minimal", "low", "medium", "high"],
      }),
    );

    expect(result?.thinkingLevelMap).toEqual({
      off: null,
      minimal: null,
      low: "low",
      medium: "medium",
      high: "high",
      xhigh: "xhigh",
      max: null,
    });
  });

  it("leaves standard levels absent without a public opinion", () => {
    const result = reduceModelGroup(
      [
        row({
          model_info: { supports_reasoning: true, supported_openai_params: ["reasoning_effort"] },
          litellm_params: { model: "internal/reasoner" },
        }),
      ],
      () => ({ reasoning: true }),
    );

    expect(result?.thinkingLevelMap).toEqual({ xhigh: null, max: null });
  });

  it("keeps off denied for an always-thinking Kimi generation when the catalog supplies effort levels", () => {
    const result = reduceModelGroup(
      [
        row({
          model_info: { supports_reasoning: true, supported_openai_params: ["reasoning_effort"] },
          litellm_params: { model: "moonshot/kimi-k2.7-code" },
        }),
      ],
      () => ({
        provider: "moonshotai",
        reasoning: true,
        semanticModel: "kimi-k2.7-code",
        effortLevels: ["low", "medium", "high"],
      }),
    );

    expect(result?.reasoningPolicy.thinkingLevelMap).toEqual({
      off: null,
      minimal: null,
      low: "low",
      medium: "medium",
      high: "high",
      xhigh: null,
      max: null,
    });
  });

  it("keeps catalog tiers off a deployment whose prices the operator configured", () => {
    const tiers = [{ inputTokensAbove: 200_000, input: 6, output: 22.5, cacheRead: 0.6, cacheWrite: 7.5 }];
    const catalog: CatalogResolver = () => ({
      provider: "openai",
      reasoning: true,
      cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75, tiers },
    });
    const priced = row({
      model_info: {
        id: "custom",
        mode: "chat",
        input_cost_per_token: 0.000001,
        output_cost_per_token: 0.000002,
        cache_read_input_token_cost: 0.0000001,
        cache_creation_input_token_cost: 0.0000002,
      },
    });

    const result = reduceModelGroup([priced], catalog);

    expect(result?.cost.tiers).toBeUndefined();
    expect(result?.cost).toMatchObject({ input: 1, output: 2 });
  });

  it("substitutes an operator price into every catalog tier for that field only", () => {
    const tiers = [{ inputTokensAbove: 200_000, input: 6, output: 22.5, cacheRead: 0.6, cacheWrite: 7.5 }];
    const catalog: CatalogResolver = () => ({
      provider: "openai",
      reasoning: true,
      cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75, tiers },
    });
    const partial = row({
      model_info: { id: "partial", mode: "chat", ...CATALOG_PRICED, input_cost_per_token: 0.000001 },
    });

    expect(reduceModelGroup([partial], catalog)?.cost.tiers).toEqual([
      { inputTokensAbove: 200_000, input: 1, output: 22.5, cacheRead: 0.6, cacheWrite: 7.5 },
    ]);
  });

  it("adopts tiered pricing when identical tiers are declared in any property order", () => {
    const tiers = [{ inputTokensAbove: 200_000, input: 6, output: 22.5, cacheRead: 0.6, cacheWrite: 7.5 }];
    const reordered = [{ cacheWrite: 7.5, output: 22.5, input: 6, cacheRead: 0.6, inputTokensAbove: 200_000 }];
    const withTiers =
      (value: typeof tiers): CatalogResolver =>
      () => ({
        provider: "anthropic",
        reasoning: true,
        vision: true,
        contextWindow: 200_000,
        maxTokens: 64_000,
        cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75, tiers: value },
      });
    const rows = [
      row({ model_info: { id: "a", mode: "chat", ...CATALOG_PRICED } }),
      row({ model_info: { id: "b", mode: "chat", ...CATALOG_PRICED } }),
    ];

    expect(reduceModelGroup(rows, withTiers(tiers))?.cost.tiers).toEqual(tiers);
    // Property order is not evidence of disagreement.
    let call = 0;
    const alternating: CatalogResolver = (entry) => withTiers(call++ === 0 ? tiers : reordered)(entry);
    expect(reduceModelGroup(rows, alternating)?.cost.tiers).toEqual(tiers);
  });

  it("builds the union-threshold envelope for deployments with different ladders", () => {
    const rows = [
      row({ model_info: { id: "a", mode: "chat", ...CATALOG_PRICED } }),
      row({ model_info: { id: "b", mode: "chat", ...CATALOG_PRICED } }),
    ];
    let call = 0;
    const differing: CatalogResolver = () => ({
      provider: "anthropic",
      reasoning: true,
      vision: true,
      contextWindow: 200_000,
      maxTokens: 64_000,
      cost: {
        input: 3,
        output: 15,
        cacheRead: 0.3,
        cacheWrite: 3.75,
        tiers: [
          call++ === 0
            ? { inputTokensAbove: 200_000, input: 6, output: 18, cacheRead: 0.6, cacheWrite: 4 }
            : { inputTokensAbove: 400_000, input: 5, output: 22.5, cacheRead: 0.5, cacheWrite: 7.5 },
        ],
      },
    });

    expect(reduceModelGroup(rows, differing)?.cost.tiers).toEqual([
      { inputTokensAbove: 200_000, input: 6, output: 18, cacheRead: 0.6, cacheWrite: 4 },
      { inputTokensAbove: 400_000, input: 6, output: 22.5, cacheRead: 0.6, cacheWrite: 7.5 },
    ]);
  });

  it("omits tiered pricing and thinking maps entirely when no catalog declares them", () => {
    const result = reduceModelGroup([row()], resolveCatalog);

    expect(result?.cost.tiers).toBeUndefined();
    expect(result?.thinkingLevelMap).toBeUndefined();
  });

  it("disables catalog authority for conflicting provider identities", () => {
    const result = reduceModelGroup(
      [
        row({ model_info: { id: "anthropic", mode: "chat" } }),
        row({
          model_info: { id: "openai", mode: "chat" },
          litellm_params: { model: "openai/gpt-4o" },
        }),
      ],
      resolveCatalog,
    );

    expect(result).not.toHaveProperty("catalogProvider");

    expect(result?.thinkingLevelMap).toBeUndefined();
  });

  // LiteLLM omits the list (or returns null) for an off-map deployment; only that
  // absence is an operator opt-in. Present but malformed data is no carrier evidence.
  it.each([
    [undefined, true],
    [null, true],
    ["reasoning_effort", false],
    [{ reasoning_effort: true }, false],
  ])("treats supported_openai_params=%j with supports_reasoning as carrier=%s", (params, expected) => {
    const result = reduceModelGroup(
      [
        row({
          model_info: { supported_openai_params: params as never, supports_reasoning: true },
          litellm_params: { model: "internal/reasoner" },
        }),
      ],
      () => undefined,
    );

    expect(result?.acceptsResponsesReasoningControl).toBe(expected);
  });

  it("intersects accepted parameters across deployments", () => {
    const result = reduceModelGroup(
      [
        row({
          model_info: { id: "a", mode: "chat", supported_openai_params: ["temperature", "reasoning_effort"] },
          litellm_params: { model: "internal/a" },
        }),
        row({
          model_info: { id: "b", mode: "chat", supported_openai_params: ["reasoning_effort", "thinking"] },
          litellm_params: { model: "internal/b", allowed_openai_params: ["reasoning_effort"] },
        }),
      ],
      resolveCatalog,
    );

    expect(result?.acceptedOpenAIParams).toEqual(["reasoning_effort"]);
  });

  it.each([
    {
      name: "Kimi K2.6 with binary thinking",
      semanticModel: "kimi-k2.5-k2.6" as const,
      params: ["thinking"],
      expected: {
        reasoning: true,
        thinkingLevelMap: {
          off: "off",
          minimal: null,
          low: null,
          medium: null,
          high: "high",
          xhigh: null,
          max: null,
        },
        compat: { thinkingFormat: "deepseek", supportsReasoningEffort: false },
      },
    },
    {
      // K2.7 Code cannot be switched off, so `off` stays denied while `high`
      // rides the accepted `thinking` param.
      name: "Kimi K2.7 Code with accepted thinking",
      semanticModel: "kimi-k2.7-code" as const,
      params: ["thinking"],
      expected: {
        reasoning: true,
        thinkingLevelMap: {
          off: null,
          minimal: null,
          low: null,
          medium: null,
          high: "high",
          xhigh: null,
          max: null,
        },
        compat: {
          supportsReasoningEffort: false,
          requiresReasoningContentOnAssistantMessages: true,
          thinkingFormat: "deepseek",
        },
      },
    },
    {
      name: "Kimi K2.7 Code without accepted controls",
      semanticModel: "kimi-k2.7-code" as const,
      params: undefined,
      expected: {
        reasoning: true,
        thinkingLevelMap: {
          off: null,
          minimal: null,
          low: null,
          medium: null,
          high: null,
          xhigh: null,
          max: null,
        },
        compat: { supportsReasoningEffort: false, requiresReasoningContentOnAssistantMessages: true },
      },
    },
    {
      name: "Kimi K2.6 without accepted controls",
      semanticModel: "kimi-k2.5-k2.6" as const,
      params: undefined,
      expected: {
        reasoning: true,
        thinkingLevelMap: {
          off: null,
          minimal: null,
          low: null,
          medium: null,
          high: null,
          xhigh: null,
          max: null,
        },
        compat: { supportsReasoningEffort: false },
      },
    },
    {
      name: "DeepSeek V4 through a thinking-only route",
      semanticModel: "deepseek-v4" as const,
      params: ["thinking"],
      expected: {
        reasoning: true,
        thinkingLevelMap: {
          off: "off",
          minimal: null,
          low: null,
          medium: null,
          high: "high",
          xhigh: null,
          max: null,
        },
        compat: {
          thinkingFormat: "deepseek",
          supportsReasoningEffort: false,
          requiresReasoningContentOnAssistantMessages: true,
        },
      },
    },
  ])("derives $name policy from semantic and accepted-control evidence", ({ semanticModel, params, expected }) => {
    const result = reduceModelGroup(
      [
        row({
          model_info: { supported_openai_params: params },
          litellm_params: { model: "internal/model" },
        }),
      ],
      () => ({ semanticModel }),
    );

    expect(result?.reasoningPolicy).toEqual(expected);
  });

  it.each([
    {
      // Without `thinking` there is no carrier for K2.7 Code's binary control, and
      // an accepted `reasoning_effort` with no public level opinion cannot reopen
      // the levels the generation denies.
      name: "Kimi K2.7 Code with effort",
      semanticModel: "kimi-k2.7-code" as const,
      params: ["reasoning_effort"],
      expected: {
        reasoning: true,
        thinkingLevelMap: NO_TRANSMISSIBLE_LEVELS,
        compat: {
          supportsReasoningEffort: false,
          requiresReasoningContentOnAssistantMessages: true,
        },
      },
    },
    {
      name: "Kimi K3 with effort",
      semanticModel: "kimi-k3" as const,
      params: ["reasoning_effort"],
      expected: {
        reasoning: true,
        thinkingLevelMap: { off: null, xhigh: null, max: null },
        compat: {
          thinkingFormat: "openai",
          supportsReasoningEffort: true,
          requiresReasoningContentOnAssistantMessages: true,
        },
      },
    },
    {
      name: "DeepSeek V4 with native controls",
      semanticModel: "deepseek-v4" as const,
      params: ["thinking", "reasoning_effort"],
      expected: {
        reasoning: true,
        thinkingLevelMap: { off: "off", xhigh: null, max: null },
        compat: {
          thinkingFormat: "deepseek",
          supportsReasoningEffort: true,
          requiresReasoningContentOnAssistantMessages: true,
        },
      },
    },
    {
      name: "DeepSeek V4 through an effort-only route",
      semanticModel: "deepseek-v4" as const,
      params: ["reasoning_effort"],
      expected: {
        reasoning: true,
        thinkingLevelMap: { off: null, xhigh: null, max: null },
        compat: {
          thinkingFormat: "openai",
          supportsReasoningEffort: true,
          requiresReasoningContentOnAssistantMessages: true,
        },
      },
    },
  ])("derives $name policy from literal level-map expectations", ({ semanticModel, params, expected }) => {
    const result = reduceModelGroup(
      [
        row({
          model_info: { supported_openai_params: params },
          litellm_params: { model: "internal/model" },
        }),
      ],
      () => ({ semanticModel }),
    );

    expect(result?.reasoningPolicy).toEqual(expected);
  });

  // LiteLLM reports both carriers for K2.5/K2.6 on OpenRouter and Databricks.
  // The generation still has one on/off switch, so the deployment map's absent standard levels
  // must not become five selectable efforts.
  it("keeps the binary Kimi map when a host also accepts reasoning_effort", () => {
    const result = reduceModelGroup(
      [
        row({
          model_info: { supports_reasoning: true, supported_openai_params: ["thinking", "reasoning_effort"] },
          litellm_params: { model: "openrouter/moonshotai/kimi-k2.6" },
        }),
      ],
      () => ({ provider: "moonshotai", reasoning: true, semanticModel: "kimi-k2.5-k2.6" }),
    );

    expect(result?.reasoningPolicy.thinkingLevelMap).toEqual({
      off: "off",
      minimal: null,
      low: null,
      medium: null,
      high: "high",
      xhigh: null,
      max: null,
    });
  });

  it("lets an explicit LiteLLM denial close a level the binary Kimi map keeps", () => {
    const result = reduceModelGroup(
      [
        row({
          model_info: {
            supports_reasoning: true,
            supports_none_reasoning_effort: false,
            supported_openai_params: ["thinking", "reasoning_effort"],
          },
          litellm_params: { model: "openrouter/moonshotai/kimi-k2.6" },
        }),
      ],
      () => ({ provider: "moonshotai", reasoning: true, semanticModel: "kimi-k2.5-k2.6" }),
    );

    expect(result?.reasoningPolicy.thinkingLevelMap).toMatchObject({ off: null, high: "high", low: null });
  });

  it("lets a public effort list govern Kimi levels while preserving semantic off", () => {
    const result = reduceModelGroup(
      [
        row({
          model_info: { supports_reasoning: true, supported_openai_params: ["thinking", "reasoning_effort"] },
          litellm_params: { model: "openrouter/moonshotai/kimi-k2.6" },
        }),
      ],
      () => ({
        provider: "moonshotai",
        reasoning: true,
        semanticModel: "kimi-k2.5-k2.6",
        effortLevels: ["low", "high"],
      }),
    );

    expect(result?.reasoningPolicy.thinkingLevelMap).toEqual({
      off: "off",
      minimal: null,
      low: "low",
      medium: null,
      high: "high",
      xhigh: null,
      max: null,
    });
  });

  it("keeps the binary Kimi map when a public effort list has no recognized values", () => {
    const result = reduceModelGroup(
      [
        row({
          model_info: { supports_reasoning: true, supported_openai_params: ["thinking", "reasoning_effort"] },
          litellm_params: { model: "openrouter/moonshotai/kimi-k2.6" },
        }),
      ],
      () => ({
        provider: "moonshotai",
        reasoning: true,
        semanticModel: "kimi-k2.5-k2.6",
        effortLevels: ["adaptive"],
      }),
    );

    expect(result?.reasoningPolicy.thinkingLevelMap).toEqual({
      off: "off",
      minimal: null,
      low: null,
      medium: null,
      high: "high",
      xhigh: null,
      max: null,
    });
  });

  it("preserves catalog map denials when its public effort list is empty", () => {
    const result = reduceModelGroup(
      [
        row({
          model_info: { supports_reasoning: true, supported_openai_params: ["reasoning_effort"] },
          litellm_params: { model: "openai/private-reasoning" },
        }),
      ],
      () => ({
        provider: "openai",
        reasoning: true,
        effortLevels: [],
        thinkingLevelMap: { off: null },
      }),
    );

    expect(result?.thinkingLevelMap).toEqual({ off: null, xhigh: null, max: null });
  });

  it("lets catalog map denials override a models.dev effort list", () => {
    const result = reduceModelGroup(
      [
        row({
          model_info: { supports_reasoning: true, supported_openai_params: ["reasoning_effort"] },
          litellm_params: { model: "openai/private-reasoning" },
        }),
      ],
      () => ({
        provider: "openai",
        reasoning: true,
        effortLevels: ["low", "high"],
        thinkingLevelMap: { off: null, low: null },
      }),
    );

    expect(result?.thinkingLevelMap).toEqual({
      off: null,
      minimal: null,
      low: null,
      medium: null,
      high: "high",
      xhigh: null,
      max: null,
    });
  });

  it.each([
    ["lets semantic off through without an explicit denial", undefined, "off"],
    ["lets an explicit LiteLLM denial override semantic off", false, null],
  ] as const)("%s for DeepSeek V4", (_name, supportsNone, expectedOff) => {
    const result = reduceModelGroup(
      [
        row({
          model_info: {
            supports_reasoning: true,
            ...(supportsNone === undefined ? {} : { supports_none_reasoning_effort: supportsNone }),
            supported_openai_params: ["thinking", "reasoning_effort"],
          },
          litellm_params: { model: "deepseek/deepseek-v4-pro" },
        }),
      ],
      () => ({
        provider: "deepseek",
        reasoning: true,
        semanticModel: "deepseek-v4",
        effortLevels: ["low", "high"],
      }),
    );

    expect(result?.reasoningPolicy.thinkingLevelMap).toEqual({
      off: expectedOff,
      minimal: null,
      low: "low",
      medium: null,
      high: "high",
      xhigh: null,
      max: null,
    });
  });

  it("keeps a standard level a wildcard parent leaves at Pi's default but closes extended ones", () => {
    expect(intersectThinkingLevelMaps([undefined, { high: "high" }])).toEqual({ high: "high" });
    expect(intersectThinkingLevelMaps([undefined, { max: "max" }])).toEqual({ max: null });
  });

  it("closes a level when wildcard parents disagree on its wire value", () => {
    expect(intersectThinkingLevelMaps([{ high: "high" }, { high: "max" }])).toEqual({ high: null });
  });

  it.each([
    {
      name: "Kimi K3",
      semanticModel: "kimi-k3" as const,
    },
    {
      name: "DeepSeek V4",
      semanticModel: "deepseek-v4" as const,
    },
  ])("preserves $name capability and replay without accepted-control evidence", ({ semanticModel }) => {
    const result = reduceModelGroup(
      [row({ model_info: { supports_reasoning: true }, litellm_params: { model: `internal/${semanticModel}` } })],
      () => ({ semanticModel }),
    );

    expect(result?.reasoningPolicy).toEqual({
      reasoning: true,
      thinkingLevelMap: { off: null, minimal: null, low: null, medium: null, high: null, xhigh: null, max: null },
      compat: {
        requiresReasoningContentOnAssistantMessages: true,
        supportsReasoningEffort: false,
      },
    });
  });

  it("lets any explicit reasoning denial override accepted-control promotion", () => {
    const result = reduceModelGroup(
      [
        row({
          model_info: { id: "denied", supports_reasoning: false, supported_openai_params: ["thinking"] },
          litellm_params: { model: "moonshot/kimi-k2.6" },
        }),
        row({
          model_info: { id: "accepted", supports_reasoning: true, supported_openai_params: ["thinking"] },
          litellm_params: { model: "moonshot/kimi-k2.6" },
        }),
      ],
      () => ({ semanticModel: "kimi-k2.5-k2.6", reasoning: true }),
    );

    expect(result?.reasoning).toBe(false);
    expect(result?.reasoningPolicy).toEqual({ reasoning: false, compat: { supportsReasoningEffort: false } });
  });

  it.each(["moonshot/kimi-k2-thinking", "moonshot/kimi_k2_thinking", "moonshot/kimi.k2.thinking"])(
    "preserves always-thinking Kimi display behavior for %s",
    (model) => {
      const result = reduceModelGroup(
        [
          row({
            model_name: "misleading-public-route",
            litellm_params: { model },
            model_info: { supports_reasoning: true },
          }),
        ],
        () => undefined,
      );

      expect(result).toMatchObject({ normalizeThinkTags: false, suppressReasoningVisibility: false });
    },
  );

  it.each([
    {
      name: "Azure-hosted Kimi",
      litellm_params: { model: "azure/FW-Kimi-K3" },
      model_info: {
        base_model: "fireworks/accounts/fireworks/models/kimi-k3",
        litellm_provider: "azure",
      },
    },
    {
      name: "Bedrock-hosted Kimi",
      litellm_params: { model: "bedrock/moonshotai.kimi-k2.5" },
      model_info: { base_model: "moonshotai.kimi-k2.5", litellm_provider: "bedrock_converse" },
    },
  ])("normalizes $name responses without sending Moonshot visibility parameters", ({ litellm_params, model_info }) => {
    const result = reduceModelGroup(
      [
        row({
          model_name: "hosted-kimi",
          litellm_params,
          model_info: { ...model_info, mode: "chat", supports_reasoning: true },
        }),
      ],
      () => ({ semanticFamily: "kimi" }),
    );

    expect(result).toMatchObject({ normalizeThinkTags: true, suppressReasoningVisibility: false });
  });

  it("suppresses reasoning visibility on Moonshot transport", () => {
    const result = reduceModelGroup(
      [
        row({
          model_name: "k3-prod",
          litellm_params: { model: "moonshot/kimi-k2.5" },
          model_info: { mode: "chat", supports_reasoning: true },
        }),
      ],
      () => ({ semanticFamily: "kimi" }),
    );

    expect(result).toMatchObject({ normalizeThinkTags: true, suppressReasoningVisibility: true });
  });

  it("keeps conflicting routing signals from enabling Moonshot visibility parameters", () => {
    const result = reduceModelGroup(
      [
        row({
          model_name: "conflicting-route",
          litellm_params: { model: "azure_ai/FW-Kimi-K3", custom_llm_provider: "moonshot" },
          model_info: { mode: "chat", supports_reasoning: true },
        }),
      ],
      () => ({ semanticFamily: "kimi" }),
    );

    expect(result).toMatchObject({ normalizeThinkTags: true, suppressReasoningVisibility: false });
  });

  it("does not suppress visibility when any Kimi deployment is always-thinking", () => {
    const result = reduceModelGroup(
      [
        row({
          model_name: "mixed-kimi-route",
          litellm_params: { model: "moonshot/kimi-k2.6" },
          model_info: { id: "normal", supports_reasoning: true },
        }),
        row({
          model_name: "mixed-kimi-route",
          litellm_params: { model: "moonshot/kimi-k2-thinking" },
          model_info: { id: "thinking", supports_reasoning: true },
        }),
      ],
      () => ({ semanticFamily: "kimi" }),
    );

    expect(result).toMatchObject({ normalizeThinkTags: false, suppressReasoningVisibility: false });
  });

  it.each([
    { name: "Claude", model: "anthropic/claude-sonnet-4-6" },
    { name: "OpenAI", model: "openai/gpt-4o" },
  ])("does not normalize think tags for a mixed Kimi/$name route", ({ model }) => {
    const result = reduceModelGroup(
      [
        row({
          model_name: "mixed-family-route",
          litellm_params: { model: "moonshot/kimi-k2.6" },
          model_info: { id: "kimi", supports_reasoning: true },
        }),
        row({
          model_name: "mixed-family-route",
          litellm_params: { model },
          model_info: { id: "other", supports_reasoning: true },
        }),
      ],
      resolveCatalog,
    );

    expect(result).toMatchObject({ normalizeThinkTags: false, suppressReasoningVisibility: false });
  });

  it("lets explicit unanimous reasoning denial override the K2.7 Code contract", () => {
    const result = reduceModelGroup(
      [
        row({
          model_info: { supports_reasoning: false, supported_openai_params: ["thinking"] },
          litellm_params: { model: "moonshot/kimi-k2.7-code" },
        }),
      ],
      () => ({ semanticModel: "kimi-k2.7-code", reasoning: true }),
    );

    expect(result?.reasoning).toBe(false);
    expect(result?.reasoningPolicy).toEqual({
      reasoning: false,
      compat: { supportsReasoningEffort: false, requiresReasoningContentOnAssistantMessages: true },
    });
  });

  it("fails closed for mixed semantic generations and accepted controls", () => {
    const deployments = [
      row({
        model_info: { id: "k2", supported_openai_params: ["thinking"] },
        litellm_params: { model: "moonshot/kimi-k2.6" },
      }),
      row({
        model_info: { id: "k3", supported_openai_params: ["reasoning_effort"] },
        litellm_params: { model: "moonshot/kimi-k3" },
      }),
    ];
    const result = reduceModelGroup(deployments, (entry) => ({
      semanticModel: entry.model_info?.id === "k2" ? "kimi-k2.5-k2.6" : "kimi-k3",
    }));

    expect(result?.acceptedOpenAIParams).toEqual([]);
    expect(result?.reasoningPolicy).toEqual({ reasoning: false });
  });
});

describe("upstream reduction regressions", () => {
  it("takes the per-field maximum from duplicate thresholds in one ladder", () => {
    const cost = {
      input: 1,
      output: 2,
      cacheRead: 3,
      cacheWrite: 4,
      tiers: [
        { inputTokensAbove: 100, input: 10, output: 2, cacheRead: 30, cacheWrite: 4 },
        { inputTokensAbove: 100, input: 1, output: 20, cacheRead: 3, cacheWrite: 40 },
      ],
    };

    expect(conservativeCostTiers([cost])).toEqual([
      { inputTokensAbove: 100, input: 10, output: 20, cacheRead: 30, cacheWrite: 40 },
    ]);
  });

  it("floors every matched tier field at its deployment base rate", () => {
    const cost = {
      input: 10,
      output: 20,
      cacheRead: 3,
      cacheWrite: 4,
      tiers: [{ inputTokensAbove: 100, input: 1, output: 2, cacheRead: 0.3, cacheWrite: 0.4 }],
    };

    expect(conservativeCostTiers([cost])).toEqual([
      { inputTokensAbove: 100, input: 10, output: 20, cacheRead: 3, cacheWrite: 4 },
    ]);
  });

  it.each([
    ["negative", -1],
    ["not a number", Number.NaN],
    ["infinite", Number.POSITIVE_INFINITY],
  ])("ignores a %s tier threshold", (_case, invalidThreshold) => {
    const cost = {
      input: 1,
      output: 2,
      cacheRead: 3,
      cacheWrite: 4,
      tiers: [
        { inputTokensAbove: invalidThreshold, input: 100, output: 200, cacheRead: 300, cacheWrite: 400 },
        { inputTokensAbove: 100, input: 10, output: 20, cacheRead: 30, cacheWrite: 40 },
      ],
    };

    expect(conservativeCostTiers([cost])).toEqual([
      { inputTokensAbove: 100, input: 10, output: 20, cacheRead: 30, cacheWrite: 40 },
    ]);
  });

  it("filters rows without a readable route name before reducing limits and capabilities", () => {
    const roomy = row({ model_info: { id: "roomy", mode: "chat", max_input_tokens: 200_000 } });
    const badName = row({
      model_name: 42 as unknown as string,
      model_info: { id: "bad-name", mode: "chat", supports_reasoning: false, max_input_tokens: 8_000 },
    });

    expect(reduceModelGroup([roomy, badName], resolveCatalog)).toMatchObject({
      contextWindow: 200_000,
      reasoning: true,
    });
  });

  it("tracks metadata completeness independently from complete router pricing", () => {
    const explicit = row({
      litellm_params: { model: "internal/unknown" },
      model_info: {
        id: "explicit",
        mode: "chat",
        supports_reasoning: false,
        supports_vision: false,
        max_input_tokens: 32_000,
        max_output_tokens: 4_000,
        input_cost_per_token: 0.000003,
        output_cost_per_token: 0.000015,
        cache_read_input_token_cost: 0.0000003,
        cache_creation_input_token_cost: 0.00000375,
      },
    });
    const defaults: ModelInfoEntry = {
      model_name: "route",
      litellm_params: { model: "internal/unknown" },
      model_info: {
        id: "defaults",
        mode: "chat",
        input_cost_per_token: 0.000003,
        output_cost_per_token: 0.000015,
        cache_read_input_token_cost: 0.0000003,
        cache_creation_input_token_cost: 0.00000375,
      },
    };

    expect(reduceModelGroup([explicit], resolveCatalog)).toMatchObject({
      hasCompleteCost: true,
      hasCompleteMetadata: true,
    });
    expect(reduceModelGroup([defaults], resolveCatalog)).toMatchObject({
      hasCompleteCost: true,
      hasCompleteMetadata: false,
    });
  });

  it("resolves catalog metadata once per routable deployment", () => {
    const seen: ModelInfoEntry[] = [];
    const record: CatalogResolver = (entry) => {
      seen.push(entry);
      return undefined;
    };
    const chat = row({ model_info: { id: "chat", mode: "chat" } });
    const embedding = row({ model_info: { id: "embed", mode: "embedding" } });
    const other = row({ model_info: { id: "other", mode: "chat" } });
    const conflicting = row({ model_info: { id: "chat", mode: "chat", max_input_tokens: 8_000 } });

    reduceModelGroup([chat], record);
    expect(seen).toEqual([chat]);

    seen.length = 0;
    reduceModelGroup([chat, chat], record);
    expect(seen).toEqual([chat]);

    // An incompatible sibling withholds the route before catalog resolution.
    seen.length = 0;
    expect(reduceModelGroup([chat, embedding], record)).toBeUndefined();
    expect(seen).toEqual([]);

    seen.length = 0;
    reduceModelGroup([chat, other], record);
    expect(seen).toEqual([chat, other]);

    seen.length = 0;
    reduceModelGroup([chat, conflicting], record);
    expect(seen).toHaveLength(2);
  });

  it("withholds groups containing an explicitly incompatible deployment mode", () => {
    const responses = row({ model_info: { id: "responses", mode: "responses" } });
    const chat = row({ model_info: { id: "chat", mode: "chat" } });
    const unsupported = row({
      model_info: { id: "embed", mode: "embedding", max_input_tokens: 1, max_output_tokens: 1 },
      litellm_params: { model: "internal/embedding" },
    });

    expect(reduceModelGroup([responses, unsupported], resolveCatalog)).toBeUndefined();
    expect(reduceModelGroup([chat, unsupported], resolveCatalog)).toBeUndefined();
  });

  it("takes the smaller valid limit when explicit and public catalog values disagree", () => {
    const largerExplicit = row({
      model_info: { id: "larger-explicit", mode: "chat", max_input_tokens: 300_000, max_output_tokens: 80_000 },
    });
    const smallerExplicit = row({
      model_info: { id: "smaller-explicit", mode: "chat", max_input_tokens: 100_000, max_output_tokens: 8_000 },
    });

    expect(reduceModelGroup([largerExplicit], resolveCatalog)).toMatchObject({
      contextWindow: 200_000,
      maxTokens: 64_000,
    });
    expect(reduceModelGroup([smallerExplicit], resolveCatalog)).toMatchObject({
      contextWindow: 100_000,
      maxTokens: 8_000,
    });
  });

  it("keeps explicit prices and fills modalities from public catalog metadata", () => {
    const explicit = row({
      model_info: {
        id: "explicit",
        mode: "chat",
        supports_vision: undefined,
        input_cost_per_token: 0.000009,
        output_cost_per_token: 0.000019,
      },
    });

    expect(reduceModelGroup([explicit], resolveCatalog)).toMatchObject({
      vision: true,
      cost: { input: 9, output: 19, cacheRead: 0.3, cacheWrite: 3.75 },
    });
  });

  it("withholds catalog authority for different concrete models from one provider", () => {
    const result = reduceModelGroup(
      [
        row({
          model_info: {
            id: "a",
            mode: "chat",
            supports_reasoning: undefined,
            supports_vision: undefined,
            max_input_tokens: undefined,
            max_output_tokens: undefined,
            input_cost_per_token: undefined,
            output_cost_per_token: undefined,
            cache_read_input_token_cost: undefined,
            cache_creation_input_token_cost: undefined,
          },
          litellm_params: { model: "route-a" },
        }),
        row({
          model_info: {
            id: "b",
            mode: "chat",
            supports_reasoning: undefined,
            supports_vision: undefined,
            max_input_tokens: undefined,
            max_output_tokens: undefined,
            input_cost_per_token: undefined,
            output_cost_per_token: undefined,
            cache_read_input_token_cost: undefined,
            cache_creation_input_token_cost: undefined,
          },
          litellm_params: { model: "route-b" },
        }),
      ],
      (entry) => ({
        provider: "anthropic",
        catalogModelId: entry.litellm_params?.model === "route-a" ? "claude-sonnet-4-6" : "claude-opus-4-5",
        reasoning: true,
        vision: true,
        contextWindow: entry.litellm_params?.model === "route-a" ? 200_000 : 180_000,
        maxTokens: 64_000,
        cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
      }),
    );

    expect(result).toMatchObject({
      catalogAuthorityAmbiguous: true,
      contextWindow: 128_000,
      maxTokens: 16_384,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    });
    expect(result).not.toHaveProperty("catalogProvider");
    expect(result).not.toHaveProperty("thinkingLevelMap");
  });

  it("keeps standard levels a catalog thinking map omits at Pi's defaults", () => {
    const thinkingLevelMap = { low: "low", high: "high" } as const;
    const result = reduceModelGroup([row({ model_info: { supported_openai_params: ["reasoning_effort"] } })], () => ({
      provider: "xai",
      reasoning: true,
      thinkingLevelMap,
      vision: false,
      contextWindow: 128_000,
      maxTokens: 16_384,
      cost: { input: 1, output: 2, cacheRead: 0, cacheWrite: 0 },
    }));

    expect(result?.thinkingLevelMap).toEqual({ ...thinkingLevelMap, xhigh: null, max: null });
  });

  it("intersects differing catalog thinking maps per level regardless of deployment order", () => {
    const entries = [
      row({ model_info: { supported_openai_params: ["reasoning_effort"], id: "a", mode: "chat" } }),
      row({ model_info: { supported_openai_params: ["reasoning_effort"], id: "b", mode: "chat" } }),
    ];
    const maps = {
      a: { off: "none", low: "low", high: "high", max: null },
      b: { off: "none", low: null, high: "high", xhigh: "xhigh" },
    } as const;

    for (const order of permutations(entries)) {
      const result = reduceModelGroup(order, (entry) => ({
        provider: "openai",
        reasoning: true,
        thinkingLevelMap: maps[entry.model_info?.id as keyof typeof maps],
        vision: false,
        contextWindow: 128_000,
        maxTokens: 16_384,
        cost: { input: 1, output: 2, cacheRead: 0, cacheWrite: 0 },
      }));

      expect(result?.thinkingLevelMap).toEqual({ off: "none", low: null, high: "high", xhigh: null, max: null });
    }
  });

  it("keeps catalog levels a deployment without a thinking map leaves at Pi's defaults", () => {
    const entries = [
      row({ model_info: { supported_openai_params: ["reasoning_effort"], id: "mapped", mode: "chat" } }),
      row({ model_info: { supported_openai_params: ["reasoning_effort"], id: "absent", mode: "chat" } }),
    ];

    for (const order of permutations(entries)) {
      const result = reduceModelGroup(order, (entry) => ({
        provider: "openai",
        reasoning: true,
        ...(entry.model_info?.id === "mapped" ? { thinkingLevelMap: { low: "low", high: "high" } as const } : {}),
        vision: false,
        contextWindow: 128_000,
        maxTokens: 16_384,
        cost: { input: 1, output: 2, cacheRead: 0, cacheWrite: 0 },
      }));

      expect(result?.thinkingLevelMap).toEqual({ low: "low", high: "high", xhigh: null, max: null });
    }
  });

  it("suppresses reasoning controls when the router explicitly disables reasoning", () => {
    const result = reduceModelGroup(
      [
        row({
          model_info: {
            id: "non-reasoner",
            mode: "chat",
            supports_reasoning: false,
            supports_low_reasoning_effort: true,
          },
        }),
      ],
      () => ({
        provider: "openai",
        reasoning: true,
        thinkingLevelMap: { low: "low", high: "high" },
        vision: true,
        contextWindow: 128_000,
        maxTokens: 16_384,
        cost: { input: 1, output: 2, cacheRead: 0, cacheWrite: 0 },
      }),
    );

    expect(result).toMatchObject({ reasoning: false });
    expect(result).not.toHaveProperty("thinkingLevelMap");
  });

  it("preserves explicit router reasoning efforts for a singleton", () => {
    const result = reduceModelGroup(
      [
        row({
          model_info: {
            supported_openai_params: ["reasoning_effort"],
            id: "reasoner",
            mode: "chat",
            supports_none_reasoning_effort: true,
            supports_minimal_reasoning_effort: false,
            supports_xhigh_reasoning_effort: true,
          },
          litellm_params: { model: "internal/reasoner" },
        }),
      ],
      resolveCatalog,
    );

    expect(result?.thinkingLevelMap).toEqual({ off: "none", minimal: null, xhigh: "xhigh", max: null });
  });

  it("preserves defaults for missing effort flags and honors explicit opt-outs", () => {
    const supportsLow = (id: string, low: unknown) =>
      row({
        model_info: {
          supported_openai_params: ["reasoning_effort"],
          id,
          mode: "chat",
          supports_low_reasoning_effort: low as boolean,
          supports_xhigh_reasoning_effort: true,
        },
        litellm_params: { model: `internal/${id}` },
      });

    expect(
      reduceModelGroup([supportsLow("a", true), supportsLow("b", true)], resolveCatalog)?.thinkingLevelMap,
    ).toEqual({ low: "low", xhigh: "xhigh", max: null });
    for (const missing of [undefined, null]) {
      expect(
        reduceModelGroup([supportsLow("a", true), supportsLow("b", missing)], resolveCatalog)?.thinkingLevelMap,
      ).toEqual({ xhigh: "xhigh", max: null });
    }
    expect(
      reduceModelGroup([supportsLow("a", true), supportsLow("b", false)], resolveCatalog)?.thinkingLevelMap,
    ).toEqual({ low: null, xhigh: "xhigh", max: null });
  });

  it("overlays conservative router reasoning evidence on catalog metadata", () => {
    const catalogThinkingLevelMap = { off: "none", low: "low", high: "high", max: "max" } as const;
    const result = reduceModelGroup(
      [
        row({
          model_info: {
            supported_openai_params: ["reasoning_effort"],
            id: "a",
            mode: "chat",
            supports_low_reasoning_effort: true,
          },
        }),
        row({
          model_info: {
            supported_openai_params: ["reasoning_effort"],
            id: "b",
            mode: "chat",
            supports_low_reasoning_effort: false,
          },
        }),
      ],
      () => ({
        provider: "openai",
        reasoning: true,
        thinkingLevelMap: catalogThinkingLevelMap,
        vision: true,
        contextWindow: 128_000,
        maxTokens: 16_384,
        cost: { input: 1, output: 2, cacheRead: 0, cacheWrite: 0 },
      }),
    );

    expect(result?.thinkingLevelMap).toEqual({ ...catalogThinkingLevelMap, xhigh: null, max: null, low: null });
  });

  it("merges different rates at identical tier thresholds conservatively regardless of order", () => {
    const rows = ["a", "b"].map((id) =>
      row({
        model_info: {
          id,
          mode: "chat",
          input_cost_per_token: undefined,
          output_cost_per_token: undefined,
          cache_read_input_token_cost: undefined,
          cache_creation_input_token_cost: undefined,
        },
      }),
    );
    const costs = [
      {
        input: 3,
        output: 15,
        cacheRead: 0.3,
        cacheWrite: 3.75,
        tiers: [{ inputTokensAbove: 200_000, input: 6, output: 20, cacheRead: 0.6, cacheWrite: 7.5 }],
      },
      {
        input: 4,
        output: 12,
        cacheRead: 0.4,
        cacheWrite: 4,
        tiers: [{ inputTokensAbove: 200_000, input: 5, output: 22.5, cacheRead: 0.5, cacheWrite: 8 }],
      },
    ];

    for (const order of permutations(costs)) {
      let call = 0;
      const result = reduceModelGroup(rows, () => ({
        provider: "anthropic",
        reasoning: true,
        vision: true,
        contextWindow: 200_000,
        maxTokens: 64_000,
        cost: order[call++],
      }));
      expect(result).toMatchObject({
        cost: {
          input: 4,
          output: 15,
          cacheRead: 0.4,
          cacheWrite: 4,
          tiers: [{ inputTokensAbove: 200_000, input: 6, output: 22.5, cacheRead: 0.6, cacheWrite: 8 }],
        },
        hasCompleteCost: true,
        hasCompleteMetadata: true,
      });
    }
  });

  it("constructs a safe piecewise envelope for different tier thresholds regardless of order", () => {
    const rateAt = (cost: ModelCost, inputTokens: number): ModelCost =>
      [...(cost.tiers ?? [])]
        .sort((left, right) => right.inputTokensAbove - left.inputTokensAbove)
        .find((tier) => tier.inputTokensAbove < inputTokens) ?? cost;
    const sampledCost = (cost: ModelCost, inputTokens: number, outputTokens: number): number => {
      const rate = rateAt(cost, inputTokens);
      return rate.input * inputTokens + rate.output * outputTokens;
    };
    const rows = ["a", "b", "c"].map((id) =>
      row({
        model_info: {
          id,
          mode: "chat",
          input_cost_per_token: undefined,
          output_cost_per_token: undefined,
          cache_read_input_token_cost: undefined,
          cache_creation_input_token_cost: undefined,
        },
      }),
    );
    const costs = [
      {
        input: 3,
        output: 15,
        cacheRead: 0.3,
        cacheWrite: 3.75,
        tiers: [
          { inputTokensAbove: 200_000, input: 6, output: 18, cacheRead: 0.6, cacheWrite: 7.5 },
          { inputTokensAbove: 500_000, input: 12, output: 36, cacheRead: 1.2, cacheWrite: 15 },
        ],
      },
      {
        input: 4,
        output: 16,
        cacheRead: 0.4,
        cacheWrite: 4,
        tiers: [{ inputTokensAbove: 400_000, input: 8, output: 32, cacheRead: 0.8, cacheWrite: 8 }],
      },
      { input: 5, output: 14, cacheRead: 0.5, cacheWrite: 5 },
    ];
    const expected = {
      input: 5,
      output: 16,
      cacheRead: 0.5,
      cacheWrite: 5,
      tiers: [
        { inputTokensAbove: 200_000, input: 6, output: 18, cacheRead: 0.6, cacheWrite: 7.5 },
        { inputTokensAbove: 400_000, input: 8, output: 32, cacheRead: 0.8, cacheWrite: 8 },
        { inputTokensAbove: 500_000, input: 12, output: 36, cacheRead: 1.2, cacheWrite: 15 },
      ],
    };

    for (const order of permutations(costs)) {
      let call = 0;
      const result = reduceModelGroup(rows, () => ({
        provider: "anthropic",
        reasoning: true,
        vision: true,
        contextWindow: 200_000,
        maxTokens: 64_000,
        cost: order[call++],
      }));
      expect(result).toBeDefined();
      if (!result) throw new Error("expected a reduced model group");
      expect(result.cost).toEqual(expected);
      expect(result).toMatchObject({ hasCompleteCost: true, hasCompleteMetadata: true });
      for (const inputTokens of [0, 200_000, 200_001, 400_000, 400_001, 500_000, 500_001]) {
        const envelopeCost = sampledCost(result.cost, inputTokens, 1_000);
        for (const source of costs) {
          expect(envelopeCost).toBeGreaterThanOrEqual(sampledCost(source, inputTokens, 1_000));
        }
      }
    }
  });

  it("drops the catalog tier ladder when every base-price field is explicit", () => {
    const result = reduceModelGroup(
      [
        row({
          model_info: {
            id: "explicit-prices",
            mode: "chat",
            input_cost_per_token: 0.000004,
            output_cost_per_token: 0.00002,
            cache_read_input_token_cost: 0.0000004,
            cache_creation_input_token_cost: 0.000005,
          },
        }),
      ],
      () => ({
        provider: "anthropic",
        reasoning: true,
        vision: true,
        contextWindow: 200_000,
        maxTokens: 64_000,
        cost: {
          input: 3,
          output: 15,
          cacheRead: 0.3,
          cacheWrite: 3.75,
          tiers: [{ inputTokensAbove: 200_000, input: 6, output: 30, cacheRead: 0.6, cacheWrite: 7.5 }],
        },
      }),
    );

    expect(result).toMatchObject({
      hasCompleteCost: true,
      cost: { input: 4, output: 20, cacheWrite: 5 },
    });
    expect(result?.cost.cacheRead).toBeCloseTo(0.4);
    expect(result?.cost.tiers).toBeUndefined();
  });

  it("retains catalog tiers for fields without an explicit router base price", () => {
    const result = reduceModelGroup(
      [
        row({
          model_info: {
            id: "partial-explicit-prices",
            mode: "chat",
            input_cost_per_token: 0.000004,
            output_cost_per_token: undefined,
            cache_read_input_token_cost: undefined,
            cache_creation_input_token_cost: undefined,
          },
        }),
      ],
      () => ({
        provider: "anthropic",
        reasoning: true,
        vision: true,
        contextWindow: 200_000,
        maxTokens: 64_000,
        cost: {
          input: 3,
          output: 15,
          cacheRead: 0.3,
          cacheWrite: 3.75,
          tiers: [{ inputTokensAbove: 200_000, input: 6, output: 30, cacheRead: 0.6, cacheWrite: 7.5 }],
        },
      }),
    );

    expect(result).toMatchObject({
      hasCompleteCost: true,
      cost: {
        input: 4,
        output: 15,
        cacheRead: 0.3,
        cacheWrite: 3.75,
        tiers: [{ inputTokensAbove: 200_000, input: 4, output: 30, cacheRead: 0.6, cacheWrite: 7.5 }],
      },
    });
  });

  it("does not attach catalog tiers when base price evidence is incomplete", () => {
    const rows = [
      row({
        model_info: {
          id: "a",
          mode: "chat",
          input_cost_per_token: undefined,
          output_cost_per_token: undefined,
          cache_read_input_token_cost: undefined,
          cache_creation_input_token_cost: undefined,
        },
      }),
      row({
        model_info: {
          id: "b",
          mode: "chat",
          input_cost_per_token: undefined,
          output_cost_per_token: undefined,
          cache_read_input_token_cost: undefined,
          cache_creation_input_token_cost: undefined,
        },
      }),
    ];
    let call = 0;
    const result = reduceModelGroup(rows, () => {
      const cost =
        call++ === 0
          ? {
              input: 3,
              output: 15,
              cacheRead: 0.3,
              cacheWrite: 3.75,
              tiers: [{ inputTokensAbove: 200_000, input: 6, output: 30, cacheRead: 0.6, cacheWrite: 7.5 }],
            }
          : undefined;
      return {
        provider: "anthropic",
        reasoning: true,
        vision: true,
        contextWindow: 200_000,
        maxTokens: 64_000,
        cost,
      };
    });

    expect(result).toMatchObject({
      hasCompleteCost: false,
      hasCompleteMetadata: false,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    });
    expect(result?.cost.tiers).toBeUndefined();
  });

  it("flags ambiguous catalog authority only when resolved identities disagree", () => {
    const anthropic = row({ model_info: { id: "anthropic", mode: "chat" } });
    const openai = row({
      model_info: { id: "openai", mode: "chat" },
      litellm_params: { model: "openai/gpt-4o" },
    });
    const unresolved = row({
      model_info: { id: "unknown", mode: "chat" },
      litellm_params: { model: "internal/unknown" },
    });

    // Unanimous identity, and wholly unknown identity, are not ambiguity.
    expect(reduceModelGroup([anthropic], resolveCatalog)).not.toHaveProperty("catalogAuthorityAmbiguous");
    expect(reduceModelGroup([anthropic, anthropic], resolveCatalog)).not.toHaveProperty("catalogAuthorityAmbiguous");
    expect(reduceModelGroup([unresolved], resolveCatalog)).not.toHaveProperty("catalogAuthorityAmbiguous");
    expect(reduceModelGroup([unresolved, unresolved], resolveCatalog)).not.toHaveProperty("catalogAuthorityAmbiguous");

    // Conflicting identities, and partial evidence, both withhold authority.
    expect(reduceModelGroup([anthropic, openai], resolveCatalog)?.catalogAuthorityAmbiguous).toBe(true);
    expect(reduceModelGroup([anthropic, unresolved], resolveCatalog)?.catalogAuthorityAmbiguous).toBe(true);
  });
});

describe("stableJson", () => {
  it("preserves array order in canonical identity", () => {
    const first = [{ threshold: 1 }, { threshold: 2 }];
    const reversed = [...first].reverse();

    expect(stableJson(first)).not.toBe(stableJson(reversed));
    expect(stableJson([{ b: 2, a: 1 }])).toBe(stableJson([{ a: 1, b: 2 }]));
  });
});

describe("native Messages route selection", () => {
  it("closes native reasoning without inheriting OpenAI controls", () => {
    expect(
      closeSerializerPolicy({
        api: "anthropic-messages",
        reasoning: true,
        vendorCompat: { forceAdaptiveThinking: true, supportsReasoningEffort: true },
        semanticCompat: { thinkingFormat: "openai" },
        semanticLevels: { off: "none", max: "xhigh" },
        catalogLevels: { off: null, max: "max" },
      }),
    ).toEqual({
      reasoning: true,
      compat: { forceAdaptiveThinking: true },
      thinkingLevelMap: { off: null, max: "max" },
    });
  });

  const claude = (compat: { forceAdaptiveThinking?: boolean } = {}) => ({
    provider: "amazon-bedrock",
    semanticFamily: "claude" as const,
    messagesCompat: compat,
  });

  it("selects Messages for a homogeneous strongly evidenced Claude group", () => {
    const result = reduceModelGroup(
      [
        { model_name: "claude-route", model_info: { id: "a", mode: "chat" } },
        { model_name: "claude-route", model_info: { id: "b", mode: "chat" } },
      ],
      () => claude({ forceAdaptiveThinking: true }),
    );

    expect(result).toMatchObject({
      api: "anthropic-messages",
      catalogProvider: "amazon-bedrock",
      semanticFamily: "claude",
      messagesCompat: { forceAdaptiveThinking: true },
    });
  });

  it.each([
    { name: "all affirmative", evidence: [true, true], expected: true },
    { name: "conflicting evidence", evidence: [true, false], expected: undefined },
    { name: "denial plus unknown", evidence: [false, undefined], expected: undefined },
    { name: "partial evidence", evidence: [true, undefined], expected: undefined },
    { name: "no evidence", evidence: [undefined, undefined], expected: undefined },
  ])("reduces strict-tool evidence independently for $name deployments", ({ evidence, expected }) => {
    const remaining = [...evidence];
    const result = reduceModelGroup(
      [
        { model_name: "claude-route", model_info: { id: "a", mode: "chat" } },
        { model_name: "claude-route", model_info: { id: "b", mode: "chat" } },
      ],
      () => ({ ...claude({ forceAdaptiveThinking: true }), messagesStrictTools: remaining.shift() }),
    );

    expect(result?.api).toBe("anthropic-messages");
    expect(result?.messagesCompat).toEqual({ forceAdaptiveThinking: true });
    expect(result?.messagesStrictTools).toBe(expected);
  });

  it("respects explicit Messages endpoint capability", () => {
    const supported = reduceModelGroup(
      [
        {
          model_name: "claude-route",
          model_info: { id: "a", mode: "chat", supported_endpoints: ["/v1/chat/completions", "/v1/messages"] },
        },
      ],
      () => claude({}),
    );
    const excluded = reduceModelGroup(
      [
        {
          model_name: "claude-route",
          model_info: { id: "a", mode: "chat", supported_endpoints: ["/v1/chat/completions"] },
        },
      ],
      () => claude({}),
    );
    const partiallyMalformedIncludingMessages = reduceModelGroup(
      [
        {
          model_name: "claude-route",
          model_info: { id: "a", mode: "chat", supported_endpoints: ["/v1/messages", 42] as never },
        },
      ],
      () => claude({}),
    );
    const partiallyMalformedOmittingMessages = reduceModelGroup(
      [
        {
          model_name: "claude-route",
          model_info: { id: "a", mode: "chat", supported_endpoints: ["/v1/chat/completions", 42] as never },
        },
      ],
      () => claude({}),
    );

    expect(supported?.api).toBe("anthropic-messages");
    expect(excluded?.api).toBe("openai-completions");
    expect(partiallyMalformedIncludingMessages?.api).toBe("anthropic-messages");
    expect(partiallyMalformedOmittingMessages?.api).toBe("openai-completions");
  });

  it.each([null, "/v1/chat/completions", "/v1/messages", 42, { endpoint: "/v1/messages" }])(
    "withholds Messages for malformed endpoint metadata %j",
    (endpoints) => {
      const result = reduceModelGroup(
        [{ model_name: "claude-route", model_info: { mode: "chat", supported_endpoints: endpoints as never } }],
        () => claude({}),
      );
      expect(result?.api).toBe("openai-completions");
    },
  );

  it.each([
    ["mixed family", [claude({}), { provider: "openai", semanticFamily: "openai" as const }]],
    ["unknown sibling", [claude({}), undefined]],
    ["different Messages compatibility", [claude({ forceAdaptiveThinking: true }), claude({})]],
  ])("keeps %s groups on Chat Completions", (_name, evidence) => {
    const result = reduceModelGroup(
      [
        { model_name: "mixed-route", model_info: { id: "a", mode: "chat" } },
        { model_name: "mixed-route", model_info: { id: "b", mode: "chat" } },
      ],
      () => evidence.shift(),
    );

    expect(result?.api).toBe("openai-completions");
  });

  it("keeps explicit Responses mode authoritative for Claude", () => {
    const result = reduceModelGroup([{ model_name: "claude-route", model_info: { id: "a", mode: "responses" } }], () =>
      claude({}),
    );

    expect(result?.api).toBe("openai-responses");
  });

  it("does not inject router OpenAI effort values into a Messages model", () => {
    const result = reduceModelGroup(
      [
        {
          model_name: "claude-route",
          model_info: {
            id: "a",
            mode: "chat",
            supports_minimal_reasoning_effort: true,
            supports_xhigh_reasoning_effort: true,
          },
        },
      ],
      () => ({
        ...claude({ forceAdaptiveThinking: true }),
        messagesThinkingLevelMap: { max: "max" },
      }),
    );

    expect(result).toMatchObject({ api: "anthropic-messages", thinkingLevelMap: { max: "max" } });
    expect(result?.thinkingLevelMap).not.toHaveProperty("minimal");
    expect(result?.thinkingLevelMap).not.toHaveProperty("xhigh");
  });

  it("lets router evidence disable but not rename a catalogued Messages effort", () => {
    const result = reduceModelGroup(
      [
        {
          model_name: "claude-route",
          model_info: { id: "a", mode: "chat", supports_xhigh_reasoning_effort: false },
        },
      ],
      () => ({
        ...claude({ forceAdaptiveThinking: true }),
        messagesThinkingLevelMap: { xhigh: "xhigh", max: "max" },
      }),
    );

    expect(result?.thinkingLevelMap).toEqual({ xhigh: null, max: "max" });
  });
});
