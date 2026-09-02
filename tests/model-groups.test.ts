import { describe, expect, it } from "vitest";
import {
  type CatalogResolution,
  type CatalogResolver,
  closeSerializerPolicy,
  meetVendorCompat,
  reduceModelGroup,
  toResponsesLevels,
} from "../src/model-groups.js";
import type { ModelInfoEntry } from "../src/types.js";

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
        max: null,
      },
    },
  ])("never widens Responses beyond Chat for $name", ({ levels, expected }) => {
    expect(toResponsesLevels(levels)).toEqual(expected);
  });
});

describe("closeSerializerPolicy", () => {
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
    "makes denyLevels explicitly disable reasoning effort for %s",
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
        compat: { supportsStore: false, supportsReasoningEffort: false },
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
      max: null,
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
    const seen: boolean[] = [];
    reduceModelGroup([anonymous, anonymous], (_entry, singleton) => {
      seen.push(singleton);
      return undefined;
    });
    expect(seen).toEqual([false, false]);
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

  it("uses catalog thinking maps for unambiguous identities", () => {
    const thinkingLevelMap = { low: "low", high: "high" } as const;
    const result = reduceModelGroup([row()], () => ({
      provider: "xai",
      reasoning: true,
      thinkingLevelMap,
      vision: false,
      contextWindow: 128_000,
      maxTokens: 16_384,
      cost: { input: 1, output: 2, cacheRead: 0, cacheWrite: 0 },
    }));

    expect(result?.thinkingLevelMap).toEqual(thinkingLevelMap);
  });

  it("preserves explicit router reasoning efforts for a singleton", () => {
    const result = reduceModelGroup(
      [
        row({
          model_info: {
            id: "reasoner",
            mode: "chat",
            supports_none_reasoning_effort: true,
            supports_minimal_reasoning_effort: false,
            supports_high_reasoning_effort: true,
          },
          litellm_params: { model: "internal/reasoner" },
        }),
      ],
      resolveCatalog,
    );

    expect(result?.thinkingLevelMap).toEqual({ off: "none", minimal: null, high: "high" });
  });

  it("advertises a router reasoning effort only when every deployment explicitly supports it", () => {
    const supportsLow = (id: string, low: boolean | undefined) =>
      row({
        model_info: {
          id,
          mode: "chat",
          supports_low_reasoning_effort: low,
          supports_high_reasoning_effort: true,
        },
        litellm_params: { model: `internal/${id}` },
      });

    expect(
      reduceModelGroup([supportsLow("a", true), supportsLow("b", true)], resolveCatalog)?.thinkingLevelMap,
    ).toEqual({ low: "low", high: "high" });
    expect(
      reduceModelGroup([supportsLow("a", true), supportsLow("b", undefined)], resolveCatalog)?.thinkingLevelMap,
    ).toEqual({ low: null, high: "high" });
    expect(
      reduceModelGroup([supportsLow("a", true), supportsLow("b", false)], resolveCatalog)?.thinkingLevelMap,
    ).toEqual({ low: null, high: "high" });
  });

  it("overlays conservative router reasoning evidence on catalog metadata", () => {
    const catalogThinkingLevelMap = { off: "none", low: "low", high: "high", max: "max" } as const;
    const result = reduceModelGroup(
      [
        row({
          model_info: { id: "a", mode: "chat", supports_low_reasoning_effort: true },
        }),
        row({
          model_info: { id: "b", mode: "chat", supports_low_reasoning_effort: false },
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

    expect(result?.thinkingLevelMap).toEqual({ ...catalogThinkingLevelMap, low: null });
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
    const rows = [row({ model_info: { id: "a", mode: "chat" } }), row({ model_info: { id: "b", mode: "chat" } })];

    expect(reduceModelGroup(rows, withTiers(tiers))?.cost.tiers).toEqual(tiers);
    // Property order is not evidence of disagreement.
    let call = 0;
    const alternating: CatalogResolver = (entry, singleton) =>
      withTiers(call++ === 0 ? tiers : reordered)(entry, singleton);
    expect(reduceModelGroup(rows, alternating)?.cost.tiers).toEqual(tiers);
  });

  it("treats a reordered tier ladder as disagreement rather than as equal", () => {
    // Tier order is semantic: the ladder is evaluated in sequence. Canonicalization
    // sorts object KEYS but must not sort array ELEMENTS, or two different ladders
    // would compare equal and one deployment's pricing would be adopted for both.
    const ladder = [
      { inputTokensAbove: 200_000, input: 6, output: 22.5, cacheRead: 0.6, cacheWrite: 7.5 },
      { inputTokensAbove: 400_000, input: 9, output: 30, cacheRead: 0.9, cacheWrite: 11.25 },
    ];
    const rows = [row({ model_info: { id: "a", mode: "chat" } }), row({ model_info: { id: "b", mode: "chat" } })];
    const withLadder =
      (value: typeof ladder): CatalogResolver =>
      () => ({
        provider: "anthropic",
        reasoning: true,
        vision: true,
        contextWindow: 200_000,
        maxTokens: 64_000,
        cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75, tiers: value },
      });

    let call = 0;
    const swapped: CatalogResolver = (entry, singleton) =>
      withLadder(call++ === 0 ? ladder : [...ladder].reverse())(entry, singleton);
    expect(reduceModelGroup(rows, swapped)?.cost.tiers).toBeUndefined();

    // Identical ladders in identical order still adopt, so the check above is not
    // simply rejecting every multi-element ladder.
    expect(reduceModelGroup(rows, withLadder(ladder))?.cost.tiers).toEqual(ladder);
  });

  it("withholds tiered pricing when deployments genuinely disagree", () => {
    const rows = [row({ model_info: { id: "a", mode: "chat" } }), row({ model_info: { id: "b", mode: "chat" } })];
    let call = 0;
    const conflicting: CatalogResolver = () => ({
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
          {
            inputTokensAbove: call++ === 0 ? 200_000 : 400_000,
            input: 6,
            output: 22.5,
            cacheRead: 0.6,
            cacheWrite: 7.5,
          },
        ],
      },
    });

    expect(reduceModelGroup(rows, conflicting)?.cost.tiers).toBeUndefined();
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
      name: "Kimi K3 with effort",
      semanticModel: "kimi-k3" as const,
      params: ["reasoning_effort"],
      expected: {
        reasoning: true,
        thinkingLevelMap: {
          off: null,
          minimal: null,
          low: "low",
          medium: null,
          high: "high",
          xhigh: null,
          max: "max",
        },
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
        thinkingLevelMap: {
          off: "off",
          minimal: null,
          low: null,
          medium: null,
          high: "high",
          xhigh: null,
          max: "max",
        },
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
        thinkingLevelMap: {
          off: null,
          minimal: null,
          low: null,
          medium: null,
          high: "high",
          xhigh: null,
          max: "max",
        },
        compat: {
          thinkingFormat: "openai",
          supportsReasoningEffort: true,
          requiresReasoningContentOnAssistantMessages: true,
        },
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
