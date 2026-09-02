import { describe, expect, it } from "vitest";
import { type CatalogResolution, type CatalogResolver, reduceModelGroup } from "../src/model-groups.js";
import type { ModelInfoEntry } from "../src/types.js";

const catalog = new Map<string, CatalogResolution>([
  [
    "openai/gpt-4o",
    {
      provider: "openai",
      catalogModelId: "gpt-4o",
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
      catalogModelId: "claude-sonnet-4-6",
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
      catalogModelId: "anthropic.claude-sonnet-4-6",
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
      vision: true,
      contextWindow: 150_000,
      maxTokens: 16_000,
      cost: { input: 3, output: 20, cacheRead: 0.3, cacheWrite: 3.75 },
      hasCompleteMetadata: false,
      catalogAuthorityAmbiguous: true,
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
    // Conflicting variants of one deployment id stay plural, so they can never be
    // mistaken for a single deployment and re-admit public route text as evidence.
    const expected = reduceModelGroup([repeated, conflicting], resolveCatalog);
    expect(expected).toMatchObject({ contextWindow: 8_000 });
    expect(reduceModelGroup([conflicting, repeated], resolveCatalog)).toEqual(expected);
    // Exact id-less repeats remain plural: equal content is not enough evidence
    // that two rows describe the same deployment.
    let seen = 0;
    reduceModelGroup([anonymous, anonymous], () => {
      seen++;
      return undefined;
    });
    expect(seen).toBe(2);
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

  it("lets unsupported transport evidence force Chat without affecting metadata", () => {
    const responses = row({ model_info: { id: "responses", mode: "responses" } });
    const unsupported = row({
      model_info: { id: "embed", mode: "embedding", max_input_tokens: 1, max_output_tokens: 1 },
      litellm_params: { model: "internal/embedding" },
    });

    expect(reduceModelGroup([responses, unsupported], resolveCatalog)).toMatchObject({
      api: "openai-completions",
      contextWindow: 200_000,
      maxTokens: 32_000,
    });
  });

  it("falls back to Chat without reducing metadata from unsupported rows", () => {
    const result = reduceModelGroup(
      [
        row(),
        row({
          model_info: {
            id: "embed",
            mode: "embedding",
            max_input_tokens: 8_000,
            max_output_tokens: 1,
            input_cost_per_token: undefined,
          },
          litellm_params: { model: "internal/embedding" },
        }),
      ],
      resolveCatalog,
    );
    expect(result).toMatchObject({
      api: "openai-completions",
      // The embedding row votes on transport but does not reduce metadata.
      contextWindow: 200_000,
      maxTokens: 32_000,
      cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
    });
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
    // Genuinely non-chat: excluded from metadata, and only votes on transport.
    expect(reduceModelGroup([roomy, embedding], resolveCatalog)).toMatchObject({ contextWindow: 200_000 });

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

  it("marks metadata complete only when every published field is resolved", () => {
    const fullyExplicit = row();
    expect(reduceModelGroup([fullyExplicit], resolveCatalog)).toMatchObject({ hasCompleteMetadata: true });

    const pricedOnly = row({
      model_info: {
        id: "priced-only",
        mode: "chat",
        supports_reasoning: undefined,
        supports_vision: undefined,
        max_input_tokens: undefined,
        max_output_tokens: undefined,
      },
      litellm_params: { model: "internal/unknown" },
    });
    expect(reduceModelGroup([pricedOnly], resolveCatalog)).toMatchObject({
      hasCompleteMetadata: false,
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
      hasCompleteMetadata: true,
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
      hasCompleteMetadata: false,
      cost: { input: 4, output: 0, cacheRead: 0.3, cacheWrite: 3.75 },
    });
  });

  it("retains proven display prices and zeroes only unresolved fields without catalog authority", () => {
    // Characterization of the existing per-field cost block. No backend resolves,
    // so cache pricing is genuinely unknown rather than free: input and output
    // survive at their proven values, the unresolved cache fields read zero, and
    // `hasCompleteMetadata` stays false so the model can be marked incomplete.
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
      hasCompleteMetadata: false,
      cost: { input: 3, output: 15, cacheRead: 0, cacheWrite: 0 },
    });
    expect(singleton).not.toHaveProperty("catalogProvider");
    expect(singleton?.cost.tiers).toBeUndefined();

    // Proven fields still reduce to the maximum across a group.
    expect(
      reduceModelGroup([priced("a", 0.000003, 0.000015), priced("b", 0.000004, 0.000015)], resolveCatalog),
    ).toMatchObject({
      hasCompleteMetadata: false,
      cost: { input: 4, output: 15, cacheRead: 0, cacheWrite: 0 },
    });
  });

  it("withholds catalog authority for different concrete models from one provider", () => {
    const result = reduceModelGroup(
      [
        row({ model_info: { id: "a", mode: "chat" }, litellm_params: { model: "route-a" } }),
        row({ model_info: { id: "b", mode: "chat" }, litellm_params: { model: "route-b" } }),
      ],
      (entry) => ({
        provider: "anthropic",
        catalogModelId: entry.litellm_params?.model === "route-a" ? "claude-sonnet-4-6" : "claude-opus-4-5",
        semanticFamily: "claude",
        reasoning: true,
        vision: true,
        contextWindow: entry.litellm_params?.model === "route-a" ? 200_000 : 180_000,
        maxTokens: 64_000,
        cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
      }),
    );

    expect(result).toMatchObject({
      semanticFamily: "claude",
      catalogAuthorityAmbiguous: true,
      // Explicit router evidence remains usable; only conflicting catalog
      // authority is withheld.
      contextWindow: 200_000,
      cost: { input: 3, output: 15, cacheRead: 0.3, cacheWrite: 3.75 },
    });
    expect(result).not.toHaveProperty("catalogProvider");
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
    const alternating: CatalogResolver = (entry) => withTiers(call++ === 0 ? tiers : reordered)(entry);
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
    const swapped: CatalogResolver = (entry) => withLadder(call++ === 0 ? ladder : [...ladder].reverse())(entry);
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
    // The disagreement is reportable so the silent downgrade is diagnosable.
    expect(result?.catalogAuthorityAmbiguous).toBe(true);
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

describe("native Messages route selection", () => {
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
        thinkingLevelMap: { max: "max" },
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
        thinkingLevelMap: { xhigh: "xhigh", max: "max" },
      }),
    );

    expect(result?.thinkingLevelMap).toEqual({ xhigh: null, max: "max" });
  });
});
