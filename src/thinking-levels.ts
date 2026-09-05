import type { ModelThinkingLevel } from "@earendil-works/pi-ai";
import type { DiscoveredModel } from "./types.js";

export const THINKING_LEVEL_DEFINITIONS = [
  ["off", "none", "supports_none_reasoning_effort"],
  ["minimal", "minimal", "supports_minimal_reasoning_effort"],
  ["low", "low", "supports_low_reasoning_effort"],
  ["medium", "medium", "supports_medium_reasoning_effort"],
  ["high", "high", "supports_high_reasoning_effort"],
  ["xhigh", "xhigh", "supports_xhigh_reasoning_effort"],
  ["max", "max", "supports_max_reasoning_effort"],
] as const;

type AssertNever<T extends never> = T;
// biome-ignore lint/correctness/noUnusedVariables: This assertion must fail if Pi adds an uncovered thinking level.
type ThinkingLevelDefinitionsAreExhaustive = AssertNever<
  Exclude<ModelThinkingLevel, (typeof THINKING_LEVEL_DEFINITIONS)[number][0]>
>;

type ThinkingLevelMap = DiscoveredModel["thinkingLevelMap"];

// Intersects independently authoritative maps. A level mentioned by any source
// is denied unless every source supplies the same mapping for that level.
export function intersectThinkingLevelMaps(maps: readonly ThinkingLevelMap[]): ThinkingLevelMap {
  if (maps.every((map) => map === undefined)) return undefined;

  const intersection: NonNullable<ThinkingLevelMap> = {};
  for (const [level] of THINKING_LEVEL_DEFINITIONS) {
    const values = maps.map((map) => map?.[level]);
    if (values.every((value) => value === undefined)) continue;
    const first = values[0];
    intersection[level] = first !== undefined && values.every((value) => value === first) ? first : null;
  }
  return Object.keys(intersection).length > 0 ? intersection : undefined;
}
