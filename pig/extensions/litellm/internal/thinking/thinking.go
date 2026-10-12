// Ports src/thinking-levels.ts
package thinking

import "github.com/MichaelKinsy/PiG/ai"

// Definition is one row of THINKING_LEVEL_DEFINITIONS: Pi level, LiteLLM effort value, support flag.
type Definition struct {
	Level  ai.ThinkingLevel
	Effort string
	Flag   string
}

var Definitions = []Definition{
	{ai.ThinkingOff, "none", "supports_none_reasoning_effort"},
	{ai.ThinkingMinimal, "minimal", "supports_minimal_reasoning_effort"},
	{ai.ThinkingLow, "low", "supports_low_reasoning_effort"},
	{ai.ThinkingMedium, "medium", "supports_medium_reasoning_effort"},
	{ai.ThinkingHigh, "high", "supports_high_reasoning_effort"},
	{ai.ThinkingXHigh, "xhigh", "supports_xhigh_reasoning_effort"},
	{ai.ThinkingMax, "max", "supports_max_reasoning_effort"},
}

// Pi sends an omitted standard level as its own name. An omitted off sends no disable value
// and omitted xhigh/max are unsupported, so those stay distinct.
func implicit(level ai.ThinkingLevel) bool {
	switch level {
	case ai.ThinkingMinimal, ai.ThinkingLow, ai.ThinkingMedium, ai.ThinkingHigh:
		return true
	}
	return false
}

// Intersect is intersectThinkingLevelMaps. A level mentioned by any source is denied (nil value)
// unless every source supplies the same mapping for it. A nil map is "absent"; the result is nil
// when every input is nil or no level is mentioned.
func Intersect(maps []ai.ThinkingLevelMap) ai.ThinkingLevelMap {
	allAbsent := true
	for _, m := range maps {
		if m != nil {
			allAbsent = false
		}
	}
	if allAbsent {
		return nil
	}

	out := ai.ThinkingLevelMap{}
	for _, d := range Definitions {
		mentioned := false
		for _, m := range maps {
			if _, ok := m[d.Level]; ok {
				mentioned = true
			}
		}
		if !mentioned {
			continue
		}
		var first *string
		agreed := true
		for i, m := range maps {
			v, ok := m[d.Level]
			if !ok && implicit(d.Level) {
				s := string(d.Level)
				v, ok = &s, true
			}
			if !ok || v == nil {
				agreed = false
				break
			}
			if i == 0 {
				first = v
			} else if *v != *first {
				agreed = false
				break
			}
		}
		if agreed {
			out[d.Level] = first
		} else {
			out[d.Level] = nil
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
