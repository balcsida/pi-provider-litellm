// Ports tests/thinking-levels.test.ts
package thinking

import (
	"reflect"
	"testing"

	"github.com/MichaelKinsy/PiG/ai"
)

func s(v string) *string { return &v }

func TestIntersectThinkingLevelMaps(t *testing.T) {
	t.Run("keeps only unanimously identical values and explicitly denies disagreement or absence", func(t *testing.T) {
		got := Intersect([]ai.ThinkingLevelMap{
			{"off": s("none"), "low": s("low"), "high": s("high"), "max": nil},
			{"off": s("none"), "low": nil, "high": s("high"), "xhigh": s("xhigh")},
		})
		want := ai.ThinkingLevelMap{"off": s("none"), "low": nil, "high": s("high"), "xhigh": nil, "max": nil}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("got %v want %v", got, want)
		}
	})
	t.Run("treats an omitted standard level as Pi's default spelling", func(t *testing.T) {
		cases := []struct{ in, want []ai.ThinkingLevelMap }{
			{[]ai.ThinkingLevelMap{{"low": s("low"), "high": s("high")}, {}}, []ai.ThinkingLevelMap{{"low": s("low"), "high": s("high")}}},
			{[]ai.ThinkingLevelMap{{"low": s("minimal")}, nil}, []ai.ThinkingLevelMap{{"low": nil}}},
			{[]ai.ThinkingLevelMap{{"off": s("none"), "max": s("max")}, {}}, []ai.ThinkingLevelMap{{"off": nil, "max": nil}}},
		}
		for i, c := range cases {
			if got := Intersect(c.in); !reflect.DeepEqual(got, c.want[0]) {
				t.Errorf("case %d: got %v want %v", i, got, c.want[0])
			}
		}
	})
	t.Run("preserves the all-absent distinction and covers every supported thinking level", func(t *testing.T) {
		if got := Intersect([]ai.ThinkingLevelMap{nil, nil}); got != nil {
			t.Errorf("got %v", got)
		}
		var levels []ai.ThinkingLevel
		for _, d := range Definitions {
			levels = append(levels, d.Level)
		}
		want := []ai.ThinkingLevel{"off", "minimal", "low", "medium", "high", "xhigh", "max"}
		if !reflect.DeepEqual(levels, want) {
			t.Errorf("got %v", levels)
		}
	})
}
