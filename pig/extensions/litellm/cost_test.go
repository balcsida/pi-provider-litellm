package litellm

// Ports tests/cost.test.ts and the cost cases of tests/features.test.ts "feature parity".

import "testing"

func assistant(provider string, usage map[string]any) map[string]any {
	return map[string]any{"role": "assistant", "provider": provider, "model": "m", "usage": usage}
}

func totalOf(t *testing.T, message map[string]any) float64 {
	t.Helper()
	if message == nil {
		t.Fatal("no message returned")
	}
	return message["usage"].(map[string]any)["cost"].(map[string]any)["total"].(float64)
}

func TestCostTracking(t *testing.T) {
	names := map[string]bool{"litellm": true, "litellm-anthropic": true}
	costHeader := map[string]any{"x-litellm-response-cost": "0.42"}
	newTracker := func() *costTracker { return &costTracker{pending: map[string]float64{}} }

	t.Run("preserves Pi's precomputed cost when LiteLLM omits its response-cost header", func(t *testing.T) {
		usage := map[string]any{"input": 100.0, "output": 50.0, "cost": map[string]any{"total": 0.002}}
		if got := newTracker().attach(assistant("litellm", usage), names); got != nil {
			t.Fatalf("got %v", got)
		}
	})

	t.Run("applies LiteLLM response-cost headers to alias provider messages", func(t *testing.T) {
		tracker := newTracker()
		tracker.record("litellm-anthropic", costHeader)
		got := tracker.attach(assistant("litellm-anthropic", map[string]any{"input": 100.0, "output": 50.0}), names)
		if totalOf(t, got) != 0.42 {
			t.Fatalf("got %v", got)
		}
	})

	t.Run("does not let one provider's headerless response clear another provider's pending cost", func(t *testing.T) {
		tracker := newTracker()
		tracker.record("litellm", costHeader)
		tracker.record("litellm-anthropic", map[string]any{})
		got := tracker.attach(assistant("litellm", map[string]any{"input": 100.0, "output": 50.0}), names)
		if totalOf(t, got) != 0.42 {
			t.Fatalf("got %v", got)
		}
	})

	t.Run("splits the cost by token usage and consumes it", func(t *testing.T) {
		tracker := newTracker()
		tracker.record("litellm", map[string]any{"X-Litellm-Response-Cost": "0.42"})
		usage := map[string]any{"input": 100.0, "output": 50.0, "cacheRead": 10.0, "cacheWrite": 5.0}
		got := tracker.attach(assistant("litellm", usage), names)
		cost := got["usage"].(map[string]any)["cost"].(map[string]any)
		if cost["input"].(float64) != 0.42*(100.0/165.0) || cost["cacheWrite"].(float64) != 0.42*(5.0/165.0) || cost["total"] != 0.42 {
			t.Fatalf("cost = %v", cost)
		}
		if _, original := usage["cost"]; original {
			t.Fatal("input usage mutated")
		}
		if tracker.attach(assistant("litellm", usage), names) != nil {
			t.Fatal("cost applied twice")
		}
	})

	t.Run("a zero-token message gets zero shares and the full total", func(t *testing.T) {
		tracker := newTracker()
		tracker.record("litellm", costHeader)
		cost := tracker.attach(assistant("litellm", map[string]any{"input": 0.0, "output": 0.0}), names)["usage"].(map[string]any)["cost"].(map[string]any)
		if cost["input"] != 0.0 || cost["total"] != 0.42 {
			t.Fatalf("cost = %v", cost)
		}
	})

	t.Run("a non-numeric or empty header clears the pending cost", func(t *testing.T) {
		for _, header := range []any{"abc", "", nil} {
			tracker := newTracker()
			tracker.record("litellm", costHeader)
			tracker.record("litellm", map[string]any{"x-litellm-response-cost": header})
			if tracker.attach(assistant("litellm", map[string]any{"input": 1.0}), names) != nil {
				t.Fatalf("header %v kept the cost", header)
			}
		}
	})

	t.Run("does not apply LiteLLM model costs to other providers' messages", func(t *testing.T) {
		tracker := newTracker()
		tracker.record("litellm", costHeader)
		if tracker.attach(assistant("anthropic", map[string]any{"input": 100.0}), names) != nil {
			t.Fatal("applied to another provider")
		}
	})

	t.Run("ignores LiteLLM cost headers captured from non-LiteLLM responses", func(t *testing.T) {
		// The after_provider_response handler records only for configured providers (inScope), so a
		// foreign response never reaches record; the message then carries Pi's own cost.
		tracker := newTracker()
		usage := map[string]any{"input": 100.0, "output": 50.0, "cost": map[string]any{"total": 0.00107175}}
		if tracker.attach(assistant("litellm", usage), names) != nil {
			t.Fatal("cost replaced")
		}
	})

	t.Run("skips non-assistant messages and messages without usage", func(t *testing.T) {
		tracker := newTracker()
		tracker.record("litellm", costHeader)
		user := assistant("litellm", map[string]any{"input": 1.0})
		user["role"] = "user"
		if tracker.attach(user, names) != nil || tracker.attach(map[string]any{"role": "assistant", "provider": "litellm"}, names) != nil {
			t.Fatal("applied")
		}
	})

	t.Run("parses like Number.parseFloat", func(t *testing.T) {
		for text, want := range map[string]float64{"0.42": 0.42, " 1e-3x": 0.001, ".5": 0.5, "-2": -2} {
			if got, ok := parseFloatPrefix(text); !ok || got != want {
				t.Fatalf("%q = %v %v", text, got, ok)
			}
		}
		if _, ok := parseFloatPrefix("abc"); ok {
			t.Fatal("abc parsed")
		}
	})
}
