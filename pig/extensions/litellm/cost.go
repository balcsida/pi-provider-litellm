package litellm

// Ports src/cost.ts: LiteLLM's x-litellm-response-cost header becomes the assistant message's cost.

import (
	"fmt"
	"maps"
	"math"
	"regexp"
	"strconv"
	"strings"
	"sync"

	sdk "github.com/MichaelKinsy/PiG/extensions/sdk"
)

var leadingFloat = regexp.MustCompile(`^\s*[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?`)

// parseFloatPrefix is Number.parseFloat for finite numbers: it reads the longest numeric prefix.
func parseFloatPrefix(text string) (float64, bool) {
	prefix := leadingFloat.FindString(text)
	if prefix == "" {
		return 0, false
	}
	value, err := strconv.ParseFloat(strings.TrimSpace(prefix), 64)
	if err != nil || math.IsInf(value, 0) {
		return 0, false
	}
	return value, true
}

func headerValue(headers map[string]any, names ...string) any {
	for _, name := range names {
		if value, ok := headers[name]; ok {
			return value
		}
	}
	return nil
}

func number(value any) float64 {
	n, _ := value.(float64)
	return n
}

// costTracker holds the pending response cost per provider: with several LiteLLM providers active,
// overlapping requests interleave, so one shared slot would let a headerless response clobber another
// provider's pending cost.
type costTracker struct {
	mu      sync.Mutex
	pending map[string]float64
}

// record keeps the response's cost for provider, or clears it when the header is absent or not a number.
func (c *costTracker) record(provider string, headers map[string]any) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if raw := headerValue(headers, "x-litellm-response-cost", "X-Litellm-Response-Cost"); raw != nil && raw != "" {
		if cost, ok := parseFloatPrefix(fmt.Sprint(raw)); ok {
			c.pending[provider] = cost
			return
		}
	}
	delete(c.pending, provider)
}

// attach returns the message with usage.cost split by token usage, or nil when it has no pending cost.
func (c *costTracker) attach(message map[string]any, providerNames map[string]bool) map[string]any {
	if message["role"] != "assistant" {
		return nil
	}
	provider, _ := message["provider"].(string)
	usage, ok := message["usage"].(map[string]any)
	if !providerNames[provider] || !ok {
		return nil
	}
	c.mu.Lock()
	total, found := c.pending[provider]
	delete(c.pending, provider)
	c.mu.Unlock()
	if !found {
		return nil
	}
	totalTokens := number(usage["input"]) + number(usage["output"]) + number(usage["cacheRead"]) + number(usage["cacheWrite"])
	share := func(tokens float64) float64 {
		if totalTokens > 0 {
			return total * (tokens / totalTokens)
		}
		return 0
	}
	withCost := maps.Clone(usage)
	withCost["cost"] = map[string]any{
		"input":      share(number(usage["input"])),
		"output":     share(number(usage["output"])),
		"cacheRead":  share(number(usage["cacheRead"])),
		"cacheWrite": share(number(usage["cacheWrite"])),
		"total":      total,
	}
	updated := maps.Clone(message)
	updated["usage"] = withCost
	return updated
}

// setupCostTracking registers the response-cost capture and the message_end cost attachment. Register it
// before the budget hooks and setupRequestPolicy: the cost after_provider_response handler must run first,
// and its message_end handler runs before the think-tag normalizer, which then sees the costed message.
func setupCostTracking(e hookRegistrar, s *extensionState) {
	tracker := &costTracker{pending: map[string]float64{}}
	e.OnEvent(sdk.EventAfterProviderResponse, func(ctx hookHost, data map[string]any) (any, error) {
		// Global hook: responses from other providers must not feed LiteLLM cost state.
		if s.inScope(ctx) {
			headers, _ := data["headers"].(map[string]any)
			tracker.record(ctx.ModelProvider(), headers)
		}
		return nil, nil
	})
	e.OnEvent(sdk.EventMessageEnd, func(_ hookHost, data map[string]any) (any, error) {
		message, _ := data["message"].(map[string]any)
		if updated := tracker.attach(message, s.providerNames); updated != nil {
			return map[string]any{"message": updated}, nil
		}
		return nil, nil
	})
}
