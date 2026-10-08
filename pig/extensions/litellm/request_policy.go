package litellm

// Ports src/index.ts: isReasoningItem, normalizeStrictToolMessages, prepareLiteLLMRequestPayload,
// normalizeThinkTags and the before_provider_request / message_end registrations. Payloads and messages
// are the JSON maps PiG's wire hands to handlers; nil results mean "keep the current value".

import (
	"maps"
	"slices"
	"strings"

	sdk "github.com/MichaelKinsy/PiG/extensions/sdk"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/discover"
	"github.com/balcsida/pi-provider-litellm/pig/extensions/litellm/internal/types"
)

// hookModel is the part of a model the request hook reads: api and thinkingLevelMap come from the
// host's registry, the family from the policy sidecar. An empty API means the model declares none.
type hookModel struct {
	ID     string
	API    string
	Family types.BackendFamily
	// OffEffort is thinkingLevelMap.off when it is a string.
	OffEffort *string
}

// reasoningVisibilityDefaults suppress duplicate Kimi reasoning without changing the model-specific
// generation control selected by the driver.
var reasoningVisibilityDefaults = []struct {
	key   string
	value any
}{
	{"include_reasoning", false},
	{"reasoning_content", false},
	{"merge_reasoning_content_in_choices", true},
}

var reasoningRequestKeys = []string{
	"reasoning", "reasoning_effort", "include_reasoning", "reasoning_content", "merge_reasoning_content_in_choices", "thinking",
}

func isReasoningItem(item any) bool {
	object, ok := item.(map[string]any)
	return ok && object["type"] == "reasoning"
}

// normalizeStrictToolMessages repairs messages for Moonshot/Kimi's strict OpenAI schema validation:
// assistant tool calls carry string content and tool results are plain text. The bool reports a change.
func normalizeStrictToolMessages(messages []any) ([]any, bool) {
	normalized := make([]any, 0, len(messages))
	var imageParts []any
	changed := false
	flushImages := func() {
		if len(imageParts) == 0 {
			return
		}
		content := append([]any{map[string]any{"type": "text", "text": "Attached image(s) from tool result:"}}, imageParts...)
		normalized = append(normalized, map[string]any{"role": "user", "content": content})
		imageParts = nil
	}
	for _, raw := range messages {
		message, isMap := raw.(map[string]any)
		if !isMap || message["role"] != "tool" {
			flushImages()
		}
		if !isMap {
			normalized = append(normalized, raw)
			continue
		}
		toolCalls, _ := message["tool_calls"].([]any)
		parts, partsIsArray := message["content"].([]any)
		switch {
		case message["role"] == "assistant" && len(toolCalls) > 0 && message["content"] == nil:
			copied := maps.Clone(message)
			copied["content"] = ""
			normalized = append(normalized, copied)
			changed = true
		case message["role"] == "tool" && partsIsArray:
			previousImageCount := len(imageParts)
			var content strings.Builder
			for _, part := range parts {
				if text, ok := part.(string); ok {
					content.WriteString(text)
					continue
				}
				object, _ := part.(map[string]any)
				switch object["type"] {
				case "text":
					text, _ := object["text"].(string)
					content.WriteString(text)
				case "image_url":
					imageParts = append(imageParts, part)
				}
			}
			text := content.String()
			if text == "" {
				text = "(no tool output)"
				if len(imageParts) > previousImageCount {
					text = "(see attached image)"
				}
			}
			copied := maps.Clone(message)
			copied["content"] = text
			normalized = append(normalized, copied)
			changed = true
		default:
			normalized = append(normalized, raw)
		}
	}
	flushImages()
	if !changed {
		return messages, false
	}
	return normalized, true
}

// prepareRequestPayload is prepareLiteLLMRequestPayload. It returns nil when the payload needs no change
// and never mutates its input.
func prepareRequestPayload(payload map[string]any, model hookModel, policy *types.LiteLLMModelPolicy) map[string]any {
	api := model.API
	if api == "" {
		api = "openai-completions"
	}
	if policy == nil {
		policy = &types.LiteLLMModelPolicy{}
	}
	var next map[string]any
	current := func() map[string]any {
		if next != nil {
			return next
		}
		return payload
	}
	mutable := func() map[string]any {
		if next == nil {
			next = maps.Clone(payload)
		}
		return next
	}

	if api == "openai-completions" && policy.SuppressReasoningVisibility {
		for _, entry := range reasoningVisibilityDefaults {
			if _, present := payload[entry.key]; !present {
				mutable()[entry.key] = entry.value
			}
		}
	}

	// GPT-5.5 and later reject reasoning alongside function tools on Chat Completions. Discovery decides
	// from deployment evidence; only a model with no backend evidence uses its route name.
	routeOnly := model.Family == "" && model.ID != ""
	tools, _ := payload["tools"].([]any)
	if api == "openai-completions" && len(tools) > 0 &&
		(policy.DropToolReasoning || (routeOnly && discover.IsGpt55OrNewerModel(model.ID))) {
		for _, key := range reasoningRequestKeys {
			if _, present := payload[key]; present {
				delete(mutable(), key)
			}
		}
		// Send the model's own off effort when it declares one. GPT-6 also rejects an omitted effort, and
		// its error names `none` as the fix, so it gets `none` even without carrier evidence.
		explicitOff := policy.ExplicitToolReasoningOff || (routeOnly && discover.IsGpt6OrNewerModel(model.ID))
		off := ""
		switch {
		case model.OffEffort != nil:
			off = *model.OffEffort
		case explicitOff:
			off = "none"
		}
		if model.OffEffort != nil || explicitOff {
			if existing, present := current()["reasoning_effort"]; !present || existing != off {
				mutable()["reasoning_effort"] = off
			}
		}
		if include, ok := current()["include"].([]any); ok && slices.Contains(include, any("reasoning.encrypted_content")) {
			filtered := slices.DeleteFunc(slices.Clone(include), func(value any) bool { return value == "reasoning.encrypted_content" })
			if len(filtered) == 0 {
				delete(mutable(), "include")
			} else {
				mutable()["include"] = filtered
			}
		}
		// Prior turns may have replayed reasoning items into the input; they are rejected once reasoning
		// is stripped.
		if input, ok := current()["input"].([]any); ok && slices.ContainsFunc(input, isReasoningItem) {
			mutable()["input"] = slices.DeleteFunc(slices.Clone(input), isReasoningItem)
		}
	}

	// Moonshot/Kimi applies strict OpenAI schema validation. The rewrite requires the deployment-backed
	// policy; route text is not evidence.
	if api == "openai-completions" && policy.NormalizeStrictToolMessages {
		if messages, ok := current()["messages"].([]any); ok {
			if normalized, changed := normalizeStrictToolMessages(messages); changed {
				mutable()["messages"] = normalized
			}
		}
	}

	if (api == "openai-completions" || api == "openai-responses") && policy.NormalizeGeminiReasoningEffort {
		if effort, ok := current()["reasoning_effort"].(string); ok {
			if lower := strings.ToLower(effort); lower != effort {
				mutable()["reasoning_effort"] = lower
			}
		}
		if reasoning, ok := current()["reasoning"].(map[string]any); ok {
			if effort, ok := reasoning["effort"].(string); ok {
				if lower := strings.ToLower(effort); lower != effort {
					replaced := maps.Clone(reasoning)
					replaced["effort"] = lower
					mutable()["reasoning"] = replaced
				}
			}
		}
	}
	return next
}

// normalizeThinkTags unwraps `<think>` text into thinking blocks. It returns nil when the message is
// out of scope or has no tags.
func normalizeThinkTags(message map[string]any, providerNames map[string]bool, api string, policy *types.LiteLLMModelPolicy) map[string]any {
	provider, _ := message["provider"].(string)
	if !providerNames[provider] || (api != "" && api != "openai-completions") || policy == nil || !policy.NormalizeThinkTags {
		return nil
	}
	blocks, _ := message["content"].([]any)
	changed := false
	var content []any
	appendKind := func(kind, field, text string) {
		if text == "" {
			return
		}
		if len(content) > 0 {
			if last, ok := content[len(content)-1].(map[string]any); ok && last["type"] == kind {
				// A malformed block (no string field) is not merged into; a new block follows it.
				if previous, ok := last[field].(string); ok {
					last[field] = previous + text
					return
				}
			}
		}
		content = append(content, map[string]any{"type": kind, field: text})
	}
	appendText := func(text string) { appendKind("text", "text", text) }
	appendThinking := func(text string) { appendKind("thinking", "thinking", text) }

	for blockIndex, raw := range blocks {
		block, _ := raw.(map[string]any)
		text, isText := block["text"].(string)
		if block["type"] != "text" || !isText {
			content = append(content, raw)
			continue
		}
		index := 0
		for index < len(text) {
			start := strings.Index(text[index:], "<think>")
			if start == -1 {
				appendText(text[index:])
				break
			}
			start += index
			changed = true
			appendText(text[index:start])
			thinkingStart := start + len("<think>")
			end := strings.Index(text[thinkingStart:], "</think>")
			if end == -1 {
				beforeNonText := slices.ContainsFunc(blocks[blockIndex+1:], func(next any) bool {
					object, _ := next.(map[string]any)
					return object["type"] != "text"
				})
				if beforeNonText {
					appendThinking(text[thinkingStart:])
				} else {
					appendText(text[thinkingStart:])
				}
				break
			}
			end += thinkingStart
			appendThinking(text[thinkingStart:end])
			index = end + len("</think>")
		}
	}
	if !changed {
		return nil
	}
	result := maps.Clone(message)
	if content == nil {
		content = []any{}
	}
	result["content"] = content
	return result
}

// requestModelFor resolves what the hooks need about a model: api and thinkingLevelMap from the host's
// registry (nil when the model is unknown), policy and family from the sidecar.
func (s *extensionState) requestModelFor(ctx hookHost, provider, modelID string) (hookModel, *types.LiteLLMModelPolicy) {
	model := hookModel{ID: modelID}
	if found := ctx.FindModel(provider, modelID); found != nil {
		model.API, _ = found["api"].(string)
		if levels, ok := found["thinkingLevelMap"].(map[string]any); ok {
			if off, ok := levels["off"].(string); ok {
				model.OffEffort = &off
			}
		}
	}
	policy, family, _ := s.policies.policyFor(provider, modelID)
	model.Family = family
	return model, policy
}

// setupRequestPolicy registers the request-payload hook and the think-tag message normalizer.
func setupRequestPolicy(e hookRegistrar, s *extensionState) {
	e.OnEvent(sdk.EventBeforeProviderRequest, func(ctx hookHost, data map[string]any) (any, error) {
		if !s.inScope(ctx) {
			return nil, nil
		}
		payload, ok := data["payload"].(map[string]any)
		if !ok {
			return nil, nil
		}
		model, policy := s.requestModelFor(ctx, ctx.ModelProvider(), ctx.Model())
		if next := prepareRequestPayload(payload, model, policy); next != nil {
			return next, nil
		}
		return nil, nil
	})

	e.OnEvent(sdk.EventMessageEnd, func(ctx hookHost, data map[string]any) (any, error) {
		message, ok := data["message"].(map[string]any)
		if !ok || message["role"] != "assistant" {
			return nil, nil
		}
		provider, _ := message["provider"].(string)
		if !s.providerNames[provider] {
			return nil, nil
		}
		modelID, _ := message["model"].(string)
		// The discovered conclusion, not the route name, decides display normalization; an unresolvable
		// model carries none and is left untouched.
		model, policy := s.requestModelFor(ctx, provider, modelID)
		if next := normalizeThinkTags(message, s.providerNames, model.API, policy); next != nil {
			return map[string]any{"message": next}, nil
		}
		return nil, nil
	})
}
