package bridge

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/MichaelKinsy/PiG/ai"
)

// decodeTranscript reads Pi's TranscriptContext. PiG's host sends {"messages": [...]} with the
// system prompt and tools folded into a leading system message (toolsAdded); Pi-shaped callers may
// instead send systemPrompt and tools beside messages, which ai.NormalizeContext folds the same way.
func decodeTranscript(in map[string]any) (ai.TranscriptContext, error) {
	data, err := json.Marshal(in)
	if err != nil {
		return ai.TranscriptContext{}, err
	}
	var wire struct {
		SystemPrompt string            `json:"systemPrompt"`
		Messages     []json.RawMessage `json:"messages"`
		Tools        []ai.ToolSchema   `json:"tools"`
	}
	if err := json.Unmarshal(data, &wire); err != nil {
		return ai.TranscriptContext{}, fmt.Errorf("decode transcript: %w", err)
	}
	messages := make([]ai.Message, 0, len(wire.Messages))
	for i, raw := range wire.Messages {
		message, err := decodeMessage(raw)
		if err != nil {
			return ai.TranscriptContext{}, fmt.Errorf("decode transcript messages[%d]: %w", i, err)
		}
		messages = append(messages, message)
	}
	transcript := ai.NormalizeContext(ai.Context{SystemPrompt: wire.SystemPrompt, Messages: messages, Tools: wire.Tools})
	// TranscriptContext keeps its validation error private and answers Messages() with nil.
	if len(messages) > 0 && len(transcript.Messages()) == 0 {
		return ai.TranscriptContext{}, errors.New("invalid transcript")
	}
	return transcript, nil
}

func decodeMessage(raw json.RawMessage) (ai.Message, error) {
	var head struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(raw, &head); err != nil {
		return nil, err
	}
	switch head.Role {
	case "system":
		var wire struct {
			Sections     ai.OrderedSections `json:"sections"`
			ToolsAdded   []ai.ToolSchema    `json:"toolsAdded"`
			ToolsRemoved []ai.ToolReference `json:"toolsRemoved"`
			Timestamp    int64              `json:"timestamp"`
		}
		if err := json.Unmarshal(raw, &wire); err != nil {
			return nil, err
		}
		content, err := decodeSystemContent(head.Content)
		if err != nil {
			return nil, err
		}
		return ai.SystemMessage{Content: content, Sections: wire.Sections, ToolsAdded: wire.ToolsAdded, ToolsRemoved: wire.ToolsRemoved, Timestamp: wire.Timestamp}, nil
	case "user":
		var wire struct {
			Timestamp int64 `json:"timestamp"`
		}
		if err := json.Unmarshal(raw, &wire); err != nil {
			return nil, err
		}
		content, err := decodeUserContent(head.Content)
		if err != nil {
			return nil, err
		}
		return ai.UserMessage{Content: content, Timestamp: wire.Timestamp}, nil
	case "assistant":
		var message ai.AssistantMessage
		err := json.Unmarshal(raw, &message)
		return message, err
	case "toolResult", "tool":
		var message ai.ToolResultMessage
		err := json.Unmarshal(raw, &message)
		return message, err
	default:
		return nil, fmt.Errorf("unsupported role %q", head.Role)
	}
}

func decodeSystemContent(raw json.RawMessage) (ai.SystemContent, error) {
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return ai.SystemText(text), nil
	}
	var blocks ai.SystemTextBlocks
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return nil, errors.New("system content must be a string or text-block array")
	}
	return blocks, nil
}

func decodeUserContent(raw json.RawMessage) (ai.UserContent, error) {
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return ai.UserText(text), nil
	}
	var items []json.RawMessage
	if err := json.Unmarshal(raw, &items); err != nil {
		return nil, errors.New("user content must be a string or block array")
	}
	blocks := make(ai.UserContentBlocks, 0, len(items))
	for _, item := range items {
		block, err := ai.UnmarshalContentBlock(item)
		if err != nil {
			return nil, err
		}
		user, ok := block.(ai.UserContentBlock)
		if !ok {
			return nil, fmt.Errorf("user content cannot hold a %T block", block)
		}
		blocks = append(blocks, user)
	}
	return blocks, nil
}
