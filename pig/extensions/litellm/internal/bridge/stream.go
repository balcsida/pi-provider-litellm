package bridge

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/MichaelKinsy/PiG/ai"
	"github.com/MichaelKinsy/PiG/extensions/sdk"
)

// callbackKeys are wire option names that name Go func or non-JSON fields of ai.StreamOptions.
var callbackKeys = []string{"onPayload", "onResponse", "transformHeaders", "onProviderStreamEvent", "fetch", "signal", "telemetryContext"}

func wrapStream(call ai.ModelsStreamFunction) sdk.ProviderStreamFunc {
	return func(wire, transcriptWire map[string]any, in sdk.ProviderStreamOptions) (*sdk.ModelEventStream, error) {
		model, err := streamModel(wire)
		if err != nil {
			return nil, err
		}
		transcript, err := decodeTranscript(transcriptWire)
		if err != nil {
			return nil, err
		}
		options, err := decodeOptions(in.Values)
		if err != nil {
			return nil, err
		}
		parent := in.Signal
		if parent == nil {
			parent = context.Background()
		}
		ctx, cancel := context.WithCancel(parent)
		options.Signal = ctx
		wireCallbacks(&options, wire, in)
		events, err := call(ctx, model, transcript, options)
		if err != nil {
			cancel()
			return nil, err
		}
		out := sdk.CreateAssistantMessageEventStream()
		go pump(ctx, cancel, events, out, model)
		return out, nil
	}
}

// decodeOptions turns the wire option object into ai.StreamOptions. encoding/json matches keys
// case-insensitively, so apiKey, headers, env, maxTokens, sessionId, ... land on their fields;
// keys without a field are ignored. An explicit temperature sets TemperatureSet.
func decodeOptions(values map[string]any) (ai.StreamOptions, error) {
	var options ai.StreamOptions
	if len(values) == 0 {
		return options, nil
	}
	clean := make(map[string]any, len(values))
	for key, value := range values {
		clean[key] = value
	}
	for _, key := range callbackKeys {
		delete(clean, key)
	}
	if err := fromMap(clean, &options); err != nil {
		return options, fmt.Errorf("decode stream options: %w", err)
	}
	_, options.TemperatureSet = clean["temperature"]
	return options, nil
}

func wireCallbacks(options *ai.StreamOptions, modelWire map[string]any, in sdk.ProviderStreamOptions) {
	if in.OnPayload != nil {
		options.OnPayload = func(payload any, _ *ai.Model) (any, error) { return in.OnPayload(payload, modelWire) }
	}
	if in.OnResponse != nil {
		options.OnResponse = func(_ context.Context, response ai.ProviderResponse, _ *ai.Model) error {
			wire, err := toMap(response)
			if err != nil {
				return err
			}
			return in.OnResponse(wire, modelWire)
		}
	}
	if in.TransformHeaders != nil {
		// ponytail: entries with a nil value (header deletions) cannot cross the map[string]string
		// wire, so they are omitted on the way out; the returned map replaces the headers wholesale.
		options.TransformHeaders = func(_ context.Context, headers ai.ProviderHeaders) (ai.ProviderHeaders, error) {
			returned, err := in.TransformHeaders(stringHeaders(headers))
			if err != nil {
				return nil, err
			}
			return ai.ProviderHeadersFromStrings(returned), nil
		}
	}
}

// pump forwards every ai event as its JSON object. It ends when the ai stream ends or ctx is
// cancelled, and always leaves out terminal: a done/error event ends it with the final message
// as the result, otherwise an aborted or error message is synthesized.
func pump(ctx context.Context, cancel context.CancelFunc, events *ai.AssistantMessageEventStream, out *sdk.ModelEventStream, model *ai.Model) {
	defer cancel()
	for event := range events.Events(ctx) {
		wire, err := toMap(event)
		if err != nil {
			out.Push(errorEvent(model, ai.StopReasonError, err))
			return
		}
		out.Push(wire)
		if kind := event.EventType(); kind == ai.EventDone || kind == ai.EventError {
			return
		}
	}
	reason, err := ai.StopReasonError, errors.New("provider stream ended without a terminal event")
	if ctx.Err() != nil {
		reason, err = ai.StopReasonAborted, context.Cause(ctx)
	}
	out.Push(errorEvent(model, reason, err))
}

func errorEvent(model *ai.Model, reason ai.StopReason, cause error) map[string]any {
	message := ai.AssistantMessage{Content: []ai.AssistantContentBlock{}, API: model.ProviderMeta.API, Provider: model.ProviderMeta.ProviderID,
		Model: model.ID, StopReason: reason, ErrorMessage: cause.Error()}
	data, _ := json.Marshal(message)
	var wire map[string]any
	_ = json.Unmarshal(data, &wire)
	return map[string]any{"type": "error", "reason": string(reason), "error": wire}
}
