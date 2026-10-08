package litellm

import (
	sdk "github.com/MichaelKinsy/PiG/extensions/sdk"
)

func Extension() *sdk.Extension {
	e := sdk.New("litellm")

	e.Tool("litellm_ping", "Return a short pong so you can confirm the extension is wired up.",
		sdk.Schema{"type": "object", "properties": map[string]any{}},
		func(ctx sdk.Context, params map[string]any) (any, error) {
			return "pong from litellm", nil
		})

	e.Command("litellm", "Say hello from the litellm extension.", func(ctx sdk.Context, args string) error {
		ctx.Notify("hello from litellm", "info")
		return nil
	})

	e.OnSessionStart(func(ctx sdk.Context, _ map[string]any) (any, error) {
		ctx.SetStatus("litellm", "litellm loaded")
		return nil, nil
	})

	return e
}
